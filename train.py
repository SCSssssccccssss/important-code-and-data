"""High-level training entry for level-wise 2D ionosphere diffusion."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .checkpoint import save_model_weights, save_training_checkpoint
from .config import ExperimentConfig
from .data import build_dataloader
from .ema import ExponentialMovingAverage
from .score_model import AmortizedHeightScoreModel
from .sde import VPSDE
from .trainer import UniversalAmortizedHeightTrainer


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        weight_decay=cfg.optim.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    name = cfg.scheduler.name.lower()
    if name in {"none", "off"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs,
            eta_min=cfg.scheduler.min_lr,
        )
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler.gamma)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.scheduler.gamma)
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


def build_training_stack(
    backbone: nn.Module,
    sample_shape: tuple[int, int, int],
    cfg: ExperimentConfig,
) -> tuple[AmortizedHeightScoreModel, VPSDE]:
    score_model = AmortizedHeightScoreModel(
        backbone=backbone,
        noisy_channels=cfg.diffusion.noisy_channels,
        condition_channels=cfg.diffusion.condition_channels,
    )

    sde = VPSDE(
        eps_model=score_model,
        shape=(cfg.diffusion.noisy_channels, *sample_shape[-2:]),
        alpha=cfg.diffusion.alpha_schedule,
        eta=cfg.diffusion.eta,
        model_type=cfg.diffusion.model_type,
    )
    return score_model, sde


def run_training(
    train_dataset: Dataset,
    backbone: nn.Module,
    cfg: ExperimentConfig,
    valid_dataset: Optional[Dataset] = None,
) -> dict:
    """Train the universal amortised 2D model on height-slice datasets."""

    output_dir = Path(cfg.train.output_dir)
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg.train.device if cfg.train.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=cfg.train.drop_last,
    )

    valid_loader = None
    if valid_dataset is not None:
        valid_loader = build_dataloader(
            dataset=valid_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            drop_last=False,
        )

    sample_x, _ = train_dataset[0]
    if sample_x.shape[0] != cfg.diffusion.noisy_channels:
        raise ValueError(
            "Dataset sample channels do not match cfg.diffusion.window * data_channels: "
            f"{sample_x.shape[0]} vs {cfg.diffusion.noisy_channels}."
        )

    score_model, sde = build_training_stack(
        backbone=backbone,
        sample_shape=tuple(sample_x.shape),
        cfg=cfg,
    )
    score_model = score_model.to(device)
    sde = sde.to(device)

    optimizer = build_optimizer(score_model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    ema = None
    if cfg.model.use_ema:
        ema = ExponentialMovingAverage(score_model, decay=cfg.model.ema_decay)

    trainer = UniversalAmortizedHeightTrainer(
        score_model=score_model,
        sde=sde,
        optimizer=optimizer,
        conditioning_config=cfg.conditioning,
        data_channels=cfg.diffusion.data_channels,
        device=device,
        ema=ema,
        grad_clip_norm=cfg.train.grad_clip_norm,
        use_amp=cfg.train.use_amp,
    )

    history: list[dict] = []
    best_val = float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        train_stats = trainer.train_one_epoch(
            dataloader=train_loader,
            epoch=epoch,
            log_every=cfg.train.log_every,
        )

        val_stats = None
        if valid_loader is not None:
            val_stats = trainer.validate_one_epoch(valid_loader)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "train_mean_conditioned_frames": train_stats.mean_conditioned_frames,
            "val_loss": None if val_stats is None else val_stats.loss,
            "val_mean_conditioned_frames": None if val_stats is None else val_stats.mean_conditioned_frames,
            "lr": current_lr,
        }
        history.append(epoch_record)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_stats.loss:.6f} "
            f"train_mean_C={train_stats.mean_conditioned_frames:.2f} "
            f"{'' if val_stats is None else f'val_loss={val_stats.loss:.6f} val_mean_C={val_stats.mean_conditioned_frames:.2f} '} "
            f"lr={current_lr:.6e}"
        )

        if epoch % cfg.train.checkpoint_every == 0:
            save_training_checkpoint(
                path=ckpt_dir / "latest.pt",
                model=score_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                ema_state=None if ema is None else ema.state_dict(),
                extra={"history": history, "config": asdict(cfg)},
            )

        current_val = train_stats.loss if val_stats is None else val_stats.loss
        if current_val < best_val:
            best_val = current_val
            save_model_weights(ckpt_dir / "score_last.pt", score_model)
            if ema is not None:
                with ema.average_parameters(score_model):
                    save_model_weights(ckpt_dir / "score_ema.pt", score_model)

    save_model_weights(ckpt_dir / "score_last.pt", score_model)
    if ema is not None:
        with ema.average_parameters(score_model):
            save_model_weights(ckpt_dir / "score_ema.pt", score_model)

    return {
        "score_model": score_model,
        "sde": sde,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "ema": ema,
        "history": history,
        "output_dir": output_dir,
    }
