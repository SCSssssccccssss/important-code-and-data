"""Smoke test for level-wise 2D ionosphere diffusion training.

This test uses small random data so it can run on CPU/GPU without allocating
the full [31, 96, 82, 91, 91] tensor. Replace `raw_data` with your real tensor
when training on real electron-density data.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Diffusion_Ionosphere_2D.config import (  # noqa: E402
    ConditioningConfig,
    DiffusionConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    SchedulerConfig,
    TrainConfig,
)
from Diffusion_Ionosphere_2D.data import make_height_2d_train_valid_datasets  # noqa: E402
from Diffusion_Ionosphere_2D.train import run_training  # noqa: E402
from Diffusion_Ionosphere_2D.unet2D import HeightConditionedUNet2D  # noqa: E402


def make_random_electron_density_data(
    days: int = 2,
    time_steps: int = 12,
    height_levels: int = 4,
    lat: int = 32,
    lon: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Create random electron-density data shaped [days, T, Z, H, W]."""

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        days,
        time_steps,
        height_levels,
        lat,
        lon,
        generator=generator,
    ).float()


def build_demo_config(window: int, data_channels: int, batch_size: int) -> ExperimentConfig:
    """Configuration for the 2D height-slice smoke training run."""

    return ExperimentConfig(
        diffusion=DiffusionConfig(
            window=window,
            data_channels=data_channels,
            alpha_schedule="cos",
            eta=1e-3,
            model_type="noise",
        ),
        conditioning=ConditioningConfig(
            min_condition_frames=1,
            max_condition_frames=window - 1,
            shared_across_batch=True,
            condition_noise_std=0.0,
        ),
        model=ModelConfig(
            time_embedding_dim=64,
            height_embedding_dim=64,
            use_ema=True,
            ema_decay=0.999,
        ),
        optim=OptimConfig(
            lr=2e-4,
            weight_decay=1e-4,
            beta1=0.9,
            beta2=0.999,
        ),
        scheduler=SchedulerConfig(
            name="cosine",
            min_lr=1e-6,
        ),
        train=TrainConfig(
            batch_size=batch_size,
            epochs=2,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_amp=torch.cuda.is_available(),
            grad_clip_norm=1.0,
            log_every=2,
            checkpoint_every=1,
            output_dir=Path("Diffusion_Ionosphere_2D") / "smoke_outputs",
        ),
    )


def build_demo_backbone(cfg: ExperimentConfig) -> HeightConditionedUNet2D:
    """Small 2D U-Net for fast smoke testing."""

    return HeightConditionedUNet2D(
        in_channels=cfg.diffusion.backbone_in_channels,
        out_channels=cfg.diffusion.backbone_out_channels,
        base_channels=16,
        channel_multipliers=(1, 2, 4),
        time_embedding_dim=cfg.model.time_embedding_dim,
        height_embedding_dim=cfg.model.height_embedding_dim,
        dropout=0.0,
    )


def main() -> None:
    torch.manual_seed(42)

    window = 4
    data_channels = 1
    batch_size = 4

    raw_data = make_random_electron_density_data()
    train_dataset, valid_dataset, base_dataset = make_height_2d_train_valid_datasets(
        raw_data,
        window=window,
        train_ratio=0.8,
        seed=42,
        shuffle=True,
    )

    cfg = build_demo_config(
        window=window,
        data_channels=data_channels,
        batch_size=batch_size,
    )
    backbone = build_demo_backbone(cfg)

    sample_x, sample_h = base_dataset[0]
    real_num_samples = 31 * 82 * (96 - window + 1)

    print("=== 2D height-slice diffusion smoke test ===")
    print(f"smoke raw shape: {tuple(raw_data.shape)}")
    print(f"sample x_h shape: {tuple(sample_x.shape)}")
    print(f"sample h_norm: {float(sample_h):.3f}")
    print(f"train/valid samples: {len(train_dataset)} / {len(valid_dataset)}")
    print(f"real-data sample count if [31,96,82,91,91]: {real_num_samples}")
    print(f"backbone in/out: {cfg.diffusion.backbone_in_channels} -> {cfg.diffusion.backbone_out_channels}")
    print(f"model parameters: {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"device: {cfg.train.device}")

    results = run_training(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        backbone=backbone,
        cfg=cfg,
    )

    print("\n=== Training finished ===")
    for record in results["history"]:
        print(record)
    print(f"\nArtifacts written to: {results['output_dir']}")


if __name__ == "__main__":
    main()
