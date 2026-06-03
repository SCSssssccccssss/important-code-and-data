"""Training loop for universal amortised level-wise 2D diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .conditioning import build_universal_condition_2d, sample_conditioned_frames
from .config import ConditioningConfig
from .ema import ExponentialMovingAverage
from .score_model import AmortizedHeightScoreModel
from .sde import VPSDE


@dataclass
class EpochStats:
    loss: float
    mean_conditioned_frames: float
    num_batches: int


class UniversalAmortizedHeightTrainer:
    """Encapsulates train/valid loop for height-conditioned 2D training."""

    def __init__(
        self,
        score_model: AmortizedHeightScoreModel,
        sde: VPSDE,
        optimizer: torch.optim.Optimizer,
        conditioning_config: ConditioningConfig,
        data_channels: int,
        device: torch.device,
        ema: Optional[ExponentialMovingAverage] = None,
        grad_clip_norm: Optional[float] = None,
        use_amp: bool = False,
    ):
        self.score_model = score_model
        self.sde = sde
        self.optimizer = optimizer
        self.conditioning_config = conditioning_config
        self.data_channels = data_channels
        self.device = device
        self.ema = ema
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _prepare_context(self, x: Tensor, height: Tensor) -> Tensor:
        window = x.shape[1] // self.data_channels
        conditioned_frames = sample_conditioned_frames(
            batch_size=x.shape[0],
            window=window,
            min_condition_frames=self.conditioning_config.min_condition_frames,
            max_condition_frames=self.conditioning_config.resolve_max_condition_frames(window),
            shared_across_batch=self.conditioning_config.shared_across_batch,
            device=x.device,
        )
        condition, _ = build_universal_condition_2d(
            clean_window=x,
            conditioned_frames=conditioned_frames,
            data_channels=self.data_channels,
            condition_noise_std=self.conditioning_config.condition_noise_std,
        )
        self.score_model.set_context(condition=condition, height=height)
        return conditioned_frames

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_every: int = 20,
    ) -> EpochStats:
        self.score_model.train()
        self.sde.train()

        running_loss = 0.0
        running_conditioned_frames = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader, start=1):
            x, height = batch
            x = x.to(self.device, non_blocking=True)
            height = height.to(self.device, non_blocking=True).reshape(-1)

            conditioned_frames = self._prepare_context(x=x, height=height)
            running_conditioned_frames += conditioned_frames.float().mean().item()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss = self.sde.loss(x)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.score_model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    clip_grad_norm_(self.score_model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self.score_model)

            num_batches += 1
            running_loss += loss.item()

            if log_every > 0 and step % log_every == 0:
                avg_loss = running_loss / num_batches
                avg_cond = running_conditioned_frames / num_batches
                print(
                    f"[train] epoch={epoch:03d} step={step:05d} "
                    f"loss={avg_loss:.6f} mean_C={avg_cond:.2f}"
                )

        return EpochStats(
            loss=running_loss / max(num_batches, 1),
            mean_conditioned_frames=running_conditioned_frames / max(num_batches, 1),
            num_batches=num_batches,
        )

    @torch.no_grad()
    def validate_one_epoch(self, dataloader: DataLoader) -> EpochStats:
        self.score_model.eval()
        self.sde.eval()

        running_loss = 0.0
        running_conditioned_frames = 0.0
        num_batches = 0

        for batch in dataloader:
            x, height = batch
            x = x.to(self.device, non_blocking=True)
            height = height.to(self.device, non_blocking=True).reshape(-1)

            conditioned_frames = self._prepare_context(x=x, height=height)
            running_conditioned_frames += conditioned_frames.float().mean().item()

            loss = self.sde.loss(x)

            num_batches += 1
            running_loss += loss.item()

        return EpochStats(
            loss=running_loss / max(num_batches, 1),
            mean_conditioned_frames=running_conditioned_frames / max(num_batches, 1),
            num_batches=num_batches,
        )
