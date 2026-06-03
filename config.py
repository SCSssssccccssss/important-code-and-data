"""Typed configuration objects for level-wise 2D ionosphere diffusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DiffusionConfig:
    """Settings that define the 2D diffusion event shape."""

    window: int
    data_channels: int = 1
    alpha_schedule: str = "cos"
    eta: float = 1e-3
    model_type: str = "noise"

    @property
    def noisy_channels(self) -> int:
        return self.window * self.data_channels

    @property
    def condition_channels(self) -> int:
        return 2 * self.noisy_channels

    @property
    def backbone_in_channels(self) -> int:
        return self.condition_channels + self.noisy_channels

    @property
    def backbone_out_channels(self) -> int:
        return self.noisy_channels


@dataclass
class ConditioningConfig:
    """How universal amortised history conditioning is sampled."""

    min_condition_frames: int = 1
    max_condition_frames: Optional[int] = None
    shared_across_batch: bool = True
    condition_noise_std: float = 0.0

    def resolve_max_condition_frames(self, window: int) -> int:
        max_frames = window - 1 if self.max_condition_frames is None else self.max_condition_frames
        if not 0 <= self.min_condition_frames <= max_frames <= window - 1:
            raise ValueError(
                "Conditioning range must satisfy "
                "0 <= min_condition_frames <= max_condition_frames <= window - 1."
            )
        return max_frames


@dataclass
class ModelConfig:
    """2D U-Net and conditioning embedding settings."""

    time_embedding_dim: int = 128
    height_embedding_dim: int = 128
    use_ema: bool = True
    ema_decay: float = 0.999


@dataclass
class OptimConfig:
    """Optimizer hyperparameters."""

    lr: float = 2e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class SchedulerConfig:
    """Epoch-level LR scheduler settings."""

    name: str = "cosine"
    gamma: float = 0.99
    min_lr: float = 1e-6


@dataclass
class TrainConfig:
    """Loop and device settings."""

    batch_size: int = 8
    epochs: int = 100
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True
    device: str = "cuda"
    use_amp: bool = False
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 20
    checkpoint_every: int = 1
    output_dir: Path = Path("Diffusion_Ionosphere_2D") / "outputs"


@dataclass
class ExperimentConfig:
    """Top-level experiment bundle used by train.py."""

    diffusion: DiffusionConfig
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
