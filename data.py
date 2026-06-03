"""Data helpers for level-wise 2D ionosphere diffusion.

The main dataset follows the tropical-cyclone paper's idea: a 3D volume is
decomposed into independent 2D height slices during training, but each slice
keeps a scalar normalized height label.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


class HeightSliceWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Lazy level-wise 2D sliding-window dataset.

    Supported raw input formats:
        [N, T, Z, H, W] for single-variable electron density
        [N, T, C_phys, Z, H, W] for multiple physical variables

    Returned sample:
        x_h:    [window * C_phys, H, W]
        h_norm: scalar tensor in [0, 1]

    This deliberately does not materialize all windows. For real data shaped
    [31, 96, 82, 91, 91], materializing all windows would create more than
    200k samples and waste memory.
    """

    def __init__(
        self,
        data: Tensor,
        window: int,
        height_values: Optional[Tensor] = None,
    ):
        if data.ndim not in {5, 6}:
            raise ValueError(
                "Expected data with shape [N, T, Z, H, W] or [N, T, C, Z, H, W]."
            )
        if window <= 0:
            raise ValueError("window must be positive.")
        if data.shape[1] < window:
            raise ValueError("T must be at least as large as window.")

        self.data = data
        self.window = window
        self.has_phys_channel = data.ndim == 6
        self.num_sequences = data.shape[0]
        self.num_starts_per_sequence = data.shape[1] - window + 1
        self.num_levels = data.shape[3] if self.has_phys_channel else data.shape[2]
        self.height_norm = self._build_height_norm(height_values)

    def _build_height_norm(self, height_values: Optional[Tensor]) -> Tensor:
        if height_values is None:
            return torch.linspace(0.0, 1.0, self.num_levels, dtype=torch.float32)

        values = torch.as_tensor(height_values, dtype=torch.float32).reshape(-1)
        if values.numel() != self.num_levels:
            raise ValueError(
                f"height_values must contain {self.num_levels} values, got {values.numel()}."
            )
        min_value = values.min()
        value_range = values.max() - min_value
        if float(value_range) == 0.0:
            return torch.zeros_like(values)
        return (values - min_value) / value_range

    @property
    def data_channels(self) -> int:
        return self.data.shape[2] if self.has_phys_channel else 1

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return tuple(self.data.shape[-2:])

    def __len__(self) -> int:
        return self.num_sequences * self.num_levels * self.num_starts_per_sequence

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start_idx = index % self.num_starts_per_sequence
        level_idx = (index // self.num_starts_per_sequence) % self.num_levels
        sequence_idx = index // (self.num_starts_per_sequence * self.num_levels)

        if self.has_phys_channel:
            window = self.data[
                sequence_idx,
                start_idx : start_idx + self.window,
                :,
                level_idx,
            ]
            x_h = window.flatten(0, 1).contiguous()
        else:
            window = self.data[
                sequence_idx,
                start_idx : start_idx + self.window,
                level_idx,
            ]
            x_h = window.unsqueeze(1).flatten(0, 1).contiguous()

        return x_h.float(), self.height_norm[level_idx].clone()


def split_dataset_indices(
    num_samples: int,
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[Tensor, Tensor]:
    """Create deterministic train/validation indices."""

    if num_samples < 2:
        raise ValueError("At least two samples are required for a train/valid split.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(num_samples, generator=generator)
    else:
        indices = torch.arange(num_samples)

    train_count = int(num_samples * train_ratio)
    train_count = min(max(train_count, 1), num_samples - 1)
    return indices[:train_count], indices[train_count:]


def make_height_2d_train_valid_datasets(
    data: Tensor,
    window: int,
    train_ratio: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
    height_values: Optional[Tensor] = None,
) -> tuple[Subset, Subset, HeightSliceWindowDataset]:
    """Build train/valid subsets from raw ionosphere data."""

    dataset = HeightSliceWindowDataset(
        data=data,
        window=window,
        height_values=height_values,
    )
    train_indices, valid_indices = split_dataset_indices(
        num_samples=len(dataset),
        train_ratio=train_ratio,
        seed=seed,
        shuffle=shuffle,
    )
    return Subset(dataset, train_indices.tolist()), Subset(dataset, valid_indices.tolist()), dataset


def build_dataloader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    """Thin helper so train.py stays focused on orchestration."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
