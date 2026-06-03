"""Universal amortised temporal conditioning utilities for 2D slices."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Generator, Tensor


def sample_conditioned_frames(
    batch_size: int,
    window: int,
    min_condition_frames: int = 1,
    max_condition_frames: Optional[int] = None,
    shared_across_batch: bool = True,
    device: Optional[torch.device] = None,
    generator: Optional[Generator] = None,
) -> Tensor:
    """Sample how many history frames are exposed for this batch."""

    if max_condition_frames is None:
        max_condition_frames = window - 1
    if not 0 <= min_condition_frames <= max_condition_frames <= window - 1:
        raise ValueError(
            "Expected 0 <= min_condition_frames <= max_condition_frames <= window - 1."
        )

    high = max_condition_frames + 1
    if shared_across_batch:
        value = torch.randint(
            low=min_condition_frames,
            high=high,
            size=(1,),
            device=device,
            generator=generator,
        )
        return value.expand(batch_size)

    return torch.randint(
        low=min_condition_frames,
        high=high,
        size=(batch_size,),
        device=device,
        generator=generator,
    )


def build_universal_condition_2d(
    clean_window: Tensor,
    conditioned_frames: Union[int, Tensor],
    data_channels: int,
    condition_noise_std: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Create the [mask, mask * values] conditioning tensor.

    Args:
        clean_window: Shape [B, window * C_phys, H, W].
        conditioned_frames: One int shared by the batch or a [B] tensor.
        data_channels: Number of physical variables per time step.
    """

    if clean_window.ndim != 4:
        raise ValueError("clean_window must have shape [B, window * C_phys, H, W].")
    if clean_window.shape[1] % data_channels != 0:
        raise ValueError("Channel axis must be divisible by data_channels.")

    batch_size, flattened_channels, _, _ = clean_window.shape
    window = flattened_channels // data_channels

    if isinstance(conditioned_frames, int):
        conditioned_frames = torch.full(
            (batch_size,),
            fill_value=conditioned_frames,
            device=clean_window.device,
            dtype=torch.long,
        )
    else:
        conditioned_frames = conditioned_frames.to(clean_window.device, dtype=torch.long)

    if conditioned_frames.shape != (batch_size,):
        raise ValueError("conditioned_frames must be an int or a tensor of shape [B].")

    mask = torch.zeros_like(clean_window)
    for batch_index, n_frames in enumerate(conditioned_frames.tolist()):
        if not 0 <= n_frames <= window - 1:
            raise ValueError("Each conditioned frame count must be within [0, window - 1].")
        observed_channels = n_frames * data_channels
        mask[batch_index, :observed_channels] = 1.0

    values = clean_window
    if condition_noise_std > 0.0:
        values = values + condition_noise_std * torch.randn_like(values)

    condition = torch.cat([mask, mask * values], dim=1)
    return condition, mask
