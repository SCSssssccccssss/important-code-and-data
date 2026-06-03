"""Amortised score wrapper for height-conditioned 2D backbones."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class AmortizedHeightScoreModel(nn.Module):
    """Universal amortised score model for 2D height slices.

    Expected shapes:
        x_t:       [B, window * C_phys, H, W]
        condition: [B, 2 * window * C_phys, H, W]
        height:    [B] normalized height labels in [0, 1]
        backbone:  [B, 3 * window * C_phys, H, W], t, height
                   -> [B, window * C_phys, H, W]
    """

    def __init__(
        self,
        backbone: nn.Module,
        noisy_channels: int,
        condition_channels: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.noisy_channels = noisy_channels
        self.condition_channels = condition_channels
        self.condition: Optional[Tensor] = None
        self.height: Optional[Tensor] = None

    def set_context(self, condition: Tensor, height: Tensor) -> None:
        self.set_condition(condition)
        self.set_height(height)

    def set_condition(self, condition: Tensor) -> None:
        if condition.ndim != 4:
            raise ValueError("condition must have shape [B, 2 * window * C_phys, H, W].")
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"Expected {self.condition_channels} condition channels, got {condition.shape[1]}."
            )
        self.condition = condition

    def set_height(self, height: Tensor) -> None:
        height = height.reshape(-1).float()
        self.height = height

    def clear_context(self) -> None:
        self.condition = None
        self.height = None

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        if self.condition is None or self.height is None:
            raise RuntimeError("Call score_model.set_context(condition, height) before forward().")
        if x_t.ndim != 4:
            raise ValueError("x_t must have shape [B, window * C_phys, H, W].")
        if x_t.shape[1] != self.noisy_channels:
            raise ValueError(
                f"Expected {self.noisy_channels} noisy channels, got {x_t.shape[1]}."
            )

        condition = self.condition
        height = self.height.to(device=x_t.device, dtype=x_t.dtype)

        if condition.shape[0] == 1 and x_t.shape[0] > 1:
            condition = condition.expand(x_t.shape[0], -1, -1, -1)
        elif condition.shape[0] != x_t.shape[0]:
            raise ValueError("Condition batch size must match x_t batch size.")

        if height.shape[0] == 1 and x_t.shape[0] > 1:
            height = height.expand(x_t.shape[0])
        elif height.shape[0] != x_t.shape[0]:
            raise ValueError("Height batch size must match x_t batch size.")

        model_input = torch.cat([condition.to(x_t.device, x_t.dtype), x_t], dim=1)
        prediction = self.backbone(model_input, t, height)
        if prediction.shape != x_t.shape:
            raise ValueError(
                "Backbone output shape must match the noisy window shape. "
                f"Expected {tuple(x_t.shape)}, got {tuple(prediction.shape)}."
            )
        return prediction
