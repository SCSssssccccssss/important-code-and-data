"""Lightweight EMA implementation so the project stays self-contained."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import torch
import torch.nn as nn
from torch import Tensor


class ExponentialMovingAverage:
    """Tracks a shadow copy of trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: Dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {
            "decay": torch.tensor(self.decay),
            "shadow": {name: tensor.clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = float(state_dict["decay"])
        self.shadow = {
            name: tensor.clone() for name, tensor in state_dict["shadow"].items()
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.shadow[name].data)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.copy_to(model)
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                param.data.copy_(self.backup[name].data)
            self.backup = {}
