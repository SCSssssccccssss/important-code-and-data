"""Variance-preserving SDE utilities used during 2D training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Size, Tensor


class VPSDE(nn.Module):
    """Compact VP-SDE module adapted from the reference repo."""

    def __init__(
        self,
        eps_model: nn.Module,
        shape: Size,
        alpha: str = "cos",
        eta: float = 1e-3,
        model_type: str = "noise",
    ):
        super().__init__()
        self.net = eps_model
        self.shape = tuple(shape)
        self.eta = eta
        self.model_type = model_type
        self.dims = tuple(range(-len(self.shape), 0))

        if alpha == "lin":
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == "cos":
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == "exp":
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError(f"Unsupported alpha schedule: {alpha}")

        self.register_buffer("_dummy", torch.empty(()), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t).sqrt()

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.mu(t) ** 2 + self.eta**2).sqrt()

    def base_sample(self, reference: Tensor) -> Tensor:
        return torch.randn_like(reference)

    def noise_prediction_fn(self, x_t: Tensor, t: Tensor) -> Tensor:
        if self.model_type == "noise":
            return self.net(x_t, t)
        if self.model_type == "x_start":
            return (x_t - self.mu(t) * self.net(x_t, t)) / self.sigma(t)
        if self.model_type == "v_prediction":
            return self.mu(t) * self.net(x_t, t) + self.sigma(t) * x_t
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def data_prediction_fn(self, x_t: Tensor, t: Tensor) -> Tensor:
        noise = self.noise_prediction_fn(x_t, t)
        return (x_t - self.sigma(t) * noise) / self.mu(t)

    def forward_diffusion(self, x: Tensor, t: Tensor, return_target: bool = False):
        t_reshaped = t.reshape(t.shape + (1,) * len(self.shape))
        eps = self.base_sample(x)
        x_t = self.mu(t_reshaped) * x + self.sigma(t_reshaped) * eps

        if not return_target:
            return x_t

        if self.model_type == "noise":
            target = eps
        elif self.model_type == "x_start":
            target = x
        elif self.model_type == "v_prediction":
            target = self.mu(t_reshaped) * eps - self.sigma(t_reshaped) * x
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return x_t, target

    def loss(self, x: Tensor, return_details: bool = False):
        if x.shape[1:] != self.shape:
            raise ValueError(
                f"Expected event shape {self.shape}, got {tuple(x.shape[1:])}."
            )

        t = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
        x_t, target = self.forward_diffusion(x, t, return_target=True)
        prediction = self.net(x_t, t)

        if self.model_type == "noise":
            loss_weight = torch.ones_like(t)
        elif self.model_type == "x_start":
            loss_weight = (self.mu(t) / self.sigma(t)) ** 2
        elif self.model_type == "v_prediction":
            loss_weight = torch.ones_like(t)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        mse = (prediction - target).square().flatten(1).mean(1)
        loss = (loss_weight * mse).mean()

        if not return_details:
            return loss

        return loss, {
            "t": t.detach(),
            "x_t": x_t.detach(),
            "target": target.detach(),
            "prediction": prediction.detach(),
            "per_example_mse": mse.detach(),
        }
