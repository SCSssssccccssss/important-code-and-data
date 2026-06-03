"""2D U-Net with pdediff-style time injection and height conditioning."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class PDEDiffScalarEmbedding(nn.Module):
    """Scalar embedding matching `pdediff.nn.embedding.TimeEmbedding`.

    The original repo maps a scalar diffusion time `t` through fixed
    cos/sin frequencies and a small MLP. We use the same form for both
    diffusion time and normalized height.
    """

    def __init__(self, features: int):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive.")
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, features),
        )
        self.register_buffer("freqs", torch.pi / 2 * 1e3 ** torch.linspace(0, 1, 64))

    def forward(self, value: Tensor) -> Tensor:
        value = value.reshape(-1).float()
        freqs = self.freqs.to(device=value.device, dtype=value.dtype)
        angles = freqs * value.unsqueeze(dim=-1)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.net(embedding)


class ContextResidualBlock2D(nn.Module):
    """Residual block with context injection following `pdediff.nn.unet`.

    Original pattern:
        x + residue(x + project(context))

    Here `context` contains both diffusion time and normalized height.
    """

    def __init__(self, channels: int, context_dim: int, dropout: float = 0.0):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(context_dim, channels),
            nn.Unflatten(-1, (-1, 1, 1)),
        )
        self.residue = nn.Sequential(
            nn.GroupNorm(_num_groups(channels), channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        return x + self.residue(x + self.project(context))


class UpsampleTail2D(nn.Module):
    """Nearest-neighbor upsample followed by a convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.norm = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, target_shape: tuple[int, int] | None = None) -> Tensor:
        x = F.silu(self.norm(x))
        x = F.interpolate(x, scale_factor=self.stride, mode="nearest")
        x = self.conv(x)
        if target_shape is not None and x.shape[-2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode="nearest")
        return x


class HeightConditionedUNet2D(nn.Module):
    """2D denoiser D_theta(x_t, t, h_norm).

    Input channels already include amortised condition channels:
        [mask, mask * x_condition, x_t]

    The time injection follows the reference repo's `pdediff/nn/unet.py`:
    every residual block receives a projected context vector and adds it to the
    feature map before the residual convolution. Height is embedded with the
    same scalar embedding style and fused with time as a shared context.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        hidden_blocks: Sequence[int] | int = 2,
        time_embedding_dim: int = 128,
        height_embedding_dim: int = 128,
        dropout: float = 0.0,
        stride: int = 2,
    ):
        super().__init__()
        if not channel_multipliers:
            raise ValueError("channel_multipliers must contain at least one value.")
        if isinstance(hidden_blocks, int):
            hidden_blocks = tuple([hidden_blocks] * len(channel_multipliers))
        if len(hidden_blocks) != len(channel_multipliers):
            raise ValueError("hidden_blocks must match channel_multipliers length.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding = PDEDiffScalarEmbedding(time_embedding_dim)
        self.height_embedding = PDEDiffScalarEmbedding(height_embedding_dim)

        context_dim = max(time_embedding_dim, height_embedding_dim)
        self.time_to_context = nn.Linear(time_embedding_dim, context_dim)
        self.height_to_context = nn.Linear(height_embedding_dim, context_dim)
        self.context_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

        hidden_channels = [base_channels * multiplier for multiplier in channel_multipliers]

        heads: list[nn.Module] = []
        tails: list[nn.Module] = []
        descent: list[nn.ModuleList] = []
        ascent: list[nn.ModuleList] = []

        for level, num_blocks in enumerate(hidden_blocks):
            channels = hidden_channels[level]
            if level == 0:
                heads.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
                tails.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
            else:
                heads.append(
                    nn.Conv2d(
                        hidden_channels[level - 1],
                        channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                    )
                )
                tails.append(
                    UpsampleTail2D(
                        in_channels=channels,
                        out_channels=hidden_channels[level - 1],
                        stride=stride,
                    )
                )

            descent.append(
                nn.ModuleList(
                    ContextResidualBlock2D(channels, context_dim, dropout)
                    for _ in range(num_blocks)
                )
            )
            ascent.append(
                nn.ModuleList(
                    ContextResidualBlock2D(channels, context_dim, dropout)
                    for _ in range(num_blocks)
                )
            )

        self.heads = nn.ModuleList(heads)
        self.tails = nn.ModuleList(reversed(tails))
        self.descent = nn.ModuleList(descent)
        self.ascent = nn.ModuleList(reversed(ascent))

    def _context_embedding(self, t: Tensor, height: Tensor) -> Tensor:
        time_context = self.time_to_context(self.time_embedding(t))
        height_context = self.height_to_context(self.height_embedding(height))
        return self.context_mlp(time_context + height_context)

    def forward(self, x: Tensor, t: Tensor, height: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, C, H, W].")

        context = self._context_embedding(t=t, height=height)
        memory: list[Tensor] = []

        for head, blocks in zip(self.heads, self.descent):
            x = head(x)
            for block in blocks:
                x = block(x, context)
            memory.append(x)

        memory.pop()

        for blocks, tail in zip(self.ascent, self.tails):
            for block in blocks:
                x = block(x, context)

            if memory:
                skip = memory.pop()
                x = tail(x, target_shape=skip.shape[-2:]) + skip
            else:
                x = tail(x)

        return x
