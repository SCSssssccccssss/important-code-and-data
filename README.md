# Diffusion_Ionosphere_2D

Level-wise 2D universal-amortised diffusion for ionosphere electron-density data.

This follows the pressure-level decomposition idea used in the tropical-cyclone
paper: a 3D volume is split into 2D height slices during training, but each
slice keeps a scalar normalized height label.

## Data Shape

Single-variable electron density:

```text
raw data: [N, 96, Z, H, W]
example:  [31, 96, 82, 91, 91]
```

Each single-time height slice is:

```text
x_h: [1, 91, 91]
h_norm: scalar in [0, 1]
```

With temporal `window=8`, each diffusion training sample becomes:

```text
x_window: [8, 91, 91]
h_norm:   scalar
```

For universal amortised diffusion:

```text
x_t:           [B, window*C_phys, H, W]
condition:     [B, 2*window*C_phys, H, W]  # [mask, mask*x]
network input: [B, 3*window*C_phys, H, W]
height label:  [B]
output noise:  [B, window*C_phys, H, W]
```

## Training Smoke Test

Run from the repository root:

```bash
python Diffusion_Ionosphere_2D/test_train_2d.py
```

The smoke test uses small random data to avoid allocating the full real tensor.
For real training, replace the `raw_data` tensor in `test_train_2d.py` with your
loaded data shaped `[31, 96, 82, 91, 91]`, then increase model size, batch size,
and epochs as needed.

## Why Height Is A Condition

If height is split into independent 2D samples, the model must still know which
height layer it is denoising. This code follows the context-residual injection
style in `pdediff/nn/unet.py`: diffusion time `t` and normalized height `h_norm`
are embedded as scalar context vectors, fused, projected to each U-Net stage, and
added inside every residual block.

The denoiser therefore learns:

```text
epsilon_theta(x_t, t, h_norm, condition)
```

instead of learning one height-agnostic 2D distribution.
