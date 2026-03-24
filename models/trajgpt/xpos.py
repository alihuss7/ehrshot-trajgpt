from __future__ import annotations

"""XPOS: Position encoding with learnable scaling for TrajGPT.

Adapted from Microsoft TorchScale's XPOS implementation. Unlike standard
RoPE, XPOS applies asymmetric scaling to Q and K for better extrapolation:
- Q gets upscaled (scale)
- K gets downscaled (1/scale)

For TrajGPT, token indices are replaced with actual timestamps to handle
irregularly-sampled time series.

Reference: Sun et al., "A Length-Extrapolatable Transformer" (2022)
"""

import torch
import torch.nn as nn
import math


class XPOS(nn.Module):
    """XPOS position encoding with learnable scaling for irregular timestamps.

    Matches the official TrajGPT implementation:
    - Sinusoidal rotation (RoPE-style)
    - Learnable scale factors with asymmetric Q/K scaling
    - Supports offset for recurrent inference
    """

    def __init__(self, head_dim: int, base: float = 10000.0, scale_base: int = 512):
        """
        Args:
            head_dim: Per-head dimension (must be even).
            base: Base for rotation frequencies.
            scale_base: Base for scale decay computation.
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim

        # Rotation frequencies: θ_j = base^{-2j/d}
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Learnable scale factors (from XPOS paper)
        scale = (torch.arange(0, head_dim, 2).float() + 0.4 * head_dim) / (1.4 * head_dim)
        self.register_buffer("scale", scale)
        self.scale_base = scale_base

    def _compute_sin_cos(self, timestamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sin/cos rotation values from timestamps.

        Args:
            timestamps: (batch, seq_len) in consistent time units.

        Returns:
            (sin, cos) each of shape (batch, seq_len, head_dim/2).
        """
        # timestamps: (B, N) -> (B, N, 1)
        t = timestamps.unsqueeze(-1).float()
        # freqs: (B, N, head_dim/2)
        freqs = t * self.inv_freq.unsqueeze(0).unsqueeze(0)
        return torch.sin(freqs), torch.cos(freqs)

    def _compute_scale(self, timestamps: torch.Tensor, downscale: bool = False) -> torch.Tensor:
        """Compute position-dependent scale factors.

        Args:
            timestamps: (batch, seq_len).
            downscale: If True, invert scale (used for K).

        Returns:
            Scale factors (batch, seq_len, head_dim/2).
        """
        t = timestamps.unsqueeze(-1).float()
        # Compute in log-space and clamp exponent to keep gradients finite.
        # log(scale^(t/scale_base)) = (t/scale_base) * log(scale)
        log_scale = torch.log(self.scale).unsqueeze(0).unsqueeze(0)  # (1,1,head_dim/2)
        exp_arg = (t / self.scale_base) * log_scale
        if downscale:
            exp_arg = -exp_arg
        exp_arg = torch.clamp(exp_arg, min=-12.0, max=12.0)
        return torch.exp(exp_arg)

    def rotate_queries_and_keys(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply XPOS rotation with asymmetric scaling.

        Q is upscaled, K is downscaled (for better length extrapolation).

        Args:
            q: Query (batch, heads, seq_len, head_dim).
            k: Key (batch, heads, seq_len, head_dim).
            timestamps: (batch, seq_len) timestamps.

        Returns:
            Rotated (q, k).
        """
        sin_vals, cos_vals = self._compute_sin_cos(timestamps)
        q_scale = self._compute_scale(timestamps, downscale=False)
        k_scale = self._compute_scale(timestamps, downscale=True)

        # Expand for heads: (B, N, dim/2) -> (B, 1, N, dim/2)
        sin_vals = sin_vals.unsqueeze(1)
        cos_vals = cos_vals.unsqueeze(1)
        q_scale = q_scale.unsqueeze(1)
        k_scale = k_scale.unsqueeze(1)

        q_rot = self._apply_rotation(q, cos_vals, sin_vals, q_scale)
        k_rot = self._apply_rotation(k, cos_vals, sin_vals, k_scale)

        return q_rot, k_rot

    def _apply_rotation(
        self,
        x: torch.Tensor,
        cos_vals: torch.Tensor,
        sin_vals: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding with scaling.

        x1' = (x1 * cos - x2 * sin) * scale
        x2' = (x1 * sin + x2 * cos) * scale
        """
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x1_rot = (x1 * cos_vals - x2 * sin_vals) * scale
        x2_rot = (x1 * sin_vals + x2 * cos_vals) * scale

        out = torch.stack([x1_rot, x2_rot], dim=-1)
        return out.flatten(-2)
