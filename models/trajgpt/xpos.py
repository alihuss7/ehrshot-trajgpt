from __future__ import annotations

"""XPOS module for TrajGPT."""

import torch
import torch.nn as nn


def fixed_pos_embedding(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, device=x.device, dtype=torch.float32) / dim))
    sinusoid_inp = torch.einsum(
        "i,j->ij",
        torch.arange(0, seq_len, device=x.device, dtype=torch.float32),
        inv_freq,
    ).to(x)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def duplicate_interleave(m: torch.Tensor) -> torch.Tensor:
    dim0 = m.shape[0]
    m = m.view(-1, 1)
    m = m.repeat(1, 2)
    m = m.view(dim0, -1)
    return m


def apply_rotary_pos_emb(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    scale: torch.Tensor | float = 1,
) -> torch.Tensor:
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    """XPOS with offset-based, asymmetric Q/K scaling."""

    def __init__(self, head_dim: int, scale_base: int = 512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale",
            (torch.arange(0, head_dim, 2, dtype=torch.float32) + 0.4 * head_dim)
            / (1.4 * head_dim),
        )

    def forward(self, x: torch.Tensor, offset: int = 0, downscale: bool = False) -> torch.Tensor:
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos

        scale = self.scale ** (
            torch.arange(min_pos, max_pos, 1, device=self.scale.device, dtype=torch.float32)
            .div(self.scale_base)[:, None]
        )
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x

    def rotate_queries_and_keys(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.forward(q, offset=offset, downscale=False)
        k = self.forward(k, offset=offset, downscale=True)
        return q, k
