from __future__ import annotations

"""Selective Recurrent Attention module for TrajGPT."""

import torch
import torch.nn as nn

from models.trajgpt.xpos import XPOS


class SelectiveRecurrentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        qk_dim: int,
        v_dim: int,
        num_heads: int,
        tau: float = 20.0,
        dropout: float = 0.0,
        use_bias_in_sra: bool = False,
        use_bias_in_sra_out: bool = False,
        use_default_gamma: bool = False,
    ):
        super().__init__()
        _ = dropout
        if qk_dim % num_heads != 0:
            raise ValueError(f"qk_dim={qk_dim} must be divisible by num_heads={num_heads}")
        if v_dim % num_heads != 0:
            raise ValueError(f"v_dim={v_dim} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.tau = tau

        self.head_qk_dim = qk_dim // num_heads
        self.head_v_dim = v_dim // num_heads

        self.qkv = nn.Linear(d_model, qk_dim * 2 + v_dim, bias=use_bias_in_sra)
        self.gamma_proj = nn.Linear(d_model, num_heads, bias=True)
        self.silu = nn.SiLU()
        self.gated = nn.Linear(d_model, v_dim, bias=False)
        self.out_proj = nn.Linear(v_dim, d_model, bias=use_bias_in_sra_out)

        self.gn = nn.GroupNorm(num_groups=num_heads, num_channels=v_dim, affine=False)
        # Apply XPOS before splitting heads, dimension=qk_dim.
        self.xpos = XPOS(qk_dim)
        if use_default_gamma:
            gamma = 1 - 2 ** (-5 - torch.arange(0, num_heads, dtype=torch.float32))
        else:
            s = torch.log(torch.tensor(1 / 64, dtype=torch.float32))
            e = torch.log(torch.tensor(1 / 512, dtype=torch.float32))
            gamma = 1 - torch.exp(torch.linspace(s, e, num_heads))
        self.decay = nn.Parameter(gamma, requires_grad=False)

    def _split_heads(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = q.view(B, T, self.num_heads, self.head_qk_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.num_heads, self.head_qk_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_heads, self.head_v_dim).permute(0, 2, 1, 3)
        return q, k, v

    def get_data_dependent_decay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (B, T, H)
        gamma = self.gamma_proj(hidden_states)
        return torch.sigmoid(gamma).pow(1.0 / self.tau)

    def get_parallel_decay_mask(
        self,
        t: torch.Tensor,
        decay: torch.Tensor,
        retention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = t
        # decay: (B, T, H)
        B, T, _ = decay.shape
        b = torch.cumprod(decay, dim=1)
        ratio = b.unsqueeze(2) / b.unsqueeze(1)
        D = ratio.permute(0, 3, 1, 2)  # (B, H, T, T)
        decay_mask = torch.tril(D, diagonal=0)

        if retention_mask is not None:
            retention_mask = retention_mask.float().view(B, 1, 1, T)
            decay_mask = decay_mask * retention_mask
        return decay_mask

    def parallel_retention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = k.size(-1) ** -0.5
        retention = torch.matmul(q, k.transpose(-1, -2)) * scale
        retention = retention * decay_mask
        output = torch.matmul(retention, v)

        current_kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        intra_decay = decay_mask[:, :, -1, :, None, None]
        current_kv = (current_kv * intra_decay).sum(2)

        return output, current_kv, retention

    def recurrent_retention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: torch.Tensor | None = None,
        decay: torch.Tensor | None = None,
        retention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, _, _ = q.shape
        past_key_value = 0 if past_key_value is None else past_key_value
        decay = 0 if decay is None else decay
        decay = decay.view(B, H, 1, 1)

        if retention_mask is None:
            retention_mask = 1
        else:
            retention_mask = retention_mask.view(B, 1, 1, 1)

        current_kv = decay * past_key_value + retention_mask * torch.matmul(k.transpose(-1, -2), v)
        output = torch.matmul(q, current_kv) * (k.size(-1) ** -0.5)
        return output, current_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        retention_mask: torch.Tensor | None = None,
        past_key_value: torch.Tensor | None = None,
        forward_impl: str = "parallel",
        sequence_offset: int = 0,
        output_retentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        B, T, _ = hidden_states.size()

        q, k, v = self.qkv(hidden_states).split([self.qk_dim, self.qk_dim, self.v_dim], dim=-1)
        q, k = self.xpos.rotate_queries_and_keys(q, k, offset=sequence_offset)
        q, k, v = self._split_heads(q, k, v, B, T)
        decay = self.get_data_dependent_decay(hidden_states)

        if forward_impl == "parallel":
            decay_mask = self.get_parallel_decay_mask(t=t, decay=decay, retention_mask=retention_mask)
            retention_out, curr_kv, retention_weights = self.parallel_retention(q, k, v, decay_mask)
        elif forward_impl == "recurrent":
            retention_weights = None
            for n in range(T):
                gamma_n = decay[:, n, :]
                retention_mask_n = retention_mask[:, n] if retention_mask is not None else None
                retention_out, curr_kv = self.recurrent_retention(
                    q,
                    k,
                    v,
                    past_key_value=past_key_value,
                    decay=gamma_n,
                    retention_mask=retention_mask_n,
                )
        else:
            raise ValueError(f"Unknown forward_impl: {forward_impl}")

        retention_out = retention_out.transpose(1, 2).contiguous().view(B, T, self.v_dim)
        normed = self.gn(retention_out.view(B * T, self.v_dim)).view(B, T, self.v_dim)
        out = self.silu(self.gated(hidden_states)) * normed

        outputs = (self.out_proj(out), curr_kv)
        if output_retentions:
            outputs += (retention_weights,)
        return outputs


class SRABlock(nn.Module):
    """LayerNorm -> SRA -> residual -> LayerNorm -> FFN -> residual."""

    def __init__(
        self,
        d_model: int,
        qk_dim: int,
        v_dim: int,
        ff_dim: int,
        num_heads: int,
        tau: float = 20.0,
        dropout: float = 0.0,
        use_bias_in_sra: bool = False,
        use_bias_in_mlp: bool = True,
        use_bias_in_sra_out: bool = False,
        use_default_gamma: bool = False,
    ):
        super().__init__()
        _ = dropout  # kept for API compatibility; FFN block has no dropout.

        self.norm1 = nn.LayerNorm(d_model)
        self.sra = SelectiveRecurrentAttention(
            d_model=d_model,
            qk_dim=qk_dim,
            v_dim=v_dim,
            num_heads=num_heads,
            tau=tau,
            dropout=0.0,
            use_bias_in_sra=use_bias_in_sra,
            use_bias_in_sra_out=use_bias_in_sra_out,
            use_default_gamma=use_default_gamma,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=use_bias_in_mlp),
            nn.GELU(),
            nn.Linear(ff_dim, d_model, bias=use_bias_in_mlp),
        )

    def forward(
        self,
        X: torch.Tensor,
        timestamps: torch.Tensor,
        retention_mask: torch.Tensor | None = None,
        forward_impl: str = "parallel",
        past_key_value: torch.Tensor | None = None,
        sequence_offset: int = 0,
        output_retentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        sra_outs = self.sra(
            self.norm1(X),
            t=timestamps,
            retention_mask=retention_mask,
            past_key_value=past_key_value,
            forward_impl=forward_impl,
            sequence_offset=sequence_offset,
            output_retentions=output_retentions,
        )
        sra = sra_outs[0]
        curr_kv = sra_outs[1]
        x = X + sra
        y = x + self.mlp(self.norm2(x))

        outputs = (y, curr_kv)
        if output_retentions:
            outputs += (sra_outs[2],)
        return outputs
