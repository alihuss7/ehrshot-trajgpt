from __future__ import annotations

"""Selective Recurrent Attention (SRA) module.

The core innovation of TrajGPT. SRA combines linear attention with a
data-dependent decay mechanism to adaptively forget irrelevant past
information based on context.

Recurrent form (Eq. 2):
    S_n = γ_n * S_{n-1} + K_n^T @ V_n
    O_n = Q_n @ S_n
    γ_n = Sigmoid(X_n @ w_γ^T)^{1/τ},  τ=20

Parallel form (Eq. 3):
    O = (Q @ K^T ⊙ D) @ V
    D_nm = b_n/b_m (n≥m), 0 (n<m)
    b_n = ∏_{t=1}^{n} γ_t

Multi-head extension (Eq. 4):
    Each head h has its own w_γ^h capturing different clinical dynamics.

Reference: Song et al., "TrajGPT: Irregular Time-Series Representation
Learning of Health Trajectory", IEEE JBHI 2025.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trajgpt.xpos import XPOS


class SelectiveRecurrentAttention(nn.Module):
    """Multi-head Selective Recurrent Attention module.

    Matches the official TrajGPT implementation with:
    - Combined QKV projection (single linear layer)
    - XPOS position encoding (asymmetric Q/K scaling)
    - SiLU gating (not sigmoid)
    - GroupNorm with affine=False
    - 1/√d_k scaling on attention scores
    """

    def __init__(
        self,
        d_model: int,
        qk_dim: int,
        v_dim: int,
        num_heads: int,
        tau: float = 20.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.tau = tau
        self.head_qk_dim = qk_dim // num_heads
        self.head_v_dim = v_dim // num_heads
        self.scale = self.head_qk_dim ** -0.5

        # Combined QKV projection (matches official)
        self.qkv = nn.Linear(d_model, qk_dim * 2 + v_dim, bias=False)

        # Data-dependent decay projection: hidden_states -> per-head gamma
        self.gamma_proj = nn.Linear(d_model, num_heads, bias=False)

        # Output gating with SiLU activation (matches official)
        self.gated = nn.Linear(d_model, v_dim, bias=False)
        self.silu = nn.SiLU()

        # Output projection
        self.out_proj = nn.Linear(v_dim, d_model, bias=False)

        # Group normalization per head, no learnable affine (matches official)
        self.gn = nn.GroupNorm(
            num_groups=num_heads, num_channels=v_dim, affine=False
        )

        # XPOS position encoding (matches official, replaces simple RoPE)
        self.xpos = XPOS(self.head_qk_dim)

        self.dropout = nn.Dropout(dropout)

    def _compute_gamma(self, X: torch.Tensor) -> torch.Tensor:
        """Compute data-dependent decay γ_n for each head.

        γ_n^h = Sigmoid(X_n @ w_γ^{hT})^{1/τ}

        Args:
            X: Input embeddings (batch, seq_len, d_model).

        Returns:
            Decay values (batch, num_heads, seq_len) in (0, 1].
        """
        # (B, N, d_model) -> (B, N, num_heads)
        gamma = self.gamma_proj(X)
        # γ = Sigmoid(gamma)^{1/τ}
        gamma = torch.sigmoid(gamma).pow(1.0 / self.tau)
        # (B, N, H) -> (B, H, N)
        return gamma.permute(0, 2, 1)

    def _project_qkv(self, X: torch.Tensor):
        """Combined QKV projection and reshape to multi-head format."""
        B, N, _ = X.shape
        qkv = self.qkv(X)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.v_dim], dim=-1)
        q = q.view(B, N, self.num_heads, self.head_qk_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, self.head_qk_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_v_dim).permute(0, 2, 1, 3)
        return q, k, v

    def forward_parallel(
        self,
        X: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel forward pass for training.

        O = (Q @ K^T * scale ⊙ D) @ V

        Args:
            X: Input embeddings (batch, seq_len, d_model).
            timestamps: Timestamps (batch, seq_len) in days.

        Returns:
            Output (batch, seq_len, d_model).
        """
        B, N, _ = X.shape

        # QKV projection
        Q, K, V = self._project_qkv(X)

        # Apply XPOS with irregular timestamps
        Q, K = self.xpos.rotate_queries_and_keys(Q, K, timestamps)

        # Compute data-dependent decay
        gamma = self._compute_gamma(X)  # (B, H, N)

        # Build cumulative decay in log-space for numerical stability
        log_gamma = torch.log(gamma + 1e-8)  # (B, H, N)
        log_b = torch.cumsum(log_gamma, dim=-1)  # (B, H, N)

        # Causal decay matrix: D_nm = exp(log_b[n] - log_b[m]) for n >= m.
        # Mask before exp to avoid overflow on the non-causal branch during backward.
        log_D = log_b.unsqueeze(-1) - log_b.unsqueeze(-2)  # (B, H, N, N)
        causal_mask = torch.tril(torch.ones(N, N, device=X.device, dtype=torch.bool))
        log_D = log_D.masked_fill(~causal_mask, float("-inf"))
        log_D = torch.clamp(log_D, min=-80.0, max=0.0)
        D = torch.exp(log_D)

        # Scaled attention with decay mask
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn * D
        O = torch.matmul(attn, V)  # (B, H, N, head_v_dim)

        # Reshape: (B, H, N, head_v_dim) -> (B, N, v_dim)
        O = O.permute(0, 2, 1, 3).contiguous().view(B, N, self.v_dim)

        # GroupNorm (affine=False)
        O = self.gn(O.transpose(1, 2)).transpose(1, 2)

        # SiLU gating (matches official)
        O = self.silu(self.gated(X)) * O

        # Output projection
        O = self.out_proj(O)
        O = self.dropout(O)

        return O

    def forward_recurrent(
        self,
        x_n: torch.Tensor,
        timestamp_n: torch.Tensor,
        state: torch.Tensor | None = None,
        prev_timestamp: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent forward pass for inference (O(1) per step).

        S_n = γ_n * S_{n-1} + K_n^T @ V_n
        O_n = Q_n @ S_n * scale
        """
        B = x_n.shape[0]

        Q, K, V = self._project_qkv(x_n)

        # Apply XPOS
        Q, K = self.xpos.rotate_queries_and_keys(Q, K, timestamp_n)

        # Compute decay
        gamma = self._compute_gamma(x_n)  # (B, H, 1)
        gamma = gamma.unsqueeze(-1)  # (B, H, 1, 1)

        # K^T @ V
        kv = torch.matmul(K.transpose(-2, -1), V)

        # State update: S_n = γ_n * S_{n-1} + K_n^T @ V_n
        if state is None:
            new_state = kv
        else:
            new_state = gamma * state + kv

        # Output: O_n = Q_n @ S_n * scale
        O = torch.matmul(Q, new_state) * self.scale

        # Reshape + norm + gating + projection
        O = O.permute(0, 2, 1, 3).contiguous().view(B, 1, self.v_dim)
        O = self.gn(O.transpose(1, 2)).transpose(1, 2)
        O = self.silu(self.gated(x_n)) * O
        O = self.out_proj(O)

        return O, new_state

    def forward(
        self,
        X: torch.Tensor,
        timestamps: torch.Tensor,
        forward_impl: str = "parallel",
    ) -> torch.Tensor:
        """Forward pass dispatching to parallel or recurrent implementation."""
        if forward_impl == "parallel":
            return self.forward_parallel(X, timestamps)
        elif forward_impl == "recurrent":
            B, N, _ = X.shape
            state = None
            outputs = []
            prev_t = None
            for i in range(N):
                x_i = X[:, i:i+1, :]
                t_i = timestamps[:, i:i+1]
                out, state = self.forward_recurrent(x_i, t_i, state, prev_t)
                outputs.append(out)
                prev_t = t_i
            return torch.cat(outputs, dim=1)
        else:
            raise ValueError(f"Unknown forward_impl: {forward_impl}")


class SRABlock(nn.Module):
    """SRA block: LayerNorm -> SRA -> Residual -> LayerNorm -> MLP -> Residual."""

    def __init__(
        self,
        d_model: int,
        qk_dim: int,
        v_dim: int,
        ff_dim: int,
        num_heads: int,
        tau: float = 20.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.sra = SelectiveRecurrentAttention(
            d_model=d_model,
            qk_dim=qk_dim,
            v_dim=v_dim,
            num_heads=num_heads,
            tau=tau,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model, bias=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        X: torch.Tensor,
        timestamps: torch.Tensor,
        forward_impl: str = "parallel",
    ) -> torch.Tensor:
        X = X + self.sra(self.norm1(X), timestamps, forward_impl)
        X = X + self.mlp(self.norm2(X))
        return X
