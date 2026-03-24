from __future__ import annotations

"""TrajGPT: Trajectory Generative Pre-trained Transformer.

Full model implementation following the paper architecture (Table V):
- 8 decoder (SRA) layers
- 4 attention heads
- Q/K dim: 200, V dim: 400, FF dim: 400
- ~7.5M parameters
- Token embedding + [SOS] prepending
- L × SRA_Block
- Final LayerNorm
- Output head (pretrain or classification)

Reference: Song et al., "TrajGPT: Irregular Time-Series Representation
Learning of Health Trajectory", IEEE JBHI 2025.
"""

import torch
import torch.nn as nn

from models.trajgpt.sra import SRABlock
from models.trajgpt.heads import PretrainHead, ClfHead


class TrajGPT(nn.Module):
    """TrajGPT model for irregularly-sampled EHR time series."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 200,
        qk_dim: int = 200,
        v_dim: int = 400,
        ff_dim: int = 400,
        num_layers: int = 8,
        num_heads: int = 4,
        tau: float = 20.0,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_id: int = 0,
        sos_id: int = 1,
    ):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            d_model: Model hidden dimension (also Q/K dim).
            qk_dim: Query/Key projection dimension.
            v_dim: Value projection dimension.
            ff_dim: Feed-forward hidden dimension.
            num_layers: Number of SRA blocks.
            num_heads: Number of attention heads.
            tau: Temperature for data-dependent decay.
            dropout: Dropout rate.
            max_seq_len: Maximum sequence length.
            pad_id: Padding token ID.
            sos_id: Start-of-sequence token ID.
        """
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Learnable SOS token (matches official: nn.Parameter + normal init)
        self.sos = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.sos)

        # SRA blocks
        self.layers = nn.ModuleList([
            SRABlock(
                d_model=d_model,
                qk_dim=qk_dim,
                v_dim=v_dim,
                ff_dim=ff_dim,
                num_heads=num_heads,
                tau=tau,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        timestamps: torch.Tensor,
        forward_impl: str = "parallel",
    ) -> torch.Tensor:
        """Forward pass through the TrajGPT encoder.

        Args:
            token_ids: Input token IDs (batch, seq_len).
            timestamps: Timestamps in days (batch, seq_len).
            forward_impl: "parallel" for training, "recurrent" for inference.

        Returns:
            Hidden states (batch, seq_len, d_model).
        """
        # Token embedding
        X = self.token_embedding(token_ids)  # (B, N, d_model)

        # Pass through SRA blocks
        for layer in self.layers:
            X = layer(X, timestamps, forward_impl)

        # Final normalization
        X = self.final_norm(X)

        return X

    def pretrain_forward(
        self,
        token_ids: torch.Tensor,
        timestamps: torch.Tensor,
        pretrain_head: PretrainHead,
        forward_impl: str = "parallel",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for pretraining with next-token prediction.

        Prepends [SOS] token and shifts targets right.

        Args:
            token_ids: Input token IDs (batch, seq_len).
            timestamps: Timestamps (batch, seq_len).
            pretrain_head: PretrainHead module.
            forward_impl: Forward implementation mode.

        Returns:
            (loss, logits).
        """
        B, N = token_ids.shape
        device = token_ids.device

        # Embed input tokens (right-shifted: drop last, prepend SOS)
        token_emb = self.token_embedding(token_ids[:, :-1])  # (B, N-1, d_model)

        # Prepend learnable SOS embedding (matches official)
        sos_emb = self.sos.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, d_model)
        X = torch.cat([sos_emb, token_emb], dim=1)  # (B, N, d_model)

        # Timestamps: prepend first timestamp for SOS position
        t0 = timestamps[:, :1]
        input_timestamps = torch.cat([t0, timestamps[:, :-1]], dim=1)

        # Forward through SRA blocks (bypass token_embedding since we already embedded)
        for layer in self.layers:
            X = layer(X, input_timestamps, forward_impl)
        hidden_states = self.final_norm(X)

        # Compute loss against original tokens (targets)
        loss, logits = pretrain_head(hidden_states, token_ids)

        return loss, logits

    def extract_representations(
        self,
        token_ids: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
        forward_impl: str = "parallel",
    ) -> torch.Tensor:
        """Extract patient representations (embeddings).

        Uses the last valid hidden state as the patient representation.

        Args:
            token_ids: Input token IDs (batch, seq_len).
            timestamps: Timestamps (batch, seq_len).
            mask: Optional boolean mask (batch, seq_len), True for valid tokens.
            forward_impl: Forward implementation mode.

        Returns:
            Representations (batch, d_model).
        """
        hidden_states = self.forward(token_ids, timestamps, forward_impl)

        if mask is not None:
            # Get last valid position for each sequence
            lengths = mask.sum(dim=1).long()  # (B,)
            # Gather the hidden state at the last valid position
            indices = (lengths - 1).clamp(min=0)
            representations = hidden_states[torch.arange(hidden_states.size(0)), indices]
        else:
            # Use last position
            representations = hidden_states[:, -1, :]

        return representations

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
