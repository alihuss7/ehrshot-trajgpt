from __future__ import annotations

"""TrajGPT core model for EHRSHOT adaptation.

This implementation follows the official TrajGPT repo block structure:
- token embedding + learnable SOS
- stacked SRA blocks
- final layer norm
"""

import torch
import torch.nn as nn

from models.trajgpt.heads import PretrainHead
from models.trajgpt.sra import SRABlock


class TrajGPT(nn.Module):
    """TrajGPT model for irregularly sampled EHR trajectories."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 200,
        qk_dim: int = 200,
        v_dim: int = 400,
        ff_dim: int = 800,
        num_layers: int = 8,
        num_heads: int = 4,
        tau: float = 20.0,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        pad_id: int = 0,
        sos_id: int = 1,
        forecast_method: str = "time_specific",
    ):
        super().__init__()

        if forecast_method not in {"time_specific", "auto_regressive"}:
            raise ValueError(
                f"Invalid forecast_method={forecast_method}. "
                "Expected one of {'time_specific', 'auto_regressive'}."
            )

        self.d_model = d_model
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.max_seq_len = max_seq_len
        self.forecast_method = forecast_method

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Official repo uses learnable SOS embedding parameter.
        self.sos = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList(
            [
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
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

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
        X = self.token_embedding(token_ids)

        for layer in self.layers:
            layer_out = layer(X, timestamps, forward_impl=forward_impl)
            X = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return self.final_norm(X)

    def pretrain_forward(
        self,
        token_ids: torch.Tensor,
        timestamps: torch.Tensor,
        pretrain_head: PretrainHead,
        forward_impl: str = "parallel",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = token_ids.shape

        token_emb = self.token_embedding(token_ids[:, :-1])
        sos_emb = self.sos.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        X = torch.cat([sos_emb, token_emb], dim=1)

        t0 = timestamps[:, :1]
        input_timestamps = torch.cat([t0, timestamps[:, :-1]], dim=1)

        for layer in self.layers:
            layer_out = layer(X, input_timestamps, forward_impl=forward_impl)
            X = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        hidden_states = self.final_norm(X)

        loss, logits = pretrain_head(hidden_states, token_ids)
        return loss, logits

    def extract_representations(
        self,
        token_ids: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
        forward_impl: str = "parallel",
    ) -> torch.Tensor:
        """Return time-specific (last valid step) hidden-state representation."""
        hidden_states = self.forward(token_ids, timestamps, forward_impl)

        if mask is not None:
            lengths = mask.sum(dim=1).long()
            indices = (lengths - 1).clamp(min=0)
            return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), indices]

        return hidden_states[:, -1, :]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
