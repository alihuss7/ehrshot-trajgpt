from __future__ import annotations

"""Output heads for TrajGPT.

- PretrainHead: Next-token prediction (cross-entropy loss)
- ForecastHead: Autoregressive/forecast token prediction head
- ClfHead: Classification with average pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainHead(nn.Module):
    """Next-token prediction head for autoregressive pretraining.

    Projects hidden states to vocabulary logits and computes
    cross-entropy loss with right-shifted targets.
    """

    def __init__(self, d_model: int, vocab_size: int, pad_id: int = 0):
        super().__init__()
        # +1 to include padding token output
        self.proj = nn.Linear(d_model, vocab_size + 1)
        self.pad_id = pad_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, d_model) — model output.
            targets: (batch, seq_len) — right-shifted input token IDs.

        Returns:
            (loss, logits): Cross-entropy loss and vocab logits.
        """
        logits = self.proj(hidden_states)  # (B, N, vocab_size)

        # Flatten for cross-entropy
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.pad_id,
        )
        return loss, logits


class ForecastHead(nn.Module):
    """Forecasting head."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size + 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class ClfHead(nn.Module):
    """Classification head with average pooling.

    Pools sequence representations and projects to class logits.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model).
            mask: Optional (batch, seq_len) boolean mask (True = valid token).

        Returns:
            Logits (batch, num_classes).
        """
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
            pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        return self.proj(pooled)
