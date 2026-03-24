from __future__ import annotations

"""Medical code tokenizer for EHRSHOT data.

Builds a vocabulary from OMOP concept codes in the MEDS dataset
and maps them to integer token IDs.
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd


# Special tokens
PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
UNK_TOKEN = "[UNK]"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, UNK_TOKEN]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2


class EHRTokenizer:
    """Tokenizer for structured EHR codes."""

    def __init__(self, vocab: dict[str, int] | None = None):
        """
        Args:
            vocab: Optional pre-built vocabulary mapping code -> token_id.
        """
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def sos_id(self) -> int:
        return SOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def encode(self, codes: list[str]) -> list[int]:
        """Convert a list of medical codes to token IDs."""
        return [self.vocab.get(c, UNK_ID) for c in codes]

    def decode(self, token_ids: list[int]) -> list[str]:
        """Convert token IDs back to medical codes."""
        return [self.inv_vocab.get(tid, UNK_TOKEN) for tid in token_ids]

    @classmethod
    def build_from_meds(
        cls,
        meds_df: pd.DataFrame,
        min_count: int = 1,
    ) -> "EHRTokenizer":
        """Build vocabulary from MEDS DataFrame.

        Args:
            meds_df: MEDS DataFrame with a 'code' column.
            min_count: Minimum occurrence count to include a code.

        Returns:
            EHRTokenizer with built vocabulary.
        """
        code_counts = Counter(meds_df["code"].dropna().tolist())

        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        idx = len(SPECIAL_TOKENS)

        for code, count in sorted(code_counts.items()):
            if count >= min_count:
                vocab[code] = idx
                idx += 1

        print(f"Built vocabulary: {len(vocab)} tokens "
              f"({len(vocab) - len(SPECIAL_TOKENS)} medical codes + "
              f"{len(SPECIAL_TOKENS)} special tokens)")

        return cls(vocab=vocab)

    def save(self, path: str | Path):
        """Save vocabulary to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EHRTokenizer":
        """Load vocabulary from JSON file."""
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
