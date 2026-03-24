from __future__ import annotations

"""TrajGPT configuration compatibility layer.

Keeps key TrajGPT fields while allowing
EHRSHOT-specific pipeline keys in one YAML.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrajGPTConfig:
    # Core model config
    num_layers: int = 8
    num_heads: int = 4
    d_model: int = 200
    qk_dim: int = 200
    v_dim: int = 400
    tau: float = 20.0
    ffn_proj_size: int = 800
    dropout: float = 0.0
    max_seq_len: int = 256
    use_bias_in_sra: bool = False
    use_bias_in_mlp: bool = True
    use_bias_in_sra_out: bool = False
    use_default_gamma: bool = False
    output_retentions: bool = False
    use_cache: bool = True
    forward_impl: str = "parallel"
    forecast_method: str = "time_specific"

    # EHRSHOT pipeline fields
    model_name: str = "trajgpt"
    meds_data_dir: str = "data/EHRSHOT_MEDS/data"
    assets_dir: str = "data/EHRSHOT_ASSETS"
    pretrain_epochs: int = 20
    pretrain_lr: float = 3e-5
    pretrain_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_steps: int = 500
    tokenizer_min_count: int = 1
    tokenizer_max_vocab_size: int | None = None
    checkpoint_dir: str = "results/trajgpt_gen3_2026-03-24/trajgpt/checkpoints"
    embedding_output_dir: str = "results/trajgpt_gen3_2026-03-24/trajgpt/embeddings"
    embedding_batch_size: int = 64
    results_dir: str = "results/trajgpt_gen3_2026-03-24/trajgpt"
    device: str = "auto"

    @property
    def ff_dim(self) -> int:
        return int(self.ffn_proj_size)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajGPTConfig":
        payload = dict(data)
        if "ffn_proj_size" not in payload and "ff_dim" in payload:
            payload["ffn_proj_size"] = payload["ff_dim"]
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load_yaml(cls, path: str | Path) -> "TrajGPTConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ff_dim"] = int(self.ffn_proj_size)
        return out
