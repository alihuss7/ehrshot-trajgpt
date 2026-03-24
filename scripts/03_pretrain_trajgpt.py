#!/usr/bin/env python3
"""Pretrain TrajGPT on EHRSHOT data.

Configuration choices:
- One sequence per patient (tail-truncated to max_seq_len)
- Code-token vocabulary only
- Warmup + linear decay scheduler
- EHRSHOT train/val/test split usage
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ehrshot.data_loading import build_patient_sequences, load_meds_dataset
from models.trajgpt.config import TrajGPTConfig
from models.trajgpt.heads import PretrainHead
from models.trajgpt.model import TrajGPT
from models.trajgpt.tokenizer import EHRTokenizer


class EHRPretrainDataset(Dataset):
    """One sequence sample per patient."""

    def __init__(
        self,
        patient_data: dict[int, dict],
        tokenizer: EHRTokenizer,
        max_seq_len: int = 256,
    ):
        self.samples = []

        for _, data in patient_data.items():
            codes = data["codes"]
            times = data["times"]

            if len(codes) < 2:
                continue

            token_ids = tokenizer.encode(codes)

            # Tail-truncate to max_seq_len
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[-max_seq_len:]
                times = times[-max_seq_len:]

            t0 = times[0]
            days = []
            for t in times:
                delta = t - t0
                if hasattr(delta, "total_seconds"):
                    days.append(delta.total_seconds() / 86400.0)
                else:
                    days.append(float(delta) / 1e9 / 86400.0 if hasattr(delta, "__float__") else 0.0)

            self.samples.append(
                {
                    "token_ids": token_ids,
                    "timestamps": days,
                    "length": len(token_ids),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(item["length"] for item in batch)

    token_ids = []
    timestamps = []
    masks = []

    for item in batch:
        n = item["length"]
        pad_len = max_len - n

        token_ids.append(item["token_ids"] + [0] * pad_len)
        timestamps.append(item["timestamps"] + [0.0] * pad_len)
        masks.append([True] * n + [False] * pad_len)

    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "timestamps": torch.tensor(timestamps, dtype=torch.float32),
        "mask": torch.tensor(masks, dtype=torch.bool),
    }


def resolve_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_ehrshot_splits(assets_dir: str) -> tuple[set[int], set[int], set[int]]:
    split_path = Path(assets_dir) / "splits" / "person_id_map.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"EHRSHOT split file not found: {split_path}")

    split_df = pd.read_csv(split_path)
    required_cols = {"split", "omop_person_id"}
    if not required_cols.issubset(split_df.columns):
        raise ValueError(
            f"Split file must contain columns {required_cols}, got {set(split_df.columns)}"
        )

    train_ids = set(split_df.loc[split_df["split"] == "train", "omop_person_id"].astype(int))
    val_ids = set(split_df.loc[split_df["split"] == "val", "omop_person_id"].astype(int))
    test_ids = set(split_df.loc[split_df["split"] == "test", "omop_person_id"].astype(int))
    return train_ids, val_ids, test_ids


def build_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    warmup_steps = max(0, warmup_steps)
    total_steps = max(1, total_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Pretrain TrajGPT")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = TrajGPTConfig.load_yaml(args.config)
    config = cfg.to_dict()

    device = resolve_device(config.get("device", "auto"))

    print("Loading MEDS dataset...")
    meds_df = load_meds_dataset(config["meds_data_dir"])
    print(f"  {len(meds_df)} events, {meds_df['subject_id'].nunique()} patients")

    print("Building tokenizer...")
    tokenizer = EHRTokenizer.build_from_meds(
        meds_df,
        min_count=int(config.get("tokenizer_min_count", 1)),
        max_vocab_size=config.get("tokenizer_max_vocab_size", None),
    )

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(checkpoint_dir / "tokenizer.json")

    print("Building patient sequences...")
    max_seq_len = int(config.get("max_seq_len", 256))
    patient_data = build_patient_sequences(meds_df, max_length=max_seq_len)

    assets_dir = config.get("assets_dir")
    if not assets_dir:
        raise ValueError("Missing `assets_dir` in config; required for EHRSHOT splits")

    train_ids, val_ids, test_ids = load_ehrshot_splits(assets_dir)
    available_ids = set(patient_data.keys())
    train_ids = train_ids & available_ids
    val_ids = val_ids & available_ids
    test_ids = test_ids & available_ids

    train_data = {k: v for k, v in patient_data.items() if k in train_ids}
    val_data = {k: v for k, v in patient_data.items() if k in val_ids}

    print(
        f"  EHRSHOT splits: Train={len(train_data)} patients, "
        f"Val={len(val_data)} patients, Test={len(test_ids)} patients"
    )

    train_dataset = EHRPretrainDataset(train_data, tokenizer, max_seq_len)
    val_dataset = EHRPretrainDataset(val_data, tokenizer, max_seq_len)

    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    batch_size = int(config.get("pretrain_batch_size", 32))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = TrajGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=int(config.get("d_model", 200)),
        qk_dim=int(config.get("qk_dim", config.get("d_model", 200))),
        v_dim=int(config.get("v_dim", 400)),
        ff_dim=int(config.get("ffn_proj_size", config.get("ff_dim", 800))),
        num_layers=int(config.get("num_layers", 8)),
        num_heads=int(config.get("num_heads", 4)),
        tau=float(config.get("tau", 20.0)),
        dropout=float(config.get("dropout", 0.0)),
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_id,
        sos_id=tokenizer.sos_id,
        forecast_method=str(config.get("forecast_method", "time_specific")),
        use_bias_in_sra=bool(config.get("use_bias_in_sra", False)),
        use_bias_in_mlp=bool(config.get("use_bias_in_mlp", True)),
        use_bias_in_sra_out=bool(config.get("use_bias_in_sra_out", False)),
        use_default_gamma=bool(config.get("use_default_gamma", False)),
        output_retentions=bool(config.get("output_retentions", False)),
        use_cache=bool(config.get("use_cache", True)),
        forward_impl=str(config.get("forward_impl", "parallel")),
    ).to(device)

    pretrain_head = PretrainHead(
        d_model=int(config.get("d_model", 200)),
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    ).to(device)

    total_params = model.count_parameters() + sum(
        p.numel() for p in pretrain_head.parameters() if p.requires_grad
    )
    print(f"\nModel parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    all_params = list(model.parameters()) + list(pretrain_head.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=float(config.get("pretrain_lr", 3e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    num_epochs = int(config.get("pretrain_epochs", 20))
    warmup_steps = int(config.get("warmup_steps", 500))
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = build_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    forward_impl = str(config.get("forward_impl", "parallel"))

    print(f"\nStarting pretraining for {num_epochs} epochs...")
    print(f"  Forward: {forward_impl}, Device: {device}")

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        pretrain_head.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in train_loader:
            token_ids = batch["token_ids"].to(device)
            timestamps = batch["timestamps"].to(device)

            optimizer.zero_grad()
            loss, _ = model.pretrain_forward(token_ids, timestamps, pretrain_head, forward_impl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        model.eval()
        pretrain_head.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch["token_ids"].to(device)
                timestamps = batch["timestamps"].to(device)
                loss, _ = model.pretrain_forward(token_ids, timestamps, pretrain_head, forward_impl)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1:>2d}/{num_epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={elapsed:.1f}s"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "pretrain_head_state_dict": pretrain_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": config,
                },
                checkpoint_dir / "best_model.pt",
            )
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "pretrain_head_state_dict": pretrain_head.state_dict(),
                    "val_loss": avg_val_loss,
                },
                checkpoint_dir / f"checkpoint_epoch{epoch+1}.pt",
            )

    torch.save(
        {
            "epoch": num_epochs - 1,
            "model_state_dict": model.state_dict(),
            "pretrain_head_state_dict": pretrain_head.state_dict(),
            "val_loss": avg_val_loss,
            "config": config,
        },
        checkpoint_dir / "final_model.pt",
    )

    print(f"\nPretraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
