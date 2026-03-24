#!/usr/bin/env python3
"""Pretrain TrajGPT on EHRSHOT data.

Trains TrajGPT from scratch using next-token prediction on patient
event sequences from the EHRSHOT MEDS dataset.

Usage:
    python scripts/03_pretrain_trajgpt.py --config configs/trajgpt_ehrshot.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ehrshot.data_loading import load_meds_dataset, build_patient_sequences
from models.trajgpt.model import TrajGPT
from models.trajgpt.heads import PretrainHead
from models.trajgpt.tokenizer import EHRTokenizer


class EHRPretrainDataset(Dataset):
    """Dataset for TrajGPT pretraining.

    Each sample is a patient's event sequence:
    - token_ids: encoded medical codes
    - timestamps: event times in days (relative to first event)
    """

    def __init__(
        self,
        patient_data: dict[int, dict],
        tokenizer: EHRTokenizer,
        max_seq_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        for subject_id, data in patient_data.items():
            codes = data["codes"]
            times = data["times"]

            if len(codes) < 2:  # need at least 2 events for next-token prediction
                continue

            # Encode codes to token IDs
            token_ids = tokenizer.encode(codes)

            # Convert timestamps to days relative to first event
            t0 = times[0]
            days = []
            for t in times:
                delta = t - t0
                if hasattr(delta, "total_seconds"):
                    days.append(delta.total_seconds() / 86400.0)
                else:
                    # Already a numeric type (e.g., from pandas Timedelta)
                    days.append(float(delta) / 1e9 / 86400.0 if hasattr(delta, '__float__') else 0.0)

            # Truncate to max_seq_len
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[-max_seq_len:]
                days = days[-max_seq_len:]

            self.samples.append({
                "token_ids": token_ids,
                "timestamps": days,
                "length": len(token_ids),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function with padding."""
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


def load_official_splits(assets_dir: str) -> tuple[set[int], set[int], set[int]]:
    """Load official EHRSHOT patient splits from EHRSHOT_ASSETS."""
    split_path = Path(assets_dir) / "splits" / "person_id_map.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Official split file not found: {split_path}")

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


def main():
    parser = argparse.ArgumentParser(description="Pretrain TrajGPT")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cpu")

    # Load data
    print("Loading MEDS dataset...")
    meds_df = load_meds_dataset(config["meds_data_dir"])
    print(f"  {len(meds_df)} events, {meds_df['subject_id'].nunique()} patients")

    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = EHRTokenizer.build_from_meds(meds_df)

    # Save tokenizer
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(checkpoint_dir / "tokenizer.json")

    # Build patient sequences
    print("Building patient sequences...")
    patient_data = build_patient_sequences(meds_df, max_length=config.get("max_seq_len", 256))

    # Use official EHRSHOT patient splits
    assets_dir = config.get("assets_dir")
    if not assets_dir:
        raise ValueError("Missing `assets_dir` in config; required for official EHRSHOT splits")

    train_ids, val_ids, test_ids = load_official_splits(assets_dir)
    available_ids = set(patient_data.keys())
    train_ids = train_ids & available_ids
    val_ids = val_ids & available_ids
    test_ids = test_ids & available_ids

    train_data = {k: v for k, v in patient_data.items() if k in train_ids}
    val_data = {k: v for k, v in patient_data.items() if k in val_ids}
    test_data = {k: v for k, v in patient_data.items() if k in test_ids}

    print(
        f"  Official splits: Train={len(train_data)} patients, "
        f"Val={len(val_data)} patients, Test={len(test_data)} patients"
    )

    # Create datasets
    max_seq_len = config.get("max_seq_len", 256)
    train_dataset = EHRPretrainDataset(train_data, tokenizer, max_seq_len)
    val_dataset = EHRPretrainDataset(val_data, tokenizer, max_seq_len)

    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("pretrain_batch_size", 32),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("pretrain_batch_size", 32),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Initialize model
    model = TrajGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 200),
        qk_dim=config.get("d_model", 200),
        v_dim=config.get("v_dim", 400),
        ff_dim=config.get("ff_dim", 400),
        num_layers=config.get("num_layers", 8),
        num_heads=config.get("num_heads", 4),
        tau=config.get("tau", 20.0),
        dropout=config.get("dropout", 0.1),
        max_seq_len=max_seq_len,
        pad_id=tokenizer.pad_id,
        sos_id=tokenizer.sos_id,
    ).to(device)

    pretrain_head = PretrainHead(
        d_model=config.get("d_model", 200),
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    ).to(device)

    total_params = model.count_parameters() + sum(
        p.numel() for p in pretrain_head.parameters() if p.requires_grad
    )
    print(f"\nModel parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Optimizer
    all_params = list(model.parameters()) + list(pretrain_head.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=config.get("pretrain_lr", 3e-5),
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Learning rate scheduler with warmup
    warmup_steps = config.get("warmup_steps", 500)
    total_steps = config.get("pretrain_epochs", 20) * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.1, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    forward_impl = config.get("forward_impl", "parallel")
    num_epochs = config.get("pretrain_epochs", 20)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)

    # Training loop
    print(f"\nStarting pretraining for {num_epochs} epochs...")
    print(f"  Forward: {forward_impl}, Device: {device}")
    print(f"  Batch size: {config.get('pretrain_batch_size', 32)}, "
          f"Grad accum: {grad_accum_steps}")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        pretrain_head.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            token_ids = batch["token_ids"].to(device)
            timestamps = batch["timestamps"].to(device)

            loss, _ = model.pretrain_forward(
                token_ids, timestamps, pretrain_head, forward_impl
            )
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps
            num_batches += 1

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation
        model.eval()
        pretrain_head.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch["token_ids"].to(device)
                timestamps = batch["timestamps"].to(device)

                loss, _ = model.pretrain_forward(
                    token_ids, timestamps, pretrain_head, forward_impl
                )
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1:>2d}/{num_epochs}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed:.1f}s")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "pretrain_head_state_dict": pretrain_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "config": config,
            }, checkpoint_dir / "best_model.pt")
            print(f"  → Saved best model (val_loss={avg_val_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "pretrain_head_state_dict": pretrain_head.state_dict(),
                "val_loss": avg_val_loss,
            }, checkpoint_dir / f"checkpoint_epoch{epoch+1}.pt")

    # Save final model
    torch.save({
        "epoch": num_epochs - 1,
        "model_state_dict": model.state_dict(),
        "pretrain_head_state_dict": pretrain_head.state_dict(),
        "val_loss": avg_val_loss,
        "config": config,
    }, checkpoint_dir / "final_model.pt")

    print(f"\nPretraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
