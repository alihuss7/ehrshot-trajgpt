#!/usr/bin/env python3
"""Extract TrajGPT embeddings in CLMBR-compatible feature format.

Uses TrajGPTEmbedder to extract embeddings for all 406K (patient, time)
pairs in the EHRSHOT benchmark, then saves in the same pickle format
as clmbr_features.pkl for use with scripts/02_run_evaluation.py.

Usage:
    python scripts/04_extract_trajgpt_embeddings.py --config configs/trajgpt_ehrshot.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ehrshot.data_loading import build_patient_sequences, load_meds_dataset
from models.trajgpt.config import TrajGPTConfig
from models.trajgpt_embedder import TrajGPTEmbedder


def main():
    parser = argparse.ArgumentParser(
        description="Extract TrajGPT embeddings (CLMBR-compatible output)"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = TrajGPTConfig.load_yaml(args.config).to_dict()

    assets_dir = Path(config["assets_dir"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_path = str(
        Path(args.checkpoint) if args.checkpoint else checkpoint_dir / "best_model.pt"
    )
    tokenizer_path = str(checkpoint_dir / "tokenizer.json")

    # ── Load MEDS data and build patient sequences ──
    print("Loading MEDS dataset...")
    t0 = time.time()
    meds_df = load_meds_dataset(config["meds_data_dir"])
    print(
        f"  {len(meds_df)} events, {meds_df['subject_id'].nunique()} patients ({time.time()-t0:.1f}s)"
    )

    print("Building patient sequences...")
    t0 = time.time()
    patient_data = build_patient_sequences(meds_df, max_length=None)
    print(f"  {len(patient_data)} patients ({time.time()-t0:.1f}s)")
    del meds_df  # free memory

    # ── Load benchmark label index ──
    print("Loading all_labels.csv...")
    labels_df = pd.read_csv(assets_dir / "benchmark" / "all_labels.csv")
    labels_df["prediction_time"] = pd.to_datetime(labels_df["prediction_time"])
    prediction_times = [
        (int(pid), ts.to_pydatetime())
        for pid, ts in zip(labels_df["patient_id"], labels_df["prediction_time"])
    ]
    print(f"  {len(prediction_times)} label rows")

    # ── Initialize embedder ──
    embedder = TrajGPTEmbedder(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=config.get("device", "cpu"),
        batch_size=config.get("embedding_batch_size", 64),
        max_seq_len=config.get("max_seq_len", 256),
    )

    # Pre-encode all patient tokens once (avoids re-encoding per label row)
    print("Pre-encoding patient tokens...")
    t0 = time.time()
    embedder.precompute_patient_tokens(patient_data)
    print(f"  Done ({time.time()-t0:.1f}s)")
    del patient_data  # free memory

    # ── Extract embeddings ──
    print(f"\nExtracting {len(prediction_times)} embeddings...")
    embeddings = embedder.embed_patients({}, prediction_times)

    # ── Save in CLMBR-compatible pickle format ──
    output_dir = Path(config["embedding_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    features = {
        "data_path": config["meds_data_dir"],
        "model": checkpoint_path,
        "data_matrix": embeddings.astype(np.float16),
        "patient_ids": labels_df["patient_id"].to_numpy(dtype=np.int64),
        "labeling_time": np.array(
            [t.to_pydatetime() for t in labels_df["prediction_time"]],
            dtype=object,
        ),
        "label_values": labels_df["value"].to_numpy(),
    }

    features_path = output_dir / "trajgpt_features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved to {features_path}")
    print(
        f"  data_matrix: {features['data_matrix'].shape} ({features['data_matrix'].dtype})"
    )
    print(f"\nRun evaluation with:")
    print(
        f"  python scripts/02_run_evaluation.py --assets_dir {assets_dir} "
        f"--features {features_path} --model_name trajgpt --output_dir {config['results_dir']}"
    )


if __name__ == "__main__":
    main()
