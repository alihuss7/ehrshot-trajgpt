#!/usr/bin/env python3
"""Extract CLMBR-T-base embeddings for all EHRSHOT tasks.

Usage:
    python scripts/01_extract_embeddings.py --config configs/clmbr_ehrshot.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ehrshot.tasks import get_tasks
from ehrshot.data_loading import load_meds_dataset, load_task_labels
from models.embedder import CLMBRBaseEmbedder


def main():
    parser = argparse.ArgumentParser(description="Extract CLMBR-T-base embeddings")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load MEDS dataset
    print("Loading MEDS dataset...")
    meds_df = load_meds_dataset(config["meds_data_dir"])
    print(f"  Loaded {len(meds_df)} events, {meds_df['subject_id'].nunique()} patients")

    # Collect all unique (subject_id, prediction_time) pairs across tasks
    tasks = get_tasks(config.get("tasks", "all"))
    all_pairs = set()
    task_labels = {}

    for task in tasks:
        print(f"Loading labels for {task.name}...")
        labels_df = load_task_labels(config["labels_dir"], task)
        task_labels[task.name] = labels_df
        pairs = set(zip(labels_df["subject_id"], labels_df["prediction_time"]))
        all_pairs.update(pairs)
        print(f"  {len(labels_df)} labels, {len(pairs)} unique (patient, time) pairs")

    prediction_times = sorted(all_pairs)
    print(
        f"\nTotal unique (patient, time) pairs across all tasks: {len(prediction_times)}"
    )

    # Build patient data dict
    print("Building patient sequences...")
    from ehrshot.data_loading import build_patient_sequences

    patient_data = build_patient_sequences(meds_df)

    # Initialize embedder
    embedder = CLMBRBaseEmbedder(
        model_hub_id=config["model_hub_id"],
        device=config.get("device", "cpu"),
        batch_size=config.get("batch_size", 64),
    )

    # Extract embeddings
    print(f"\nExtracting {len(prediction_times)} embeddings...")
    embeddings = embedder.embed_patients(patient_data, prediction_times)

    # Save
    output_dir = Path(config["embedding_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)
    index_df = pd.DataFrame(prediction_times, columns=["subject_id", "prediction_time"])
    index_df.to_parquet(output_dir / "index.parquet", index=False)

    print(f"\nSaved embeddings: {embeddings.shape} to {output_dir}/")
    print(f"  embeddings.npy: {embeddings.shape}")
    print(f"  index.parquet: {len(index_df)} rows")


if __name__ == "__main__":
    main()
