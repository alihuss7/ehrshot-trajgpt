#!/usr/bin/env python3
"""Run EHRSHOT few-shot evaluation using precomputed embeddings.

Uses the EHRSHOT_ASSETS benchmark data:
- Precomputed CLMBR features from features/clmbr_features.pkl
- Per-task labels from benchmark/<task>/labeled_patients.csv
- Pre-generated few-shot splits from benchmark/<task>/all_shots_data.json
- Patient train/val/test splits from splits/person_id_map.csv

Usage:
    python scripts/02_run_evaluation.py --assets_dir data/EHRSHOT_ASSETS
    python scripts/02_run_evaluation.py --assets_dir data/EHRSHOT_ASSETS --tasks guo_los guo_readmission
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_features(features_path: str) -> dict:
    """Load precomputed feature embeddings from pickle."""
    import pickle
    with open(features_path, "rb") as f:
        return pickle.load(f)


def build_feature_lookup(feats: dict) -> dict[tuple[int, str], int]:
    """Build (patient_id, prediction_time_str) -> feature row index lookup."""
    lookup = {}
    for i in range(len(feats["patient_ids"])):
        pid = int(feats["patient_ids"][i])
        t = feats["labeling_time"][i]
        t_str = t.strftime("%Y-%m-%d %H:%M:%S")
        lookup[(pid, t_str)] = i
    return lookup


def match_task_embeddings(
    task_labels: pd.DataFrame,
    feat_lookup: dict,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match task labels to precomputed embeddings.

    Returns arrays aligned with task_labels row order.
    X_all[i] is the embedding for task_labels row i.
    """
    n = len(task_labels)
    dim = embeddings.shape[1]
    X_all = np.zeros((n, dim), dtype=np.float32)
    y_all = np.empty(n, dtype=object)
    matched = np.zeros(n, dtype=bool)

    for i, (_, row) in enumerate(task_labels.iterrows()):
        pid = int(row["patient_id"])
        t_str = row["prediction_time"].replace("T", " ")
        key = (pid, t_str)
        if key in feat_lookup:
            X_all[i] = embeddings[feat_lookup[key]].astype(np.float32)
            val = row["value"]
            # Convert string booleans to int
            if isinstance(val, str):
                y_all[i] = 1 if val == "True" else 0
            elif isinstance(val, bool):
                y_all[i] = int(val)
            else:
                y_all[i] = val
            matched[i] = True

    y_all = y_all.astype(int)
    return X_all, y_all, matched


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_type: str = "boolean",
) -> dict[str, float]:
    """Train LR on train set, evaluate AUROC/AUPRC on test set."""
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    is_multiclass = label_type == "categorical" or n_classes > 2

    clf = LogisticRegression(
        max_iter=10000,
        solver="lbfgs",
        random_state=42,
        n_jobs=1,
        multi_class="multinomial" if is_multiclass else "auto",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)

    try:
        if is_multiclass:
            # OvR macro AUROC/AUPRC for multiclass
            classes = clf.classes_
            y_test_bin = label_binarize(y_test, classes=classes)
            # If binarize collapsed (only 2 classes), reshape
            if y_test_bin.ndim == 1:
                y_test_bin = np.column_stack([1 - y_test_bin, y_test_bin])
            valid = y_test_bin.sum(axis=0) > 0
            if valid.sum() < 2:
                return {"auroc": float("nan"), "auprc": float("nan")}
            auroc = roc_auc_score(y_test_bin[:, valid], y_score[:, valid], average="macro")
            auprc = average_precision_score(y_test_bin[:, valid], y_score[:, valid], average="macro")
        else:
            scores = y_score[:, 1] if y_score.ndim == 2 else y_score
            auroc = roc_auc_score(y_test, scores)
            auprc = average_precision_score(y_test, scores)
    except ValueError:
        return {"auroc": float("nan"), "auprc": float("nan")}

    return {"auroc": auroc, "auprc": auprc}


def main():
    parser = argparse.ArgumentParser(description="Run EHRSHOT few-shot evaluation")
    parser.add_argument("--assets_dir", type=str, default="data/EHRSHOT_ASSETS")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to features pickle (default: assets_dir/features/clmbr_features.pkl)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Tasks to evaluate (default: all)")
    parser.add_argument("--k_shots", nargs="*", type=str, default=None,
                        help="k values to evaluate (default: all available)")
    parser.add_argument("--model_name", type=str, default="clmbr-t-base")
    parser.add_argument("--output_dir", type=str, default="results/clmbr-t-base")
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    benchmark_dir = assets_dir / "benchmark"
    splits_path = assets_dir / "splits" / "person_id_map.csv"

    # Load features
    features_path = args.features or str(assets_dir / "features" / "clmbr_features.pkl")
    print(f"Loading features from {features_path}...")
    feats = load_features(features_path)
    embeddings = feats["data_matrix"]
    print(f"  Embeddings: {embeddings.shape} ({embeddings.dtype})")

    # Build feature lookup
    print("Building feature lookup...")
    feat_lookup = build_feature_lookup(feats)
    print(f"  {len(feat_lookup)} (patient, time) entries")

    # Load patient splits
    splits_df = pd.read_csv(splits_path)
    test_pids = set(splits_df[splits_df["split"] == "test"]["omop_person_id"])
    print(f"  Test patients: {len(test_pids)}")

    # Discover tasks
    all_tasks = sorted([
        d.name for d in benchmark_dir.iterdir()
        if d.is_dir() and (d / "labeled_patients.csv").exists()
    ])
    tasks = args.tasks if args.tasks else all_tasks
    print(f"\nTasks to evaluate: {tasks}")

    output_dir = Path(args.output_dir)
    all_results = []

    for task_name in tasks:
        task_dir = benchmark_dir / task_name
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        # Load task labels
        task_labels = pd.read_csv(task_dir / "labeled_patients.csv")
        print(f"  Labels: {len(task_labels)}, type: {task_labels['label_type'].iloc[0]}")
        print(f"  Value dist: {task_labels['value'].value_counts().to_dict()}")

        # Match embeddings (aligned with task_labels row order)
        X_all, y_all, matched = match_task_embeddings(task_labels, feat_lookup, embeddings)
        n_matched = matched.sum()
        print(f"  Matched embeddings: {n_matched}/{len(task_labels)}")

        if n_matched == 0:
            print(f"  WARNING: No embeddings matched, skipping")
            continue

        # Identify test rows (test-split patients)
        test_mask = task_labels["patient_id"].isin(test_pids).values & matched
        test_rows = np.where(test_mask)[0]
        X_test = X_all[test_rows]
        y_test = y_all[test_rows]
        print(f"  Test set: {len(X_test)} examples")

        if len(X_test) < 10:
            print(f"  WARNING: Too few test examples, skipping")
            continue

        # Load pre-generated few-shot splits
        shots_path = task_dir / "all_shots_data.json"
        with open(shots_path) as f:
            shots_data = json.load(f)
        task_key = list(shots_data.keys())[0]
        shots = shots_data[task_key]

        # Determine k values
        available_ks = sorted(shots.keys(), key=lambda x: int(x))
        if args.k_shots:
            k_values = [k for k in args.k_shots if k in available_ks]
        else:
            k_values = available_ks

        for k_str in k_values:
            k = int(k_str)
            replicates = shots[k_str]

            rep_results = []
            for rep_id in sorted(replicates.keys()):
                rep = replicates[rep_id]
                train_idxs = rep["train_idxs"]

                # Filter to matched rows only
                valid_train = [i for i in train_idxs if matched[i]]
                if len(valid_train) < 2:
                    continue

                X_train = X_all[valid_train]
                y_train = y_all[valid_train]

                # Need at least 2 classes
                if len(np.unique(y_train)) < 2:
                    continue

                label_type = task_labels["label_type"].iloc[0]
                metrics = train_and_evaluate(X_train, y_train, X_test, y_test, label_type)
                rep_results.append(metrics)

            if not rep_results:
                print(f"  k={k:>4}: no valid replicates")
                continue

            aurocs = [r["auroc"] for r in rep_results if not np.isnan(r["auroc"])]
            auprcs = [r["auprc"] for r in rep_results if not np.isnan(r["auprc"])]

            auroc_mean = np.mean(aurocs) if aurocs else float("nan")
            auprc_mean = np.mean(auprcs) if auprcs else float("nan")
            auroc_std = np.std(aurocs) if aurocs else float("nan")
            auprc_std = np.std(auprcs) if auprcs else float("nan")

            label = "all" if k == -1 else str(k)
            print(f"  k={label:>4}: AUROC={auroc_mean:.4f}+/-{auroc_std:.4f}  "
                  f"AUPRC={auprc_mean:.4f}+/-{auprc_std:.4f}  ({len(rep_results)} reps)")

            all_results.append({
                "model": args.model_name,
                "task": task_name,
                "k": k,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
                "n_replicates": len(rep_results),
                "n_test": len(X_test),
            })

    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "model_name": args.model_name,
            "assets_dir": str(assets_dir),
            "features_path": str(features_path),
            "tasks": tasks,
            "k_shots": args.k_shots if args.k_shots else "all",
            "summary_path": str(summary_path),
            "n_rows": int(len(summary_df)),
        }
        with open(output_dir / "evaluation_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\nSummary saved to {summary_path}")
        print(f"\n{'='*60}")
        print("Best AUROC per task:")
        print(f"{'='*60}")
        for task_name in summary_df["task"].unique():
            task_df = summary_df[summary_df["task"] == task_name]
            valid = task_df.dropna(subset=["auroc_mean"])
            if valid.empty:
                print(f"  {task_name:>25s}: no valid results")
            else:
                best = valid.loc[valid["auroc_mean"].idxmax()]
                print(f"  {task_name:>25s}: AUROC={best['auroc_mean']:.4f} (k={int(best['k'])})")


if __name__ == "__main__":
    main()
