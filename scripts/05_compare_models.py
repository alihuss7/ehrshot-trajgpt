#!/usr/bin/env python3
"""Compare CLMBR-T-base and TrajGPT summary results.

Reads per-model `summary.csv` files produced by scripts/02_run_evaluation.py.

Usage:
    python scripts/05_compare_models.py --results-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TASK_TO_CATEGORY = {
    # Operational
    "guo_icu": "operational",
    "guo_los": "operational",
    "guo_readmission": "operational",
    # Lab
    "lab_anemia": "lab",
    "lab_hyperkalemia": "lab",
    "lab_hypoglycemia": "lab",
    "lab_hyponatremia": "lab",
    "lab_thrombocytopenia": "lab",
    # Diagnosis
    "new_acutemi": "diagnosis",
    "new_celiac": "diagnosis",
    "new_hyperlipidemia": "diagnosis",
    "new_hypertension": "diagnosis",
    "new_lupus": "diagnosis",
    "new_pancan": "diagnosis",
    # Imaging
    "chexpert": "imaging",
}


def load_summaries(results_dir: Path) -> pd.DataFrame:
    """Load and combine all `<model>/summary.csv` files."""
    dfs = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "summary.csv"
        if not summary_path.exists():
            continue

        df = pd.read_csv(summary_path)
        if "model" not in df.columns:
            df["model"] = model_dir.name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No summary.csv files found under {results_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    required_cols = {"model", "task", "k", "auroc_mean", "auprc_mean"}
    missing = required_cols - set(combined.columns)
    if missing:
        raise ValueError(
            f"summary.csv format mismatch; missing columns: {sorted(missing)}"
        )

    combined["category"] = combined["task"].map(TASK_TO_CATEGORY).fillna("unknown")
    return combined


def print_macro_by_k(df: pd.DataFrame, metric_col: str) -> None:
    """Print macro mean across tasks for each model and k."""
    print(f"\n{'=' * 72}")
    print(f"Macro {metric_col.upper()} By K")
    print(f"{'=' * 72}")

    macro_df = (
        df.groupby(["model", "k"], as_index=False)[metric_col]
        .mean()
        .sort_values(["k", "model"])
    )
    models = sorted(macro_df["model"].unique())

    header = f"{'k':>5s}"
    for model in models:
        header += f"  {model:>16s}"
    print(header)
    print("-" * len(header))

    for k in sorted(macro_df["k"].unique()):
        row = f"{int(k):>5d}"
        for model in models:
            vals = macro_df[(macro_df["k"] == k) & (macro_df["model"] == model)][
                metric_col
            ]
            row += f"  {vals.iloc[0]:>16.4f}" if len(vals) else f"  {'N/A':>16s}"
        print(row)


def print_best_by_task(df: pd.DataFrame, metric_col: str) -> None:
    """Print each model's best score per task."""
    print(f"\n{'=' * 72}")
    print(f"Best {metric_col.upper()} Per Task")
    print(f"{'=' * 72}")

    models = sorted(df["model"].unique())
    tasks = sorted(df["task"].unique())

    for task in tasks:
        task_df = df[df["task"] == task]
        row = f"{task:>22s}: "
        chunks = []
        for model in models:
            model_task = task_df[task_df["model"] == model]
            if model_task.empty:
                chunks.append(f"{model}=N/A")
                continue
            best_idx = model_task[metric_col].idxmax()
            best = model_task.loc[best_idx]
            chunks.append(f"{model}={best[metric_col]:.4f} (k={int(best['k'])})")
        row += " | ".join(chunks)
        print(row)


def save_artifacts(df: pd.DataFrame, output_dir: Path) -> None:
    """Save combined and macro comparison artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "all_results.csv", index=False)

    for metric_col in ["auroc_mean", "auprc_mean"]:
        macro_by_k = (
            df.groupby(["model", "k"], as_index=False)[metric_col]
            .mean()
            .rename(columns={metric_col: "macro_mean"})
        )
        macro_by_k.to_csv(output_dir / f"macro_by_k_{metric_col}.csv", index=False)

        macro_by_category = (
            df.groupby(["model", "category", "k"], as_index=False)[metric_col]
            .mean()
            .rename(columns={metric_col: "macro_mean"})
        )
        macro_by_category.to_csv(
            output_dir / f"macro_by_category_{metric_col}.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = load_summaries(results_dir)

    print(f"Loaded models: {sorted(df['model'].unique().tolist())}")
    print(f"Tasks: {len(df['task'].unique())}, rows: {len(df)}")

    print_macro_by_k(df, "auroc_mean")
    print_macro_by_k(df, "auprc_mean")
    print_best_by_task(df, "auroc_mean")

    output_dir = results_dir / "comparison"
    save_artifacts(df, output_dir)
    print(f"\nSaved comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
