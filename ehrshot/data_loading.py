from __future__ import annotations

"""EHRSHOT data loading utilities.

Handles loading MEDS-formatted patient data and task labels from parquet files.
MEDS schema: subject_id (int64), time (timestamp), code (string),
             numeric_value (float32), text_value (string).
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ehrshot.tasks import TaskSpec


def load_meds_dataset(data_dir: str) -> pd.DataFrame:
    """Load MEDS-formatted patient data from parquet files.

    Args:
        data_dir: Path to directory containing MEDS parquet files
                  (e.g., data/ehrshot_meds/data/).

    Returns:
        DataFrame with columns: subject_id, time, code, numeric_value, text_value
        sorted by (subject_id, time).
    """
    data_path = Path(data_dir)

    # MEDS data may be in a single parquet or sharded across multiple files
    parquet_files = sorted(data_path.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    # Standardize column names (MEDS spec)
    expected_cols = {"subject_id", "time", "code"}
    if not expected_cols.issubset(df.columns):
        # Try alternative MEDS column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "subject" in col_lower or "patient" in col_lower:
                col_map[col] = "subject_id"
            elif col_lower in ("time", "timestamp", "datetime"):
                col_map[col] = "time"
            elif col_lower == "code":
                col_map[col] = "code"
            elif "numeric" in col_lower:
                col_map[col] = "numeric_value"
            elif "text" in col_lower:
                col_map[col] = "text_value"
        df = df.rename(columns=col_map)

    # Ensure time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values(["subject_id", "time"]).reset_index(drop=True)
    return df


def load_task_labels(labels_dir: str, task: TaskSpec) -> pd.DataFrame:
    """Load labels for a specific EHRSHOT task.

    Args:
        labels_dir: Path to directory containing label parquet files
                    (e.g., data/ehrshot_meds/label/).
        task: TaskSpec defining which task to load.

    Returns:
        DataFrame with columns: subject_id, prediction_time, label.
    """
    label_path = Path(labels_dir) / task.label_key
    if label_path.is_dir():
        parquet_files = sorted(label_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No label files found for task {task.name} in {label_path}")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
    elif label_path.with_suffix(".parquet").exists():
        df = pd.read_parquet(label_path.with_suffix(".parquet"))
    else:
        raise FileNotFoundError(
            f"Label file not found for task {task.name}. "
            f"Looked in {label_path} and {label_path.with_suffix('.parquet')}"
        )

    # Standardize columns
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "subject" in col_lower or "patient" in col_lower:
            col_map[col] = "subject_id"
        elif "time" in col_lower:
            col_map[col] = "prediction_time"
        elif "boolean" in col_lower:
            col_map[col] = "label"
        elif "integer" in col_lower or "value" in col_lower:
            col_map[col] = "label"
    df = df.rename(columns=col_map)

    # Ensure we have required columns
    if "prediction_time" not in df.columns:
        raise ValueError(f"No prediction_time column found in labels for {task.name}")
    if "label" not in df.columns:
        raise ValueError(f"No label column found in labels for {task.name}")

    if not pd.api.types.is_datetime64_any_dtype(df["prediction_time"]):
        df["prediction_time"] = pd.to_datetime(df["prediction_time"])

    # Convert boolean labels to int
    if df["label"].dtype == bool:
        df["label"] = df["label"].astype(int)

    return df[["subject_id", "prediction_time", "label"]].copy()


def build_patient_sequences(
    meds_df: pd.DataFrame,
    max_length: int | None = 256,
) -> dict[int, dict]:
    """Convert MEDS events into per-patient sequences for model input.

    Args:
        meds_df: MEDS DataFrame (output of load_meds_dataset).
        max_length: Maximum sequence length (truncate from the right/most recent).

    Returns:
        Dict mapping subject_id -> {
            "codes": list[str],           # code strings
            "times": list[datetime],      # event timestamps
            "numeric_values": list[float],# numeric values (NaN if absent)
            "units": list[str | None],    # measurement units if present
            "omop_tables": list[str | None], # source tables if present
        }
    """
    patients = {}
    for subject_id, group in meds_df.groupby("subject_id"):
        group = group.sort_values("time")
        if max_length is not None and len(group) > max_length:
            group = group.tail(max_length)

        patients[subject_id] = {
            "codes": group["code"].tolist(),
            "times": group["time"].tolist(),
            "numeric_values": group.get("numeric_value", pd.Series([np.nan] * len(group))).tolist(),
            "units": group.get("unit", pd.Series([None] * len(group))).tolist(),
            "omop_tables": group.get("omop_table", pd.Series([None] * len(group))).tolist(),
        }
    return patients


def get_prediction_time_patients(
    meds_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> dict[tuple[int, datetime], dict]:
    """Get patient sequences truncated at each prediction time.

    For each (subject_id, prediction_time) in labels, returns the patient's
    event sequence up to (and including) that prediction time. This prevents
    future information leakage.

    Args:
        meds_df: Full MEDS DataFrame.
        labels_df: Labels DataFrame with subject_id and prediction_time.

    Returns:
        Dict mapping (subject_id, prediction_time) -> {
            "codes": list[str],
            "times": list[datetime],
            "numeric_values": list[float]
        }
    """
    result = {}
    unique_pairs = labels_df[["subject_id", "prediction_time"]].drop_duplicates()

    for _, row in unique_pairs.iterrows():
        sid = row["subject_id"]
        pred_time = row["prediction_time"]

        patient_events = meds_df[
            (meds_df["subject_id"] == sid) & (meds_df["time"] <= pred_time)
        ].sort_values("time")

        if len(patient_events) == 0:
            continue

        result[(sid, pred_time)] = {
            "codes": patient_events["code"].tolist(),
            "times": patient_events["time"].tolist(),
            "numeric_values": patient_events.get(
                "numeric_value", pd.Series([np.nan] * len(patient_events))
            ).tolist(),
        }

    return result


def get_all_unique_codes(meds_df: pd.DataFrame) -> list[str]:
    """Get sorted list of all unique medical codes in the dataset."""
    return sorted(meds_df["code"].dropna().unique().tolist())
