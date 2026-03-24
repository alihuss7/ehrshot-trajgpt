from __future__ import annotations

"""EHRSHOT evaluation utilities.

Implements the EHRSHOT few-shot evaluation protocol:
- k-shot stratified sampling (k per class)
- Logistic regression classifier training
- AUROC / AUPRC metric computation (binary, multiclass, multilabel)
- Bootstrapped 95% confidence intervals over multiple seeds
"""

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


def sample_k_shot(
    labels: np.ndarray,
    k: int,
    seed: int,
    task_type: str = "binary",
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified k-shot sampling following the EHRSHOT protocol.

    Samples k examples per class for training, remaining for test.

    Args:
        labels: Array of labels for all examples.
        k: Number of examples per class for training.
        seed: Random seed for reproducibility.
        task_type: "binary", "multiclass", or "multilabel".

    Returns:
        (train_indices, test_indices) as numpy arrays.
    """
    rng = np.random.RandomState(seed)

    if task_type == "multilabel":
        # For multilabel, sample k examples that are positive for each label
        n = len(labels)
        train_set = set()
        labels_array = np.array(labels)
        for col in range(labels_array.shape[1]):
            pos_indices = np.where(labels_array[:, col] == 1)[0]
            if len(pos_indices) >= k:
                selected = rng.choice(pos_indices, size=k, replace=False)
            else:
                selected = pos_indices
            train_set.update(selected.tolist())
        # Also sample k negative examples (no positive labels)
        neg_indices = np.where(labels_array.sum(axis=1) == 0)[0]
        if len(neg_indices) >= k:
            selected = rng.choice(neg_indices, size=k, replace=False)
            train_set.update(selected.tolist())

        train_indices = np.array(sorted(train_set))
        test_indices = np.array(sorted(set(range(n)) - train_set))
    else:
        # Binary or multiclass: k per class
        unique_labels = np.unique(labels)
        train_list = []
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            if len(indices) >= k:
                selected = rng.choice(indices, size=k, replace=False)
            else:
                selected = indices
            train_list.extend(selected.tolist())

        train_indices = np.array(sorted(train_list))
        all_indices = np.arange(len(labels))
        test_indices = np.setdiff1d(all_indices, train_indices)

    return train_indices, test_indices


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str = "binary",
) -> LogisticRegression | OneVsRestClassifier:
    """Train a logistic regression classifier.

    Args:
        X_train: Training features (N x D).
        y_train: Training labels.
        task_type: "binary", "multiclass", or "multilabel".

    Returns:
        Trained classifier.
    """
    base_lr = LogisticRegression(
        max_iter=10000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )

    if task_type == "multilabel":
        clf = OneVsRestClassifier(base_lr)
    else:
        clf = base_lr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)

    return clf


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_type: str = "binary",
    num_classes: int = 2,
) -> dict[str, float]:
    """Compute AUROC and AUPRC.

    Args:
        y_true: True labels.
        y_score: Predicted probabilities. Shape depends on task_type:
            - binary: (N,) or (N, 2) — use column 1 if 2D
            - multiclass: (N, C)
            - multilabel: (N, C)
        task_type: "binary", "multiclass", or "multilabel".
        num_classes: Number of classes.

    Returns:
        Dict with "auroc" and "auprc" keys.
    """
    metrics = {}

    try:
        if task_type == "binary":
            scores = y_score[:, 1] if y_score.ndim == 2 else y_score
            metrics["auroc"] = roc_auc_score(y_true, scores)
            metrics["auprc"] = average_precision_score(y_true, scores)

        elif task_type == "multiclass":
            # Binarize for OvR AUROC
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            # Handle case where not all classes appear in test set
            valid_cols = y_true_bin.sum(axis=0) > 0
            if valid_cols.sum() < 2:
                metrics["auroc"] = float("nan")
                metrics["auprc"] = float("nan")
            else:
                metrics["auroc"] = roc_auc_score(
                    y_true_bin[:, valid_cols],
                    y_score[:, valid_cols],
                    average="macro",
                    multi_class="ovr",
                )
                metrics["auprc"] = average_precision_score(
                    y_true_bin[:, valid_cols],
                    y_score[:, valid_cols],
                    average="macro",
                )

        elif task_type == "multilabel":
            valid_cols = np.array(y_true).sum(axis=0) > 0
            if valid_cols.sum() == 0:
                metrics["auroc"] = float("nan")
                metrics["auprc"] = float("nan")
            else:
                y_true_valid = np.array(y_true)[:, valid_cols]
                y_score_valid = y_score[:, valid_cols]
                metrics["auroc"] = roc_auc_score(
                    y_true_valid, y_score_valid, average="macro"
                )
                metrics["auprc"] = average_precision_score(
                    y_true_valid, y_score_valid, average="macro"
                )

    except ValueError:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    return metrics


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrapped confidence interval.

    Args:
        values: List of metric values across seeds.
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level (0.05 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with "mean", "ci_lower", "ci_upper".
    """
    values = [v for v in values if not np.isnan(v)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    rng = np.random.RandomState(seed)
    arr = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    }


def run_single_evaluation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    seed: int,
    task_type: str = "binary",
    num_classes: int = 2,
) -> dict:
    """Run a single k-shot evaluation.

    Args:
        embeddings: Feature matrix (N x D).
        labels: Label array.
        k: Number of shots per class.
        seed: Random seed.
        task_type: "binary", "multiclass", or "multilabel".
        num_classes: Number of classes.

    Returns:
        Dict with k, seed, and metric values.
    """
    train_idx, test_idx = sample_k_shot(labels, k, seed, task_type)

    if len(train_idx) == 0 or len(test_idx) == 0:
        return {"k": k, "seed": seed, "auroc": float("nan"), "auprc": float("nan")}

    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_test, y_test = embeddings[test_idx], labels[test_idx]

    clf = train_classifier(X_train, y_train, task_type)

    y_score = clf.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_score, task_type, num_classes)

    return {"k": k, "seed": seed, **metrics}


def save_result(result: dict, output_path: str | Path):
    """Save a single evaluation result as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
