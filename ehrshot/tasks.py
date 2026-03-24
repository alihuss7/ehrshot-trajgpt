from __future__ import annotations

"""EHRSHOT benchmark task definitions.

Defines all 15 prediction tasks from the EHRSHOT benchmark across 4 categories:
- Operational outcomes (3 binary tasks)
- Lab result anticipation (5 multiclass tasks, 5-way)
- Diagnosis assignment (6 binary tasks)
- Imaging (1 multilabel task, 14-way)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """Specification for an EHRSHOT prediction task."""
    name: str
    category: str          # "operational" | "lab" | "diagnosis" | "imaging"
    task_type: str         # "binary" | "multiclass" | "multilabel"
    num_classes: int
    label_key: str         # subdirectory/filename in MEDS label files


# ── Operational Outcomes (binary) ─────────────────────────────────────────────

LONG_LOS = TaskSpec(
    name="long_los",
    category="operational",
    task_type="binary",
    num_classes=2,
    label_key="long_length_of_stay",
)

READMISSION_30D = TaskSpec(
    name="readmission_30d",
    category="operational",
    task_type="binary",
    num_classes=2,
    label_key="30_day_readmission",
)

ICU_TRANSFER = TaskSpec(
    name="icu_transfer",
    category="operational",
    task_type="binary",
    num_classes=2,
    label_key="icu_transfer",
)

# ── Lab Result Anticipation (5-way multiclass) ───────────────────────────────

THROMBOCYTOPENIA = TaskSpec(
    name="thrombocytopenia",
    category="lab",
    task_type="multiclass",
    num_classes=5,
    label_key="thrombocytopenia",
)

HYPERKALEMIA = TaskSpec(
    name="hyperkalemia",
    category="lab",
    task_type="multiclass",
    num_classes=5,
    label_key="hyperkalemia",
)

HYPOGLYCEMIA = TaskSpec(
    name="hypoglycemia",
    category="lab",
    task_type="multiclass",
    num_classes=5,
    label_key="hypoglycemia",
)

HYPONATREMIA = TaskSpec(
    name="hyponatremia",
    category="lab",
    task_type="multiclass",
    num_classes=5,
    label_key="hyponatremia",
)

ANEMIA = TaskSpec(
    name="anemia",
    category="lab",
    task_type="multiclass",
    num_classes=5,
    label_key="anemia",
)

# ── Diagnosis Assignment (binary) ─────────────────────────────────────────────

HYPERTENSION = TaskSpec(
    name="hypertension",
    category="diagnosis",
    task_type="binary",
    num_classes=2,
    label_key="hypertension",
)

HYPERLIPIDEMIA = TaskSpec(
    name="hyperlipidemia",
    category="diagnosis",
    task_type="binary",
    num_classes=2,
    label_key="hyperlipidemia",
)

PANCREATIC_CANCER = TaskSpec(
    name="pancreatic_cancer",
    category="diagnosis",
    task_type="binary",
    num_classes=2,
    label_key="pancreatic_cancer",
)

CELIAC = TaskSpec(
    name="celiac",
    category="diagnosis",
    task_type="binary",
    num_classes=2,
    label_key="celiac",
)

LUPUS = TaskSpec(
    name="lupus",
    category="diagnosis",
    task_type="binary",
    num_classes=2,
    label_key="lupus",
)

ACUTE_MI = TaskSpec(
    name="acute_mi",
    category="diagnosis",
    task_type="binary",
    num_classes=2,
    label_key="acute_mi",
)

# ── Imaging (14-way multilabel) ───────────────────────────────────────────────

CHEST_XRAY = TaskSpec(
    name="chest_xray",
    category="imaging",
    task_type="multilabel",
    num_classes=14,
    label_key="chest_xray_findings",
)

# ── Registry ──────────────────────────────────────────────────────────────────

TASK_REGISTRY: dict[str, TaskSpec] = {
    task.name: task
    for task in [
        # Operational
        LONG_LOS, READMISSION_30D, ICU_TRANSFER,
        # Lab
        THROMBOCYTOPENIA, HYPERKALEMIA, HYPOGLYCEMIA, HYPONATREMIA, ANEMIA,
        # Diagnosis
        HYPERTENSION, HYPERLIPIDEMIA, PANCREATIC_CANCER, CELIAC, LUPUS, ACUTE_MI,
        # Imaging
        CHEST_XRAY,
    ]
}

CATEGORIES = ["operational", "lab", "diagnosis", "imaging"]


def get_tasks(names: str | list[str] = "all") -> list[TaskSpec]:
    """Resolve task names to TaskSpec objects.

    Args:
        names: "all" for all 15 tasks, or a list of task names,
               or a category name like "operational".
    """
    if names == "all":
        return list(TASK_REGISTRY.values())

    if isinstance(names, str):
        if names in CATEGORIES:
            return [t for t in TASK_REGISTRY.values() if t.category == names]
        return [TASK_REGISTRY[names]]

    return [TASK_REGISTRY[n] for n in names]
