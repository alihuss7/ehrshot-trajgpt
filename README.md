# ehrshot-trajgpt

Comparing **TrajGPT** against **CLMBR-T-base** on the EHRSHOT benchmark.

Track model changes and result snapshots in:

```text
MODEL_ACTIVITY.md
```

## Data Prerequisites
Expected local folders:

```text
data/EHRSHOT_ASSETS
data/EHRSHOT_MEDS
```

These come from the EHRSHOT release (DUA-restricted).

## Environment (working setup used)

```bash
python3 -m venv .venv39
source .venv39/bin/activate
pip install -r requirements.txt
pip install torch==2.1.2 pandas pyarrow scikit-learn scipy tqdm pyyaml "numpy<2"
```

## Canonical Config

Use:

```text
configs/trajgpt_ehrshot.yaml
```

## Commands We Used

### 1) CLMBR-T-base baseline (from precomputed EHRSHOT_ASSETS features)

```bash
./.venv39/bin/python scripts/02_run_evaluation.py \
  --assets_dir data/EHRSHOT_ASSETS \
  --model_name clmbr-t-base \
  --output_dir results/gen2/clmbr-t-base
```

This uses `data/EHRSHOT_ASSETS/features/clmbr_features.pkl` by default.

Optional/legacy: `scripts/01_extract_embeddings.py` exists for regenerating
CLMBR embeddings from scratch, but it is not used in the canonical pipeline
documented here.

### 2) TrajGPT pretraining

```bash
./.venv39/bin/python scripts/03_pretrain_trajgpt.py \
  --config configs/trajgpt_ehrshot.yaml
```

Outputs checkpoints under:

```text
results/gen2/trajgpt/checkpoints
```

### 3) TrajGPT embedding extraction (CLMBR-compatible pickle output)

```bash
./.venv39/bin/python scripts/04_extract_trajgpt_embeddings.py \
  --config configs/trajgpt_ehrshot.yaml
```

Output:

```text
results/gen2/trajgpt/embeddings/trajgpt_features.pkl
```

### 4) TrajGPT evaluation on EHRSHOT tasks

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
./.venv39/bin/python scripts/02_run_evaluation.py \
  --assets_dir data/EHRSHOT_ASSETS \
  --features results/gen2/trajgpt/embeddings/trajgpt_features.pkl \
  --model_name trajgpt \
  --output_dir results/gen2/trajgpt
```

### 5) CLMBR vs TrajGPT comparison

```bash
./.venv39/bin/python scripts/05_compare_models.py --results-dir results/gen2
```

Outputs under:

```text
results/gen2/comparison
```

## Notes

- `scripts/04_extract_trajgpt_embeddings.py` already sets `KMP_DUPLICATE_LIB_OK=TRUE` internally.
- On macOS, the evaluation command above with `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` is the most stable option.
- The current TrajGPT code path is strict repo/paper-aligned; older TrajGPT checkpoints from prior code variants are not load-compatible and must be retrained.
