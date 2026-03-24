# ehrshot-trajgpt

Comparing **TrajGPT** (Selective Recurrent Attention) against **CLMBR-T-base** (pretrained Transformer) on the [EHRSHOT benchmark](https://ehrshot.stanford.edu/) — 15 clinical prediction tasks across 6,739 patients.

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd ehrshot-trajgpt
python3 -m venv .venv39
source .venv39/bin/activate
pip install -r requirements.txt
```

### 2. Download EHRSHOT data

Request access at [Redivis](https://redivis.com/) and download:

| File                 | Size   | Required                                                    |
| -------------------- | ------ | ----------------------------------------------------------- |
| `EHRSHOT_ASSETS.zip` | 4.3 GB | Yes — precomputed CLMBR embeddings, labels, few-shot splits |
| `EHRSHOT_MEDS.zip`   | 191 MB | Yes — MEDS-format patient event data for TrajGPT training   |

Extract both into the `data/` directory:

```bash
mkdir -p data
unzip EHRSHOT_ASSETS.zip -d data/
unzip EHRSHOT_MEDS.zip -d data/
```

You should have:

```
data/
  EHRSHOT_ASSETS/
    benchmark/     # 15 task labels + few-shot splits
    features/      # clmbr_features.pkl (precomputed 768-dim embeddings)
    splits/        # train/val/test patient splits
    ...
  EHRSHOT_MEDS/
    data/          # data.parquet (41.6M clinical events)
    labels/        # per-task label parquets
    metadata/      # codes, splits
```

## Running the Pipeline

### Step 1: Evaluate CLMBR-T-base (baseline)

CLMBR embeddings are precomputed in `EHRSHOT_ASSETS`. No model download needed.

```bash
python scripts/02_run_evaluation.py \
  --assets_dir data/EHRSHOT_ASSETS \
  --model_name clmbr-t-base \
  --output_dir results/clmbr-t-base
```

Output: `results/clmbr-t-base/summary.csv`

### Step 2: Pretrain TrajGPT

Trains TrajGPT from scratch on EHRSHOT patient sequences (~20 epochs, ~60 min on Apple Silicon MPS).

```bash
python scripts/03_pretrain_trajgpt.py --config configs/trajgpt_ehrshot.yaml
```

Output: `results/trajgpt/checkpoints/best_model.pt` + `tokenizer.json`

### Step 3: Extract TrajGPT embeddings

Extracts 200-dim embeddings for all 406K (patient, prediction_time) pairs.

```bash
python scripts/04_extract_trajgpt_embeddings.py --config configs/trajgpt_ehrshot.yaml
```

Output: `results/trajgpt/embeddings/trajgpt_features.pkl`

### Step 4: Evaluate TrajGPT

Same evaluation script as CLMBR, just point to TrajGPT features.

```bash
python scripts/02_run_evaluation.py \
  --assets_dir data/EHRSHOT_ASSETS \
  --features results/trajgpt/embeddings/trajgpt_features.pkl \
  --model_name trajgpt \
  --output_dir results/trajgpt
```

Output: `results/trajgpt/summary.csv`

### Step 5: Compare models

```bash
python scripts/05_compare_models.py --results-dir results/
```

## Project Structure

```
ehrshot-trajgpt/
  configs/
    trajgpt_ehrshot.yaml        # TrajGPT hyperparameters + data paths
  ehrshot/                      # Shared utilities
    tasks.py                    # 15 EHRSHOT task definitions
    data_loading.py             # MEDS parquet loading
    evaluation.py               # k-shot LR, AUROC/AUPRC, bootstrap CIs
  models/
    embedder.py                 # PatientEmbedder ABC + CLMBRBaseEmbedder
    trajgpt_embedder.py         # TrajGPTEmbedder (loads checkpoint, extracts embeddings)
    trajgpt/
      model.py                  # TrajGPT model (8 SRA blocks, 200-dim)
      sra.py                    # Selective Recurrent Attention (parallel + recurrent)
      xpos.py                   # XPOS position encoding for irregular timestamps
      tokenizer.py              # EHR code tokenizer
      heads.py                  # Pretrain (next-code) + classification heads
  scripts/
    01_extract_embeddings.py    # (Optional) CLMBR embedding extraction from HuggingFace
    02_run_evaluation.py        # Few-shot LR evaluation (works for both models)
    03_pretrain_trajgpt.py      # TrajGPT pretraining
    04_extract_trajgpt_embeddings.py  # TrajGPT embedding extraction
    05_compare_models.py        # Side-by-side comparison
  data/                         # EHRSHOT_ASSETS + EHRSHOT_MEDS (not in repo)
  results/                      # Outputs (not in repo)
```

## Model Comparison

|                  | CLMBR-T-base              | TrajGPT                           |
| ---------------- | ------------------------- | --------------------------------- |
| Parameters       | 141M                      | ~11M                              |
| Attention        | Standard softmax O(N^2)   | Selective Recurrent O(N)          |
| Time encoding    | Standard positional       | XPOS with real timestamps         |
| Temporal decay   | None                      | Data-dependent (learned per head) |
| Embedding dim    | 768                       | 200                               |
| Pretraining data | 2.57M patients (Stanford) | 6,739 patients (EHRSHOT)          |

## Troubleshooting

**OpenMP crash on macOS:** If you see `Initializing libiomp5.dylib, but found libomp.dylib already initialized`, prefix your command with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/...
```

The extraction script sets `KMP_DUPLICATE_LIB_OK` automatically.

## References

- [EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models](https://arxiv.org/abs/2307.02028) (Wornow et al., NeurIPS 2023)
- [TrajGPT: Irregular Time-Series Representation Learning for Health Prediction](https://github.com/li-lab-mcgill/TrajGPT) (Song et al., IEEE JBHI 2025)
- [CLMBR-T-base on HuggingFace](https://huggingface.co/StanfordShahLab/clmbr-t-base)
