# Model Activity Log (CLMBR vs TrajGPT)

## Current Baseline Snapshot

- Baseline commit: `20534c8` (`main`)
- Repo: `https://github.com/alihuss7/ehrshot-trajgpt`
- Benchmark: EHRSHOT (15 tasks)
- CLMBR source: precomputed `EHRSHOT_ASSETS/features/clmbr_features.pkl`
- TrajGPT source: local pretraining + extracted embeddings

## Model Size (Current TrajGPT Config)

- TrajGPT backbone: `11,392,432`
- TrajGPT + pretrain head: `17,674,888`
- These numbers apply to the current architecture/vocab setup.

## Latest Comparison Results (Local Run)

Source files:

- `results/trajgpt_gen3_2026-03-24/comparison/macro_by_k_auroc_mean.csv`
- `results/trajgpt_gen3_2026-03-24/comparison/macro_by_k_auprc_mean.csv`
- `results/trajgpt_gen3_2026-03-24/comparison/macro_by_category_auroc_mean.csv`
- `results/trajgpt_gen3_2026-03-24/comparison/macro_by_category_auprc_mean.csv`

Gen3 snapshot:

- Gen3 evaluation is complete at `results/trajgpt_gen3_2026-03-24/summary.csv`.
- Gen3 macro at `k=-1` (TrajGPT only): `AUROC=0.6280`, `AUPRC=0.2168`.
- The comparison tables below are the Gen3 comparison snapshot against CLMBR-T-base.

### Macro AUROC by k-shot

|   k | CLMBR-T-base | TrajGPT | Delta (TrajGPT - CLMBR) |
| --: | -----------: | ------: | ----------------------: |
|  -1 |       0.8066 |  0.6280 |                 -0.1787 |
|   1 |       0.5548 |  0.5099 |                 -0.0449 |
|   4 |       0.5797 |  0.5194 |                 -0.0602 |
|  16 |       0.5963 |  0.5244 |                 -0.0719 |
|  64 |       0.6394 |  0.5232 |                 -0.1162 |
| 128 |       0.6708 |  0.5347 |                 -0.1360 |

### Macro AUPRC by k-shot

|   k | CLMBR-T-base | TrajGPT | Delta (TrajGPT - CLMBR) |
| --: | -----------: | ------: | ----------------------: |
|  -1 |       0.3808 |  0.2168 |                 -0.1640 |
|   1 |       0.2776 |  0.2509 |                 -0.0266 |
|   4 |       0.2489 |  0.2209 |                 -0.0281 |
|  16 |       0.2479 |  0.2021 |                 -0.0458 |
|  64 |       0.2485 |  0.1784 |                 -0.0701 |
| 128 |       0.2469 |  0.1619 |                 -0.0849 |

### Macro AUROC by Category at k = -1

| Category    | CLMBR-T-base | TrajGPT |   Delta |
| ----------- | -----------: | ------: | ------: |
| diagnosis   |       0.7737 |  0.5930 | -0.1807 |
| imaging     |       0.7926 |  0.7360 | -0.0566 |
| lab         |       0.8352 |  0.6381 | -0.1971 |
| operational |       0.8295 |  0.6450 | -0.1845 |

<!--
## Change History (Modeling + Pipeline)

### v1 (baseline)

- Implemented TrajGPT architecture modules:
  - `models/trajgpt/sra.py`
  - `models/trajgpt/xpos.py`
  - `models/trajgpt/model.py`
  - `models/trajgpt/heads.py`
  - `models/trajgpt/tokenizer.py`
- Aligned key SRA/XPOS details to target implementation behavior (QKV projection, gamma projection, XPOS rotation/scaling, SiLU gating, GroupNorm setup).
- Pretraining script uses EHRSHOT patient splits from `EHRSHOT_ASSETS/splits/person_id_map.csv`.
- TrajGPT embedding extraction writes CLMBR-compatible pickle format so `scripts/02_run_evaluation.py` can evaluate both models with one interface.
- Evaluation script supports EHRSHOT assets/few-shot format and binary + categorical metrics.
- Comparison script writes unified outputs under `results/trajgpt_gen2_2026-03-24/comparison/`.

### v2 (Gen2 configuration alignment)

- Updated TrajGPT settings to match the published implementation closely for EHRSHOT:
  - `d_model=200`, `qk_dim=200`, `v_dim=400`, `ff_dim=800`, `tau=20`, `dropout=0.0`, `max_seq_len=256`, `forecast_method=time_specific`.
- SRA/XPOS path aligned to current behavior:
  - XPOS offset-based rotation applied before head split,
  - `gamma_proj` uses bias,
  - parallel decay mask uses cumulative-product ratio (no log-space custom path),
  - FFN block is 2-layer GELU without dropout.
- Tokenization is code-only (removed structured/composed token features from the TrajGPT path).
- Embedding extraction is canonical time-specific last hidden state only.
- Existing v1 checkpoints were not compatible after the architecture/settings update; archived old outputs to:
  - `results/trajgpt_gen1_2026-03-24/`
- Gen2 full run completed end-to-end (pretrain -> extract -> eval -> compare) and outputs written under:
  - `results/trajgpt_gen2_2026-03-24/trajgpt/`
  - `results/trajgpt_gen2_2026-03-24/comparison/`

### v3 (Gen3 results)

- Gen3 output path:
  - `results/trajgpt_gen3_2026-03-24/`
- Pretraining setup:
  - `configs/trajgpt_ehrshot.yaml`
  - EHRSHOT split mapping (`train=2295`, `val=2232`, `test=2212`)
  - vocabulary size `31,255` (31,252 codes + 3 special tokens)
- Pretraining complete:
  - best/final checkpoints written under `results/trajgpt_gen3_2026-03-24/checkpoints/`
- Embedding extraction complete:
  - `406,379 / 406,379` samples
  - `results/trajgpt_gen3_2026-03-24/embeddings/trajgpt_features.pkl`
  - `data_matrix: (406379, 200)` (`float16`)
- Evaluation complete:
  - `results/trajgpt_gen3_2026-03-24/summary.csv`
  - macro (`k=-1`) TrajGPT: `AUROC=0.6280`, `AUPRC=0.2168`
- Gen3 comparison artifacts vs CLMBR-T-base generated at:
- `results/trajgpt_gen3_2026-03-24/comparison/`
-->

## Experiment Log

| Run ID           | Date       | Commit         | Config                         | Checkpoint                                                          | Features file                                                             | Macro AUROC (-1) | Macro AUPRC (-1) | Notes                                  |
| ---------------- | ---------- | -------------- | ------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------: | ---------------: | -------------------------------------- |
| trajgpt_gen2_001 | 2026-03-24 | `20534c8`      | `configs/trajgpt_ehrshot.yaml` | `results/trajgpt_gen2_2026-03-24/trajgpt/checkpoints/best_model.pt` | `results/trajgpt_gen2_2026-03-24/trajgpt/embeddings/trajgpt_features.pkl` |           0.6435 |           0.2226 | TrajGPT-aligned Gen2 run completed     |
| trajgpt_gen3_001 | 2026-03-24 | `working tree` | `configs/trajgpt_ehrshot.yaml` | `results/trajgpt_gen3_2026-03-24/checkpoints/best_model.pt`         | `results/trajgpt_gen3_2026-03-24/embeddings/trajgpt_features.pkl`         |           0.6280 |           0.2168 | Gen3 run (pretrain -> extract -> eval) |
