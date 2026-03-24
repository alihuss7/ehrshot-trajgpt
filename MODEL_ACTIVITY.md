# Model Activity Log (CLMBR vs TrajGPT)

This file tracks:
- what changed in the modeling pipeline,
- what was run,
- and the latest comparison outcomes.

## Current Baseline Snapshot

- Baseline commit: `20534c8` (`main`)
- Repo: `https://github.com/alihuss7/ehrshot-trajgpt`
- Benchmark: EHRSHOT (15 tasks)
- CLMBR source: precomputed `EHRSHOT_ASSETS/features/clmbr_features.pkl`
- TrajGPT source: local pretraining + extracted embeddings

## Latest Comparison Results (Local Run)

Source files:
- `results/gen2/comparison/macro_by_k_auroc_mean.csv`
- `results/gen2/comparison/macro_by_k_auprc_mean.csv`
- `results/gen2/comparison/macro_by_category_auroc_mean.csv`
- `results/gen2/comparison/macro_by_category_auprc_mean.csv`

### Macro AUROC by k-shot (selected)

| k | CLMBR-T-base | TrajGPT | Delta (TrajGPT - CLMBR) |
|---:|---:|---:|---:|
| -1 | 0.8066 | 0.6435 | -0.1631 |
| 1 | 0.5548 | 0.5115 | -0.0433 |
| 4 | 0.5797 | 0.5170 | -0.0627 |
| 16 | 0.5963 | 0.5266 | -0.0697 |
| 64 | 0.6394 | 0.5298 | -0.1096 |
| 128 | 0.6708 | 0.5386 | -0.1322 |

### Macro AUPRC by k-shot (selected)

| k | CLMBR-T-base | TrajGPT | Delta (TrajGPT - CLMBR) |
|---:|---:|---:|---:|
| -1 | 0.3808 | 0.2226 | -0.1582 |
| 1 | 0.2776 | 0.2513 | -0.0262 |
| 4 | 0.2489 | 0.2214 | -0.0275 |
| 16 | 0.2479 | 0.2011 | -0.0468 |
| 64 | 0.2485 | 0.1792 | -0.0693 |
| 128 | 0.2469 | 0.1649 | -0.0820 |

### Macro AUROC by Category at k = -1

| Category | CLMBR-T-base | TrajGPT | Delta |
|---|---:|---:|---:|
| diagnosis | 0.7737 | 0.6107 | -0.1631 |
| imaging | 0.7926 | 0.7315 | -0.0612 |
| lab | 0.8352 | 0.6488 | -0.1863 |
| operational | 0.8295 | 0.6709 | -0.1586 |

## Change History (Modeling + Pipeline)

### v1 (baseline in repo)

- Implemented TrajGPT architecture modules:
  - `models/trajgpt/sra.py`
  - `models/trajgpt/xpos.py`
  - `models/trajgpt/model.py`
  - `models/trajgpt/heads.py`
  - `models/trajgpt/tokenizer.py`
- Aligned key SRA/XPOS details to official implementation behavior (QKV projection, gamma projection, XPOS rotation/scaling, SiLU gating, GroupNorm setup).
- Pretraining script uses official EHRSHOT patient splits from `EHRSHOT_ASSETS/splits/person_id_map.csv`.
- TrajGPT embedding extraction writes CLMBR-compatible pickle format so `scripts/02_run_evaluation.py` can evaluate both models with one interface.
- Evaluation script supports EHRSHOT assets/few-shot format and binary + categorical metrics.
- Comparison script writes unified outputs under `results/gen2/comparison/`.

### v2 (gen2 patch set applied)

- Switched to strict TrajGPT repo/paper defaults for the model path used in EHRSHOT:
  - `d_model=200`, `qk_dim=200`, `v_dim=400`, `ff_dim=800`, `tau=20`, `dropout=0.0`, `max_seq_len=256`, `forecast_method=time_specific`.
- SRA/XPOS path aligned to official behavior:
  - XPOS offset-based rotation applied before head split,
  - `gamma_proj` uses bias,
  - parallel decay mask uses cumulative-product ratio (no log-space custom path),
  - FFN block is 2-layer GELU without dropout.
- Tokenization is code-only (removed structured/composed token features from the TrajGPT path).
- Embedding extraction is canonical time-specific last hidden state only.
- Existing v1 checkpoints became intentionally incompatible with strict architecture; archived old outputs to:
  - `results/trajgpt_gen1_2026-03-24/`
- Gen2 full run completed end-to-end (pretrain -> extract -> eval -> compare) and outputs written under:
  - `results/gen2/trajgpt/`
  - `results/gen2/comparison/`
- Parameter counts observed in Gen2:
  - TrajGPT backbone: `11,392,432`
  - TrajGPT + pretrain head: `17,674,888`

## Experiment Log Template (Fill for each new run)

| Run ID | Date | Commit | Config | Checkpoint | Features file | Macro AUROC (-1) | Macro AUPRC (-1) | Notes |
|---|---|---|---|---|---|---:|---:|---|
| trajgpt_gen2_001 | 2026-03-24 | `20534c8` | `configs/trajgpt_ehrshot.yaml` | `results/gen2/trajgpt/checkpoints/best_model.pt` | `results/gen2/trajgpt/embeddings/trajgpt_features.pkl` | 0.6435 | 0.2226 | strict TrajGPT-style run completed |
