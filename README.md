# ehrshot-trajgpt

Comparing **TrajGPT** against **CLMBR-T-base** on the EHRSHOT benchmark for clinical prediction tasks.

## Project Goals
1. Reproduce CLMBR-T-base baseline results on EHRSHOT benchmark tasks
2. Implement TrajGPT and train it on EHRSHOT
3. Compare both models on tasks from the EHRSHOT paper:
   - Predicting new diagnoses
   - Anticipating lab results
   - Operational outcomes (readmission, ICU transfer, length of stay)

## Structure
```
ehrshot-trajgpt/
├── data/               # EHR data (not committed - DUA restricted)
├── models/             # Model definitions and checkpoints
├── notebooks/          # Exploration and analysis notebooks
├── scripts/            # Training and evaluation scripts
├── results/            # Evaluation outputs and plots
└── configs/            # Hyperparameter and experiment configs
```

## Setup
```bash
conda create -n ehrshot python=3.10
conda activate ehrshot
pip install -r requirements.txt
```

## Models
- **CLMBR-T-base**: 141M parameter clinical foundation model from Stanford Shah Lab
- **TrajGPT**: Trajectory-based GPT model trained on EHRSHOT

## Data
Data sourced from [EHRSHOT on Redivis](http://ehrshot.stanford.edu). Access requires DUA approval.
Data must remain on encrypted local machine per the EHRSHOT Data Set License 1.0.
