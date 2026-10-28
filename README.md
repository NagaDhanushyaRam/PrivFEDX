# PrivFedXAI

A privacy-preserving explainable federated learning framework for clinical decision support.
COSC 880 Capstone, Towson University, Fall 2026. Advisor: Dr. Wassila Lalouani.

PTB-XL (12-lead ECG) is the validation dataset. Recording sites are used as simulated hospital clients.
The aim is to train one shared diagnostic model without pooling patient records (federated learning),
limit what the model can reveal about individuals (differential privacy), and show where in the ECG the
model looked (attention heatmaps), then measure what each step costs in accuracy.

## Status (October 27, 2026)

| Phase | State |
| --- | --- |
| Sprint 1: data pipeline, five-client partitions, classical baselines (LR, SVM, RF) | Done |
| Per-site SMOTE variants of the baselines (advisor request) | Done |
| Sprint 2: centralized 1D CNN (CNN v0) | Trained, evaluated on fold 9, backbone frozen | Bahdanau attention and heatmaps | Next |
| Sprint 3: Flower federated simulation (FedAvg), Opacus DP-SGD, per-class fairness | Not started |
| Sprint 4: attention-to-ECG IoU alignment, SHAP, full metrics | Not started |
| Sprint 5: Streamlit app, IEEE report | Not started |

Final deliverables are due December 1, 2026.

## What has been built

- **Data pipeline:** records are read at 100 Hz, band-pass filtered (0.5-40 Hz, zero-phase), then scaled by one
  fixed constant (0.209 mV). No per-record or per-client statistics are used, so the same preprocessing works
  in federated and DP training.
- **Labels:** multi-label, five diagnostic superclasses (NORM, MI, STTC, CD, HYP). 21,388 of 21,799 records
  have at least one superclass.
- **Five clients:** recording sites 0, 1, 2, 3 and one pooled "Rest" client (sites 4-50 plus records with no
  site). PTB-XL folds 1-8 are train, fold 9 is validation, fold 10 is test.
- **Classical baselines:** logistic regression, SVM and random forest on 84 hand-crafted features, plus
  per-site SMOTE variants.
- **CNN v0:** four Conv1d + GroupNorm + ReLU blocks, 165,157 parameters, class-weighted loss. GroupNorm is used
  instead of BatchNorm so the model is compatible with Opacus.
- **Evaluation:** per-class and macro AUC, F1 and MCC at a fixed threshold, record-level bootstrap confidence
  intervals, and paired bootstrap differences between models.

## Results so far

Pooled macro AUC over the five superclasses.

| Model | Fold 9 (validation) | Fold 10 (test) |
| --- | --- | --- |
| Logistic regression | 0.842 | 0.831 |
| Random forest | 0.861 | 0.850 |
| SVM | 0.865 | 0.852 |
| CNN v0 | 0.927 | not scored yet |

- The CNN is higher than every baseline on fold 9 (paired bootstrap difference +0.061 to +0.088, 95% CIs
  exclude 0). This is one seed, and fold 9 was monitored during CNN training, so these are validation numbers.
- Per-site SMOTE did not help: macro AUC was slightly lower than the class-weighted baselines.
- Fold 10 is scored once for the CNN models, after the attention model is frozen.
- The baselines use default hyperparameters (no tuning).

Full tables: `results/baselines/baseline_table_test.csv` and `results/cnn/`.

## Setup

1. Install Anaconda or Miniconda.
2. Create and activate a conda environment:
```bash
conda create -n PrivFEDX python=3.11
conda activate PrivFEDX
```
   Install all packages below with pip, not conda.
3. Install PyTorch for your hardware (pip option) from https://pytorch.org/get-started/locally/
4. Install the remaining dependencies:
```bash
python -m pip install -r requirements.txt
```
5. Verify:
```bash
python check_env.py
```

## Dataset

PTB-XL v1.0.3 (PhysioNet): https://physionet.org/content/ptb-xl/1.0.3/

Download it and place the files in `data/raw/ptbxl/` (the folder that contains `ptbxl_database.csv`).
Raw data and derived files in `data/processed/` are not committed to this repository.

## Reproducing the results

Run each command from the repository root, as modules (no `.py`).

```bash
python -m src.data.build_features          # 84 baseline features
python -m src.data.build_waveform_cache    # cached waveforms for the CNN
python -m src.models.lr_baseline           # also rf_baseline, svm_baseline
python -m src.models.lr_smote              # also rf_smote, svm_smote
python -m src.eval.make_table              # baseline table on the test fold
python -m src.models.train                 # CNN v0 (30 epochs, fold 9 monitored)
python -m src.eval.compare_val             # CNN vs baselines on fold 9
python -m pytest tests                     # unit tests
```

## Repository structure

```
src/data/        preprocessing, client partitions, features, waveform cache
src/models/      classical baselines, SMOTE variants, 1D CNN, training script
src/eval/        metrics, bootstrap intervals, comparison tables
src/federated/   Flower client and server (planned)
src/privacy/     Opacus DP-SGD wrapper (planned)
src/explain/     attention heatmaps, SHAP, IoU alignment (planned)
tests/           unit tests
sanity/          throwaway library checks
notebooks/       EDA
results/         tables and figures (large arrays and model weights are git-ignored)
```

## Limitations

- Clients are recording sites, which are partly confounded with recording device, so non-IID effects mix label
  and device differences.
- Differential privacy via Opacus protects individual records, not patients (a patient can have up to 10
  records).
- The project is a simulation on one machine. No real hospital data or clinician study is involved.
