# GlucoTwin

A student proof-of-concept Digital Twin for Type 1 Diabetes focused on early hypoglycemia risk prediction.

## Current research specification

- Dataset: HUPA-UCM Diabetes Dataset
- Population: 25 real T1D participants
- Target: probability that glucose will be <70 mg/dL at any point during the next 30 minutes
- Development: 20 patients
- Final test: 5 untouched patients
- Development validation: 5-fold patient-wise CV
- Candidate models: Logistic Regression, Random Forest, XGBoost
- Primary evaluation: PR-AUC, Brier score/calibration
- Operational evaluation: sensitivity, false alarms/day, event sensitivity, warning lead time

## Phase 1

The first implementation milestone is a leakage-safe HUPA data loader and audit.

The HUPA ZIP itself is not copied into this repository. Put/extract the dataset separately and pass its `Preprocessed` directory to the audit script.

## Expected preprocessed CSV format

HUPA files are semicolon-separated and contain:

`time;glucose;calories;heart_rate;steps;basal_rate;bolus_volume_delivered;carb_input`

## Run

```bash
pip install -r requirements.txt
python scripts/audit_hupa.py --data-dir "/path/to/HUPA-UCM Diabetes Dataset/Preprocessed"
```
