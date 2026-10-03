import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all_patients
from src.data.preprocessing import clean_patient, audit_patient
from src.labels.hypoglycemia import add_30min_hypoglycemia_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="data/processed/patient_audit.csv")
    args = parser.parse_args()

    patients = load_all_patients(args.data_dir)

    audits = []
    total_usable = 0
    total_positive = 0

    for patient_id, raw in patients.items():
        clean = clean_patient(raw)
        labelled = add_30min_hypoglycemia_label(clean)

        valid = labelled["target_hypo_30m"].notna()
        usable = int(valid.sum())
        positive = int(labelled.loc[valid, "target_hypo_30m"].sum())

        row = audit_patient(clean)
        row["usable_prediction_rows"] = usable
        row["positive_30m_windows"] = positive
        row["positive_rate"] = positive / usable if usable else 0.0
        audits.append(row)

        total_usable += usable
        total_positive += positive

    audit_df = pd.DataFrame(audits).sort_values("patient_id")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output, index=False)

    print("\nGlucoTwin HUPA audit")
    print("=" * 70)
    print(f"Patients: {len(patients)}")
    print(f"Usable prediction windows: {total_usable:,}")
    print(f"Positive 30-min windows: {total_positive:,}")
    print(f"Positive prevalence: {total_positive / total_usable:.4%}")
    print(f"\nSaved: {output}")
    print("\nPatient summary:")
    print(audit_df[
        [
            "patient_id", "rows", "duration_days",
            "positive_30m_windows", "positive_rate",
            "negative_bolus_rows", "max_gap_minutes"
        ]
    ].to_string(index=False))

if __name__ == "__main__":
    main()
