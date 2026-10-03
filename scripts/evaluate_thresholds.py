from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = "data/processed/random_forest_calibrated_oof.csv"

# Candidate probability thresholds.
THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]


def calculate_metrics(df, threshold):
    y_true = df["hypo_next_30m"].to_numpy()
    probability = df["calibrated_probability"].to_numpy()

    predicted_positive = probability >= threshold

    tp = np.sum((predicted_positive == 1) & (y_true == 1))
    tn = np.sum((predicted_positive == 0) & (y_true == 0))
    fp = np.sum((predicted_positive == 1) & (y_true == 0))
    fn = np.sum((predicted_positive == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    # Each row represents approximately one 5-minute observation.
    total_days = len(df) * 5 / (60 * 24)

    false_alarms_per_day = (
        fp / total_days if total_days > 0 else np.nan
    )

    alerts_per_day = (
        (tp + fp) / total_days if total_days > 0 else np.nan
    )

    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "false_alarms_per_day": false_alarms_per_day,
        "alerts_per_day": alerts_per_day,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def main():
    print("Loading calibrated development OOF predictions...")

    df = pd.read_csv(INPUT_PATH, parse_dates=["time"])

    print(f"Rows: {len(df):,}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Positive windows: {df['hypo_next_30m'].sum():,}")
    print()

    print("=" * 90)
    print("DEVELOPMENT THRESHOLD ANALYSIS")
    print("=" * 90)

    results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(df, threshold)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    display_columns = [
        "threshold",
        "sensitivity",
        "specificity",
        "precision",
        "false_alarms_per_day",
        "alerts_per_day",
        "tp",
        "fp",
        "fn",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 90)
    print("PATIENT-LEVEL SENSITIVITY")
    print("=" * 90)

    patient_results = []

    for patient_id, patient_df in df.groupby("patient_id"):

        y_true = patient_df["hypo_next_30m"].to_numpy()
        probability = patient_df["calibrated_probability"].to_numpy()

        for threshold in THRESHOLDS:

            predicted_positive = probability >= threshold

            tp = np.sum(
                (predicted_positive == 1) &
                (y_true == 1)
            )

            fn = np.sum(
                (predicted_positive == 0) &
                (y_true == 1)
            )

            if (tp + fn) > 0:
                sensitivity = tp / (tp + fn)
            else:
                sensitivity = np.nan

            patient_results.append(
                {
                    "patient_id": patient_id,
                    "threshold": threshold,
                    "sensitivity": sensitivity,
                }
            )

    patient_df = pd.DataFrame(patient_results)

    summary = (
        patient_df
        .groupby("threshold")["sensitivity"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    output_path = Path(
        "data/processed/development_threshold_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    patient_output_path = Path(
        "data/processed/development_patient_threshold_results.csv"
    )

    patient_df.to_csv(
        patient_output_path,
        index=False,
    )

    print()
    print(
        f"Saved threshold results to: {output_path}"
    )

    print(
        f"Saved patient threshold results to: "
        f"{patient_output_path}"
    )


if __name__ == "__main__":
    main()