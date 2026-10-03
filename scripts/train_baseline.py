from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score


FEATURES = [
    "glucose",
    "slope_15m",
    "slope_30m",
    "slope_60m",
    "slope_120m",
    "glucose_mean_30m",
    "glucose_min_30m",
    "glucose_std_30m",
    "glucose_mean_60m",
    "glucose_min_60m",
    "bolus_exposure_15m",
    "bolus_exposure_30m",
    "bolus_exposure_60m",
    "bolus_exposure_120m",
    "basal_exposure_30m",
    "basal_exposure_60m",
    "basal_exposure_120m",
    "carb_exposure_30m",
    "carb_exposure_60m",
    "carb_exposure_120m",
    "minutes_since_carb",
    "hour_sin",
    "hour_cos",
]

DATA_PATH = "data/processed/hupa_features.csv"
FOLD_PATH = "data/processed/hupa_cv_folds.csv"


def patient_balanced_weights(patient_ids):
    counts = pd.Series(patient_ids).value_counts()
    weights = pd.Series(patient_ids).map(1.0 / counts)

    # Normalize so the average training weight is 1.
    weights = weights / weights.mean()

    return weights.to_numpy()


def evaluate_predictions(y_true, probabilities):
    pr_auc = average_precision_score(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)

    return pr_auc, roc_auc


def main():
    print("Loading feature dataset...")

    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    folds = pd.read_csv(FOLD_PATH)

    df = df.merge(
        folds[["patient_id", "fold"]],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )

    # Remove rows where the 30-minute future target does not exist.
    df = df[df["hypo_next_30m"].notna()].copy()

    # Require the full 120-minute historical window.
    required_history = [
        "slope_15m",
        "slope_30m",
        "slope_60m",
        "slope_120m",
    ]

    df = df.dropna(subset=required_history).copy()

    df["hypo_next_30m"] = df["hypo_next_30m"].astype(int)

    print(f"Usable rows: {len(df):,}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Positive windows: {df['hypo_next_30m'].sum():,}")
    print()

    all_oof_predictions = []

    print("=" * 70)
    print("5-FOLD PATIENT-WISE CROSS-VALIDATION")
    print("=" * 70)

    for fold in range(1, 6):

        print()
        print(f"Validation Fold {fold}")

        train_df = df[df["fold"] != fold].copy()
        val_df = df[df["fold"] == fold].copy()

        X_train = train_df[FEATURES]
        y_train = train_df["hypo_next_30m"]

        X_val = val_df[FEATURES]
        y_val = val_df["hypo_next_30m"]

        print(f"Training patients: {train_df['patient_id'].nunique()}")
        print(f"Validation patients: {val_df['patient_id'].nunique()}")
        print(f"Training rows: {len(train_df):,}")
        print(f"Validation rows: {len(val_df):,}")

        # Preprocessing is fitted ONLY on the training patients.
        imputer = SimpleImputer(
            strategy="median",
            add_indicator=True,
        )

        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        # Equalize total contribution of each patient.
        sample_weights = patient_balanced_weights(
            train_df["patient_id"].to_numpy()
        )

        # Simple baseline: regularized logistic regression.
        baseline = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=42,
        )

        baseline.fit(
            X_train_imp,
            y_train,
            sample_weight=sample_weights,
        )

        val_prob = baseline.predict_proba(X_val_imp)[:, 1]

        pr_auc, roc_auc = evaluate_predictions(
            y_val,
            val_prob,
        )

        print(f"PR-AUC:  {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        fold_predictions = val_df[
            ["patient_id", "time", "hypo_next_30m"]
        ].copy()

        fold_predictions["fold"] = fold
        fold_predictions["probability"] = val_prob

        all_oof_predictions.append(fold_predictions)

    oof = pd.concat(
        all_oof_predictions,
        ignore_index=True,
    )

    overall_pr_auc, overall_roc_auc = evaluate_predictions(
        oof["hypo_next_30m"],
        oof["probability"],
    )

    print()
    print("=" * 70)
    print("OVERALL DEVELOPMENT OOF RESULTS")
    print("=" * 70)
    print(f"PR-AUC:  {overall_pr_auc:.4f}")
    print(f"ROC-AUC: {overall_roc_auc:.4f}")

    output_path = Path("data/processed/baseline_oof_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    oof.to_csv(output_path, index=False)

    print()
    print(f"Saved OOF predictions to: {output_path}")


if __name__ == "__main__":
    main()