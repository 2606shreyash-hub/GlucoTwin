from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
    print("5-FOLD PATIENT-WISE RANDOM FOREST")
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

        # Fit preprocessing ONLY on training patients.
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

        model = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
        )

        model.fit(
            X_train_imp,
            y_train,
            sample_weight=sample_weights,
        )

        val_prob = model.predict_proba(X_val_imp)[:, 1]

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

    print()
    print("=" * 70)
    print("PATIENT-LEVEL RESULTS")
    print("=" * 70)

    patient_results = []

    for patient_id, patient_df in oof.groupby("patient_id"):

        y_true = patient_df["hypo_next_30m"]
        probabilities = patient_df["probability"]

        # PR-AUC requires both classes to be present.
        if y_true.nunique() >= 2:
            patient_pr_auc = average_precision_score(
                y_true,
                probabilities,
            )
        else:
            patient_pr_auc = np.nan

        # ROC-AUC also requires both classes.
        if y_true.nunique() >= 2:
            patient_roc_auc = roc_auc_score(
                y_true,
                probabilities,
            )
        else:
            patient_roc_auc = np.nan

        patient_results.append(
            {
                "patient_id": patient_id,
                "pr_auc": patient_pr_auc,
                "roc_auc": patient_roc_auc,
                "rows": len(patient_df),
                "positive_windows": int(y_true.sum()),
            }
        )

    patient_results = pd.DataFrame(patient_results)

    print(
        f"Median patient PR-AUC: "
        f"{patient_results['pr_auc'].median():.4f}"
    )

    print(
        f"Patient PR-AUC IQR: "
        f"{patient_results['pr_auc'].quantile(0.25):.4f} - "
        f"{patient_results['pr_auc'].quantile(0.75):.4f}"
    )

    print(
        f"Median patient ROC-AUC: "
        f"{patient_results['roc_auc'].median():.4f}"
    )

    print(
        f"Patient ROC-AUC IQR: "
        f"{patient_results['roc_auc'].quantile(0.25):.4f} - "
        f"{patient_results['roc_auc'].quantile(0.75):.4f}"
    )

    print()
    print("Per-patient results:")
    print(
        patient_results
        .sort_values("patient_id")
        .to_string(index=False)
    )

    output_path = Path(
        "data/processed/random_forest_oof_predictions.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof.to_csv(
        output_path,
        index=False,
    )

    patient_output_path = Path(
        "data/processed/random_forest_patient_results.csv"
    )

    patient_results.to_csv(
        patient_output_path,
        index=False,
    )

    print()
    print(
        f"Saved OOF predictions to: {output_path}"
    )

    print(
        f"Saved patient results to: {patient_output_path}"
    )


if __name__ == "__main__":
    main()