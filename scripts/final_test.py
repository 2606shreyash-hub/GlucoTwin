from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)


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
OOF_PATH = "data/processed/random_forest_oof_predictions.csv"

ALERT_THRESHOLD = 0.50


def patient_balanced_weights(patient_ids):
    counts = pd.Series(patient_ids).value_counts()

    weights = pd.Series(patient_ids).map(
        1.0 / counts
    )

    weights = weights / weights.mean()

    return weights.to_numpy()


def fit_sigmoid_calibrator(oof):
    """
    Fit the previously specified sigmoid calibration
    using development OOF predictions only.
    """

    probability = np.clip(
        oof["probability"].to_numpy(),
        1e-6,
        1 - 1e-6,
    )

    logit_probability = np.log(
        probability / (1 - probability)
    )

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(
        logit_probability.reshape(-1, 1),
        oof["hypo_next_30m"].to_numpy(),
    )

    return calibrator


def main():

    print("Loading HUPA feature data...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["time"],
    )

    folds = pd.read_csv(FOLD_PATH)

    df = df.merge(
        folds[
            [
                "patient_id",
                "fold",
            ]
        ],
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )

    df = df[
        df["hypo_next_30m"].notna()
    ].copy()

    # Require complete 120-minute history.
    df = df.dropna(
        subset=[
            "slope_15m",
            "slope_30m",
            "slope_60m",
            "slope_120m",
        ]
    ).copy()

    df["hypo_next_30m"] = (
        df["hypo_next_30m"].astype(int)
    )

    development = df[
        df["fold"] != 0
    ].copy()

    test = df[
        df["fold"] == 0
    ].copy()

    print()
    print("=" * 70)
    print("LOCKED FINAL TEST")
    print("=" * 70)

    print(
        f"Development patients: "
        f"{development['patient_id'].nunique()}"
    )

    print(
        f"Test patients: "
        f"{test['patient_id'].nunique()}"
    )

    print(
        f"Development rows: "
        f"{len(development):,}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    print()
    print("Test patients:")

    for patient in sorted(
        test["patient_id"].unique()
    ):
        print(f"  {patient}")

    # ------------------------------------------------------------
    # 1. Fit preprocessing on ALL development patients.
    # ------------------------------------------------------------

    X_development = development[
        FEATURES
    ]

    y_development = development[
        "hypo_next_30m"
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
        "hypo_next_30m"
    ]

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    X_development_imp = (
        imputer.fit_transform(
            X_development
        )
    )

    X_test_imp = (
        imputer.transform(
            X_test
        )
    )

    # ------------------------------------------------------------
    # 2. Patient-balanced final training.
    # ------------------------------------------------------------

    sample_weights = (
        patient_balanced_weights(
            development[
                "patient_id"
            ].to_numpy()
        )
    )

    print()
    print(
        "Training final Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_development_imp,
        y_development,
        sample_weight=sample_weights,
    )

    # ------------------------------------------------------------
    # 3. Raw test probabilities.
    # ------------------------------------------------------------

    raw_probability = (
        model.predict_proba(
            X_test_imp
        )[:, 1]
    )

    # ------------------------------------------------------------
    # 4. Fit calibration ONLY from development OOF.
    # ------------------------------------------------------------

    print(
        "Loading development OOF predictions..."
    )

    oof = pd.read_csv(
        OOF_PATH
    )

    calibrator = fit_sigmoid_calibrator(
        oof
    )

    raw_probability_clipped = np.clip(
        raw_probability,
        1e-6,
        1 - 1e-6,
    )

    test_logit = np.log(
        raw_probability_clipped
        / (
            1
            - raw_probability_clipped
        )
    )

    calibrated_probability = (
        calibrator.predict_proba(
            test_logit.reshape(-1, 1)
        )[:, 1]
    )

    # ------------------------------------------------------------
    # 5. Frozen alert threshold.
    # ------------------------------------------------------------

    predicted_alert = (
        calibrated_probability
        >= ALERT_THRESHOLD
    )

    # ------------------------------------------------------------
    # 6. Final test metrics.
    # ------------------------------------------------------------

    pr_auc = average_precision_score(
        y_test,
        calibrated_probability,
    )

    roc_auc = roc_auc_score(
        y_test,
        calibrated_probability,
    )

    brier = brier_score_loss(
        y_test,
        calibrated_probability,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predicted_alert,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"PR-AUC:       {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC:      {roc_auc:.4f}"
    )

    print(
        f"Brier score:  {brier:.6f}"
    )

    print(
        f"Sensitivity:  {sensitivity:.4f}"
    )

    print(
        f"Specificity:  {specificity:.4f}"
    )

    print(
        f"Precision:    {precision:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        f"  TN: {tn}"
    )

    print(
        f"  FP: {fp}"
    )

    print(
        f"  FN: {fn}"
    )

    print(
        f"  TP: {tp}"
    )

    # ------------------------------------------------------------
    # 7. Per-patient final test results.
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("PER-PATIENT FINAL TEST RESULTS")
    print("=" * 70)

    test_results = []

    test_output = test[
        [
            "patient_id",
            "time",
            "hypo_next_30m",
        ]
    ].copy()

    test_output[
        "raw_probability"
    ] = raw_probability

    test_output[
        "calibrated_probability"
    ] = calibrated_probability

    test_output[
        "predicted_alert"
    ] = predicted_alert.astype(int)

    for patient_id, patient_df in (
        test_output.groupby(
            "patient_id"
        )
    ):

        y_patient = (
            patient_df[
                "hypo_next_30m"
            ]
        )

        p_patient = (
            patient_df[
                "calibrated_probability"
            ]
        )

        patient_pr_auc = np.nan
        patient_roc_auc = np.nan

        if y_patient.nunique() >= 2:

            patient_pr_auc = (
                average_precision_score(
                    y_patient,
                    p_patient,
                )
            )

            patient_roc_auc = (
                roc_auc_score(
                    y_patient,
                    p_patient,
                )
            )

        test_results.append(
            {
                "patient_id": patient_id,
                "rows": len(
                    patient_df
                ),
                "positive_windows": int(
                    y_patient.sum()
                ),
                "pr_auc": patient_pr_auc,
                "roc_auc": patient_roc_auc,
            }
        )

    patient_results = pd.DataFrame(
        test_results
    )

    print(
        patient_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    valid_patient_pr = (
        patient_results[
            "pr_auc"
        ].dropna()
    )

    valid_patient_roc = (
        patient_results[
            "roc_auc"
        ].dropna()
    )

    print()

    print(
        f"Median patient PR-AUC: "
        f"{valid_patient_pr.median():.4f}"
    )

    print(
        f"Median patient ROC-AUC: "
        f"{valid_patient_roc.median():.4f}"
    )

    # ------------------------------------------------------------
    # 8. Save final test predictions.
    # ------------------------------------------------------------

    prediction_path = Path(
        "data/processed/"
        "final_test_predictions.csv"
    )

    test_output.to_csv(
        prediction_path,
        index=False,
    )

    patient_path = Path(
        "data/processed/"
        "final_test_patient_results.csv"
    )

    patient_results.to_csv(
        patient_path,
        index=False,
    )

    print()
    print(
        f"Saved predictions to: "
        f"{prediction_path}"
    )

    print(
        f"Saved patient results to: "
        f"{patient_path}"
    )


if __name__ == "__main__":
    main()