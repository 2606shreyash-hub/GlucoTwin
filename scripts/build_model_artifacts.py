from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "hupa_features.csv"
OOF_FILE = PROJECT_ROOT / "data" / "processed" / "random_forest_oof_predictions.csv"
SPLIT_FILE = PROJECT_ROOT / "data" / "processed" / "hupa_patient_split.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


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

TARGET = "hypo_next_30m"

THRESHOLD_MG_DL = 70
HORIZON_MINUTES = 30
SAMPLING_MINUTES = 5
ALERT_THRESHOLD = 0.50


def patient_balanced_weights(patient_ids):
    counts = pd.Series(patient_ids).value_counts()
    n_patients = len(counts)

    weights = pd.Series(patient_ids).map(
        lambda patient: n_patients / counts[patient]
    )

    return weights.to_numpy(dtype=float)


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature dataset...")
    df = pd.read_csv(FEATURE_FILE)

    split = pd.read_csv(SPLIT_FILE)

    split_map = dict(zip(split["patient_id"], split["split"]))
    df["split"] = df["patient_id"].map(split_map)

    if df["split"].isna().any():
        missing = df.loc[df["split"].isna(), "patient_id"].unique()
        raise ValueError(
            f"Patients missing from split file: {missing}"
        )

    development = df[
        (df["split"] == "development")
        & df[TARGET].notna()
    ].copy()

    development["time"] = pd.to_datetime(development["time"])

    development = development.sort_values(
        ["patient_id", "time"]
    )

    first_time = development.groupby(
        "patient_id"
    )["time"].transform("min")

    elapsed_minutes = (
        development["time"] - first_time
    ).dt.total_seconds() / 60.0

    development = development[
        elapsed_minutes >= 120
    ].copy()

    X = development[FEATURES]
    y = development[TARGET].astype(int).to_numpy()

    patient_ids = development["patient_id"].to_numpy()

    weights = patient_balanced_weights(patient_ids)

    print(f"Development rows: {len(development):,}")
    print(
        f"Development patients: "
        f"{development['patient_id'].nunique()}"
    )
    print(f"Positive rows: {y.sum():,}")

    print("\nFitting feature imputer...")

    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    X_imp = imputer.fit_transform(X)

    print("Training final Random Forest...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_imp,
        y,
        sample_weight=weights,
    )

    print("Loading development OOF predictions...")

    oof = pd.read_csv(OOF_FILE)

    if "probability" not in oof.columns:
        raise ValueError(
            "Expected 'probability' column in OOF predictions."
    )

    raw_oof = oof["probability"].to_numpy()
    oof_y = oof["hypo_next_30m"].astype(int).to_numpy()

    print("Fitting sigmoid calibration...")

    eps = 1e-6
    clipped = np.clip(raw_oof, eps, 1 - eps)

    oof_logit = np.log(
        clipped / (1 - clipped)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=42,
    )

    calibrator.fit(
        oof_logit,
        oof_y,
    )

    print("\nSaving model artifacts...")

    joblib.dump(
        model,
        ARTIFACT_DIR / "random_forest.joblib",
    )

    joblib.dump(
        imputer,
        ARTIFACT_DIR / "feature_imputer.joblib",
    )

    joblib.dump(
        calibrator,
        ARTIFACT_DIR / "sigmoid_calibrator.joblib",
    )

    config = {
        "project": "GlucoTwin",
        "prediction_target": (
            "Probability that blood glucose will fall below "
            "70 mg/dL at any point within the next 30 minutes."
        ),
        "target_column": TARGET,
        "hypoglycemia_threshold_mg_dl": THRESHOLD_MG_DL,
        "prediction_horizon_minutes": HORIZON_MINUTES,
        "sampling_interval_minutes": SAMPLING_MINUTES,
        "alert_threshold": ALERT_THRESHOLD,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 20,
            "random_state": 42,
            "patient_balanced_sample_weights": True,
        },
        "calibration": {
            "type": "sigmoid",
            "training_source": "development OOF predictions",
        },
        "features": FEATURES,
        "notes": [
            "Final model trained only on development patients.",
            "Held-out test patients were not used for training.",
            "Alert threshold frozen at 0.50.",
            "Retrospective proof-of-concept only.",
            "Not clinically validated.",
        ],
    }

    with open(
        ARTIFACT_DIR / "model_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )

    print("\nCreated artifacts:")

    for path in sorted(ARTIFACT_DIR.iterdir()):
        print(f"  {path.name}")

    print("\nModel artifact packaging complete.")


if __name__ == "__main__":
    main()