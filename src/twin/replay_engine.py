from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "hupa_features.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


class ReplayEngine:
    """Time-synchronized Digital Twin replay engine.

    At replay time T, only observations with timestamp <= T are exposed
    to the feature and prediction pipeline.
    """

    def __init__(
        self,
        feature_file: Path = FEATURE_FILE,
        artifact_dir: Path = ARTIFACT_DIR,
    ) -> None:
        self.feature_file = Path(feature_file)
        self.artifact_dir = Path(artifact_dir)

        self.model = joblib.load(
            self.artifact_dir / "random_forest.joblib"
        )

        self.imputer = joblib.load(
            self.artifact_dir / "feature_imputer.joblib"
        )

        self.calibrator = joblib.load(
            self.artifact_dir / "sigmoid_calibrator.joblib"
        )

        with open(
            self.artifact_dir / "model_config.json",
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        self.features = self.config["features"]

        self.alert_threshold = float(
            self.config["alert_threshold"]
        )

        self.horizon_minutes = int(
            self.config["prediction_horizon_minutes"]
        )

        self.sampling_minutes = int(
            self.config["sampling_interval_minutes"]
        )

        self._data: pd.DataFrame | None = None

    def load_data(self) -> None:
        """Load the processed HUPA dataset once."""

        df = pd.read_csv(self.feature_file)

        if "patient_id" not in df.columns:
            raise ValueError(
                "Expected 'patient_id' column in feature dataset."
            )

        if "time" not in df.columns:
            raise ValueError(
                "Expected 'time' column in feature dataset."
            )

        df["time"] = pd.to_datetime(
            df["time"],
            errors="raise",
        )

        df = df.sort_values(
            ["patient_id", "time"]
        ).reset_index(drop=True)

        self._data = df

    def _ensure_data_loaded(self) -> pd.DataFrame:
        if self._data is None:
            self.load_data()

        assert self._data is not None
        return self._data

    def list_patients(self) -> list[str]:
        """Return available patient identifiers."""

        df = self._ensure_data_loaded()

        return sorted(
            df["patient_id"].dropna().unique().tolist()
        )

    def get_patient_data(
        self,
        patient_id: str,
    ) -> pd.DataFrame:
        """Return the complete processed timeline for one patient."""

        df = self._ensure_data_loaded()

        patient = df[
            df["patient_id"] == patient_id
        ].copy()

        if patient.empty:
            raise ValueError(
                f"Unknown patient_id: {patient_id}"
            )

        return patient.sort_values("time").reset_index(
            drop=True
        )

    def get_replay_state(
        self,
        patient_id: str,
        timestamp: str | pd.Timestamp,
    ) -> dict[str, Any]:
        """Generate the Digital Twin state at replay time T.

        Only observations with time <= T are used.
        """

        timestamp = pd.Timestamp(timestamp)

        patient = self.get_patient_data(patient_id)

        available = patient[
            patient["time"] <= timestamp
        ].copy()

        if available.empty:
            raise ValueError(
                f"No observations available for {patient_id} "
                f"at or before {timestamp}."
            )

        # Never expose the future target to the feature pipeline.
        future_target_columns = [
            column
            for column in available.columns
            if column.startswith("hypo_")
            or column.startswith("target_")
        ]

        available = available.drop(
            columns=future_target_columns,
            errors="ignore",
        )

        # The original HUPA processed data contains the causal raw
        # measurements. Rebuild features using only observations
        # available up to the replay timestamp.
        featured = build_features(
            available,
            freq_minutes=self.sampling_minutes,
        )

        current = featured.iloc[-1]

        missing_features = [
            feature
            for feature in self.features
            if feature not in featured.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing model features: "
                + ", ".join(missing_features)
            )

        X = pd.DataFrame(
            [
                current[self.features].to_dict()
            ]
        )

        X_imputed = self.imputer.transform(X)

        raw_probability = float(
            self.model.predict_proba(X_imputed)[0, 1]
        )

        clipped = np.clip(
            raw_probability,
            1e-6,
            1 - 1e-6,
        )

        logit = np.log(
            clipped / (1.0 - clipped)
        )

        calibrated_probability = float(
            self.calibrator.predict_proba(
                np.array([[logit]])
            )[0, 1]
        )

        alert_active = (
            calibrated_probability
            >= self.alert_threshold
        )

        alert_state = (
            "ALERT"
            if alert_active
            else "MONITOR"
        )

        state = {
            "patient_id": patient_id,
            "timestamp": timestamp.isoformat(),
            "latest_observation": {
                "time": current["time"].isoformat(),
                "glucose_mg_dl": self._safe_float(
                    current.get("glucose")
                ),
                "heart_rate_bpm": self._safe_float(
                    current.get("heart_rate")
                ),
                "steps": self._safe_float(
                    current.get("steps")
                ),
                "calories": self._safe_float(
                    current.get("calories")
                ),
                "basal_rate": self._safe_float(
                    current.get("basal_rate")
                ),
                "bolus_volume_delivered": self._safe_float(
                    current.get("bolus_volume_delivered")
                ),
                "carb_input": self._safe_float(
                    current.get("carb_input")
                ),
            },
            "prediction": {
                "hypoglycemia_probability": calibrated_probability,
                "horizon_minutes": self.horizon_minutes,
                "threshold_mg_dl": self.config[
                    "hypoglycemia_threshold_mg_dl"
                ],
                "alert_threshold": self.alert_threshold,
            },
            "alert": {
                "state": alert_state,
                "active": alert_active,
            },
            "data_boundary": {
                "replay_timestamp": timestamp.isoformat(),
                "latest_used_observation": current[
                    "time"
                ].isoformat(),
                "future_observations_used": False,
            },
        }

        return state

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert numeric values safely for API/dashboard use."""

        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(value):
            return None

        return value