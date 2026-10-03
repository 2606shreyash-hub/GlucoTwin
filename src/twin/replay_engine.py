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

    def get_timeline(
        self,
        patient_id: str,
        start_timestamp: str | None = None,
        end_timestamp: str | None = None,
    ) -> list[dict]:
        """Return historical observations for a patient.

        Only observations within the requested time range are returned.
        This endpoint exposes observed data only; it does not reveal
        future observations beyond the requested end timestamp.
        """

        df = self._ensure_data_loaded()

        if patient_id not in df["patient_id"].unique():
            raise ValueError(f"Unknown patient_id: {patient_id}")

        patient = df[df["patient_id"] == patient_id].copy()

        patient["time"] = pd.to_datetime(patient["time"])

        if start_timestamp is not None:
            start = pd.to_datetime(start_timestamp)
            patient = patient[patient["time"] >= start]

        if end_timestamp is not None:
            end = pd.to_datetime(end_timestamp)
            patient = patient[patient["time"] <= end]

        if patient.empty:
            return []

        timeline = []

        for _, row in patient.iterrows():
            timeline.append(
                {
                    "timestamp": row["time"].isoformat(),
                    "glucose_mg_dl": self._safe_float(row["glucose"]),
                    "heart_rate_bpm": self._safe_float(
                        row.get("heart_rate")
                    ),
                    "steps": self._safe_float(row.get("steps")),
                    "calories": self._safe_float(row.get("calories")),
                    "basal_rate": self._safe_float(
                        row.get("basal_rate")
                    ),
                    "bolus_volume_delivered": self._safe_float(
                        row.get("bolus_volume_delivered")
                    ),
                    "carb_input": self._safe_float(
                        row.get("carb_input")
                    ),
                }
            )

        return timeline

    def get_risk_timeline(
        self,
        patient_id: str,
        start_timestamp: str | None = None,
        end_timestamp: str | None = None,
    ) -> list[dict]:
        """Return replayed glucose and hypoglycemia risk over time.

        Each prediction is generated using only observations available
        at that timestamp. Future observations are never passed to the
        prediction pipeline.
        """

        timeline = self.get_timeline(
            patient_id=patient_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

        risk_timeline = []

        for observation in timeline:
            timestamp = observation["timestamp"]

            state = self.get_replay_state(
                patient_id=patient_id,
                timestamp=timestamp,
            )

            risk_timeline.append(
                {
                    "timestamp": timestamp,
                    "glucose_mg_dl": observation["glucose_mg_dl"],
                    "hypoglycemia_probability": state[
                        "prediction"
                    ]["hypoglycemia_probability"],
                    "horizon_minutes": state[
                        "prediction"
                    ]["horizon_minutes"],
                    "alert_threshold": state[
                        "prediction"
                    ]["alert_threshold"],
                    "alert_state": state["alert"]["state"],
                    "alert_active": state["alert"]["active"],
                }
            )

        return risk_timeline  

    def find_next_hypoglycemia_event(
        self,
        patient_id: str,
        after_timestamp: str | None = None,
    ) -> dict | None:
        """Find the next observed hypoglycemia episode for a patient.

        An episode begins when glucose crosses from >=70 mg/dL to <70 mg/dL.
        Consecutive below-70 observations are treated as one episode.
        """

        patient = self.get_patient_data(patient_id).copy()
        patient["time"] = pd.to_datetime(patient["time"])
        patient = patient.sort_values("time").reset_index(drop=True)

        if after_timestamp is not None:
            after = pd.to_datetime(after_timestamp)
            patient = patient[patient["time"] > after].reset_index(drop=True)

        if patient.empty:
            return None

        glucose = patient["glucose"].astype(float)
        below = glucose < 70.0

        episodes = []
        in_episode = False
        start_index = None

        for index, is_below in enumerate(below):
            if is_below and not in_episode:
                in_episode = True
                start_index = index

            elif not is_below and in_episode:
                end_index = index - 1
                episodes.append((start_index, end_index))
                in_episode = False
                start_index = None

        if in_episode and start_index is not None:
            episodes.append((start_index, len(patient) - 1))

        if not episodes:
            return None

        start_index, end_index = episodes[0]

        start_row = patient.iloc[start_index]
        end_row = patient.iloc[end_index]

        return {
            "patient_id": patient_id,
            "event_start": start_row["time"].isoformat(),
            "event_end": end_row["time"].isoformat(),
            "duration_minutes": (
                end_row["time"] - start_row["time"]
            ).total_seconds() / 60.0,
            "minimum_glucose_mg_dl": float(
                patient.iloc[start_index:end_index + 1]["glucose"].min()
            ),
        }   


    def get_event_detection(
        self,
        patient_id: str,
        event_start: str,
    ) -> dict:
        """Determine whether the model alerted before an observed
        hypoglycemia event.

        Only predictions generated at timestamps before the event
        are considered valid detections.
        """

        event_time = pd.to_datetime(event_start)

        patient = self.get_patient_data(patient_id).copy()
        patient["time"] = pd.to_datetime(patient["time"])
        patient = patient.sort_values("time").reset_index(drop=True)

        # Only use observations available before the event.
        patient = patient[patient["time"] < event_time].copy()

        if patient.empty:
            return {
                "patient_id": patient_id,
                "event_start": event_time.isoformat(),
                "detected": False,
                "alert_timestamp": None,
                "lead_time_minutes": None,
                "alert_probability": None,
                "glucose_at_alert_mg_dl": None,
            }

        # Check predictions chronologically.
        for timestamp in patient["time"]:
            state = self.get_replay_state(
                patient_id=patient_id,
                timestamp=timestamp.isoformat(),
            )

            probability = state["prediction"]["hypoglycemia_probability"]

            if probability >= state["prediction"]["alert_threshold"]:
                lead_time = (
                    event_time - timestamp
                ).total_seconds() / 60.0

                return {
                    "patient_id": patient_id,
                    "event_start": event_time.isoformat(),
                    "detected": True,
                    "alert_timestamp": timestamp.isoformat(),
                    "lead_time_minutes": lead_time,
                    "alert_probability": probability,
                    "glucose_at_alert_mg_dl": state[
                        "latest_observation"
                    ]["glucose_mg_dl"],
                }

        return {
            "patient_id": patient_id,
            "event_start": event_time.isoformat(),
            "detected": False,
            "alert_timestamp": None,
            "lead_time_minutes": None,
            "alert_probability": None,
            "glucose_at_alert_mg_dl": None,
        } 

    def get_event_replay(
        self,
        patient_id: str,
        event_start: str,
        before_minutes: int = 60,
        after_minutes: int = 30,
    ) -> dict:
        """Return a frontend-ready replay around an observed
        hypoglycemia event.
        """

        event_time = pd.to_datetime(event_start)

        start_time = event_time - pd.Timedelta(minutes=before_minutes)
        end_time = event_time + pd.Timedelta(minutes=after_minutes)

        timeline = self.get_risk_timeline(
            patient_id=patient_id,
            start_timestamp=start_time.isoformat(),
            end_timestamp=end_time.isoformat(),
        )

        detection = self.get_event_detection(
            patient_id=patient_id,
            event_start=event_time.isoformat(),
        )

        event = self.find_next_hypoglycemia_event(
            patient_id=patient_id,
            after_timestamp=(
                event_time - pd.Timedelta(minutes=1)
            ).isoformat(),
        )

        event_end = event["event_end"] if event else None
        minimum_glucose = (
            event["minimum_glucose_mg_dl"]
            if event
            else None
        )

        return {
            "patient": {
                "patient_id": patient_id,
                "condition": "Type 1 Diabetes",
            },
            "event": {
                "type": "hypoglycemia",
                "threshold_mg_dl": 70.0,
                "start": event_time.isoformat(),
                "end": event_end,
                "minimum_glucose_mg_dl": minimum_glucose,
            },
            "alert": {
                "detected": detection["detected"],
                "timestamp": detection["alert_timestamp"],
                "probability": detection["alert_probability"],
                "threshold": self.alert_threshold,
                "lead_time_minutes": detection["lead_time_minutes"],
                "glucose_at_alert_mg_dl": detection[
                    "glucose_at_alert_mg_dl"
                ],
            },
            "window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "before_minutes": before_minutes,
                "after_minutes": after_minutes,
            },
            "timeline": timeline,
        }                  

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