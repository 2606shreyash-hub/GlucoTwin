from pathlib import Path

import pandas as pd

from src.twin.replay_engine import ReplayEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_replay_engine_loads():
    engine = ReplayEngine()

    patients = engine.list_patients()

    assert len(patients) == 25
    assert "HUPA0026P" in patients


def test_replay_uses_only_past_data():
    engine = ReplayEngine()

    patient = engine.get_patient_data("HUPA0026P")

    timestamp = patient["time"].iloc[200]

    state = engine.get_replay_state(
        "HUPA0026P",
        timestamp,
    )

    assert state["patient_id"] == "HUPA0026P"

    assert (
        state["data_boundary"]["future_observations_used"]
        is False
    )

    latest_used = pd.Timestamp(
        state["data_boundary"]["latest_used_observation"]
    )

    assert latest_used <= timestamp


def test_replay_returns_prediction():
    engine = ReplayEngine()

    patient = engine.get_patient_data("HUPA0026P")

    timestamp = patient["time"].iloc[200]

    state = engine.get_replay_state(
        "HUPA0026P",
        timestamp,
    )

    probability = state[
        "prediction"
    ]["hypoglycemia_probability"]

    assert 0.0 <= probability <= 1.0

    assert state["prediction"]["horizon_minutes"] == 30

    assert state["prediction"]["threshold_mg_dl"] == 70

    assert state["alert"]["state"] in {
        "ALERT",
        "MONITOR",
    }