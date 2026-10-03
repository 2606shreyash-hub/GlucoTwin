from fastapi import FastAPI, HTTPException, Query
from src.twin.replay_engine import ReplayEngine
from backend.profiles import get_patient_profile


app = FastAPI(
    title="GlucoTwin API",
    description="Digital Twin API for early hypoglycemia risk prediction.",
    version="0.1.0",
)


replay_engine = ReplayEngine()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "GlucoTwin API",
    }


@app.get("/patients")
def get_patients():
    return {
        "patients": replay_engine.list_patients()
    }


@app.get("/patients/{patient_id}/profile")
def get_patient_profile_endpoint(patient_id: str):
    try:
        return get_patient_profile(patient_id)
    except ValueError as exc:
raise HTTPException(status_code=404, detail=str(exc)) from exc 


@app.get("/patients/{patient_id}/timeline")
def get_patient_timeline(
    patient_id: str,
    start_timestamp: str | None = Query(
        default=None,
        description="Optional start timestamp in ISO format.",
    ),
    end_timestamp: str | None = Query(
        default=None,
        description="Optional end timestamp in ISO format.",
    ),
):
    try:
        return {
            "patient_id": patient_id,
            "timeline": replay_engine.get_timeline(
                patient_id=patient_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            ),
        }
    except ValueError as exc:
raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/patients/{patient_id}/state")
def get_patient_state(
    patient_id: str,
    timestamp: str = Query(
        ...,
        description="Replay timestamp in ISO format.",
    ),
):
    try:
        return replay_engine.get_replay_state(
            patient_id=patient_id,
            timestamp=timestamp,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc