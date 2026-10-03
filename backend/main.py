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


@app.get("/patients/{patient_id}/risk-timeline")
def get_patient_risk_timeline(
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
            "risk_timeline": replay_engine.get_risk_timeline(
                patient_id=patient_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/patients/{patient_id}/next-hypoglycemia")
def get_next_hypoglycemia(
    patient_id: str,
    after_timestamp: str | None = Query(
        default=None,
        description="Only return events after this timestamp.",
    ),
):
    try:
        event = replay_engine.find_next_hypoglycemia_event(
            patient_id=patient_id,
            after_timestamp=after_timestamp,
        )

        if event is None:
            raise HTTPException(
                status_code=404,
                detail="No hypoglycemia event found.",
            )

        return event

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc        


@app.get("/patients/{patient_id}/event-detection")
def get_event_detection(
    patient_id: str,
    event_start: str = Query(
        ...,
        description="Observed hypoglycemia event start timestamp.",
    ),
):
    try:
        return replay_engine.get_event_detection(
            patient_id=patient_id,
            event_start=event_start,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/patients/{patient_id}/event-replay")
def get_event_replay(
    patient_id: str,
    event_start: str = Query(
        ...,
        description="Observed hypoglycemia event start timestamp.",
    ),
    before_minutes: int = Query(
        default=60,
        ge=5,
        le=240,
    ),
    after_minutes: int = Query(
        default=30,
        ge=5,
        le=120,
    ),
):
    try:
        return replay_engine.get_event_replay(
            patient_id=patient_id,
            event_start=event_start,
            before_minutes=before_minutes,
            after_minutes=after_minutes,
        )
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