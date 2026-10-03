from fastapi import FastAPI, HTTPException, Query

from src.twin.replay_engine import ReplayEngine


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