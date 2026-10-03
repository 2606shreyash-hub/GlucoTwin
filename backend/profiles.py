"""Synthetic static patient profiles for the GlucoTwin demo.

These profiles are demonstration metadata only.
They are not claimed to be the actual EHR records of HUPA-UCM participants.
"""

PATIENT_PROFILES = {
    "HUPA0001P": {
        "patient_id": "HUPA0001P",
        "data_type": "synthetic_demo_profile",
        "age": 24,
        "sex": "Female",
        "condition": "Type 1 Diabetes",
        "years_since_diagnosis": 9,
        "treatment_modality": "Insulin pump",
    },
    "HUPA0002P": {
        "patient_id": "HUPA0002P",
        "data_type": "synthetic_demo_profile",
        "age": 31,
        "sex": "Male",
        "condition": "Type 1 Diabetes",
        "years_since_diagnosis": 14,
        "treatment_modality": "Insulin pump",
    },
}


def get_patient_profile(patient_id: str) -> dict:
    """Return the synthetic demo profile for a patient."""
    if patient_id not in PATIENT_PROFILES:
        raise ValueError(f"No demo profile found for patient {patient_id}")

    return PATIENT_PROFILES[patient_id]