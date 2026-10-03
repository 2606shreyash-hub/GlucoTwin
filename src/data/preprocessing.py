import numpy as np
import pandas as pd

def clean_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only causal/data-quality transformations that are known at time t.

    Important:
    - Do not interpolate future glucose values.
    - Do not use centered rolling windows.
    - Negative delivered-bolus values are treated as invalid exposure values.
    """
    out = df.copy()

    # Delivered insulin cannot physically be negative. HUPA0017P contains
    # four negative values that correspond to a different raw-data field.
    out["bolus_valid"] = out["bolus_volume_delivered"].ge(0)
    out["bolus_exposure"] = out["bolus_volume_delivered"].where(
        out["bolus_valid"], np.nan
    )

    # Keep the original column for auditability.
    # Exposure calculations should use bolus_exposure, never the raw negative value.

    out = out.drop_duplicates(subset=["time"], keep="first")
    out = out.sort_values("time").reset_index(drop=True)

    return out

def audit_patient(df: pd.DataFrame) -> dict:
    d = df.sort_values("time")
    deltas = d["time"].diff().dropna().dt.total_seconds().div(60)

    return {
        "patient_id": d["patient_id"].iloc[0],
        "rows": len(d),
        "start": d["time"].min(),
        "end": d["time"].max(),
        "duration_days": (d["time"].max() - d["time"].min()).total_seconds() / 86400,
        "duplicate_timestamps": int(d["time"].duplicated().sum()),
        "missing_glucose": int(d["glucose"].isna().sum()),
        "negative_bolus_rows": int((d["bolus_volume_delivered"] < 0).sum()),
        "max_gap_minutes": float(deltas.max()) if len(deltas) else 0.0,
        "median_gap_minutes": float(deltas.median()) if len(deltas) else 0.0,
        "glucose_below_70": int((d["glucose"] < 70).sum()),
    }
