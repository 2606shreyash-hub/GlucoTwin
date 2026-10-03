import numpy as np
import pandas as pd

def add_30min_hypoglycemia_label(
    df: pd.DataFrame,
    threshold: float = 70.0,
    horizon_minutes: int = 30,
    sampling_minutes: int = 5,
) -> pd.DataFrame:
    """Create Y(t)=1 if any observed glucose in (t,t+30m] is < threshold.

    This implementation requires a complete 30-minute future horizon on the
    processed grid. It does not use future values as model features.
    """
    out = df.copy().sort_values("time").reset_index(drop=True)

    steps = horizon_minutes // sampling_minutes
    future_min = pd.Series(np.nan, index=out.index, dtype=float)

    # For each t, inspect t+5 ... t+30.
    future_arrays = [
        out["glucose"].shift(-k) for k in range(1, steps + 1)
    ]
    future_matrix = pd.concat(future_arrays, axis=1)

    complete = future_matrix.notna().all(axis=1)
    future_min.loc[complete] = future_matrix.loc[complete].min(axis=1)

    out["target_hypo_30m"] = np.where(
        complete,
        (future_min < threshold).astype("int8"),
        np.nan,
    )

    return out
