import numpy as np
import pandas as pd

from src.features.feature_engineering import build_features


def make_df(glucose):
    n = len(glucose)
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="5min"),
        "glucose": glucose,
        "basal_rate": np.ones(n),
        "bolus_volume_delivered": np.zeros(n),
        "carb_input": np.zeros(n),
        "heart_rate": np.full(n, 70.0),
        "steps": np.zeros(n),
        "calories": np.zeros(n),
    })


def test_future_value_does_not_change_causal_feature():
    base = make_df([100.0] * 25)
    changed = base.copy()
    changed.loc[20:, "glucose"] = 300.0

    f1 = build_features(base)
    f2 = build_features(changed)

    # Row 10 is 50 minutes before the changed region, so its causal features
    # must be identical.
    cols = ["slope_15m", "slope_30m", "slope_60m", "glucose_mean_30m", "glucose_min_30m"]
    pd.testing.assert_series_equal(f1.loc[10, cols], f2.loc[10, cols], check_names=False)


def test_slope_direction():
    glucose = np.arange(25, dtype=float) + 100.0
    features = build_features(make_df(glucose))
    assert features.loc[24, "slope_30m"] > 0
