import pandas as pd
from src.labels.hypoglycemia import add_30min_hypoglycemia_label

def test_future_below_70_creates_positive_label():
    times = pd.date_range("2026-01-01", periods=8, freq="5min")
    glucose = [100, 100, 100, 100, 100, 100, 69, 100]
    df = pd.DataFrame({"time": times, "glucose": glucose})
    out = add_30min_hypoglycemia_label(df)
    assert out.loc[0, "target_hypo_30m"] == 1

def test_no_future_below_70_creates_negative_label():
    times = pd.date_range("2026-01-01", periods=8, freq="5min")
    glucose = [100] * 8
    df = pd.DataFrame({"time": times, "glucose": glucose})
    out = add_30min_hypoglycemia_label(df)
    assert out.loc[0, "target_hypo_30m"] == 0
