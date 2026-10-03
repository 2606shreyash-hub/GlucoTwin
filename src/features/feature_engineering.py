"""Causal feature engineering for the HUPA-UCM 5-minute dataset.

All rolling features use observations at or before the prediction time t.
The functions intentionally do not inspect future rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (15, 30, 60, 120)


def _minutes_to_rows(minutes: int, freq_minutes: int = 5) -> int:
    if minutes % freq_minutes != 0:
        raise ValueError("Feature window must be divisible by the sampling interval.")
    return minutes // freq_minutes


def _past_rolling(series: pd.Series, minutes: int, agg: str, freq_minutes: int = 5) -> pd.Series:
    rows = _minutes_to_rows(minutes, freq_minutes)
    window = rows + 1  # include current observation t
    rolling = series.rolling(window=window, min_periods=1)
    if agg == "mean":
        return rolling.mean()
    if agg == "min":
        return rolling.min()
    if agg == "std":
        return rolling.std(ddof=0)
    if agg == "sum":
        return rolling.sum()
    raise ValueError(f"Unsupported aggregation: {agg}")


def _past_slope(series: pd.Series, minutes: int, freq_minutes: int) -> pd.Series:
    window = _minutes_to_rows(minutes, freq_minutes) + 1

    def slope(values: np.ndarray) -> float:
        mask = np.isfinite(values)
        if mask.sum() < 2:
            return np.nan

        y = values[mask]
        x = np.arange(len(values), dtype=float)[mask]

        x_centered = x - x.mean()
        y_centered = y - y.mean()

        denominator = np.sum(x_centered ** 2)

        if denominator == 0:
            return np.nan

        return np.sum(x_centered * y_centered) / denominator

    return series.rolling(
        window=window,
        min_periods=window
    ).apply(slope, raw=True)


def add_glucose_features(df: pd.DataFrame, freq_minutes: int = 5) -> pd.DataFrame:
    out = df.copy()
    for minutes in WINDOWS:
        out[f"slope_{minutes}m"] = _past_slope(out["glucose"], minutes, freq_minutes)
    out["glucose_mean_30m"] = _past_rolling(out["glucose"], 30, "mean", freq_minutes)
    out["glucose_min_30m"] = _past_rolling(out["glucose"], 30, "min", freq_minutes)
    out["glucose_std_30m"] = _past_rolling(out["glucose"], 30, "std", freq_minutes)
    out["glucose_mean_60m"] = _past_rolling(out["glucose"], 60, "mean", freq_minutes)
    out["glucose_min_60m"] = _past_rolling(out["glucose"], 60, "min", freq_minutes)
    return out


def _past_sum(df: pd.DataFrame, column: str, minutes: int, freq_minutes: int = 5) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return _past_rolling(df[column], minutes, "sum", freq_minutes)


def add_insulin_and_carb_features(df: pd.DataFrame, freq_minutes: int = 5) -> pd.DataFrame:
    out = df.copy()
    for minutes in (15, 30, 60, 120):
        out[f"bolus_exposure_{minutes}m"] = _past_sum(out, "bolus_volume_delivered", minutes, freq_minutes)
    for minutes in (30, 60, 120):
        # basal_rate is a rate in units/hour. Convert each 5-minute rate
        # observation to approximate delivered units before summing.
        if "basal_rate" in out.columns:
            out[f"basal_exposure_{minutes}m"] = _past_sum(
                out, "basal_rate", minutes, freq_minutes
            ) * (freq_minutes / 60.0)
        else:
            out[f"basal_exposure_{minutes}m"] = np.nan
        out[f"carb_exposure_{minutes}m"] = _past_sum(out, "carb_input", minutes, freq_minutes)

    if "carb_input" in out.columns:
        carb = out["carb_input"].fillna(0)
        seen = np.where(carb.to_numpy() > 0, np.arange(len(out)), np.nan)
        last = pd.Series(seen, index=out.index).ffill()
        out["minutes_since_carb"] = (np.arange(len(out)) - last.to_numpy()) * freq_minutes
        out.loc[last.isna(), "minutes_since_carb"] = np.nan
    else:
        out["minutes_since_carb"] = np.nan
    return out


def add_wearable_features(df: pd.DataFrame, freq_minutes: int = 5) -> pd.DataFrame:
    out = df.copy()
    if "heart_rate" in out.columns:
        out["heart_rate_current"] = out["heart_rate"]
        out["heart_rate_mean_30m"] = _past_rolling(out["heart_rate"], 30, "mean", freq_minutes)
    if "steps" in out.columns:
        out["steps_30m"] = _past_sum(out, "steps", 30, freq_minutes)
    if "calories" in out.columns:
        out["calories_30m"] = _past_sum(out, "calories", 30, freq_minutes)
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hours = out["time"].dt.hour + out["time"].dt.minute / 60.0
    angle = 2.0 * np.pi * hours / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    return out


def build_features(df: pd.DataFrame, freq_minutes: int = 5) -> pd.DataFrame:
    """Return a feature table while preserving original columns."""
    required = {"time", "glucose"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="raise")
    out = out.sort_values("time").reset_index(drop=True)
    out = add_glucose_features(out, freq_minutes)
    out = add_insulin_and_carb_features(out, freq_minutes)
    out = add_wearable_features(out, freq_minutes)
    out = add_time_features(out)
    return out
