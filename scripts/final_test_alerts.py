from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = (
    "data/processed/final_test_predictions.csv"
)

FEATURES_PATH = (
    "data/processed/hupa_features.csv"
)

THRESHOLD = 0.50
SAMPLING_MINUTES = 5
HORIZON_MINUTES = 30


def find_true_hypoglycemia_events(patient_df):
    patient_df = (
        patient_df
        .sort_values("time")
        .reset_index(drop=True)
    )

    low = (
        patient_df["glucose"].to_numpy() < 70.0
    )

    times = patient_df["time"].to_numpy()

    events = []
    i = 0

    while i < len(patient_df):

        if not low[i]:
            i += 1
            continue

        start = i

        while (
            i + 1 < len(patient_df)
            and low[i + 1]
            and (
                (
                    times[i + 1] - times[i]
                )
                / np.timedelta64(1, "m")
                <= SAMPLING_MINUTES
            )
        ):
            i += 1

        end = i

        events.append(
            {
                "start_time": pd.Timestamp(
                    times[start]
                ),
                "end_time": pd.Timestamp(
                    times[end]
                ),
            }
        )

        i += 1

    return events


def find_alert_episodes(patient_df):
    patient_df = (
        patient_df
        .sort_values("time")
        .reset_index(drop=True)
    )

    alert = (
        patient_df[
            "calibrated_probability"
        ].to_numpy()
        >= THRESHOLD
    )

    times = patient_df["time"].to_numpy()

    episodes = []
    i = 0

    while i < len(patient_df):

        if not alert[i]:
            i += 1
            continue

        start = i

        while (
            i + 1 < len(patient_df)
            and alert[i + 1]
            and (
                (
                    times[i + 1] - times[i]
                )
                / np.timedelta64(1, "m")
                <= SAMPLING_MINUTES
            )
        ):
            i += 1

        end = i

        episodes.append(
            {
                "start_time": pd.Timestamp(
                    times[start]
                ),
                "end_time": pd.Timestamp(
                    times[end]
                ),
            }
        )

        i += 1

    return episodes


def evaluate_patient(patient_df):
    patient_df = (
        patient_df
        .sort_values("time")
        .reset_index(drop=True)
    )

    true_events = find_true_hypoglycemia_events(
        patient_df
    )

    alert_episodes = find_alert_episodes(
        patient_df
    )

    matched_events = set()
    matched_alerts = set()
    lead_times = []

    for alert_index, alert in enumerate(
        alert_episodes
    ):

        alert_time = alert["start_time"]

        for event_index, event in enumerate(
            true_events
        ):

            if event_index in matched_events:
                continue

            lead_minutes = (
                event["start_time"] - alert_time
            ) / np.timedelta64(1, "m")

            # Strictly before event onset and
            # no more than 30 minutes ahead.
            if (
                0
                < lead_minutes
                <= HORIZON_MINUTES
            ):
                matched_events.add(event_index)
                matched_alerts.add(alert_index)
                lead_times.append(
                    float(lead_minutes)
                )
                break

    event_sensitivity = (
        len(matched_events) / len(true_events)
        if true_events
        else np.nan
    )

    false_alerts = (
        len(alert_episodes)
        - len(matched_alerts)
    )

    duration_days = (
        (
            patient_df["time"].max()
            - patient_df["time"].min()
        ).total_seconds()
        / (60 * 60 * 24)
    )

    false_alerts_per_day = (
        false_alerts / duration_days
        if duration_days > 0
        else np.nan
    )

    return {
        "patient_id": patient_df[
            "patient_id"
        ].iloc[0],
        "true_events": len(true_events),
        "alert_episodes": len(alert_episodes),
        "matched_events": len(matched_events),
        "false_alerts": false_alerts,
        "event_sensitivity": event_sensitivity,
        "false_alerts_per_day": false_alerts_per_day,
        "median_lead_time_minutes": (
            np.median(lead_times)
            if lead_times
            else np.nan
        ),
    }


def main():

    print("Loading final test predictions...")
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["time"],
    )

    print("Loading observed glucose...")
    features = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["time"],
    )

    glucose = features[
        [
            "patient_id",
            "time",
            "glucose",
        ]
    ]

    df = predictions.merge(
        glucose,
        on=[
            "patient_id",
            "time",
        ],
        how="left",
        validate="one_to_one",
    )

    if df["glucose"].isna().any():
        raise RuntimeError(
            "Some final test predictions could not "
            "be matched to observed glucose."
        )

    print()
    print("=" * 80)
    print("LOCKED FINAL TEST — ALERT EPISODES")
    print("=" * 80)

    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Rows: {len(df):,}")

    results = []

    for patient_id, patient_df in (
        df.groupby("patient_id")
    ):
        results.append(
            evaluate_patient(patient_df)
        )

    results_df = pd.DataFrame(results)

    print()
    print("Per-patient results:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 80)
    print("FINAL TEST ALERT SUMMARY")
    print("=" * 80)

    print(
        f"Median event sensitivity: "
        f"{results_df['event_sensitivity'].median():.4f}"
    )

    print(
        f"Event sensitivity IQR: "
        f"{results_df['event_sensitivity'].quantile(0.25):.4f} - "
        f"{results_df['event_sensitivity'].quantile(0.75):.4f}"
    )

    print(
        f"Median false alerts/day: "
        f"{results_df['false_alerts_per_day'].median():.4f}"
    )

    print(
        f"False alerts/day IQR: "
        f"{results_df['false_alerts_per_day'].quantile(0.25):.4f} - "
        f"{results_df['false_alerts_per_day'].quantile(0.75):.4f}"
    )

    print(
        f"Median lead time: "
        f"{results_df['median_lead_time_minutes'].median():.2f} minutes"
    )

    output_path = Path(
        "data/processed/"
        "final_test_alert_episode_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()