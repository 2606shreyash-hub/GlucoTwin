from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = (
    "data/processed/random_forest_calibrated_oof.csv"
)

FEATURES_PATH = (
    "data/processed/hupa_features.csv"
)

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]

SAMPLING_MINUTES = 5
HORIZON_MINUTES = 30


def find_true_hypoglycemia_events(patient_df):
    """
    Identify contiguous periods where glucose is below 70 mg/dL.

    Consecutive 5-minute observations below 70 mg/dL
    are treated as one hypoglycemia episode.
    """

    patient_df = (
        patient_df
        .sort_values("time")
        .reset_index(drop=True)
    )

    low = (
        patient_df["glucose"].to_numpy()
        < 70.0
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
                    times[i + 1]
                    - times[i]
                )
                / np.timedelta64(1, "m")
                <= SAMPLING_MINUTES
            )
        ):
            i += 1

        end = i

        events.append(
            {
                "start_index": start,
                "end_index": end,
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


def find_alert_episodes(
    patient_df,
    threshold,
):
    """
    Convert consecutive positive model predictions
    into one alert episode.
    """

    patient_df = (
        patient_df
        .sort_values("time")
        .reset_index(drop=True)
    )

    alert = (
        patient_df[
            "calibrated_probability"
        ].to_numpy()
        >= threshold
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
                    times[i + 1]
                    - times[i]
                )
                / np.timedelta64(1, "m")
                <= SAMPLING_MINUTES
            )
        ):
            i += 1

        end = i

        episodes.append(
            {
                "start_index": start,
                "end_index": end,
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


def evaluate_patient(
    patient_df,
    threshold,
):
    """
    Match alert episodes against future
    hypoglycemia events.
    """

    patient_df = (
        patient_df
        .sort_values("time")
        .reset_index(drop=True)
    )

    true_events = find_true_hypoglycemia_events(
        patient_df
    )

    alert_episodes = find_alert_episodes(
        patient_df,
        threshold,
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

            event_start = event["start_time"]

            lead_minutes = (
                event_start - alert_time
            ) / np.timedelta64(1, "m")

            if (
                0
                < lead_minutes
                <= HORIZON_MINUTES
            ):

                matched_events.add(
                    event_index
                )

                matched_alerts.add(
                    alert_index
                )

                lead_times.append(
                    float(lead_minutes)
                )

                break

    event_sensitivity = (
        len(matched_events)
        / len(true_events)
        if len(true_events) > 0
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

        "threshold": threshold,

        "true_events": len(true_events),

        "alert_episodes": len(
            alert_episodes
        ),

        "matched_events": len(
            matched_events
        ),

        "false_alerts": false_alerts,

        "event_sensitivity": (
            event_sensitivity
        ),

        "false_alerts_per_day": (
            false_alerts_per_day
        ),

        "median_lead_time_minutes": (
            np.median(lead_times)
            if lead_times
            else np.nan
        ),
    }


def main():

    print(
        "Loading calibrated OOF predictions..."
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["time"],
    )

    print(
        "Loading glucose data..."
    )

    features = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["time"],
    )

    # We only need glucose and the identifiers
    # from the feature dataset.
    glucose = features[
        [
            "patient_id",
            "time",
            "glucose",
        ]
    ].copy()

    # Merge the model predictions with the
    # corresponding observed glucose.
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
        missing = int(
            df["glucose"].isna().sum()
        )

        raise RuntimeError(
            f"{missing} prediction rows "
            "could not be matched to glucose."
        )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Patients: "
        f"{df['patient_id'].nunique()}"
    )

    print()

    all_results = []

    for threshold in THRESHOLDS:

        print(
            f"Evaluating threshold: "
            f"{threshold:.2f}"
        )

        for patient_id, patient_df in (
            df.groupby("patient_id")
        ):

            result = evaluate_patient(
                patient_df,
                threshold,
            )

            all_results.append(result)

    results = pd.DataFrame(
        all_results
    )

    # ------------------------------------------------------------
    # Patient-level summary
    # ------------------------------------------------------------

    summary = (
        results
        .groupby("threshold")
        .agg(
            median_event_sensitivity=(
                "event_sensitivity",
                "median",
            ),

            q25_event_sensitivity=(
                "event_sensitivity",
                lambda x: x.quantile(0.25),
            ),

            q75_event_sensitivity=(
                "event_sensitivity",
                lambda x: x.quantile(0.75),
            ),

            median_false_alerts_per_day=(
                "false_alerts_per_day",
                "median",
            ),

            q25_false_alerts_per_day=(
                "false_alerts_per_day",
                lambda x: x.quantile(0.25),
            ),

            q75_false_alerts_per_day=(
                "false_alerts_per_day",
                lambda x: x.quantile(0.75),
            ),

            median_lead_time_minutes=(
                "median_lead_time_minutes",
                "median",
            ),
        )
        .reset_index()
    )

    print()

    print(
        "=" * 100
    )

    print(
        "PATIENT-LEVEL ALERT EPISODE SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    output_path = Path(
        "data/processed/"
        "alert_episode_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    summary_path = Path(
        "data/processed/"
        "alert_episode_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()

    print(
        f"Saved patient results to: "
        f"{output_path}"
    )

    print(
        f"Saved summary to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()