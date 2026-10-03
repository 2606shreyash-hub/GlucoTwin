from pathlib import Path
import argparse
import pandas as pd


def make_patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df[df["hypo_next_30m"].notna()].copy()

    summary = labeled.groupby("patient_id").agg(
        rows=("patient_id", "size"),
        positive=("hypo_next_30m", "sum"),
        start=("time", "min"),
        end=("time", "max"),
    )

    summary["prevalence"] = summary["positive"] / summary["rows"]
    summary["duration_days"] = (
        summary["end"] - summary["start"]
    ).dt.total_seconds() / 86400.0

    return summary.sort_index()


def choose_test_patients(summary: pd.DataFrame) -> list[str]:
    """
    Reproducible, pre-specified selection:

    1. Keep the longest-recording patient in development so that the
       unusually long HUPA0027P record does not dominate the final test.
    2. Select five final-test patients using fixed duration strata:
       - shortest-duration patient
       - second-shortest-duration patient
       - median-duration patient
       - first patient at/above the 75th duration percentile
       - longest patient EXCLUDING the single longest patient

    Ties are resolved by patient ID.
    """

    s = summary.sort_values(["duration_days", "patient_id"])

    longest = s.index[-1]

    shortest = s.index[0]
    second_shortest = s.index[1]

    median_duration = s["duration_days"].median()
    median_candidates = s.copy()
    median_candidates["distance"] = (
        median_candidates["duration_days"] - median_duration
    ).abs()
    median_patient = (
        median_candidates.sort_values(["distance", "patient_id"]).index[0]
    )

    q75 = s["duration_days"].quantile(0.75)
    q75_candidates = s[s["duration_days"] >= q75]
    q75_patient = q75_candidates.index[0]

    remaining = s.drop(index=[longest])
    longest_remaining = remaining.index[-1]

    selected = [
        shortest,
        second_shortest,
        median_patient,
        q75_patient,
        longest_remaining,
    ]

    # Remove accidental duplicates, then deterministically fill if necessary.
    selected = list(dict.fromkeys(selected))

    for patient in s.index:
        if len(selected) >= 5:
            break
        if patient not in selected and patient != longest:
            selected.append(patient)

    return selected[:5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/hupa_features.csv",
    )
    parser.add_argument(
        "--output",
        default="data/processed/hupa_patient_split.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["time"])

    summary = make_patient_summary(df)

    if len(summary) != 25:
        raise ValueError(
            f"Expected 25 patients, found {len(summary)}."
        )

    test_patients = choose_test_patients(summary)
    development_patients = [
        p for p in summary.index if p not in test_patients
    ]

    if set(test_patients) & set(development_patients):
        raise RuntimeError("Patient overlap detected.")

    if len(test_patients) != 5:
        raise RuntimeError(
            f"Expected 5 test patients, got {len(test_patients)}."
        )

    manifest = summary.copy()

    manifest["split"] = [
    "test" if p in test_patients else "development"
    for p in manifest.index
    ]

    manifest = manifest.reset_index()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print("Patient split created.")
    print()
    print("Development patients:", len(development_patients))
    print("Test patients:", len(test_patients))
    print()
    print("Test patients:")
    for p in test_patients:
        print(f"  {p}")
    print()
    print("Full manifest:")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()