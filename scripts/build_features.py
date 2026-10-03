"""Build the causal HUPA feature table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.feature_engineering import build_features
from src.labels.hypoglycemia import add_30min_hypoglycemia_label


def load_patient(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["patient_id"] = path.stem
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="HUPA Preprocessed directory")
    parser.add_argument("--output", default="data/processed/hupa_features.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    tables = []
    for path in paths:
        df = load_patient(path)
        df = df.sort_values("time").reset_index(drop=True)
        features = build_features(df)
        labels = add_30min_hypoglycemia_label(features[["time", "glucose"]])
        features["hypo_next_30m"] = labels["target_hypo_30m"].to_numpy()
        tables.append(features)

    result = pd.concat(tables, ignore_index=True)
    result = result.sort_values(["patient_id", "time"]).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result):,} rows to {output}")
    print(f"Patients: {result['patient_id'].nunique()}")
    print(f"Positive labels: {int(result['hypo_next_30m'].sum()):,}")


if __name__ == "__main__":
    main()
