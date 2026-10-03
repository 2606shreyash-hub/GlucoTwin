from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "time",
    "glucose",
    "calories",
    "heart_rate",
    "steps",
    "basal_rate",
    "bolus_volume_delivered",
    "carb_input",
]

def find_patient_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {root}")
    return files

def load_patient_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=";")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns: {missing}")

    df = df[EXPECTED_COLUMNS].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    numeric_cols = [c for c in EXPECTED_COLUMNS if c != "time"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)
    df["patient_id"] = path.stem

    return df

def load_all_patients(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    patients = {}
    for path in find_patient_files(data_dir):
        patients[path.stem] = load_patient_csv(path)
    return patients
