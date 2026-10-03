from pathlib import Path
import pandas as pd


INPUT = "data/processed/hupa_patient_split.csv"
OUTPUT = "data/processed/hupa_cv_folds.csv"


def main():
    manifest = pd.read_csv(INPUT)

    development = manifest[
        manifest["split"] == "development"
    ].copy()

    if len(development) != 20:
        raise ValueError(
            f"Expected 20 development patients, found {len(development)}"
        )

    # Target totals across five equally sized folds.
    target_rows = development["rows"].sum() / 5
    target_positive = development["positive"].sum() / 5

    # Process the largest patients first so they are distributed
    # across different folds.
    development = development.sort_values(
        ["rows", "positive", "patient_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    folds = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }

    fold_rows = {k: 0.0 for k in folds}
    fold_positive = {k: 0.0 for k in folds}

    for _, row in development.iterrows():

        available = [
            fold for fold in folds
            if len(folds[fold]) < 4
        ]

        # Normalize row and positive-event burden so both contribute
        # to the balancing decision.
        def score(fold):
            row_ratio = (
                fold_rows[fold] / target_rows
                if target_rows > 0 else 0
            )

            positive_ratio = (
                fold_positive[fold] / target_positive
                if target_positive > 0 else 0
            )

            return (
                row_ratio + positive_ratio,
                fold_rows[fold],
                fold,
            )

        chosen = min(available, key=score)

        folds[chosen].append(row["patient_id"])
        fold_rows[chosen] += row["rows"]
        fold_positive[chosen] += row["positive"]

    records = []

    for fold, patients in folds.items():
        if len(patients) != 4:
            raise RuntimeError(
                f"Fold {fold} has {len(patients)} patients instead of 4."
            )

        for patient_id in patients:
            records.append({
                "patient_id": patient_id,
                "fold": fold,
            })

    # Final test patients get fold 0.
    test_patients = manifest.loc[
        manifest["split"] == "test",
        "patient_id",
    ].tolist()

    for patient_id in test_patients:
        records.append({
            "patient_id": patient_id,
            "fold": 0,
        })

    result = pd.DataFrame(records)

    # Verify exactly one assignment per patient.
    if result["patient_id"].duplicated().any():
        raise RuntimeError("A patient appears more than once.")

    if len(result) != 25:
        raise RuntimeError(
            f"Expected 25 patients, found {len(result)}."
        )

    # Add patient statistics.
    stats = manifest[
        [
            "patient_id",
            "rows",
            "positive",
            "prevalence",
            "duration_days",
        ]
    ]

    result = result.merge(
        stats,
        on="patient_id",
        how="left",
        validate="one_to_one",
    )

    result = result.sort_values(
        ["fold", "patient_id"]
    ).reset_index(drop=True)

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("5-fold development split created.")
    print()

    for fold in range(1, 6):
        subset = result[result["fold"] == fold]

        print(f"Fold {fold}:")
        print(f"  Patients: {len(subset)}")
        print(f"  Rows: {int(subset['rows'].sum()):,}")
        print(f"  Positive windows: {int(subset['positive'].sum()):,}")
        print(
            f"  Positive prevalence: "
            f"{subset['positive'].sum() / subset['rows'].sum():.4%}"
        )
        print("  Patients:")
        for patient in subset["patient_id"]:
            print(f"    {patient}")
        print()

    print("Final test patients:")
    for patient in test_patients:
        print(f"  {patient}")

    print()
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()