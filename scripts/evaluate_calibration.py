from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


INPUT_PATH = "data/processed/random_forest_oof_predictions.csv"


def main():
    print("Loading Random Forest OOF predictions...")

    df = pd.read_csv(INPUT_PATH)

    y_true = df["hypo_next_30m"].to_numpy()
    raw_probability = df["probability"].to_numpy()

    print(f"Rows: {len(df):,}")
    print(f"Positive events: {y_true.sum():,}")
    print(f"Observed prevalence: {y_true.mean():.4f}")
    print()

    # ------------------------------------------------------------
    # 1. RAW RANDOM FOREST CALIBRATION
    # ------------------------------------------------------------

    raw_brier = brier_score_loss(
        y_true,
        raw_probability,
    )

    raw_log_loss = log_loss(
        y_true,
        raw_probability,
    )

    print("=" * 70)
    print("RAW RANDOM FOREST")
    print("=" * 70)

    print(f"Brier score: {raw_brier:.6f}")
    print(f"Log loss:    {raw_log_loss:.6f}")

    # ------------------------------------------------------------
    # 2. CALIBRATION CURVE
    # ------------------------------------------------------------

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        raw_probability,
        n_bins=10,
        strategy="quantile",
    )

    calibration_table = pd.DataFrame(
        {
            "mean_predicted_probability": mean_predicted,
            "observed_fraction_positive": fraction_positive,
        }
    )

    print()
    print("Calibration curve:")
    print(calibration_table.to_string(index=False))

    # ------------------------------------------------------------
    # 3. PLATT / SIGMOID CALIBRATION
    # ------------------------------------------------------------
    #
    # IMPORTANT:
    # The calibration model is fitted only on OOF development
    # predictions. The final test set is not touched.
    #
    # We use the logit of the RF probability as the input to
    # logistic regression.
    # ------------------------------------------------------------

    epsilon = 1e-6

    clipped_probability = np.clip(
        raw_probability,
        epsilon,
        1 - epsilon,
    )

    logit_probability = np.log(
        clipped_probability / (1 - clipped_probability)
    )

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(
        logit_probability.reshape(-1, 1),
        y_true,
    )

    calibrated_probability = calibrator.predict_proba(
        logit_probability.reshape(-1, 1)
    )[:, 1]

    calibrated_brier = brier_score_loss(
        y_true,
        calibrated_probability,
    )

    calibrated_log_loss = log_loss(
        y_true,
        calibrated_probability,
    )

    print()
    print("=" * 70)
    print("SIGMOID-CALIBRATED RANDOM FOREST")
    print("=" * 70)

    print(f"Brier score: {calibrated_brier:.6f}")
    print(f"Log loss:    {calibrated_log_loss:.6f}")

    # ------------------------------------------------------------
    # 4. CALIBRATED CURVE
    # ------------------------------------------------------------

    calibrated_fraction, calibrated_mean = calibration_curve(
        y_true,
        calibrated_probability,
        n_bins=10,
        strategy="quantile",
    )

    calibrated_table = pd.DataFrame(
        {
            "mean_predicted_probability": calibrated_mean,
            "observed_fraction_positive": calibrated_fraction,
        }
    )

    print()
    print("Calibrated curve:")
    print(calibrated_table.to_string(index=False))

    # ------------------------------------------------------------
    # 5. SAVE CALIBRATED OOF PREDICTIONS
    # ------------------------------------------------------------

    output = df.copy()

    output["calibrated_probability"] = calibrated_probability

    output_path = Path(
        "data/processed/random_forest_calibrated_oof.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Saved calibrated predictions to: {output_path}")

    # Save calibration parameters for later final-test use.
    calibration_parameters = pd.DataFrame(
        {
            "intercept": [calibrator.intercept_[0]],
            "coefficient": [calibrator.coef_[0, 0]],
        }
    )

    parameter_path = Path(
        "data/processed/calibration_parameters.csv"
    )

    calibration_parameters.to_csv(
        parameter_path,
        index=False,
    )

    print(
        f"Saved calibration parameters to: {parameter_path}"
    )


if __name__ == "__main__":
    main()