"""Entry point: validate the approach, then fit on everything and predict.

    python -m src.run validate     # temporal cross-validation table
    python -m src.run predict      # writes the two submission files
    python -m src.run all          # both
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .evaluate import holdout_reference, temporal_cv
from .features import MarketIndexTable
from .model import ModelConfig, RateModel

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def validate() -> None:
    train = data.load_train()
    table = temporal_cv(train)
    pd.set_option("display.width", 200)
    print("Temporal cross-validation (each fold predicts dates after its training data)\n")
    print(table.round(3).to_string())
    ref = holdout_reference(train)
    print(
        f"\nRandom-split reference (leaks same-day market level, not a validation estimate): "
        f"MAPE_clean={ref['MAPE_clean_%']:.3f}%  MAE_clean={ref['MAE_clean']:.2f}"
    )
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUTS / "validation_metrics.csv")
    print(f"\nWrote {OUTPUTS / 'validation_metrics.csv'}")


def predict() -> None:
    train = data.load_train()
    validation = data.load_validation()
    december = data.load_december()

    coords = data.city_coordinates(train, validation)
    december = data.attach_coordinates(december, coords)
    missing = december[["pickup_lat", "delivery_lat"]].isna().any(axis=1)
    if missing.any():
        raise SystemExit("December inputs reference a city with no known coordinates")

    # The December chart rows carry no market_index, so they take the daily
    # level observed for the same dates in validation.
    market = MarketIndexTable(train, validation)
    model = RateModel(ModelConfig()).fit(train, market)

    print(f"Fitted on {len(train):,} rows; {model.n_trimmed:,} trimmed as corrupted "
          f"({100 * model.n_trimmed / len(train):.2f}%). Residual sd = {model.residual_sd:.4f} log units.")
    print("\nLinear-stage coefficients (log rate per mile):")
    coef = model.linear_coefficients()
    print(coef[~coef.index.str.startswith("quarter_ramp")].round(5).to_string())

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    predicted = model.predict(validation)
    out = pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": np.round(predicted, 2)})
    template = data.load_template()
    out = template[["load_id"]].merge(out, on="load_id", how="left")
    if out["predicted_rate"].isna().any():
        raise SystemExit("some template load_ids received no prediction")
    path = OUTPUTS / "validation_predictions.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}  ({len(out):,} rows)")

    dec = data.load_december()
    dec_features = data.attach_coordinates(dec, coords)
    dec_features["market_index"] = np.nan
    dec["predicted_rate"] = np.round(model.predict(dec_features), 2)
    dec_path = OUTPUTS / "december_chart_inputs.csv"
    dec.to_csv(dec_path, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {dec_path}  ({len(dec)} rows)")
    print(f"  December range ${dec['predicted_rate'].min():,.2f} to ${dec['predicted_rate'].max():,.2f} "
          f"(+{100 * (dec['predicted_rate'].iloc[-1] / dec['predicted_rate'].iloc[0] - 1):.1f}% across the month)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "predict", "all"])
    args = parser.parse_args()
    if args.command in ("validate", "all"):
        validate()
    if args.command in ("predict", "all"):
        if args.command == "all":
            print("\n" + "=" * 70 + "\n")
        predict()


if __name__ == "__main__":
    main()
