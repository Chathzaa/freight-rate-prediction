"""Validation protocol.

The real task predicts November and December from a model trained on January
through October, so every held-out fold here is a contiguous block of future
dates — a random split would leak the market level of the very days being
predicted and report an accuracy the submission cannot reproduce.  The last
fold reproduces the submission's two-month horizon exactly.

Two sets of numbers are reported per fold.  Roughly 1.3% of rows carry a
corrupted ``posted_rate`` that no model can recover, and they dominate squared
error; the "clean" columns exclude rows whose actual is more than 1.8x or less
than 0.6x the prediction, which isolates how the model does on the rows that
are actually predictable.  Both are shown because the clean figures are the
ones that measure modelling, and the raw figures are the ones the scorer sees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import MarketIndexTable
from .model import ModelConfig, RateModel

# Folds are (train_end, test_start, test_end).  Test blocks sit strictly after
# their training data, with the same 1-2 month horizon as the submission.
FOLDS = [
    ("2025-07-01", "2025-07-01", "2025-08-31"),
    ("2025-08-01", "2025-08-01", "2025-09-30"),
    ("2025-09-01", "2025-09-01", "2025-10-31"),
]

CORRUPT_HIGH = 1.8
CORRUPT_LOW = 0.6


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    ratio = actual / predicted
    clean = (ratio < CORRUPT_HIGH) & (ratio > CORRUPT_LOW)
    err = np.abs(actual - predicted)
    return {
        "n": len(actual),
        "n_clean": int(clean.sum()),
        "MAE": err.mean(),
        "RMSE": float(np.sqrt(((actual - predicted) ** 2).mean())),
        "MAPE_%": float(np.mean(err / actual) * 100),
        "MAE_clean": float(err[clean].mean()),
        "RMSE_clean": float(np.sqrt(((actual - predicted) ** 2)[clean].mean())),
        "MAPE_clean_%": float(np.mean(err[clean] / actual[clean]) * 100),
        "median_ratio": float(np.median(ratio)),
    }


def temporal_cv(train: pd.DataFrame, config: ModelConfig | None = None) -> pd.DataFrame:
    rows = []
    for train_end, test_start, test_end in FOLDS:
        fit_part = train[train["date"] < train_end]
        test_part = train[(train["date"] >= test_start) & (train["date"] <= test_end)]
        # The market table is built from the same information the model would
        # have at prediction time: history plus the covariates of the rows being
        # scored.  It never touches posted_rate.
        market = MarketIndexTable(fit_part, test_part)
        model = RateModel(config or ModelConfig()).fit(fit_part, market)
        predicted = model.predict(test_part)
        row = {"fold": f"train<{train_end} -> {test_start}..{test_end}"}
        row.update(metrics(test_part["posted_rate"].to_numpy(), predicted))
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("fold")
    frame.loc["mean"] = frame.mean(numeric_only=True)
    return frame


def holdout_reference(train: pd.DataFrame, config: ModelConfig | None = None) -> dict:
    """Random split, reported only as a noise floor.

    This is not a validation estimate — it leaks same-day market information —
    but the gap between it and the temporal folds is what the two-month forecast
    horizon actually costs.
    """
    rng = np.random.default_rng(0)
    mask = rng.random(len(train)) < 0.8
    fit_part, test_part = train[mask], train[~mask]
    market = MarketIndexTable(fit_part, test_part)
    model = RateModel(config or ModelConfig()).fit(fit_part, market)
    return metrics(test_part["posted_rate"].to_numpy(), model.predict(test_part))
