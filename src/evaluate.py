"""Validation protocol.

The real task predicts November and December from a model trained on January
through October, so every held-out fold here is a contiguous block of future
dates. A random split would leak the market level of the very days being
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


def booster_only(fit: pd.DataFrame, test: pd.DataFrame, market: MarketIndexTable,
                 use_time_index: bool) -> np.ndarray:
    """A single-stage booster, for the comparison that justifies the two-stage split."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from .features import FeatureBuilder, log_rate_per_mile

    builder = FeatureBuilder(fit, market)
    matrix, matrix_test = builder.build(fit), builder.build(test)
    target = log_rate_per_mile(fit)
    columns = list(matrix.columns)
    if not use_time_index:
        columns = [c for c in columns if c != "time_index"]

    centre = np.median(target)
    keep = np.abs(target - centre) < 4.0 * 1.4826 * np.median(np.abs(target - centre))
    booster = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0)
    booster.fit(matrix.loc[keep, columns], target[keep])
    return np.exp(booster.predict(matrix_test[columns])) * test["distance"].to_numpy()


def ablations(train: pd.DataFrame) -> pd.DataFrame:
    """Compare the submitted model against single-stage boosters.

    Also reports the median actual/predicted ratio on each fold's furthest-out
    month, which is where a model that cannot extrapolate the trend gives itself
    away: a ratio above 1 means it under-predicted.
    """
    rows = []
    for train_end, test_start, test_end in FOLDS:
        fit = train[train["date"] < train_end]
        test = train[(train["date"] >= test_start) & (train["date"] <= test_end)]
        market = MarketIndexTable(fit, test)
        actual = test["posted_rate"].to_numpy()
        far = (test["date"] >= pd.Timestamp(test_end) - pd.offsets.MonthBegin(1)).to_numpy()

        variants = {
            "two-stage (submitted)": RateModel(ModelConfig()).fit(fit, market).predict(test),
            "booster only, with time index": booster_only(fit, test, market, True),
            "booster only, without time index": booster_only(fit, test, market, False),
        }
        for name, predicted in variants.items():
            scored = metrics(actual, predicted)
            ratio = actual[far] / predicted[far]
            ratio = ratio[(ratio > CORRUPT_LOW) & (ratio < CORRUPT_HIGH)]
            rows.append({"variant": name, "MAPE_clean_%": scored["MAPE_clean_%"],
                         "MAE_clean": scored["MAE_clean"],
                         "far_month_median_ratio": float(np.median(ratio))})
    return pd.DataFrame(rows).groupby("variant", sort=False).mean()


def holdout_reference(train: pd.DataFrame, config: ModelConfig | None = None) -> dict:
    """Random split, reported only as a noise floor.

    This is not a validation estimate, since it leaks same-day market
    information, but the gap between it and the temporal folds is what the
    two-month forecast horizon actually costs.
    """
    rng = np.random.default_rng(0)
    mask = rng.random(len(train)) < 0.8
    fit_part, test_part = train[mask], train[~mask]
    market = MarketIndexTable(fit_part, test_part)
    model = RateModel(config or ModelConfig()).fit(fit_part, market)
    return metrics(test_part["posted_rate"].to_numpy(), model.predict(test_part))
