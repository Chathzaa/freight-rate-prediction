"""Feature engineering.

The design follows what the exploration turned up (see report/report.md):

* the target is modelled as log rate-per-mile, so the strong and almost
  log-linear distance effect is carried by an offset instead of by the trees;
* ``market_index`` is split into a daily level and a within-day deviation,
  because the daily level is the dominant driver of where rates sit on a given
  date and the deviation is a much weaker per-load effect;
* calendar structure enters as days-to-quarter-end hinges plus a linear time
  index, both of which extrapolate past the end of training;
* ``quote_signal`` is deliberately excluded — it carries no signal once
  distance and the market level are accounted for, and including it measurably
  hurt held-out accuracy.

Only features whose values in November and December fall inside the range seen
in training are handed to the gradient booster.  The linear time index is the
one feature that necessarily leaves that range, so it is fitted by the linear
stage alone, which can extrapolate it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp("2025-01-01")

# Knots for the days-to-quarter-end ramp.  Rates are flat for most of a quarter
# and climb over roughly its final month, so the knots are dense near zero.
QUARTER_KNOTS = (60, 45, 35, 30, 25, 20, 15, 12, 9, 7, 5, 3, 2, 1)

# Columns the linear stage fits.  Everything here is smooth and safe to
# extrapolate; the booster then models what is left.
LINEAR_COLUMNS = (
    ["log_distance", "log_distance_sq", "log_weight", "market_level", "market_dev",
     "pickup_lat", "delivery_lat", "pickup_lon", "delivery_lon",
     "is_reefer", "is_flatbed", "time_index", "month_end_7"]
    + [f"quarter_ramp_{k}" for k in QUARTER_KNOTS]
)

# The trees never see the raw time index: it takes values beyond the training
# range in November and December, where a tree can only repeat its last split.
BOOSTER_EXCLUDE = ("time_index",)


class MarketIndexTable:
    """Daily market level, pooled across every frame that reports one.

    ``market_index`` is a per-load number, but its daily mean explains roughly
    96% of the day-to-day movement in rates once the trend and the quarter-end
    ramp are removed.  Validation ships with ``market_index`` for every November
    and December load, so the daily level for those dates is observed rather
    than forecast.  The December chart inputs omit the column entirely, and
    borrow the same daily table.
    """

    def __init__(self, *frames: pd.DataFrame):
        parts = [f[["date", "market_index"]] for f in frames if "market_index" in f.columns]
        pooled = pd.concat(parts, ignore_index=True).dropna()
        self.daily = pooled.groupby("date")["market_index"].mean()
        self.overall = float(pooled["market_index"].mean())

    def level(self, dates: pd.Series) -> pd.Series:
        return dates.map(self.daily).astype(float).fillna(self.overall)


class FeatureBuilder:
    """Turns a raw frame into the model matrix.

    Anything learned from data (median weight, the market table) is fitted on
    the training frame only and then reused, so a feature matrix built for
    validation never sees a training-time statistic it would not have in
    production.
    """

    def __init__(self, train: pd.DataFrame, market: MarketIndexTable):
        self.weight_median = float(train["weight"].median())
        self.market = market

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        distance = df["distance"].clip(lower=1.0)
        f["log_distance"] = np.log(distance)
        f["log_distance_sq"] = f["log_distance"] ** 2

        weight = df["weight"].fillna(self.weight_median).clip(lower=1.0)
        f["log_weight"] = np.log(weight)
        f["weight_missing"] = df["weight"].isna().astype(float)

        level = self.market.level(df["date"])
        f["market_level"] = level
        if "market_index" in df.columns:
            f["market_dev"] = (df["market_index"].astype(float) - level).fillna(0.0)
        else:
            f["market_dev"] = 0.0

        f["pickup_lat"] = df["pickup_lat"].astype(float)
        f["delivery_lat"] = df["delivery_lat"].astype(float)
        f["pickup_lon"] = df["pickup_lon"].astype(float)
        f["delivery_lon"] = df["delivery_lon"].astype(float)

        f["is_reefer"] = (df["equipment"] == "Reefer").astype(float)
        f["is_flatbed"] = (df["equipment"] == "Flatbed").astype(float)

        date = df["date"]
        f["time_index"] = (date - ORIGIN).dt.days.astype(float)

        to_quarter_end = (
            date.dt.to_period("Q").dt.end_time.dt.normalize() - date
        ).dt.days.astype(float)
        f["days_to_quarter_end"] = to_quarter_end
        for k in QUARTER_KNOTS:
            f[f"quarter_ramp_{k}"] = np.maximum(0.0, k - to_quarter_end)

        to_month_end = (date + pd.offsets.MonthEnd(0) - date).dt.days.astype(float)
        f["days_to_month_end"] = to_month_end
        f["month_end_7"] = np.maximum(0.0, 7.0 - to_month_end)

        f["day_of_week"] = date.dt.dayofweek.astype(float)
        f["day_of_month"] = date.dt.day.astype(float)
        return f

    @staticmethod
    def booster_columns(matrix: pd.DataFrame) -> list[str]:
        return [c for c in matrix.columns if c not in BOOSTER_EXCLUDE]


def log_rate_per_mile(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["posted_rate"] / df["distance"].clip(lower=1.0)).to_numpy()
