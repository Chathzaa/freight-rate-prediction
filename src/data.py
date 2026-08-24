"""Loading and cleaning for the freight rate data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

TRAIN = DATA / "train_test.csv"
VALIDATION = DATA / "validation.csv"
TEMPLATE = DATA / "validation_predictions_template.csv"
DECEMBER = DATA / "december_chart_inputs.csv"

EARTH_MILES = 3958.8


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_MILES * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data-quality problems that are recoverable.

    weight arrives with a sign flip on ~0.6% of rows.  The absolute values of the
    negative entries have the same distribution as the positive ones (same min,
    max and quartiles), so the sign is corruption rather than signal and abs()
    recovers the true value.  Everything else that is wrong with the data is
    either unrecoverable (the corrupted posted_rate rows, handled by robust
    fitting during training) or simply missing (handled at feature time).
    """
    out = df.copy()
    out["weight"] = out["weight"].abs()
    return out


def load_train() -> pd.DataFrame:
    return clean(_read(TRAIN))


def load_validation() -> pd.DataFrame:
    return clean(_read(VALIDATION))


def load_template() -> pd.DataFrame:
    return pd.read_csv(TEMPLATE)


def load_december() -> pd.DataFrame:
    return pd.read_csv(DECEMBER, parse_dates=["date"])


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One lat/lon per city, pooled over every frame that carries coordinates.

    Pickup and delivery coordinates agree for every city, and each city maps to
    exactly one pair, so this is a lookup table rather than an aggregation.
    Validation introduces eight cities that never appear in training, which is
    why the table is built from both files.
    """
    parts = []
    for df in frames:
        if {"pickup", "pickup_lat", "pickup_lon"} <= set(df.columns):
            parts.append(
                df[["pickup", "pickup_lat", "pickup_lon"]].rename(
                    columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
                )
            )
        if {"delivery", "delivery_lat", "delivery_lon"} <= set(df.columns):
            parts.append(
                df[["delivery", "delivery_lat", "delivery_lon"]].rename(
                    columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
                )
            )
    table = pd.concat(parts, ignore_index=True).drop_duplicates("city")
    return table.set_index("city")


def attach_coordinates(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Fill pickup/delivery lat/lon from the lookup table.

    The December chart inputs ship without coordinates, so they are recovered
    from the city names here.
    """
    out = df.copy()
    for side, prefix in (("pickup", "pickup"), ("delivery", "delivery")):
        for axis in ("lat", "lon"):
            col = f"{prefix}_{axis}"
            mapped = out[side].map(coords[axis])
            out[col] = mapped if col not in out.columns else out[col].fillna(mapped)
    return out
