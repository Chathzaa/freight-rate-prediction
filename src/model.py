"""The rate model: a smooth linear stage plus a gradient-boosted correction.

Why two stages rather than one booster.  Validation is two months past the end
of training, and the rate level drifts upward at a steady ~0.019% per day.  A
tree cannot extrapolate that drift: beyond the last split it repeats its final
value, which would leave every November and December prediction short.  A
linear term in the time index does extrapolate it.  So the linear stage carries
the parts of the signal that are smooth and need to run past the training
window (distance, weight, market level, latitude, the time trend and the
quarter-end ramp), and the booster is fitted to whatever the linear stage leaves
behind, using only features that stay inside their training range.

Robustness.  About 1.3% of training rows have a corrupted ``posted_rate``,
scattered symmetrically above and below the true value (median ratios of about
3.3x and 0.29x).  They are unrecoverable, so the linear stage is fitted by
iteratively reweighted least squares that trims rows more than 4 robust
standard deviations out, and the booster is fitted on the surviving rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FeatureBuilder, LINEAR_COLUMNS, MarketIndexTable, log_rate_per_mile


@dataclass
class ModelConfig:
    max_iter: int = 500
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    trim_sigmas: float = 4.0
    trim_rounds: int = 3
    random_state: int = 0


@dataclass
class RateModel:
    config: ModelConfig = field(default_factory=ModelConfig)

    def fit(self, train: pd.DataFrame, market: MarketIndexTable) -> "RateModel":
        self.builder = FeatureBuilder(train, market)
        matrix = self.builder.build(train)
        target = log_rate_per_mile(train)

        design = self._design(matrix)
        keep = np.ones(len(train), dtype=bool)
        for _ in range(self.config.trim_rounds):
            beta, *_ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
            residual = target - design @ beta
            centre = np.median(residual[keep])
            scale = 1.4826 * np.median(np.abs(residual[keep] - centre))
            keep = np.abs(residual - centre) < self.config.trim_sigmas * scale

        self.beta = beta
        self.kept = keep
        self.n_trimmed = int((~keep).sum())

        linear_fit = design @ beta
        self.booster_cols = FeatureBuilder.booster_columns(matrix)
        self.booster = HistGradientBoostingRegressor(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            early_stopping=False,
            random_state=self.config.random_state,
        )
        self.booster.fit(matrix.loc[keep, self.booster_cols], (target - linear_fit)[keep])

        fitted = linear_fit + self.booster.predict(matrix[self.booster_cols])
        self.residual_sd = float(np.std((target - fitted)[keep]))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self.builder.build(df)
        log_rpm = self._design(matrix) @ self.beta
        log_rpm = log_rpm + self.booster.predict(matrix[self.booster_cols])
        return np.exp(log_rpm) * df["distance"].clip(lower=1.0).to_numpy()

    @staticmethod
    def _design(matrix: pd.DataFrame) -> np.ndarray:
        block = matrix[LINEAR_COLUMNS].to_numpy(dtype=float)
        return np.column_stack([block, np.ones(len(matrix))])

    def linear_coefficients(self) -> pd.Series:
        return pd.Series(self.beta[:-1], index=LINEAR_COLUMNS)
