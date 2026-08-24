"""Figures for the written report.

    python -m src.figures

These are print figures for a PDF, so they commit to the light surface rather
than carrying a dark variant.  Palette slots are the validated categorical
order (blue / orange / aqua); every multi-series panel is direct-labelled as
well as legended, so identity never rests on colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data
from .features import ORIGIN, QUARTER_KNOTS

FIGURES = Path(__file__).resolve().parents[1] / "outputs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e3e2dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SPOTTER = "#064A56"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def _caption(ax, text, y=-0.30, width=105):
    """Footnote under an axis, wrapped so it never widens the saved bounding box."""
    import textwrap
    ax.text(0.0, y, "\n".join(textwrap.wrap(text, width)), transform=ax.transAxes,
            fontsize=8, color=INK_2, va="top")


def _style(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, loc="left", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def daily_rate_index(train: pd.DataFrame) -> pd.Series:
    """Mean log rate-per-mile per day, net of load characteristics.

    Fits load-level effects and a free per-day effect by alternating least
    squares, so the returned series is the market level on each date with the
    mix of loads that day divided out.
    """
    df = train.copy()
    df["lrpm"] = np.log(df["posted_rate"] / df["distance"])
    lo, hi = df["lrpm"].quantile([0.005, 0.995])
    df = df[(df["lrpm"] > lo) & (df["lrpm"] < hi)].dropna(subset=["weight"])
    design = pd.DataFrame({
        "ldist": np.log(df["distance"]), "ldist2": np.log(df["distance"]) ** 2,
        "lw": np.log(df["weight"]), "plat": df["pickup_lat"], "dlat": df["delivery_lat"],
        "reefer": (df["equipment"] == "Reefer").astype(float),
        "flatbed": (df["equipment"] == "Flatbed").astype(float), "const": 1.0,
    }).to_numpy()
    y = df["lrpm"].to_numpy()
    effect = pd.Series(0.0, index=sorted(df["date"].unique()))
    for _ in range(30):
        beta, *_ = np.linalg.lstsq(design, y - df["date"].map(effect).to_numpy(), rcond=None)
        effect = pd.Series(y - design @ beta, index=df["date"]).groupby(level=0).mean()
        effect -= effect.mean()
    return effect


def fig_market_level(effect: pd.Series, train: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=200)
    smooth = effect.rolling(7, center=True, min_periods=3).mean()
    ax.plot(effect.index, effect.to_numpy(), color=MUTED, linewidth=0.7, alpha=0.65)
    ax.plot(smooth.index, smooth.to_numpy(), color=SERIES[0], linewidth=2.0)
    for q in ("2025-04-01", "2025-07-01", "2025-10-01"):
        ax.axvline(pd.Timestamp(q), color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.7)
    ax.text(pd.Timestamp("2025-03-20"), effect.max() * 0.98, "quarter end",
            ha="right", fontsize=8, color=INK_2)
    _style(ax, "Market rate level by day, load mix removed",
           "log rate per mile (centred)")
    _caption(ax, "Thin line: daily estimate.  Heavy line: 7-day mean.  Dashed rules mark "
             "quarter boundaries — the level climbs through each quarter and resets.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "fig1_market_level.png", bbox_inches="tight")
    plt.close(fig)


def fig_quarter_ramp(effect: pd.Series) -> None:
    idx = pd.DatetimeIndex(effect.index)
    frame = pd.DataFrame({
        "effect": effect.to_numpy(),
        "quarter": idx.quarter,
        "to_end": (idx.to_period("Q").end_time.normalize() - idx).days,
        "t": (idx - ORIGIN).days,
    })
    slope = np.polyfit(frame["t"], frame["effect"], 1)[0]
    frame["detrended"] = frame["effect"] - slope * frame["t"]

    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=200)
    for i, q in enumerate((1, 2, 3)):
        part = frame[frame["quarter"] == q].sort_values("to_end", ascending=False)
        # Baseline each quarter on its own flat stretch (90 to 35 days out) so the
        # three ramps are compared on shape rather than on level.
        flat = part.loc[part["to_end"] >= 35, "detrended"].mean()
        roll = (part["detrended"] - flat).rolling(5, center=True, min_periods=2).mean()
        ax.plot(part["to_end"], roll, color=SERIES[i], linewidth=2.0)
        ax.text(-1.5, roll.iloc[-1], f"Q{q}", color=SERIES[i], fontsize=9.5,
                fontweight="bold", va="center", ha="left")
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.axvline(30, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.7)
    ax.set_xlim(93, -7)
    _style(ax, "The same ramp appears in every quarter",
           "level vs. that quarter's baseline", "days until quarter end")
    ax.text(29, ax.get_ylim()[1] * 0.95, "ramp begins ~30 days out",
            fontsize=8, color=INK_2, ha="left", va="top")
    _caption(ax, "Each quarter's daily level after removing the linear trend, aligned on "
             "days-to-quarter-end, baselined on its own flat stretch and smoothed over 5 days. "
             "Rates are flat until roughly 30 days out, then climb about 5%.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "fig2_quarter_ramp.png", bbox_inches="tight")
    plt.close(fig)


def fig_data_quality(train: pd.DataFrame, raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), dpi=200)

    ax = axes[0]
    neg = raw.loc[raw["weight"] < 0, "weight"].abs()
    pos = raw.loc[raw["weight"] > 0, "weight"]
    bins = np.linspace(5000, 47500, 36)
    ax.hist(pos, bins=bins, color=GRID, edgecolor=MUTED, linewidth=0.4,
            weights=np.full(len(pos), 1 / len(pos)), label="positive (n=47,408)")
    ax.hist(neg, bins=bins, histtype="step", color=SERIES[1], linewidth=2.0,
            weights=np.full(len(neg), 1 / len(neg)), label="negative, sign flipped (n=292)")
    _style(ax, "Negative weights are a sign flip", "share of rows", "weight (lb)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left")

    ax = axes[1]
    rpm = raw["posted_rate"] / raw["distance"]
    ax.hist(np.log10(rpm), bins=90, color=GRID, edgecolor=MUTED, linewidth=0.4)
    ax.axvline(np.log10(rpm.median() * 0.6), color=SERIES[1], linewidth=1.8,
               linestyle=(0, (4, 3)))
    ax.axvline(np.log10(rpm.median() * 1.8), color=SERIES[1], linewidth=1.8,
               linestyle=(0, (4, 3)))
    ax.set_yscale("log")
    _style(ax, "Corrupted rates sit in two tails", "rows (log scale)",
           "log10 rate per mile")
    ax.set_ylim(0.7, 3e4)
    ax.text(0.5, 0.97, "~1.3% of rows fall outside the dashed bounds",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK_2)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_data_quality.png", bbox_inches="tight")
    plt.close(fig)


def fig_market_index(effect: pd.Series, train: pd.DataFrame) -> None:
    daily_mi = train.groupby("date")["market_index"].mean()
    frame = pd.DataFrame({"effect": effect, "mi": daily_mi}).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), dpi=200)

    ax = axes[0]
    ax.scatter(frame["mi"], frame["effect"], s=14, color=SERIES[0], alpha=0.55,
               edgecolors="none")
    b = np.polyfit(frame["mi"], frame["effect"], 1)
    xs = np.linspace(frame["mi"].min(), frame["mi"].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=INK_2, linewidth=1.6)
    r = np.corrcoef(frame["mi"], frame["effect"])[0, 1]
    _style(ax, "Daily market index tracks the rate level",
           "daily rate level", "daily mean market_index")
    ax.text(0.04, 0.90, f"r = {r:.2f}", transform=ax.transAxes, fontsize=9, color=INK)

    ax = axes[1]
    dow = pd.DataFrame({
        "mi": frame["mi"], "effect": frame["effect"],
        "dow": pd.DatetimeIndex(frame.index).dayofweek}).groupby("dow").mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.bar(np.arange(7), dow["effect"], color=SERIES[0], width=0.62)
    ax.set_xticks(np.arange(7), names)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    _style(ax, "Rates follow a weekly cycle", "deviation in log rate per mile")
    _caption(ax, "Peak Thursday, trough Sunday — about a 2.1% swing, carried by "
             "market_index rather than by the weekday itself.", width=58)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "fig4_market_index.png", bbox_inches="tight")
    plt.close(fig)


def fig_holdout(train: pd.DataFrame) -> None:
    from .features import MarketIndexTable
    from .model import ModelConfig, RateModel

    fit = train[train["date"] < "2025-09-01"]
    test = train[train["date"] >= "2025-09-01"]
    model = RateModel(ModelConfig()).fit(fit, MarketIndexTable(fit, test))
    pred = model.predict(test)
    actual = test["posted_rate"].to_numpy()
    ratio = actual / pred
    clean = (ratio > 0.6) & (ratio < 1.8)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), dpi=200)
    ax = axes[0]
    ax.scatter(pred[clean], actual[clean], s=5, color=SERIES[0], alpha=0.25, edgecolors="none")
    ax.scatter(pred[~clean], actual[~clean], s=11, color=SERIES[1], alpha=0.8,
               edgecolors="none", label="corrupted actuals")
    lim = [0, max(pred.max(), 9000)]
    ax.plot(lim, lim, color=INK_2, linewidth=1.2)
    ax.set_xlim(0, 8000); ax.set_ylim(0, 9000)
    _style(ax, "Held-out Sep-Oct: predicted vs actual", "actual rate ($)", "predicted rate ($)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower right")

    ax = axes[1]
    ax.hist(np.log(ratio[clean]) * 100, bins=70, color=SERIES[0])
    ax.axvline(0, color=INK_2, linewidth=1.2)
    _style(ax, "Error distribution on predictable rows", "rows", "log error (%)")
    ax.text(0.03, 0.90, f"MAPE {np.mean(np.abs(actual - pred)[clean] / actual[clean]) * 100:.2f}%\n"
                        f"median ratio {np.median(ratio):.3f}",
            transform=ax.transAxes, fontsize=8.5, color=INK, va="top")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_holdout.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data.TRAIN, parse_dates=["date"])
    train = data.load_train()
    effect = daily_rate_index(train)
    fig_market_level(effect, train)
    fig_quarter_ramp(effect)
    fig_data_quality(train, raw)
    fig_market_index(effect, train)
    fig_holdout(train)
    for p in sorted(FIGURES.glob("fig*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
