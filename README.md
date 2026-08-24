# Freight Rate Prediction

Solution for the Spotter Machine Learning Engineer assessment: predict `posted_rate`
for 12,000 held-out freight loads, and produce the fixed December 2025 chart.

Full write-up: [`report/freight_rate_report.pdf`](report/freight_rate_report.pdf)
(source: [`report/report.md`](report/report.md)).

## Result

Forward-chaining temporal validation, averaged over three folds:

| Metric | All rows | Predictable rows only |
|---|---|---|
| MAE | $89.82 | $36.82 |
| RMSE | $619.86 | $53.05 |
| MAPE | 3.95% | 1.58% |

About 1.3% of rows carry a corrupted `posted_rate` (roughly 3.3x or 0.29x the true
value) that no model can recover. They dominate squared error, so both views are
reported. A random split, which leaks same-day market information and is therefore
only a noise floor, reaches 1.35% MAPE — the small gap to 1.58% is what the
two-month forecast horizon costs.

## Approach

The target is **log rate-per-mile**, not the dollar rate, because equipment, market
conditions and geography all act multiplicatively.

The model has two stages:

1. **Linear stage** over smooth features that must extrapolate past the end of
   training: log distance, log weight, the daily market level and the within-day
   deviation from it, coordinates, equipment, a linear time index, and a hinge basis
   on days-to-quarter-end.
2. **Gradient-boosted stage** (`HistGradientBoostingRegressor`) fitted to the first
   stage's residuals, over the same features minus the time index.

Validation sits one to two months past the end of training and the rate level drifts
upward throughout, so the split matters: a tree cannot extrapolate that drift, but a
linear time term can. The booster only sees features whose November and December
values fall inside their training range.

On the same folds, a single-stage booster with no time index under-predicts the
furthest-out month by 4.1% and scores 3.30% MAPE; given the time index it reaches
1.87%, still behind the two-stage model's 1.58%. `python -m src.run validate`
reproduces that comparison.

The key structural finding is that rates climb through each quarter and reset at the
quarter boundary, rising about 5% over a quarter's final month. December is a
quarter-end month, which is what shapes the required chart.

## Setup

Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

`reportlab` is needed only to rebuild the PDF report; everything else runs without it.

## Running

```bash
# temporal cross-validation + model-family comparison
# -> outputs/validation_metrics.csv, outputs/ablations.csv
python -m src.run validate

# fit on all 48,000 rows, write both submission files
python -m src.run predict

# both
python -m src.run all
```

`predict` writes:

- `outputs/validation_predictions.csv` — the 12,000 predictions, `load_id,predicted_rate`
- `outputs/december_chart_inputs.csv` — the chart inputs with `predicted_rate` filled

Then validate the outputs and render the chart with the provided scorer:

```bash
python score.py \
  --predictions outputs/validation_predictions.csv \
  --december-predictions outputs/december_chart_inputs.csv \
  --output-dir outputs/figures
```

To regenerate the report figures and the PDF:

```bash
python -m src.figures
python -m src.build_report
```

Everything runs locally on CPU. A full `validate` plus `predict` takes about two
minutes.

## Layout

```
data/            provided CSVs (train_test, validation, template, december inputs)
src/
  data.py        loading, cleaning, city-coordinate lookup
  features.py    feature engineering; which features are safe to extrapolate
  model.py       two-stage model with robust fitting
  evaluate.py    temporal cross-validation protocol and metrics
  run.py         CLI entry point
  figures.py     report figures
  build_report.py  renders report/report.md to PDF
outputs/         predictions, metrics, figures
report/          report source and PDF
score.py         provided scorer, unmodified
```

## Notes on the data

Handled explicitly (details and counts in the report):

- 292 rows with sign-flipped `weight`, recovered with `abs()`
- ~1.3% of rows with corrupted `posted_rate`, trimmed from fitting by robust regression
- missing `weight` (465 rows) and `market_index` (623 rows), imputed with a missingness flag
- 8 cities that appear only in validation (1,447 rows, 12.1%), handled by using
  coordinates rather than city identity
- `quote_signal` carries no signal once distance and market level are accounted for,
  and including it made held-out accuracy worse; it is deliberately excluded
