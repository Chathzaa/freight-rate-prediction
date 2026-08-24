# Freight Rate Prediction

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
only a noise floor, reaches 1.35% MAPE. The small gap to 1.58% is what the
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

### Data

Drop the four files supplied with the assessment into `data/`,
keeping these names:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

Everything below then runs as written.

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

- `outputs/validation_predictions.csv`, the 12,000 predictions as `load_id,predicted_rate`
  (submitted through the application form rather than committed here)
- `outputs/december_chart_inputs.csv`, the chart inputs with `predicted_rate` filled

Then validate the outputs and render the chart with the provided scorer:

```bash
python score.py \
  --predictions outputs/validation_predictions.csv \
  --december-predictions outputs/december_chart_inputs.csv \
  --output-dir outputs/figures
```

To regenerate the report figures:

```bash
python -m src.figures
```


## Layout

```
data/            put the four provided CSVs here
src/
  data.py        loading, cleaning, city-coordinate lookup
  features.py    feature engineering; which features are safe to extrapolate
  model.py       two-stage model with robust fitting
  evaluate.py    temporal cross-validation protocol and metrics
  run.py         CLI entry point
  figures.py     report figures
outputs/         metrics, figures, December predictions
report/          the written report (PDF)
score.py         provided scorer
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
