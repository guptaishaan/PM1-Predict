# PM1 Forecast (CS109)

Forecasts PM1 mass concentration 1 hour ahead using PM2.5 and PM10 from hourly air quality data.


## Install and Run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the training CSV at `data/extract_0.csv`, then:

```bash
python -m src.run_pipeline --data_path data/extract_0.csv
```

Optionally filter to calibrated-ready PM2.5 rows:

```bash
python -m src.run_pipeline --data_path data/extract_0.csv --use_calibrated_ready
```

Change output directory:

```bash
python -m src.run_pipeline --data_path data/extract_0.csv --output_dir outputs
```

### CS109 extensions (optional flags)

Baseline always runs. Extensions run only when flagged:

| Flag | Description |
|------|-------------|
| `--run_bootstrap` | Bootstrap training resamples; uncertainty for test MAE/MSE and coefficients |
| `--bootstrap_B 200` | Number of bootstrap replicates (default 200) |
| `--run_map` | MAP/Ridge vs OLS comparison; validation MAE selects `alpha` |
| `--run_distribution_fit` | Fit Normal/Lognormal/Gamma to hourly PM1, daily PM1, test residuals |

Example (all extensions):

```bash
python -m src.run_pipeline --data_path data/extract_0.csv --run_bootstrap --run_map --run_distribution_fit
```

## Outputs

### Baseline

| Path | Description |
|------|-------------|
| `outputs/metrics.json` | Test MAE, MSE, 90%/95% interval coverage |
| `outputs/best_model.json` | OLS parameters (`model_name`: `baseline_ols`), σ̂, k, feature names |
| `outputs/predictions_test.csv` | Timestamp, y_true, y_pred, lower_90, upper_90, lower_95, upper_95 |
| `outputs/figures/timeseries_week.png` | 7-day window: true vs predicted with 90% band |
| `outputs/figures/parity_plot.png` | Scatter of true vs predicted |
| `outputs/figures/interval_coverage.png` | Bar plot of nominal vs empirical coverage |

### Bootstrap (`--run_bootstrap`)

| Path | Description |
|------|-------------|
| `outputs/bootstrap_metrics.json` | Observed vs bootstrap MAE/MSE; coefficient means and 95% intervals |
| `outputs/bootstrap_coefficients.csv` | One row per bootstrap: intercept + coefficients |
| `outputs/figures/bootstrap_mae_hist.png` | Histogram of bootstrap test MAE |
| `outputs/figures/bootstrap_coef_intervals.png` | Coefficient 95% intervals |

### MAP/Ridge (`--run_map`)

| Path | Description |
|------|-------------|
| `outputs/map_vs_mle_comparison.csv` | baseline_ols vs map_ridge: k, alpha, val/test MAE/MSE, coverage |
| `outputs/best_ridge_model.json` | Ridge parameters and `alpha` |
| `outputs/figures/map_vs_mle_bar.png` | Test MAE bar comparison |
| `outputs/figures/ridge_coef_shrinkage.png` | OLS vs Ridge coefficients |

### Distribution fitting (`--run_distribution_fit`)

| Path | Description |
|------|-------------|
| `outputs/distribution_fits_hourly_pm1.csv` | Normal/Lognormal/Gamma fits + AIC/BIC |
| `outputs/distribution_fits_daily_pm1.csv` | Same for daily-average PM1 |
| `outputs/distribution_fits_residuals.csv` | Same for test residuals (Normal only if residuals ≤ 0 exist) |
| `outputs/figures/hourly_pm1_distribution_fit.png` | Histogram + overlaid densities |
| `outputs/figures/daily_pm1_distribution_fit.png` | Same |
| `outputs/figures/residual_distribution_fit.png` | Same |

### Additional diagnostic figures (always or extension-gated)

| Path | Concept |
|------|--------|
| `outputs/figures/residual_vs_fitted.png` | **Homoscedasticity** — residuals vs ŷ; funnel shape suggests non-constant variance |
| `outputs/figures/residual_qq_normal.png` | **Gaussian assumption** — Q–Q vs Normal for test residuals |
| `outputs/figures/bootstrap_mse_hist.png` | **Bootstrap** — sampling distribution of test MSE (with `--run_bootstrap`) |
| `outputs/figures/ridge_val_mae_vs_alpha.png` | **MAP/regularization path** — validation MAE vs α (log scale); chosen α marked |
| `outputs/figures/hourly_pm1_aic_comparison.png` | **Model comparison** — AIC bars for Normal/Lognormal/Gamma (`--run_distribution_fit`) |
| `outputs/figures/daily_pm1_aic_comparison.png` | Same for daily PM1 |
| `outputs/figures/residuals_aic_comparison.png` | Same for residuals (often single bar if Normal only) |

### Tests

```bash
cd PM1-Predict
PYTHONPATH=. python3 -m pytest tests/test_smoke.py -v
```
