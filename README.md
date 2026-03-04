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

## Outputs

| Path | Description |
|------|-------------|
| `outputs/metrics.json` | Test MAE, MSE, 90%/95% interval coverage |
| `outputs/best_model.json` | OLS parameters (intercept, coefficients, σ̂, k, feature names) |
| `outputs/predictions_test.csv` | Timestamp, y_true, y_pred, lower_90, upper_90, lower_95, upper_95 |
| `outputs/figures/timeseries_week.png` | 7-day window: true vs predicted with 90% band |
| `outputs/figures/parity_plot.png` | Scatter of true vs predicted |
| `outputs/figures/interval_coverage.png` | Bar plot of nominal vs empirical coverage |
