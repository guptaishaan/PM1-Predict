import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .features import make_supervised
from .io import load_data
from .metrics import coverage, mae, mse
from .models import fit_ols, predict_intervals, predict_mean, serialize_model
from .plots import plot_interval_coverage, plot_parity, plot_timeseries_week
from .split import time_block_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/extract_0.csv")
    parser.add_argument("--use_calibrated_ready", action="store_true")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    cfg = Config()
    out_dir = Path(args.output_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, diag = load_data(args.data_path)
    print(f"Loaded: {diag['n_rows_after_dedup']} rows, duplicates dropped: {diag['n_duplicates_dropped']}")

    if args.use_calibrated_ready:
        before = len(df)
        df = df[df[cfg.col_pm25_status] == cfg.filter_value].reset_index(drop=True)
        print(f"Filtered to calibrated-ready: {len(df)} rows (removed {before - len(df)})")

    best_k = None
    best_val_mae = float("inf")
    best_result = None

    for k in cfg.candidate_ks:
        try:
            X, y, timestamps, feat_cols = make_supervised(
                df, k, cfg.horizon,
                cfg.col_target, cfg.col_pm25, cfg.col_pm10, cfg.col_time,
            )
        except Exception as e:
            print(f"k={k}: skipped - {e}")
            continue

        n = len(y)
        if n < 100:
            print(f"k={k}: skipped - too few rows ({n}) after dropping NaNs")
            continue

        train_sl, val_sl, test_sl = time_block_split(
            n, cfg.train_frac, cfg.val_frac
        )

        X_train, y_train = X[train_sl], y[train_sl]
        X_val, y_val = X[val_sl], y[val_sl]

        model, sigma_hat = fit_ols(X_train, y_train)
        y_val_pred = predict_mean(X_val, model)
        val_mae = mae(y_val, y_val_pred)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_k = k
            best_result = {
                "X": X, "y": y, "timestamps": timestamps, "feat_cols": feat_cols,
                "model": model, "sigma_hat": sigma_hat,
                "train_sl": train_sl, "val_sl": val_sl, "test_sl": test_sl,
            }

    if best_k is None:
        raise RuntimeError("No valid k; cannot proceed")

    print(f"Best k={best_k} (val MAE={best_val_mae:.4f})")

    X = best_result["X"]
    y = best_result["y"]
    timestamps = best_result["timestamps"]
    feat_cols = best_result["feat_cols"]
    train_sl = best_result["train_sl"]
    val_sl = best_result["val_sl"]
    test_sl = best_result["test_sl"]

    n_train = train_sl.stop
    n_val = val_sl.stop - val_sl.start
    trainval_end = n_train + n_val
    X_trainval = X[:trainval_end]
    y_trainval = y[:trainval_end]

    model, sigma_hat = fit_ols(X_trainval, y_trainval)

    X_test = X[test_sl]
    y_test = y[test_sl]
    ts_test = timestamps.iloc[test_sl]

    y_pred = predict_mean(X_test, model)
    lower_90, upper_90 = predict_intervals(y_pred, sigma_hat, 0.90)
    lower_95, upper_95 = predict_intervals(y_pred, sigma_hat, 0.95)

    test_mae = mae(y_test, y_pred)
    test_mse = mse(y_test, y_pred)
    cov_90 = coverage(y_test, lower_90, upper_90)
    cov_95 = coverage(y_test, lower_95, upper_95)

    metrics = {
        "best_k": best_k,
        "test_mae": test_mae,
        "test_mse": test_mse,
        "coverage_90": cov_90,
        "coverage_95": cov_95,
    }

    pred_df = pd.DataFrame({
        "timestamp": ts_test.iloc[:, 0].values,
        "y_true": y_test,
        "y_pred": y_pred,
        "lower_90": lower_90,
        "upper_90": upper_90,
        "lower_95": lower_95,
        "upper_95": upper_95,
    })
    pred_path = out_dir / "predictions_test.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    best_model = serialize_model(model, sigma_hat, best_k, feat_cols)
    best_model_path = out_dir / "best_model.json"
    with open(best_model_path, "w") as f:
        json.dump(best_model, f, indent=2)

    plot_timeseries_week(
        ts_test, y_test, y_pred, lower_90, upper_90,
        str(figures_dir / "timeseries_week.png"),
    )
    plot_parity(y_test, y_pred, str(figures_dir / "parity_plot.png"))
    plot_interval_coverage(cov_90, cov_95, str(figures_dir / "interval_coverage.png"))

    print(f"Test MAE: {test_mae:.4f}, MSE: {test_mse:.4f}")
    print(f"Coverage 90%: {cov_90:.4f}, 95%: {cov_95:.4f}")
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
