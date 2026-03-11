import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap_analysis import run_bootstrap
from .config import Config
from .distribution_fits import (
    best_by_aic,
    daily_average_pm1,
    fit_all_distributions,
    fit_normal,
    fits_to_dataframe,
)
from .features import make_supervised
from .io import load_data
from .map_models import fit_ridge, ridge_val_mae_path, run_map_ridge
from .metrics import coverage, mae, mse
from .models import fit_ols, predict_intervals, predict_mean, serialize_model
from .plots import (
    plot_aic_comparison,
    plot_distribution_fit_overlay,
    plot_interval_coverage,
    plot_map_vs_mle_bar,
    plot_parity,
    plot_qq_residuals,
    plot_residual_vs_fitted,
    plot_ridge_coef_shrinkage,
    plot_ridge_val_mae_vs_alpha,
    plot_timeseries_week,
)
from .split import time_block_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/extract_0.csv")
    parser.add_argument("--use_calibrated_ready", action="store_true")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run_bootstrap", action="store_true")
    parser.add_argument("--bootstrap_B", type=int, default=200)
    parser.add_argument("--run_map", action="store_true")
    parser.add_argument("--run_distribution_fit", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    out_dir = Path(args.output_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, diag = load_data(args.data_path)
    n_loaded = diag["n_rows_after_dedup"]
    print(f"Loaded: {n_loaded} rows, duplicates dropped: {diag['n_duplicates_dropped']}")

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
    X_train = X[train_sl]
    y_train = y[train_sl]
    X_val = X[val_sl]
    y_val = y[val_sl]

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
    best_model["model_name"] = "baseline_ols"
    best_model_path = out_dir / "best_model.json"
    with open(best_model_path, "w") as f:
        json.dump(best_model, f, indent=2)

    plot_timeseries_week(
        ts_test, y_test, y_pred, lower_90, upper_90,
        str(figures_dir / "timeseries_week.png"),
    )
    plot_parity(y_test, y_pred, str(figures_dir / "parity_plot.png"))
    plot_interval_coverage(cov_90, cov_95, str(figures_dir / "interval_coverage.png"))
    resid_test = y_test - y_pred
    plot_residual_vs_fitted(y_pred, resid_test, str(figures_dir / "residual_vs_fitted.png"))
    plot_qq_residuals(resid_test, str(figures_dir / "residual_qq_normal.png"))

    n_supervised = len(y)
    print(f"Rows used after lag construction: {n_supervised}")

    summary_state = {
        "baseline_best_k": best_k,
        "baseline_test_mae": test_mae,
        "baseline_test_mse": test_mse,
        "ridge_best_alpha": None,
        "ridge_test_mae": None,
        "bootstrap_mae_ci": None,
        "best_dist_hourly": None,
        "best_dist_daily": None,
        "best_dist_residuals": None,
    }

    if args.run_bootstrap:
        B = max(1, args.bootstrap_B)
        print(f"Running bootstrap B={B}...")
        boot_summary = run_bootstrap(
            X_train, y_train, X_test, y_test, feat_cols, B,
            out_dir=out_dir,
            observed_mae=test_mae,
            observed_mse=test_mse,
        )
        ci = (boot_summary["bootstrap_mae_ci95_low"], boot_summary["bootstrap_mae_ci95_high"])
        summary_state["bootstrap_mae_ci"] = ci
        print(f"Bootstrap MAE mean={boot_summary['bootstrap_mae_mean']:.4f}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    if args.run_map:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        print("Running MAP/Ridge...")
        ridge_result = run_map_ridge(
            X_train, y_train, X_val, y_val, X_trainval, y_trainval,
            X_test, y_test, best_k, feat_cols, alphas, out_dir=out_dir,
        )
        ols_row = {
            "model_name": "baseline_ols",
            "best_k": best_k,
            "best_alpha": None,
            "validation_MAE": best_val_mae,
            "test_MAE": test_mae,
            "test_MSE": test_mse,
            "coverage_90": cov_90,
            "coverage_95": cov_95,
        }
        comp_rows = [ols_row, ridge_result]
        pd.DataFrame(comp_rows).to_csv(out_dir / "map_vs_mle_comparison.csv", index=False)
        plot_map_vs_mle_bar(comp_rows, str(figures_dir / "map_vs_mle_bar.png"))
        model_ridge, _ = fit_ridge(X_trainval, y_trainval, ridge_result["best_alpha"])
        plot_ridge_coef_shrinkage(
            np.array(model.coef_), np.array(model_ridge.coef_), feat_cols,
            str(figures_dir / "ridge_coef_shrinkage.png"),
        )
        alphas_path, val_maes_path = ridge_val_mae_path(X_train, y_train, X_val, y_val, alphas)
        plot_ridge_val_mae_vs_alpha(
            alphas_path, val_maes_path, ridge_result["best_alpha"],
            str(figures_dir / "ridge_val_mae_vs_alpha.png"),
        )
        summary_state["ridge_best_alpha"] = ridge_result["best_alpha"]
        summary_state["ridge_test_mae"] = ridge_result["test_MAE"]
        print(f"Ridge best_alpha={ridge_result['best_alpha']}, test MAE={ridge_result['test_MAE']:.4f}")

    if args.run_distribution_fit:
        print("Running distribution fitting...")
        hourly_pm1 = df[cfg.col_target].dropna().values.astype(float)
        hourly_pm1 = hourly_pm1[np.isfinite(hourly_pm1)]
        daily_pm1 = daily_average_pm1(df, cfg.col_time, cfg.col_target)
        residuals = (y_test - y_pred).astype(float)

        fits_hourly = fit_all_distributions(hourly_pm1, "hourly_pm1")
        fits_daily = fit_all_distributions(daily_pm1, "daily_avg_pm1")
        if np.any(residuals <= 0):
            r = fit_normal(residuals)
            fits_resid = [r] if r else []
            print("Residuals contain nonpositive values; fitting Normal only for residuals.")
        else:
            fits_resid = fit_all_distributions(residuals, "residuals")

        fits_to_dataframe(fits_hourly).to_csv(out_dir / "distribution_fits_hourly_pm1.csv", index=False)
        fits_to_dataframe(fits_daily).to_csv(out_dir / "distribution_fits_daily_pm1.csv", index=False)
        fits_to_dataframe(fits_resid).to_csv(out_dir / "distribution_fits_residuals.csv", index=False)

        if fits_hourly:
            plot_distribution_fit_overlay(
                hourly_pm1, fits_hourly, str(figures_dir / "hourly_pm1_distribution_fit.png"),
                "Hourly PM1: distribution fits",
            )
        if fits_daily:
            plot_distribution_fit_overlay(
                daily_pm1, fits_daily, str(figures_dir / "daily_pm1_distribution_fit.png"),
                "Daily average PM1: distribution fits",
            )
        if fits_resid:
            plot_distribution_fit_overlay(
                residuals, fits_resid, str(figures_dir / "residual_distribution_fit.png"),
                "Test residuals: distribution fits",
            )
        if fits_hourly:
            plot_aic_comparison(
                fits_hourly, str(figures_dir / "hourly_pm1_aic_comparison.png"),
                "Hourly PM1: AIC by distribution",
            )
        if fits_daily:
            plot_aic_comparison(
                fits_daily, str(figures_dir / "daily_pm1_aic_comparison.png"),
                "Daily PM1: AIC by distribution",
            )
        if fits_resid:
            plot_aic_comparison(
                fits_resid, str(figures_dir / "residuals_aic_comparison.png"),
                "Residuals: AIC by distribution",
            )

        summary_state["best_dist_hourly"] = best_by_aic(fits_hourly)
        summary_state["best_dist_daily"] = best_by_aic(fits_daily)
        summary_state["best_dist_residuals"] = best_by_aic(fits_resid)
        print(f"Best AIC hourly PM1: {summary_state['best_dist_hourly']}")
        print(f"Best AIC daily PM1: {summary_state['best_dist_daily']}")
        print(f"Best AIC residuals: {summary_state['best_dist_residuals']}")

    print("--- Summary ---")
    print(f"Rows loaded (after dedup): {n_loaded}")
    print(f"Rows after lag construction: {n_supervised}")
    print(f"Baseline best_k={best_k}, test MAE={test_mae:.4f}, test MSE={test_mse:.4f}")
    if summary_state["ridge_best_alpha"] is not None:
        print(f"Ridge best_alpha={summary_state['ridge_best_alpha']}, test MAE={summary_state['ridge_test_mae']:.4f}")
    if summary_state["bootstrap_mae_ci"] is not None:
        lo, hi = summary_state["bootstrap_mae_ci"]
        print(f"Bootstrap MAE 95% interval: [{lo:.4f}, {hi:.4f}]")
    if summary_state["best_dist_hourly"] is not None:
        print(f"Best distribution (AIC) hourly PM1: {summary_state['best_dist_hourly']}")
        print(f"Best distribution (AIC) daily PM1: {summary_state['best_dist_daily']}")
        print(f"Best distribution (AIC) residuals: {summary_state['best_dist_residuals']}")
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
