from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import mae, mse
from .models import fit_ols, predict_mean


def run_bootstrap(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_cols: list,
    B: int,
    seed: int = 42,
    out_dir: Path | None = None,
    observed_mae: float | None = None,
    observed_mse: float | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train = len(y_train)
    test_maes = []
    test_mses = []
    intercepts = []
    coef_rows = []

    for b in range(B):
        idx = rng.integers(0, n_train, size=n_train)
        X_b = X_train[idx]
        y_b = y_train[idx]
        model, _ = fit_ols(X_b, y_b)
        y_pred = predict_mean(X_test, model)
        test_maes.append(mae(y_test, y_pred))
        test_mses.append(mse(y_test, y_pred))
        intercepts.append(float(model.intercept_))
        coef_rows.append([float(c) for c in model.coef_])

    test_maes = np.array(test_maes)
    test_mses = np.array(test_mses)
    intercepts = np.array(intercepts)
    coef_arr = np.array(coef_rows)

    def pct(a, q):
        return float(np.percentile(a, q))

    if observed_mae is None or observed_mse is None:
        model_obs, _ = fit_ols(X_train, y_train)
        y_obs = predict_mean(X_test, model_obs)
        if observed_mae is None:
            observed_mae = mae(y_test, y_obs)
        if observed_mse is None:
            observed_mse = mse(y_test, y_obs)

    summary = {
        "B": B,
        "observed_test_mae": observed_mae,
        "observed_test_mse": observed_mse,
        "bootstrap_mae_mean": float(np.mean(test_maes)),
        "bootstrap_mae_std": float(np.std(test_maes, ddof=1)),
        "bootstrap_mae_ci95_low": pct(test_maes, 2.5),
        "bootstrap_mae_ci95_high": pct(test_maes, 97.5),
        "bootstrap_mse_mean": float(np.mean(test_mses)),
        "bootstrap_mse_std": float(np.std(test_mses, ddof=1)),
        "bootstrap_mse_ci95_low": pct(test_mses, 2.5),
        "bootstrap_mse_ci95_high": pct(test_mses, 97.5),
        "intercept_mean": float(np.mean(intercepts)),
        "intercept_std": float(np.std(intercepts, ddof=1)),
        "intercept_ci95_low": pct(intercepts, 2.5),
        "intercept_ci95_high": pct(intercepts, 97.5),
    }

    coef_summary = []
    for j, name in enumerate(feat_cols):
        col = coef_arr[:, j]
        coef_summary.append({
            "feature": name,
            "mean": float(np.mean(col)),
            "std": float(np.std(col, ddof=1)),
            "ci95_low": pct(col, 2.5),
            "ci95_high": pct(col, 97.5),
        })
    summary["coefficients"] = coef_summary

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        from .plots import (
            plot_bootstrap_coef_intervals,
            plot_bootstrap_mae_hist,
            plot_bootstrap_mse_hist,
        )
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_bootstrap_mae_hist(test_maes, observed_mae, str(fig_dir / "bootstrap_mae_hist.png"))
        plot_bootstrap_mse_hist(test_mses, observed_mse, str(fig_dir / "bootstrap_mse_hist.png"))
        plot_bootstrap_coef_intervals(coef_summary, str(fig_dir / "bootstrap_coef_intervals.png"))
        summary_dump = {k: v for k, v in summary.items() if k != "coefficients"}
        summary_dump["coefficients"] = coef_summary
        with open(out_dir / "bootstrap_metrics.json", "w") as f:
            json.dump(summary_dump, f, indent=2)

        coef_df = pd.DataFrame(coef_rows, columns=feat_cols)
        coef_df.insert(0, "bootstrap_id", range(B))
        coef_df.insert(1, "intercept", intercepts)
        coef_df.to_csv(out_dir / "bootstrap_coefficients.csv", index=False)

    return summary
