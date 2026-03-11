import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_timeseries_week(
    timestamps: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_90: np.ndarray,
    upper_90: np.ndarray,
    out_path: str,
):
    col = timestamps.columns[0]
    ts = pd.to_datetime(timestamps[col])
    n = len(ts)
    n_week = min(7 * 24, n)
    start = n // 2 - n_week // 2
    end = start + n_week
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts.iloc[start:end], y_true[start:end], label="True", color="black", linewidth=1.5)
    ax.plot(ts.iloc[start:end], y_pred[start:end], label="Predicted", color="C0", linewidth=1)
    ax.fill_between(
        ts.iloc[start:end],
        lower_90[start:end],
        upper_90[start:end],
        alpha=0.3,
        color="C0",
        label="90% interval",
    )
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("PM1 (μg/m³)")
    ax.set_title("PM1 Forecast: 7-Day Window from Test Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    mx = max(y_true.max(), y_pred.max())
    mn = min(y_true.min(), y_pred.min())
    ax.plot([mn, mx], [mn, mx], "r--", label="y = x")
    ax.set_xlabel("True PM1 (μg/m³)")
    ax.set_ylabel("Predicted PM1 (μg/m³)")
    ax.set_title("Parity Plot (Test Set)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_interval_coverage(
    cov_90: float, cov_95: float, out_path: str
):
    fig, ax = plt.subplots(figsize=(6, 4))
    levels = ["90%", "95%"]
    nominal = [0.90, 0.95]
    empirical = [cov_90, cov_95]
    x = np.arange(len(levels))
    width = 0.35
    ax.bar(x - width / 2, nominal, width, label="Nominal", color="lightgray")
    ax.bar(x + width / 2, empirical, width, label="Empirical", color="C0")
    ax.set_ylabel("Coverage")
    ax.set_title("Prediction Interval Coverage (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_bootstrap_mse_hist(test_mses: np.ndarray, observed_mse: float, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(test_mses, bins=30, density=True, alpha=0.7, color="C2", edgecolor="white")
    ax.axvline(observed_mse, color="red", linestyle="--", linewidth=2, label=f"Observed MSE={observed_mse:.4f}")
    ax.set_xlabel("Test MSE (bootstrap refits)")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap Distribution of Test MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_bootstrap_mae_hist(test_maes: np.ndarray, observed_mae: float, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(test_maes, bins=30, density=True, alpha=0.7, color="C0", edgecolor="white")
    ax.axvline(observed_mae, color="red", linestyle="--", linewidth=2, label=f"Observed MAE={observed_mae:.4f}")
    ax.set_xlabel("Test MAE (bootstrap refits)")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap Distribution of Test MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_bootstrap_coef_intervals(coef_summary: list, out_path: str, max_features: int = 30):
    names = [c["feature"] for c in coef_summary[:max_features]]
    means = [c["mean"] for c in coef_summary[:max_features]]
    lows = [c["ci95_low"] for c in coef_summary[:max_features]]
    highs = [c["ci95_high"] for c in coef_summary[:max_features]]
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.2)))
    y = np.arange(len(names))
    err_low = np.array(means) - np.array(lows)
    err_high = np.array(highs) - np.array(means)
    ax.errorbar(means, y, xerr=[err_low, err_high], fmt="o", capsize=3)
    ax.axvline(0, color="gray", linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Coefficient (95% bootstrap interval)")
    ax.set_title("Bootstrap Coefficient Uncertainty")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_map_vs_mle_bar(rows: list, out_path: str):
    names = [r["model_name"] for r in rows]
    maes = [r["test_MAE"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, maes, color=["lightgray", "C0"][: len(names)])
    ax.set_ylabel("Test MAE")
    ax.set_title("MLE (OLS) vs MAP (Ridge)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_ridge_coef_shrinkage(coef_ols: np.ndarray, coef_ridge: np.ndarray, feat_names: list, out_path: str, max_features: int = 40):
    n = min(len(feat_names), max_features)
    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.15)))
    y = np.arange(n)
    width = 0.35
    ax.barh(y - width / 2, coef_ols[:n], width, label="OLS", color="lightgray")
    ax.barh(y + width / 2, coef_ridge[:n], width, label="Ridge", color="C0")
    ax.set_yticks(y)
    ax.set_yticklabels(feat_names[:n], fontsize=7)
    ax.set_xlabel("Coefficient")
    ax.legend()
    ax.set_title("OLS vs Ridge Coefficients")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_ridge_val_mae_vs_alpha(alphas: list, val_maes: list, best_alpha: float, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, val_maes, "o-", color="C0")
    ax.axvline(best_alpha, color="red", linestyle="--", label=f"Selected α={best_alpha}")
    ax.set_xscale("log")
    ax.set_xlabel("Ridge α (regularization strength)")
    ax.set_ylabel("Validation MAE")
    ax.set_title("Ridge: validation error vs α (MAP prior precision)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_aic_comparison(fits: list, out_path: str, title: str):
    if not fits:
        return
    names = [f["distribution"] for f in fits]
    aics = [f["AIC"] for f in fits]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    ax.bar(x, aics, color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("AIC (lower is better)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_qq_residuals(residuals: np.ndarray, out_path: str):
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot: Test Residuals vs Normal")
    ax.get_lines()[0].set_markersize(3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_residual_vs_fitted(y_pred: np.ndarray, residuals: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.4, s=8)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Fitted value ŷ (test)")
    ax.set_ylabel("Residual (y − ŷ)")
    ax.set_title("Residuals vs Fitted (homoscedasticity check)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def plot_distribution_fit_overlay(data: np.ndarray, fits: list, out_path: str, title: str, n_bins: int = 50):
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(x, bins=n_bins, density=True, alpha=0.5, color="gray", label="Data")
    xs = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 300)
    for f in fits:
        dist_name = f["distribution"]
        if dist_name == "Normal":
            params = stats.norm.fit(x)
            pdf = stats.norm.pdf(xs, *params)
        elif dist_name == "Lognormal":
            xp = x[x > 0]
            if len(xp) < 5:
                continue
            params = stats.lognorm.fit(xp, floc=0)
            pdf = stats.lognorm.pdf(xs, *params)
            pdf = np.where(xs > 0, pdf, np.nan)
        elif dist_name == "Gamma":
            xp = x[x > 0]
            if len(xp) < 5:
                continue
            params = stats.gamma.fit(xp, floc=0)
            pdf = stats.gamma.pdf(xs, *params)
            pdf = np.where(xs > 0, pdf, np.nan)
        else:
            continue
        ax.plot(xs, pdf, label=f"{dist_name} (AIC={f['AIC']:.1f})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()
