import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
