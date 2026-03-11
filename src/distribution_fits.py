from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _aic(n, k_params, loglik):
    return 2 * k_params - 2 * loglik


def _bic(n, k_params, loglik):
    return k_params * np.log(n) - 2 * loglik


def fit_normal(x: np.ndarray) -> dict | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return None
    params = stats.norm.fit(x)
    loglik = np.sum(stats.norm.logpdf(x, *params))
    k = 2
    return {
        "distribution": "Normal",
        "params": {"loc": float(params[0]), "scale": float(params[1])},
        "log_likelihood": float(loglik),
        "AIC": float(_aic(len(x), k, loglik)),
        "BIC": float(_bic(len(x), k, loglik)),
        "n": int(len(x)),
    }


def fit_lognormal(x: np.ndarray) -> dict | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 5:
        return None
    params = stats.lognorm.fit(x, floc=0)
    loglik = np.sum(stats.lognorm.logpdf(x, *params))
    k = 2
    return {
        "distribution": "Lognormal",
        "params": {"s": float(params[0]), "scale": float(params[2])},
        "log_likelihood": float(loglik),
        "AIC": float(_aic(len(x), k, loglik)),
        "BIC": float(_bic(len(x), k, loglik)),
        "n": int(len(x)),
    }


def fit_gamma(x: np.ndarray) -> dict | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 5:
        return None
    try:
        params = stats.gamma.fit(x, floc=0)
    except Exception:
        return None
    loglik = np.sum(stats.gamma.logpdf(x, *params))
    k = 2
    return {
        "distribution": "Gamma",
        "params": {"a": float(params[0]), "scale": float(params[2])},
        "log_likelihood": float(loglik),
        "AIC": float(_aic(len(x), k, loglik)),
        "BIC": float(_bic(len(x), k, loglik)),
        "n": int(len(x)),
    }


def fit_all_distributions(x: np.ndarray, name: str) -> list[dict]:
    fits = []
    n = fit_normal(x)
    if n:
        fits.append(n)
    if np.all(np.asarray(x, dtype=float)[np.isfinite(x)] > 0):
        ln = fit_lognormal(x)
        if ln:
            fits.append(ln)
        g = fit_gamma(x)
        if g:
            fits.append(g)
    else:
        pos = np.asarray(x, dtype=float)
        pos = pos[np.isfinite(pos) & (pos > 0)]
        if len(pos) >= 5:
            ln = fit_lognormal(pos)
            if ln:
                ln["note"] = "fitted_to_positive_subset_only"
                fits.append(ln)
            g = fit_gamma(pos)
            if g:
                g["note"] = "fitted_to_positive_subset_only"
                fits.append(g)
    for f in fits:
        f["quantity"] = name
    return fits


def fits_to_dataframe(fits: list[dict]) -> pd.DataFrame:
    rows = []
    for f in fits:
        row = {
            "quantity": f.get("quantity", ""),
            "distribution": f["distribution"],
            "log_likelihood": f["log_likelihood"],
            "AIC": f["AIC"],
            "BIC": f["BIC"],
            "n": f["n"],
        }
        row["params_json"] = str(f["params"])
        if "note" in f:
            row["note"] = f["note"]
        rows.append(row)
    return pd.DataFrame(rows)


def best_by_aic(fits: list[dict]) -> str | None:
    if not fits:
        return None
    return min(fits, key=lambda f: f["AIC"])["distribution"]


def daily_average_pm1(df: pd.DataFrame, col_time: str, col_pm1: str) -> np.ndarray:
    df = df.copy()
    df["_date"] = pd.to_datetime(df[col_time]).dt.normalize()
    daily = df.groupby("_date")[col_pm1].mean().dropna()
    return daily.values
