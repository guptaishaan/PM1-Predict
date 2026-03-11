from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from .metrics import coverage, mae, mse
from .models import fit_ols, predict_intervals, predict_mean, serialize_model


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float) -> tuple[object, float]:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    resid = y_train - model.predict(X_train)
    sigma_hat = float(np.std(resid, ddof=1))
    return model, sigma_hat


def serialize_ridge(model: object, sigma_hat: float, k: int, feature_names: list, alpha: float) -> dict:
    d = serialize_model(model, sigma_hat, k, feature_names)
    d["model_name"] = "map_ridge"
    d["alpha"] = float(alpha)
    return d


def ridge_val_mae_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: list[float],
) -> tuple[list[float], list[float]]:
    val_maes = []
    for alpha in alphas:
        model, _ = fit_ridge(X_train, y_train, alpha)
        y_val_pred = predict_mean(X_val, model)
        val_maes.append(mae(y_val, y_val_pred))
    return list(alphas), val_maes


def select_ridge_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: list[float],
) -> tuple[float, float]:
    alphas_list, val_maes = ridge_val_mae_path(X_train, y_train, X_val, y_val, alphas)
    i = int(np.argmin(val_maes))
    return alphas_list[i], val_maes[i]


def run_map_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_k: int,
    feat_cols: list,
    alphas: list[float],
    out_dir: Path | None = None,
) -> dict:
    best_alpha, val_mae = select_ridge_alpha(X_train, y_train, X_val, y_val, alphas)
    model, sigma_hat = fit_ridge(X_trainval, y_trainval, best_alpha)
    y_pred = predict_mean(X_test, model)
    lower_90, upper_90 = predict_intervals(y_pred, sigma_hat, 0.90)
    lower_95, upper_95 = predict_intervals(y_pred, sigma_hat, 0.95)
    test_mae = mae(y_test, y_pred)
    test_mse = mse(y_test, y_pred)
    cov_90 = coverage(y_test, lower_90, upper_90)
    cov_95 = coverage(y_test, lower_95, upper_95)

    result = {
        "model_name": "map_ridge",
        "best_k": best_k,
        "best_alpha": best_alpha,
        "validation_MAE": val_mae,
        "test_MAE": test_mae,
        "test_MSE": test_mse,
        "coverage_90": cov_90,
        "coverage_95": cov_95,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "best_ridge_model.json", "w") as f:
            json.dump(serialize_ridge(model, sigma_hat, best_k, feat_cols, best_alpha), f, indent=2)

    return result
