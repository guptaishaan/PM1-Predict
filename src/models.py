import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression


def fit_ols(X_train: np.ndarray, y_train: np.ndarray) -> tuple[object, float]:
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    resid = y_train - model.predict(X_train)
    sigma_hat = float(np.std(resid, ddof=1))
    return model, sigma_hat


def predict_mean(X: np.ndarray, model: object) -> np.ndarray:
    return model.predict(X)


def predict_intervals(
    mu: np.ndarray, sigma: float, level: float
) -> tuple[np.ndarray, np.ndarray]:
    z = float(norm.ppf(0.5 + level / 2))
    lower = mu - z * sigma
    upper = mu + z * sigma
    return lower, upper


def serialize_model(model: object, sigma_hat: float, k: int, feature_names: list) -> dict:
    return {
        "intercept": float(model.intercept_),
        "coefficients": [float(c) for c in model.coef_],
        "sigma_hat": float(sigma_hat),
        "k": int(k),
        "feature_names": list(feature_names),
    }
