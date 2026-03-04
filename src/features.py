import numpy as np
import pandas as pd


def make_supervised(
    df: pd.DataFrame,
    k: int,
    h: int,
    col_target: str,
    col_pm25: str,
    col_pm10: str,
    col_time: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list]:
    df = df.copy()
    lags = list(range(k + 1))
    feat_cols = []
    for j in lags:
        df[f"pm25_t_minus_{j}"] = df[col_pm25].shift(j)
        df[f"pm10_t_minus_{j}"] = df[col_pm10].shift(j)
        feat_cols.extend([f"pm25_t_minus_{j}", f"pm10_t_minus_{j}"])
    df["y"] = df[col_target].shift(-h)
    keep = [col_time, "y"] + feat_cols
    df_sub = df[keep].dropna(how="any")
    X = df_sub[feat_cols].values
    y = df_sub["y"].values
    timestamps = df_sub[[col_time]].reset_index(drop=True)
    return X, y, timestamps, feat_cols
