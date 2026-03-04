import pandas as pd


def load_data(path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    col_time = "Start of Period"
    df[col_time] = pd.to_datetime(df[col_time], utc=True)
    df = df.sort_values(col_time).reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=[col_time], keep="first").reset_index(drop=True)
    n_after = len(df)
    diagnostics = {
        "path": path,
        "n_rows_raw": n_before,
        "n_rows_after_dedup": n_after,
        "n_duplicates_dropped": n_before - n_after,
    }
    return df, diagnostics
