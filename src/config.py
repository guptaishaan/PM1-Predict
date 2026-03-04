from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    col_time: str = "Start of Period"
    col_target: str = "PM1 mass concentration, 1-hour mean raw"
    col_pm25: str = "PM2.5 mass concentration, 1-hour mean raw"
    col_pm10: str = "PM10 mass concentration, 1-hour mean raw"
    col_pm25_status: str = "PM2.5 mass concentration, 1-hour mean status"
    filter_value: str = "calibrated-ready"
    filter_flag: str = "use_calibrated_ready"
    horizon: int = 1
    candidate_ks: tuple = (0, 1, 3, 6, 12)
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
