# src/features.py
import numpy as np
import pandas as pd


def add_features(df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các feature:
    - log_return
    - ma_5, ma_20, vol_20
    - vol_ma_5, vol_ma_20
    - các lag: close_lag_*, ret_lag_*

    Ưu tiên dùng Close_adj, Volume_adj nếu có, nếu không thì dùng Close, Volume.
    """
    df = df_t.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Chọn cột giá & volume
    price_col = "Close_adj" if "Close_adj" in df.columns else "Close"
    volume_col = "Volume_adj" if "Volume_adj" in df.columns else "Volume"

    # log_return
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    # Moving averages & volatility
    df["ma_5"] = df[price_col].rolling(window=5).mean()
    df["ma_20"] = df[price_col].rolling(window=20).mean()
    df["vol_20"] = df["log_return"].rolling(window=20).std()

    # Volume MA
    df["vol_ma_5"] = df[volume_col].rolling(window=5).mean()
    df["vol_ma_20"] = df[volume_col].rolling(window=20).mean()

    # Lag features
    lags = [1, 2, 3, 5, 10]
    for lag in lags:
        df[f"close_lag_{lag}"] = df[price_col].shift(lag)
        df[f"ret_lag_{lag}"] = df["log_return"].shift(lag)

    return df


def get_feature_cols(df: pd.DataFrame):
    base_features = [
        "Close",
        "Close_adj",
        "Volume",
        "Volume_adj",
        "log_return",
        "ma_5",
        "ma_20",
        "vol_20",
        "vol_ma_5",
        "vol_ma_20",
    ]

    lag_features = [
        c
        for c in df.columns
        if c.startswith("close_lag_") or c.startswith("ret_lag_")
    ]

    feature_cols = base_features + lag_features
    # Giữ những cột thực sự tồn tại
    feature_cols = [c for c in feature_cols if c in df.columns]
    # Bỏ trùng
    feature_cols = list(dict.fromkeys(feature_cols))

    return feature_cols
