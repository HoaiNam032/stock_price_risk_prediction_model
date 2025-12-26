# src/train_risk_models.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

from .corporate_actions import adjust_all_tickers
from .features import add_features


# Số ngày giả định cho 1 quý / 2 quý ( trading days )
HORIZON_1Q_DAYS = 63
HORIZON_2Q_DAYS = 126

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


def build_base_df() -> pd.DataFrame:
    """
    1) Đọc cleaned_prices.csv
    2) Adjust corporate actions cho toàn bộ mã
    3) Thêm features (lag, log_return, indicators...)
    """
    data_path = DATA_DIR / "cleaned_prices.csv"
    df_raw = pd.read_csv(data_path)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print(f"Kích thước df_raw: {df_raw.shape}")

    # Adjust tất cả ticker
    df_adj = adjust_all_tickers(df_raw, ratio_down=0.7, ratio_up=1.5)
    print(f"Kích thước df sau adjust: {df_adj.shape}")

    # Thêm features
    df_feat = add_features(df_adj.copy())
    print(f"Kích thước df_feat: {df_feat.shape}")

    return df_feat


def add_drawdown_targets(
    df_feat: pd.DataFrame,
    price_col_preferred: str = "Close_adj",
    horizon_1q: int = HORIZON_1Q_DAYS,
    horizon_2q: int = HORIZON_2Q_DAYS,
) -> pd.DataFrame:
    """
    Thêm 2 cột target:
      - dd_1q: max drawdown trong 1 quý tới (dựa trên giá thấp nhất)
      - dd_2q: max drawdown trong 2 quý tới

    Công thức:
      dd_1q = min_future_price_1q / current_price - 1.0  ( <= 0 )
    """

    df = df_feat.sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()

    # Chọn cột giá dùng cho risk (ưu tiên Close_adj)
    if price_col_preferred in df.columns:
        price_col = price_col_preferred
    else:
        price_col = "Close"
    if price_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột giá '{price_col}' trong df_feat")

    print(f"Dùng cột giá cho drawdown: {price_col}")

    all_chunks = []

    for ticker, g in df.groupby("Ticker", sort=False):
        g = g.sort_values("Date").copy()
        s = g[price_col].astype(float)

        # future_min_1q: min của [hôm nay .. hôm nay + 1Q-1] (bao gồm hôm nay)
        future_min_1q = (
            s.iloc[::-1]
            .rolling(window=horizon_1q, min_periods=horizon_1q)
            .min()
            .iloc[::-1]
        )

        future_min_2q = (
            s.iloc[::-1]
            .rolling(window=horizon_2q, min_periods=horizon_2q)
            .min()
            .iloc[::-1]
        )

        dd_1q = future_min_1q / s - 1.0
        dd_2q = future_min_2q / s - 1.0

        g["dd_1q"] = dd_1q
        g["dd_2q"] = dd_2q

        all_chunks.append(g)

    df_risk = pd.concat(all_chunks, ignore_index=True)
    print(f"Kích thước df_risk (sau khi thêm dd_1q, dd_2q): {df_risk.shape}")

    return df_risk


def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Chọn danh sách feature numeric để train ML risk:
      - bỏ cột định danh, target
    """
    drop_cols = {"Ticker", "Date", "dd_1q", "dd_2q"}
    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feature_cols


def train_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    horizon_name: str,
    model_dir: Path = MODEL_DIR,
):
    """
    Train 4 mô hình GradientBoostingRegressor quantile cho 1 horizon:

      - alpha = 0.05  -> VaR 5%   (Rating A)
      - alpha = 0.03  -> VaR 3%   (Rating B)
      - alpha = 0.01  -> VaR 1%   (Rating C)
      - alpha = 0.001 -> VaR 0.1% (Rating D)

    horizon_name: "1q" hoặc "2q"
    """

    os.makedirs(model_dir, exist_ok=True)

    alphas = [0.05, 0.03, 0.01, 0.001]

    for alpha in alphas:
        print(f"  -> Train GB quantile for {horizon_name}, alpha={alpha} ...")

        gb = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )

        gb.fit(X_train, y_train)

        # ví dụ tên file:
        #  alpha=0.05  -> risk_gb_dd_1q_q50.pkl
        #  alpha=0.03  -> risk_gb_dd_1q_q30.pkl
        #  alpha=0.01  -> risk_gb_dd_1q_q10.pkl
        #  alpha=0.001 -> risk_gb_dd_1q_q1.pkl
        q_tag = int(alpha * 1000)
        model_path = model_dir / f"risk_gb_dd_{horizon_name}_q{q_tag}.pkl"
        joblib.dump(gb, model_path)
        print(f"     💾 Saved: {model_path}")


def main():
    print("===========================================")
    print("  TRAIN RISK MODELS: MAX DRAWDOWN 1Q / 2Q")
    print("===========================================\n")

    # 1) Xây df_feat (đã adjust + features)
    df_feat = build_base_df()

    # 2) Thêm target dd_1q, dd_2q
    df_risk = add_drawdown_targets(df_feat)
    print(f"Kích thước df_risk: {df_risk.shape}")

    # 3) Bỏ các dòng không có dd_1q hoặc dd_2q
    df_risk = df_risk.dropna(subset=["dd_1q", "dd_2q"]).copy()
    print(f"Sau khi drop NaN dd_1q/dd_2q: {df_risk.shape}")

    # 4) Chia train/test theo thời gian (80% train, 20% test) – chỉ để log, không dùng test cho gì nặng
    df_risk = df_risk.sort_values("Date").reset_index(drop=True)
    cutoff_date = df_risk["Date"].quantile(0.8)

    train = df_risk[df_risk["Date"] <= cutoff_date].copy()
    test = df_risk[df_risk["Date"] > cutoff_date].copy()

    print(f"Train size (ban đầu): {train.shape} Test size (ban đầu): {test.shape}")

    # 5) Chọn feature columns
    feature_cols = get_feature_cols(df_risk)
    print(f"Số lượng feature: {len(feature_cols)}")

    # 6) Bỏ các dòng có NaN trong feature (cho sạch dữ liệu train)
    train = train.dropna(subset=feature_cols).copy()
    test = test.dropna(subset=feature_cols).copy()
    print(f"Sau khi drop NaN feature:")
    print(f"  Train size: {train.shape} Test size: {test.shape}\n")

    # 7) Chuẩn bị X_train, y_train cho 2 horizon
    X_train = train[feature_cols].values
    y_train_1q = train["dd_1q"].values
    y_train_2q = train["dd_2q"].values

    # 8) Train multi-quantile GB models
    print("=== Train MULTI-QUANTILE GB models (ML VaR) ===")
    print("Horizon 1Q (dd_1q):")
    train_quantile_models(X_train, y_train_1q, horizon_name="1q")

    print("\nHorizon 2Q (dd_2q):")
    train_quantile_models(X_train, y_train_2q, horizon_name="2q")

    print("\n✅ Done. All ML quantile risk models trained and saved.\n")


if __name__ == "__main__":
    main()
