# src/corporate_actions.py
from typing import Tuple

import numpy as np
import pandas as pd


def detect_corporate_actions_for_ticker(
    df_t: pd.DataFrame,
    ratio_down: float = 0.7,
    ratio_up: float = 1.5,
) -> pd.DataFrame:
    """
    Phát hiện ngày nghi ngờ có corporate action dựa trên nhảy giá lớn
    giữa hai phiên liên tiếp.

    - df_t: DataFrame của 1 ticker, đã sort theo Date.
      Cần có cột 'Close'.

    Trả về df_t có thêm:
      - prev_close
      - price_ratio (Close / prev_close)
      - is_corporate_action (bool): True nếu ratio < ratio_down hoặc > ratio_up
    """
    df = df_t.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["prev_close"] = df["Close"].shift(1)
    df["price_ratio"] = df["Close"] / df["prev_close"]

    # corporate action xảy ra tại ngày i,
    # tức là giữa prev_close[i] và Close[i]
    df["is_corporate_action"] = False
    mask_event = (df["price_ratio"] < ratio_down) | (df["price_ratio"] > ratio_up)
    df.loc[mask_event, "is_corporate_action"] = True

    return df


def compute_adjustment_factors(df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Dựa vào cột 'is_corporate_action' + 'price_ratio',
    tính adj_factor cum cho từng ngày để backward-adjust.

    Logic:
      - Nếu ngày i là corporate_action:
          factor_i = price_ratio_i = Close[i] / Close[i-1]
          (áp dụng cho toàn bộ ngày < i)
      - cumulative_factor[i] = tích của tất cả factor_j với j > i

    Ta tính cumulative_factor bằng cách duyệt ngược thời gian.
    """
    df = df_t.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    n = len(df)
    # factor_j = 1 nếu không phải corporate action
    factor = np.ones(n)

    # factor tại ngày i: dùng price_ratio_i
    event_idx = df.index[df["is_corporate_action"]].tolist()
    for idx in event_idx:
        ratio = df.loc[idx, "price_ratio"]
        # tránh chia cho 0 / NaN
        if pd.notna(ratio) and ratio > 0:
            factor[idx] = ratio

    # cumulative_factor[i] = tích factor[j] với j > i
    cum_factor = np.ones(n)
    for i in range(n - 2, -1, -1):
        cum_factor[i] = cum_factor[i + 1] * factor[i + 1]

    df["adj_factor"] = cum_factor

    return df


def apply_price_adjustment(df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện full pipeline cho 1 ticker:
      1) detect_corporate_actions_for_ticker
      2) compute_adjustment_factors
      3) Tạo thêm các cột giá đã điều chỉnh:
         - Close_adj, Open_adj, High_adj, Low_adj
         (Volume có thể điều chỉnh ngược lại, tùy nhu cầu)

    Lưu ý: Không ghi đè lên giá gốc, chỉ thêm cột *_adj.
    """
    df = detect_corporate_actions_for_ticker(df_t)
    df = compute_adjustment_factors(df)

    # Áp dụng adj_factor cho giá
    for col in ["Close", "Open", "High", "Low"]:
        if col in df.columns:
            df[f"{col}_adj"] = df[col] * df["adj_factor"]

    # Tuỳ bạn: có thể muốn điều chỉnh Volume ngược lại (chia cho factor)
    # Ví dụ:
    # if "Volume" in df.columns:
    #     df["Volume_adj"] = df["Volume"] / df["adj_factor"]

    return df


def adjust_all_tickers(
    df_all: pd.DataFrame,
    ratio_down: float = 0.7,
    ratio_up: float = 1.3,
) -> pd.DataFrame:
    """
    Áp dụng adjust cho toàn bộ DataFrame chứa nhiều ticker.

    - Nhóm theo 'Ticker'
    - Với mỗi nhóm: detect + compute adj_factor + tạo *_adj

    Trả về DataFrame đã ghép lại.
    """
    if "Ticker" not in df_all.columns:
        raise ValueError("DataFrame phải có cột 'Ticker'.")

    def _adjust_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date")
        # Gọi lại hàm với threshold mong muốn
        g = detect_corporate_actions_for_ticker(
            g, ratio_down=ratio_down, ratio_up=ratio_up
        )
        g = compute_adjustment_factors(g)
        for col in ["Close", "Open", "High", "Low"]:
            if col in g.columns:
                g[f"{col}_adj"] = g[col] * g["adj_factor"]
        return g

    df_adj = (
        df_all.sort_values(["Ticker", "Date"])
        .groupby("Ticker", group_keys=False)
        .apply(_adjust_one)
        .reset_index(drop=True)
    )

    return df_adj
