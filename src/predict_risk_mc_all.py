# src/predict_risk_mc_all.py
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .monte_carlo import mc_drawdown_quantiles_1q_2q

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def run_mc_risk_all(
    price_file: str = "cleaned_prices_adj.csv",
    output_prefix: str = "risk_mc_all_tickers",
    days_1q: int = 66,
    days_2q: int = 132,
    n_sim: int = 40_000,          # 👈 mô phỏng 40k path
    min_log_return_points: int = 10,
    log_every: int = 20,
) -> None:
    """
    Chạy Monte Carlo risk cho toàn bộ mã và sinh ra 3 file CSV:

    1) MC_1Q  (ước lượng mu,sigma từ ~1 quý gần nhất ≈ 66 phiên)
       -> data/risk_mc_all_tickers_mc1q.csv

    2) MC_1Y  (ước lượng mu,sigma từ ~1 năm gần nhất ≈ 250 phiên)
       -> data/risk_mc_all_tickers_mc1y.csv

    3) MC_FULL (ước lượng mu,sigma từ toàn bộ lịch sử log_return)
       -> data/risk_mc_all_tickers_mcfull.csv

    Mỗi file có dạng:
      - Ticker
      - current_price          (đơn vị giống file giá, hiện tại là 'nghìn VND')
      - last_date
      - mc_dd_q05_1q,  mc_dd_q03_1q,  mc_dd_q01_1q,  mc_dd_q001_1q
      - mc_price_q05_1q, ..., mc_price_q001_1q
      - mc_dd_q05_2q,  ...,   mc_dd_q001_2q
      - mc_price_q05_2q, ..., mc_price_q001_2q

    Lưu ý:
    - 66 & 132 ngày là horizon mô phỏng (1Q, 2Q)
    - Cửa sổ ước lượng mu,sigma:
        + 1Q  : 66 phiên gần nhất
        + 1Y  : 250 phiên gần nhất
        + FULL: toàn bộ lịch sử
    """

    print("===========================================")
    print("  RUN MONTE CARLO RISK FOR ALL TICKERS")
    print("  WINDOWS FOR MU,SIGMA: 1Q (66d), 1Y (250d), FULL")
    print(f"  n_sim per ticker/horizon = {n_sim}")
    print("===========================================\n")

    # ===== 1. Đọc & chuẩn hóa dữ liệu giá =====
    price_path = DATA_DIR / price_file
    if not price_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {price_path}")

    df_price = pd.read_csv(price_path)
    if "Date" not in df_price.columns:
        raise ValueError(f"{price_file} phải có cột 'Date'")

    df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce")

    if "Ticker" not in df_price.columns:
        raise ValueError(f"{price_file} phải có cột 'Ticker'")

    # Chọn cột giá: ưu tiên Close_adj, nếu không có thì dùng Close
    price_col = "Close_adj" if "Close_adj" in df_price.columns else "Close"
    if price_col not in df_price.columns:
        raise ValueError(
            f"{price_file} phải có cột '{price_col}' (Close hoặc Close_adj)"
        )

    # Sort theo Ticker, Date trước khi tính log_return
    df_price = df_price.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print(f"✅ Loaded price df: {price_path.name}, shape={df_price.shape}")
    print(f"   Dùng cột giá: {price_col}\n")

    # ===== 2. TÍNH log_return TỪ {price_col} THEO TỪNG MÃ =====
    df_price["log_return"] = (
        df_price
        .groupby("Ticker")[price_col]
        .transform(lambda s: np.log(s / s.shift(1)))
    )
    df_price.replace([np.inf, -np.inf], np.nan, inplace=True)

    all_tickers = sorted(df_price["Ticker"].dropna().unique().tolist())
    n_total = len(all_tickers)
    print(f"✅ Tìm thấy {n_total} mã trong dữ liệu.\n")

    # 👇 3 cửa sổ ước lượng mu,sigma:
    #    - win_1q   : ~ 1 quý gần nhất (66 phiên)
    #    - win_1y   : ~ 1 năm gần nhất (250 phiên)
    #    - win_full : toàn bộ lịch sử
    window_configs: dict[str, int | None] = {
        "win_1q": 66,
        "win_1y": 250,
        "win_full": None,
    }

    # output files tương ứng
    output_paths = {
        "win_1q": DATA_DIR / f"{output_prefix}_mc1q.csv",
        "win_1y": DATA_DIR / f"{output_prefix}_mc1y.csv",
        "win_full": DATA_DIR / f"{output_prefix}_mcfull.csv",
    }

    # rows cho từng window
    rows_by_window: dict[str, list[dict]] = {
        "win_1q": [],
        "win_1y": [],
        "win_full": [],
    }

    n_ok_any = 0
    n_skip_short = 0
    n_err = 0

    # ===== 3. LOOP QUA TỪNG MÃ & CHẠY MONTE CARLO =====
    for i, ticker in enumerate(all_tickers, start=1):
        df_t = df_price[df_price["Ticker"] == ticker].copy()
        if df_t.empty:
            continue

        # Nếu ticker nào không đủ log_return thì bỏ qua
        n_log = df_t["log_return"].dropna().shape[0]
        if n_log < min_log_return_points:
            n_skip_short += 1
            if i % log_every == 0 or i == 1:
                print(
                    f"[{i}/{n_total}] {ticker}: skip (log_return points={n_log} < {min_log_return_points})"
                )
            continue

        current_price = float(df_t[price_col].iloc[-1])
        last_date = df_t["Date"].max()

        if i % log_every == 0 or i == 1:
            print(
                f"[{i}/{n_total}] Đang xử lý ticker: {ticker} | "
                f"n_log={n_log}, current_price={current_price:.2f}"
            )

        base_row: dict = {
            "Ticker": ticker,
            "current_price": current_price,
            "last_date": last_date,
        }

        has_any_window = False

        # ---- Chạy MC cho từng window (1Q, 1Y, FULL) ----
        for j, (wkey, est_window) in enumerate(window_configs.items()):
            try:
                mc_res = mc_drawdown_quantiles_1q_2q(
                    current_price=current_price,
                    df_t=df_t,          # df_t đã có cột log_return
                    days_1q=days_1q,
                    days_2q=days_2q,
                    # 👇 mô phỏng 40k path (hoặc override ở tham số hàm run_mc_risk_all)
                    n_sim=n_sim,
                    seed=42 + j,        # đổi seed nhẹ giữa các window
                    est_window=est_window,
                    min_window=min_log_return_points,
                    # batching & dtype dùng mặc định trong monte_carlo (float32 + batch)
                )
            except ValueError as e:
                n_err += 1
                print(f"  [WARN] {ticker}: window={wkey} -> {e}")
                continue

            # row riêng cho window này
            row_win = dict(base_row)
            row_win.update(mc_res)   # giữ nguyên tên cột: mc_dd_q05_1q, ...
            rows_by_window[wkey].append(row_win)
            has_any_window = True

        if has_any_window:
            n_ok_any += 1

    # ===== 4. Ghi ra 3 file kết quả (1Q / 1Y / FULL) =====
    print("\n===========================================")
    print(f"✅ Tổng số mã xử lý được ít nhất 1 window : {n_ok_any}")
    print(f"   Số mã skip do thiếu data log_return    : {n_skip_short}")
    print(f"   Số lỗi khác (mu/sigma, v.v.)           : {n_err}")
    print("===========================================\n")

    for wkey, rows in rows_by_window.items():
        if not rows:
            print(f"⚠️ Window {wkey}: không có mã nào đủ điều kiện, không ghi file.")
            continue

        df_out = pd.DataFrame(rows)
        out_path = output_paths[wkey]
        df_out.to_csv(out_path, index=False)

        print(f"✅ Đã ghi file Monte Carlo risk cho {wkey}: {out_path} (shape={df_out.shape})")


if __name__ == "__main__":
    run_mc_risk_all()
