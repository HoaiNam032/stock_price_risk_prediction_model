# src/predict_risk_all.py
import os
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
try:
    import xgboost as xgb
except ImportError:
    xgb = None


from .corporate_actions import adjust_all_tickers
from .features import add_features
from .monte_carlo import estimate_mu_sigma

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# LTV tối đa tại đáy stress (ví dụ 0.5 = 50%)
LTV_FUTURE_MAX_DEFAULT = 0.5


# =========================
# 1. Load ML quantile models (GB / XGB / LGBM)
# =========================
def load_quantile_models(model_type: str = "gb") -> Dict[str, Dict[float, object]]:
    """
    Load các mô hình quantile:
      - horizon: "1q", "2q"
      - alpha: 0.05, 0.03, 0.01, 0.001

    model_type:
      - "gb"   -> risk_gb_dd_{h}_q{q_tag}.pkl
      - "xgb"  -> risk_xgb_dd_{h}_q{q_tag}.pkl
      - "lgbm" -> risk_lgbm_dd_{h}_q{q_tag}.pkl

    Trả về dict: models[horizon][alpha] = model
    """
    # chuẩn hóa input
    model_type = model_type.lower()
    if model_type not in {"gb", "xgb", "lgbm"}:
        raise ValueError(f"model_type không hợp lệ: {model_type}")

    alphas = [0.05, 0.03, 0.01, 0.001]
    horizons = ["1q", "2q"]

    prefix = f"risk_{model_type}"  # risk_gb, risk_xgb, risk_lgbm

    models: Dict[str, Dict[float, object]] = {}
    for h in horizons:
        models[h] = {}
        for alpha in alphas:
            q_tag = int(alpha * 1000)  # 0.05 -> 50, 0.001 -> 1
            model_path = MODEL_DIR / f"{prefix}_dd_{h}_q{q_tag}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy model: {model_path} (model_type={model_type})"
                )
            models[h][alpha] = joblib.load(model_path)
    return models

def predict_dd_one(model, X_last, model_type: str) -> float:
    """
    Dự đoán 1 giá trị drawdown cho X_last (shape: 1 x n_features)
    - GB / LGBM: model là sklearn-like, nhận numpy array trực tiếp.
    - XGB: model là xgboost.Booster, phải bọc vào DMatrix.
    """
    if model_type == "xgb":
        if xgb is None:
            raise RuntimeError("xgboost chưa được import.")
        dmat = xgb.DMatrix(X_last)
        return float(model.predict(dmat)[0])
    else:
        # GradientBoosting, LightGBM (sklearn API)
        return float(model.predict(X_last)[0])

# =========================
# 2. Monte Carlo helpers
# =========================
def mc_min_price_distribution(
    current_price: float,
    mu: float,
    sigma: float,
    horizon_days: int,
    n_sim: int = 2000,
) -> np.ndarray:
    """
    Monte Carlo: mô phỏng đường giá theo log-return N(mu, sigma^2)
    Trả về phân phối MIN PRICE của từng path.
    """
    if current_price <= 0 or np.isnan(mu) or np.isnan(sigma):
        return np.array([])

    mins = []
    for _ in range(n_sim):
        daily_ret = np.random.normal(loc=mu, scale=sigma, size=horizon_days)
        price_path = current_price * np.exp(np.cumsum(daily_ret))
        mins.append(price_path.min())
    return np.array(mins)


# =========================
# 3. Build feature df
# =========================
def build_feature_df() -> pd.DataFrame:
    """
    Đọc cleaned_prices.csv, adjust corporate actions, add_features.
    """
    data_path = DATA_DIR / "cleaned_prices.csv"
    df_raw = pd.read_csv(data_path)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Adjust toàn bộ (chống split / thưởng cổ tức, v.v.)
    df_adj = adjust_all_tickers(df_raw, ratio_down=0.7, ratio_up=1.5)

    # Thêm features (EMA, volatility, lag returns, target dd_1q, dd_2q, ...)
    df_feat = add_features(df_adj.copy())
    return df_feat


def get_feature_cols(df_feat: pd.DataFrame) -> List[str]:
    """
    Chọn danh sách feature dùng cho ML risk.
    Lấy tất cả cột numeric, bỏ các cột id / target.
    """
    drop_cols = {
        "Ticker",
        "Date",
        "dd_1q",
        "dd_2q",
    }
    feature_cols = [
        c
        for c in df_feat.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df_feat[c])
    ]
    return feature_cols


# =========================
# 4. Load rating (A/B/C/D)
# =========================
def load_ratings(path: Path = DATA_DIR / "ticker_ratings.csv") -> pd.DataFrame:
    """
    Đọc file rating: yêu cầu cột Ticker, Rating (A/B/C/D).
    Nếu không có file thì trả df rỗng.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

    cols_lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols_lower)

    if "ticker" not in df.columns or "rating" not in df.columns:
        return pd.DataFrame()

    df["Ticker"] = df["ticker"].astype(str).str.strip()
    df["Rating"] = df["rating"].astype(str).str.strip().str.upper()
    return df[["Ticker", "Rating"]]


# =========================
# 5. Main pipeline (cho 1 loại model)
# =========================
def run_predict_for_all(model_type: str = "gb") -> None:
    """
    Chạy full pipeline risk cho tất cả tickers với 1 loại model:
      - model_type: "gb" / "xgb" / "lgbm"
    Xuất file:
      data/risk_all_tickers_{model_type}.csv
    """
    print("===========================================")
    print(f"  PREDICT RISK FOR ALL TICKERS ({model_type.upper()} ML + MC)")
    print("===========================================\n")

    # 5.0 Build feature df
    df_feat = build_feature_df()
    feature_cols = get_feature_cols(df_feat)
    print(f"Kích thước df_feat: {df_feat.shape}")
    print(f"Số lượng feature dùng cho ML risk: {len(feature_cols)}\n")

    # 5.1 Load các mô hình quantile ML
    models_q = load_quantile_models(model_type=model_type)
    print(f"✅ Đã load multi-quantile {model_type.upper()} models.\n")

    # 5.2 Load rating A/B/C/D (nếu có)
    ratings_df = load_ratings()
    if ratings_df.empty:
        print("⚠️ Không tìm thấy file ticker_ratings.csv hoặc thiếu cột. Sẽ không có rating_group.")
    else:
        print(f"✅ Đã load {len(ratings_df)} dòng rating.\n")

    tickers = sorted(df_feat["Ticker"].dropna().unique().tolist())
    print(f"Tìm thấy {len(tickers)} mã trong dữ liệu.\n")

    rows: List[Dict] = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] Đang xử lý ticker: {ticker}")
        df_t = df_feat[df_feat["Ticker"] == ticker].copy()
        if df_t.empty:
            print("  -> Bỏ qua (không có dữ liệu).")
            continue

        df_t = df_t.sort_values("Date")
        df_t = df_t.dropna(subset=feature_cols).copy()
        if df_t.empty:
            print("  -> Bỏ qua (feature bị NaN hết).")
            continue

        # Giá hiện tại: dùng Close_adj nếu có, không thì Close
        price_col = "Close_adj" if "Close_adj" in df_t.columns else "Close"
        if price_col not in df_t.columns:
            print(f"  -> Bỏ qua (không có cột {price_col}).")
            continue

        last_row = df_t.iloc[-1]
        current_price = float(last_row[price_col])
        last_date = last_row["Date"]

        X_last = last_row[feature_cols].values.reshape(1, -1)

        # ====== 5.3 ML quantile drawdowns (4 mức) cho 1Q / 2Q ======
        # A: 5%, B: 3%, C: 2%, D: 0.1%
        dd_ml: Dict[str, float] = {}

        def set_dd(h: str, alpha: float, dd_val: float):
            """
            h: "1q" hoặc "2q"
            alpha: 0.05, 0.03, 0.02, 0.001 -> map sang dd_q05_*, dd_q03_*, dd_q02_*, dd_q001_*
            """
            if alpha == 0.05:
                dd_ml[f"dd_q05_{h}"] = dd_val
            elif alpha == 0.03:
                dd_ml[f"dd_q03_{h}"] = dd_val
            elif alpha == 0.01:
                dd_ml[f"dd_q01_{h}"] = dd_val
            elif alpha == 0.001:
                dd_ml[f"dd_q001_{h}"] = dd_val

        for h in ["1q", "2q"]:
            for alpha in [0.05, 0.03, 0.01, 0.001]:
                m = models_q[h][alpha]
                dd_hat = predict_dd_one(m, X_last, model_type)  # dùng helper mới
                set_dd(h, alpha, dd_hat)

        # ====== 5.4 Monte Carlo: min price tails (5%, 3%, 1%, 0.1%) ======
        try:
            mu, sigma = estimate_mu_sigma(df_t, window=60)
        except Exception:
            mu = sigma = np.nan

        mc_min_p5_1q = mc_min_p3_1q = mc_min_p1_1q = mc_min_p01_1q = np.nan
        mc_min_p5_2q = mc_min_p3_2q = mc_min_p1_2q = mc_min_p01_2q = np.nan

        if not np.isnan(mu) and not np.isnan(sigma):
            for h_tag, days in [("1q", 63), ("2q", 126)]:
                sims_min = mc_min_price_distribution(
                    current_price=current_price,
                    mu=mu,
                    sigma=sigma,
                    horizon_days=days,
                    n_sim=2000,
                )

                if sims_min.size > 0:
                    p5, p3, p1, p01 = np.percentile(sims_min, [5, 3, 1, 0.1])
                    p5 = float(p5)
                    p3 = float(p3)
                    p1 = float(p1)
                    p01 = float(p01)
                else:
                    p5 = p3 = p1 = p01 = np.nan

                if h_tag == "1q":
                    mc_min_p5_1q, mc_min_p3_1q, mc_min_p1_1q, mc_min_p01_1q = (
                        p5,
                        p3,
                        p1,
                        p01,
                    )
                else:
                    mc_min_p5_2q, mc_min_p3_2q, mc_min_p1_2q, mc_min_p01_2q = (
                        p5,
                        p3,
                        p1,
                        p01,
                    )

        # ====== 5.5 Rating group (A/B/C/D) ======
        rating_group = None
        if not ratings_df.empty:
            r_row = ratings_df[ratings_df["Ticker"] == ticker]
            if not r_row.empty:
                rating_group = str(r_row.iloc[0]["Rating"]).upper().strip()

        # ====== 5.6 Giá ML VaR5 & ML min cho 1Q và 2Q ======
        ml_var5_price_1q = np.nan
        ml_min_price_1q = np.nan
        ml_var5_price_2q = np.nan
        ml_min_price_2q = np.nan

        # 1Q
        dd_vals_1q = []
        for k in ["dd_q05_1q", "dd_q03_1q", "dd_q01_1q", "dd_q001_1q"]:
            if k in dd_ml and not np.isnan(dd_ml[k]):
                dd_vals_1q.append(dd_ml[k])

        if "dd_q05_1q" in dd_ml and not np.isnan(dd_ml["dd_q05_1q"]):
            ml_var5_price_1q = current_price * (1.0 + dd_ml["dd_q05_1q"])

        if dd_vals_1q:
            dd_min_1q = min(dd_vals_1q)  # tail sâu nhất trong 4 mức ML 1Q
            ml_min_price_1q = current_price * (1.0 + dd_min_1q)

        # 2Q
        dd_vals_2q = []
        for k in ["dd_q05_2q", "dd_q03_2q", "dd_q01_2q", "dd_q001_2q"]:
            if k in dd_ml and not np.isnan(dd_ml[k]):
                dd_vals_2q.append(dd_ml[k])

        if "dd_q05_2q" in dd_ml and not np.isnan(dd_ml["dd_q05_2q"]):
            ml_var5_price_2q = current_price * (1.0 + dd_ml["dd_q05_2q"])

        if dd_vals_2q:
            dd_min_2q = min(dd_vals_2q)  # tail sâu nhất trong 4 mức ML 2Q
            ml_min_price_2q = current_price * (1.0 + dd_min_2q)

        # ====== 5.7 Giá huề vốn (loan-per-share) theo rating cho 1Q và 2Q ======
        breakeven_price_rating_1q = np.nan
        breakeven_price_rating_2q = np.nan

        if rating_group is not None:
            # Map rating -> cột dd ML 1Q
            rating_dd_map_1q = {
                "A": "dd_q05_1q",
                "B": "dd_q03_1q",
                "C": "dd_q01_1q",
                "D": "dd_q001_1q",
            }
            col_dd_1q = rating_dd_map_1q.get(rating_group)
            if col_dd_1q in dd_ml and not np.isnan(dd_ml.get(col_dd_1q, np.nan)):
                dd_r_1q = dd_ml[col_dd_1q]
                p_stress_r_1q = current_price * (1.0 + dd_r_1q)
                ltv_r_1q = LTV_FUTURE_MAX_DEFAULT * (p_stress_r_1q / current_price)
                ltv_r_1q = float(np.clip(ltv_r_1q, 0.0, 1.0))
                breakeven_price_rating_1q = current_price * ltv_r_1q

            # Map rating -> cột dd ML 2Q
            rating_dd_map_2q = {
                "A": "dd_q05_2q",
                "B": "dd_q03_2q",
                "C": "dd_q01_2q",
                "D": "dd_q001_2q",
            }
            col_dd_2q = rating_dd_map_2q.get(rating_group)
            if col_dd_2q in dd_ml and not np.isnan(dd_ml.get(col_dd_2q, np.nan)):
                dd_r_2q = dd_ml[col_dd_2q]
                p_stress_r_2q = current_price * (1.0 + dd_r_2q)
                ltv_r_2q = LTV_FUTURE_MAX_DEFAULT * (p_stress_r_2q / current_price)
                ltv_r_2q = float(np.clip(ltv_r_2q, 0.0, 1.0))
                breakeven_price_rating_2q = current_price * ltv_r_2q

        # ====== 5.8 Gom row output ======
        row_out: Dict = {
            "Ticker": ticker,
            "last_date": last_date.date(),
            "current_price": current_price,

            # Group rating
            "rating_group": rating_group,

            # Giá ML (1Q & 2Q)
            "ml_var5_price_1q": ml_var5_price_1q,
            "ml_min_price_1q": ml_min_price_1q,
            "ml_var5_price_2q": ml_var5_price_2q,
            "ml_min_price_2q": ml_min_price_2q,

            # Giá huề vốn theo rating (loan per share, với LTV_future_max mặc định)
            "breakeven_price_rating_1q": breakeven_price_rating_1q,
            "breakeven_price_rating_2q": breakeven_price_rating_2q,

            # Monte Carlo min price tails 1Q
            "mc_min_p5_1q": mc_min_p5_1q,
            "mc_min_p3_1q": mc_min_p3_1q,
            "mc_min_p1_1q": mc_min_p1_1q,
            "mc_min_p01_1q": mc_min_p01_1q,

            # Monte Carlo min price tails 2Q
            "mc_min_p5_2q": mc_min_p5_2q,
            "mc_min_p3_2q": mc_min_p3_2q,
            "mc_min_p1_2q": mc_min_p1_2q,
            "mc_min_p01_2q": mc_min_p01_2q,
        }

        # Thêm toàn bộ dd ML (4 mức × 2 horizon)
        row_out.update(dd_ml)

        rows.append(row_out)

    if not rows:
        print("❌ Không ticker nào được xử lý.")
        return

    out_df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = DATA_DIR / f"risk_all_tickers_{model_type}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n===========================================")
    print(f"✅ Đã ghi file: {out_path}")
    print("===========================================")


def main():
    parser = argparse.ArgumentParser(
        description="Predict risk for all tickers with different ML models (GB / XGB / LGBM)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gb",
        choices=["gb", "xgb", "lgbm"],
        help="Loại model dùng để dự đoán risk: gb / xgb / lgbm (default: gb)",
    )

    args = parser.parse_args()
    run_predict_for_all(model_type=args.model_type)


if __name__ == "__main__":
    main()
