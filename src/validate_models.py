# src/validate_models.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib

from .corporate_actions import adjust_all_tickers
from .features import add_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

ALPHAS = [0.05, 0.03, 0.01, 0.001]
HORIZONS = ["1q", "2q"]  # tương ứng dd_1q, dd_2q


# =========================
# 0. Build feature df + TARGET (dd_1q, dd_2q)
# =========================

def build_feature_df_with_targets(
    days_1q: int = 63,
    days_2q: int = 126,
) -> pd.DataFrame:
    """
    Đọc cleaned_prices.csv, adjust corporate actions, add_features,
    rồi tính thêm 2 cột target:
        - dd_1q: drawdown tệ nhất trong ~1 quý tiếp theo
        - dd_2q: drawdown tệ nhất trong ~2 quý tiếp theo

    dd = (min_future_price / current_price) - 1  (âm, vd -0.35 = -35%).
    """
    data_path = DATA_DIR / "cleaned_prices.csv"
    df_raw = pd.read_csv(data_path)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # 1) Adjust corporate actions
    df_adj = adjust_all_tickers(df_raw, ratio_down=0.7, ratio_up=1.5)

    # 2) Thêm features
    df_feat = add_features(df_adj.copy())

    # 3) Tính dd_1q, dd_2q cho từng ticker trên cột Close_adj (nếu có) hoặc Close
    price_col = "Close_adj" if "Close_adj" in df_feat.columns else "Close"
    if price_col not in df_feat.columns:
        raise ValueError(f"Thiếu cột giá {price_col} trong df_feat")

    def _add_dd_per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").reset_index(drop=True)
        prices = g[price_col].astype(float).values
        n = len(prices)

        dd_1q = np.full(n, np.nan, dtype=float)
        dd_2q = np.full(n, np.nan, dtype=float)

        for i in range(n):
            p0 = prices[i]
            if not np.isfinite(p0) or p0 <= 0:
                continue

            # 1Q: min giá trong (i+1 .. i+days_1q)
            j1_end = min(n, i + 1 + days_1q)
            if j1_end > i + 1:
                future_1q = prices[i + 1 : j1_end]
                if future_1q.size > 0:
                    dd_1q[i] = future_1q.min() / p0 - 1.0

            # 2Q: min giá trong (i+1 .. i+days_2q)
            j2_end = min(n, i + 1 + days_2q)
            if j2_end > i + 1:
                future_2q = prices[i + 1 : j2_end]
                if future_2q.size > 0:
                    dd_2q[i] = future_2q.min() / p0 - 1.0

        g["dd_1q"] = dd_1q
        g["dd_2q"] = dd_2q
        return g

    df_feat = (
        df_feat
        .groupby("Ticker", group_keys=False)
        .apply(_add_dd_per_ticker)
        .reset_index(drop=True)
    )

    return df_feat


def get_feature_cols(df_feat: pd.DataFrame) -> List[str]:
    """
    Danh sách feature để đưa vào ML:
      - lấy tất cả cột numeric
      - loại bỏ Ticker, Date, dd_1q, dd_2q (targets)
    """
    drop_cols = {"Ticker", "Date", "dd_1q", "dd_2q"}
    feature_cols = [
        c
        for c in df_feat.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df_feat[c])
    ]
    return feature_cols


# =========================
# 1. Metric helpers
# =========================

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Quantile (pinball) loss cho quantile alpha.
    Dùng đúng chuẩn để đánh giá quantile regression.
    """
    diff = y_true - y_pred
    loss = np.where(diff >= 0, alpha * diff, (1 - alpha) * -diff)
    return float(np.mean(loss))


def coverage_violation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coverage (tỷ lệ vi phạm tail) cho drawdown:
      - dd là số âm (rớt giá).
      - Vi phạm khi thực tế tệ hơn dự báo: y_true < y_pred
        (vd dự báo dd = -0.30, thực tế dd = -0.45 => vi phạm).
    Với quantile alpha (vd 5%), coverage nên ≈ alpha.
    """
    violations = (y_true < y_pred)
    return float(np.mean(violations))


# =========================
# 2. Load models cho từng loại
# =========================

def load_quantile_models(model_type: str) -> Dict[str, Dict[float, object]]:
    """
    Load các mô hình quantile cho một model_type:
      - model_type: "gb", "xgb", "lgbm"
      - horizon: "1q", "2q"
      - alpha: 0.05, 0.03, 0.02, 0.001

    Trả về: models[horizon][alpha] = model_object
    """
    prefix_map = {
        "gb": "risk_gb",
        "xgb": "risk_xgb",
        "lgbm": "risk_lgbm",
    }
    if model_type not in prefix_map:
        raise ValueError(f"model_type không hợp lệ: {model_type}")

    prefix = prefix_map[model_type]

    models: Dict[str, Dict[float, object]] = {}
    for h in HORIZONS:
        models[h] = {}
        for alpha in ALPHAS:
            q_tag = int(alpha * 1000)  # 0.05 -> 50, 0.001 -> 1
            model_path = MODEL_DIR / f"{prefix}_dd_{h}_q{q_tag}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Thiếu file model: {model_path} "
                    f"(model_type={model_type}, horizon={h}, alpha={alpha})"
                )
            models[h][alpha] = joblib.load(model_path)
    return models


# =========================
# 3. Build validation set (time-series split per ticker)
# =========================

def build_valid_set(
    df_feat: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    valid_ratio: float = 0.2,
    min_len_per_ticker: int = 50,
) -> pd.DataFrame:
    """
    Tạo tập validation theo time-series split per ticker:
      - sort theo (Ticker, Date)
      - với mỗi ticker:
          + nếu số mẫu < min_len_per_ticker: bỏ qua (không dùng eval)
          + ngược lại: lấy khoảng valid_ratio (vd 20%) cuối cùng làm valid.

    Chỉ giữ các dòng không NaN cả target_col lẫn feature_cols.
    """
    df = df_feat.dropna(subset=[target_col] + feature_cols).copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    valid_rows = []
    tickers = sorted(df["Ticker"].dropna().unique().tolist())

    for ticker in tickers:
        g = df[df["Ticker"] == ticker].copy()
        n = len(g)
        if n < min_len_per_ticker:
            continue

        n_valid = max(int(n * valid_ratio), 1)
        g_valid = g.iloc[-n_valid:]
        valid_rows.append(g_valid)

    if not valid_rows:
        return pd.DataFrame(columns=df.columns)

    df_valid = pd.concat(valid_rows, ignore_index=True)
    return df_valid


# =========================
# 4. Đánh giá 1 model_type
# =========================

def evaluate_one_model_type(
    model_type: str,
    df_feat: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Đánh giá 1 model_type (gb / xgb / lgbm) trên cả 2 horizon (1q, 2q)
    bằng pinball loss + coverage.

    Trả về DataFrame có các cột:
      - model_type
      - horizon (1q / 2q)
      - alpha
      - pinball_loss
      - coverage
      - n_valid
    """
    print(f"\n================ MODEL TYPE: {model_type.upper()} ================")

    try:
        models = load_quantile_models(model_type)
    except FileNotFoundError as e:
        print(f"⚠️  Bỏ qua {model_type} vì thiếu model file:\n   {e}")
        return pd.DataFrame(
            columns=["model_type", "horizon", "alpha", "pinball_loss", "coverage", "n_valid"]
        )

    xgb_module = None
    if model_type == "xgb":
        import xgboost as xgb  # type: ignore
        xgb_module = xgb

    rows_metric: List[dict] = []

    for h in HORIZONS:
        target_col = f"dd_{h}"  # dd_1q hoặc dd_2q
        if target_col not in df_feat.columns:
            print(f"⚠️  Bỏ qua horizon={h} vì thiếu cột target {target_col}")
            continue

        df_valid = build_valid_set(
            df_feat=df_feat,
            feature_cols=feature_cols,
            target_col=target_col,
            valid_ratio=0.2,
            min_len_per_ticker=50,
        )

        if df_valid.empty:
            print(f"⚠️  Validation set rỗng cho horizon={h}, bỏ qua.")
            continue

        X_valid = df_valid[feature_cols].values
        y_valid = df_valid[target_col].values.astype(float)
        n_valid = len(df_valid)

        print(
            f"\n--- Horizon: {h} | target={target_col} | "
            f"số mẫu valid: {n_valid}"
        )

        # DMatrix 1 lần cho XGB
        dmatrix = None
        if model_type == "xgb" and xgb_module is not None:
            dmatrix = xgb_module.DMatrix(X_valid, feature_names=feature_cols)

        for alpha in ALPHAS:
            model = models[h][alpha]

            if model_type == "xgb":
                y_pred = model.predict(dmatrix)  # type: ignore
                y_pred = np.asarray(y_pred).reshape(-1)
            else:
                y_pred = model.predict(X_valid)
                y_pred = np.asarray(y_pred).reshape(-1)

            pb = pinball_loss(y_valid, y_pred, alpha=alpha)
            cov = coverage_violation(y_valid, y_pred)

            print(
                f"  alpha={alpha:0.3f} | "
                f"pinball_loss={pb: .6f} | "
                f"coverage={cov: .4f} (kỳ vọng ≈ {alpha:0.3f})"
            )

            rows_metric.append(
                {
                    "model_type": model_type,
                    "horizon": h,
                    "alpha": alpha,
                    "pinball_loss": pb,
                    "coverage": cov,
                    "n_valid": n_valid,
                }
            )

    if not rows_metric:
        return pd.DataFrame(
            columns=["model_type", "horizon", "alpha", "pinball_loss", "coverage", "n_valid"]
        )

    return pd.DataFrame(rows_metric)


# =========================
# 5. Main
# =========================

def main():
    print("===========================================")
    print("  VALIDATE ML QUANTILE MODELS (GB / LGBM / XGB)")
    print("  Metrics: Pinball Loss + Coverage (tail)")
    print("===========================================\n")

    # 1) Build feature df có target
    df_feat = build_feature_df_with_targets(days_1q=63, days_2q=126)
    feature_cols = get_feature_cols(df_feat)

    print(f"✅ Loaded feature df with targets: shape={df_feat.shape}")
    print(f"✅ Số lượng feature dùng cho ML risk: {len(feature_cols)}")
    print(f"   Feature cols: {feature_cols}\n")

    # 2) Evaluate cho từng model_type
    all_metrics = []

    for model_type in ["gb", "lgbm", "xgb"]:
        m_df = evaluate_one_model_type(
            model_type=model_type,
            df_feat=df_feat,
            feature_cols=feature_cols,
        )
        if not m_df.empty:
            all_metrics.append(m_df)

    if not all_metrics:
        print("❌ Không có model nào được đánh giá (thiếu file?).")
        return

    metrics_df = pd.concat(all_metrics, ignore_index=True)

    # 3) In bảng tổng hợp
    print("\n===========================================")
    print("  TỔNG HỢP METRIC THEO MODEL / HORIZON / ALPHA")
    print("===========================================\n")

    metrics_df = metrics_df.sort_values(
        by=["horizon", "alpha", "model_type"]
    ).reset_index(drop=True)
    print(metrics_df.to_string(index=False))

    # 4) Ghi ra CSV để soi thêm
    out_csv = DATA_DIR / "model_validation_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã ghi metrics ra: {out_csv}")



if __name__ == "__main__":
    main()
