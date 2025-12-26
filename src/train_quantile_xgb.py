# src/train_quantile_xgb.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from .corporate_actions import adjust_all_tickers
from .features import add_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

ALPHAS = [0.05, 0.03, 0.01, 0.001]
HORIZON_DAYS = {
    "1q": 63,   # ~3 tháng
    "2q": 126,  # ~6 tháng
}


# ======================================================
# 1. Tính forward min & drawdown target (dd_1q, dd_2q)
# ======================================================
def compute_forward_min(series: pd.Series, horizon: int) -> pd.Series:
    """
    future_min[t] = min(price[t+1 ... t+horizon])
    Dùng trick đảo ngược + rolling để vectorize.
    """
    s_rev = series.iloc[::-1]
    roll_min = s_rev.rolling(window=horizon, min_periods=1).min()
    future_min = roll_min.iloc[::-1].shift(-1)
    return future_min


def add_drawdown_targets(
    df: pd.DataFrame,
    price_col: str = "Close_adj",
    horizons: Dict[str, int] = HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Thêm cột:
      - dd_1q: drawdown (%) trong 63 ngày tới
      - dd_2q: drawdown (%) trong 126 ngày tới
    dd_h = future_min / price_today - 1  (<= 0, vì future_min là giá thấp nhất)
    """
    if price_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột giá '{price_col}' trong df_feat")

    df = df.sort_values(["Ticker", "Date"]).copy()

    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        px = g[price_col]
        for tag, days in horizons.items():
            fmin = compute_forward_min(px, days)
            dd = fmin / px - 1.0
            g[f"dd_{tag}"] = dd
        return g

    df_out = (
        df
        .groupby("Ticker", group_keys=False)
        .apply(_per_ticker)
    )

    return df_out


# ======================================================
# 2. Build feature df (giống LightGBM)
# ======================================================
def build_feature_df() -> pd.DataFrame:
    """
    Đọc cleaned_prices.csv, adjust corporate actions, add_features.
    Sau đó tự tính dd_1q, dd_2q cho training.
    """
    data_path = DATA_DIR / "cleaned_prices.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {data_path}")

    df_raw = pd.read_csv(data_path)
    if "Date" not in df_raw.columns or "Ticker" not in df_raw.columns:
        raise ValueError("cleaned_prices.csv phải có cột 'Date' và 'Ticker'")

    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
    df_raw = df_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Adjust toàn bộ corporate actions
    df_adj = adjust_all_tickers(df_raw, ratio_down=0.7, ratio_up=1.5)

    # Thêm feature kỹ thuật
    df_feat = add_features(df_adj.copy())

    # Thêm target dd_1q, dd_2q
    price_col = "Close_adj" if "Close_adj" in df_feat.columns else "Close"
    df_feat = add_drawdown_targets(df_feat, price_col=price_col)

    return df_feat


def get_feature_cols(df_feat: pd.DataFrame) -> List[str]:
    """
    Chọn danh sách feature numeric dùng cho ML risk.
    Bỏ các cột id / target / Date / Ticker.
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


# ======================================================
# 3. Pinball loss & custom quantile objective cho XGBoost
# ======================================================
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Pinball loss cho quantile regression.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = y_true - y_pred
    loss = np.where(diff >= 0, alpha * diff, (1 - alpha) * (-diff))
    return float(loss.mean())


def make_quantile_obj(alpha: float):
    """
    Custom objective cho XGBoost để tối ưu pinball loss.
    grad = dL/dy_hat, hess ~ 1 (xấp xỉ).
    """
    def quantile_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        labels = dtrain.get_label()
        diff = labels - preds
        # diff >= 0: y >= y_hat -> dL/dy_hat = -alpha
        # diff < 0: y <  y_hat -> dL/dy_hat = 1 - alpha
        grad = np.where(diff >= 0, -alpha, 1.0 - alpha)
        hess = np.ones_like(grad)  # hessian xấp xỉ hằng số
        return grad, hess
    return quantile_obj


def make_quantile_eval(alpha: float):
    """
    Eval metric cho XGBoost: pinball loss.
    """
    def quantile_eval(preds: np.ndarray, dtrain: xgb.DMatrix):
        labels = dtrain.get_label()
        loss = pinball_loss(labels, preds, alpha=alpha)
        return "pinball", loss
    return quantile_eval


# ======================================================
# 4. Train XGB quantile cho 1 horizon + 1 alpha
# ======================================================
def train_one_model(
    df_feat: pd.DataFrame,
    feature_cols: List[str],
    horizon_tag: str,
    alpha: float,
    valid_frac: float = 0.2,
) -> Tuple[xgb.Booster, float]:
    """
    Train 1 model quantile XGBoost cho:
      - horizon_tag: "1q" hoặc "2q"
      - alpha: 0.05, 0.03, 0.02, 0.001
    Trả về: (booster, pinball_loss_valid)
    """
    target_col = f"dd_{horizon_tag}"
    if target_col not in df_feat.columns:
        raise ValueError(f"Thiếu target {target_col} trong df_feat")

    df = df_feat.dropna(subset=[target_col] + feature_cols).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    n = len(df)
    if n < 1000:
        raise ValueError(f"Dữ liệu quá ít cho horizon {horizon_tag}: n={n}")

    n_valid = int(n * valid_frac)
    n_train = n - n_valid
    if n_valid < 100:
        raise ValueError(f"Validation set quá nhỏ (n_valid={n_valid})")

    df_train = df.iloc[:n_train]
    df_valid = df.iloc[n_train:]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[target_col].values

    print(f"    -> Số mẫu train: {n_train:,}, valid: {n_valid:,}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "max_depth": 5,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
        "objective": "reg:squarederror",  # sẽ override bằng custom obj
        "eval_metric": "rmse",            # placeholder
    }

    watchlist = [(dtrain, "train"), (dvalid, "valid")]
    obj = make_quantile_obj(alpha)
    feval = make_quantile_eval(alpha)

    t0 = time.time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=watchlist,
        obj=obj,
        feval=feval,
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    t1 = time.time()

    # Đánh giá lại pinball loss trên valid
    y_pred_valid = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    q_loss = pinball_loss(y_valid, y_pred_valid, alpha=alpha)

    print(
        f"    -> Done alpha={alpha:.3f} | best_iter={booster.best_iteration} "
        f"| pinball_loss_valid={q_loss:.6f} | time={t1 - t0:.1f}s"
    )

    return booster, q_loss


# ======================================================
# 5. MAIN
# ======================================================
def main():
    print("===========================================")
    print("  TRAIN QUANTILE MODELS - XGBoost")
    print("  (dd_1q, dd_2q với alphas 5%,3%,1%,0.1%)")
    print("===========================================\n")

    # 5.1 Build feature df + targets
    df_feat = build_feature_df()
    print(f"✅ Loaded feature df: shape={df_feat.shape}")

    feature_cols = get_feature_cols(df_feat)
    print(f"✅ Số lượng feature numeric dùng cho ML risk: {len(feature_cols)}\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for h_tag in ["1q", "2q"]:
        target_col = f"dd_{h_tag}"
        print(f"\n================ Horizon: {h_tag} (target={target_col}) ================")

        if target_col not in df_feat.columns:
            print(f"❌ Thiếu cột target {target_col}, bỏ qua horizon này.")
            continue

        for i, alpha in enumerate(ALPHAS, start=1):
            q_tag = int(alpha * 1000)  # 0.05 -> 50, 0.001 -> 1
            print(f"\n  [{i}/{len(ALPHAS)}] Train XGB quantile alpha={alpha:.3f} (q{q_tag})")

            try:
                booster, q_loss = train_one_model(
                    df_feat=df_feat,
                    feature_cols=feature_cols,
                    horizon_tag=h_tag,
                    alpha=alpha,
                    valid_frac=0.2,
                )
            except Exception as e:
                print(f"  ⚠️ Lỗi khi train horizon={h_tag}, alpha={alpha}: {e}")
                continue

            model_path = MODEL_DIR / f"risk_xgb_dd_{h_tag}_q{q_tag}.pkl"
            joblib.dump(booster, model_path)
            print(f"  💾 Saved model to: {model_path}")

    print("\n===========================================")
    print("✅ HOÀN TẤT TRAIN XGBoost QUANTILE MODELS")
    print("===========================================")


if __name__ == "__main__":
    main()
