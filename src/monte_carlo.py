
# from __future__ import annotations

# import math
# from typing import Dict, Tuple, Iterable

# import numpy as np
# import pandas as pd


# # ---------------------------------------------------------
# # 1) Ước lượng mu, sigma từ lịch sử (log_return)
# # ---------------------------------------------------------

# def estimate_mu_sigma(
#     df_t: pd.DataFrame,
#     window: int | None = 60,
#     min_window: int = 10,
# ) -> Tuple[float, float]:
#     """
#     Ước lượng mu, sigma (log-return hàng ngày) từ df_t.

#     - df_t phải có cột 'log_return'
#     - Nếu window là số nguyên: dùng min(window, len(df_t)) phiên gần nhất
#     - Nếu window = None: dùng toàn bộ lịch sử log_return
#     - Nếu dữ liệu < min_window -> raise lỗi
#     """
#     if "log_return" not in df_t.columns:
#         raise ValueError("df_t chưa có cột 'log_return' - hãy tạo trước")

#     df = df_t.dropna(subset=["log_return"]).copy()
#     n = len(df)
#     if n < min_window:
#         raise ValueError(f"Không đủ dữ liệu log_return (n={n}) để ước lượng mu,sigma")

#     if window is None:
#         recent = df["log_return"]
#     else:
#         w = min(window, n)
#         recent = df["log_return"].iloc[-w:]

#     mu = float(recent.mean())
#     sigma = float(recent.std(ddof=1))  # sample std
#     return mu, sigma


# # ---------------------------------------------------------
# # 2) Vectorized Monte Carlo – giá THẤP NHẤT trong horizon
# #    (hỗ trợ batching để tránh OOM khi n_sim rất lớn)
# # ---------------------------------------------------------

# def _simulate_min_prices_batch(
#     current_price: float,
#     horizon_days: int,
#     mu: float,
#     sigma: float,
#     n_batch: int,
#     rng: np.random.Generator,
#     dtype: np.dtype = np.float32,
# ) -> np.ndarray:
#     """
#     Simulate `n_batch` paths (vectorized) và trả về min_prices (n_batch,)
#     - dùng float32 theo mặc định để tiết kiệm RAM
#     - rng: là np.random.Generator đã khởi tạo (deterministic theo seed)
#     """
#     # Generate shocks: shape (n_batch, horizon_days)
#     # Use standard_normal for speed. Cast to dtype early to save memory.
#     z = rng.standard_normal(size=(n_batch, horizon_days)).astype(dtype)

#     # daily log-return according to GBM discrete approximation
#     dt = 1.0
#     drift = np.float32(mu * dt)
#     vol_scale = np.float32(sigma * math.sqrt(dt))

#     daily_log_ret = drift + vol_scale * z  # shape (n_batch, horizon_days)

#     # cumulative log-price paths: log(S0) + cumsum(daily_log_ret)
#     # use float32 cumsum to reduce memory
#     log_s0 = np.float32(np.log(current_price))
#     log_paths = np.cumsum(daily_log_ret, axis=1, dtype=dtype)
#     log_paths += log_s0  # broadcast add

#     # min log-price per path
#     min_log_price = np.min(log_paths, axis=1)

#     # convert back to price
#     min_prices = np.exp(min_log_price.astype(np.float64)).astype(np.float64)
#     # return as float64 for downstream percentile accuracy
#     return min_prices


# def monte_carlo_min_prices(
#     current_price: float,
#     horizon_days: int,
#     mu: float,
#     sigma: float,
#     n_sim: int = 20000,
#     seed: int | None = 42,
#     batch_size: int | None = None,
#     dtype: str = "float32",
# ) -> np.ndarray:
#     """
#     Vectorized Monte Carlo min-prices with optional batching.

#     Parameters
#     ----------
#     - current_price: float
#     - horizon_days: int
#     - mu, sigma: floats (daily)
#     - n_sim: total number of simulated paths
#     - seed: RNG seed
#     - batch_size: nếu None -> tự chọn sao cho vừa vặn (giả sử ~10k-50k)
#     - dtype: 'float32' hoặc 'float64' (mặc định float32 để tiết kiệm RAM)

#     Trả về
#     -------
#     - numpy.ndarray shape (n_sim,) dtype float64 (để percentile chính xác)

#     Ghi chú
#     -------
#     - Hàm này sẽ chia n_sim thành nhiều batch để tránh cấp phát ma trận lớn quá mức.
#     - Sử dụng np.random.Generator để đảm bảo reproducible across batches.
#     """
#     if n_sim <= 0:
#         return np.array([], dtype=np.float64)

#     if dtype not in ("float32", "float64"):
#         raise ValueError("dtype must be 'float32' or 'float64'")

#     dtype_np = np.float32 if dtype == "float32" else np.float64

#     # heuristic default batch size: cố gắng giữ mỗi batch ~ (n_batch * horizon) ~ 5e6 elements
#     # nếu n_sim * horizon_days nhỏ thì chạy 1 batch.
#     if batch_size is None:
#         target_elems = 5_000_000
#         est_batch = max(1, int(target_elems // max(1, horizon_days)))
#         batch_size = min(n_sim, max(1, est_batch))

#     rng = np.random.default_rng(seed)

#     mins: list[np.ndarray] = []
#     sims_left = n_sim
#     while sims_left > 0:
#         cur_batch = min(batch_size, sims_left)
#         mins_batch = _simulate_min_prices_batch(
#             current_price=current_price,
#             horizon_days=horizon_days,
#             mu=mu,
#             sigma=sigma,
#             n_batch=cur_batch,
#             rng=rng,
#             dtype=dtype_np,
#         )
#         mins.append(mins_batch)
#         sims_left -= cur_batch

#     min_prices = np.concatenate(mins, axis=0)
#     return min_prices


# # ---------------------------------------------------------
# # 3) (Tùy chọn) Monte Carlo terminal prices (vectorized)
# # ---------------------------------------------------------

# def monte_carlo_terminal_prices(
#     current_price: float,
#     horizon_days: int,
#     mu: float,
#     sigma: float,
#     n_sim: int = 10000,
#     seed: int | None = 42,
#     dtype: str = "float32",
# ) -> np.ndarray:
#     """
#     Vectorized terminal price simulation (1-step equivalent using normal distribution
#     of total log-return) but implemented in vectorized form for consistency.
#     """
#     if n_sim <= 0:
#         return np.array([], dtype=np.float64)

#     dtype_np = np.float32 if dtype == "float32" else np.float64
#     rng = np.random.default_rng(seed)

#     total_mean = mu * horizon_days
#     total_std = sigma * math.sqrt(horizon_days)

#     z = rng.standard_normal(size=n_sim).astype(dtype_np)
#     total_log = total_mean + total_std * z
#     simulated = current_price * np.exp(total_log.astype(np.float64))
#     return simulated


# # ---------------------------------------------------------
# # 4) Quantiles & helper 1Q/2Q
# # ---------------------------------------------------------

# def mc_drawdown_quantiles_min(
#     current_price: float,
#     horizon_days: int,
#     mu: float,
#     sigma: float,
#     n_sim: int = 20000,
#     seed: int | None = 42,
#     probs: Tuple[float, float, float, float] = (5.0, 3.0, 1.0, 0.1),
#     batch_size: int | None = None,
#     dtype: str = "float32",
# ) -> Dict[str, float]:
#     """
#     Tính các mức worst-case dựa trên min_prices (vectorized + batching).
#     probs: percent values (e.g. 5.0 means 5th percentile)
#     """
#     min_prices = monte_carlo_min_prices(
#         current_price=current_price,
#         horizon_days=horizon_days,
#         mu=mu,
#         sigma=sigma,
#         n_sim=n_sim,
#         seed=seed,
#         batch_size=batch_size,
#         dtype=dtype,
#     )

#     # np.percentile expects percent in [0,100]
#     p5, p3, p1, p01 = np.percentile(min_prices, probs)

#     res = {
#         "mc_price_q05": float(p5),
#         "mc_price_q03": float(p3),
#         "mc_price_q01": float(p1),
#         "mc_price_q001": float(p01),
#         "mc_dd_q05": float(p5 / current_price - 1.0),
#         "mc_dd_q03": float(p3 / current_price - 1.0),
#         "mc_dd_q01": float(p1 / current_price - 1.0),
#         "mc_dd_q001": float(p01 / current_price - 1.0),
#     }
#     return res


# def mc_drawdown_quantiles_1q_2q(
#     current_price: float,
#     df_t: pd.DataFrame,
#     days_1q: int = 66,
#     days_2q: int = 132,
#     n_sim: int = 20000,
#     seed: int | None = 42,
#     est_window: int | None = 60,
#     min_window: int = 10,
#     batch_size: int | None = None,
#     dtype: str = "float32",
# ) -> Dict[str, float]:
#     """
#     Tính full bộ MC drawdown cho 1Q & 2Q (hỗ trợ batching và dtype)
#     """
#     mu, sigma = estimate_mu_sigma(
#         df_t,
#         window=est_window,
#         min_window=min_window,
#     )

#     out: Dict[str, float] = {}

#     res_1q = mc_drawdown_quantiles_min(
#         current_price=current_price,
#         horizon_days=days_1q,
#         mu=mu,
#         sigma=sigma,
#         n_sim=n_sim,
#         seed=seed,
#         batch_size=batch_size,
#         dtype=dtype,
#     )
#     for k, v in res_1q.items():
#         out[f"{k}_1q"] = v

#     res_2q = mc_drawdown_quantiles_min(
#         current_price=current_price,
#         horizon_days=days_2q,
#         mu=mu,
#         sigma=sigma,
#         n_sim=n_sim,
#         seed=None if seed is None else seed + 1,
#         batch_size=batch_size,
#         dtype=dtype,
#     )
#     for k, v in res_2q.items():
#         out[f"{k}_2q"] = v

#     return out


# src/monte_carlo.py
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# =========================================================
# 0) CẤU HÌNH STRESS ĐUÔI RỦI RO
# =========================================================
DRIFT_MODE = "zero"
# Nhân sigma lên 1.x để làm đuôi rủi ro dày hơn 
SIGMA_STRESS_FACTOR = 1  # 1.30 = +30% volatility;
def apply_risk_stress(mu: float, sigma: float) -> Tuple[float, float]:
    """
    Áp dụng stress cho mu, sigma để làm mô phỏng bảo thủ hơn.
    - DRIFT_MODE = "zero": ép mu = 0  (không kỳ vọng tăng trưởng)
    - SIGMA_STRESS_FACTOR > 1: nhân volatility lên để làm đuôi dày hơn
    """
    if DRIFT_MODE == "zero":
        mu_stress = 0.0
    else:
        mu_stress = mu
    sigma_stress = sigma * SIGMA_STRESS_FACTOR
    return mu_stress, sigma_stress


# =========================================================
# 1) Ước lượng mu, sigma từ lịch sử (log_return)
# =========================================================

def estimate_mu_sigma(
    df_t: pd.DataFrame,
    window: int | None = 60,
    min_window: int = 10,
) -> Tuple[float, float]:
    """
    Ước lượng mu, sigma (log-return hàng ngày) từ df_t.

    - df_t phải có cột 'log_return'
    - Nếu window là số nguyên: dùng min(window, len(df_t)) phiên gần nhất
    - Nếu window = None: dùng toàn bộ lịch sử log_return
    - Nếu dữ liệu < min_window -> raise lỗi
    """
    if "log_return" not in df_t.columns:
        raise ValueError("df_t chưa có cột 'log_return' - hãy tạo trước")

    df = df_t.dropna(subset=["log_return"]).copy()
    n = len(df)
    if n < min_window:
        raise ValueError(f"Không đủ dữ liệu log_return (n={n}) để ước lượng mu,sigma")

    if window is None:
        recent = df["log_return"]
    else:
        w = min(window, n)
        recent = df["log_return"].iloc[-w:]

    mu = float(recent.mean())
    sigma = float(recent.std(ddof=1))  # sample std

    return mu, sigma


# =========================================================
# 2) Vectorized Monte Carlo – giá THẤP NHẤT trong horizon
#    (hỗ trợ batching để tránh OOM khi n_sim rất lớn)
# =========================================================

def _simulate_min_prices_batch(
    current_price: float,
    horizon_days: int,
    mu: float,
    sigma: float,
    n_batch: int,
    rng: np.random.Generator,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Simulate `n_batch` paths (vectorized) và trả về min_prices (n_batch,)
    - dùng float32 theo mặc định để tiết kiệm RAM
    - rng: là np.random.Generator đã khởi tạo (deterministic theo seed)
    """
    # Generate shocks: shape (n_batch, horizon_days)
    z = rng.standard_normal(size=(n_batch, horizon_days)).astype(dtype)

    # daily log-return theo GBM
    dt = 1.0
    drift = np.float32(mu * dt)
    vol_scale = np.float32(sigma * math.sqrt(dt))

    daily_log_ret = drift + vol_scale * z  # shape (n_batch, horizon_days)

    # cumulative log-price paths: log(S0) + cumsum(daily_log_ret)
    log_s0 = np.float32(np.log(current_price))
    log_paths = np.cumsum(daily_log_ret, axis=1, dtype=dtype)
    log_paths += log_s0  # broadcast add

    # min log-price per path
    min_log_price = np.min(log_paths, axis=1)

    # convert back to price (float64 để percentile chính xác)
    min_prices = np.exp(min_log_price.astype(np.float64)).astype(np.float64)
    return min_prices


def monte_carlo_min_prices(
    current_price: float,
    horizon_days: int,
    mu: float,
    sigma: float,
    n_sim: int = 20000,
    seed: int | None = 42,
    batch_size: int | None = None,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Vectorized Monte Carlo min-prices with optional batching.

    Parameters
    ----------
    - current_price: float
    - horizon_days: int
    - mu, sigma: floats (daily)
    - n_sim: tổng số path
    - seed: RNG seed
    - batch_size: nếu None -> tự chọn (khoảng vài chục nghìn path / batch)
    - dtype: 'float32' hoặc 'float64'

    Trả về
    -------
    - numpy.ndarray shape (n_sim,) dtype float64
    """
    if n_sim <= 0:
        return np.array([], dtype=np.float64)

    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")

    dtype_np = np.float32 if dtype == "float32" else np.float64

    # heuristic: giữ mỗi batch ~ 5e6 phần tử
    if batch_size is None:
        target_elems = 5_000_000
        est_batch = max(1, int(target_elems // max(1, horizon_days)))
        batch_size = min(n_sim, max(1, est_batch))

    rng = np.random.default_rng(seed)

    mins: list[np.ndarray] = []
    sims_left = n_sim
    while sims_left > 0:
        cur_batch = min(batch_size, sims_left)
        mins_batch = _simulate_min_prices_batch(
            current_price=current_price,
            horizon_days=horizon_days,
            mu=mu,
            sigma=sigma,
            n_batch=cur_batch,
            rng=rng,
            dtype=dtype_np,
        )
        mins.append(mins_batch)
        sims_left -= cur_batch

    min_prices = np.concatenate(mins, axis=0)
    return min_prices


# =========================================================
# 3) (Optional) Monte Carlo terminal prices (nếu sau này cần)
# =========================================================

def monte_carlo_terminal_prices(
    current_price: float,
    horizon_days: int,
    mu: float,
    sigma: float,
    n_sim: int = 10000,
    seed: int | None = 42,
    dtype: str = "float32",
) -> np.ndarray:
    """
    Vectorized terminal price simulation (1-step equivalent).
    Dùng nếu muốn mô phỏng GIÁ CUỐI KỲ thay vì MIN-PRICE.
    """
    if n_sim <= 0:
        return np.array([], dtype=np.float64)

    dtype_np = np.float32 if dtype == "float32" else np.float64
    rng = np.random.default_rng(seed)

    total_mean = mu * horizon_days
    total_std = sigma * math.sqrt(horizon_days)

    z = rng.standard_normal(size=n_sim).astype(dtype_np)
    total_log = total_mean + total_std * z
    simulated = current_price * np.exp(total_log.astype(np.float64))
    return simulated


# =========================================================
# 4) Quantiles & helper 1Q/2Q
# =========================================================

def mc_drawdown_quantiles_min(
    current_price: float,
    horizon_days: int,
    mu: float,
    sigma: float,
    n_sim: int = 20000,
    seed: int | None = 42,
    probs: Tuple[float, float, float, float] = (5.0, 3.0, 1.0, 0.1),
    batch_size: int | None = None,
    dtype: str = "float32",
) -> Dict[str, float]:
    """
    Tính các mức worst-case dựa trên min_prices (vectorized + batching).
    probs: percent values (e.g. 5.0 = 5th percentile)
    """
    min_prices = monte_carlo_min_prices(
        current_price=current_price,
        horizon_days=horizon_days,
        mu=mu,
        sigma=sigma,
        n_sim=n_sim,
        seed=seed,
        batch_size=batch_size,
        dtype=dtype,
    )

    p5, p3, p1, p01 = np.percentile(min_prices, probs)

    res = {
        "mc_price_q05": float(p5),
        "mc_price_q03": float(p3),
        "mc_price_q01": float(p1),
        "mc_price_q001": float(p01),
        "mc_dd_q05": float(p5 / current_price - 1.0),
        "mc_dd_q03": float(p3 / current_price - 1.0),
        "mc_dd_q01": float(p1 / current_price - 1.0),
        "mc_dd_q001": float(p01 / current_price - 1.0),
    }
    return res


def mc_drawdown_quantiles_1q_2q(
    current_price: float,
    df_t: pd.DataFrame,
    days_1q: int = 66,
    days_2q: int = 132,
    n_sim: int = 20000,
    seed: int | None = 42,
    est_window: int | None = 60,
    min_window: int = 10,
    batch_size: int | None = None,
    dtype: str = "float32",
) -> Dict[str, float]:
    """
    Tính full bộ MC drawdown cho 1Q & 2Q (hỗ trợ batching và dtype).
    LƯU Ý: Ở ĐÂY ĐÃ ÁP DỤNG STRESS (mu,sigma) QUA apply_risk_stress(...)
    """
    # 1) Ước lượng mu,sigma lịch sử
    mu_raw, sigma_raw = estimate_mu_sigma(
        df_t,
        window=est_window,
        min_window=min_window,
    )

    # 2) Áp dụng stress để làm mô hình bảo thủ hơn
    mu, sigma = apply_risk_stress(mu_raw, sigma_raw)

    out: Dict[str, float] = {}

    # --- 1Q ---
    res_1q = mc_drawdown_quantiles_min(
        current_price=current_price,
        horizon_days=days_1q,
        mu=mu,
        sigma=sigma,
        n_sim=n_sim,
        seed=seed,
        batch_size=batch_size,
        dtype=dtype,
    )
    for k, v in res_1q.items():
        out[f"{k}_1q"] = v

    # --- 2Q ---
    res_2q = mc_drawdown_quantiles_min(
        current_price=current_price,
        horizon_days=days_2q,
        mu=mu,
        sigma=sigma,
        n_sim=n_sim,
        seed=None if seed is None else seed + 1,
        batch_size=batch_size,
        dtype=dtype,
    )
    for k, v in res_2q.items():
        out[f"{k}_2q"] = v

    return out
