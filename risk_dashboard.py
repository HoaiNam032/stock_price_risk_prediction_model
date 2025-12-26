

# risk_dashboard.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# --- MC density helper for right-side histogram ---
def compute_mc_density_for_ticker(
    df_all: pd.DataFrame,
    ticker: str,
    trading_days: int,
    n_sim: int = 10_000,
    window: int | None = 60,
) -> tuple[pd.DataFrame, float, float, float, float] | None:
    """
    Tạo phân phối giá tương lai cho 1 ticker bằng Monte Carlo (terminal price).

    Trả về:
      - df_density: DataFrame có cột price_k, count
      - q5_k, q3_k, q1_k: các quantile 5%, 3%, 1% theo đơn vị 'k VND'
      - lowest_k: giá thấp nhất trong các đường mô phỏng (terminal, k VND)
    """
    df_t = df_all[df_all["Ticker"] == ticker].copy()
    if df_t.empty:
        return None

    df_t = df_t.sort_values("Date")
    price_col_mc = "Close_adj" if "Close_adj" in df_t.columns else "Close"
    if price_col_mc not in df_t.columns:
        return None

    df_t["price_k"] = df_t[price_col_mc].astype(float)
    df_t["log_return"] = np.log(df_t["price_k"] / df_t["price_k"].shift(1))
    rets = df_t["log_return"].dropna()
    if len(rets) < 5:
        return None

    # chọn window ước lượng mu, sigma
    if window is None:          # FULL history
        recent = rets
    else:
        w = min(window, len(rets))
        recent = rets.iloc[-w:]

    mu = float(recent.mean())
    sigma = float(recent.std(ddof=1))
    if np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
        return None

    current_price_k = float(df_t["price_k"].iloc[-1])

    rng = np.random.default_rng(123)
    # shape: (n_sim, trading_days)
    daily = rng.normal(mu, sigma, size=(n_sim, trading_days))
    log_paths = np.cumsum(daily, axis=1)
    prices_k = current_price_k * np.exp(log_paths)
    terminal_k = prices_k[:, -1]

    counts, bin_edges = np.histogram(terminal_k, bins=60)
    centers_k = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    df_density = pd.DataFrame({"price_k": centers_k, "count": counts})

    q5_k, q3_k, q1_k = np.percentile(terminal_k, [5, 3, 1])
    lowest_k = float(terminal_k.min())

    return df_density, float(q5_k), float(q3_k), float(q1_k), lowest_k


# ==== GLOBAL SETTINGS ====
st.set_page_config(page_title="Stock Risk Dashboard", layout="wide")

UNIT_MULTIPLIER = 1000.0  # giá trong risk_* là "nghìn VND"


# ============ 1. Load data helpers ============

@st.cache_data
def load_risk_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "last_date" in df.columns:
        df["last_date"] = pd.to_datetime(df["last_date"], errors="coerce")
    return df


@st.cache_data
def load_ratings(path: str = "data/ticker_ratings.csv") -> pd.DataFrame:
    """Rating file requires columns: Ticker, Rating (case-insensitive)."""
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


@st.cache_data
def load_price_history(path: str = "data/cleaned_prices.csv") -> pd.DataFrame:
    """Historical price data: must have columns Ticker, Date, Close (and/or Close_adj)."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


# ============ 1b. Export helper (1 sheet / horizon) ============

def build_export_sheet(
    df_risk: pd.DataFrame,
    ratings_df: pd.DataFrame,
    hk_tag: str,        # "1q" hoặc "2q"
    dd_prefix: str,     # "dd" (ML) hoặc "mc_dd" (Monte Carlo)
) -> pd.DataFrame:
    """
    Tạo bảng xuất Excel cho 1 horizon (1q hoặc 2q) và 1 loại model (ML / MC).

    Cột trả về:
        - Ticker
        - Group           (Rating A/B/C/D)
        - Breakeven_price (VND) = sàn ứng với rating hiện tại
        - Lowest_price_D  (VND) = sàn của nhóm D
    """
    if df_risk.empty:
        return pd.DataFrame(
            columns=["Ticker", "Group", "Breakeven_price", "Lowest_price_D"]
        )

    if dd_prefix == "mc_dd":
        rating_dd_cols = {
            "A": f"mc_dd_q05_{hk_tag}",
            "B": f"mc_dd_q03_{hk_tag}",
            "C": f"mc_dd_q01_{hk_tag}",
            "D": f"mc_dd_q001_{hk_tag}",
        }
    else:
        rating_dd_cols = {
            "A": f"dd_q05_{hk_tag}",
            "B": f"dd_q03_{hk_tag}",
            "C": f"dd_q01_{hk_tag}",
            "D": f"dd_q001_{hk_tag}",
        }
    df = df_risk.copy()
    if not ratings_df.empty:
        df = df.merge(ratings_df, on="Ticker", how="left")
    else:
        df["Rating"] = np.nan

    df["current_price_k"] = df["current_price"].astype(float)

    # Giá stressed từng rating (k VND)
    for r, col in rating_dd_cols.items():
        if col in df.columns:
            df[f"price_{r}_k"] = df["current_price_k"] * (1.0 + df[col].astype(float))
        else:
            df[f"price_{r}_k"] = np.nan

    order = ["A", "B", "C", "D"]
    floors_A, floors_B, floors_C, floors_D = [], [], [], []

    # ép thứ tự A >= B >= C >= D
    for _, row_ in df.iterrows():
        prev = None
        cur_out = {}
        for r in order:
            v = row_[f"price_{r}_k"]
            if pd.isna(v):
                cur_out[r] = np.nan
            else:
                if prev is None:
                    use = v
                else:
                    use = min(prev, v)
                prev = use
                cur_out[r] = use
        floors_A.append(cur_out["A"])
        floors_B.append(cur_out["B"])
        floors_C.append(cur_out["C"])
        floors_D.append(cur_out["D"])

    df["floor_A_k"] = floors_A
    df["floor_B_k"] = floors_B
    df["floor_C_k"] = floors_C
    df["floor_D_k"] = floors_D

    # Breakeven = floor của rating hiện tại
    def _breakeven_row(rw):
        g = str(rw.get("Rating") or "").upper()
        if g not in ["A", "B", "C", "D"]:
            return np.nan
        return rw.get(f"floor_{g}_k", np.nan)

    df["breakeven_k"] = df.apply(_breakeven_row, axis=1)

    df["Breakeven_price"] = df["breakeven_k"] * UNIT_MULTIPLIER
    df["Lowest_price_D"] = df["floor_D_k"] * UNIT_MULTIPLIER

    out = df[["Ticker", "Rating", "Breakeven_price", "Lowest_price_D"]].copy()
    out = out.rename(columns={"Rating": "Group"})

    return out


# ============ 2. UI: model + ticker & parameters ============

st.title("📊 Stock Risk Dashboard")

# --- chọn nguồn model ---
st.sidebar.header("Model & ticker")

model_type = st.sidebar.selectbox(
    "Model source",
    ["LGBM", "XGB", "MC_1Q", "MC_1Y", "MC_FULL"],
    index=0,
)

# map model -> file risk
MODEL_FILE_MAP = {
    "LGBM": "data/risk_all_tickers_lgbm.csv",
    "XGB": "data/risk_all_tickers_xgb.csv",
    "MC_1Q": "data/risk_mc_all_tickers_mc1q.csv",
    "MC_1Y": "data/risk_mc_all_tickers_mc1y.csv",
    "MC_FULL": "data/risk_mc_all_tickers_mcfull.csv",
}
risk_path = MODEL_FILE_MAP[model_type]

try:
    df_risk = load_risk_data(risk_path)
except FileNotFoundError:
    st.error(f"Không tìm thấy file risk cho model {model_type}: `{risk_path}`")
    st.stop()

ratings_df = load_ratings()
df_hist_all = load_price_history()

if df_risk.empty:
    st.error(
        f"File `{risk_path}` rỗng hoặc không đọc được. "
        "Hãy chạy script predict tương ứng trước."
    )
    st.stop()

# chọn ticker
all_tickers = sorted(df_risk["Ticker"].dropna().unique().tolist())
if not all_tickers:
    st.error("No tickers found in risk file.")
    st.stop()

selected_ticker = st.sidebar.selectbox("Ticker", all_tickers, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk horizon & lending setting")

horizon_ltv = st.sidebar.selectbox(
    "Horizon used for stress",
    ["1Q", "2Q"],
    index=1,
)

ltv_future_max = st.sidebar.number_input(
    "Max LTV at stressed price (e.g. 0.5 = 50%)",
    min_value=0.1,
    max_value=1.5,
    value=0.5,
    step=0.05,
)

qty_pledged = st.sidebar.number_input(
    "Number of pledged shares (for position size)",
    min_value=0,
    value=10_000,
    step=100,
)


# ============ 3. Data for selected ticker ============

df_one = df_risk[df_risk["Ticker"] == selected_ticker].copy()
if df_one.empty:
    st.error("No risk data available for the selected ticker.")
    st.stop()

row = df_one.iloc[0]
current_price_k = float(row["current_price"])
current_price_vnd = current_price_k * UNIT_MULTIPLIER

last_date = (
    row["last_date"] if "last_date" in row and not pd.isna(row["last_date"]) else None
)

current_rating = None
if not ratings_df.empty:
    r_row = ratings_df[ratings_df["Ticker"] == selected_ticker]
    if not r_row.empty:
        current_rating = str(r_row.iloc[0]["Rating"]).upper().strip()

# Price history (raw cleaned_prices)
df_hist = df_hist_all[df_hist_all["Ticker"] == selected_ticker].copy()
df_hist = df_hist.sort_values("Date")

# chỉ giữ từ 11/2024 trở về sau cho gọn
cutoff_date = pd.Timestamp("2024-11-01")
df_hist = df_hist[df_hist["Date"] >= cutoff_date].copy()

price_col = "Close_adj" if "Close_adj" in df_hist.columns else "Close"
df_hist = df_hist.dropna(subset=["Date", price_col])

if df_hist.empty:
    st.error(
        "No historical prices found for this ticker in cleaned_prices "
        "(from 11/2024 onward)."
    )
    st.stop()

df_hist["price_k"] = df_hist[price_col].astype(float)
df_hist["price_vnd"] = df_hist["price_k"] * UNIT_MULTIPLIER

last_hist_date = df_hist["Date"].max()

trading_days_map = {
    "1Q": 66,   # ~ 3 tháng
    "2Q": 132,  # ~ 6 tháng
}
trading_days = trading_days_map.get(horizon_ltv, 66)


# ============ 4. Header metrics ============

st.subheader(f"Basic information: `{selected_ticker}`")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Current price", f"{current_price_vnd:,.0f} VND")
with c2:
    st.metric(
        "Last data date", str(last_date.date()) if last_date is not None else "N/A"
    )
with c3:
    st.metric("Credit Score", current_rating or "Not available")
with c4:
    st.metric("Model", model_type)


# ============ 5. Compute stressed prices by rating (A/B/C/D) ============

hk_tag = "1q" if horizon_ltv == "1Q" else "2q"

is_mc_model = model_type.startswith("MC")  # MC_1Q, MC_1Y, MC_FULL

if is_mc_model:
    dd_prefix = "mc_dd"
    rating_dd_cols = {
        "A": f"mc_dd_q05_{hk_tag}",
        "B": f"mc_dd_q03_{hk_tag}",
        "C": f"mc_dd_q01_{hk_tag}",
        "D": f"mc_dd_q001_{hk_tag}",
    }
else:
    dd_prefix = "dd"
    rating_dd_cols = {
        "A": f"dd_q05_{hk_tag}",
        "B": f"dd_q03_{hk_tag}",
        "C": f"dd_q01_{hk_tag}",
        "D": f"dd_q001_{hk_tag}",
    }

# raw stressed (k VND)
stress_price_raw_k = {}
for r, col in rating_dd_cols.items():
    if col in row and not pd.isna(row[col]):
        dd_val = float(row[col])  # drawdown (âm)
        stress_price_raw_k[r] = current_price_k * (1.0 + dd_val)

# ép thứ tự A ≥ B ≥ C ≥ D
line_value_by_rating_k = {}
rating_order = ["A", "B", "C", "D"]
prev_price_k = None

for r in rating_order:
    if r not in stress_price_raw_k:
        continue
    p_raw_k = stress_price_raw_k[r]
    if prev_price_k is None:
        p_use_k = p_raw_k
    else:
        p_use_k = min(prev_price_k, p_raw_k)
    prev_price_k = p_use_k
    line_value_by_rating_k[r] = p_use_k  # k VND

price_by_rating_vnd = {
    r: line_value_by_rating_k[r] * UNIT_MULTIPLIER for r in line_value_by_rating_k
}

# LTV & loan per share
ltv_by_rating = {}
loan_per_share_by_rating_k = {}
for r in rating_order:
    if r not in line_value_by_rating_k:
        continue
    p_stress_k = line_value_by_rating_k[r]
    ltv = ltv_future_max * (p_stress_k / current_price_k)
    ltv = float(np.clip(ltv, 0.0, 1.0))
    ltv_by_rating[r] = ltv
    loan_per_share_by_rating_k[r] = current_price_k * ltv  # k VND

ltv_rec = None
breakeven_price_vnd = None
loan_max_vnd = None

if current_rating is not None and current_rating in price_by_rating_vnd:
    breakeven_price_vnd = price_by_rating_vnd[current_rating]

if current_rating is not None and current_rating in ltv_by_rating:
    ltv_rec = ltv_by_rating[current_rating]
    loan_per_share_current_k = loan_per_share_by_rating_k[current_rating]
    if qty_pledged > 0:
        loan_max_k = loan_per_share_current_k * qty_pledged
        loan_max_vnd = loan_max_k * UNIT_MULTIPLIER


# ============ 6. Price distribution text (suggested levels) ============

lines = []
lines.append(
    f"The distribution plot describes probabilities of possible prices "
    f"in the next {trading_days} trading days "
    f"based on **{model_type}** model."
)

if "A" in price_by_rating_vnd:
    lines.append(
        f"- For Group A, we propose to take higher risk, "
        f"at {price_by_rating_vnd['A']:,.0f} dong."
    )
if "B" in price_by_rating_vnd:
    lines.append(
        f"- For Group B, we propose to take moderate risk, "
        f"at {price_by_rating_vnd['B']:,.0f} dong."
    )
if "C" in price_by_rating_vnd:
    lines.append(
        f"- For Group C, we propose to take minor risk, "
        f"at {price_by_rating_vnd['C']:,.0f} dong."
    )
if "D" in price_by_rating_vnd:
    lines.append(
        f"- For Group D, we propose to take no risk, "
        f"at {price_by_rating_vnd['D']:,.0f} dong."
    )

st.markdown(
    "<div style='font-size:12px; line-height:1.3; margin-bottom:0.5rem;'>"
    + "<br>".join(lines)
    + "</div>",
    unsafe_allow_html=True,
)


# ============ 7. ONE combined chart: price history + zone + breakeven + density ============

# worst-case level (D) – đáy zone (Group D)
if "D" in line_value_by_rating_k:
    worst_k = line_value_by_rating_k["D"]
else:
    worst_k = float(df_hist["price_k"].min() * 0.9)

# đỉnh zone: dùng max giữa giá hiện tại và giá cao nhất lịch sử
highest_hist_k = float(df_hist["price_k"].max())
top_k = max(current_price_k, highest_hist_k)

# số điểm trong vùng forecast
n_points = trading_days + 1
future_dates = pd.date_range(last_hist_date, periods=n_points, freq="D")

t = np.linspace(0.0, 1.0, n_points)

# 1) Đường cơ sở cho nhánh trên & dưới
upper_base = current_price_k + (top_k - current_price_k) * (t ** 0.6)      # cong nhẹ lên
lower_base = current_price_k + (worst_k - current_price_k) * (t ** 0.6)    # cong nhẹ xuống

# 2) Thêm random walk nhẹ để tạo "ghồ ghề"
rng = np.random.default_rng(40)

def add_jitter(base: np.ndarray, amp_frac: float = 0.09) -> np.ndarray:
    """
    Thêm dao động nhỏ kiểu random-walk nhưng giữ nguyên 2 đầu.
    amp_frac ~ biên độ dao động tương đối.
    """
    n = base.shape[0]
    steps = rng.normal(0.0, 1.0, size=n)
    walk = np.cumsum(steps)

    # chuẩn hoá walk về [-0.5, 0.5]
    walk = (walk - walk.min()) / (walk.max() - walk.min() + 1e-9) - 0.5

    amp = amp_frac * (base.max() - base.min())
    noisy = base + walk * amp

    # neo lại 2 đầu cho đúng shape
    noisy[0] = base[0]
    noisy[-1] = base[-1]
    return noisy

upper_k = add_jitter(upper_base, amp_frac=0.085)
lower_k = add_jitter(lower_base, amp_frac=0.085)

df_future = pd.DataFrame(
    {
        "Date": future_dates,
        "upper_k": upper_k,
        "lower_k": lower_k,
    }
)

df_zone = pd.DataFrame(
    {
        "Date": future_dates,
        "bottom_k": lower_k,
        "top_k": upper_k,
    }
)

min_hist_price_k = df_hist["price_k"].min()

base_hist = alt.Chart(df_hist).encode(
    x=alt.X(
        "Date:T",
        title="Date",
        axis=alt.Axis(
            format="%m/%Y",
            tickCount={"interval": "month", "step": 2},
            labelFontSize=12,
            labelFontWeight="bold",
            titleFontSize=13,
            titleFontWeight="bold",
        ),
    ),
    y=alt.Y(
        "price_k:Q",
        title="Stock Price",
        axis=alt.Axis(
            format=",.0f",
            labelExpr="datum.value + 'k'",
            labelFontSize=12,
            labelFontWeight="bold",
            titleFontSize=13,
            titleFontWeight="bold",
        ),
    ),
    tooltip=[
        alt.Tooltip("Date:T", title="Date"),
        alt.Tooltip("price_vnd:Q", title="Price (VND)", format=",.0f"),
    ],
)

hist_line = base_hist.mark_line(color="#1f77b4", size=2)

hist_points = base_hist.mark_circle(size=20, color="red").encode(
    opacity=alt.condition(
        alt.datum.price_k == min_hist_price_k, alt.value(1), alt.value(0)
    )
)

layers = [hist_line, hist_points]

last_row = df_hist.iloc[[-1]]
hist_last_text = (
    alt.Chart(last_row)
    .mark_text(
        align="left",
        dx=5,
        dy=-8,
        fontSize=13,
        fontWeight="bold",
        color="#1c4a6b",
    )
    .encode(
        x="Date:T",
        y="price_k:Q",
        text=alt.Text("price_vnd:Q", format=",.0f"),
    )
)
layers.append(hist_last_text)

zone_area = (
    alt.Chart(df_zone)
    .mark_area(color="rgba(144,238,144,0.45)", strokeWidth=0)
    .encode(
        x="Date:T",
        y="bottom_k:Q",
        y2="top_k:Q",
    )
)

upper_boundary = (
    alt.Chart(df_future)
    .mark_line(color="#CE5050", size=2)
    .encode(
        x="Date:T",
        y="upper_k:Q",
    )
)

lower_boundary = (
    alt.Chart(df_future)
    .mark_line(color="#CE5050", size=2)
    .encode(
        x="Date:T",
        y="lower_k:Q",
    )
)

layers.extend([zone_area, upper_boundary, lower_boundary])


def compute_stressed_prices_for_row(
    row_in: pd.Series,
    hk_tag: str,
    model_kind: str,   # "ML" hoặc "MC"
) -> dict:
    """
    Trả về dict { "A": price_vnd, "B": ..., "C": ..., "D": ... }
    cho 1 ticker, 1 horizon, 1 loại model (ML hoặc MC),
    đã ép thứ tự A >= B >= C >= D giống trên chart.
    """
    current_price_k = float(row_in["current_price"])
    if model_kind == "MC":
        rating_dd_cols = {
            "A": f"mc_dd_q05_{hk_tag}",
            "B": f"mc_dd_q03_{hk_tag}",
            "C": f"mc_dd_q01_{hk_tag}",
            "D": f"mc_dd_q001_{hk_tag}",
        }
    else:  # ML
        rating_dd_cols = {
            "A": f"dd_q05_{hk_tag}",
            "B": f"dd_q03_{hk_tag}",
            "C": f"dd_q01_{hk_tag}",
            "D": f"dd_q001_{hk_tag}",
        }

    rating_order_local = ["A", "B", "C", "D"]
    stress_raw = {}
    for r in rating_order_local:
        col = rating_dd_cols[r]
        if col in row_in and not pd.isna(row_in[col]):
            dd_val = float(row_in[col])
            stress_raw[r] = current_price_k * (1.0 + dd_val)

    line_k = {}
    prev = None
    for r in rating_order_local:
        if r not in stress_raw:
            continue
        p_raw_k = stress_raw[r]
        if prev is None:
            use_k = p_raw_k
        else:
            use_k = min(prev, p_raw_k)
        prev = use_k
        line_k[r] = use_k

    return {r: line_k[r] * UNIT_MULTIPLIER for r in line_k}  # VND


# --- Breakeven Price line (1 đường ngang đỏ) ---
if breakeven_price_vnd is not None:
    breakeven_k = breakeven_price_vnd / UNIT_MULTIPLIER

    breakeven_rule = (
        alt.Chart(pd.DataFrame({"price_k": [breakeven_k]}))
        .mark_rule(color="#CC0000", strokeDash=[4, 4])
        .encode(
            y="price_k:Q",
        )
    )

    mid_date = last_hist_date + (future_dates[-1] - last_hist_date) / 2
    df_breakeven_text = pd.DataFrame(
        {
            "Date": [mid_date],
            "price_k": [breakeven_k],
            "label": [f"Breakeven Price: {breakeven_price_vnd:,.0f}"],
        }
    )

    breakeven_text = (
        alt.Chart(df_breakeven_text)
        .mark_text(
            align="center",
            baseline="bottom",
            dy=-4,
            fontSize=11,
            color="#CC0000",
        )
        .encode(
            x="Date:T",
            y="price_k:Q",
            text="label:N",
        )
    )

    layers.extend([breakeven_rule, breakeven_text])


# chart trái
chart_left = (
    alt.layer(*layers)
    .properties(
        height=380,
        width=550,
    )
)

# ---- 2-panel: left = main chart, right = MC density (chỉ cho các model MC*) ----
if is_mc_model:
    # chọn window density phù hợp với từng model MC
    if model_type == "MC_1Q":
        density_window = 66
    elif model_type == "MC_1Y":
        density_window = 250
    else:  # MC_FULL
        density_window = None  # dùng full lịch sử log_return

    mc_density_res = compute_mc_density_for_ticker(
        df_hist_all,
        selected_ticker,
        trading_days,
        n_sim=10_000,
        window=density_window,
    )

    if mc_density_res is None:
        chart_single = (
            chart_left
            .configure_axis(labelFontSize=12, titleFontSize=13)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart_single, use_container_width=True)
    else:
        df_density, _q5_k_sim, _q3_k_sim, _q1_k_sim, _lowest_k_sim = mc_density_res

        max_count = float(df_density["count"].max() or 1)

        density_base = (
            alt.Chart(df_density)
            .mark_bar(orient="horizontal")
            .encode(
                y=alt.Y(
                    "price_k:Q",
                    axis=alt.Axis(
                        title=None,
                        labels=False,   # dùng chung trục Y với chart trái
                    ),
                ),
                x=alt.X(
                    "count:Q",
                    axis=alt.Axis(
                        title="Density",
                        titleFontWeight="bold",
                        labels=False,     # tắt số 0, 100, 200...
                    ),
                ),
            )
        )

        # --- DÙNG GIÁ NHÓM A/B/C/D LÀM 5% / 3% / 1% / LOWEST ---
        q5_k = line_value_by_rating_k.get("A", np.nan)
        q3_k = line_value_by_rating_k.get("B", np.nan)
        q1_k = line_value_by_rating_k.get("C", np.nan)
        lowest_k = line_value_by_rating_k.get("D", np.nan)

        levels = []
        if not np.isnan(q5_k):
            levels.append(("5% Worst Case", q5_k))
        if not np.isnan(q3_k):
            levels.append(("3% Worst Case", q3_k))
        if not np.isnan(q1_k):
            levels.append(("1% Worst Case", q1_k))
        if not np.isnan(lowest_k):
            levels.append(("Lowest Price", lowest_k))

        df_levels = pd.DataFrame(
            [
                {
                    "price_k": val_k,
                    "x": max_count * 1.02,
                    "label": f"{label}: {val_k * UNIT_MULTIPLIER:,.0f}",
                    "is_tail": 1 if label == "Lowest Price" else 0,
                }
                for (label, val_k) in levels
            ]
        )

        # vùng tail (Lowest -> 1%) tô hồng nhạt
        if not np.isnan(lowest_k) and not np.isnan(q1_k):
            df_tail = pd.DataFrame(
                {
                    "y1": [lowest_k],
                    "y2": [q1_k],
                    "x1": [0.0],
                    "x2": [max_count],
                }
            )
            tail_bg = (
                alt.Chart(df_tail)
                .mark_rect(color="rgba(255,0,0,0.08)")
                .encode(
                    y="y1:Q",
                    y2="y2:Q",
                    x="x1:Q",
                    x2="x2:Q",
                )
            )
        else:
            tail_bg = None

        level_rules = (
            alt.Chart(df_levels[df_levels["label"].str.startswith(("5%", "3%", "1%"))])
            .mark_rule(color="red", strokeDash=[4, 2])
            .encode(y="price_k:Q")
        )

        level_texts = (
            alt.Chart(df_levels)
            .mark_text(
                align="left",
                dx=4,
                dy=-2,
                fontSize=10,
                color="red",
            )
            .encode(
                y="price_k:Q",
                x="x:Q",
                text="label:N",
            )
        )

        if tail_bg is not None:
            density_chart = (
                alt.layer(tail_bg, density_base, level_rules, level_texts)
                .properties(width=240, height=380)
            )
        else:
            density_chart = (
                alt.layer(density_base, level_rules, level_texts)
                .properties(width=240, height=380)
            )

        full_chart = (
            alt.hconcat(chart_left, density_chart, spacing=3)
            .resolve_scale(y="shared")
            .configure_axis(labelFontSize=12, titleFontSize=13)
            .configure_view(strokeWidth=0)
        )

        st.altair_chart(full_chart, use_container_width=True)
else:
    chart_single = (
        chart_left
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart_single, use_container_width=True)


# ============ 7b. So sánh LGBM vs XGB vs MC_1Q vs MC_1Y vs MC_FULL (A/B/C/D) ============

st.subheader("Model comparison: LGBM vs XGB vs MC_1Q vs MC_1Y vs MC_FULL (A/B/C/D)")

models_to_compare = ["LGBM", "XGB", "MC_1Q", "MC_1Y", "MC_FULL"]
compare_results = {}

for m in models_to_compare:
    risk_file = MODEL_FILE_MAP.get(m)
    try:
        df_m = load_risk_data(risk_file)
    except Exception:
        df_m = None

    if df_m is None or df_m.empty or "Ticker" not in df_m.columns:
        compare_results[m] = None
        continue

    row_m = df_m[df_m["Ticker"] == selected_ticker]
    if row_m.empty:
        compare_results[m] = None
        continue

    row_m = row_m.iloc[0]
    kind = "MC" if m.startswith("MC") else "ML"
    prices = compute_stressed_prices_for_row(row_m, hk_tag, model_kind=kind)
    compare_results[m] = prices  # dict {A: price_vnd, ...}

data_rows = []
for g in ["A", "B", "C", "D"]:
    row_dict = {"Group": g}
    for m in models_to_compare:
        col_name = f"{m}_price (VND)"
        prices = compare_results.get(m)
        if prices is None:
            row_dict[col_name] = np.nan
        else:
            row_dict[col_name] = prices.get(g, np.nan)
    data_rows.append(row_dict)

df_compare_all = pd.DataFrame(data_rows)

if df_compare_all[[c for c in df_compare_all.columns if "price" in c]].isna().all().all():
    st.info("⚠️ Không đủ dữ liệu để so sánh các model cho ticker này.")
else:
    st.dataframe(
        df_compare_all.style.format(
            {col: "{:,.0f}" for col in df_compare_all.columns if "VND" in col}
        ),
        use_container_width=True,
    )

# ============ 8. Lending summary ============

st.subheader("Lending policy summary")

c2, c3, c4 = st.columns(3)

with c2:
    if ltv_rec is not None:
        st.metric("LTV (internal rating)", f"{ltv_rec*100:.1f}%")
    else:
        st.metric("LTV (internal rating)", "N/A")

with c3:
    if breakeven_price_vnd is not None:
        st.metric("Breakeven price", f"{breakeven_price_vnd:,.0f} VND")
    else:
        st.metric("Breakeven price", "N/A")

with c4:
    if loan_max_vnd is not None:
        st.metric("Margin Limit", f"{loan_max_vnd:,.0f} VND")
    else:
        st.metric("Margin Limit", "Enter pledged quantity & rating LTV")


# ============ 9. Export Excel (2 sheet: 1Q & 2Q) ============

df_1q_export = build_export_sheet(df_risk, ratings_df, "1q", dd_prefix)
df_2q_export = build_export_sheet(df_risk, ratings_df, "2q", dd_prefix)

if not df_1q_export.empty or not df_2q_export.empty:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_1q_export.to_excel(writer, sheet_name="1Q", index=False)
        df_2q_export.to_excel(writer, sheet_name="2Q", index=False)
    buffer.seek(0)

    st.download_button(
        label=f"📥 Download risk levels (1Q & 2Q) - {model_type}",
        data=buffer,
        file_name=f"risk_levels_by_group_{model_type.lower()}.xlsx",
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
    )


