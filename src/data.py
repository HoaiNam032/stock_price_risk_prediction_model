# src/data.py
import pandas as pd


def load_cleaned_prices(path: str = "data/cleaned_prices.csv") -> pd.DataFrame:
    """
    Đọc file cleaned_prices.csv, chuẩn hóa cột Date và sort.
    """
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def get_ticker_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Lọc ra dữ liệu của 1 mã cổ phiếu cụ thể.
    """
    df_t = df[df["Ticker"] == ticker].copy()
    if "Date" in df_t.columns:
        df_t = df_t.sort_values("Date").reset_index(drop=True)
    return df_t
