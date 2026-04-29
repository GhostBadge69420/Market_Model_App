import pandas as pd
import numpy as np


def create_features(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "Close" not in df.columns:
        return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = 0

    numeric_cols = ["Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df["Returns"] = df["Close"].pct_change()

    # price features
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MeanPrice"] = df["Close"].rolling(20).mean()

    # volatility
    df["Volatility"] = df["Returns"].rolling(20).std()

    # volume features
    df["Volume_MA"] = df["Volume"].rolling(20).mean()

    # momentum
    df["Momentum"] = df["Close"] - df["Close"].shift(5)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["Close", "MA20", "MA50", "Volatility", "Volume_MA", "Momentum"]
    )

    return df
