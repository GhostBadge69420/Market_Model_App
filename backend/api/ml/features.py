import pandas as pd
import numpy as np


def _safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _rsi(close, window=14):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = _safe_divide(avg_gain, avg_loss)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return rsi


def create_features(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "Close" not in df.columns:
        return pd.DataFrame()

    for column in ["Open", "High", "Low", "Volume"]:
        if column not in df.columns:
            df[column] = df["Close"] if column != "Volume" else 0

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df["Returns"] = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"]).diff()
    df["Intraday_Range"] = _safe_divide(df["High"] - df["Low"], df["Close"])
    df["Close_Position"] = _safe_divide(df["Close"] - df["Low"], df["High"] - df["Low"])
    df["Gap"] = _safe_divide(df["Open"] - df["Close"].shift(1), df["Close"].shift(1))

    # price features
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    df["MeanPrice"] = df["Close"].rolling(20).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["Price_to_MA20"] = _safe_divide(df["Close"], df["MA20"]) - 1
    df["Price_to_MA50"] = _safe_divide(df["Close"], df["MA50"]) - 1
    df["MA20_to_MA50"] = _safe_divide(df["MA20"], df["MA50"]) - 1
    df["MA5_Slope"] = df["MA5"].pct_change(5)
    df["MA20_Slope"] = df["MA20"].pct_change(10)

    # volatility
    df["Volatility"] = df["Returns"].rolling(20).std()
    df["Volatility_5"] = df["Returns"].rolling(5).std()
    df["Volatility_50"] = df["Returns"].rolling(50).std()
    df["Realized_Volatility"] = df["Log_Returns"].rolling(20).std() * np.sqrt(252)
    rolling_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["MA20"] + (rolling_std * 2)
    df["BB_Lower"] = df["MA20"] - (rolling_std * 2)
    df["BB_Width"] = _safe_divide(df["BB_Upper"] - df["BB_Lower"], df["MA20"])
    df["BB_Position"] = _safe_divide(df["Close"] - df["BB_Lower"], df["BB_Upper"] - df["BB_Lower"])

    # volume features
    df["Volume_MA"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = _safe_divide(df["Volume"], df["Volume_MA"])
    df["Dollar_Volume"] = df["Close"] * df["Volume"]

    # momentum
    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Momentum_20"] = df["Close"] - df["Close"].shift(20)
    df["Return_5"] = df["Close"].pct_change(5)
    df["Return_10"] = df["Close"].pct_change(10)
    df["Return_20"] = df["Close"].pct_change(20)
    df["RSI14"] = _rsi(df["Close"], 14)

    df = df.replace([np.inf, -np.inf], np.nan)
    required_columns = ["Close", "MA20", "MA50", "Volatility", "Volume_MA", "Momentum", "RSI14"]
    df = df.dropna(subset=required_columns)

    return df
