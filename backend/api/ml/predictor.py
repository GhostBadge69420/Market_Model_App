from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf
from django.conf import settings

from .features import create_features

BASE_DIR = settings.BASE_DIR
MODELS_DIR = Path(BASE_DIR) / "api" / "ml" / "models"
RF_FEATURE_COLUMNS = ["MA20", "MA50", "Volatility", "Volume_MA"]

rf_model = None
arima_model = None


def get_rf_model():
    global rf_model

    if rf_model is None:
        rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")

    return rf_model


def get_arima_model():
    global arima_model

    if arima_model is None:
        arima_model = joblib.load(MODELS_DIR / "arima_model.pkl")

    return arima_model


@lru_cache(maxsize=32)
def load_market_data(symbol):
    df = yf.download(symbol, period="2y", interval="1d", progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return create_features(df)


def predict_with_arima(close_series):
    try:
        saved_model = get_arima_model()
        if hasattr(saved_model, "apply"):
            updated_model = saved_model.apply(close_series)
            forecast = updated_model.forecast(steps=1)
            return float(forecast.iloc[0] if hasattr(forecast, "iloc") else forecast[0])
    except Exception:
        return None


@lru_cache(maxsize=32)
def predict(symbol):
    symbol = str(symbol).strip().upper()

    try:
        df = load_market_data(symbol)
    except Exception:
        return None, None

    if df is None or df.empty:
        return None, None

    latest = df.iloc[-1]

    # ---------------- RF prediction ----------------
    rf_input = pd.DataFrame(
        [[latest[column] for column in RF_FEATURE_COLUMNS]],
        columns=RF_FEATURE_COLUMNS,
    )

    try:
        rf_pred = float(get_rf_model().predict(rf_input)[0])
    except Exception:
        rf_pred = None

    # ---------------- ARIMA prediction ----------------
    close_series = df["Close"]

    arima_pred = predict_with_arima(close_series)

    # ---------------- ensemble ----------------
    predictions = [pred for pred in (rf_pred, arima_pred) if pred is not None]

    if not predictions:
        return None, latest

    final_pred = sum(predictions) / len(predictions)

    return float(final_pred), latest
