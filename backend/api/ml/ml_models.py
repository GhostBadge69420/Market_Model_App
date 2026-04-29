from functools import lru_cache

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

from .features import create_features


@lru_cache(maxsize=32)
def load_market_history(symbol):
    df = yf.download(symbol, period="2y", interval="1d", progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna().copy()


def compare_history_models(df, label="CUSTOM"):
    label = str(label).strip() or "CUSTOM"

    if df is None or df.empty or "Close" not in df.columns:
        return {"Error": "No data available"}

    featured_df = create_features(df)

    if featured_df.empty:
        return {"Error": "Unable to create model features"}

    if len(featured_df) < 60:
        return {"Error": "Not enough historical data"}

    close = featured_df["Close"]
    features = featured_df[["MA20", "MA50", "Volatility", "Volume_MA"]]
    target = featured_df["Close"]

    split = int(len(features) * 0.8)
    if split <= 0 or split >= len(features):
        return {"Error": "Not enough test data"}

    X_train = features[:split]
    X_test = features[split:]

    y_train = target[:split]
    y_test = target[split:]
    close_train = close[:split]
    close_test = close[split:]

    if len(y_test) == 0:
        return {"Error": "Not enough test data"}

    # ---------------------------
    # BENCHMARK MODEL
    # ---------------------------
    benchmark_pred = close_test.shift(1)
    benchmark_pred.iloc[0] = close_train.iloc[-1]
    benchmark_pred = benchmark_pred.astype(float)

    benchmark_mae = mean_absolute_error(close_test, benchmark_pred)
    benchmark_rmse = np.sqrt(mean_squared_error(close_test, benchmark_pred))
    benchmark_r2 = r2_score(close_test, benchmark_pred)

    # ---------------------------
    # ARIMA MODEL
    # ---------------------------
    try:
        model = ARIMA(close_train, order=(5, 1, 0))
        model_fit = model.fit()
        arima_forecast = model_fit.forecast(steps=len(close_test))
        arima_pred = pd.Series(np.asarray(arima_forecast, dtype=float), index=close_test.index, dtype=float)

        arima_mae = mean_absolute_error(close_test, arima_pred)
        arima_rmse = np.sqrt(mean_squared_error(close_test, arima_pred))
        arima_r2 = r2_score(close_test, arima_pred)
    except Exception:
        arima_pred = pd.Series(index=close_test.index, dtype=float)
        arima_mae = None
        arima_rmse = None
        arima_r2 = None

    rf = RandomForestRegressor(
        n_estimators=60,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    rf_pred = pd.Series(rf.predict(X_test), index=y_test.index, dtype=float)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    metric_table = {
        "Benchmark": {
            "MAE": round(float(benchmark_mae), 4),
            "RMSE": round(float(benchmark_rmse), 4),
            "R2": round(float(benchmark_r2), 4),
        },
        "ARIMA": {
            "MAE": None if arima_mae is None else round(float(arima_mae), 4),
            "RMSE": None if arima_rmse is None else round(float(arima_rmse), 4),
            "R2": None if arima_r2 is None else round(float(arima_r2), 4),
        },
        "Random Forest": {
            "MAE": round(float(rf_mae), 4),
            "RMSE": round(float(rf_rmse), 4),
            "R2": round(float(rf_r2), 4),
        },
    }

    eligible_models = {
        name: values["RMSE"]
        for name, values in metric_table.items()
        if values["RMSE"] is not None
    }
    best_model = min(eligible_models, key=eligible_models.get)

    forecasting_models = {
        name: values["RMSE"]
        for name, values in metric_table.items()
        if name in {"ARIMA", "Random Forest"} and values["RMSE"] is not None
    }
    best_forecasting_model = None
    if forecasting_models:
        best_forecasting_model = min(forecasting_models, key=forecasting_models.get)

    comparison_frame = pd.DataFrame(
        {
            "Date": close_test.index.astype(str),
            "Actual": close_test.astype(float).round(4).tolist(),
            "Benchmark": benchmark_pred.astype(float).round(4).tolist(),
            "ARIMA": arima_pred.reindex(close_test.index).round(4).tolist(),
            "Random Forest": rf_pred.reindex(close_test.index).round(4).tolist(),
        }
    )

    results = {
        "symbol": label,
        "best_model": best_model,
        "best_forecasting_model": best_forecasting_model,
        "test_points": int(len(close_test)),
        "test_period_start": str(close_test.index[0].date()),
        "test_period_end": str(close_test.index[-1].date()),
        "metrics": metric_table,
        "comparison_frame": comparison_frame.to_dict("records"),
    }

    return results


def forecast_period_returns(df, periods):
    if df is None or df.empty or "Close" not in df.columns:
        return {}

    history = df.sort_index().copy()
    history["Close"] = pd.to_numeric(history["Close"], errors="coerce")
    history = history.dropna(subset=["Close"])
    featured_df = create_features(history)

    if history.empty:
        return {}

    feature_columns = ["MA20", "MA50", "Volatility", "Volume_MA"]
    output = {}

    for period in periods:
        label = period["label"]
        start_date = pd.Timestamp(period["start"])
        end_date = pd.Timestamp(period["end"])

        period_close = history.loc[(history.index >= start_date) & (history.index <= end_date), "Close"]
        close_train = history.loc[history.index < start_date, "Close"]
        feat_train = featured_df.loc[featured_df.index < start_date]
        feat_test = featured_df.loc[(featured_df.index >= start_date) & (featured_df.index <= end_date)]

        arima_return = None
        rf_return = None

        if len(period_close) >= 2 and len(close_train) >= 60:
            start_close = float(period_close.iloc[0])

            try:
                arima_model = ARIMA(close_train, order=(5, 1, 0)).fit()
                arima_forecast = np.asarray(
                    arima_model.forecast(steps=len(period_close)),
                    dtype=float,
                )
                if len(arima_forecast):
                    arima_return = float((arima_forecast[-1] / start_close) - 1)
            except Exception:
                arima_return = None

            if len(feat_train) >= 60 and not feat_test.empty:
                try:
                    rf = RandomForestRegressor(
                        n_estimators=60,
                        random_state=42,
                        n_jobs=-1,
                    )
                    rf.fit(feat_train[feature_columns], feat_train["Close"])
                    rf_forecast = rf.predict(feat_test[feature_columns])
                    if len(rf_forecast):
                        rf_return = float((float(rf_forecast[-1]) / start_close) - 1)
                except Exception:
                    rf_return = None

        output[label] = {
            "ARIMA Forecast": arima_return,
            "Random Forest Forecast": rf_return,
        }

    return output


@lru_cache(maxsize=32)
def compare_models(symbol):
    symbol = str(symbol).strip().upper()

    df = load_market_history(symbol)

    return compare_history_models(df, symbol)
