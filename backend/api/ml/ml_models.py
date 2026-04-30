from functools import lru_cache

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.arima.model import ARIMA

from .features import create_features

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

ADVANCED_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Returns",
    "Log_Returns",
    "Intraday_Range",
    "Close_Position",
    "Gap",
    "MA5",
    "MA20",
    "MA50",
    "MA100",
    "EMA12",
    "EMA26",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Price_to_MA20",
    "Price_to_MA50",
    "MA20_to_MA50",
    "MA5_Slope",
    "MA20_Slope",
    "Volatility",
    "Volatility_5",
    "Volatility_50",
    "Realized_Volatility",
    "BB_Width",
    "BB_Position",
    "Volume_MA",
    "Volume_Ratio",
    "Dollar_Volume",
    "Momentum",
    "Momentum_10",
    "Momentum_20",
    "Return_5",
    "Return_10",
    "Return_20",
    "RSI14",
]
TRANSFORMER_SEQUENCE_LENGTH = 16
TRANSFORMER_MIN_TRAIN_SEQUENCES = 48


class TimeSeriesTransformer(nn.Module if nn is not None else object):
    def __init__(self, feature_count, model_dim=32, nhead=4, layers=1, dropout=0.06):
        super().__init__()
        self.input_projection = nn.Linear(feature_count, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, TRANSFORMER_SEQUENCE_LENGTH, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(self, sequence):
        encoded = self.input_projection(sequence)
        encoded = encoded + self.position_embedding[:, : encoded.shape[1], :]
        encoded = self.encoder(encoded)
        return self.head(encoded[:, -1, :]).squeeze(-1)


@lru_cache(maxsize=32)
def load_market_history(symbol):
    df = yf.download(symbol, period="2y", interval="1d", progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna().copy()


def _metric_dict(actual, predicted):
    actual = pd.Series(actual, dtype=float)
    predicted = pd.Series(predicted, index=actual.index, dtype=float)
    valid = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()

    if valid.empty:
        return {"MAE": None, "RMSE": None, "R2": None, "MAPE": None, "Directional Accuracy": None}

    actual_values = valid["actual"]
    predicted_values = valid["predicted"]
    previous_actual = actual_values.shift(1)
    actual_direction = np.sign(actual_values - previous_actual)
    predicted_direction = np.sign(predicted_values - previous_actual)
    directional_mask = previous_actual.notna()
    directional_accuracy = None
    if directional_mask.any():
        directional_accuracy = float((actual_direction[directional_mask] == predicted_direction[directional_mask]).mean())

    non_zero_actual = actual_values.replace(0, np.nan)
    mape = (np.abs((actual_values - predicted_values) / non_zero_actual)).dropna().mean()

    return {
        "MAE": round(float(mean_absolute_error(actual_values, predicted_values)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(actual_values, predicted_values))), 4),
        "R2": round(float(r2_score(actual_values, predicted_values)), 4) if len(valid) > 1 else None,
        "MAPE": None if pd.isna(mape) else round(float(mape), 4),
        "Directional Accuracy": None if directional_accuracy is None else round(directional_accuracy, 4),
    }


def _build_supervised_frame(featured_df):
    features = featured_df[[column for column in ADVANCED_FEATURE_COLUMNS if column in featured_df.columns]].copy()
    features = features.dropna(axis=1, how="all")
    features = features.ffill().bfill().fillna(0)
    target = featured_df["Close"].shift(-1).rename("Target_Close")
    target_dates = pd.Series(featured_df.index, index=featured_df.index).shift(-1)
    supervised = features.join(target).dropna(subset=["Target_Close"])
    supervised["Target_Date"] = target_dates.reindex(supervised.index)
    supervised = supervised.dropna(subset=["Target_Date"])

    return supervised


def _model_candidates(sample_count):
    leaf_size = max(2, min(8, sample_count // 35))
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=leaf_size,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=260,
            min_samples_leaf=leaf_size,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.82,
            random_state=42,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=42,
        ),
        "Robust Ridge": make_pipeline(RobustScaler(), Ridge(alpha=1.6)),
    }


def _fit_predict_models(X_train, y_train, X_test):
    predictions = {}
    fitted_models = {}

    for name, model in _model_candidates(len(X_train)).items():
        try:
            model.fit(X_train, y_train)
            predictions[name] = pd.Series(model.predict(X_test), index=X_test.index, dtype=float)
            fitted_models[name] = model
        except Exception:
            predictions[name] = pd.Series(index=X_test.index, dtype=float)

    return predictions, fitted_models


def _build_transformer_sequences(features, target, sequence_length):
    X_sequences = []
    y_values = []
    row_positions = []
    feature_values = features.to_numpy(dtype=np.float32)
    target_values = target.to_numpy(dtype=np.float32)

    for position in range(sequence_length - 1, len(features)):
        X_sequences.append(feature_values[position - sequence_length + 1 : position + 1])
        y_values.append(target_values[position])
        row_positions.append(position)

    if not X_sequences:
        return None, None, []

    return np.asarray(X_sequences, dtype=np.float32), np.asarray(y_values, dtype=np.float32), row_positions


def _fit_predict_transformer(X_train, y_train, X_test):
    if torch is None or nn is None:
        return pd.Series(index=X_test.index, dtype=float), {"enabled": False, "reason": "PyTorch unavailable"}

    all_features = pd.concat([X_train, X_test])
    all_target = pd.concat([y_train, pd.Series(index=X_test.index, dtype=float)])
    train_count = len(X_train)
    sequence_length = min(TRANSFORMER_SEQUENCE_LENGTH, max(6, train_count // 4))

    if train_count < sequence_length + TRANSFORMER_MIN_TRAIN_SEQUENCES:
        return pd.Series(index=X_test.index, dtype=float), {
            "enabled": False,
            "reason": "Not enough sequence data",
            "sequence_length": sequence_length,
        }

    feature_scaler = RobustScaler()
    scaled_train = feature_scaler.fit_transform(X_train)
    scaled_all = feature_scaler.transform(all_features)
    scaled_features = pd.DataFrame(scaled_all, index=all_features.index, columns=all_features.columns)

    target_mean = float(y_train.mean())
    target_std = float(y_train.std(ddof=0)) or 1.0
    scaled_target = (all_target - target_mean) / target_std

    X_sequences, y_values, row_positions = _build_transformer_sequences(
        scaled_features,
        scaled_target,
        sequence_length,
    )
    if X_sequences is None:
        return pd.Series(index=X_test.index, dtype=float), {"enabled": False, "reason": "No sequences"}

    train_mask = np.asarray(row_positions) < train_count
    test_mask = np.asarray(row_positions) >= train_count

    if train_mask.sum() < TRANSFORMER_MIN_TRAIN_SEQUENCES or test_mask.sum() == 0:
        return pd.Series(index=X_test.index, dtype=float), {
            "enabled": False,
            "reason": "Insufficient train/test sequences",
            "sequence_length": sequence_length,
        }

    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(feature_count=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss()

    X_tensor = torch.tensor(X_sequences[train_mask], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_values[train_mask], dtype=torch.float32, device=device)
    epochs = 22 if train_mask.sum() < 220 else 30
    batch_size = min(32, len(X_tensor))

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(X_tensor), device=device)
        for start in range(0, len(X_tensor), batch_size):
            batch_index = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(X_tensor[batch_index])
            loss = loss_fn(prediction, y_tensor[batch_index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_sequences[test_mask], dtype=torch.float32, device=device)
        scaled_predictions = model(test_tensor).detach().cpu().numpy()

    predictions = (scaled_predictions * target_std) + target_mean
    prediction_positions = np.asarray(row_positions)[test_mask] - train_count
    prediction_index = X_test.index[prediction_positions]
    metadata = {
        "enabled": True,
        "device": str(device),
        "epochs": epochs,
        "sequence_length": sequence_length,
        "train_sequences": int(train_mask.sum()),
    }

    return pd.Series(predictions, index=prediction_index, dtype=float).reindex(X_test.index), metadata


def _fit_predict_automated_bench(X_train, y_train, X_test):
    predictions, fitted_models = _fit_predict_models(X_train, y_train, X_test)
    transformer_predictions, transformer_metadata = _fit_predict_transformer(X_train, y_train, X_test)
    if transformer_metadata.get("enabled") or not transformer_predictions.dropna().empty:
        predictions["PyTorch Transformer"] = transformer_predictions
    return predictions, fitted_models, transformer_metadata


def _weighted_ensemble(prediction_map, metrics):
    eligible = {
        name: values["RMSE"]
        for name, values in metrics.items()
        if name not in {"Benchmark", "ARIMA"} and values.get("RMSE") not in (None, 0)
    }

    if not eligible:
        return None, {}

    inverse_errors = {name: 1 / rmse for name, rmse in eligible.items()}
    total_weight = sum(inverse_errors.values())
    weights = {name: weight / total_weight for name, weight in inverse_errors.items()}
    ensemble = None

    for name, weight in weights.items():
        weighted_prediction = prediction_map[name] * weight
        ensemble = weighted_prediction if ensemble is None else ensemble.add(weighted_prediction, fill_value=0)

    return ensemble, weights


def compare_history_models(df, label="CUSTOM"):
    label = str(label).strip() or "CUSTOM"

    if df is None or df.empty or "Close" not in df.columns:
        return {"Error": "No data available"}

    featured_df = create_features(df)

    if featured_df.empty:
        return {"Error": "Unable to create model features"}

    supervised = _build_supervised_frame(featured_df)

    if len(supervised) < 80:
        return {"Error": "Not enough historical data"}

    feature_columns = [column for column in ADVANCED_FEATURE_COLUMNS if column in supervised.columns]
    features = supervised[feature_columns]
    target = supervised["Target_Close"]
    target_dates = pd.DatetimeIndex(supervised["Target_Date"])

    split = int(len(features) * 0.8)
    if split <= 0 or split >= len(features):
        return {"Error": "Not enough test data"}

    X_train = features[:split]
    X_test = features[split:]

    y_train = target[:split]
    y_test = target[split:]
    close_history = featured_df["Close"].astype(float)
    close_train = close_history.loc[:features.index[split - 1]]

    if len(y_test) == 0:
        return {"Error": "Not enough test data"}

    # ---------------------------
    # BENCHMARK MODEL
    # ---------------------------
    benchmark_pred = featured_df.loc[X_test.index, "Close"].astype(float)
    benchmark_pred = benchmark_pred.astype(float)

    # ---------------------------
    # ARIMA MODEL
    # ---------------------------
    try:
        model = ARIMA(close_train, order=(5, 1, 0))
        model_fit = model.fit()
        arima_forecast = model_fit.forecast(steps=len(y_test))
        arima_pred = pd.Series(np.asarray(arima_forecast, dtype=float), index=y_test.index, dtype=float)
    except Exception:
        arima_pred = pd.Series(index=y_test.index, dtype=float)

    model_predictions, _, transformer_metadata = _fit_predict_automated_bench(X_train, y_train, X_test)
    prediction_map = {
        "Benchmark": pd.Series(benchmark_pred.to_numpy(), index=y_test.index, dtype=float),
        "ARIMA": arima_pred.reindex(y_test.index),
        **{name: prediction.reindex(y_test.index) for name, prediction in model_predictions.items()},
    }

    metric_table = {
        name: _metric_dict(y_test, prediction)
        for name, prediction in prediction_map.items()
    }

    ensemble_pred, ensemble_weights = _weighted_ensemble(prediction_map, metric_table)
    if ensemble_pred is not None:
        prediction_map["Advanced Ensemble"] = ensemble_pred.reindex(y_test.index)
        metric_table["Advanced Ensemble"] = _metric_dict(y_test, prediction_map["Advanced Ensemble"])

    eligible_models = {
        name: values["RMSE"]
        for name, values in metric_table.items()
        if values["RMSE"] is not None
    }
    best_model = min(eligible_models, key=eligible_models.get)

    forecasting_models = {
        name: values["RMSE"]
        for name, values in metric_table.items()
        if name != "Benchmark" and values["RMSE"] is not None
    }
    best_forecasting_model = None
    if forecasting_models:
        best_forecasting_model = min(forecasting_models, key=forecasting_models.get)

    comparison_data = {
        "Date": target_dates[split:].astype(str),
        "Actual": y_test.astype(float).round(4).tolist(),
    }
    for name, prediction in prediction_map.items():
        comparison_data[name] = prediction.reindex(y_test.index).round(4).tolist()

    comparison_frame = pd.DataFrame(comparison_data)

    results = {
        "symbol": label,
        "best_model": best_model,
        "best_forecasting_model": best_forecasting_model,
        "test_points": int(len(y_test)),
        "test_period_start": str(target_dates[split].date()),
        "test_period_end": str(target_dates[-1].date()),
        "feature_count": int(len(feature_columns)),
        "ensemble_weights": {name: round(float(weight), 4) for name, weight in ensemble_weights.items()},
        "transformer": transformer_metadata,
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

    if history.empty or featured_df.empty:
        return {}

    supervised = _build_supervised_frame(featured_df)
    if supervised.empty:
        return {}
    feature_columns = [column for column in ADVANCED_FEATURE_COLUMNS if column in supervised.columns]
    output = {}

    for period in periods:
        label = period["label"]
        start_date = pd.Timestamp(period["start"])
        end_date = pd.Timestamp(period["end"])

        period_close = history.loc[(history.index >= start_date) & (history.index <= end_date), "Close"]
        close_train = history.loc[history.index < start_date, "Close"]
        feat_train = supervised.loc[supervised["Target_Date"] < start_date]
        feat_test = supervised.loc[
            (supervised["Target_Date"] >= start_date) & (supervised["Target_Date"] <= end_date)
        ]

        arima_return = None
        rf_return = None
        transformer_return = None
        ensemble_return = None

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
                    X_train = feat_train[feature_columns]
                    y_train = feat_train["Target_Close"]
                    X_test = feat_test[feature_columns]
                    predictions, _, _ = _fit_predict_automated_bench(X_train, y_train, X_test)
                    rf_forecast = predictions["Random Forest"].to_numpy()
                    transformer_forecast = predictions.get("PyTorch Transformer", pd.Series(dtype=float)).dropna().to_numpy()
                    forecast_matrix = pd.DataFrame(predictions)

                    if not forecast_matrix.empty:
                        ensemble_forecast = forecast_matrix.mean(axis=1)
                        if len(ensemble_forecast):
                            ensemble_return = float((float(ensemble_forecast.iloc[-1]) / start_close) - 1)

                    if len(rf_forecast):
                        rf_return = float((float(rf_forecast[-1]) / start_close) - 1)
                    if len(transformer_forecast):
                        transformer_return = float((float(transformer_forecast[-1]) / start_close) - 1)
                except Exception:
                    rf_return = None
                    transformer_return = None
                    ensemble_return = None

        output[label] = {
            "ARIMA Forecast": arima_return,
            "Random Forest Forecast": rf_return,
            "PyTorch Transformer Forecast": transformer_return,
            "Advanced Ensemble Forecast": ensemble_return,
        }

    return output


@lru_cache(maxsize=32)
def compare_models(symbol):
    symbol = str(symbol).strip().upper()

    df = load_market_history(symbol)

    return compare_history_models(df, symbol)
