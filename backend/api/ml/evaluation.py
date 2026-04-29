from sklearn.metrics import mean_absolute_error

def compare_models(actual, rf_pred, arima_pred):

    rf_error = mean_absolute_error(actual, rf_pred)
    arima_error = mean_absolute_error(actual, arima_pred)

    return {
        "rf_mae": rf_error,
        "arima_mae": arima_error,
        "better_model": "RF" if rf_error < arima_error else "ARIMA"
    }