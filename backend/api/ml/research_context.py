RESEARCH_CONTEXT = {
    "title": "Big-Data Equity Forecasting Framework",
    "problem": (
        "Equity prices are influenced by price history, volume, sentiment, and macro conditions. "
        "Single-source or price-only models often miss that broader market context."
    ),
    "objectives": [
        "Analyze how price, volume, sentiment, and macro variables relate to equity-market movement.",
        "Compare ARIMA, tree ensembles, boosted models, regularized regression, a PyTorch Transformer, and a price-only benchmark using common forecast metrics.",
        "Build a practical forecasting workflow that is reproducible inside the dashboard.",
    ],
    "hypotheses": [
        "A multi-source model should reduce forecast error compared with a price-only benchmark.",
        "An ensemble of nonlinear models and sequence-aware Transformer forecasts should match or outperform a price-only ARIMA baseline.",
    ],
    "methodology": [
        "Collect daily OHLCV market data, sentiment data, and macro proxies.",
        "Create engineered features such as moving averages, volatility regimes, Bollinger position, RSI, MACD, momentum, gaps, and volume ratios.",
        "Train ARIMA on price-only history, multiple machine-learning regressors on the richer feature set, and an automated PyTorch Transformer on rolling feature sequences.",
        "Evaluate models on the same held-out test period using MAE, RMSE, R², MAPE, and directional accuracy.",
    ],
    "data_sources": [
        "Yahoo Finance for OHLCV and market prices",
        "Financial news headlines for sentiment scoring",
        "Macro proxies such as market returns, volatility, and rates",
    ],
}
