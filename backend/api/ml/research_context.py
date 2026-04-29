RESEARCH_CONTEXT = {
    "title": "Big-Data Equity Forecasting Framework",
    "problem": (
        "Equity prices are influenced by price history, volume, sentiment, and macro conditions. "
        "Single-source or price-only models often miss that broader market context."
    ),
    "objectives": [
        "Analyze how price, volume, sentiment, and macro variables relate to equity-market movement.",
        "Compare ARIMA, Random Forest, and a price-only benchmark using common forecast metrics.",
        "Build a practical forecasting workflow that is reproducible inside the dashboard.",
    ],
    "hypotheses": [
        "A multi-source model should reduce forecast error compared with a price-only benchmark.",
        "Random Forest with extended features should match or outperform a price-only ARIMA baseline.",
    ],
    "methodology": [
        "Collect daily OHLCV market data, sentiment data, and macro proxies.",
        "Create engineered features such as moving averages, volatility, RSI, and MACD.",
        "Train ARIMA on price-only history and Random Forest on the richer multi-source feature set.",
        "Evaluate models on the same held-out test period using MAE, RMSE, and R².",
    ],
    "data_sources": [
        "Yahoo Finance for OHLCV and market prices",
        "Financial news headlines for sentiment scoring",
        "Macro proxies such as market returns, volatility, and rates",
    ],
}

