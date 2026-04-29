def get_decision(current, predicted, sentiment):
    if current in (None, 0) or predicted is None or sentiment is None:
        return "HOLD", 0, "NEUTRAL"

    change = (predicted - current) / current
    clamped_sentiment = max(0, min(100, sentiment))
    confidence = abs(change) * 100 + clamped_sentiment * 0.4

    if change > 0.02 and clamped_sentiment > 55:
        return "BUY", min(confidence, 95), "BULLISH"

    elif change < -0.02 and clamped_sentiment < 45:
        return "SELL", min(confidence, 95), "BEARISH"

    else:
        return "HOLD", min(confidence, 70), "NEUTRAL"
