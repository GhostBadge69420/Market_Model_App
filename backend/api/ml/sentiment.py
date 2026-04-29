from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def sentiment_breakdown(score):
    score = max(0, min(100, score))
    bullish = max(0, score - 50) * 2
    bearish = max(0, 50 - score) * 2
    neutral = max(0, 100 - bullish - bearish)

    return {
        "bullish_percent": round(bullish, 2),
        "bearish_percent": round(bearish, 2),
        "neutral_percent": round(neutral, 2)
    }


def sentiment_score(news_list):
    if not news_list:
        return 50  # neutral

    scores = []

    for text in news_list:
        score = analyzer.polarity_scores(text)["compound"]
        scores.append(score)

    avg = sum(scores) / len(scores)

    # convert -1 → 1 into 0 → 100
    return round(max(0, min(100, (avg + 1) * 50)), 2)
