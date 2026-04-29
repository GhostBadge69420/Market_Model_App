import re

from django.http import JsonResponse
from .ml.news import get_news
from .ml.sentiment import sentiment_score, sentiment_breakdown

SAFE_SYMBOL = re.compile(r"^[A-Z0-9.\-_=^]{1,20}$")


def news_sentiment(request, symbol):
    normalized_symbol = str(symbol).strip().upper()
    if not SAFE_SYMBOL.fullmatch(normalized_symbol):
        return JsonResponse({"error": "Invalid symbol format."}, status=400)

    news = get_news(normalized_symbol)
    score = sentiment_score(news)
    breakdown = sentiment_breakdown(score)

    return JsonResponse({
        "symbol": normalized_symbol,
        "news": news,
        "sentiment_score": score,
        "market_sentiment": breakdown
    })
