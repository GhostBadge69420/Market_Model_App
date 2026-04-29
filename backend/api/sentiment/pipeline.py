from .analyzer import analyze_sentiment
from .news_loader import fetch_news


def get_news_sentiment(symbol):
    news_list = fetch_news(symbol)

    if not news_list:
        return 0

    scores = []

    for headline in news_list:
        try:
            score = analyze_sentiment(headline)
            scores.append(score)
        except:
            continue

    if len(scores) == 0:
        return 0

    return round(sum(scores) / len(scores), 4)
