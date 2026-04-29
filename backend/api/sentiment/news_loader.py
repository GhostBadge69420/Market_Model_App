from api.ml.news import get_news


def fetch_news(symbol):
    try:
        return get_news(symbol)
    except Exception:
        return []
