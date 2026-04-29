from transformers import pipeline

sentiment_model = None


def get_sentiment_model():
    global sentiment_model

    if sentiment_model is None:
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )

    return sentiment_model

def analyze_sentiment(text):
    try:
        model = get_sentiment_model()
        result = model(text)[0]
    except Exception:
        return 0.0

    label = result["label"]   # positive / negative / neutral
    score = result["score"]

    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0
