from django.test import TestCase
from django.urls import reverse

from api.ml.decision import get_decision
from api.ml.features import create_features
from api.ml.sentiment import sentiment_breakdown, sentiment_score


class MlHelpersTests(TestCase):
    def test_create_features_returns_rows_after_warmup(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "Close": [100 + idx for idx in range(80)],
                "Volume": [1000 + idx * 10 for idx in range(80)],
            }
        )

        featured = create_features(df)

        self.assertFalse(featured.empty)
        self.assertTrue({"MA20", "MA50", "Volatility", "Volume_MA"}.issubset(featured.columns))

    def test_sentiment_helpers_stay_in_expected_range(self):
        score = sentiment_score(["Markets rally on strong earnings", "Investors remain cautious"])
        breakdown = sentiment_breakdown(score)

        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertEqual(
            round(
                breakdown["bullish_percent"]
                + breakdown["bearish_percent"]
                + breakdown["neutral_percent"],
                2,
            ),
            100.0,
        )

    def test_decision_handles_invalid_current_price(self):
        decision, confidence, market = get_decision(0, 120, 75)

        self.assertEqual(decision, "HOLD")
        self.assertEqual(confidence, 0)
        self.assertEqual(market, "NEUTRAL")

    def test_news_endpoint_rejects_invalid_symbols(self):
        response = self.client.get(reverse("news_sentiment", kwargs={"symbol": "BAD$"}))

        self.assertEqual(response.status_code, 400)
