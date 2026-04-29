from django.urls import path
from .views import news_sentiment

urlpatterns = [
    path("news/<str:symbol>/", news_sentiment, name="news_sentiment"),
]
