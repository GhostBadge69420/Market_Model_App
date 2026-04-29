from html import unescape
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests

SAFE_SYMBOL = re.compile(r"^[A-Z0-9.\-_=^]{1,20}$")
STRIP_MARKET_SUFFIX = re.compile(r"(\.NS|\.BO|-USD|=X|=F)$")
MAX_NEWS_ITEMS = 8
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def _build_google_news_query(symbol):
    normalized_symbol = str(symbol).strip().upper()
    base_symbol = STRIP_MARKET_SUFFIX.sub("", normalized_symbol)

    if normalized_symbol.endswith("=X"):
        return f'"{base_symbol}" forex OR currency market'
    if normalized_symbol.endswith("-USD"):
        return f'"{base_symbol}" crypto OR cryptocurrency market'
    if normalized_symbol.endswith("=F"):
        return f'"{base_symbol}" commodity OR futures market'
    if normalized_symbol.startswith("^"):
        return f'"{base_symbol[1:]}" index market'

    return f'"{base_symbol}" stock OR shares OR market'


def _clean_headline(text):
    cleaned = unescape(str(text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def get_news(symbol):
    normalized_symbol = str(symbol).strip().upper()
    if not SAFE_SYMBOL.fullmatch(normalized_symbol):
        return []

    query = _build_google_news_query(normalized_symbol)
    url = (
        f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}"
        "&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError):
        return []

    headlines = []
    for item in root.findall(".//item"):
        title = _clean_headline(item.findtext("title"))
        if not title or title in headlines:
            continue
        headlines.append(title)
        if len(headlines) >= MAX_NEWS_ITEMS:
            break

    return headlines
