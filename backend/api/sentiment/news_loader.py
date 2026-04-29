import re

import requests

SAFE_SYMBOL = re.compile(r"^[A-Z0-9.\-_=^]{1,20}$")


def fetch_news(symbol):
    normalized_symbol = str(symbol).strip().upper()
    if not SAFE_SYMBOL.fullmatch(normalized_symbol):
        return []

    url = f"http://127.0.0.1:8000/api/news/{normalized_symbol}/"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return []

    return data.get("news", [])
