import logging
import httpx
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("scanner")

GAMMA_API = "https://gamma-api.polymarket.com"

THEME_KEYWORDS = {
    "iran":     ["iran","iranian","tehran","nuclear iran","iaea"],
    "oil":      ["oil","opec","crude","brent","wti","petroleum"],
    "war":      ["war","attack","strike","invasion","missile","nuclear"],
    "peace":    ["ceasefire","peace","deal","agreement","surrender"],
    "ukraine":  ["ukraine","zelensky","donbas","crimea"],
    "russia":   ["russia","putin","kremlin","moscow"],
    "crypto":   ["bitcoin","btc","crypto","ethereum","blockchain"],
    "fed":      ["federal reserve","powell","rate","inflation","cpi"],
    "china":    ["china","taiwan","beijing","xi jinping"],
    "trump":    ["trump","executive order","tariff","maga"],
    "gold":     ["gold","xau","precious metal"],
    "election": ["election","vote","president","congress","senate"],
    "israel":   ["israel","hamas","gaza","hezbollah","netanyahu"],
}

def _parse_end_date(raw) -> Optional[datetime]:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None

def detect_theme(question: str) -> str:
    lower = question.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return theme
    return "other"

class PolymarketScanner:
    def __init__(self, config: dict):
        self.config = config
        self.client = httpx.AsyncClient(timeout=15.0)

    async def fetch(self) -> list:
        try:
            markets = []
            filtered = 0
            offset  = 0
            while len(markets) < 500:
                r = await self.client.get(f"{GAMMA_API}/markets", params={
                    "active": "true", "closed": "false",
                    "order": "volume24hr", "ascending": "false",
                    "limit": 100, "offset": offset,
                })
                batch = r.json() or []
                if not batch: break
                for m in batch:
                    vol = float(m.get("volume") or 0)
                    liq = float(m.get("liquidity") or 0)
                    if vol < self.config["MIN_VOLUME"] or liq < 5000:
                        filtered += 1
                        continue
                    raw_prices = m.get("outcomePrices") or ["0.5","0.5"]
                    if isinstance(raw_prices, str):
                        import json as _json
                        raw_prices = _json.loads(raw_prices)
                    yes_price = float(raw_prices[0])
                    no_price  = float(raw_prices[1]) if len(raw_prices) > 1 else 1 - yes_price
                    if yes_price > 0.97 or yes_price < 0.03:
                        filtered += 1
                        continue
                    end_date = _parse_end_date(m.get("endDate"))
                    # URL: use event slug if available, fall back to market slug
                    events = m.get("events") or []
                    event_slug = events[0].get("slug", "") if events else ""
                    url_slug = event_slug or m.get("slug", "")
                    markets.append({
                        "id":        m["id"],
                        "slug":      m.get("slug",""),
                        "question":  m.get("question",""),
                        "yes_price": round(yes_price, 4),
                        "no_price":  round(no_price, 4),
                        "volume":    vol,
                        "volume_24h":float(m.get("volume24hr") or 0),
                        "liquidity": liq,
                        "end_date":  end_date,
                        "theme":     detect_theme(m.get("question","")),
                        "url":       f"https://polymarket.com/event/{url_slug}",
                    })
                offset += 100
                if len(batch) < 100: break
            log.info(f"[SCANNER] {len(markets)} рынков (filtered out: {filtered})")
            return markets
        except Exception as e:
            log.error(f"[SCANNER] {e}")
            return []

    async def get_market(self, market_id: str) -> dict | None:
        try:
            r = await self.client.get(f"{GAMMA_API}/markets/{market_id}")
            return r.json()
        except Exception as e:
            log.error(f"[SCANNER] get_market {market_id}: {e}")
            return None

    async def close(self):
        await self.client.aclose()
