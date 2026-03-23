import json as _json
import logging
import httpx
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("scanner")

GAMMA_API = "https://gamma-api.polymarket.com"


def _parse_token_ids(m: dict) -> tuple:
    """Extract YES and NO token IDs from market data."""
    token_ids = m.get("clobTokenIds") or []
    if isinstance(token_ids, str):
        token_ids = _json.loads(token_ids)
    yes_token = token_ids[0] if len(token_ids) > 0 else None
    no_token = token_ids[1] if len(token_ids) > 1 else None
    return yes_token, no_token

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

SPORTS_KEYWORDS = [
    # Match patterns
    "vs.", "vs ", "spread:", "o/u ", "over/under", "moneyline",
    "win on 2026", "win on 2025", "win the 2026", "win the 2025",
    # Leagues
    "nba", "nfl", "mlb", "nhl", "ncaa", "mls", "pga", "atp", "wta",
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    "champions league", "europa league",
    "ufc", "mma", "boxing", "tennis", "golf", "masters tournament",
    "miami open", "french open", "wimbledon", "us open",
    "round of", "semifinal", "quarterfinal",
    # Esports
    "counter-strike", "dota", "league of legends", "valorant", "blast open",
    # Teams / athletes
    "panthers", "razorbacks", "hawkeyes", "gators", "wildcats", "wolverines",
    "bulldogs", "tigers", "eagles", "bears", "lakers", "celtics",
    "warriors", "nets", "yankees", "dodgers", "chiefs", "49ers",
    "timberwolves", "raptors", "blue jackets", "islanders",
    "feyenoord", "manchester city", "real madrid", "atletico",
    "san diego fc", "lazio",
    "scheffler", "berrettini", "djokovic", "nadal", "sinner",
]


def is_sports(question: str) -> bool:
    lower = question.lower()
    return any(kw in lower for kw in SPORTS_KEYWORDS)


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
                    # Skip sports/esports markets
                    if self.config.get("SKIP_SPORTS", True) and is_sports(m.get("question", "")):
                        filtered += 1
                        continue
                    raw_prices = m.get("outcomePrices") or ["0.5","0.5"]
                    if isinstance(raw_prices, str):
                        raw_prices = _json.loads(raw_prices)
                    yes_price = float(raw_prices[0])
                    no_price  = float(raw_prices[1]) if len(raw_prices) > 1 else 1 - yes_price
                    if yes_price > 0.97 or yes_price < 0.03:
                        filtered += 1
                        continue
                    yes_token, no_token = _parse_token_ids(m)
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
                        "volume_1wk":float(m.get("volume1wk") or 0),
                        "volume_1mo":float(m.get("volume1mo") or 0),
                        "liquidity": liq,
                        "spread":    float(m.get("spread") or 0),
                        "best_ask":  float(m.get("bestAsk") or yes_price),
                        "competitive": float(m.get("competitive") or 0),
                        "price_change_1wk": float(m.get("oneWeekPriceChange") or 0),
                        "price_change_1mo": float(m.get("oneMonthPriceChange") or 0),
                        "neg_risk":  bool(m.get("negRisk")),
                        "neg_risk_market_id": m.get("negRiskMarketID") or "",
                        "end_date":  end_date,
                        "theme":     detect_theme(m.get("question","")),
                        "url":       f"https://polymarket.com/event/{url_slug}",
                        "yes_token": yes_token,
                        "no_token":  no_token,
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
            data = r.json()
            if data:
                yes_token, no_token = _parse_token_ids(data)
                data["yes_token"] = yes_token
                data["no_token"] = no_token
            return data
        except Exception as e:
            log.error(f"[SCANNER] get_market {market_id}: {e}")
            return None

    async def close(self):
        await self.client.aclose()
