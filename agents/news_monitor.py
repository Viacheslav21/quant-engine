import logging
import httpx
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

log = logging.getLogger("news")

RSS_FEEDS = [
    # ✅ Работающие публичные RSS фиды (2026)
    {"url": "http://feeds.bbci.co.uk/news/world/rss.xml",             "source": "BBC"},
    {"url": "https://www.aljazeera.com/xml/rss/all.xml",              "source": "AlJazeera"},
    {"url": "https://www.theguardian.com/world/rss",                  "source": "Guardian"},
    {"url": "https://feeds.npr.org/1001/rss.xml",                     "source": "NPR"},
    {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",  "source": "CNBC"},
    {"url": "https://www.cnbc.com/id/10000664/device/rss/rss.html",   "source": "CNBC_Politics"},
    {"url": "https://feeds.skynews.com/feeds/rss/world.xml",          "source": "SkyNews"},
    {"url": "https://rss.dw.com/rss/en-all",                          "source": "DeutscheWelle"},
]

THEME_KEYWORDS = {
    "iran":     ["iran","iranian","tehran","nuclear","hormuz","iaea"],
    "oil":      ["opec","crude oil","oil price","petroleum","brent","wti"],
    "war":      ["war","attack","strike","invasion","missile","airstrike","ceasefire"],
    "ukraine":  ["ukraine","zelensky","donbas","crimea","nato","russia"],
    "crypto":   ["bitcoin","cryptocurrency","btc","ethereum","crypto"],
    "fed":      ["federal reserve","powell","interest rate","inflation","rate cut"],
    "china":    ["china","taiwan","beijing","xi jinping"],
    "trump":    ["trump","white house","executive order","tariff"],
    "election": ["election","vote","ballot","congress","senate"],
    "gold":     ["gold price","xau","precious metal","safe haven"],
    "israel":   ["israel","hamas","gaza","hezbollah","netanyahu"],
}

BULLISH = ["deal","agreement","ceasefire","peace","victory","signed","approved"]
BEARISH = ["attack","war","crisis","collapse","failed","rejected","sanction","threat"]

def detect_theme(text: str) -> str:
    lower = text.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return theme
    return "other"

def detect_sentiment(text: str) -> str:
    lower = text.lower()
    bull  = sum(1 for w in BULLISH if w in lower)
    bear  = sum(1 for w in BEARISH if w in lower)
    if bull > bear: return "bullish"
    if bear > bull: return "bearish"
    return "neutral"

def extract_keywords(text: str) -> list:
    lower = text.lower()
    found = []
    for keywords in THEME_KEYWORDS.values():
        for kw in keywords:
            if kw in lower:
                found.append(kw)
    return list(set(found))[:10]

def parse_date(date_str: Optional[str]) -> datetime:
    if not date_str:
        return datetime.now(timezone.utc)
    try:
        return parsedate_to_datetime(date_str).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

class NewsMonitor:
    def __init__(self, db):
        self.db     = db
        self.client = httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; QuantBot/1.0)"}
        )

    async def scan(self) -> list:
        new_items = []
        for feed in RSS_FEEDS:
            try:
                items = await self._fetch_feed(feed["url"], feed["source"])
                for item in items:
                    is_new = await self.db.save_news(item)
                    if is_new:
                        new_items.append(item)
            except Exception as e:
                log.warning(f"[NEWS] {feed['source']}: {e}")
        if new_items:
            log.info(f"[NEWS] 🆕 {len(new_items)} новых новостей")
        return new_items

    async def _fetch_feed(self, url: str, source: str) -> list:
        try:
            r = await self.client.get(url, follow_redirects=True)
            if r.status_code != 200:
                log.warning(f"[NEWS] {source}: HTTP {r.status_code}")
                return []
            root  = ET.fromstring(r.text)
        except ET.ParseError as e:
            log.warning(f"[NEWS] {source}: XML parse error {e}")
            return []
        except Exception as e:
            log.warning(f"[NEWS] {source}: {e}")
            return []

        items = []
        for item in root.iter("item"):
            title = item.findtext("title","").strip()
            link  = item.findtext("link","").strip()
            date  = item.findtext("pubDate","")
            if not title or not link: continue
            theme = detect_theme(title)
            if theme == "other": continue
            items.append({
                "source":       source,
                "title":        title[:500],
                "url":          link[:1000],
                "keywords":     extract_keywords(title),
                "theme":        theme,
                "sentiment":    detect_sentiment(title),
                "published_at": parse_date(date),
            })
        return items

    async def find_relevant_markets(self, news_item: dict, markets: list, max_matches: int = 5) -> list:
        theme    = news_item["theme"]
        keywords = news_item.get("keywords", [])
        # Filter to specific keywords (3+ chars) to avoid generic matches like "war"
        specific_kw = [kw for kw in keywords if len(kw) >= 4]
        scored = []
        for market in markets:
            m_text       = market["question"].lower()
            theme_match  = market.get("theme") == theme
            # Require BOTH theme match AND at least one keyword in market question
            kw_hits      = sum(1 for kw in specific_kw if kw in m_text)
            if not (theme_match and kw_hits >= 1):
                continue
            scored.append((kw_hits, market))
        # Sort by relevance (more keyword hits = better match), take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        relevant = []
        for kw_hits, market in scored[:max_matches]:
            price_unchanged = await self._price_unchanged(market["id"])
            if price_unchanged:
                log.info(f"[NEWS] Match ({kw_hits}kw): '{news_item['title'][:60]}' → '{market['question'][:60]}'")
                relevant.append({
                    **market,
                    "news_sentiment": news_item["sentiment"],
                    "news_title":     news_item["title"],
                })
        if relevant:
            log.info(f"[NEWS] Found {len(relevant)} relevant markets for theme={theme}")
        return relevant

    async def _price_unchanged(self, market_id: str) -> bool:
        try:
            async with self.db.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT yes_price FROM price_snapshots
                    WHERE market_id=$1 AND snapshot_at > NOW() - INTERVAL '10 minutes'
                    ORDER BY snapshot_at DESC LIMIT 5
                """, market_id)
                if len(rows) < 2: return True
                prices = [r["yes_price"] for r in rows]
                return abs(prices[0] - prices[-1]) < 0.02
        except Exception:
            return True

    async def close(self):
        await self.client.aclose()
