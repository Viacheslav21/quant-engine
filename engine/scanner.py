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
    # Geopolitics & conflicts
    "iran":       ["iran","iranian","tehran","nuclear iran","iaea","persian gulf","strait of hormuz"],
    "israel":     ["israel","hamas","gaza","hezbollah","netanyahu","idf","west bank","golan"],
    "ukraine":    ["ukraine","zelensky","donbas","crimea","kherson","zaporizhzhia"],
    "russia":     ["russia","putin","kremlin","moscow","wagner","navalny"],
    "china":      ["china","taiwan","beijing","xi jinping","south china sea","ccp","uyghur"],
    "war":        ["war","attack","strike","invasion","missile","nuclear","military","troops","bomb","drone"],
    "peace":      ["ceasefire","peace","deal","agreement","surrender","truce","negotiations","treaty"],
    "nkorea":     ["north korea","pyongyang","kim jong"],
    "india":      ["india","modi","kashmir","delhi","mumbai"],
    "pakistan":    ["pakistan","islamabad","afghanistan","taliban"],
    "yemen":      ["yemen","houthi","aden","sanaa"],
    "syria":      ["syria","assad","damascus"],

    # US Politics
    "trump":      ["trump","executive order","tariff","maga","mar-a-lago","trump approval","trumps"],
    "biden":      ["biden","white house","kamala","harris"],
    "congress":   ["congress","senate","house of representatives","speaker","filibuster","debt ceiling"],
    "scotus":     ["supreme court","scotus","justice","roe","constitutional"],
    "usgov":      ["doge","government shutdown","federal budget","pentagon","cia","fbi","doj","attorney general",
                   "secretary of state","cabinet","impeach","pardon","classified"],
    "election":   ["election","vote","president","referendum","governor","mayor","minister","parliament",
                   "primary","caucus","midterm","ballot","polling","swing state","electoral",
                   "democratic presidential","republican presidential","win the 2028","win the 2026",
                   "nomination","nominee","running mate"],

    # Commodities & markets
    "oil":        ["oil","opec","crude","brent","wti","petroleum","natural gas","lng"],
    "gold":       ["gold","xau","precious metal","silver","platinum","palladium"],
    "crypto":     ["bitcoin","btc","crypto","ethereum","eth","solana","sol","dogecoin","doge","xrp",
                   "ripple","cardano","polkadot","avalanche","chainlink","defi","nft","stablecoin",
                   "binance","coinbase","memecoin","altcoin","halving"],
    "stocks":     ["s&p","sp500","spx","nasdaq","dow jones","russell","stock market","ipo","earnings",
                   "market cap","fdv","bull market","bear market"],

    # Economy & macro
    "fed":        ["federal reserve","powell","rate cut","rate hike","inflation","cpi","pce",
                   "interest rate","fomc","quantitative","monetary policy","tapering",
                   "fed chair","bessent","shelton"],
    "economy":    ["gdp","unemployment","jobs","recession","nonfarm","payroll","consumer spending",
                   "retail sales","housing","mortgage","debt","deficit","trade balance"],

    # Tech & science
    "tech":       ["ai ","artificial intelligence","openai","anthropic","google","apple","nvidia",
                   "tesla","microsoft","meta","amazon","semiconductor","chip","quantum","robotics"],
    "space":      ["nasa","spacex","rocket","satellite","mars","moon","orbit","launch","starship",
                   "blue origin","artemis","iss"],
    "musk":       ["elon musk","musk","tweet","twitter","x.com","truth social post"],
    "social":     ["post","followers","tiktok",
                   "instagram","youtube","subscribers","views","downloads",
                   "mrbeast","mr beast","pewdiepie","streamer","influencer","viral"],

    # Society
    "health":     ["covid","pandemic","vaccine","fda","who ","health","disease","outbreak",
                   "bird flu","h5n1","monkeypox","drug","pharma","approval"],
    "climate":    ["climate","hurricane","earthquake","wildfire","flood","weather","tornado",
                   "drought","emissions","carbon","paris agreement","cop2"],
    "legal":      ["court","ruling","lawsuit","indictment","trial","verdict","conviction",
                   "acquittal","sentence","extradition","arrest","charged"],
    "film":       ["box office","movie","film","oscar","academy award","opening weekend",
                   "grammy","emmy","golden globe","netflix","disney","streaming"],

    # Regions
    "europe":     ["eu ","european","macron","scholz","starmer","brexit","nato","ecb",
                   "germany","france","uk ","britain","italy","spain","poland","european council"],
    "latam":      ["brazil","lula","mexico","amlo","argentina","milei","venezuela","maduro",
                   "colombia","peru","chile","bolivia","ecuador","cuba"],
    "africa":     ["africa","nigeria","south africa","kenya","ethiopia","egypt","morocco","sahel"],
    "mideast":    ["saudi","mbs","qatar","uae","emirates","bahrain","oman","iraq","baghdad","kurdish"],

    # Other categories
    "sports_off": ["world cup","olympics","fifa","ioc"],  # non-betting sports events
    "culture":    ["pope","vatican","royal family","king charles","queen","celebrity","scandal",
                   "eurovision","music","album","concert","grammy"],
    "education":  ["university","college","student","tuition","scholarship"],
    "transport":  ["boeing","airbus","airline","aviation","faa","shipping","port","suez"],
}

SPORTS_KEYWORDS = [
    # Match patterns
    "vs.", "vs ", "spread:", "o/u ", "over/under", "moneyline",
    "win on 2026", "win on 2025", "win the 2026", "win the 2025",
    "points scored", "map handicap", "map winner", "game winner",
    "total goals", "total points", "total runs", "total sets",
    # Leagues & sports
    "nba", "nfl", "mlb", "nhl", "ncaa", "mls", "pga", "atp", "wta", "lpga",
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1", "eredivisie",
    "champions league", "europa league", "conference league",
    "ufc", "mma", "boxing", "tennis", "golf", "cricket", "rugby", "f1 ",
    "formula 1", "nascar", "indycar", "tour de france", "grand prix",
    "f1 drivers", "fifa world cup", "world cup 2026",
    "win on 2026-", "win on 2025-",  # "Will X FC win on 2026-03-22?"
    "masters tournament", "the open", "pga championship", "ryder cup",
    "miami open", "french open", "wimbledon", "us open", "australian open",
    "roland garros", "indian wells",
    "round of 16", "round of 32", "round of 64", "semifinal", "quarterfinal",
    "sweet 16", "elite eight", "final four",
    "world series", "super bowl", "stanley cup", "nba finals",
    # Esports
    "counter-strike", "dota", "league of legends", "valorant", "blast open",
    "esl ", "iem ", "major ", "bo1", "bo3", "bo5",
    "faze", "navi", "g2 esports", "team liquid", "fnatic", "vitality",
    "t1 ", "gen.g", "cloud9",
    # Football (soccer) teams
    "feyenoord", "manchester city", "manchester united", "real madrid",
    "barcelona", "atletico", "bayern", "psg", "juventus", "inter milan",
    "ac milan", "napoli", "liverpool", "arsenal", "chelsea", "tottenham",
    "borussia", "roma", "lazio", "benfica", "porto", "ajax",
    "olympique", "nottingham", "midtjylland", "san diego fc",
    # US college / pro teams
    "panthers", "razorbacks", "hawkeyes", "gators", "wildcats", "wolverines",
    "bulldogs", "tigers", "eagles", "bears", "lakers", "celtics",
    "warriors", "nets", "yankees", "dodgers", "chiefs", "49ers",
    "timberwolves", "raptors", "blue jackets", "islanders", "bruins",
    "canadiens", "penguins", "capitals", "rangers", "red wings",
    "packers", "cowboys", "patriots", "steelers", "ravens",
    "red sox", "mets", "cubs", "astros", "braves",
    "duke", "gonzaga", "kentucky", "villanova", "baylor",
    "michigan", "arizona", "houston", "marquette", "purdue",
    "st. john", "northern iowa", "high point",
    # Athletes
    "scheffler", "berrettini", "djokovic", "nadal", "sinner", "alcaraz",
    "swiatek", "gauff", "medvedev", "zverev", "rublev", "tsitsipas",
    "lebron", "curry", "durant", "giannis", "jokic", "luka",
    "mahomes", "allen", "lamar", "ohtani", "judge",
    "verstappen", "hamilton", "leclerc", "norris",
    "mcilroy", "koepka", "rahm", "hovland", "morikawa",
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
                if r.status_code == 429:
                    for retry_delay in [2, 5, 10]:
                        log.warning(f"[SCANNER] Rate limited at offset={offset}, waiting {retry_delay}s...")
                        import asyncio
                        await asyncio.sleep(retry_delay)
                        r = await self.client.get(f"{GAMMA_API}/markets", params={
                            "active": "true", "closed": "false",
                            "order": "volume24hr", "ascending": "false",
                            "limit": 100, "offset": offset,
                        })
                        if r.status_code != 429:
                            break
                if r.status_code != 200:
                    raise Exception(f"HTTP {r.status_code} at offset={offset} after retry")
                batch = r.json() or []
                if not batch: break
                for m in batch:
                    vol = float(m.get("volume") or 0)
                    liq = float(m.get("liquidity") or 0)
                    if vol < self.config["MIN_VOLUME"] or liq < 5000:
                        filtered += 1
                        continue
                    # Skip markets not accepting orders (in review / paused)
                    if m.get("acceptingOrders") is False:
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
