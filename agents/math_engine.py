import math
import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("math_engine")

PROSPECT_GAMMA   = 0.65
MIN_BETS_HISTORY = 10
VOLUME_SPIKE_THR = 2.5

def prospect_weight(p: float, gamma: float = PROSPECT_GAMMA) -> float:
    if p <= 0.001: return 0.001
    if p >= 0.999: return 0.999
    num = p ** gamma
    den = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
    return num / den

def prospect_true_price(p_market: float) -> float:
    lo, hi = 0.001, 0.999
    for _ in range(50):
        mid = (lo + hi) / 2
        if prospect_weight(mid) < p_market: lo = mid
        else: hi = mid
    return round((lo + hi) / 2, 4)

def expected_value(p_true: float, price: float) -> float:
    if not (0 < price < 1): return 0.0
    return round(p_true * ((1/price)-1) - (1-p_true), 4)

def kelly_fraction(p_true: float, price: float, fraction: float = 0.25) -> float:
    if not (0 < price < 1) or not (0 < p_true < 1): return 0.0
    b = (1/price)-1
    k = (b*p_true-(1-p_true))/b
    return round(max(0, k*fraction), 4)

def kl_divergence(p: float, q: float) -> float:
    p = max(0.001, min(0.999, p))
    q = max(0.001, min(0.999, q))
    return round(p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q)), 4)

def entropy(p: float) -> float:
    if p <= 0 or p >= 1: return 0.0
    return round(-p*math.log2(p)-(1-p)*math.log2(1-p), 4)

class MathEngine:
    def __init__(self, config: dict, db):
        self.config    = config
        self.db        = db
        self._patterns: dict = {}
        self._vol_history: dict = {}

    async def load_patterns(self):
        self._patterns = await self.db.get_patterns()
        log.info(f"[MATH] Загружено {len(self._patterns)} паттернов")

    def analyze(self, market: dict) -> Optional[dict]:
        p_market = market["yes_price"]
        theme    = market.get("theme","other")

        p_prospect = prospect_true_price(p_market)
        p_history  = self._apply_history(p_market, theme)
        vol_signal, vol_dir = self._volume_signal(market["id"], market.get("volume_24h",0))
        p_time     = self._time_decay(p_market, market.get("end_date"))

        w = {
            "prospect": 0.40,
            "history":  0.35 if p_history else 0.0,
            "volume":   0.15 if vol_signal > 1.5 else 0.0,
            "time":     0.10,
        }
        total_w = sum(w.values()) or 1.0
        p_final = (
            w["prospect"] * p_prospect +
            w["history"]  * (p_history or p_prospect) +
            w["volume"]   * self._vol_adjusted(p_market, vol_signal, vol_dir) +
            w["time"]     * p_time
        ) / total_w
        p_final = max(0.02, min(0.98, p_final))

        if p_final > p_market:
            side, p_side, price_side = "YES", p_final, p_market
        else:
            side, p_side, price_side = "NO", 1-p_final, 1-p_market

        ev    = expected_value(p_side, price_side)
        kl    = kl_divergence(p_final, p_market)
        kelly = kelly_fraction(p_side, price_side)
        edge  = abs(p_final - p_market)

        if ev    < self.config["MIN_EV"]:        return None
        if kl    < self.config["MIN_KL"]:        return None
        if kelly < self.config["MIN_KELLY_FRAC"]: return None
        if edge  < 0.05:                          return None

        return {
            "market_id":  market["id"],
            "question":   market["question"],
            "theme":      theme,
            "url":        market.get("url",""),
            "side":       side,
            "side_price": price_side,
            "p_market":   p_market,
            "p_prospect": p_prospect,
            "p_history":  p_history,
            "p_final":    round(p_final, 4),
            "p_side":     round(p_side, 4),
            "ev":         ev,
            "kl":         kl,
            "kelly":      kelly,
            "entropy":    entropy(p_market),
            "edge":       round(edge, 4),
            "vol_signal": round(vol_signal, 2),
            "vol_dir":    vol_dir,
            "source":     "math",
        }

    def _apply_history(self, p_market: float, theme: str) -> Optional[float]:
        pat = self._patterns.get(theme)
        if not pat or pat.get("sample_size",0) < MIN_BETS_HISTORY: return None
        p_hist = 0.7 * pat["base_rate"] * pat.get("prospect_factor",1.0) + 0.3 * p_market
        return round(max(0.02, min(0.98, p_hist)), 4)

    def _volume_signal(self, market_id: str, volume_24h: float) -> tuple:
        history = self._vol_history.setdefault(market_id, [])
        history.append(volume_24h)
        if len(history) > 48: history.pop(0)
        if len(history) < 3: return 1.0, "neutral"
        avg = sum(history[:-1]) / len(history[:-1])
        if avg <= 0: return 1.0, "neutral"
        ratio = volume_24h / avg
        if ratio > VOLUME_SPIKE_THR:
            direction = "up" if volume_24h > history[-2] else "down"
            log.info(f"[MATH] 📊 Volume spike {market_id[:8]}: {ratio:.1f}x → {direction}")
            return ratio, direction
        return ratio, "neutral"

    def _vol_adjusted(self, p_market: float, vol_ratio: float, direction: str) -> float:
        if vol_ratio < VOLUME_SPIKE_THR or direction == "neutral": return p_market
        if direction == "up":   return min(0.98, p_market*(1+0.1*min(vol_ratio/5,1)))
        return max(0.02, p_market*(1-0.1*min(vol_ratio/5,1)))

    def _time_decay(self, p_market: float, end_date) -> float:
        if not end_date: return p_market
        try:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z","+00:00"))
            days_left = (end_date - datetime.now(timezone.utc)).days
        except Exception:
            return p_market
        if days_left <= 0:  return p_market
        if days_left <= 3:  return p_market*0.95 + prospect_true_price(p_market)*0.05
        if days_left <= 14: return p_market*0.7  + prospect_true_price(p_market)*0.3
        return prospect_true_price(p_market)

    def compute_stake(self, bankroll: float, kelly: float) -> float:
        stake = bankroll * kelly
        return round(max(1.0, min(stake, bankroll*self.config["MAX_KELLY_FRAC"])), 2)
