import math
import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("math_engine")

PROSPECT_GAMMA   = 0.65
MIN_BETS_HISTORY = 10
VOLUME_SPIKE_THR = 2.5

# --- Core math functions ---

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

def prob_to_logodds(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))

def logodds_to_prob(lo: float) -> float:
    return 1 / (1 + math.exp(-lo))

# --- Bayesian fusion: combine evidence in log-odds space ---

def bayesian_update(prior: float, *evidence_probs) -> float:
    """Combine prior with independent evidence using log-odds (Bayesian fusion).
    Each evidence_prob is an independent estimate of P(YES).
    Prior is typically the market price."""
    lo = prob_to_logodds(prior)
    base_lo = prob_to_logodds(0.5)  # uninformative prior in log-odds
    for p_ev in evidence_probs:
        if p_ev is None:
            continue
        # Each evidence contributes its log-likelihood ratio vs base rate
        lo += prob_to_logodds(p_ev) - base_lo
    return max(0.02, min(0.98, logodds_to_prob(lo)))


class MathEngine:
    def __init__(self, config: dict, db, calibrator=None):
        self.config      = config
        self.db          = db
        self.calibrator  = calibrator
        self._patterns: dict = {}
        self._vol_history: dict = {}
        self._price_cache: dict = {}  # market_id -> list of recent prices

    async def load_patterns(self):
        self._patterns = await self.db.get_patterns()
        log.info(f"[MATH] Загружено {len(self._patterns)} паттернов")

    def analyze(self, market: dict) -> Optional[dict]:
        p_market = market["yes_price"]
        theme    = market.get("theme","other")

        # 1. Prospect theory — invert human probability weighting
        p_prospect = prospect_true_price(p_market)

        # 2. Historical base rate for this theme
        p_history = self._apply_history(p_market, theme)

        # 3. Volume spike detection
        vol_signal, vol_dir = self._volume_signal(market["id"], market.get("volume_24h",0))
        p_volume = self._vol_adjusted(p_market, vol_signal, vol_dir) if vol_signal > 1.5 else None

        # 4. Time decay — markets converge to truth near expiry
        p_time = self._time_decay(p_market, market.get("end_date"))

        # 5. Price momentum — trend from recent snapshots
        p_momentum = self._price_momentum(market["id"], p_market)

        # --- Bayesian fusion in log-odds space ---
        # Prior: prospect-adjusted market price
        # Evidence: history, volume, time, momentum
        evidence = [e for e in [p_history, p_volume, p_time, p_momentum] if e is not None]
        if evidence:
            p_final = bayesian_update(p_prospect, *evidence)
        else:
            p_final = p_prospect

        # 6. Apply calibration correction if available
        if self.calibrator:
            p_final = self.calibrator.adjust(p_final)

        p_final = max(0.02, min(0.98, round(p_final, 4)))

        if p_final > p_market:
            side, p_side, price_side = "YES", p_final, p_market
        else:
            side, p_side, price_side = "NO", 1-p_final, 1-p_market

        ev    = expected_value(p_side, price_side)
        kl    = kl_divergence(p_final, p_market)
        kelly = kelly_fraction(p_side, price_side)
        edge  = abs(p_final - p_market)

        if ev < self.config["MIN_EV"]:
            log.debug(f"[MATH] Rejected {market['id'][:8]}: EV {ev:.4f} < {self.config['MIN_EV']}")
            return None
        if kl < self.config["MIN_KL"]:
            log.debug(f"[MATH] Rejected {market['id'][:8]}: KL {kl:.4f} < {self.config['MIN_KL']}")
            return None
        if kelly < self.config["MIN_KELLY_FRAC"]:
            log.debug(f"[MATH] Rejected {market['id'][:8]}: Kelly {kelly:.4f} < {self.config['MIN_KELLY_FRAC']}")
            return None
        if edge < 0.05:
            log.debug(f"[MATH] Rejected {market['id'][:8]}: Edge {edge:.4f} < 0.05")
            return None

        log.info(f"[MATH] Signal: {side} '{market['question'][:50]}' EV:+{ev*100:.1f}% Kelly:{kelly*100:.1f}% Edge:{edge*100:.1f}%")
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
            "p_momentum": p_momentum,
            "p_final":    p_final,
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
            # Compare to average, not just previous point (less noisy)
            direction = "up" if volume_24h > avg * 1.2 else "down"
            log.info(f"[MATH] Volume spike {market_id[:8]}: {ratio:.1f}x avg → {direction}")
            return ratio, direction
        return ratio, "neutral"

    def _vol_adjusted(self, p_market: float, vol_ratio: float, direction: str) -> float:
        if vol_ratio < VOLUME_SPIKE_THR or direction == "neutral": return p_market
        strength = min(0.10, 0.05 * (vol_ratio / VOLUME_SPIKE_THR))
        if direction == "up":   return min(0.98, p_market * (1 + strength))
        return max(0.02, p_market * (1 - strength))

    def _time_decay(self, p_market: float, end_date) -> float:
        if not end_date: return p_market
        try:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z","+00:00"))
            days_left = (end_date - datetime.now(timezone.utc)).days
        except Exception:
            return p_market
        if days_left <= 0:  return p_market
        # Smooth decay: blend market → prospect as time increases
        # Near expiry: trust market price more (it's converging to truth)
        # Far from expiry: prospect theory has more room to add value
        decay = min(1.0, max(0.0, (days_left - 3) / 30))
        return p_market * (1 - decay) + prospect_true_price(p_market) * decay

    def _price_momentum(self, market_id: str, current_price: float) -> Optional[float]:
        """Track price trend from recent snapshots. Rising prices = more YES signal."""
        history = self._price_cache.setdefault(market_id, [])
        history.append(current_price)
        if len(history) > 30: history.pop(0)
        if len(history) < 5: return None

        # Linear regression slope over recent prices
        n = len(history)
        x_mean = (n - 1) / 2
        y_mean = sum(history) / n
        num = sum((i - x_mean) * (history[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den == 0: return None
        slope = num / den

        # Slope per tick → extrapolated probability shift
        # Positive slope = price trending up = YES more likely
        # Cap at ±5% adjustment
        momentum_shift = max(-0.05, min(0.05, slope * n * 0.5))
        p_mom = current_price + momentum_shift
        return round(max(0.02, min(0.98, p_mom)), 4)

    def compute_stake(self, bankroll: float, kelly: float) -> float:
        stake = bankroll * kelly
        stake = round(max(1.0, min(stake, bankroll*self.config["MAX_KELLY_FRAC"])), 2)
        log.info(f"[MATH] Stake: ${stake:.2f} (kelly={kelly*100:.1f}% bankroll=${bankroll:.2f})")
        return stake
