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

def kelly_fraction(p_true: float, price: float, fraction: float = 0.15) -> float:
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

def bayesian_update(prior: float, evidence_with_weights: list) -> float:
    """Combine prior with weighted evidence using log-odds (Bayesian fusion).
    evidence_with_weights: list of (p_evidence, weight) tuples.
    Weight < 1.0 discounts correlated or weak evidence."""
    lo = prob_to_logodds(prior)
    base_lo = prob_to_logodds(0.5)
    for item in evidence_with_weights:
        if item is None:
            continue
        p_ev, weight = item
        if p_ev is None:
            continue
        # Each evidence contributes its log-likelihood ratio, scaled by weight
        lo += (prob_to_logodds(p_ev) - base_lo) * weight
    return max(0.02, min(0.98, logodds_to_prob(lo)))


class MathEngine:
    def __init__(self, config: dict, db, calibrator=None):
        self.config      = config
        self.db          = db
        self.calibrator  = calibrator
        self._patterns: dict = {}
        self._vol_history: dict = {}
        self._price_cache: dict = {}  # market_id -> list of recent prices (30 points, ~5 min)
        self._long_price_cache: dict = {}  # market_id -> list of recent prices (180 points, ~30 min)
        self._neg_risk_groups: dict = {}  # neg_risk_market_id -> [market dicts] for arbitrage
        self._ml_url = config.get("ML_API_URL")  # e.g. http://quant-ml.railway.internal:8080
        self._ml_client = None

    def _get_ml_client(self):
        if self._ml_client is None and self._ml_url:
            import httpx
            self._ml_client = httpx.AsyncClient(timeout=3.0)
        return self._ml_client

    async def ml_predict(self, market: dict) -> dict | None:
        """Call ML API for prediction. Returns {p_yes, p_mispriced} or None."""
        client = self._get_ml_client()
        if not client:
            return None
        try:
            from datetime import datetime, timezone
            days_to_expiry = 30
            end_date = market.get("end_date")
            if end_date:
                if isinstance(end_date, str):
                    end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                days_to_expiry = max(0, (end_date - datetime.now(timezone.utc)).days)

            r = await client.get(f"{self._ml_url}/predict", params={
                "yes_price": market["yes_price"],
                "theme": market.get("theme", "other"),
                "volume": market.get("volume", 0),
                "days_to_expiry": days_to_expiry,
                "volume_per_day": market.get("volume_24h", 0),
                "neg_risk": market.get("neg_risk", False),
                "question_length": len(market.get("question", "")),
                "has_numbers": any(c.isdigit() for c in market.get("question", "")),
                "spread": market.get("spread", 0),
            })
            return r.json()
        except Exception as e:
            log.debug(f"[ML] Predict failed: {e}")
            return None

    async def load_patterns(self):
        self._patterns = await self.db.get_patterns()
        log.info(f"[MATH] Загружено {len(self._patterns)} паттернов")

    def analyze(self, market: dict) -> Optional[dict]:
        p_market = market["yes_price"]
        theme    = market.get("theme","other")

        # 0. Skip markets too far in the future (don't freeze capital)
        max_days = int(self.config.get("MAX_MARKET_DAYS", 90))
        end_date = market.get("end_date")
        if end_date:
            try:
                if isinstance(end_date, str):
                    from datetime import datetime as _dt, timezone as _tz
                    end_date = _dt.fromisoformat(end_date.replace("Z", "+00:00"))
                days_left = (end_date - datetime.now(timezone.utc)).days
                if days_left > max_days:
                    return None
            except Exception:
                pass

        # 1. Prospect theory — invert human probability weighting (configurable)
        use_prospect = self.config.get("USE_PROSPECT", True)
        p_prospect = prospect_true_price(p_market) if use_prospect else p_market

        # 2. Historical base rate for this theme
        p_history = self._apply_history(p_market, theme)

        # 3. Volume spike detection
        vol_signal, vol_dir = self._volume_signal(market["id"], market.get("volume_24h",0))
        p_volume = self._vol_adjusted(p_market, vol_signal, vol_dir) if vol_signal > VOLUME_SPIKE_THR else None

        # 4. Time decay — markets converge to truth near expiry
        p_time = self._time_decay(p_market, market.get("end_date"))

        # 5. Price momentum — trend from recent snapshots
        p_momentum = self._price_momentum(market["id"], p_market)

        # 6. Mean reversion / contrarian signal
        # Update long price cache from short cache overflow
        long_hist = self._long_price_cache.setdefault(market["id"], [])
        long_hist.append(p_market)
        if len(long_hist) > 180:
            long_hist.pop(0)

        p_contrarian, contrarian_conf = self._mean_reversion(market["id"], p_market)

        # 7. Long-term momentum from API (week/month price changes)
        p_long_mom = self._long_momentum(market)

        # 8. Volume trend (24h vs weekly average)
        p_vol_trend = self._volume_trend(market)

        # 9. NegRisk arbitrage (multi-outcome events)
        p_arb = self._neg_risk_arb(market)

        # 10. Order book imbalance — buy/sell pressure from WS book events
        p_book = self._book_imbalance(market)

        # --- Bayesian fusion in log-odds space ---
        # Merge correlated pairs BEFORE fusion (one source each, not two at 0.5)
        # Momentum: pick stronger of short-term vs long-term
        if p_momentum is not None and p_long_mom is not None:
            p_mom_combined = p_momentum if abs(p_momentum - p_market) >= abs(p_long_mom - p_market) else p_long_mom
        else:
            p_mom_combined = p_momentum or p_long_mom

        # Volume: pick stronger of spike vs trend
        if p_volume is not None and p_vol_trend is not None:
            p_vol_combined = p_volume if abs(p_volume - p_market) >= abs(p_vol_trend - p_market) else p_vol_trend
        else:
            p_vol_combined = p_volume or p_vol_trend

        # 7 independent sources after de-duplication
        # time_decay at 0.5 (partially priced into market), book at 0.5 (short-term signal)
        evidence = [
            (p_history, 1.0),
            (p_vol_combined, 1.0),
            (p_time, 0.5),
            (p_mom_combined, 1.0),
            (p_contrarian, 1.0),
            (p_arb, 1.0),
            (p_book, 0.5),
        ]
        active_evidence = [(p, w) for p, w in evidence if p is not None]

        if active_evidence:
            p_final = bayesian_update(p_prospect, active_evidence)
        else:
            p_final = p_prospect

        # Apply calibration correction if available
        if self.calibrator:
            p_final = self.calibrator.adjust(p_final)

        # Adaptive drift cap: more evidence = wider cap
        # 0-1 sources: ±12%, 2-3: ±18%, 4+: ±25%
        n_evidence = len(active_evidence)
        if n_evidence <= 1:
            max_drift = 0.12
        elif n_evidence <= 3:
            max_drift = 0.18
        else:
            max_drift = 0.25
        if p_final > p_market + max_drift:
            p_final = p_market + max_drift
        elif p_final < p_market - max_drift:
            p_final = p_market - max_drift

        p_final = max(0.02, min(0.98, round(p_final, 4)))

        if p_final > p_market:
            side, p_side, price_side = "YES", p_final, p_market
        else:
            side, p_side, price_side = "NO", 1-p_final, 1-p_market

        # Use bestAsk as real entry price for YES side only
        # (API doesn't provide bestBid, so can't compute real NO entry price)
        best_ask = market.get("best_ask")
        if side == "YES" and best_ask and 0.01 < best_ask < 0.99:
            real_price = best_ask
        else:
            real_price = price_side

        ev    = expected_value(p_side, real_price)
        kl    = kl_divergence(p_final, p_market)
        kelly = kelly_fraction(p_side, real_price)
        edge  = abs(p_final - p_market)

        # Kelly uncertainty: fewer evidence sources = less confident sizing
        # fraction = 0.5 + 0.5 × (n_sources / 7) → range [0.57, 1.0]
        # 1 source → Kelly × 0.57, all 7 → Kelly × 1.0
        MAX_SOURCES = 7
        confidence_mult = 0.5 + 0.5 * (n_evidence / MAX_SOURCES)
        kelly = round(kelly * confidence_mult, 4)

        # Spread penalty: wide spread = less confident sizing
        spread_mult = self._spread_penalty(market)
        if spread_mult < 1.0:
            kelly = round(kelly * spread_mult, 4)

        # Per-theme EV threshold: bad themes need higher EV to enter
        theme = market.get("theme", "other")
        theme_pat = self._patterns.get(theme, {})
        ev_mult = theme_pat.get("ev_mult", 1.0) or 1.0
        min_ev = self.config["MIN_EV"] * ev_mult
        min_kl = self.config["MIN_KL"] * ev_mult
        min_edge = self.config.get("MIN_EDGE", 0.08) * ev_mult

        if ev < min_ev:
            log.debug(f"[MATH] Rejected {market['id'][:8]}: EV {ev:.4f} < {min_ev:.4f} (×{ev_mult:.2f} {theme})")
            return None
        if kl < min_kl:
            log.debug(f"[MATH] Rejected {market['id'][:8]}: KL {kl:.4f} < {min_kl:.4f} (×{ev_mult:.2f} {theme})")
            return None
        if kelly < self.config["MIN_KELLY_FRAC"]:
            log.debug(f"[MATH] Rejected {market['id'][:8]}: Kelly {kelly:.4f} < {self.config['MIN_KELLY_FRAC']}")
            return None
        if edge < min_edge:
            log.debug(f"[MATH] Rejected {market['id'][:8]}: Edge {edge:.4f} < {min_edge:.4f} (×{ev_mult:.2f} {theme})")
            return None

        # Determine if contrarian is the dominant evidence
        is_contrarian = False
        if p_contrarian is not None and contrarian_conf > 0.3:
            # Contrarian is dominant if its contribution exceeds other evidence
            contrarian_shift = abs(p_contrarian - p_market)
            other_shifts = [abs(e - p_market) for e in [p_history, p_volume, p_time, p_momentum] if e is not None]
            max_other = max(other_shifts) if other_shifts else 0
            if contrarian_shift > max_other:
                is_contrarian = True

        arb_tag = " [ARB]" if p_arb is not None else ""
        log.info(f"[MATH] Signal: {side} '{market['question'][:50]}' EV:+{ev*100:.1f}% Kelly:{kelly*100:.1f}% Edge:{edge*100:.1f}%{' [CONTRARIAN]' if is_contrarian else ''}{arb_tag}")
        return {
            "market_id":  market["id"],
            "question":   market["question"],
            "theme":      theme,
            "url":        market.get("url",""),
            "side":       side,
            "side_price": real_price,  # bestAsk for YES, mid-price for NO
            "p_market":   p_market,
            "p_prospect": p_prospect,
            "p_history":  p_history,
            "p_momentum": p_momentum,
            "p_long_mom": p_long_mom,
            "p_contrarian": p_contrarian,
            "p_vol_trend": p_vol_trend,
            "p_arb":      p_arb,
            "p_final":    p_final,
            "p_side":     round(p_side, 4),
            "ev":         ev,
            "kl":         kl,
            "kelly":      kelly,
            "entropy":    entropy(p_market),
            "edge":       round(edge, 4),
            "spread":     market.get("spread", 0),
            "liquidity":  market.get("liquidity", 0),
            "spread_mult": spread_mult,
            "vol_signal": round(vol_signal, 2),
            "vol_dir":    vol_dir,
            "contrarian": is_contrarian,
            "contrarian_conf": round(contrarian_conf, 3) if p_contrarian is not None else 0,
            "volatility": self._market_volatility(market["id"]),
            "neg_risk_market_id": market.get("neg_risk_market_id", ""),
            "n_evidence": n_evidence,
            "p_book":     p_book,
            "source":     "math",
        }

    def _market_volatility(self, market_id: str) -> float:
        """Calculate market volatility as average absolute price change (ATR-style).
        Uses long price cache (up to 180 points / ~30 min)."""
        hist = self._long_price_cache.get(market_id, [])
        if len(hist) < 10:
            return 0.0
        changes = [abs(hist[i] - hist[i-1]) for i in range(1, len(hist))]
        return round(sum(changes) / len(changes), 5)

    def _apply_history(self, p_market: float, theme: str) -> Optional[float]:
        pat = self._patterns.get(theme)
        if not pat or pat.get("sample_size",0) < MIN_BETS_HISTORY: return None
        adjusted_rate = min(0.95, max(0.05, pat["base_rate"] * pat.get("prospect_factor", 1.0)))
        p_hist = 0.7 * adjusted_rate + 0.3 * p_market
        return round(max(0.02, min(0.98, p_hist)), 4)

    def _volume_signal(self, market_id: str, volume_24h: float) -> tuple:
        history = self._vol_history.setdefault(market_id, [])
        # Only record if value actually changed (volume_24h updates infrequently in API)
        if not history or abs(history[-1] - volume_24h) > 1.0:
            history.append(volume_24h)
        if len(history) > 48: history.pop(0)
        if len(history) < 3: return 1.0, "neutral"
        avg = sum(history[:-1]) / len(history[:-1])
        if avg <= 0: return 1.0, "neutral"
        ratio = volume_24h / avg
        if ratio > VOLUME_SPIKE_THR:
            # Use price momentum to determine spike direction (volume alone is always "up" here)
            price_hist = self._price_cache.get(market_id, [])
            if len(price_hist) >= 3:
                direction = "up" if price_hist[-1] > price_hist[-3] else "down"
            else:
                direction = "up"  # default to bullish if no price history
            log.info(f"[MATH] Volume spike {market_id[:8]}: {ratio:.1f}x avg → {direction}")
            return ratio, direction
        return ratio, "neutral"

    def _vol_adjusted(self, p_market: float, vol_ratio: float, direction: str) -> float:
        if vol_ratio < VOLUME_SPIKE_THR or direction == "neutral": return p_market
        strength = min(0.10, 0.05 * (vol_ratio / VOLUME_SPIKE_THR))
        if direction == "up":   return min(0.98, p_market * (1 + strength))
        return max(0.02, p_market * (1 - strength))

    def _time_decay(self, p_market: float, end_date) -> Optional[float]:
        if not end_date: return None
        try:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z","+00:00"))
            days_left = (end_date - datetime.now(timezone.utc)).days
        except Exception:
            return None
        if days_left <= 0: return None
        # Smooth decay: blend market → prospect as time increases
        # Near expiry: trust market price more (it's converging to truth)
        # Far from expiry: prospect theory has more room to add value
        # Near expiry (decay→0): trust market price. Far (decay→1): slightly away from market.
        # Don't re-apply prospect here — it's already the Bayesian prior.
        decay = min(1.0, max(0.0, (days_left - 3) / 30))
        # Use 0.5 as neutral anchor: far from expiry → market price is less reliable
        p_time = p_market * (1 - decay * 0.3) + 0.5 * (decay * 0.3)
        # Only return if meaningfully different from market price
        if abs(p_time - p_market) < 0.005:
            return None
        return p_time

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

    def _mean_reversion(self, market_id: str, current_price: float) -> tuple:
        """Detect sharp price moves on low volume that are likely to revert.
        Returns (p_contrarian, confidence) or (None, 0) if no signal."""
        # Use long price cache for detection (30 min of data)
        long_hist = self._long_price_cache.get(market_id, [])
        if len(long_hist) < 10:  # need ~2 min minimum history
            return None, 0

        # Price 30 min ago (or oldest available)
        old_price = long_hist[0]
        move = current_price - old_price

        # Need >5% absolute move
        if abs(move) < 0.05:
            return None, 0

        # Check volume ratio from _vol_history
        # Need enough volume data to assess — without it, can't tell informed vs noise
        vol_hist = self._vol_history.get(market_id, [])
        if len(vol_hist) < 6:
            return None, 0  # not enough volume data to assess contrarian signal
        else:
            avg_vol = sum(vol_hist[:-1]) / len(vol_hist[:-1])
            vol_ratio = vol_hist[-1] / avg_vol if avg_vol > 0 else 1.0

        # Volume filter: high volume = informed money, skip
        if vol_ratio > 2.5:
            log.debug(f"[MATH] Contrarian skip {market_id[:8]}: vol_ratio {vol_ratio:.1f}x (informed money)")
            return None, 0

        # Confidence: lower volume = higher confidence, bigger move = higher confidence
        move_confidence = min(1, abs(move) / 0.20)
        if vol_ratio >= 1.5:
            # Moderate volume: might be informed, lower confidence
            vol_confidence = max(0.1, 1 - (vol_ratio - 1.5) / 1.0)  # 1.5x→1.0, 2.5x→0.1
            confidence = vol_confidence * move_confidence
            log.info(f"[MATH] Contrarian WEAK {market_id[:8]}: move={move:+.3f} vol={vol_ratio:.1f}x conf={confidence:.2f}")
        else:
            # Low volume: likely noise, high confidence in reversion
            confidence = move_confidence
            log.info(f"[MATH] Contrarian STRONG {market_id[:8]}: move={move:+.3f} vol={vol_ratio:.1f}x conf={confidence:.2f}")

        # Compute EWMA as reversion target
        alpha = 2 / (len(long_hist) + 1)
        ewma = long_hist[0]
        for p in long_hist[1:]:
            ewma = alpha * p + (1 - alpha) * ewma

        # Shift probability toward EWMA (reversion target)
        # Blend: current price shifted toward EWMA, weighted by confidence
        p_contrarian = current_price + (ewma - current_price) * confidence * 0.7
        p_contrarian = round(max(0.02, min(0.98, p_contrarian)), 4)

        return p_contrarian, confidence

    def _spread_penalty(self, market: dict) -> float:
        """Spread > 3¢ reduces confidence. Returns multiplier 0.0–1.0."""
        spread = market.get("spread", 0)
        if spread <= 0.03:
            return 1.0
        if spread >= 0.10:
            return 0.3
        # Linear scale: 3¢→1.0, 10¢→0.3
        return round(1.0 - (spread - 0.03) / 0.07 * 0.7, 3)

    def _long_momentum(self, market: dict) -> Optional[float]:
        """Use API-provided week/month price changes as long-term momentum signal."""
        chg_1wk = market.get("price_change_1wk", 0)
        chg_1mo = market.get("price_change_1mo", 0)
        # Require at least 2% weekly or 5% monthly change to signal
        if abs(chg_1wk) < 0.02 and abs(chg_1mo) < 0.05:
            return None
        # Weighted: week is more recent, weight 0.7; month gives context, weight 0.3
        blended = chg_1wk * 0.7 + chg_1mo * 0.3
        # Cap at ±8% shift
        shift = max(-0.08, min(0.08, blended * 0.5))
        p_long = market["yes_price"] + shift
        return round(max(0.02, min(0.98, p_long)), 4)

    def _volume_trend(self, market: dict) -> Optional[float]:
        """Compare 24h volume to weekly average. Rising interest = trust market price more."""
        vol_24h = market.get("volume_24h", 0)
        vol_1wk = market.get("volume_1wk", 0)
        if vol_1wk <= 0 or vol_24h <= 0:
            return None
        daily_avg = vol_1wk / 7
        if daily_avg <= 0:
            return None
        ratio = vol_24h / daily_avg
        if 0.5 < ratio < 2.0:
            # Normal volume range — no signal
            return None
        p_market = market["yes_price"]
        if ratio >= 2.0:
            # Rising interest: market moving toward truth, trust market direction
            # Amplify current deviation from 0.5
            strength = min(0.05, (ratio - 2.0) * 0.02)
            direction = 1 if p_market > 0.5 else -1
            p_vol_trend = p_market + direction * strength
        else:
            # Dying interest (ratio < 0.5): market may be stale, pull toward 0.5
            strength = min(0.03, (0.5 - ratio) * 0.03)
            p_vol_trend = p_market + (0.5 - p_market) * strength
        return round(max(0.02, min(0.98, p_vol_trend)), 4)

    def _book_imbalance(self, market: dict) -> Optional[float]:
        """Order book imbalance signal from WS book events.
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol), range [-1, 1].
        Positive = buy pressure → YES price likely up.
        Only fires when |imbalance| > 0.3 (strong directional pressure).
        Returns adjusted p as evidence source, weight 0.5 in fusion."""
        imbalance = market.get("book_imbalance")
        if imbalance is None or abs(imbalance) < 0.3:
            return None
        p_market = market["yes_price"]
        # Scale: |imbalance| 0.3→0.01 shift, 1.0→0.04 shift
        strength = min(0.04, (abs(imbalance) - 0.3) * 0.043 + 0.01)
        if imbalance > 0:
            p_book = p_market + strength   # buy pressure → YES up
        else:
            p_book = p_market - strength   # sell pressure → YES down
        return round(max(0.02, min(0.98, p_book)), 4)

    def build_neg_risk_groups(self, markets: list):
        """Group neg-risk markets by their shared event ID for arbitrage detection."""
        self._neg_risk_groups.clear()
        for m in markets:
            nrm_id = m.get("neg_risk_market_id", "")
            if nrm_id and m.get("neg_risk"):
                self._neg_risk_groups.setdefault(nrm_id, []).append(m)
        grouped = sum(1 for g in self._neg_risk_groups.values() if len(g) > 1)
        if grouped:
            log.info(f"[MATH] NegRisk: {grouped} multi-outcome events ({sum(len(g) for g in self._neg_risk_groups.values())} markets)")

    def _neg_risk_arb(self, market: dict) -> Optional[float]:
        """Detect mispricing in multi-outcome events where probabilities should sum to 1."""
        nrm_id = market.get("neg_risk_market_id", "")
        if not nrm_id:
            return None
        group = self._neg_risk_groups.get(nrm_id)
        if not group or len(group) < 2:
            return None
        # Sum of all YES prices in this event
        total = sum(m["yes_price"] for m in group)
        if total <= 0:
            return None
        # Fair sum = 1.0; overpriced if > 1, underpriced if < 1
        # Adjust this market's probability proportionally
        # E.g. total=1.10 means 10% overpriced, each outcome should be ~9% lower
        fair_price = market["yes_price"] / total
        # Don't return if adjustment is tiny (< 1%)
        if abs(fair_price - market["yes_price"]) < 0.01:
            return None
        log.debug(f"[MATH] NegRisk arb {market['id'][:8]}: sum={total:.3f} price={market['yes_price']:.3f}→{fair_price:.3f}")
        return round(max(0.02, min(0.98, fair_price)), 4)

    def get_market_metrics(self, market_id: str) -> dict:
        """Export current metrics for a market (for DB persistence)."""
        short = self._price_cache.get(market_id, [])
        long_ = self._long_price_cache.get(market_id, [])
        vol = self._market_volatility(market_id)

        # Momentum slope
        momentum = 0.0
        if len(short) >= 5:
            n = len(short)
            x_mean = (n - 1) / 2
            y_mean = sum(short) / n
            num = sum((i - x_mean) * (short[i] - y_mean) for i in range(n))
            den = sum((i - x_mean) ** 2 for i in range(n))
            if den > 0:
                momentum = round(num / den * n * 0.5, 5)

        # Volume ratio
        vol_hist = self._vol_history.get(market_id, [])
        vol_ratio = 1.0
        if len(vol_hist) >= 3:
            avg = sum(vol_hist) / len(vol_hist)
            if avg > 0:
                vol_ratio = round(vol_hist[-1] / avg, 2)

        return {
            "volatility": vol,
            "momentum": momentum,
            "vol_ratio": vol_ratio,
            "long_prices": long_[-180:],
            "short_prices": short[-30:],
        }

    def restore_market_metrics(self, market_id: str, metrics: dict):
        """Restore caches from DB on startup."""
        long_p = metrics.get("long_prices") or []
        short_p = metrics.get("short_prices") or []
        if long_p:
            self._long_price_cache[market_id] = list(long_p)
        if short_p:
            self._price_cache[market_id] = list(short_p)

    # ── Portfolio correlation constants ──
    THEME_CORRELATION = 0.5      # assumed correlation between positions in same theme
    NEGRISK_CORRELATION = 1.0    # positions sharing negRiskMarketID are ~identical risk
    MAX_EFFECTIVE_STAKE_PCT = 0.05   # max 5% of bankroll per effective independent bet
    MAX_WORST_CASE_PCT = 0.15        # max 15% equity loss if entire cluster hits SL

    def compute_stake(self, bankroll: float, kelly: float, theme: str = None,
                       open_positions: list = None, liquidity: float = 0,
                       neg_risk_market_id: str = None, sl_pct: float = 0.25) -> float:
        """Compute stake with Bayesian theme calibration, correlation penalty, and liquidity."""
        if bankroll <= 0:
            return 0.0

        # Per-theme Kelly multiplier from Bayesian performance calibration
        if theme:
            theme_pat = self._patterns.get(theme, {})
            kelly_mult = theme_pat.get("kelly_mult", 1.0) or 1.0
            if kelly_mult != 1.0:
                kelly = kelly * kelly_mult
                log.info(f"[MATH] Theme Kelly: '{theme}' ×{kelly_mult:.2f} → Kelly {kelly*100:.2f}%")

        stake = bankroll * kelly
        stake = min(stake, bankroll * self.config["MAX_KELLY_FRAC"])

        # ── Portfolio correlation penalty ──
        # Accounts for correlated risk: positions in same negRisk group (~100% correlated)
        # and same theme (~50% correlated) are effectively fewer independent bets.
        if open_positions and bankroll > 0:
            penalty = self._correlation_penalty(
                theme, neg_risk_market_id, open_positions, bankroll, sl_pct)
            if penalty < 1.0:
                stake *= penalty

        # Liquidity penalty: thin markets → smaller stake to avoid slippage
        if liquidity > 0:
            liq_mult = min(1.0, liquidity / 50_000)
            if liq_mult < 1.0:
                stake *= liq_mult
                log.info(f"[MATH] Liquidity penalty: ${liquidity:,.0f} → stake ×{liq_mult:.2f}")

        stake = round(max(1.0, stake), 2)
        log.info(f"[MATH] Stake: ${stake:.2f} (kelly={kelly*100:.1f}% bankroll=${bankroll:.2f})")
        return stake

    def _correlation_penalty(self, theme: str, neg_risk_market_id: str,
                              open_positions: list, bankroll: float, sl_pct: float) -> float:
        """Compute correlation-aware penalty for portfolio risk concentration.

        Groups positions by negRiskMarketID (ρ≈1.0) and theme (ρ≈0.5).
        Calculates effective number of independent bets:
          effective_n = n / (1 + (n-1) × ρ)
        Then limits stake_per_effective_bet to MAX_EFFECTIVE_STAKE_PCT of bankroll.
        Also checks worst-case scenario (entire cluster hits SL).
        """
        penalty = 1.0

        # 1. NegRisk group check (correlation ≈ 1.0) — most dangerous
        #    Look up which market_ids share the same negRiskMarketID
        if neg_risk_market_id:
            nrg_market_ids = {m["id"] for m in self._neg_risk_groups.get(neg_risk_market_id, [])}
            nrg_positions = [p for p in open_positions
                             if p.get("market_id") in nrg_market_ids]
            if nrg_positions:
                n = len(nrg_positions)
                nrg_stake = sum(p.get("stake_amt", 0) for p in nrg_positions)
                # With ρ=1.0, effective_n = 1 regardless of n
                effective_n = n / (1 + (n - 1) * self.NEGRISK_CORRELATION)
                stake_per_eff = nrg_stake / effective_n if effective_n > 0 else nrg_stake
                max_allowed = bankroll * self.MAX_EFFECTIVE_STAKE_PCT
                if stake_per_eff > max_allowed:
                    nrg_penalty = max_allowed / stake_per_eff
                    penalty = min(penalty, nrg_penalty)
                    log.info(f"[CORR] negRisk group: {n} pos, ${nrg_stake:.0f} staked, "
                             f"eff={effective_n:.1f} → penalty ×{nrg_penalty:.2f}")

        # 2. Theme cluster check (correlation ≈ 0.5)
        if theme:
            theme_positions = [p for p in open_positions if p.get("theme") == theme]
            if theme_positions:
                n = len(theme_positions)
                theme_stake = sum(p.get("stake_amt", 0) for p in theme_positions)
                effective_n = n / (1 + (n - 1) * self.THEME_CORRELATION)
                stake_per_eff = theme_stake / effective_n if effective_n > 0 else theme_stake
                max_allowed = bankroll * self.MAX_EFFECTIVE_STAKE_PCT
                if stake_per_eff > max_allowed:
                    theme_penalty = max_allowed / stake_per_eff
                    penalty = min(penalty, theme_penalty)
                    log.info(f"[CORR] Theme '{theme}': {n} pos, ${theme_stake:.0f} staked, "
                             f"eff={effective_n:.1f} → penalty ×{theme_penalty:.2f}")

                # 3. Worst-case check: if all theme positions hit SL simultaneously
                worst_case_loss = theme_stake * sl_pct
                max_loss = bankroll * self.MAX_WORST_CASE_PCT
                if worst_case_loss > max_loss:
                    wc_penalty = max_loss / worst_case_loss
                    if wc_penalty < penalty:
                        log.info(f"[CORR] Worst-case '{theme}': SL on all {n} = -${worst_case_loss:.0f} "
                                 f"> {self.MAX_WORST_CASE_PCT*100:.0f}% bankroll → penalty ×{wc_penalty:.2f}")
                    penalty = min(penalty, wc_penalty)

        if penalty < 0.15:
            log.info(f"[CORR] Penalty ×{penalty:.2f} too low, will likely reject (stake < $1)")

        return round(max(0.05, penalty), 3)
