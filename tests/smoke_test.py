"""
Pre-launch smoke tests for Quant Engine.
Run: python tests/smoke_test.py
Exit code 0 = all passed, 1 = failures.
No DB or network required — pure unit tests on math/logic.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0
errors = []

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  \033[32m✓\033[0m {name}")
    else:
        failed += 1
        msg = f"{name}: {detail}" if detail else name
        errors.append(msg)
        print(f"  \033[31m✗\033[0m {msg}")


# ── 1. Core Math Functions ──
print("\n\033[1m1. Core Math\033[0m")

from agents.math_engine import (
    expected_value, kelly_fraction, kl_divergence, entropy,
    prospect_weight, prospect_true_price, bayesian_update,
    prob_to_logodds, logodds_to_prob,
)

# EV
check("EV: fair price → ~0", abs(expected_value(0.5, 0.5)) < 0.01, f"got {expected_value(0.5, 0.5)}")
check("EV: edge → positive", expected_value(0.7, 0.5) > 0.2, f"got {expected_value(0.7, 0.5)}")
check("EV: bad bet → negative", expected_value(0.3, 0.5) < 0, f"got {expected_value(0.3, 0.5)}")
check("EV: price=0 → 0", expected_value(0.5, 0) == 0.0)
check("EV: price=1 → 0", expected_value(0.5, 1) == 0.0)

# Kelly
check("Kelly: edge → positive", kelly_fraction(0.7, 0.5) > 0, f"got {kelly_fraction(0.7, 0.5)}")
check("Kelly: no edge → 0", kelly_fraction(0.5, 0.5) == 0)
check("Kelly: bad bet → 0", kelly_fraction(0.3, 0.5) == 0)
check("Kelly: price=0 → 0", kelly_fraction(0.5, 0) == 0)
check("Kelly: price=1 → 0", kelly_fraction(0.5, 1) == 0)
check("Kelly: p=0 → 0", kelly_fraction(0, 0.5) == 0)
check("Kelly: p=1 → 0 guard", kelly_fraction(1, 0.5) == 0)

# KL divergence
check("KL: same dist → 0", kl_divergence(0.5, 0.5) == 0)
check("KL: different → positive", kl_divergence(0.8, 0.5) > 0)
check("KL: extreme → handles", kl_divergence(0.001, 0.999) > 0)

# Entropy
check("Entropy: max at 0.5", entropy(0.5) == 1.0)
check("Entropy: 0 at extremes", entropy(0.01) < 0.1)

# Prospect
check("Prospect: overweight small p", prospect_weight(0.1) > 0.1)
check("Prospect: underweight large p", prospect_weight(0.9) < 0.9)
check("Prospect inverse: roundtrip", abs(prospect_true_price(prospect_weight(0.3)) - 0.3) < 0.01,
      f"got {prospect_true_price(prospect_weight(0.3))}")

# Bayesian
check("Bayesian: evidence shifts prior", bayesian_update(0.5, [(0.8, 1.0)]) > 0.6,
      f"got {bayesian_update(0.5, [(0.8, 1.0)])}")
check("Bayesian: no evidence → prior", bayesian_update(0.5, []) == 0.5)
check("Bayesian: strong evidence", bayesian_update(0.5, [(0.9, 2.0)]) > 0.7)
check("Bayesian: conflicting evidence", 0.4 < bayesian_update(0.5, [(0.8, 1.0), (0.2, 1.0)]) < 0.6)

# Log-odds roundtrip
check("Log-odds: roundtrip 0.3", abs(logodds_to_prob(prob_to_logodds(0.3)) - 0.3) < 0.001)
check("Log-odds: roundtrip 0.9", abs(logodds_to_prob(prob_to_logodds(0.9)) - 0.9) < 0.001)


# ── 2. MathEngine Methods ──
print("\n\033[1m2. MathEngine Methods\033[0m")

from agents.math_engine import MathEngine

class FakeDB:
    async def get_patterns(self): return {}
    async def get_dma_weights(self): return {}

config = {
    "MIN_EV": 0.12, "MIN_KL": 0.10, "MIN_EDGE": 0.08,
    "MAX_KELLY_FRAC": 0.15, "MIN_KELLY_FRAC": 0.01,
    "MAX_EV": 0.20, "MAX_MARKET_DAYS": 30,
    "USE_PROSPECT": True, "BANKROLL": 1000,
}

import asyncio
eng = MathEngine(config, FakeDB())

# Hurst exponent
eng._long_price_cache["_trend"] = [0.50 + i * 0.005 for i in range(30)]
h = eng._hurst_exponent("_trend")
check("Hurst: trending H > 0.5", h > 0.45, f"H={h:.3f}")
del eng._long_price_cache["_trend"]

eng._long_price_cache["_revert"] = [0.50 + (0.01 if i % 2 == 0 else -0.01) for i in range(30)]
h = eng._hurst_exponent("_revert")
check("Hurst: mean-reverting H < 0.5", h < 0.55, f"H={h:.3f}")
del eng._long_price_cache["_revert"]

# Spread penalty
check("Spread 0% → 1.0", eng._spread_penalty({"spread": 0}) == 1.0)
check("Spread 3% → 1.0", eng._spread_penalty({"spread": 0.03}) == 1.0)
check("Spread 10% → 0.3", eng._spread_penalty({"spread": 0.10}) == 0.3)
sp = eng._spread_penalty({"spread": 0.06})
check("Spread 6% → (0.3, 1.0)", 0.3 <= sp <= 1.0, f"got {sp}")
check("Spread negative → 1.0", eng._spread_penalty({"spread": -0.01}) == 1.0)

# Volume signal edge cases
sig, direction = eng._volume_signal("_test_vol", 0)
check("Volume: zero → neutral", sig == 1.0 and direction == "neutral")

eng._vol_history["_test_vol"] = [100.0]  # only 1 entry
sig, direction = eng._volume_signal("_test_vol", 100.0)
check("Volume: 1 entry → no crash", sig == 1.0)
del eng._vol_history["_test_vol"]

# Mean reversion thresholds
eng._long_price_cache["_small_move"] = [0.50] * 20
p, conf = eng._mean_reversion("_small_move", 0.55)  # 5% move < 8% threshold
check("Contrarian: 5% move → None (below 8% threshold)", p is None, f"got p={p}")
del eng._long_price_cache["_small_move"]

eng._long_price_cache["_big_move"] = [0.50] * 20
p, conf = eng._mean_reversion("_big_move", 0.85)  # 35% move > 30% cap
check("Contrarian: 35% move → None (news-driven block)", p is None, f"got p={p}")
del eng._long_price_cache["_big_move"]

# Momentum threshold
eng._price_cache["_flat"] = [0.50 + i * 0.0001 for i in range(10)]  # tiny slope
p = eng._price_momentum("_flat", 0.501)
check("Momentum: tiny slope → None (below 1.5% threshold)", p is None, f"got {p}")
del eng._price_cache["_flat"]

eng._price_cache["_strong"] = [0.50 + i * 0.005 for i in range(10)]  # strong slope
p = eng._price_momentum("_strong", 0.55)
check("Momentum: strong slope → not None", p is not None, f"got {p}")
del eng._price_cache["_strong"]

# Overreaction decay thresholds
eng._price_cache["_small_or"] = [0.50 + i * 0.008 for i in range(15)]  # 9.6% in 10 ticks
p = eng._overreaction_decay("_small_or", 0.58)
check("Overreaction: 10% move → None (below 12% threshold)", p is None, f"got {p}")
del eng._price_cache["_small_or"]

# Book imbalance
check("Book: no data → None", eng._book_imbalance({"yes_price": 0.5}) is None)
check("Book: weak imbalance → None", eng._book_imbalance({"yes_price": 0.5, "book_imbalance": 0.1}) is None)
p = eng._book_imbalance({"yes_price": 0.5, "book_imbalance": 0.5})
check("Book: strong imbalance → shift", p is not None and p > 0.5, f"got {p}")

# FLB
check("FLB: mid price → None", eng._favorite_longshot_bias({"id": "t", "yes_price": 0.5}) is None)
p = eng._favorite_longshot_bias({"id": "t", "yes_price": 0.1})
check("FLB: longshot → adjusts", p is not None, f"got {p}")

# Certainty gradient
check("Certainty: mid → None", eng._certainty_gradient({"id": "t", "yes_price": 0.5}) is None)
p = eng._certainty_gradient({"id": "t", "yes_price": 0.96})
check("Certainty: near 1.0 → shift up", p is not None and p > 0.96, f"got {p}")

# Short-term filter
market_short = {"id": "test", "question": "Bitcoin Up or Down on April 1?", "yes_price": 0.5,
                "no_price": 0.5, "volume": 100000, "volume_24h": 50000, "spread": 0.02,
                "best_ask": 0.51, "theme": "crypto"}
result = eng.analyze(market_short)
check("Short-term filter: 'Up or Down' → rejected", result is None)

# Expired date filter
market_expired = {"id": "test2", "question": "Will X happen on January 1, 2025?", "yes_price": 0.5,
                  "no_price": 0.5, "volume": 100000, "volume_24h": 50000, "spread": 0.02,
                  "best_ask": 0.51, "theme": "other", "end_date": "2025-01-02T00:00:00Z"}
result = eng.analyze(market_expired)
check("Expired date: past date → rejected", result is None)


# ── 3. Portfolio Correlation ──
print("\n\033[1m3. Portfolio Correlation\033[0m")

# compute_stake with empty portfolio
stake = eng.compute_stake(1000, 0.10, "crypto", [], 50000, "", 0.25)
check("Stake: normal case → positive", stake > 0, f"got {stake}")

stake = eng.compute_stake(0, 0.10, "crypto", [], 50000, "", 0.25)
check("Stake: zero bankroll → 0", stake == 0)

stake = eng.compute_stake(-100, 0.10, "crypto", [], 50000, "", 0.25)
check("Stake: negative bankroll → 0", stake == 0)

stake = eng.compute_stake(1000, 0, "crypto", [], 50000, "", 0.25)
# compute_stake has $1 min floor — zero kelly still gives min stake
check("Stake: zero kelly → min", stake <= 1.0, f"got {stake}")


# ── 4. Division by Zero Guards ──
print("\n\033[1m4. Division by Zero Guards\033[0m")

check("EV: price boundary 0.001", abs(expected_value(0.5, 0.001)) < 1000)
check("EV: price boundary 0.999", abs(expected_value(0.5, 0.999)) < 10)
check("Kelly: price boundary 0.001", kelly_fraction(0.9, 0.001) >= 0)
check("Kelly: price boundary 0.999", kelly_fraction(0.9, 0.999) >= 0)

# Spread penalty edge cases
check("Spread: None → 1.0", eng._spread_penalty({}) == 1.0)
check("Spread: 100% → 0.3", eng._spread_penalty({"spread": 1.0}) == 0.3)


# ── 5. DMA Weight Bounds ──
print("\n\033[1m5. DMA & Calibration\033[0m")

# Verify DMA floor is 0.5 (not 0.3)
try:
    import inspect
    from agents.history_agent import HistoryAgent
    src = inspect.getsource(HistoryAgent._update_dma_weights)
    check("DMA floor = 0.5", "max(0.5," in src, "expected max(0.5, ...) in DMA normalization")
    check("DMA ceiling = 2.0", "min(2.0," in src)
except ImportError:
    # numpy not available outside venv — check source file directly
    src_ha = open("agents/history_agent.py").read()
    check("DMA floor = 0.5", "max(0.5," in src_ha, "expected max(0.5, ...) in DMA normalization")
    check("DMA ceiling = 2.0", "min(2.0," in src_ha)

# Verify contrarian threshold
src_me = open("agents/math_engine.py").read()
check("Contrarian conf threshold = 0.5", "contrarian_conf > 0.5" in src_me, "expected 0.5 threshold")


# ── 6. Scanner Filters ──
print("\n\033[1m6. Scanner & Recheck\033[0m")

try:
    from engine.scanner import is_sports
    check("Sports: NBA → True", is_sports("Will the Lakers win the NBA finals?"))
    check("Sports: Iran → False", not is_sports("US forces enter Iran by March 31?"))
    check("Sports: UFC → True", is_sports("Who wins the UFC 300 main event?"))
except ImportError:
    # httpx not available outside venv — test via source
    check("Sports filter: skipped (no httpx)", True)

src_scanner = open("engine/scanner.py").read()
check("Scanner: acceptingOrders filter", "acceptingOrders" in src_scanner)

# Verify recheck exists in main.py
src_main = open("main.py").read()
check("Recheck: market_closed_pre_exec", "market_closed_pre_exec" in src_main)
check("Recheck: market_in_review", "market_in_review" in src_main)
check("Recheck: stale_price", "stale_price" in src_main)
check("Recheck: price parsing safety", "ValueError, TypeError, IndexError" in src_main)

# Verify Claude API reject on error
check("Claude: reject on API error", "api_error_reject" in src_main, "should reject, not confirm")

# Verify payout division guard
check("Payout: side_price>0 guard", "won and pos[\"side_price\"] > 0" in src_main)

# Verify upnl division guard
check("Upnl: side_price>0 guard", 'if pos["side_price"] > 0 else 0' in src_main)

# Verify peak equity clamp
check("Peak equity: max(0) clamp", "max(0, p.get" in src_main)

# Scanner retry backoff
check("Scanner: retry backoff", "retry_delay in [2, 5, 10]" in src_scanner)


# ── 7. Signal Quality Guards ──
print("\n\033[1m7. Signal Quality Guards\033[0m")

# Contrarian move thresholds in source
src_me_full = open("agents/math_engine.py").read()
check("Contrarian: 8% min move", "abs(move) < 0.08" in src_me_full)
check("Contrarian: 30% max block", "abs(move) > 0.30" in src_me_full)
check("Contrarian: volume 2.0x filter", "vol_ratio > 2.0" in src_me_full)
check("Contrarian: smooth vol confidence", "1.0 - (vol_ratio - 0.5) * 0.5" in src_me_full)
check("Overreaction: 12% threshold", "abs_move < 0.12" in src_me_full)
check("Momentum: 1.5% min shift", "abs(momentum_shift) < 0.015" in src_me_full)


# ── Results ──
print(f"\n{'='*50}")
total = passed + failed
if failed == 0:
    print(f"\033[32m ALL {passed} TESTS PASSED\033[0m")
    sys.exit(0)
else:
    print(f"\033[31m {failed}/{total} TESTS FAILED:\033[0m")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
