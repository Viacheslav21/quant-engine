# Quant Engine: Research Report — Strategies for Profitability

> Date: 2026-03-25 | Updated: 2026-03-26
> Sources: Academic papers, Twitter/X, Reddit, Medium, CryptoNews, GitHub, Substack, arXiv

---

## Table of Contents

1. [Current State & Audit](#1-current-state--audit)
2. [Community Strategies (Reddit, Medium)](#2-community-strategies)
3. [Advanced Math Techniques](#3-advanced-math-techniques)
4. [Market Inefficiencies & Microstructure](#4-market-inefficiencies--microstructure)
5. [What We Already Cover](#5-what-we-already-cover)
6. [Gap Analysis & Implementation Plan](#6-gap-analysis--implementation-plan)
7. [References](#7-references)

---

## 1. Current State & Audit

### Log Audit (25 March 2026, 12:06–13:22 UTC)

- 5 scan cycles, ~500 markets, 0 errors
- 4 new SIM trades, 3 closed, 24 whale alerts

### Key Bottlenecks

| Bottleneck | Impact |
|---|---|
| Theme limits (crypto/social/iran all 10/10) | 15+ good signals/hour blocked |
| Only 5 signals per cycle (`confirmed[:5]`) | Half of new signals wasted |
| Trailing TP too conservative (5% pullback) | +12-13% positions oscillate, don't close |
| Iran: 10 slots, WR 24%, ROI -11.3% | Capital locked in losing theme |

### Theme Performance

| Theme | Trades | WR Adj | ROI | kelly_mult | ev_mult |
|---|---|---|---|---|---|
| crypto | 55 | 54% | +7.1% | 1.24 | 0.71 |
| social | 30 | 41% | -1.5% | 0.95 | 1.05 |
| iran | 21 | 34% | -11.3% | 0.77 | 1.48 |
| oil | 28 | 31% | -4.1% | 0.70 | 1.42 |

Brier: 0.019 (excellent), bias: -0.003

### Config Tag Performance

| Tag | Trades | WR | Total PnL | Avg PnL | Avg Stake | Key Params |
|---|---|---|---|---|---|---|
| v4 | 96 | 40% | -$9.20 | -$0.10 | $7.97 | EV≥0.08 KL≥0.05 |
| v5 | 66 | 27% | -$23.22 | -$0.35 | $4.45 | EV≥0.10 KL≥0.07 |
| v6 | 7 | 29% | -$2.13 | -$0.30 | $11.71 | EV≥0.10 KL≥0.07 Kelly:0.2 |

**Key insight**: v4 (softest thresholds) had best WR. Tighter thresholds filter out good signals too.

---

## 2. Community Strategies

### Applicable to Our Bot

| Strategy | Source | Our Coverage | Action |
|---|---|---|---|
| "Nothing Ever Happens" — fade geopolitical hype | Reddit #1 | Mean reversion + overreaction_decay | Increase contrarian weight for geopolitical |
| Positive EV Grinding — buy obvious outcomes | Reddit #6 | Prospect theory + FLB | Strengthen for high-prob markets |
| Bond Harvesting — buy >95% near resolution | Medium #1 | **quant-micro** | Already implemented as separate service |
| Mispricing Hunting | Medium #3 | Core math engine | Already our main strategy |
| Whale conviction as evidence source | Reddit #2, Medium #2 | None | New evidence source (on-chain tracking) |
| News Scalping — first 30 sec | Reddit #3 | 5-min scan too slow | RSS/Telegram fast-trigger |
| Mentions: default to NO | Reddit #9 | FLB signal | Explicit NO bias at longshots |
| Riskless Rate Discounting | Reddit #11 | time_decay signal | Add risk-free rate for long markets |
| Cultural Calendar | Reddit #14 | None | Holiday calendar modifier |

### Not Applicable (manual/different strategy)

- Personality-driven mention trading (Reddit #4) — niche, hard to automate
- "Girlfriend Poll" (Reddit #8) — demographic bias, manual
- Partial Resolution Advocacy (Reddit #13) — requires NLP of resolution criteria
- Market Making / LP (Medium #5) — separate bot entirely
- Fed Signal Trading (Reddit #5) — niche, CME integration needed

---

## 3. Advanced Math Techniques

### HIGH Priority (implement next)

**Hurst Exponent** — signal selector (~20 lines):
```
H > 0.6: trending → momentum ×1.5, contrarian ×0.5
H < 0.4: mean-reverting → contrarian ×1.5, momentum ×0.5
H ≈ 0.5: random walk → reduce all weights
```

**Dynamic Bayesian Model Averaging** — adaptive evidence weights:
```
w_k,t = (w_{k,t-1}^α × likelihood) / normalizer    α ∈ (0.95, 0.99)
```
Sources that predicted well recently → higher weight automatically.

**Thompson Sampling** — theme allocation (drop-in replacement for Bayesian shrinkage):
```
sampled_wr ~ Beta(wins + α, losses + β)
allocation = sampled_wr × kelly / Σ(sampled_wr_j × kelly_j)
```

### MEDIUM Priority

**VPIN** — order flow toxicity from WS trade data:
```
VPIN = (1/n) × Σ |V_buy - V_sell| / VBS    (n=50 volume buckets)
```
When VPIN > 0.6 → widen entry thresholds (informed money active).

**BOCPD** — regime change detection:
```
p(changepoint) > 0.3 → reset momentum/contrarian caches, widen uncertainty
```

**Logit Jump-Diffusion** — replace ATR volatility with belief volatility:
```
Jump detection: γ_t > 0.7 = news event → don't mean-revert
```

### LOW Priority (research stage)

| Technique | Purpose | Complexity |
|---|---|---|
| Conformal Prediction | Calibrated uncertainty for XGBoost | MEDIUM |
| Risk-Constrained Kelly | `f_RCK = f_Kelly × (1 - λ × σ_portfolio)` | LOW |
| Kyle's Lambda | Market impact estimation, order sizing | MEDIUM |
| Copula correlation | Replace fixed ρ with empirical tail dependence | HIGH |
| Hawkes Process | Trade clustering → momentum/reversal signal | HIGH |
| Transfer Entropy | Lead-lag detection between markets | HIGH |

---

## 4. Market Inefficiencies & Microstructure

### Documented Inefficiencies

| Finding | Scale | Source |
|---|---|---|
| $40M arbitrage extracted from Polymarket (2024-2025) | 41% of markets had arb | arXiv:2508.03474 |
| Longshots (<10¢) overpriced, buyers lose >60% | Systematic | NBER (Snowberg & Wolfers) |
| Takers lose 1.12%/trade, makers gain 1.12% | Structural | jbecker.dev |
| YES underperforms NO by up to 64pp at longshot prices | Structural | jbecker.dev |
| Entertainment markets: 4.79% maker-taker gap | Category-dependent | jbecker.dev |
| Markets overestimate certainty for distant events | 4.7-10.9pp bias at >100d | Page & Clemen 2013 |

### Actionable Microstructure

| Technique | Impact | Status |
|---|---|---|
| Order Book Imbalance (R²=0.65 for short-term) | **Increase weight 0.5→0.8** | Easy fix |
| Closing Line Value tracking | Feedback loop for weight optimization | Add columns to positions |
| Limit orders (save 50bps round-trip) | Free money | Requires CLOB API |
| Intraday patterns (execute off-hours) | Lower impact cost | Schedule-based |

### Behavioral Biases

| Bias | Exploitable? | Our Coverage |
|---|---|---|
| Favorite-Longshot | Yes — sell longshots | Prospect theory (γ=0.65) |
| Partisan bias in politics | Yes — fade partisans | None → detector needed |
| Recency / overreaction | Yes — mean reversion | Active (contrarian signal) |
| YES optimism bias | Yes — default to NO | FLB partially covers |
| "Nothing happens" in geopolitics | Yes — fade hype | Contrarian signal |

---

## 5. What We Already Cover

| Capability | Module | Notes |
|---|---|---|
| Bayesian fusion (13→8 sources) | math_engine.py | Core signal generation |
| Kelly criterion (0.15 fractional) | math_engine.py | With uncertainty scaling |
| Prospect theory (γ=0.65) | math_engine.py | Inverts human bias |
| Mean reversion + overreaction decay | math_engine.py | Two separate signals |
| Momentum (short + long-term) | math_engine.py | Linear regression + API data |
| Order book imbalance | math_engine.py + ws_client.py | Weight 0.5 (should be 0.8) |
| FLB + certainty gradient | math_engine.py | Tail mispricing |
| NegRisk arbitrage | math_engine.py | Multi-outcome normalization |
| Volume spikes & trends | math_engine.py | De-duplicated with momentum |
| ML (XGBoost) | quant-ml | 90/10 blend, ±5% cap |
| Per-theme Bayesian calibration | history_agent.py | kelly_mult + ev_mult |
| Drawdown protection (25%) | main.py | Peak equity tracking |
| Trailing TP + vol-based SL | main.py | ATR-scaled |
| WS real-time monitoring | ws_client.py | Sub-second SL/TP |
| Portfolio correlation | math_engine.py | ρ=1.0 negRisk, ρ=0.5 theme |
| Displacement | main.py | EV>25% replaces worst |
| Bond harvesting | **quant-micro** | Separate service, 93%+ markets |

---

## 6. Gap Analysis & Implementation Plan

### Phase 1: Quick Wins (hours each)

| # | Change | Expected Impact | Effort |
|---|---|---|---|
| 1 | **Dynamic theme limits** (`max(5,min(15, 10×kelly_mult))`) | Unblock 15+ signals/hour | 10 min |
| 2 | **Trailing TP pullback 5%→3%** | Close +12% positions faster | 5 min |
| 3 | **Signal cap 5→10** (`confirmed[:10]`) | 2x more trades per cycle | 5 min |
| 4 | **Book imbalance weight 0.5→0.8** | Better short-term prediction | 5 min |
| 5 | **Hurst exponent** | Right signal type per market | 1 hour |
| 6 | **Return to v4 thresholds** (EV≥0.08, KL≥0.05) | Best historical WR (40%) | 5 min |

### Phase 2: Medium Effort (days each)

| # | Change | Expected Impact | Effort |
|---|---|---|---|
| 7 | Dynamic Bayesian Model Averaging | Auto-optimize evidence weights | 1 day |
| 8 | CLV tracking | Feedback loop for improvement | 1 day |
| 9 | Thompson Sampling for themes | Better exploration vs exploitation | 1 day |
| 10 | News fast-trigger (RSS) | Catch 30-sec alpha windows | 2 days |

### Phase 3: Strategic (weeks)

| # | Change | Expected Impact | Effort |
|---|---|---|---|
| 11 | VPIN toxicity indicator | Protect from adverse selection | 1 week |
| 12 | BOCPD regime detection | Prevent losses on regime change | 1 week |
| 13 | Limit order execution | Save 50bps round-trip | 1 week |
| 14 | Whale conviction signal | New evidence source | 1 week |
| 15 | Weather market module | New vertical ($2M+ documented) | 2 weeks |

---

## 7. References

### Academic Papers
- arXiv:2510.15205 — Logit Jump-Diffusion (Black-Scholes for Prediction Markets)
- arXiv:2508.03474 — $40M Arbitrage in Prediction Markets
- arXiv:2603.03136 — Anatomy of Polymarket (2024 Election)
- arXiv:2107.07511 — Conformal Prediction
- arXiv:2503.14814 — Hawkes Processes in HFT
- arXiv:2307.02375 — BOCPD for Order Flow
- NBER — Favorite-Longshot Bias (Snowberg & Wolfers)
- Clinton & Huang 2025 — Prediction Market Efficiency

### Industry & Community
- [Polymarket Maker Rebates](https://docs.polymarket.com/market-makers/maker-rebates)
- [Microstructure of Wealth Transfer](https://www.jbecker.dev/research/prediction-market-microstructure)
- [Mathematical Execution Behind Alpha](https://navnoorbawa.substack.com/p/the-mathematical-execution-behind)
- [QuantPedia — Systematic Edges](https://quantpedia.com/systematic-edges-in-prediction-markets/)
- [Reddit — 14 Polymarket Strategies](https://www.reddit.com/r/CryptoCurrency/comments/1payslv/14_polymarket_trading_strategies/)
- [Medium — 5 Ways to $100K](https://medium.com/@monolith.vc/5-ways-to-make-100k-on-polymarket-f6368eed98f5)
- [CryptoNews — Polymarket Strategies 2026](https://cryptonews.com/cryptocurrency/polymarket-strategies/)

### Open-Source
- [warproxxx/poly-maker](https://github.com/warproxxx/poly-maker) — Market making bot
- [suislanchez/weather-bot](https://github.com/suislanchez/polymarket-kalshi-weather-bot) — GFS ensemble
- [Polymarket/agents](https://github.com/Polymarket/agents) — Official AI framework
