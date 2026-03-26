# Quant Engine: Research Report — Strategies for Profitability

> Date: 2026-03-25 | Updated: 2026-03-26
> Sources: Academic papers, Twitter/X, Reddit, Medium, CryptoNews, GitHub, Substack, arXiv

---

## Table of Contents

1. [Current State](#1-current-state)
2. [Community Strategies](#2-community-strategies)
3. [Remaining Improvements](#3-remaining-improvements)
4. [Market Inefficiencies & Microstructure](#4-market-inefficiencies--microstructure)
5. [References](#5-references)

---

## 1. Current State

### What's Implemented (this session)

| Feature | Module | Status |
|---|---|---|
| Hurst exponent (momentum vs contrarian) | math_engine.py | DONE |
| Book imbalance weight 0.5→0.8 | math_engine.py | DONE |
| CLV tracking (1h/4h/24h/close) | db.py + main.py | DONE |
| CLV analytics on dashboard | dashboard analytics.html | DONE |
| DMA (Dynamic Model Averaging) | history_agent.py + math_engine.py | DONE |
| DMA weights on dashboard | dashboard analytics.html | DONE |
| All source probs saved to trade_log | main.py | DONE |
| Smoke test at startup (engine) | main.py | DONE |
| Smoke test at startup (micro) | micro/main.py | DONE |
| quant-micro: YES token fix | micro/scanner.py + main.py | DONE |
| quant-micro: WS sanity (wild ticks, book) | micro/ws_client.py | DONE |
| quant-micro: risky filter (24 patterns) | micro/scanner.py | DONE |
| quant-micro: date parsing from questions | micro/scanner.py | DONE |
| quant-micro: API fallback for token restore | micro/main.py | DONE |
| Dashboard: Scalping page (micro stats) | dashboard app.py + scalping.html | DONE |

### Config Tag Performance

| Tag | Trades | WR | Total PnL | Key Params |
|---|---|---|---|---|
| v4 | 96 | 40% | -$9.20 | EV≥0.08 KL≥0.05 |
| v5 | 66 | 27% | -$23.22 | EV≥0.10 KL≥0.07 |
| v6 | 7 | 29% | -$2.13 | EV≥0.10 KL≥0.07 Kelly:0.2 |

**Key insight**: v4 had best WR but still lost money due to TP/SL ratio (breakeven WR=63%).

---

## 2. Community Strategies

### Applicable but NOT yet implemented

| Strategy | Source | Action |
|---|---|---|
| Whale conviction as evidence source | Reddit #2, Medium #2 | On-chain wallet tracking |
| News Scalping — first 30 sec | Reddit #3 | RSS/Telegram fast-trigger |
| Cultural Calendar — fade hype near holidays | Reddit #14 | Holiday calendar modifier |
| Riskless Rate Discounting | Reddit #11 | Risk-free rate for long markets |

### Not Applicable

- Personality-driven mention trading (Reddit #4)
- "Girlfriend Poll" (Reddit #8)
- Partial Resolution Advocacy (Reddit #13)
- Market Making / LP (Medium #5)
- Fed Signal Trading (Reddit #5)

---

## 3. Remaining Improvements

### Quick Wins (not yet done)

| # | Change | Expected Impact | Effort |
|---|---|---|---|
| 1 | **Dynamic theme limits** (`max(5,min(15, 10×kelly_mult))`) | Unblock 15+ signals/hour | 10 min |
| 2 | **Trailing TP pullback 5%→3%** | Close +12% positions faster | 5 min |
| 3 | **Signal cap 5→10** (`confirmed[:10]`) | 2x more trades per cycle | 5 min |
| 4 | **Fix TP/SL ratio** (TP:20% SL:15%) | Breakeven WR 63%→43% | 5 min |
| 5 | **Return to v4 thresholds** (EV≥0.08, KL≥0.05) | Best historical WR (40%) | 5 min |

### Medium Effort

| # | Change | Expected Impact | Effort |
|---|---|---|---|
| 6 | Thompson Sampling for themes | Better exploration vs exploitation | 1 day |
| 7 | News fast-trigger (RSS) | Catch 30-sec alpha windows | 2 days |
| 8 | Limit order execution | Save 50bps round-trip | 1 week |

### Strategic

| # | Change | Expected Impact | Effort |
|---|---|---|---|
| 9 | VPIN toxicity indicator | Protect from adverse selection | 1 week |
| 10 | BOCPD regime detection | Prevent losses on regime change | 1 week |
| 11 | Whale conviction signal | New evidence source | 1 week |
| 12 | Weather market module | New vertical ($2M+ documented) | 2 weeks |

---

## 4. Market Inefficiencies & Microstructure

### Documented Inefficiencies

| Finding | Scale | Source |
|---|---|---|
| $40M arbitrage extracted from Polymarket (2024-2025) | 41% of markets | arXiv:2508.03474 |
| Longshots (<10¢) overpriced, buyers lose >60% | Systematic | NBER |
| Takers lose 1.12%/trade, makers gain 1.12% | Structural | jbecker.dev |
| YES underperforms NO by up to 64pp at longshot prices | Structural | jbecker.dev |
| Entertainment: 4.79% maker-taker gap | Category-dependent | jbecker.dev |

### Behavioral Biases

| Bias | Exploitable? | Our Coverage |
|---|---|---|
| Favorite-Longshot | Yes | Prospect theory (γ=0.65) |
| Partisan bias | Yes | None → detector needed |
| Recency / overreaction | Yes | Contrarian signal (active) |
| YES optimism | Yes | FLB partially covers |
| "Nothing happens" | Yes | Contrarian signal |

---

## 5. References

### Academic Papers
- arXiv:2510.15205 — Logit Jump-Diffusion
- arXiv:2508.03474 — $40M Arbitrage in Prediction Markets
- arXiv:2603.03136 — Anatomy of Polymarket
- arXiv:2107.07511 — Conformal Prediction
- arXiv:2503.14814 — Hawkes Processes
- arXiv:2307.02375 — BOCPD for Order Flow
- NBER — Favorite-Longshot Bias

### Industry & Community
- [Reddit — 14 Polymarket Strategies](https://www.reddit.com/r/CryptoCurrency/comments/1payslv/14_polymarket_trading_strategies/)
- [Medium — 5 Ways to $100K](https://medium.com/@monolith.vc/5-ways-to-make-100k-on-polymarket-f6368eed98f5)
- [CryptoNews — Polymarket Strategies 2026](https://cryptonews.com/cryptocurrency/polymarket-strategies/)
- [Microstructure of Wealth Transfer](https://www.jbecker.dev/research/prediction-market-microstructure)
- [QuantPedia — Systematic Edges](https://quantpedia.com/systematic-edges-in-prediction-markets/)
