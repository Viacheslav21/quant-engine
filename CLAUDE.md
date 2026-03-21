# CLAUDE.md

This Saved by Rejection
file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Quant Engine v3 — a prediction market trading bot for Polymarket. Uses Bayesian probability fusion (9 evidence sources), prospect theory, price momentum, mean reversion, negRisk arbitrage, news monitoring, ML calibration, and Claude AI confirmation to generate and execute trading signals with Kelly criterion position sizing.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # then edit with API keys

# Run the bot
python main.py
```

No build step, no test suite, no linter configured. Logging goes to `quant.log` and stdout. Set `logging.DEBUG` in main.py to see signal rejection details.

Dashboard runs on `http://localhost:3000` (FastAPI + Uvicorn, started automatically by main.py).

Deployed via Railway (`Procfile: web: python main.py`). Graceful shutdown on SIGTERM/SIGINT.

## Architecture

**Fully async Python** — all I/O uses `asyncio`, `httpx`, and `asyncpg`.

### Pipeline Flow

```
Polymarket API → Scanner (500 markets, paginated)
    → Math Engine (9 evidence sources → Bayesian fusion → drift cap ±15%)
    → News Monitor (RSS feeds)
    → Signal Ranking (Kelly × entropy penalty)
    → Signal Cooldown (5 min per market)
    → Claude Confirmation (if EV > 20%, max 1/min, cache 30min)
    → Kelly Sizing (0.15 fraction, spread penalty)
    → Execution (max 5 per theme, 50 total, displacement)
    → Position Monitoring (per-position TP/SL, trailing TP, resolution detection)
    → History Agent → Calibrator → feedback into Math Engine
    → Daily Sonnet Analysis → Telegram
```

### Module Responsibilities

- **main.py** (~580 lines) — Orchestrator. Main event loop with tick intervals: market scan (10s), news scan (30s), historical learning (4h). Signal ranking, Claude confirmation with rate limiting (1/min, 30min cache), execution with displacement logic, position monitoring with trailing TP, daily AI analysis via Sonnet. DB write optimization (skip unchanged prices). Signal cooldown (5 min per market). Config tag tracking for A/B testing. Marks signals as executed after successful trade for backtest tracking. Graceful shutdown on SIGTERM/SIGINT.
- **engine/scanner.py** (~120 lines) — Fetches up to 500 markets from Polymarket's Gamma API with pagination. Filters by volume (>$50k), liquidity (>$5k), price bounds (3-97¢). Extracts: spread, bestAsk, competitive, oneWeekPriceChange, oneMonthPriceChange, negRisk, negRiskMarketID, volume1wk, volume1mo. Classifies into 13 themes via keyword matching.
- **agents/math_engine.py** (~467 lines) — Core signal generation. 9 evidence sources fused via Bayesian log-odds:
  1. Prospect theory (prior) — inverts human probability weighting (γ=0.65)
  2. Historical base rates per theme
  3. Volume spike detection (>2.5x average, only records changed values)
  4. Time decay (near expiry → trust market more)
  5. Price momentum (5-min linear regression, ±5% cap)
  6. Mean reversion / contrarian (>8% move on low volume → bet on reversion via EWMA)
  7. Long-term momentum (week/month price changes from API, threshold 2%/5%)
  8. Volume trend (24h vs weekly average)
  9. NegRisk arbitrage (multi-outcome events, sum ≠ 1.0)
  Correlated evidence de-duplication (max of momentum pair, max of volume pair). Drift cap ±15% from market price. Spread penalty on Kelly. bestAsk for YES-side real entry price. Per-market volatility calculation (ATR-style from 30-min price cache). Rejects: EV < 15%, KL < 0.12, edge < 10%, market > 30 days out.
- **agents/news_monitor.py** — Scans 8 RSS feeds, keyword sentiment, matches to markets. Triggers signals when price stale (< 2¢ change in 10 min).
- **agents/history_agent.py** (~107 lines) — Self-learning. Base rates & prospect factors per theme from closed markets. Volume patterns. Calibration via Brier score on RESOLVED positions only (not TP/SL).
- **ml/calibrator.py** (~76 lines) — Brier score, bias, logit-scale correction. `adjust()` applied to every `p_final`.
- **utils/db.py** (~550 lines) — PostgreSQL schema (9 tables including config_history), connection pool, CRUD, analytics queries (by theme/source/side/config_tag/calibration), cumulative PnL, signal outcomes for backtest, DB cleanup with configurable retention. Migrations: tp_pct/sl_pct/config_tag columns, backfill executed signals from positions.
- **utils/telegram.py** (~33 lines) — Async Telegram notifications with HTML formatting.
- **dashboard/app.py** (~778 lines) — FastAPI web UI (dark theme):
  - `/` — Bankroll, ROI, Win Rate, EV, open positions, PnL chart (Chart.js), signals, history with pagination (100/page). Hover tooltips on all headers.
  - `/analytics` — Config A/B comparison table, WR by theme/source/side, close reasons, calibration chart + table, cumulative PnL chart, daily PnL bar chart, win rate by theme chart, signal backtest (executed vs rejected).
  - `/api` — JSON stats endpoint.

### Key Algorithms

- **Bayesian fusion**: Prior (prospect-adjusted price) updated with up to 6 de-duplicated evidence sources in log-odds space. Drift capped at ±15% from market price.
- **Prospect weighting**: Kahneman-Tversky with γ=0.65, binary search to invert.
- **Price momentum**: Linear regression slope over 30-point cache, capped at ±5%.
- **Mean reversion**: 180-point long cache (~30 min). Detects >8% moves, volume filter (>2.5x = skip, 1.5-2.5x = weak, <1.5x = strong), EWMA reversion target, confidence-weighted shift.
- **NegRisk arbitrage**: Groups markets by negRiskMarketID, normalizes prices to sum=1.0.
- **Kelly criterion**: 0.15 fraction (conservative), spread penalty (3-10¢ → 1.0-0.3x multiplier), capped at MAX_KELLY_FRAC of bankroll. Contrarian trades: Kelly × 0.5.
- **Signal ranking**: `kelly × (1 - entropy × 0.3)` — penalizes 50/50 markets.
- **Position management**: Per-position TP/SL with volatility-based SL. SL = 2.5 × ATR / entry_price (floor 8%, cap at default). Default SL: normal 30%, contrarian 25%. TP: normal 20%, contrarian 10%. Trailing TP: tracks peak PnL, closes on 5% pullback from peak when peak ≥ 50% of TP target. Resolution detection: fetches closed markets directly via API, threshold 95¢/5¢.
- **Displacement**: When 50 slots full, new signal (EV > 25%) can close worst position. Profitable positions displaced easily; losing positions only if new EV > 2× old EV.
- **Claude confirmation**: Haiku with web search, max 1 call/min, 30-min cache, drift re-cap after p_final blending. Fallback: reject (not confirm).
- **Daily AI analysis**: Sonnet once daily (first tick after 8:00 UTC), full analytics summary → actionable recommendations in Telegram.

### Configuration

All config via environment variables. Key params:
- `SIMULATION=true` (default, no real trades)
- `MIN_EV=0.15`, `MIN_KL=0.12` (tight thresholds, quality over quantity)
- `MAX_KELLY_FRAC=0.15`, `MAX_OPEN=50` (conservative sizing, many positions)
- `CLAUDE_EV_THR=0.20` (only strong signals get Claude confirmation)
- `TAKE_PROFIT_PCT=0.20`, `STOP_LOSS_PCT=0.30`
- `TRAILING_TP=true` (trailing take-profit enabled)
- `MAX_MARKET_DAYS=30` (skip markets closing > 30 days out)
- `CONFIG_TAG=v1` (A/B testing tag, saved to DB with full config snapshot)
- `HISTORY_INTERVAL=14400` (recalibrate every 4 hours)
- `SCAN_INTERVAL=10` (seconds between market scans)

### Database

PostgreSQL required (500MB plan). Schema auto-created on startup by `db.init()`. 9 tables: markets, price_snapshots, news, signals, positions, patterns, calibration, stats, config_history. Migrations run automatically for new columns (tp_pct, sl_pct, config_tag) and backfill (executed signals from positions table). Signals marked `executed=TRUE` after trade for backtest analytics. Cleanup runs every HISTORY_INTERVAL: snapshots (1d), unexecuted signals (7d), processed news (5d). Positions and markets kept forever (needed for analytics/backtest). VACUUM after cleanup. DB writes optimized: skip unchanged market prices.

### Performance Optimizations

- Single scanner.fetch() per cycle (shared with monitor_positions)
- DB writes only for changed prices (~80% reduction)
- Signal cooldown 5 min per market (prevents spam)
- Claude: 1 call/min max, 30-min cache
- News markets not re-analyzed in math loop
- MathEngine instance reused (not recreated per execute)
- Volume history only records changed values
