# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Quant Engine v3 — a prediction market trading bot for Polymarket. Uses Bayesian probability fusion, prospect theory, price momentum, news monitoring, ML calibration, and Claude AI confirmation to generate and execute trading signals with Kelly criterion position sizing.

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

Deployed via Heroku (`Procfile: web: python main.py`). Graceful shutdown on SIGTERM/SIGINT.

## Architecture

**Fully async Python** — all I/O uses `asyncio`, `httpx`, and `asyncpg`.

### Pipeline Flow

```
Polymarket API → Scanner → Math Engine (Bayesian fusion) → News Monitor
    → Signal Ranking (Kelly × entropy penalty)
    → Claude Confirmation (if EV > 20%, cached 10min)
    → Kelly Sizing → Execution (max 5 per theme, 20 total)
    → Position Monitoring (TP/SL/Resolution)
    → History Agent → Calibrator → feedback into Math Engine
```

### Module Responsibilities

- **main.py** — Orchestrator. Runs the main event loop with three tick intervals: market scan (10s), news scan (30s), historical learning (1h). Handles signal ranking, execution, position monitoring, Claude API caching, portfolio diversification (max 5 per theme), and daily reports. Graceful shutdown on SIGTERM/SIGINT.
- **engine/scanner.py** — Fetches and filters markets from Polymarket's Gamma API. Filters by volume, liquidity, and price bounds. Classifies markets into 13 themes. Parses `outcomePrices` JSON strings and `endDate` to datetime. Builds correct Polymarket URLs from `events[0].slug`.
- **agents/math_engine.py** — Core signal generation. Bayesian probability fusion in log-odds space combining 5 sources: prospect theory (prior), historical base rates, volume spikes, time decay, and price momentum. Applies calibration correction, then filters by EV, KL divergence, Kelly fraction, and edge.
- **agents/news_monitor.py** — Scans 8 RSS feeds, detects themes and sentiment via keyword matching, matches news to markets. Only triggers signals when market price is stale (< 2¢ change in 10 min).
- **agents/history_agent.py** — Self-learning loop. Computes base rates, prospect factors, and volume patterns per theme from closed markets. Calibrates using Brier score against actual market outcomes (YES/NO), not bet results (WIN/LOSS).
- **ml/calibrator.py** — Brier score, bias, and logit-scale correction factor for probability calibration. Wired into MathEngine — `adjust()` is applied to every `p_final` before signal generation. Measures calibration against actual market outcomes.
- **utils/db.py** — PostgreSQL schema (8 tables), connection pool (asyncpg), all CRUD operations with transactional position open/close. Tables: markets, price_snapshots, news, signals, positions, patterns, calibration, stats.
- **utils/telegram.py** — Async Telegram notifications with HTML formatting and plain-text fallback.
- **dashboard/app.py** — FastAPI web UI (modern dark theme) with 15s auto-refresh. Shows bankroll, ROI, open positions, signals, and history. Error handling on both routes. Also exposes `/api` JSON endpoint.

### Key Algorithms

- **Bayesian fusion**: Prior (prospect-adjusted price) updated with evidence (history, volume, time, momentum) in log-odds space via `bayesian_update()`. Mathematically correct way to combine independent probability estimates.
- **Prospect weighting**: Kahneman-Tversky with γ=0.65, binary search to invert. Models systematic human probability mispricing.
- **Price momentum**: Linear regression slope over recent price snapshots, capped at ±5% adjustment.
- **Kelly criterion**: Quarter-Kelly (0.25x fraction), capped at MAX_KELLY_FRAC of bankroll.
- **Calibration**: Logit-scale adjustment using historical bias and correction factor from Brier score. Applied to every signal automatically.
- **Signal ranking**: `kelly × (1 - entropy × 0.3)` — Kelly already incorporates EV, entropy penalizes uncertain 50/50 markets.
- **Position management**: Take-profit (default +20%), stop-loss (default -50%), and market resolution. Closed positions free slots for re-entry.

### Configuration

All config via environment variables (see `.env.example`). Key params:
- `SIMULATION=true` (default, no real trades)
- `MIN_EV=0.05`, `MIN_KL=0.05` (lowered for data-gathering phase)
- `MAX_KELLY_FRAC=0.05`, `MAX_OPEN=20` (many small positions for learning)
- `CLAUDE_EV_THR=0.20` (only strong signals get Claude confirmation)
- `TAKE_PROFIT_PCT=0.20`, `STOP_LOSS_PCT=0.50`
- `HISTORY_INTERVAL=3600` (recalibrate every hour)

### Database

PostgreSQL required. Schema auto-created on startup by `db.init()`. Key indexes on `markets(active, volume)`, `price_snapshots(market_id, ts)`, `signals(created_at)`, `positions(status)`. Transactional writes for position open/close to prevent bankroll race conditions.
