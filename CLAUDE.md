# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Quant Engine v3 — a prediction market trading bot for Polymarket. Uses prospect theory, news monitoring, ML calibration, and Claude AI confirmation to generate and execute trading signals with Kelly criterion position sizing.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # then edit with API keys

# Run the bot
python main.py
```

No build step, no test suite, no linter configured. Logging goes to `quant.log` and stdout.

Dashboard runs on `http://localhost:3000` (FastAPI + Uvicorn, started automatically by main.py).

Deployed via Heroku (`Procfile: web: python main.py`).

## Architecture

**Fully async Python** — all I/O uses `asyncio`, `httpx`, and `asyncpg`.

### Pipeline Flow

```
Polymarket API → Scanner → Math Engine → News Monitor → Signal Ranking
    → Claude Confirmation (if EV > 15%) → Kelly Sizing → Execution → Position Monitoring
```

### Module Responsibilities

- **main.py** — Orchestrator. Runs the main event loop with three tick intervals: market scan (10s), news scan (30s), historical learning (4h). Handles signal ranking, execution, position monitoring, and daily reports.
- **engine/scanner.py** — Fetches and filters markets from Polymarket's Gamma API. Filters by volume, liquidity, and price bounds. Classifies markets into 13 themes.
- **agents/math_engine.py** — Core signal generation. Fuses four probability sources (prospect theory 40%, historical patterns 35%, volume spikes 15%, time decay 10%). Applies EV, Kelly, and KL divergence filters.
- **agents/news_monitor.py** — Scans 8 RSS feeds, detects themes and sentiment via keyword matching, matches news to markets. Only triggers signals when market price is stale (< 2¢ change in 10 min).
- **agents/history_agent.py** — Self-learning loop. Computes base rates, prospect factors, and volume patterns per theme from closed markets. Runs every 4 hours.
- **ml/calibrator.py** — Brier score, bias, and logit-scale correction factor for probability calibration.
- **utils/db.py** — PostgreSQL schema (8 tables), connection pool (asyncpg), all CRUD operations. Tables: markets, price_snapshots, news, signals, positions, patterns, calibration, stats.
- **utils/telegram.py** — Async Telegram notifications with HTML formatting.
- **dashboard/app.py** — FastAPI web UI with auto-refresh. Shows bankroll, ROI, open positions, signals, and history. Also exposes `/api` JSON endpoint.

### Key Algorithms

- **Prospect weighting**: Kahneman-Tversky with γ=0.65, binary search to invert
- **Kelly criterion**: Quarter-Kelly (0.25x fraction), capped at 15% of bankroll
- **Multi-source fusion**: Weighted average of prospect, historical, volume, and time-decay probabilities
- **Calibration**: Logit-scale adjustment using historical bias and correction factor

### Configuration

All config via environment variables (see `.env.example`). Key params: `SIMULATION=true` (default, no real trades), `MIN_EV=0.08`, `MAX_KELLY_FRAC=0.15`, `MAX_OPEN=5`, `MIN_VOLUME=50000`.

### Database

PostgreSQL required. Schema auto-created on startup by `db.init()`. Key indexes on `markets(active, volume)`, `price_snapshots(market_id, ts)`, `signals(created_at)`, `positions(status)`.
