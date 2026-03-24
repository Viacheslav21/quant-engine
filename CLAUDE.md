# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Quant Engine v3 — a prediction market trading bot for Polymarket. Uses Bayesian probability fusion (9 raw evidence sources → 6 de-duplicated), prospect theory, price momentum, mean reversion, negRisk arbitrage, ML calibration, and optional Claude AI confirmation to generate and execute trading signals with Kelly criterion position sizing. Includes drawdown protection, theme concentration limits, and comprehensive trade logging to PostgreSQL.

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

Deployed via Railway (`Procfile: worker: python main.py`). Graceful shutdown on SIGTERM/SIGINT via `_shutdown_flag`. Dashboard is a separate service (quant-dashboard).

## Architecture

**Fully async Python** — all I/O uses `asyncio`, `httpx`, `asyncpg`, and `websockets`.

### Pipeline Flow

```
Polymarket API → Scanner (500 markets, paginated, every 5 min)
    → Drawdown Check (halt new trades if equity drops ≥25% from peak)
    → Math Engine (9 evidence sources → de-duplicate correlated pairs → 6 sources → Bayesian fusion → adaptive drift cap)
    → ML Enrichment (top 5 signals: XGBoost blend 90% math + 10% ML, cap ±5%)
    → Signal Ranking (Kelly × entropy penalty)
    → Signal Cooldown (5 min per market)
    → Optional Claude Confirmation (if enabled + EV > threshold)
    → Kelly Sizing (0.15 fraction, spread penalty, theme concentration penalty, per-theme Bayesian multiplier)
    → Execution (max per-theme limit, MAX_OPEN total, displacement)
    → WS Subscribe (on new position) + instant _ws_positions refresh
    → Position Monitoring:
        - Primary: WebSocket real-time (<1s reaction to SL/TP/book events, bid-price based)
        - Fallback: REST check every scan cycle (5 min)
    → Trader Commands (dashboard → trader_commands table → LISTEN/NOTIFY → engine executes)
    → History Agent → Calibrator → Theme Performance (Bayesian shrinkage) → feedback into Math Engine
    → Daily Report → Telegram (full analytics at 08:00 UTC)
    → trade_log table (all events persisted to PostgreSQL)
```

### Module Responsibilities

- **main.py** (~1000 lines) — Orchestrator. Two-speed architecture: REST market scan every 5 min (signal generation, DB), WebSocket real-time position monitoring (<1s SL/TP reaction on price_change, trade, AND book events, using bid-price for realistic exit pricing). Drawdown halt (≥25% equity drop from peak stops new trades, peak restored from real equity on restart). Signal ranking, optional Claude confirmation, execution with displacement logic (returns bool, aborts on race), dynamic WS subscribe/unsubscribe on position open/close with instant `_ws_positions` refresh. Trailing TP, daily analytics report. DB write optimization (throttled to 30s per position). Signal cooldown (5 min per market). Config tag tracking for A/B testing. Whale alerts on positions ($500+). Trader commands processing via `trader_commands` table with LISTEN/NOTIFY for instant reaction from dashboard (manual close, etc.). Comprehensive trade_log events (22 event types incl. CLOSE_MANUAL). Graceful shutdown via `_shutdown_flag` on SIGTERM/SIGINT.
- **engine/ws_client.py** (~260 lines) — WebSocket client for real-time Polymarket price updates. Connects to `wss://ws-subscriptions-clob.polymarket.com/ws/market`. Handles `price_change`, `last_trade_price`, `book` events (all async, all trigger SL/TP callback). Dynamic subscribe/unsubscribe per market. Token membership checked BEFORE `register_market` to avoid silent subscribe failures. YES/NO token price conversion. Heartbeat (10s), auto-reconnect (5s delay). Batch subscriptions (100 tokens). Callbacks: `on_price_change` (SL/TP), `on_trade` (whale alerts), `on_disconnect`/`on_reconnect` (logged to trade_log).
- **engine/scanner.py** (~138 lines) — Fetches up to 500 markets from Polymarket's Gamma API with pagination. Filters by volume (>$50k), liquidity (>$5k), price bounds (3-97¢). Extracts: spread, bestAsk, oneWeekPriceChange, oneMonthPriceChange, negRisk, negRiskMarketID, volume1wk, volume1mo, clobTokenIds (YES/NO token IDs for WS). Classifies into 13 themes via keyword matching.
- **agents/math_engine.py** (~580 lines) — Core signal generation. 9 raw evidence sources, de-duplicated to 6 via max() on correlated pairs, fused via weighted Bayesian log-odds:
  1. Prospect theory (prior) — inverts human probability weighting (γ=0.65), configurable via `USE_PROSPECT` env var
  2. Historical base rates per theme (requires ≥10 bets, base_rate×prospect_factor clamped to [0.05, 0.95])
  3. Volume spike detection (>2.5x average, direction from price momentum cache)
  4. Time decay (near expiry → trust market more, uses neutral 0.5 anchor not prospect), weight 0.5
  5. Price momentum (5-min linear regression, ±5% cap)
  6. Mean reversion / contrarian (>8% move on low volume → bet on reversion via EWMA)
  7. Long-term momentum (week/month price changes from API, threshold 2%/5%)
  8. Volume trend (24h vs weekly average)
  9. NegRisk arbitrage (multi-outcome events, sum ≠ 1.0)
  10. Order book imbalance (WS book events: bid/ask volume ratio, threshold |imbalance| > 0.3, weight 0.5)
  Correlated pair de-duplication: max(p_momentum, p_long_mom) and max(p_volume, p_vol_trend) before fusion → 7 independent sources. Evidence weights: history=1.0, vol_combined=1.0, time=0.5, mom_combined=1.0, contrarian=1.0, arb=1.0, book=0.5. Adaptive drift cap: ±8% (0-1 sources), ±12% (2-3), ±18% (4+). Spread penalty on Kelly. bestAsk for YES-side real entry price. Per-market volatility (ATR from 30-min price cache). **Per-theme adaptive thresholds**: EV/KL/edge minimums scaled by `ev_mult` from Bayesian theme calibration (losing themes need higher EV to enter). **Portfolio correlation penalty**: positions in same negRisk group (ρ=1.0) or theme (ρ=0.5) reduce effective independent bets via `effective_n = n / (1 + (n-1) × ρ)`. Stake limited to 5% bankroll per effective bet. Worst-case check: if entire cluster hits SL > 15% bankroll → reduce. **Per-theme Kelly multiplier**: `kelly_mult` from Bayesian shrinkage scales Kelly before stake calculation (winning themes get bigger bets). `compute_stake` guards against negative bankroll (returns 0). Rejects: EV < 12%×ev_mult, KL < 0.10×ev_mult, Kelly < 0.01, edge < 8%×ev_mult, market > 30 days out.
- **agents/history_agent.py** (~180 lines) — Self-learning. Base rates & prospect factors per theme from closed markets. Volume patterns (high vs low volume win rates). **Per-theme Bayesian performance calibration**: computes `kelly_mult` and `ev_mult` per theme using empirical Bayes shrinkage (k=20) — themes with few trades shrink toward global mean, themes with many trades converge to their true performance. Outputs: `kelly_mult` (0.3–2.0, scales bet size), `ev_mult` (0.7–2.0, scales entry thresholds). ROI-adjusted: losing themes get double penalty (lower Kelly + higher EV bar). Calibration via Brier score on RESOLVED positions only (not TP/SL).
- **ml/calibrator.py** (~90 lines) — Brier score, logit-scale correction via factor only (no double bias subtraction). Window: last 300 positions (RESOLVED outcomes only, not TP/SL). Factor bounds [0.7, 1.3]. Only adjusts if |bias| > 0.05 and Brier < 0.25. `adjust()` applied to every `p_final`.
- **utils/db.py** (~850 lines) — PostgreSQL schema (12 tables including market_metrics, trade_log, and trader_commands), connection pool, CRUD, analytics queries (by theme/source/side/config_tag/calibration), cumulative PnL, signal outcomes for backtest, DB cleanup with configurable retention. `close_position` has race protection (`WHERE status='open' RETURNING id`). `log_event` method for fire-and-forget trade logging. `build_report` generates comprehensive daily analytics. Trader commands: `fetch_pending_commands` (atomic status='processing'), `complete_command`, `fail_command`, `setup_listen` (LISTEN/NOTIFY). Migrations auto-run on startup for new columns (patterns: trade_n, trade_wr, trade_roi, kelly_mult, ev_mult).
- **utils/telegram.py** (~33 lines) — Async Telegram notifications with HTML formatting.

### Key Algorithms

- **Bayesian fusion**: Prior (prospect-adjusted price) updated with up to 6 de-duplicated evidence sources in weighted log-odds space. Correlated pairs combined via max() before fusion. Adaptive drift cap: ±8% (0-1 sources), ±12% (2-3), ±18% (4+).
- **Prospect weighting**: Kahneman-Tversky with γ=0.65, binary search to invert. Configurable via `USE_PROSPECT` env var for A/B testing.
- **Price momentum**: Linear regression slope over 30-point cache, capped at ±5%.
- **Mean reversion**: 180-point long cache (~30 min). Detects >8% moves, volume filter (>2.5x = skip, 1.5-2.5x = weak, <1.5x = strong), EWMA reversion target, confidence-weighted shift.
- **NegRisk arbitrage**: Groups markets by negRiskMarketID, normalizes prices to sum=1.0.
- **Kelly criterion**: 0.15 fraction (conservative), spread penalty (3-10¢ → 1.0-0.3x multiplier), capped at MAX_KELLY_FRAC of bankroll. **Uncertainty scaling**: `kelly × (0.5 + 0.5 × n_sources/7)` — 1 evidence source → Kelly ×0.57, all 7 → Kelly ×1.0. Contrarian trades: Kelly × 0.5. Per-theme Bayesian Kelly multiplier (0.3–2.0×) from trade history. Portfolio correlation penalty (see below).
- **Bayesian theme calibration**: Empirical Bayes shrinkage with k=20. Per-theme adjusted WR = `(n × raw_wr + k × global_wr) / (n + k)`. Kelly multiplier = `adj_wr / global_wr`. EV threshold multiplier combines WR ratio with ROI penalty (losing themes: `1 + |roi| × 3`, profitable: `max(0.7, 1 - roi × 2)`). Recalculated every HISTORY_INTERVAL (4h). Protects against overfitting: themes with <20 trades stay near global behavior.
- **Portfolio correlation**: Positions in same negRisk group (ρ=1.0) or theme (ρ=0.5) are correlated. Effective independent bets = `n / (1 + (n-1) × ρ)`. Three checks: (1) negRisk group: positions sharing negRiskMarketID count as 1 effective bet, max 5% bankroll per effective bet. (2) Theme cluster: same formula with ρ=0.5, e.g. 14 iran positions = 1.9 effective bets. (3) Worst-case: if all theme positions hit SL simultaneously, loss must stay < 15% bankroll. Penalty applied as multiplier on stake (min 0.05×). Only limits new positions, never closes existing.
- **Order book imbalance**: 10th evidence source from WS book events. `imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)` over top-5 price levels. Threshold |imbalance| > 0.3. Shift: 0.3→±1%, 1.0→±4%. Weight 0.5 in Bayesian fusion (short-term, noisy). Flipped for NO tokens.
- **Bid-price SL/TP**: Position monitoring uses bid price (realistic exit price) instead of mid-price. YES positions: YES best_bid. NO positions: 1 − YES best_ask (= NO bid). Prevents premature closes from spread inflation and matches real execution price.
- **Signal ranking**: `kelly × (1 - entropy × 0.3)` — penalizes 50/50 markets.
- **Position management**: Per-position TP/SL with volatility-based SL. SL = 2.5 × ATR / entry_price (floor 8%, cap at default). Default SL: normal 30%, contrarian 25%. TP: normal 20%, contrarian 10%. Trailing TP: tracks peak PnL, closes on 5% pullback from peak when peak ≥ 50% of TP target. Resolution detection: API `is_closed` flag or extreme price (≥99¢/≤1¢); price-based uses linear payout (not binary) as safety.
- **Displacement**: When slots full, new signal (EV > 25%) can close worst position. Profitable positions displaced easily; losing positions only if new EV > 2× old EV. Returns bool, caller aborts if displacement failed (race protection).
- **Drawdown protection**: Tracks peak equity (free cash + position values). Peak restored from real equity on restart (not just BANKROLL env). Halts all new trades when drawdown ≥ 25%. Continues monitoring and closing existing positions. Resume after recovery + 30 min cooldown.
- **Double-close protection**: `close_position` uses `WHERE status='open' RETURNING id` — concurrent WS + REST close attempts are safe.
- **Claude confirmation** (optional): Haiku with web search, max 1 call/min, 30-min cache. Blends: 0.6 × p_final + 0.4 × p_claude, then re-caps drift to ±15%. Fallback: reject (not confirm). Currently disabled — math-only mode.

### Trade Log (trade_log table)

Append-only audit trail in PostgreSQL. Never cleaned up. 22 event types:
- **STARTUP**, **SHUTDOWN** — lifecycle
- **SCAN** — each cycle with equity, drawdown, signal counts
- **SIGNAL_GENERATED** — all probabilities, EV, Kelly, edge, contrarian, momentum, spread, volatility
- **SIGNAL_REJECTED** (5 reasons) — duplicate_market, theme_limit, slots_full_low_ev, no_displaceable_position, displacement_failed_race, stake_too_small
- **OPEN** — full position parameters
- **CLOSE_TP**, **CLOSE_SL**, **CLOSE_TRAILING_TP**, **CLOSE_RESOLVED** — PnL, payout, win rate
- **CLOSE_MANUAL** — manual close from dashboard via trader_commands
- **DISPLACEMENT** — closed position details
- **WHALE** — large trades on held positions
- **WS_DISCONNECT**, **WS_RECONNECT** — connection events
- **DRAWDOWN_HALT**, **DRAWDOWN_RESUME** — risk events
- **HISTORY_RECALC** — recalibration
- **DAILY_REPORT** — daily snapshot

Columns: 35 typed columns + JSONB `details` for event-specific data. Indexed by event_type, market_id, created_at.

### Configuration

All config via environment variables. Key params:
- `SIMULATION=true` (default, no real trades)
- `USE_PROSPECT=true` (prospect theory prior, disable for A/B testing)
- `MIN_EV=0.12`, `MIN_KL=0.10` (signal acceptance thresholds)
- `MIN_EDGE=0.08` (minimum |p_final - p_market|)
- `MAX_KELLY_FRAC=0.15`, `MAX_OPEN=75` (conservative sizing, many positions)
- `TAKE_PROFIT_PCT=0.20`, `STOP_LOSS_PCT=0.30`
- `TRAILING_TP=true` (trailing take-profit enabled)
- `MAX_MARKET_DAYS=30` (skip markets closing > 30 days out)
- `MAX_DRAWDOWN=0.25` (halt trading at 25% equity drawdown from peak)
- `CONFIG_TAG=v3` (A/B testing tag, saved to DB with full config snapshot)
- `HISTORY_INTERVAL=14400` (recalibrate every 4 hours)
- `SCAN_INTERVAL=300` (seconds between REST market scans; positions monitored in real-time via WebSocket)

### Database

PostgreSQL required. Schema auto-created on startup by `db.init()`. 12 tables: markets, price_snapshots, news, signals, positions, patterns, calibration, stats, config_history, market_metrics, trade_log, trader_commands. Migrations run automatically for new columns (positions: tp_pct, sl_pct, config_tag; patterns: trade_n, trade_wr, trade_roi, kelly_mult, ev_mult) and backfill (executed signals from positions table). Signals marked `executed=TRUE` after trade for backtest analytics. `trader_commands` table enables dashboard→engine communication: dashboard INSERTs commands + NOTIFY, engine LISTENs + polls each cycle. Cleanup runs every HISTORY_INTERVAL: snapshots (1d), unexecuted signals (7d), processed news (5d). Positions, markets, and trade_log kept forever. VACUUM after cleanup. DB writes throttled: price updates every 30s per position, market upserts skip unchanged prices.

### Risk Management

- **Drawdown halt**: Equity = free cash + sum(position values). Peak restored from real equity on restart. If drawdown ≥ 25% → halt all new trades, continue monitoring/closing existing positions.
- **Double-close protection**: `close_position` atomic with `WHERE status='open' RETURNING id`. WS + REST can't double-count.
- **Portfolio correlation**: Three-layer protection against correlated risk. (1) NegRisk groups (ρ=1.0): positions sharing negRiskMarketID = 1 effective bet, max 5% bankroll. (2) Theme clusters (ρ=0.5): 14 iran positions = ~2 effective bets, stake limited per effective bet. (3) Worst-case: all theme positions hit SL simultaneously must be < 15% bankroll. Only restricts new entries, never closes existing.
- **Position limits**: Max open (configurable, default 75), max 10 per theme.
- **Volatility-based SL**: ATR-scaled stop loss (2.5 × ATR / entry_price), floor 8%, cap at default SL.
- **Trailing TP**: Tracks peak profit, closes on 5% pullback once peak ≥ 50% of TP target.
- **Displacement**: Only when EV > 25%; losing positions protected unless new signal is 2× better. Race-safe (returns bool).
- **Conservative Kelly**: 0.15 fraction of full Kelly. Contrarian trades halved again. Zero stake on negative bankroll.
- **Resolution safety**: API-confirmed resolution uses binary payout; price-based (≥99¢) uses linear payout to prevent spike false positives.

### Performance Optimizations

- **WebSocket position monitoring** — sub-second SL/TP reaction on price_change, trade, AND book events. REST scan every 5 min (~30x fewer API calls)
- Dynamic WS subscribe/unsubscribe — only track tokens of open positions, not all 500 markets
- Instant `_ws_positions` refresh after position open — no 5-min gap
- REST fallback — monitor_positions still runs each scan cycle as safety net
- DB write throttling — price updates every 30s per position, market upserts skip unchanged prices
- Signal cooldown 5 min per market (prevents spam)
- MathEngine instance reused (not recreated per execute)
- Volume history only records changed values
- trade_log is fire-and-forget (never blocks main loop on failure)
