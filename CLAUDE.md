# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Quant Engine v3 — a prediction market trading bot for Polymarket. Uses Bayesian probability fusion (13 raw evidence sources → 8 de-duplicated), Hurst exponent for signal selection, DMA (Dynamic Model Averaging) for adaptive evidence weights, prospect theory, price momentum, mean reversion, negRisk arbitrage, ML calibration, and optional Claude AI confirmation to generate and execute trading signals with Kelly criterion position sizing. Includes CLV tracking, drawdown protection, theme concentration limits, smoke tests at startup, and comprehensive trade logging to PostgreSQL.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env  # then edit with API keys

# Run the bot
python main.py
```

Pre-launch smoke tests: `python tests/smoke_test.py` (85 offline unit tests, no DB required). No linter configured. Logging goes to `quant.log` and stdout. Set `logging.DEBUG` in main.py to see signal rejection details.

Deployed via Railway (`Procfile: worker: python main.py`). Graceful shutdown on SIGTERM/SIGINT via `_shutdown_flag`. Dashboard is a separate service (quant-dashboard).

## Architecture

**Fully async Python** — all I/O uses `asyncio`, `httpx`, `asyncpg`, and `websockets`.

### Pipeline Flow

```
Polymarket API → Scanner (500 markets, paginated, every 5 min)
    → Drawdown Check (halt new trades if equity drops ≥25% from peak)
    → Short-term filter (block "Up or Down", 5-min direction bets — pure noise)
    → Expired question date filter (parse date from question text for negRisk sub-markets)
    → Math Engine (13 evidence sources → de-duplicate correlated pairs → 8 sources → Bayesian fusion → adaptive drift cap)
    → ML Enrichment (top 5 signals: XGBoost blend 90% math + 10% ML, cap ±5%)
    → Signal Ranking (Kelly × entropy penalty)
    → Signal Cooldown (5 min per market)
    → Escalating Loss Cooldown (per market_id: 1st SL→2h, 2nd→8h, 3rd+→24h block)
    → Optional Claude Confirmation (if enabled + EV > threshold)
    → MAX_EV cap (default 0.18 — EV>18% signals are overconfident, data shows 46% WR at EV 20-30%)
    → Pre-Execution Price Recheck (fresh API call: reject if closed/in-review/stale price, recompute EV)
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

- **main.py** (~1680 lines) — Orchestrator. Two-speed architecture: REST market scan every 5 min (signal generation, DB), WebSocket real-time position monitoring (<1s SL/TP reaction on price_change, trade, AND book events, using bid-price for realistic exit pricing). Drawdown halt (≥25% equity drop from peak stops new trades, peak restored from real equity on restart). Signal ranking, optional Claude confirmation, pre-execution price recheck (fresh API call to catch stale prices / closed / in-review markets), execution with displacement logic (returns bool, aborts on race), dynamic WS subscribe/unsubscribe on position open/close with instant `_ws_positions` refresh. Trailing TP, daily analytics report. DB write optimization: batch market upserts + snapshots (executemany), batch CLV updates, throttled position price writes (30s). Event loop yielded every 50 markets during signal generation. REST fallback skipped for WS-active positions (<30s fresh data). Signal cooldown (5 min per market). Config tag tracking for A/B testing. Whale alerts on positions ($500+). Trader commands processing via `trader_commands` table with LISTEN/NOTIFY for instant reaction from dashboard (manual close, etc.). Comprehensive trade_log events (19 event types incl. CLOSE_MANUAL). Graceful shutdown via `_shutdown_flag` on SIGTERM/SIGINT. Contrarian signals (is_contrarian=True) rejected at execute_signal. Scanner fetch wrapped in `asyncio.wait_for(timeout=120)`. Watchdog tracks `_last_scan_at` globally with debug logging. Telegram messages prefixed with "ENGINE |".
- **engine/ws_client.py** (~280 lines) — WebSocket client for real-time Polymarket price updates. Connects to `wss://ws-subscriptions-clob.polymarket.com/ws/market`. Handles `price_change`, `last_trade_price`, `book` events (all async, all trigger SL/TP callback). Dynamic subscribe/unsubscribe per market. Token membership checked BEFORE `register_market` to avoid silent subscribe failures. YES/NO token price conversion. Heartbeat (10s), auto-reconnect (5s delay). Batch subscriptions (100 tokens). Callbacks: `on_price_change` (SL/TP), `on_trade` (whale alerts), `on_disconnect`/`on_reconnect` (logged to trade_log).
- **engine/scanner.py** (~280 lines) — Fetches up to 500 markets from Polymarket's Gamma API with pagination. Filters by volume (>$50k), liquidity (>$5k), price bounds (3-97¢), `acceptingOrders=true` (excludes markets in review). Retry: 3 attempts with backoff (2s, 5s, 10s). Extracts: spread, bestAsk, oneWeekPriceChange, oneMonthPriceChange, negRisk, negRiskMarketID, volume1wk, volume1mo, clobTokenIds (YES/NO token IDs for WS). Classifies into themes via keyword matching (crypto, iran, oil, musk, social, election, etc.). `musk` split from `social` for separate calibration/limits.
- **agents/math_engine.py** (~980 lines) — Core signal generation. 13 raw evidence sources, de-duplicated to 8 via max() on 4 correlated pairs, fused via weighted Bayesian log-odds:
  1. Prospect theory (prior) — inverts human probability weighting (γ=0.65), configurable via `USE_PROSPECT` env var
  2. Historical base rates per theme (requires ≥10 bets, base_rate×prospect_factor clamped to [0.05, 0.95])
  3. Volume spike detection (>2.5x average, direction from price momentum cache; skips if no price history to avoid YES bias)
  4. Time decay (near expiry → trust market more, uses neutral 0.5 anchor not prospect), weight 0.5
  5. Price momentum (5-min linear regression, ±5% cap)
  6. Mean reversion / contrarian (>8% move on low volume → bet on reversion via EWMA)
  7. Long-term momentum (week/month price changes from API, threshold 2%/5%)
  8. Volume trend (24h vs weekly average)
  9. NegRisk arbitrage (multi-outcome events, sum ≠ 1.0)
  10. Order book imbalance (WS book events: bid/ask volume ratio, threshold |imbalance| > 0.3, weight 0.8)
  11. Favorite-longshot bias (crowds overprice longshots, underprice favorites)
  12. Certainty gradient (irrationality near 0%/100%)
  13. Overreaction decay (>12% rapid moves tend to revert)
  Correlated pair de-duplication: max(p_momentum, p_long_mom), max(p_volume, p_vol_trend), max(p_flb, p_certainty), max(p_overreact, p_contrarian) before fusion → 8 independent sources. Evidence weights: history=1.0, vol_combined=1.0, time=0.5, mom_combined=1.0, contrarian=1.0, arb=1.0, book=0.8, crowd=0.8. DMA weight floor: 0.5 (prevents killing working sources). Adaptive drift cap: ±12% (0-1 sources), ±18% (2-3), ±25% (4+). Spread penalty on Kelly. bestAsk for YES-side real entry price. Per-market volatility (ATR from 30-min price cache). **Per-theme adaptive thresholds**: EV/KL/edge minimums scaled by `ev_mult` from Bayesian theme calibration (losing themes need higher EV to enter). **Portfolio correlation penalty**: positions in same negRisk group (ρ=1.0) or theme (ρ=0.5) reduce effective independent bets via `effective_n = n / (1 + (n-1) × ρ)`. Stake limited to 5% bankroll per effective bet. Worst-case check: if entire cluster hits SL > 15% bankroll → reduce. **Per-theme Kelly multiplier**: `kelly_mult` from Bayesian shrinkage scales Kelly before stake calculation (winning themes get bigger bets). `compute_stake` guards against negative bankroll (returns 0). **Bimodal sizing**: stakes $5-15 pushed to $4 (toxic range). Rejects: EV < 12%×ev_mult, KL < 0.08×ev_mult, Kelly < 0.03, edge < 10%×ev_mult, market > 30 days out.
- **agents/history_agent.py** (~320 lines) — Self-learning. Base rates & prospect factors per theme from closed markets. Volume patterns (high vs low volume win rates). **Per-theme Bayesian performance calibration**: computes `kelly_mult` and `ev_mult` per theme using empirical Bayes shrinkage (k=20) — themes with few trades shrink toward global mean, themes with many trades converge to their true performance. Outputs: `kelly_mult` (0.3–2.0, scales bet size), `ev_mult` (0.7–2.0, scales entry thresholds). ROI-adjusted: losing themes get double penalty (lower Kelly + higher EV bar). ROI check added: WR>55% but ROI<0 gets no reward (prevents rewarding themes that win often but lose money). Calibration via Brier score on RESOLVED positions only (not TP/SL).
- **ml/calibrator.py** (~90 lines) — Brier score, logit-scale correction via factor only (no double bias subtraction). Window: last 300 positions (RESOLVED outcomes only, not TP/SL). Factor bounds [0.7, 1.3]. Only adjusts if |bias| > 0.05 and Brier < 0.25. `adjust()` applied to every `p_final`.
- **utils/db.py** (~1120 lines) — PostgreSQL schema (13 tables including market_metrics, trade_log, trader_commands, and dma_weights), connection pool, CRUD, analytics queries (by theme/source/side/config_tag/calibration), cumulative PnL, signal outcomes for backtest, DB cleanup with configurable retention. `close_position` has race protection (`WHERE status='open' RETURNING id`). `log_event` method for fire-and-forget trade logging. `build_report` generates comprehensive daily analytics. Trader commands: `fetch_pending_commands` (atomic status='processing'), `complete_command`, `fail_command`, `setup_listen` (LISTEN/NOTIFY). Migrations auto-run on startup for new columns (patterns: trade_n, trade_wr, trade_roi, kelly_mult, ev_mult, blocked). `get_blocked_themes()` and `set_theme_blocked()` for dynamic theme blocking from dashboard.
- **utils/telegram.py** (~33 lines) — Async Telegram notifications with HTML formatting.

### Key Algorithms

- **Bayesian fusion**: Prior (prospect-adjusted price) updated with up to 8 de-duplicated evidence sources in weighted log-odds space. Correlated pairs combined via max() before fusion. Adaptive drift cap: ±12% (0-1 sources), ±18% (2-3), ±25% (4+).
- **Prospect weighting**: Kahneman-Tversky with γ=0.65, binary search to invert. Configurable via `USE_PROSPECT` env var for A/B testing.
- **Price momentum**: Linear regression slope over 30-point cache, normalized by constant (×10, not ×n), capped at ±5%. Minimum shift threshold 1.5% (filters noise).
- **Mean reversion / contrarian**: 180-point long cache (~30 min). Detects >8% moves, blocks >30% moves (news-driven). Volume filter: >2.0x = skip (informed money). Smooth vol confidence scaling: low vol → high conf (capped 0.95), high vol → low conf (floor 0.1). `is_contrarian` threshold 0.5. EWMA reversion target, confidence-weighted shift. **Note**: Contrarian trades (is_contrarian=True) are disabled — signals with is_contrarian are rejected in execute_signal. Mean reversion still works as an evidence source in Bayesian fusion, just not as a separate trading mode.
- **NegRisk arbitrage**: Groups markets by negRiskMarketID, normalizes prices to sum=1.0.
- **Kelly criterion**: 0.15 fraction (conservative), guarded against extreme odds (b < 0.01 or b > 100 → 0), spread penalty (3-10¢ → 1.0-0.3x multiplier, clamped to [0.3, 1.0]), capped at MAX_KELLY_FRAC (0.20) of bankroll. **Uncertainty scaling**: `kelly × (0.5 + 0.5 × n_sources/8)` — 1 evidence source → Kelly ×0.56, all 8 → Kelly ×1.0. Per-theme Bayesian Kelly multiplier (0.3–2.0×) from trade history. Portfolio correlation penalty (see below).
- **Nonlinear theme calibration**: Empirical Bayes shrinkage with k=20 for WR/ROI. Nonlinear penalty curves: WR<40% → steep penalty (kelly_mult 0.3-0.65, ev_mult 1.5-3.0×, effectively blocks worst themes when combined with MAX_EV cap), WR 40-55% → neutral, WR>55% → reward (kelly_mult up to 1.5, ev_mult down to 0.7) **only if ROI>0** (WR>55% but ROI<0 gets no reward). ROI double-penalty on losing themes (ev_mult × (1 + |roi| × 2)). Auto-recovery: recalculated every HISTORY_INTERVAL — as WR improves, thresholds drop automatically. Example: oil 35% WR → ev_mult=1.74 (need EV≥22.6%), crypto 59% → ev_mult=0.82 (need EV≥10.6%).
- **Portfolio correlation**: Positions in same negRisk group (ρ=1.0) or theme (ρ=0.5) are correlated. Effective independent bets = `n / (1 + (n-1) × ρ)`. Three checks: (1) negRisk group: positions sharing negRiskMarketID count as 1 effective bet, max 5% bankroll per effective bet. (2) Theme cluster: same formula with ρ=0.5, e.g. 14 iran positions = 1.9 effective bets. (3) Worst-case: if all theme positions hit SL simultaneously, loss must stay < 15% bankroll. Penalty applied as multiplier on stake (min 0.05×). Only limits new positions, never closes existing.
- **Order book imbalance**: Evidence source from WS book events. `imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)` over top-5 price levels. Threshold |imbalance| > 0.3. Shift: 0.3→±1%, 1.0→±4%. Weight 0.8 in Bayesian fusion. Flipped for NO tokens.
- **Bid-price SL/TP**: Position monitoring uses bid price (realistic exit price) instead of mid-price. YES positions: YES best_bid. NO positions: 1 − YES best_ask (= NO bid). Prevents premature closes from spread inflation and matches real execution price.
- **Signal ranking**: `kelly × (1 - entropy × 0.3)` — penalizes 50/50 markets.
- **Position management**: Per-position TP/SL. Vol SL disabled — fixed SL from config. Default SL: normal 25%, contrarian 25%. TP: normal 15%, contrarian 10%. **TP shield near resolution**: if price >92¢ our side AND market expires <48h → skip TP, wait for $1.00 payout (RESOLVED avg=+$1.31 vs TP avg=+$1.09). Trailing TP: tracks peak PnL, closes on 5% pullback from peak when peak ≥ 50% of TP target. Resolution detection: API `is_closed` flag or extreme price (≥99¢/≤1¢); price-based uses linear payout (not binary) as safety.
- **Pre-execution price recheck**: Fresh `get_market()` call before every entry. Rejects if market is `closed`, `acceptingOrders=false` (in review), or price moved enough to drop EV below MIN_EV. Updates entry price, EV, Kelly, edge with fresh data. Prevents phantom entries on stale scan prices (e.g., scanner sees 78¢ but market already at 96¢ near resolution). Logs all outcomes as `[RECHECK]`.
- **Displacement**: When slots full, new signal (EV > 25%) can close worst position. Profitable positions displaced easily; losing positions only if new EV > 2× old EV. Returns bool, caller aborts if displacement failed (race protection).
- **Drawdown protection**: Tracks peak equity (free cash + position values, each clamped to max(0)). Peak restored from real equity on restart (not just BANKROLL env). Halts all new trades when drawdown ≥ 25%. Continues monitoring and closing existing positions. Resume after recovery + 30 min cooldown.
- **Double-close protection**: `close_position` uses `WHERE status='open' RETURNING id` — concurrent WS + REST close attempts are safe.
- **Claude confirmation** (optional): Sonnet 4.6 with web search tool, max 1 call/min, configurable delay (`CONFIRM_DELAY`, default 600s). Enriched context: spread, volume, volatility, Hurst exponent, days_left, negRisk flag, drawdown %, market price history, theme ROI, confirmation count. Improved JSON parsing (supports ```json fenced blocks). Blends: 0.6 × p_final + 0.4 × p_claude, then re-caps drift to ±15%. Fallback: reject (not confirm). Toggled via `CLAUDE_CONFIRM` and `CLAUDE_WEB_SEARCH` config keys.

### Trade Log (trade_log table)

Append-only audit trail in PostgreSQL. Never cleaned up. 19 event types:
- **STARTUP**, **SHUTDOWN** — lifecycle
- **SCAN** — each cycle with equity, drawdown, signal counts
- **SIGNAL_GENERATED** — all probabilities, EV, Kelly, edge, contrarian, momentum, spread, volatility
- **SIGNAL_REJECTED** (8 reasons) — duplicate_market, theme_limit, slots_full_low_ev, no_displaceable_position, displacement_failed_race, stake_too_small, market_closed_pre_exec, market_in_review, stale_price
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

Config loaded from environment variables at startup, then overridden at runtime by `config_live` DB table. `_seed_config_live(engine_config=CONFIG)` runs at startup to populate `config_live` with current env values (ON CONFLICT DO NOTHING — preserves DB overrides). `_reload_config()` merges DB overrides into the `CONFIG` dict (safe keys only, never credentials). Triggered instantly via `LISTEN config_reload` (same connection as `trader_commands`). 22 engine parameters exposed for live editing:
- **Signals**: `MIN_EV`, `MAX_EV`, `MIN_KL`, `MIN_EDGE`, `USE_PROSPECT`
- **Risk**: `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, `TRAILING_TP`, `TRAILING_PULLBACK`
- **Sizing**: `MIN_KELLY_FRAC`, `MAX_KELLY_FRAC`
- **Capacity**: `MAX_OPEN`, `MAX_PER_THEME`, `MAX_SIGNALS`
- **Timing**: `SCAN_INTERVAL`, `HISTORY_INTERVAL`, `CONFIRM_DELAY`
- **Filters**: `MAX_MARKET_DAYS`, `MIN_VOLUME`
- **Claude**: `CLAUDE_CONFIRM`, `CLAUDE_WEB_SEARCH`
- **General**: `CONFIG_TAG`

Key env var params:
- `SIMULATION=true` (default, no real trades)
- `USE_PROSPECT=true` (prospect theory prior, disable for A/B testing)
- `MIN_EV=0.12`, `MIN_KL=0.08` (signal acceptance thresholds)
- `MIN_EDGE=0.10` (minimum |p_final - p_market|)
- `MAX_EV=0.18` (reject overconfident signals — EV>18% has 46% WR)
- `MAX_KELLY_FRAC=0.20`, `MAX_OPEN=50` (conservative sizing)
- `TAKE_PROFIT_PCT=0.15`, `STOP_LOSS_PCT=0.25`
- `TRAILING_TP=true` (trailing take-profit enabled)
- `MAX_MARKET_DAYS=30` (skip markets closing > 30 days out)
- `MAX_DRAWDOWN=0.25` (halt trading at 25% equity drawdown from peak)
- `CONFIG_TAG=v7` (A/B testing tag, saved to DB with full config snapshot)
- `HISTORY_INTERVAL=14400` (recalibrate every 4 hours)
- `SCAN_INTERVAL=300` (seconds between REST market scans; positions monitored in real-time via WebSocket)

### Database

PostgreSQL required. Schema auto-created on startup by `db.init()`. 15 tables: markets, price_snapshots, news, signals, positions, patterns, calibration, stats, config_history, market_metrics, trade_log, trader_commands, dma_weights, config_live, config_live_history. Migrations run automatically for new columns (positions: tp_pct, sl_pct, config_tag; patterns: trade_n, trade_wr, trade_roi, kelly_mult, ev_mult) and backfill (executed signals from positions table). Signals marked `executed=TRUE` after trade for backtest analytics. `trader_commands` table enables dashboard→engine communication: dashboard INSERTs commands + NOTIFY, engine LISTENs + polls each cycle. `config_live` table enables live config: dashboard writes + `NOTIFY config_reload`, engine LISTENs on same connection as `trader_commands`. `config_live_history` stores change audit trail. Cleanup runs every HISTORY_INTERVAL: snapshots (1d), unexecuted signals (7d), processed news (5d). Positions, markets, and trade_log kept forever. VACUUM after cleanup. DB writes throttled: price updates every 30s per position, market upserts skip unchanged prices.

### Risk Management

- **Drawdown halt**: Equity = free cash + sum(position values). Peak restored from real equity on restart. If drawdown ≥ 25% → halt all new trades, continue monitoring/closing existing positions.
- **Double-close protection**: `close_position` atomic with `WHERE status='open' RETURNING id`. WS + REST can't double-count.
- **Portfolio correlation**: Three-layer protection against correlated risk. (1) NegRisk groups (ρ=1.0): positions sharing negRiskMarketID = 1 effective bet, max 5% bankroll. (2) Theme clusters (ρ=0.5): 14 iran positions = ~2 effective bets, stake limited per effective bet. (3) Worst-case: all theme positions hit SL simultaneously must be < 15% bankroll. Only restricts new entries, never closes existing.
- **Position limits**: Max open (configurable, default 50), max 10 per theme.
- **Dynamic theme blocking**: Blocked themes read from DB (`patterns.blocked` column) instead of a hardcoded set. Dashboard can toggle theme blocking via API. Engine checks `get_blocked_themes()` each cycle. Migration adds `blocked BOOLEAN DEFAULT FALSE` to patterns table.
- **Staged SL with grace period**: First 4 hours: emergency SL only (1.5× default, e.g., 37.5% at SL=25%). After 4h: normal SL. Data: <1h=32% WR, 1-3h=49% WR, 3h+=63% WR — noise kills early positions. **Small stakes (≤$10) have NO stop loss** — ride to resolution (low risk, let it play out). Vol SL disabled — fixed SL from config (data showed SL=0.25-0.30 beats all vol-adjusted SLs: 8%→24% WR, 15%→34%, 20%→56%, 25%→75%+).
- **Dynamic trailing TP**: Tracks peak profit, uses dynamic pullback that widens with peak height. `pullback = base × (1 + peak_ratio × 0.5)`. At 50% TP peak: base×1.25, at 100% TP: base×1.5, at 150%: base×1.75. Lets winners run longer while protecting small gains quickly.
- **Displacement**: Only when EV > 25%; losing positions protected unless new signal is 2× better. Race-safe (returns bool).
- **Conservative Kelly**: 0.15 fraction of full Kelly. Zero stake on negative bankroll. **Bimodal sizing**: $5-15 stake range is toxic — all stakes in this range pushed to $4.
- **Resolution safety**: API-confirmed resolution uses binary payout (with division by zero guard on side_price=0); price-based (≥99¢) uses linear payout to prevent spike false positives. All PnL calculations (unrealized, pnl_pct, payout) guarded against side_price=0.
- **Duplicate position guard**: Partial unique index `positions(market_id) WHERE status='open'` — DB-level prevention. `save_position` catches `UniqueViolationError` gracefully.
- **Escalating loss cooldown**: Per market_id, after SL: 1st→2h, 2nd→8h, 3rd+→24h block. Prevents repeated entries into losing markets (e.g., 13x entries on one market losing -$9.67). `_loss_cooldown` stores expiry timestamps, `_loss_count` tracks SL count per session.
- **Short-term filter**: Blocks "Up or Down", "Higher or Lower", "Green or Red", time-specific patterns (e.g. "5:20AM ET"), and time ranges (e.g. "9AM-10AM") via 5 regex patterns in `math_engine._SHORT_TERM_PATTERNS`.
- **Expired question date filter**: `math_engine._parse_question_date()` extracts dates from question text ("on March 22, 2026?"). Blocks negRisk sub-markets where the specific date already passed, even if the event's `end_date` is still in the future.

### Performance Optimizations

- **WebSocket position monitoring** — sub-second SL/TP reaction on price_change, trade, AND book events. REST scan every 5 min (~30x fewer API calls)
- Dynamic WS subscribe/unsubscribe — only track tokens of open positions, not all 500 markets
- Instant `_ws_positions` refresh after position open — no 5-min gap
- **REST fallback skip** — positions with fresh WS data (<30s) skip REST monitor entirely; only stale/missing/closed markets use REST fallback
- **Batch DB writes** — `upsert_markets_batch` + `save_snapshots_batch` (executemany): ~600 sequential queries → 2 per cycle. `update_clv_batch`: ~150 → 3. `save_market_metrics_batch`: ~100 → 1
- **Cached per-cycle data** — `get_open_positions()` and `get_stats()` called once per cycle, passed as arguments to `execute_signal()` and filtering logic (avoids ~15 duplicate queries)
- **Event loop yield** — `asyncio.sleep(0)` every 50 markets during signal generation prevents WS callbacks from being blocked
- **Stale cache eviction** — `evict_stale_caches()` removes price/volume caches for markets no longer in scan results (prevents unbounded memory growth)
- **Hurst O(n)** — `itertools.accumulate` replaces O(n²) cumulative sum in Hurst exponent calculation
- DB write throttling — price updates every 30s per position, market upserts skip unchanged prices
- Signal cooldown 5 min per market (prevents spam)
- Startup: reuses smoke test scanner results instead of fetching twice
- Connection pool: max_size=15 (accommodates WS callbacks + main loop + LISTEN)
- trade_log is fire-and-forget (never blocks main loop on failure)
- Pre-launch smoke tests (85 offline unit tests + online checks) — bot refuses to start if any fail
- Scanner retry with backoff (3 retries: 2s, 5s, 10s) for resilience against transient API failures
