#!/usr/bin/env python3
"""
QUANT ENGINE v3
PostgreSQL + 6 агентов + самообучение
Math-first: Claude только для сильных сигналов (EV > 15%)
"""

import asyncio
import asyncpg
import logging
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from utils.db             import Database
from utils.telegram       import TelegramBot
from engine.scanner       import PolymarketScanner
from engine.ws_client     import PolymarketWS
from agents.math_engine   import MathEngine
from agents.history_agent import HistoryAgent
from ml.calibrator        import Calibrator

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [
        logging.FileHandler("quant.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("main")

CONFIG = {
    # ── Credentials ──
    "ANTHROPIC_KEY":    os.getenv("ANTHROPIC_API_KEY"),
    "TELEGRAM_TOKEN":   os.getenv("TELEGRAM_BOT_TOKEN"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
    "ML_API_URL":       os.getenv("ML_API_URL", ""),

    # ── Core ──
    "BANKROLL":         float(os.getenv("BANKROLL", "1000")),
    "SIMULATION":       os.getenv("SIMULATION", "true").lower() == "true",
    "CONFIG_TAG":       os.getenv("CONFIG_TAG", "v7"),

    # ── Timing ──
    "SCAN_INTERVAL":    int(os.getenv("SCAN_INTERVAL", "300")),
    "HISTORY_INTERVAL": int(os.getenv("HISTORY_INTERVAL", "14400")),

    # ── Signal thresholds ──
    "MIN_EV":           float(os.getenv("MIN_EV", "0.12")),
    "MIN_KL":           float(os.getenv("MIN_KL", "0.08")),
    "MIN_EDGE":         float(os.getenv("MIN_EDGE", "0.10")),
    "MIN_KELLY_FRAC":   float(os.getenv("MIN_KELLY_FRAC", "0.03")),
    "MAX_KELLY_FRAC":   float(os.getenv("MAX_KELLY_FRAC", "0.20")),

    # ── Position management ──
    "TAKE_PROFIT_PCT":  float(os.getenv("TAKE_PROFIT_PCT", "0.15")),
    "STOP_LOSS_PCT":    float(os.getenv("STOP_LOSS_PCT", "0.25")),
    "TRAILING_TP":      os.getenv("TRAILING_TP", "true").lower() == "true",
    "TRAILING_PULLBACK": float(os.getenv("TRAILING_PULLBACK", "0.05")),  # was hardcoded 5%

    # ── Capacity ──
    "MAX_OPEN":         int(os.getenv("MAX_OPEN", "50")),
    "MAX_PER_THEME":    int(os.getenv("MAX_PER_THEME", "10")),  # was hardcoded
    "MAX_SIGNALS":      int(os.getenv("MAX_SIGNALS", "5")),     # was hardcoded confirmed[:5]

    # ── Market filters ──
    "MIN_VOLUME":       float(os.getenv("MIN_VOLUME", "50000")),
    "MAX_MARKET_DAYS":  int(os.getenv("MAX_MARKET_DAYS", "30")),
    "USE_PROSPECT":     os.getenv("USE_PROSPECT", "true").lower() == "true",
    "SKIP_SPORTS":      os.getenv("SKIP_SPORTS", "true").lower() == "true",
    "MAX_EV":           float(os.getenv("MAX_EV", "0.20")),
    "CLAUDE_CONFIRM":   os.getenv("CLAUDE_CONFIRM", "false").lower() == "true",
    "CLAUDE_WEB_SEARCH": os.getenv("CLAUDE_WEB_SEARCH", "false").lower() == "true",
}

_claude_client = None
_shutdown_flag = False

def _get_claude_client(config: dict):
    global _claude_client
    if _claude_client is None:
        from anthropic import AsyncAnthropic
        _claude_client = AsyncAnthropic(api_key=config["ANTHROPIC_KEY"])
    return _claude_client

async def build_signal_context(signal: dict, config: dict, db=None, open_positions: list = None) -> dict:
    """Build rich context for Claude confirmation. Contains everything Claude needs to make a decision."""
    import time as _t
    theme = signal.get("theme", "other")
    context = {
        "health": {
            "bankroll": 0, "wr": 0, "breakeven_wr": 0, "wr_gap": 0,
            "7d_pnl": 0, "trend": "unknown",
        },
        "signal": {
            "question": signal.get("question", ""),
            "side": signal["side"],
            "side_price": signal["side_price"],
            "ev": signal["ev"],
            "kelly": signal["kelly"],
            "edge": signal.get("edge", 0),
            "p_final": signal["p_final"],
            "p_market": signal["p_market"],
            "theme": theme,
            "theme_wr": 0, "theme_trend": "?",
            "n_evidence": signal.get("n_evidence", 0),
            "is_contrarian": signal.get("contrarian", False),
            "sources": {},
            "dma_weights": {},
            "previous_sl_on_market": _loss_count.get(signal["market_id"], 0),
            "similar_theme_open": 0, "theme_stake": 0,
        },
        "portfolio": {
            "open_positions": 0, "drawdown": 0,
            "today_pnl": 0, "largest_theme": "",
        },
    }
    if db:
        try:
            stats = await db.get_stats()
            total = stats["wins"] + stats["losses"]
            wr = round(stats["wins"] / total * 100, 1) if total > 0 else 0
            # Breakeven from actual win/loss sizes (not EV/Kelly)
            async with db.pool.acquire() as _conn:
                _wl = await _conn.fetchrow("""
                    SELECT AVG(CASE WHEN pnl > 0 THEN ABS(pnl/stake_amt) END) as avg_win_pct,
                           AVG(CASE WHEN pnl < 0 THEN ABS(pnl/stake_amt) END) as avg_loss_pct
                    FROM positions WHERE status='closed' AND result IS NOT NULL AND stake_amt > 0
                """)
            avg_win_pct = float(_wl['avg_win_pct'] or 0.13)
            avg_loss_pct = float(_wl['avg_loss_pct'] or 0.17)
            breakeven = round(avg_loss_pct / (avg_win_pct + avg_loss_pct) * 100, 1) if (avg_win_pct + avg_loss_pct) > 0 else 50
            context["health"]["bankroll"] = round(stats["bankroll"])
            context["health"]["wr"] = wr
            context["health"]["breakeven_wr"] = breakeven
            context["health"]["wr_gap"] = round(wr - breakeven, 1)

            # Theme stats from patterns
            patterns = await db.get_patterns()
            tp = patterns.get(theme, {})
            context["signal"]["theme_wr"] = round(float(tp.get("trade_wr", 0)) * 100, 1)
            context["signal"]["ev_mult"] = round(float(tp.get("ev_mult", 1)), 2)
            context["signal"]["kelly_mult"] = round(float(tp.get("kelly_mult", 1)), 2)

            # DMA weights
            dma = await db.get_dma_weights()
            context["signal"]["dma_weights"] = {k: round(v, 2) for k, v in dma.items()} if dma else {}

            # Source probabilities from signal
            for src_key in ["p_history", "p_momentum", "p_long_mom", "p_contrarian", "p_vol_trend", "p_arb", "p_book", "p_flb"]:
                v = signal.get(src_key)
                if v is not None:
                    context["signal"]["sources"][src_key] = round(v, 3)

            # Portfolio context
            open_pos = open_positions or await db.get_open_positions()
            context["portfolio"]["open_positions"] = len(open_pos)
            theme_pos = [p for p in open_pos if p.get("theme") == theme]
            context["signal"]["similar_theme_open"] = len(theme_pos)
            context["signal"]["theme_stake"] = round(sum(p.get("stake_amt", 0) for p in theme_pos))

            # Today's PnL
            async with db.pool.acquire() as conn:
                today_pnl = await conn.fetchval("""
                    SELECT COALESCE(SUM(pnl), 0) FROM positions
                    WHERE status='closed' AND closed_at::date = CURRENT_DATE
                """)
                context["portfolio"]["today_pnl"] = round(float(today_pnl), 2)

                # 7d PnL
                pnl_7d = await conn.fetchval("""
                    SELECT COALESCE(SUM(pnl), 0) FROM positions
                    WHERE status='closed' AND closed_at > NOW() - INTERVAL '7 days'
                """)
                context["health"]["7d_pnl"] = round(float(pnl_7d))
                context["health"]["trend"] = "improving" if float(pnl_7d) > 0 else "declining"

            # Largest theme by stake
            from collections import Counter
            theme_stakes = Counter()
            for p in open_pos:
                theme_stakes[p.get("theme", "?")] += p.get("stake_amt", 0)
            if theme_stakes:
                top_theme = theme_stakes.most_common(1)[0]
                context["portfolio"]["largest_theme"] = f"{top_theme[0]}/${round(top_theme[1])}"
        except Exception as e:
            log.warning(f"[CLAUDE] Context build error: {e}")
    return context

async def claude_confirm(signal: dict, config: dict, db=None, open_positions: list = None) -> dict:
    """Claude confirmation with rich context. Enable via CLAUDE_CONFIRM=true env var."""
    import re, json
    client = _get_claude_client(config)
    use_web = config.get("CLAUDE_WEB_SEARCH", False)
    ctx = await build_signal_context(signal, config, db, open_positions)

    SYSTEM = f"""You are a prediction market analyst for a Polymarket trading bot.
You receive a signal with full system context. Your job: confirm or reject the trade.
{'Search the web for latest news about the market topic before deciding.' if use_web else ''}

CONFIRM unless you have SPECIFIC evidence the market price is correct and our model is wrong.
Consider: theme performance, DMA source reliability, portfolio concentration, loss history on this market.
Be skeptical of high-EV signals (>18%) — our model tends to be overconfident.
Be extra skeptical if the theme has WR < 40% or if we already lost on this market.

Return ONLY:
<json>{{"confirm": true/false, "p_claude": 0.00, "reasoning": "one sentence", "confidence": 0.00}}</json>"""

    # Compact context brief for Claude
    h = ctx["health"]
    s = ctx["signal"]
    p = ctx["portfolio"]
    prompt = (
        f"SYSTEM: WR={h['wr']}% (need {h['breakeven_wr']}%) | 7d: {h['7d_pnl']:+d}$ | Bank: ${h['bankroll']} | Open: {p['open_positions']} pos\n\n"
        f"SIGNAL: {s['side']} \"{s['question'][:100]}\" @ {s['side_price']*100:.0f}c\n"
        f"  EV: +{s['ev']*100:.1f}% | Edge: {s['edge']*100:.1f}% | Kelly: {s['kelly']*100:.1f}%\n"
        f"  p_true: {s['p_final']*100:.0f}% vs market: {s['p_market']*100:.0f}% | Sources: {s['n_evidence']}/8\n"
    )
    if s["sources"]:
        prompt += f"  Sources: {' '.join(f'{k}={v:.2f}' for k, v in s['sources'].items())}\n"
    if s["dma_weights"]:
        top_dma = sorted(s["dma_weights"].items(), key=lambda x: -x[1])[:4]
        prompt += f"  DMA (best→worst): {' '.join(f'{k}={v:.2f}' for k, v in top_dma)}\n"
    prompt += (
        f"\nTHEME [{s['theme']}]: WR={s['theme_wr']}% | ev_mult={s.get('ev_mult', 1):.2f} | kelly_mult={s.get('kelly_mult', 1):.2f}\n"
        f"  Open: {s['similar_theme_open']} pos, ${s['theme_stake']} staked\n"
    )
    if s["previous_sl_on_market"] > 0:
        prompt += f"  WARNING: {s['previous_sl_on_market']} previous SL on this market\n"
    if s["is_contrarian"]:
        prompt += f"  NOTE: Contrarian signal (betting against recent move)\n"
    prompt += f"\nPORTFOLIO: Today {p['today_pnl']:+.0f}$ | Largest: {p['largest_theme']}\n"
    prompt += f"\nConfirm this trade?"

    try:
        call_args = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 200 if not use_web else 400,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        if use_web:
            call_args["tools"] = [{"type": "web_search_20250305", "name": "web_search",
                                    "max_uses": 1}]
        r = await client.messages.create(**call_args)
        txt = "".join(b.text for b in r.content if hasattr(b, "text"))
        match = re.search(r"<json>([\s\S]*?)</json>", txt)
        if match:
            result = json.loads(match.group(1).strip())
            log.info(f"[CLAUDE] {'CONFIRM' if result.get('confirm') else 'REJECT'} {signal['market_id'][:8]} | {result.get('reasoning', '?')}")
            return result
    except Exception as e:
        log.warning(f"[CLAUDE] {e}")
    return {"confirm": True, "p_claude": signal["p_final"], "confidence": 0.3, "reasoning": "api_error_passthrough"}

async def bootstrap_history(db: Database, scanner: PolymarketScanner):
    """Load historical closed markets from Polymarket for training. Skips if already done."""
    existing = await db.get_closed_markets(limit=1)
    if existing:
        log.info(f"[BOOTSTRAP] Historical data already exists, skipping")
        return

    log.info("[BOOTSTRAP] Loading historical closed markets for training...")
    import json as _json
    from engine.scanner import detect_theme, _parse_end_date

    total = 0
    offset = 0
    while offset < 5000:
        try:
            r = await scanner.client.get("https://gamma-api.polymarket.com/markets", params={
                "closed": "true", "order": "volume", "ascending": "false",
                "limit": 100, "offset": offset,
            })
            batch = r.json() or []
            if not batch:
                break
            for m in batch:
                vol = float(m.get("volume") or 0)
                if vol < 10000:
                    continue
                raw_prices = m.get("outcomePrices") or ["0.5", "0.5"]
                if isinstance(raw_prices, str):
                    raw_prices = _json.loads(raw_prices)
                yes_price = float(raw_prices[0])
                no_price = float(raw_prices[1]) if len(raw_prices) > 1 else 1 - yes_price

                # Determine outcome from final price
                if yes_price >= 0.95:
                    outcome = "YES"
                elif yes_price <= 0.05:
                    outcome = "NO"
                else:
                    continue  # not cleanly resolved

                end_date = _parse_end_date(m.get("endDate"))
                question = m.get("question", "")
                await db.upsert_market({
                    "id": str(m["id"]),
                    "slug": m.get("slug", ""),
                    "question": question,
                    "theme": detect_theme(question),
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "volume": vol,
                    "volume_24h": 0,
                    "liquidity": 0,
                    "end_date": end_date,
                    "url": "",
                })
                # Mark as resolved
                async with db.pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE markets SET outcome=$1, is_active=FALSE, resolved_at=NOW() WHERE id=$2",
                        outcome, str(m["id"])
                    )
                total += 1
            offset += 100
        except Exception as e:
            log.warning(f"[BOOTSTRAP] Error at offset {offset}: {e}")
            break

    log.info(f"[BOOTSTRAP] Loaded {total} historical markets for training")

MAX_PER_THEME = CONFIG["MAX_PER_THEME"]
MAX_PER_OTHER = CONFIG["MAX_PER_THEME"]

DISPLACE_MIN_EV = 0.25  # new signal must have EV > 25% to displace

async def _find_displaceable(open_pos: list, signal: dict, scanner: PolymarketScanner) -> dict | None:
    """Find the worst open position that can be displaced by a stronger signal."""
    candidates = []
    for pos in open_pos:
        # Never displace a position in the same market
        if pos["market_id"] == signal["market_id"]:
            continue
        upnl_pct = pos.get("unrealized_pnl", 0) / pos["stake_amt"] if pos["stake_amt"] > 0 else 0
        # Score: lower = worse position = better displacement candidate
        # Positive PnL positions are easy to displace (we lock in profit)
        # Negative PnL need the new signal to be 2x better
        score = pos.get("ev", 0) - upnl_pct * 0.5  # blend remaining EV with current PnL
        candidates.append((score, upnl_pct, pos))

    if not candidates:
        return None

    # Sort: worst position first
    candidates.sort(key=lambda x: x[0])
    score, upnl_pct, worst = candidates[0]

    # If worst position is in profit → displace (lock in gains + open better)
    if upnl_pct > 0:
        log.info(f"[DISPLACE] Candidate (in profit +{upnl_pct*100:.1f}%): {worst['question'][:50]}")
        return worst

    # If worst position is in loss → only displace if new signal EV > 2x worst EV
    if signal["ev"] > 2 * worst.get("ev", 0):
        log.info(f"[DISPLACE] Candidate (in loss {upnl_pct*100:.1f}%, new EV {signal['ev']*100:.0f}% >> old {worst.get('ev',0)*100:.0f}%): {worst['question'][:50]}")
        return worst

    return None

async def _close_for_displacement(pos: dict, db: Database, telegram: TelegramBot) -> bool:
    """Close a position to make room for a better signal. Returns False if already closed."""
    # current_price in DB is already side price (written by update_position_price)
    price = pos.get("current_price", pos["side_price"])
    pnl_pct = (price - pos["side_price"]) / pos["side_price"] if pos["side_price"] > 0 else 0
    payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
    pnl = round(payout - pos["stake_amt"], 2)
    outcome = f"{pos['side']}@{price*100:.0f}¢"

    actually_closed = await db.close_position(pos["id"], outcome, payout, pnl)
    if not actually_closed:
        log.warning(f"[DISPLACE] Position {pos['id']} already closed, skipping")
        return False
    log.info(f"[DISPLACE] Closed {pos['id']} PnL:{pnl:+.2f} to make room")
    await db.log_event("DISPLACEMENT",
        market_id=pos["market_id"], position_id=pos["id"],
        question=pos.get("question"), theme=pos.get("theme"), side=pos.get("side"),
        side_price=pos.get("side_price"), yes_price=price,
        pnl=pnl, pnl_pct=round(pnl_pct, 4), payout=payout, stake_amt=pos["stake_amt"],
        ev=pos.get("ev"), kelly=pos.get("kelly"),
        tp_pct=pos.get("tp_pct"), sl_pct=pos.get("sl_pct"),
        details={"outcome": outcome})
    side_label = "YES" if pos.get("side") == "YES" else "NO"
    await telegram.send(
        f"🔄 <b>DISPLACEMENT</b> {'✅' if pnl > 0 else '❌'}\n\n"
        f"❓ {pos['question'][:120]}\n"
        f"🎲 {side_label} | Вход:{pos['side_price']*100:.1f}¢ → Выход:{price*100:.1f}¢\n"
        f"💰 P&L:<b>{pnl:+.2f}$</b> ({pnl_pct*100:+.1f}%) → слот для лучшего сигнала\n"
        f"🏷 Config: <b>{pos.get('config_tag', CONFIG.get('CONFIG_TAG', '?'))}</b>\n"
        f"🔗 <a href='{pos.get('url','')}'>Polymarket</a>"
    )
    return True

async def execute_signal(signal: dict, db: Database, telegram: TelegramBot, config: dict,
                         scanner: PolymarketScanner = None, math_eng: MathEngine = None):
    open_pos = await db.get_open_positions()
    _rej_base = dict(market_id=signal["market_id"], question=signal["question"],
                      theme=signal.get("theme"), side=signal["side"], ev=signal["ev"],
                      kelly=signal["kelly"], is_simulation=config["SIMULATION"],
                      config_tag=config.get("CONFIG_TAG"))
    if any(p["market_id"] == signal["market_id"] for p in open_pos):
        await db.log_event("SIGNAL_REJECTED", **_rej_base, details={"reason": "duplicate_market"})
        return False
    theme = signal.get("theme", "other")
    theme_count = sum(1 for p in open_pos if p.get("theme") == theme)
    theme_limit = MAX_PER_OTHER if theme == "other" else MAX_PER_THEME
    if theme_count >= theme_limit:
        log.info(f"[EXEC] Skipped: theme '{theme}' already has {theme_count}/{theme_limit} positions")
        await db.log_event("SIGNAL_REJECTED", **_rej_base,
                           details={"reason": "theme_limit", "theme_count": theme_count, "theme_limit": theme_limit})
        return False

    # If full, try to displace worst position
    if len(open_pos) >= config["MAX_OPEN"]:
        if signal["ev"] < DISPLACE_MIN_EV or scanner is None:
            await db.log_event("SIGNAL_REJECTED", **_rej_base, open_positions=len(open_pos),
                               details={"reason": "slots_full_low_ev", "displace_min_ev": DISPLACE_MIN_EV})
            return False
        displace = await _find_displaceable(open_pos, signal, scanner)
        if not displace:
            await db.log_event("SIGNAL_REJECTED", **_rej_base, open_positions=len(open_pos),
                               details={"reason": "no_displaceable_position"})
            return False
        displaced = await _close_for_displacement(displace, db, telegram)
        if not displaced:
            await db.log_event("SIGNAL_REJECTED", **_rej_base, open_positions=len(open_pos),
                               details={"reason": "displacement_failed_race"})
            return False
    stats    = await db.get_stats()
    bankroll = stats.get("bankroll", config["BANKROLL"])
    if math_eng is None:
        math_eng = MathEngine(config, db)

    # Contrarian trades: half Kelly, tighter TP/SL
    kelly = signal["kelly"]
    is_contrarian = signal.get("contrarian", False)
    if is_contrarian:
        kelly *= 0.5
        tp_pct = 0.10
        sl_pct = 0.25
        log.info(f"[EXEC] Contrarian sizing: Kelly {signal['kelly']*100:.1f}%→{kelly*100:.1f}%, TP:10%, SL:25%")
    else:
        tp_pct = config["TAKE_PROFIT_PCT"]
        sl_pct = config["STOP_LOSS_PCT"]

    # Vol SL disabled — data shows fixed SL=25% (74.6% WR, +$0.92) beats all vol-adjusted SLs
    # History: 8% floor → 24% WR, 15% → 34% WR, 20% → 55.6% WR. Fixed 25% is the sweet spot.
    # ATR still logged for monitoring but doesn't override SL
    volatility = signal.get("volatility", 0)
    if volatility > 0:
        vol_sl_info = 4.0 * volatility / signal["side_price"] if signal["side_price"] > 0.10 else 0
        log.info(f"[EXEC] Vol info: ATR={volatility:.5f} → would be SL:{vol_sl_info*100:.1f}% (using fixed:{sl_pct*100:.0f}%)")

    stake = math_eng.compute_stake(bankroll, kelly, signal.get("theme"), open_pos,
                                   signal.get("liquidity", 0),
                                   signal.get("neg_risk_market_id", ""), sl_pct)
    if stake < 1.0:
        await db.log_event("SIGNAL_REJECTED", **_rej_base, bankroll=bankroll,
                           details={"reason": "stake_too_small", "computed_stake": stake})
        return False
    mode = "🧪 SIM" if config["SIMULATION"] else "💰 REAL"
    log.info(f"[EXEC] {mode} {signal['side']} '{signal['question'][:50]}' | ${stake} EV:{signal['ev']*100:.1f}%{' [CONTRARIAN]' if is_contrarian else ''}")
    pos = {
        "id":         f"pos_{signal['market_id'][:8]}_{int(time.time())}",
        "market_id":  signal["market_id"],
        "signal_id":  signal.get("id"),
        "question":   signal["question"],
        "theme":      signal.get("theme","other"),
        "side":       signal["side"],
        "side_price": signal["side_price"],
        "p_final":    signal["p_final"],
        "ev":         signal["ev"],
        "kl":         signal["kl"],
        "kelly":      kelly,
        "stake_amt":  stake,
        "url":        signal.get("url",""),
        "tp_pct":     tp_pct,
        "sl_pct":     sl_pct,
        "config_tag": config.get("CONFIG_TAG", "v1"),
    }
    saved = await db.save_position(pos)
    if not saved:
        log.warning(f"[EXEC] Duplicate position blocked by DB: {signal['market_id'][:8]} {signal['question'][:50]}")
        await db.log_event("SIGNAL_REJECTED", **_rej_base, details={"reason": "duplicate_market_db_guard"})
        return False
    await db.log_event("OPEN",
        market_id=signal["market_id"], position_id=pos["id"], signal_id=signal.get("id"),
        question=signal["question"], theme=signal.get("theme"), side=signal["side"],
        side_price=signal["side_price"], p_market=signal["p_market"], p_final=signal["p_final"],
        ev=signal["ev"], kl=signal["kl"], kelly=kelly, edge=signal.get("edge"),
        stake_amt=stake, tp_pct=tp_pct, sl_pct=sl_pct, bankroll=bankroll,
        is_contrarian=is_contrarian, is_simulation=config["SIMULATION"],
        config_tag=config.get("CONFIG_TAG"),
        details={"volatility": signal.get("volatility"), "spread": signal.get("spread"),
                 "liquidity": signal.get("liquidity"), "original_kelly": signal["kelly"],
                 "source": signal.get("source", "math")})
    side_label = "✅ YES (случится)" if signal["side"] == "YES" else "❌ NO (не случится)"
    await telegram.send(
        f"🎯 <b>СИГНАЛ [{mode}]</b>{' 🔄' if is_contrarian else ''}\n\n"
        f"❓ {signal['question'][:150]}\n"
        f"🎲 Ставка: <b>{side_label}</b> по <b>{signal['side_price']*100:.1f}¢</b>\n\n"
        f"📊 EV:<b>+{signal['ev']*100:.1f}%</b> | Kelly:<b>{kelly*100:.1f}%</b> | Edge:<b>{signal.get('edge',0)*100:.1f}%</b>\n"
        f"🧮 p_true:<b>{signal['p_final']*100:.1f}%</b> vs рынок:<b>{signal['p_market']*100:.1f}%</b>\n"
        f"🎯 TP:{tp_pct*100:.0f}% | SL:{sl_pct*100:.0f}%\n\n"
        f"💵 Ставка: <b>${stake}</b> | Банк: ${bankroll:.2f}\n"
        f"🏷 Config: <b>{config.get('CONFIG_TAG', '?')}</b>\n"
        f"🔗 <a href='{signal.get('url','')}'>Polymarket</a>"
    )
    return True

_last_db_price_update: dict = {}  # pos_id -> timestamp of last DB write
DB_PRICE_UPDATE_INTERVAL = 30  # update DB every 30 seconds, not every tick
_loss_cooldown: dict = {}  # market_id -> expiry timestamp, escalating cooldown
_loss_count: dict = {}  # market_id -> total SL count (persistent within session)

def _position_age_hours(pos: dict) -> float:
    """Return position age in hours. Uses opened_at from DB."""
    from datetime import datetime, timezone
    opened = pos.get("opened_at")
    if not opened:
        return 999.0  # unknown age → treat as old (no grace)
    if isinstance(opened, str):
        opened = datetime.fromisoformat(opened)
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - opened).total_seconds() / 3600

async def process_trader_commands(db: Database, telegram: TelegramBot,
                                  scanner, config: dict, trailing_highs: dict,
                                  ws: PolymarketWS = None):
    """Poll trader_commands table and execute pending commands."""
    try:
        commands = await db.fetch_pending_commands()
    except Exception as e:
        log.error(f"[CMD] Ошибка получения команд: {e}")
        return
    if commands:
        log.info(f"[CMD] Получено {len(commands)} команд(а)")
    for cmd in commands:
        cmd_id = cmd["id"]
        command = cmd["command"]
        pos_id = cmd.get("position_id")
        log.info(f"[CMD] Обработка #{cmd_id}: {command} position={pos_id}")
        try:
            if command == "close_position":
                if not pos_id:
                    log.warning(f"[CMD] #{cmd_id}: отклонено — нет position_id")
                    await db.fail_command(cmd_id, "missing position_id")
                    continue
                # Fetch current position
                open_pos = await db.get_open_positions()
                pos = next((p for p in open_pos if p["id"] == pos_id), None)
                if not pos:
                    log.warning(f"[CMD] #{cmd_id}: позиция {pos_id} не найдена или уже закрыта")
                    await db.fail_command(cmd_id, f"position {pos_id} not found or already closed")
                    continue
                # Calculate current PnL using last known price
                price = pos.get("current_price") or pos["side_price"]
                pnl_pct = (price - pos["side_price"]) / pos["side_price"] if pos["side_price"] > 0 else 0
                payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
                pnl = round(payout - pos["stake_amt"], 2)
                log.info(f"[CMD] #{cmd_id}: закрываю {pos['side']} '{pos['question'][:50]}' | вход:{pos['side_price']*100:.1f}¢ сейчас:{price*100:.1f}¢ PnL:{pnl:+.2f}")
                closed = await db.close_position(pos_id, "MANUAL_CLOSE", payout, pnl)
                if not closed:
                    log.warning(f"[CMD] #{cmd_id}: позиция уже закрыта (race)")
                    await db.fail_command(cmd_id, "position already closed (race)")
                    continue
                # Cleanup WS subscription
                if ws:
                    try:
                        await ws.unsubscribe_market(pos["market_id"])
                        log.info(f"[CMD] WS отписка {pos['market_id'][:8]}")
                    except Exception:
                        pass
                trailing_highs.pop(pos_id, None)
                _last_db_price_update.pop(pos_id, None)
                result_str = "WIN" if pnl > 0 else "LOSS"
                stats = await db.get_stats()
                total = stats["wins"] + stats["losses"]
                wr = round(stats["wins"] / total * 100) if total > 0 else 0
                await db.log_event("CLOSE_MANUAL",
                    market_id=pos["market_id"], position_id=pos_id,
                    question=pos.get("question"), theme=pos.get("theme"), side=pos.get("side"),
                    side_price=pos.get("side_price"), yes_price=price,
                    p_final=pos.get("p_final"), ev=pos.get("ev"), kelly=pos.get("kelly"),
                    stake_amt=pos["stake_amt"], pnl=pnl, pnl_pct=round(pnl_pct, 4), payout=payout,
                    tp_pct=pos.get("tp_pct"), sl_pct=pos.get("sl_pct"),
                    bankroll=stats["bankroll"],
                    is_simulation=config["SIMULATION"], config_tag=config.get("CONFIG_TAG"),
                    details={"close_reason": "manual_dashboard", "win_rate": wr, "total_closed": total})
                side_label = "✅ YES" if pos["side"] == "YES" else "❌ NO"
                # Position lifetime
                _lifetime = ""
                if pos.get("opened_at"):
                    try:
                        from datetime import datetime as _dt, timezone as _tz
                        _opened = pos["opened_at"] if hasattr(pos["opened_at"], 'timestamp') else _dt.fromisoformat(str(pos["opened_at"]))
                        _hours = (_dt.now(_tz.utc) - _opened.replace(tzinfo=_tz.utc) if _opened.tzinfo is None else _dt.now(_tz.utc) - _opened).total_seconds() / 3600
                        _lifetime = f"\n⏱ {_hours*60:.0f}мин" if _hours < 1 else f"\n⏱ {_hours:.1f}ч"
                    except Exception:
                        pass
                await telegram.send(
                    f"🔧 <b>Ручное закрытие</b> {'✅' if pnl >= 0 else '❌'}\n\n"
                    f"❓ {pos['question'][:120]}\n"
                    f"🎲 Ставка: <b>{side_label}</b>\n\n"
                    f"📊 Вход: {pos['side_price']*100:.1f}¢ → Выход: <b>{price*100:.1f}¢</b>\n"
                    f"📈 Движение: <b>{pnl_pct*100:+.1f}%</b>\n"
                    f"💰 P&L: <b>{pnl:+.2f}$</b> (ставка ${pos['stake_amt']:.2f})\n"
                    f"📊 WR:{wr}% ({stats['wins']}W/{stats['losses']}L) | Банк:${stats['bankroll']:.2f}"
                    f"{_lifetime}\n"
                    f"🏷 Config: <b>{pos.get('config_tag', config.get('CONFIG_TAG', '?'))}</b>\n"
                    f"🔗 <a href='{pos.get('url','')}'>Polymarket</a>")
                await db.complete_command(cmd_id, {"pnl": pnl, "payout": payout, "result": result_str})
                log.info(f"[CMD] ✅ #{cmd_id}: позиция {pos_id} закрыта | {result_str} PnL:{pnl:+.2f}$")
            else:
                log.warning(f"[CMD] #{cmd_id}: неизвестная команда '{command}'")
                await db.fail_command(cmd_id, f"unknown command: {command}")
        except Exception as e:
            log.error(f"[CMD] #{cmd_id}: ошибка — {e}", exc_info=True)
            try:
                await db.fail_command(cmd_id, str(e))
            except Exception:
                pass

async def _check_position(pos: dict, price: float, is_closed: bool, yes_price: float,
                          config: dict, trailing_highs: dict,
                          db: Database, telegram: TelegramBot, ws: PolymarketWS = None) -> bool:
    """Check a single position for TP/SL/resolution. Returns True if position was closed."""
    import time as _time
    upnl = (price / pos["side_price"] - 1) * pos["stake_amt"]
    # Throttle DB writes — update price every 30s, not every WS tick
    now = _time.time()
    last_update = _last_db_price_update.get(pos["id"], 0)
    if now - last_update >= DB_PRICE_UPDATE_INTERVAL:
        await db.update_position_price(pos["id"], price, upnl)
        _last_db_price_update[pos["id"]] = now

    pnl_pct = (price - pos["side_price"]) / pos["side_price"]
    close_reason = None
    outcome = None

    tp_pct = pos.get("tp_pct") or config["TAKE_PROFIT_PCT"]
    sl_pct = pos.get("sl_pct") or config["STOP_LOSS_PCT"]

    # 1. Market resolved — only trust API is_closed flag, or extreme prices (99/1)
    # Using 95/5 caused false resolutions on price spikes with inflated binary payout
    is_resolved = is_closed or yes_price >= 0.99 or yes_price <= 0.01
    if is_resolved:
        outcome = "YES" if yes_price >= 0.50 else "NO"
        won = outcome == pos["side"]
        if is_closed:
            # API confirmed resolution — use binary payout
            payout = pos["stake_amt"] * (1 / pos["side_price"]) if won else 0.0
        else:
            # Price-based detection (99/1) — use linear payout as safety
            payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
        pnl = round(payout - pos["stake_amt"], 2)
        close_reason = "RESOLVED"

    # 2. Take profit
    elif pnl_pct >= tp_pct:
        payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
        pnl = round(payout - pos["stake_amt"], 2)
        close_reason = "TAKE_PROFIT"

    # 2b. Trailing take profit with dynamic pullback
    # The higher the peak profit, the wider the allowed pullback (let winners run)
    # Base pullback at 50% TP, scales up to 2× base at 100%+ TP
    elif config.get("TRAILING_TP") and pnl_pct >= tp_pct * 0.5:
        prev_high = trailing_highs.get(pos["id"], 0)
        current_high = max(prev_high, pnl_pct)
        trailing_highs[pos["id"]] = current_high
        pullback = current_high - pnl_pct
        # Dynamic pullback: base × (1 + peak_ratio)
        # At 50% TP peak: base × 1.5, at 100% TP peak: base × 2.0, at 150%: base × 2.5
        base_pullback = config.get("TRAILING_PULLBACK", 0.03)
        peak_ratio = current_high / tp_pct if tp_pct > 0 else 1.0
        dynamic_pullback = base_pullback * (1.0 + peak_ratio * 0.5)
        if pullback >= dynamic_pullback and current_high >= tp_pct * 0.5:
            payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
            pnl = round(payout - pos["stake_amt"], 2)
            close_reason = "TRAILING_TP"
            log.info(f"[MONITOR] Trailing TP: peak {current_high*100:.1f}% → now {pnl_pct*100:.1f}% (pullback {pullback*100:.1f}% >= {dynamic_pullback*100:.1f}%)")

    # 3. Stop loss with staged grace period
    # <1h positions had 29% WR (noise stops). Staged SL: wider first hour, then normal.
    # Emergency SL (1.5×) still protects against real crashes during grace period.
    elif pnl_pct <= -sl_pct:
        pos_age_h = _position_age_hours(pos)
        if pos_age_h >= 1.0:
            # Normal SL after 1h
            payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
            pnl = round(payout - pos["stake_amt"], 2)
            close_reason = "STOP_LOSS"
        else:
            # Grace period (<1h): only emergency SL at 1.5× default
            emergency_sl = sl_pct * 1.5
            if pnl_pct <= -emergency_sl:
                payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
                pnl = round(payout - pos["stake_amt"], 2)
                close_reason = "STOP_LOSS"
                log.info(f"[MONITOR] Emergency SL: age={pos_age_h:.1f}h pnl={pnl_pct*100:.1f}% (emergency:{emergency_sl*100:.0f}%)")
            # else: within grace period + above emergency → hold

    if not close_reason:
        return False

    if close_reason != "RESOLVED":
        outcome = f"{pos['side']}@{price*100:.0f}¢"
    # Write final price + CLV at close to DB
    await db.update_position_price(pos["id"], price, upnl)
    await db.update_clv_close(pos["id"], yes_price)
    actually_closed = await db.close_position(pos["id"], outcome, payout, pnl)
    if not actually_closed:
        return False  # already closed by concurrent WS/REST race
    _peak_pnl = trailing_highs.pop(pos["id"], None)
    _last_db_price_update.pop(pos["id"], None)

    # Unsubscribe from WS after closing
    if ws:
        log.info(f"[WS] Unsubscribing {pos['market_id'][:8]} after {close_reason}: {pos['question'][:60]}")
        await ws.unsubscribe_market(pos["market_id"])

    # Escalating loss cooldown: 1st SL → 2h, 2nd → 8h, 3rd+ → 24h (blocked for session)
    if close_reason == "STOP_LOSS":
        import time as _t
        mid = pos["market_id"]
        _loss_count[mid] = _loss_count.get(mid, 0) + 1
        n = _loss_count[mid]
        if n >= 3:
            cooldown_s = 86400  # 24h — effectively blocked
        elif n == 2:
            cooldown_s = 28800  # 8h
        else:
            cooldown_s = 7200   # 2h
        _loss_cooldown[mid] = _t.time() + cooldown_s  # store expiry time, not start time
        log.info(f"[RISK] Loss cooldown: {mid[:8]} SL #{n} → blocked {cooldown_s//3600}h")

    stats = await db.get_stats()
    total = stats["wins"] + stats["losses"]
    wr = round(stats["wins"] / total * 100) if total > 0 else 0

    _close_event = {"RESOLVED": "CLOSE_RESOLVED", "TAKE_PROFIT": "CLOSE_TP",
                    "STOP_LOSS": "CLOSE_SL", "TRAILING_TP": "CLOSE_TRAILING_TP"}[close_reason]
    await db.log_event(_close_event,
        market_id=pos["market_id"], position_id=pos["id"], signal_id=pos.get("signal_id"),
        question=pos.get("question"), theme=pos.get("theme"), side=pos.get("side"),
        side_price=pos.get("side_price"), yes_price=yes_price,
        p_final=pos.get("p_final"), ev=pos.get("ev"), kelly=pos.get("kelly"),
        stake_amt=pos["stake_amt"], pnl=pnl, pnl_pct=round(pnl_pct, 4), payout=payout,
        tp_pct=tp_pct, sl_pct=sl_pct, bankroll=stats["bankroll"],
        is_simulation=config["SIMULATION"], config_tag=pos.get("config_tag"),
        details={"outcome": outcome, "close_reason": close_reason, "current_price": price,
                 "trailing_high": _peak_pnl,
                 "win_rate": wr, "total_closed": total})

    _reason_label = {
        "RESOLVED":   "🏁 Рынок закрылся",
        "TAKE_PROFIT":"💰 Тейк-профит",
        "STOP_LOSS":  "🛑 Стоп-лосс",
        "TRAILING_TP":"📈 Трейлинг-стоп",
    }[close_reason]
    won = pnl > 0
    log.info(f"[MONITOR] {_reason_label} {'WIN' if won else 'LOSS'} P&L:{pnl:+.2f}")
    side_label = "✅ YES (случится)" if pos["side"] == "YES" else "❌ NO (не случится)"
    # Position lifetime
    _lifetime = ""
    if pos.get("opened_at"):
        try:
            from datetime import datetime as _dt, timezone as _tz
            _opened = pos["opened_at"] if hasattr(pos["opened_at"], 'timestamp') else _dt.fromisoformat(str(pos["opened_at"]))
            _hours = (_dt.now(_tz.utc) - _opened.replace(tzinfo=_tz.utc) if _opened.tzinfo is None else _dt.now(_tz.utc) - _opened).total_seconds() / 3600
            if _hours < 1:
                _lifetime = f"⏱ {_hours*60:.0f}мин"
            else:
                _lifetime = f"⏱ {_hours:.1f}ч"
        except Exception:
            pass
    await telegram.send(
        f"{_reason_label} {'✅' if won else '❌'}\n\n"
        f"❓ {pos['question'][:120]}\n"
        f"🎲 Ставка: <b>{side_label}</b>\n\n"
        f"📊 Вход: {pos['side_price']*100:.1f}¢ → Выход: <b>{price*100:.1f}¢</b>\n"
        f"📈 Движение: <b>{pnl_pct*100:+.1f}%</b>\n"
        f"💰 P&L: <b>{pnl:+.2f}$</b> (ставка ${pos['stake_amt']:.2f})\n"
        f"🎯 EV:{pos.get('ev',0)*100:+.1f}% | TP:{tp_pct*100:.0f}% | SL:{sl_pct*100:.0f}%\n"
        f"📊 WR:{wr}% ({stats['wins']}W/{stats['losses']}L) | Банк:${stats['bankroll']:.2f}\n"
        f"{_lifetime}\n"
        f"🏷 Config: <b>{pos.get('config_tag', CONFIG.get('CONFIG_TAG', '?'))}</b>\n"
        f"🔗 <a href='{pos.get('url','')}'>Polymarket</a>"
    )
    return True


async def monitor_positions(db: Database, telegram: TelegramBot, scanner: PolymarketScanner, config: dict,
                            markets: list = None, trailing_highs: dict = None, ws: PolymarketWS = None):
    """REST fallback position monitor — runs every scan cycle."""
    open_pos = await db.get_open_positions()
    if not open_pos: return
    if markets is None:
        markets = await scanner.fetch()
    if trailing_highs is None:
        trailing_highs = {}
    mmap = {m["id"]: m for m in markets}

    for pos in open_pos:
        m = mmap.get(pos["market_id"])
        is_closed = False

        # Try WS price first (fresher), fall back to REST
        ws_data = ws.get_market_data(pos["market_id"]) if ws else {}
        ws_price = ws_data.get("yes_price", 0) if ws_data else 0
        if ws_price > 0:
            yes_price = ws_price
            no_price = 1 - ws_price
            m = m or {"id": pos["market_id"], "yes_price": yes_price, "no_price": no_price}
            # Use bid price for realistic exit price
            if pos["side"] == "YES" and ws_data.get("best_bid"):
                m["yes_bid"] = ws_data["best_bid"]
            elif pos["side"] == "NO" and ws_data.get("best_ask"):
                m["no_bid"] = 1 - ws_data["best_ask"]
        elif not m:
            raw = await scanner.get_market(pos["market_id"])
            if not raw:
                continue
            raw_prices = raw.get("outcomePrices") or ["0.5", "0.5"]
            if isinstance(raw_prices, str):
                import json as _json
                raw_prices = _json.loads(raw_prices)
            m = {
                "id": raw["id"],
                "yes_price": float(raw_prices[0]),
                "no_price": float(raw_prices[1]) if len(raw_prices) > 1 else 1 - float(raw_prices[0]),
            }
            is_closed = bool(raw.get("closed"))
            # Subscribe to WS for future real-time updates
            if ws and raw.get("yes_token"):
                await ws.subscribe_market(raw["id"], raw.get("yes_token"), raw.get("no_token"),
                                          m["yes_price"], pos.get("question", ""))
                log.info(f"[WS] +subscribe (REST fallback recovery) {raw['id'][:8]} | {pos.get('question', '')[:60]}")

        yes_price = m["yes_price"]
        # Use bid price (realistic exit) when available, otherwise mid
        if pos["side"] == "YES":
            price = m.get("yes_bid", yes_price)
        else:
            price = m.get("no_bid", m.get("no_price", 1 - yes_price))
        await _check_position(pos, price, is_closed, yes_price, config, trailing_highs, db, telegram, ws)

async def main():
    log.info("🚀 QUANT ENGINE v3")
    log.info(f"💼 ${CONFIG['BANKROLL']} | {'SIM 🧪' if CONFIG['SIMULATION'] else 'REAL 💰'}")

    db         = Database()
    await db.init()
    telegram   = TelegramBot(CONFIG["TELEGRAM_TOKEN"], CONFIG["TELEGRAM_CHAT_ID"])
    scanner    = PolymarketScanner(CONFIG)
    calibrator = Calibrator(db)
    math_eng   = MathEngine(CONFIG, db, calibrator)
    history    = HistoryAgent(db, calibrator)
    ws         = PolymarketWS()

    trailing_highs = {}  # pos_id -> max pnl_pct seen (for trailing TP)

    # --- WS real-time position monitoring callback ---
    # Keep a live map of market_id -> position for instant SL/TP checks
    _ws_positions: dict[str, dict] = {}  # market_id -> position dict

    async def _refresh_ws_positions():
        """Refresh the WS position map from DB."""
        nonlocal _ws_positions
        open_pos = await db.get_open_positions()
        _ws_positions = {p["market_id"]: p for p in open_pos}

    async def on_ws_price_change(market_id: str, old_price: float, new_price: float):
        """Instant SL/TP check when WS delivers a price update."""
        pos = _ws_positions.get(market_id)
        if not pos:
            return
        yes_price = new_price
        # Use bid price (realistic exit price) instead of mid for SL/TP checks
        ws_data = ws.get_market_data(market_id)
        if pos["side"] == "YES":
            # Selling YES = YES bid
            price = ws_data.get("best_bid", yes_price)
        else:
            # Selling NO = 1 - YES ask (NO bid = 1 - YES ask)
            price = 1 - ws_data.get("best_ask", 1 - yes_price) if ws_data.get("best_ask") else 1 - yes_price
        pnl_pct = (price - pos["side_price"]) / pos["side_price"] if pos["side_price"] > 0 else 0
        sl_pct = pos.get("sl_pct") or CONFIG["STOP_LOSS_PCT"]
        tp_pct = pos.get("tp_pct") or CONFIG["TAKE_PROFIT_PCT"]
        # Log significant moves toward SL/TP
        if pnl_pct <= -sl_pct * 0.7 or pnl_pct >= tp_pct * 0.7:
            log.info(f"[WS] {market_id[:8]} {pos['side']} pnl:{pnl_pct*100:+.1f}% (SL:{-sl_pct*100:.0f}%/TP:{tp_pct*100:.0f}%) price:{old_price:.4f}→{new_price:.4f}")
        closed = await _check_position(pos, price, False, yes_price, CONFIG, trailing_highs, db, telegram, ws)
        if closed:
            _ws_positions.pop(market_id, None)

    async def on_ws_trade(market_id: str, price: float, size: float, side: str):
        """Whale alert on positions we hold."""
        if market_id in _ws_positions and size >= 500:
            pos = _ws_positions[market_id]
            log.info(f"[WHALE] ${size:.0f} {side} on {pos['question'][:50]}")
            await db.log_event("WHALE",
                market_id=market_id, position_id=pos.get("id"),
                question=pos.get("question"), side=pos.get("side"), yes_price=price,
                is_simulation=CONFIG["SIMULATION"],
                details={"trade_size": size, "trade_side": side})

    async def on_ws_disconnect():
        await db.log_event("WS_DISCONNECT", is_simulation=CONFIG["SIMULATION"],
                           details={"tracked_markets": len(ws.prices)})

    async def on_ws_reconnect():
        await db.log_event("WS_RECONNECT", is_simulation=CONFIG["SIMULATION"],
                           details={"tracked_markets": len(ws.prices), "tokens": len(ws._subscribed_tokens)})

    ws.set_callbacks(on_price_change=on_ws_price_change, on_trade=on_ws_trade,
                     on_disconnect=on_ws_disconnect, on_reconnect=on_ws_reconnect)

    # ── Smoke test: verify critical systems before trading ──
    smoke_errors = []
    try:
        # 1. Hurst exponent: known trending data → H > 0.5
        math_eng._long_price_cache["_test"] = [0.50 + i * 0.005 for i in range(30)]
        h = math_eng._hurst_exponent("_test")
        if h < 0.45:
            smoke_errors.append(f"Hurst: trending data H={h:.2f} (expected >0.5)")
        del math_eng._long_price_cache["_test"]

        # 2. Bayesian fusion: evidence should shift prior
        from agents.math_engine import bayesian_update
        result = bayesian_update(0.50, [(0.80, 1.0)])
        if result < 0.60:
            smoke_errors.append(f"Bayesian: prior=0.50 + evidence=0.80 → {result:.2f} (expected >0.60)")

        # 3. Book imbalance weight = 0.8 (not 0.5)
        import re as _re
        with open("agents/math_engine.py") as _f:
            _src = _f.read()
        # Check that book weight base is 0.8 (line like: p_book, 0.8 * dma...)
        import re as _re
        if not _re.search(r"p_book.*0\.8", _src):
            smoke_errors.append("Book imbalance weight not 0.8")

        # 4. CLV columns exist in DB
        stats = await db.get_stats()
        async with db.pool.acquire() as conn:
            cols = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name='positions'"
            )
            col_names = {r["column_name"] for r in cols}
        for clv_col in ["clv_1h", "clv_4h", "clv_24h", "clv_close"]:
            if clv_col not in col_names:
                smoke_errors.append(f"CLV column '{clv_col}' missing from positions table")

        # 5. Scanner returns markets
        test_markets = await scanner.fetch()
        if len(test_markets) < 10:
            smoke_errors.append(f"Scanner: only {len(test_markets)} markets (expected >100)")

        # 6. DB connection + bankroll
        if stats.get("bankroll", 0) <= 0:
            smoke_errors.append(f"DB bankroll={stats.get('bankroll')} (expected >0)")

        # 7. Patterns loaded
        patterns = await db.get_patterns()
        if not patterns:
            smoke_errors.append("No patterns in DB (history agent not run?)")

    except Exception as e:
        smoke_errors.append(f"Smoke test crashed: {e}")

    if smoke_errors:
        error_msg = "\n".join(f"  - {e}" for e in smoke_errors)
        log.error(f"[SMOKE] FAILED:\n{error_msg}")
        await telegram.send(
            f"🚨 <b>ENGINE SMOKE TEST FAILED</b>\n\n"
            f"<pre>{error_msg}</pre>\n\n"
            f"🏷 Config: <b>{CONFIG['CONFIG_TAG']}</b>\n"
            f"Bot will NOT start trading."
        )
        await db.close()
        return
    else:
        log.info(f"[SMOKE] All checks passed: Hurst, Bayesian, book weight, CLV, scanner ({len(test_markets)} mkts), DB")

    await telegram.send(
        f"🚀 <b>Quant Engine v3</b>\n"
        f"💼 ${CONFIG['BANKROLL']} | {'Симуляция 🧪' if CONFIG['SIMULATION'] else 'Реальный 💰'}\n"
        f"✅ Smoke test passed ({len(test_markets)} markets)\n"
        f"🧠 Hurst + Book 0.8 + CLV + DMA\n"
        f"🏷 Config: <b>{CONFIG['CONFIG_TAG']}</b>\n"
        f"🔌 WebSocket position monitoring enabled"
    )

    await db.save_config_snapshot(CONFIG["CONFIG_TAG"], CONFIG)
    await db._cleanup_arb_positions()
    await bootstrap_history(db, scanner)
    await history.analyze()
    await math_eng.load_patterns()

    # Subscribe WS to all currently open positions
    open_pos = await db.get_open_positions()
    markets_initial = await scanner.fetch()
    mmap_initial = {m["id"]: m for m in markets_initial}
    ws_registered = 0
    ws_failed = 0
    for pos in open_pos:
        m = mmap_initial.get(pos["market_id"])
        if m and (m.get("yes_token") or m.get("no_token")):
            ws.register_market(m["id"], m.get("yes_token"), m.get("no_token"),
                               m["yes_price"], m.get("question", ""))
            log.info(f"[WS] +subscribe {pos['market_id'][:8]} {pos['side']} @ {pos['side_price']*100:.1f}¢ | {pos['question'][:60]}")
            ws_registered += 1
        else:
            raw = await scanner.get_market(pos["market_id"])
            if raw and (raw.get("yes_token") or raw.get("no_token")):
                raw_prices = raw.get("outcomePrices") or ["0.5", "0.5"]
                if isinstance(raw_prices, str):
                    import json as _json
                    raw_prices = _json.loads(raw_prices)
                ws.register_market(raw["id"], raw.get("yes_token"), raw.get("no_token"),
                                   float(raw_prices[0]), raw.get("question", ""))
                log.info(f"[WS] +subscribe {pos['market_id'][:8]} (fetched tokens) {pos['side']} @ {pos['side_price']*100:.1f}¢ | {pos['question'][:60]}")
                ws_registered += 1
            else:
                log.warning(f"[WS] Failed to get tokens for {pos['market_id'][:8]} | {pos['question'][:60]}")
                ws_failed += 1
    await _refresh_ws_positions()
    log.info(f"[WS] Startup: {ws_registered} positions subscribed, {ws_failed} failed, {len(ws.prices)} total markets")
    await db.log_event("STARTUP",
        bankroll=CONFIG["BANKROLL"], open_positions=len(open_pos),
        is_simulation=CONFIG["SIMULATION"], config_tag=CONFIG["CONFIG_TAG"],
        details={"ws_registered": ws_registered, "ws_failed": ws_failed,
                 "config": {k: v for k, v in CONFIG.items()
                            if k not in ("ANTHROPIC_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID")}})

    # Start WS in background
    ws_task = asyncio.create_task(ws.connect())

    # LISTEN for trader_commands NOTIFY — instant reaction to dashboard commands
    async def _listen_commands():
        try:
            listen_conn = await asyncpg.connect(db.url)
            await listen_conn.add_listener("trader_commands",
                lambda conn, pid, channel, payload:
                    asyncio.create_task(process_trader_commands(db, telegram, scanner, CONFIG, trailing_highs, ws)))
            await db.setup_listen(listen_conn)
            log.info("[CMD] LISTEN trader_commands active")
            while not _shutdown_flag:
                await asyncio.sleep(60)
            await listen_conn.close()
        except Exception as e:
            log.warning(f"[CMD] LISTEN setup failed, polling only: {e}")
    cmd_task = asyncio.create_task(_listen_commands())

    last_history = last_metrics_save = 0
    METRICS_SAVE_INTERVAL = 300
    scan_count = 0
    _market_price_cache = {}
    _signal_cooldown = {}
    try:
        saved_metrics = await db.get_all_market_metrics()
        for m in saved_metrics:
            mid = m["market_id"]
            math_eng.restore_market_metrics(mid, m)
            if m.get("last_signal_at"):
                _signal_cooldown[mid] = m["last_signal_at"].timestamp()
        log.info(f"[MAIN] Restored {len(saved_metrics)} market caches, {len(_signal_cooldown)} cooldowns from DB")
    except Exception as e:
        log.warning(f"[MAIN] Could not restore metrics: {e}")

    # Restore loss cooldowns from DB — count SLs per market in last 24h
    try:
        async with db.pool.acquire() as conn:
            sl_rows = await conn.fetch("""
                SELECT market_id, COUNT(*) as sl_count, MAX(created_at) as last_sl
                FROM trade_log
                WHERE event_type = 'CLOSE_SL'
                  AND created_at > NOW() - INTERVAL '24 hours'
                GROUP BY market_id HAVING COUNT(*) >= 1
            """)
            import time as _t
            for r in sl_rows:
                mid = r["market_id"]
                n = r["sl_count"]
                _loss_count[mid] = n
                # Apply cooldown based on count
                if n >= 3:
                    cooldown_s = 86400
                elif n == 2:
                    cooldown_s = 28800
                else:
                    cooldown_s = 7200
                last_sl_ts = r["last_sl"].timestamp() if r["last_sl"] else _t.time()
                expiry = last_sl_ts + cooldown_s
                if expiry > _t.time():
                    _loss_cooldown[mid] = expiry
            if _loss_cooldown:
                log.info(f"[MAIN] Restored {len(_loss_cooldown)} loss cooldowns from DB ({sum(_loss_count.values())} total SLs)")
    except Exception as e:
        log.warning(f"[MAIN] Could not restore loss cooldowns: {e}")
    daily_report_sent = None
    # Compute real equity at startup for accurate drawdown protection
    _startup_stats = await db.get_stats()
    _startup_pos = await db.get_open_positions()
    _startup_pos_value = sum((p.get("stake_amt", 0) + (p.get("unrealized_pnl", 0) or 0)) for p in _startup_pos)
    _startup_equity = _startup_stats["bankroll"] + _startup_pos_value
    peak_equity = max(CONFIG["BANKROLL"], _startup_equity)  # track peak equity for drawdown
    MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.25"))  # 25% from peak → stop
    trading_halted = False
    halt_time = 0  # timestamp when halted
    HALT_COOLDOWN = 1800  # 30 min cooldown before resume
    loop = asyncio.get_event_loop()
    for sig_name in ("SIGTERM", "SIGINT"):
        try:
            import signal as _signal
            loop.add_signal_handler(
                getattr(_signal, sig_name),
                lambda: asyncio.create_task(shutdown(db, telegram, scanner, ws)),
            )
        except (NotImplementedError, AttributeError):
            pass

    while not _shutdown_flag:
        try:
            now = time.time()
            scan_count += 1

            # Drawdown check: equity = free cash + value of all open positions
            stats = await db.get_stats()
            open_pos = await db.get_open_positions()
            positions_value = sum((p.get("stake_amt", 0) + (p.get("unrealized_pnl", 0) or 0)) for p in open_pos)
            equity = stats["bankroll"] + positions_value
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            if drawdown >= MAX_DRAWDOWN and not trading_halted:
                trading_halted = True
                halt_time = now
                log.warning(f"[RISK] DRAWDOWN HALT: equity=${equity:.2f} peak=${peak_equity:.2f} dd={drawdown*100:.1f}% >= {MAX_DRAWDOWN*100:.0f}%")
                await db.log_event("DRAWDOWN_HALT",
                    bankroll=stats["bankroll"], equity=equity, peak_equity=peak_equity,
                    drawdown_pct=round(drawdown, 4), open_positions=len(open_pos),
                    is_simulation=CONFIG["SIMULATION"], config_tag=CONFIG["CONFIG_TAG"])
                await telegram.send(
                    f"🚨 <b>TRADING HALTED — DRAWDOWN</b>\n\n"
                    f"Equity: <b>${equity:.2f}</b> (peak: ${peak_equity:.2f})\n"
                    f"Drawdown: <b>{drawdown*100:.1f}%</b> >= {MAX_DRAWDOWN*100:.0f}% limit\n"
                    f"Open positions: {len(open_pos)}\n\n"
                    f"🏷 Config: <b>{CONFIG['CONFIG_TAG']}</b>\n"
                    f"No new trades until recovery + 30 min cooldown."
                )
            if trading_halted:
                # Recovery: drawdown recovered AND 30 min cooldown passed
                if drawdown < MAX_DRAWDOWN * 0.5 and (now - halt_time) >= HALT_COOLDOWN:
                    trading_halted = False
                    log.info(f"[RISK] TRADING RESUMED: equity=${equity:.2f} dd={drawdown*100:.1f}% recovered")
                    await db.log_event("DRAWDOWN_RESUME",
                        bankroll=stats["bankroll"], equity=equity, peak_equity=peak_equity,
                        drawdown_pct=round(drawdown, 4), open_positions=len(open_pos),
                        is_simulation=CONFIG["SIMULATION"], config_tag=CONFIG["CONFIG_TAG"])
                    await telegram.send(
                        f"✅ <b>TRADING RESUMED</b>\n\n"
                        f"Equity: <b>${equity:.2f}</b> (peak: ${peak_equity:.2f})\n"
                        f"Drawdown: {drawdown*100:.1f}% < {MAX_DRAWDOWN*50:.0f}% recovery threshold\n"
                        f"🏷 Config: <b>{CONFIG['CONFIG_TAG']}</b>"
                    )
                else:
                    # Still halted — process commands + monitor positions only
                    await process_trader_commands(db, telegram, scanner, CONFIG, trailing_highs, ws)
                    await monitor_positions(db, telegram, scanner, CONFIG, await scanner.fetch(), trailing_highs, ws)
                    await asyncio.sleep(CONFIG["SCAN_INTERVAL"])
                    continue

            # Process any pending trader commands (manual close, etc.)
            await process_trader_commands(db, telegram, scanner, CONFIG, trailing_highs, ws)

            markets = await scanner.fetch()
            for m in markets:
                prev = _market_price_cache.get(m["id"])
                if prev and abs(prev - m["yes_price"]) < 0.001:
                    continue
                _market_price_cache[m["id"]] = m["yes_price"]
                await db.upsert_market(m)
                await db.save_snapshot(m["id"], m["yes_price"], m["volume"], m.get("volume_24h", 0))

            math_eng.build_neg_risk_groups(markets)

            # Warmup: first scan fills price caches
            if scan_count <= 1:
                log.info(f"[MAIN] First scan complete, {len(markets)} markets. Signal generation starts next cycle.")
                # REST fallback check + refresh WS positions
                await _refresh_ws_positions()
                await monitor_positions(db, telegram, scanner, CONFIG, markets, trailing_highs, ws)
                await asyncio.sleep(CONFIG["SCAN_INTERVAL"])
                continue

            # Enrich markets with WS order book imbalance
            for m in markets:
                ws_data = ws.get_market_data(m["id"])
                if ws_data and "imbalance" in ws_data:
                    m["book_imbalance"] = ws_data["imbalance"]

            math_signals = []
            for m in markets:
                sig = math_eng.analyze(m)
                if sig:
                    math_signals.append(sig)

            all_signals = {s["market_id"]: s for s in math_signals}

            SIGNAL_COOLDOWN = 300
            _signal_cooldown = {k: v for k, v in _signal_cooldown.items() if now - v < SIGNAL_COOLDOWN}
            all_signals = {k: v for k, v in all_signals.items() if k not in _signal_cooldown}

            # Escalating loss cooldown — expiry time stored in _loss_cooldown
            for k in [k for k, v in _loss_cooldown.items() if v <= now]:
                del _loss_cooldown[k]  # purge expired, mutate in-place (avoid rebinding)
            loss_blocked = {k for k in all_signals if k in _loss_cooldown}
            if loss_blocked:
                log.info(f"[SCAN] Loss cooldown blocked {len(loss_blocked)} signals (counts: {', '.join(f'{k[:8]}={_loss_count.get(k,0)}' for k in loss_blocked)})")
            all_signals = {k: v for k, v in all_signals.items() if k not in _loss_cooldown}

            # Filter out markets where we already have an open position
            open_market_ids = {p["market_id"] for p in await db.get_open_positions()}
            new_signals = {k: v for k, v in all_signals.items() if k not in open_market_ids}

            signals = sorted(new_signals.values(), key=lambda s: s["kelly"] * (1 - s.get("entropy", 0.5) * 0.3), reverse=True)

            if signals:
                log.info(f"[SCAN #{scan_count}] {len(markets)} рынков | {len(all_signals)} сигналов | {len(signals)} новых")
                await db.log_event("SCAN",
                    bankroll=stats["bankroll"], equity=equity, peak_equity=peak_equity,
                    drawdown_pct=round(drawdown, 4), open_positions=len(open_pos),
                    is_simulation=CONFIG["SIMULATION"], config_tag=CONFIG["CONFIG_TAG"],
                    details={"scan_count": scan_count, "markets": len(markets),
                             "signals_total": len(all_signals), "signals_new": len(signals),
                             "trading_halted": trading_halted})

            # ML enrichment: blend p_ml into p_final for top signals
            mmap_for_ml = {m["id"]: m for m in markets}
            for sig in signals[:5]:
                m_data = mmap_for_ml.get(sig["market_id"])
                if m_data:
                    ml_result = await math_eng.ml_predict(m_data)
                    if ml_result:
                        p_ml = ml_result.get("p_yes")
                        sig["p_ml"] = p_ml
                        sig["p_mispriced"] = ml_result.get("p_mispriced", 0)
                        # Blend ML into p_final: 90% math + 10% ML, cap ±5% shift
                        if p_ml is not None:
                            old_final = sig["p_final"]
                            blended = round(old_final * 0.9 + p_ml * 0.1, 4)
                            max_ml_shift = 0.05
                            sig["p_final"] = max(old_final - max_ml_shift, min(old_final + max_ml_shift, blended))
                            # Recalculate EV and Kelly with updated p_final
                            from agents.math_engine import expected_value, kelly_fraction
                            sig["ev"] = expected_value(
                                sig["p_final"] if sig["side"] == "YES" else 1 - sig["p_final"],
                                sig["side_price"])
                            sig["kelly"] = kelly_fraction(
                                sig["p_final"] if sig["side"] == "YES" else 1 - sig["p_final"],
                                sig["side_price"])
                            log.info(f"[ML] {sig['market_id'][:8]} p_ml={p_ml:.2f} p_final:{old_final:.2f}→{sig['p_final']:.2f} EV:{sig['ev']*100:+.1f}%")

            # Claude confirmation filter (if enabled via CLAUDE_CONFIRM=true)
            if CONFIG.get("CLAUDE_CONFIRM") and CONFIG.get("ANTHROPIC_KEY"):
                confirmed = []
                from agents.math_engine import expected_value, kelly_fraction
                for sig in signals[:CONFIG["MAX_SIGNALS"]]:
                    result = await claude_confirm(sig, CONFIG, db, open_pos)
                    if result.get("confirm"):
                        # Blend Claude's probability: 60% math + 40% Claude, cap ±15% drift
                        p_claude = result.get("p_claude", sig["p_final"])
                        if p_claude and abs(p_claude - sig["p_final"]) < 0.15:
                            sig["p_claude"] = p_claude
                            old_final = sig["p_final"]
                            sig["p_final"] = round(old_final * 0.6 + p_claude * 0.4, 4)
                            sig["p_final"] = max(old_final - 0.15, min(old_final + 0.15, sig["p_final"]))

                        # Price freshness check — re-read from WS after Claude delay
                        ws_data = ws.get_market_data(sig["market_id"]) if ws else {}
                        fresh_price = ws_data.get("yes_price", 0) if ws_data else 0
                        if fresh_price > 0 and abs(fresh_price - sig["p_market"]) > 0.01:
                            old_price = sig["side_price"]
                            sig["p_market"] = fresh_price
                            if sig["side"] == "YES":
                                sig["side_price"] = ws_data.get("best_ask", fresh_price) if ws_data else fresh_price
                            else:
                                sig["side_price"] = round(1 - fresh_price, 4)
                            sig["ev"] = expected_value(
                                sig["p_final"] if sig["side"] == "YES" else 1 - sig["p_final"],
                                sig["side_price"])
                            sig["kelly"] = kelly_fraction(
                                sig["p_final"] if sig["side"] == "YES" else 1 - sig["p_final"],
                                sig["side_price"])
                            log.info(f"[CLAUDE] Price refresh: {sig['market_id'][:8]} {old_price*100:.1f}c→{sig['side_price']*100:.1f}c EV:{sig['ev']*100:+.1f}%")
                            # Re-check EV after price move
                            if sig["ev"] < CONFIG["MIN_EV"]:
                                log.info(f"[CLAUDE] Post-refresh EV too low: {sig['ev']*100:.1f}% < {CONFIG['MIN_EV']*100:.0f}%")
                                continue
                        else:
                            sig["ev"] = expected_value(
                                sig["p_final"] if sig["side"] == "YES" else 1 - sig["p_final"],
                                sig["side_price"])
                            sig["kelly"] = kelly_fraction(
                                sig["p_final"] if sig["side"] == "YES" else 1 - sig["p_final"],
                                sig["side_price"])
                        confirmed.append(sig)
                    else:
                        log.info(f"[CLAUDE] Rejected: {sig['market_id'][:8]} {sig['side']} | {result.get('reasoning', '?')}")
                        await db.log_event("SIGNAL_REJECTED",
                            market_id=sig["market_id"], question=sig["question"],
                            theme=sig.get("theme"), side=sig["side"], ev=sig["ev"],
                            kelly=sig["kelly"], is_simulation=CONFIG["SIMULATION"],
                            config_tag=CONFIG.get("CONFIG_TAG"),
                            details={"reason": "claude_rejected", "reasoning": result.get("reasoning"),
                                     "p_claude": result.get("p_claude"), "confidence": result.get("confidence")})
            else:
                confirmed = signals[:CONFIG["MAX_SIGNALS"]]

            mmap = {m["id"]: m for m in markets}
            for sig in confirmed:
                sig_id = f"sig_{sig['market_id'][:8]}_{int(now)}"
                await db.save_signal({
                    "id":          sig_id,
                    "market_id":   sig["market_id"],
                    "question":    sig["question"],
                    "side":        sig["side"],
                    "side_price":  sig["side_price"],
                    "p_market":    sig["p_market"],
                    "p_math":      sig.get("p_prospect"),
                    "p_history":   sig.get("p_history"),
                    "p_claude":    sig.get("p_ml") or sig.get("p_claude"),  # p_ml stored in p_claude column
                    "p_final":     sig["p_final"],
                    "ev":          sig["ev"],
                    "kl":          sig["kl"],
                    "kelly":       sig["kelly"],
                    "confidence":  sig.get("edge", 0),
                    "volume_ratio":sig.get("vol_signal", 1.0),
                    "source":      sig.get("source","math"),
                })
                sig["id"] = sig_id  # attach signal_id so position gets linked
                _signal_cooldown[sig["market_id"]] = now
                await db.mark_signal_cooldown(sig["market_id"])
                await db.log_event("SIGNAL_GENERATED",
                    market_id=sig["market_id"], signal_id=sig_id,
                    question=sig["question"], theme=sig.get("theme"), side=sig["side"],
                    side_price=sig["side_price"], yes_price=sig.get("p_market"),
                    p_market=sig["p_market"], p_final=sig["p_final"],
                    p_prospect=sig.get("p_prospect"), p_history=sig.get("p_history"),
                    p_claude=sig.get("p_claude"), p_ml=sig.get("p_ml"),
                    ev=sig["ev"], kl=sig["kl"], kelly=sig["kelly"],
                    edge=sig.get("edge"), entropy=sig.get("entropy"),
                    is_contrarian=sig.get("contrarian", False),
                    is_simulation=CONFIG["SIMULATION"], config_tag=CONFIG["CONFIG_TAG"],
                    details={"spread": sig.get("spread"), "volatility": sig.get("volatility"),
                             "liquidity": sig.get("liquidity"), "vol_signal": sig.get("vol_signal"),
                             "vol_dir": sig.get("vol_dir"), "source": sig.get("source", "math"),
                             "p_history": sig.get("p_history"),
                             "p_momentum": sig.get("p_momentum"), "p_long_mom": sig.get("p_long_mom"),
                             "p_contrarian": sig.get("p_contrarian"), "p_vol_trend": sig.get("p_vol_trend"),
                             "p_arb": sig.get("p_arb"), "p_book": sig.get("p_book"),
                             "p_flb": sig.get("p_flb"), "p_certainty": sig.get("p_certainty"),
                             "p_overreact": sig.get("p_overreact"),
                             "contrarian_conf": sig.get("contrarian_conf"),
                             "p_mispriced": sig.get("p_mispriced")})
                executed = await execute_signal(sig, db, telegram, CONFIG, scanner, math_eng)
                if executed:
                    await db.mark_signal_executed(sig_id)
                    # Subscribe WS to the new position's market
                    m_data = mmap.get(sig["market_id"])
                    if m_data and (m_data.get("yes_token") or m_data.get("no_token")):
                        await ws.subscribe_market(m_data["id"], m_data.get("yes_token"), m_data.get("no_token"),
                                                  m_data["yes_price"], m_data.get("question", ""))
                        log.info(f"[WS] +subscribe (new position) {sig['market_id'][:8]} {sig['side']} | {sig['question'][:60]}")
                    else:
                        log.warning(f"[WS] No tokens for new position {sig['market_id'][:8]}, REST fallback only")
                    # Immediately add to WS position map for instant monitoring
                    await _refresh_ws_positions()

            # Refresh WS position map and run REST fallback check
            await _refresh_ws_positions()
            await monitor_positions(db, telegram, scanner, CONFIG, markets, trailing_highs, ws)

            # CLV tracking: record market price at 1h, 4h, 24h after entry
            try:
                mmap_clv = {m["id"]: m for m in markets}
                for pos in open_pos:
                    if not pos.get("opened_at"):
                        continue
                    m = mmap_clv.get(pos["market_id"])
                    if not m:
                        continue
                    yes_price = m["yes_price"]
                    age_hours = (datetime.now(timezone.utc) - pos["opened_at"].replace(
                        tzinfo=timezone.utc) if pos["opened_at"].tzinfo is None
                        else datetime.now(timezone.utc) - pos["opened_at"]).total_seconds() / 3600
                    if age_hours >= 1:
                        await db.update_clv(pos["id"], "clv_1h", yes_price)
                    if age_hours >= 4:
                        await db.update_clv(pos["id"], "clv_4h", yes_price)
                    if age_hours >= 24:
                        await db.update_clv(pos["id"], "clv_24h", yes_price)
            except Exception as e:
                log.debug(f"[CLV] Update failed: {e}")

            # Log WS health
            ws_active = ws.active_count()
            log.info(f"[WS] {ws_active}/{len(ws.prices)} markets active | connected={ws.connected}")

            # Persist market metrics every scan
            if now - last_metrics_save >= METRICS_SAVE_INTERVAL:
                last_metrics_save = now
                try:
                    for m in markets[:100]:
                        metrics = math_eng.get_market_metrics(m["id"])
                        await db.save_market_metrics(m["id"], metrics)
                except Exception as e:
                    log.warning(f"[MAIN] Metrics save failed: {e}")

            if now - last_history >= CONFIG["HISTORY_INTERVAL"]:
                last_history = now
                await history.analyze()
                await math_eng.load_patterns()
                await db.cleanup()
                await db.log_event("HISTORY_RECALC",
                    is_simulation=CONFIG["SIMULATION"], config_tag=CONFIG["CONFIG_TAG"],
                    details={"scan_count": scan_count})

            utc = datetime.now(timezone.utc)
            if utc.hour >= 8 and daily_report_sent != utc.date():
                daily_report_sent = utc.date()
                report = await db.build_report()
                await telegram.send(report)
                await db.log_event("DAILY_REPORT",
                    bankroll=stats["bankroll"], equity=equity, peak_equity=peak_equity,
                    drawdown_pct=round(drawdown, 4), open_positions=len(open_pos),
                    is_simulation=CONFIG["SIMULATION"], config_tag=CONFIG["CONFIG_TAG"],
                    details={"wins": stats["wins"], "losses": stats["losses"],
                             "total_pnl": stats["total_pnl"], "total_bets": stats["total_bets"]})

        except Exception as e:
            log.error(f"[MAIN] {e}", exc_info=True)

        await asyncio.sleep(CONFIG["SCAN_INTERVAL"])

async def shutdown(db, telegram, scanner, ws=None):
    global _shutdown_flag
    _shutdown_flag = True
    log.info("🛑 Shutting down...")
    try:
        await db.log_event("SHUTDOWN", is_simulation=CONFIG.get("SIMULATION", True),
                           config_tag=CONFIG.get("CONFIG_TAG"), details={"reason": "signal"})
    except Exception:
        pass
    if ws:
        ws.stop()
    await scanner.close()
    await telegram.close()
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
