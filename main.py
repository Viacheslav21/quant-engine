#!/usr/bin/env python3
"""
QUANT ENGINE v3
PostgreSQL + 6 агентов + самообучение
Math-first: Claude только для сильных сигналов (EV > 15%)
"""

import asyncio
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
from agents.news_monitor  import NewsMonitor
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
    "ANTHROPIC_KEY":    os.getenv("ANTHROPIC_API_KEY"),
    "TELEGRAM_TOKEN":   os.getenv("TELEGRAM_BOT_TOKEN"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
    "BANKROLL":         float(os.getenv("BANKROLL", "1000")),
    "SIMULATION":       os.getenv("SIMULATION", "true").lower() == "true",
    "SCAN_INTERVAL":    int(os.getenv("SCAN_INTERVAL", "300")),
    "NEWS_INTERVAL":    int(os.getenv("NEWS_INTERVAL", "30")),
    "HISTORY_INTERVAL": int(os.getenv("HISTORY_INTERVAL", "14400")),
    "MIN_EV":           float(os.getenv("MIN_EV", "0.12")),
    "MIN_KL":           float(os.getenv("MIN_KL", "0.10")),
    "MIN_KELLY_FRAC":   float(os.getenv("MIN_KELLY_FRAC", "0.01")),
    "MAX_KELLY_FRAC":   float(os.getenv("MAX_KELLY_FRAC", "0.15")),
    "MAX_OPEN":         int(os.getenv("MAX_OPEN", "50")),
    "MIN_VOLUME":       float(os.getenv("MIN_VOLUME", "50000")),
    "CLAUDE_EV_THR":    float(os.getenv("CLAUDE_EV_THR", "0.20")),
    "TAKE_PROFIT_PCT":  float(os.getenv("TAKE_PROFIT_PCT", "0.20")),
    "STOP_LOSS_PCT":    float(os.getenv("STOP_LOSS_PCT", "0.30")),
    "TRAILING_TP":      os.getenv("TRAILING_TP", "true").lower() == "true",
    "MIN_EDGE":         float(os.getenv("MIN_EDGE", "0.08")),
    "MAX_MARKET_DAYS":  int(os.getenv("MAX_MARKET_DAYS", "30")),
    "CONFIG_TAG":       os.getenv("CONFIG_TAG", "v3"),
    "USE_PROSPECT":     os.getenv("USE_PROSPECT", "true").lower() == "true",
    "CLAUDE_WEB_SEARCH": os.getenv("CLAUDE_WEB_SEARCH", "false").lower() == "true",
    "SKIP_SPORTS":      os.getenv("SKIP_SPORTS", "true").lower() == "true",
    "ML_API_URL":       os.getenv("ML_API_URL", ""),  # e.g. http://quant-ml.railway.internal:8080
}

_claude_client = None

def _get_claude_client(config: dict):
    global _claude_client
    if _claude_client is None:
        from anthropic import AsyncAnthropic
        _claude_client = AsyncAnthropic(api_key=config["ANTHROPIC_KEY"])
    return _claude_client

async def claude_confirm(signal: dict, config: dict, db=None) -> dict:
    import re, json
    client = _get_claude_client(config)
    use_web = config.get("CLAUDE_WEB_SEARCH", False)
    SYSTEM = f"""Prediction market analyst. Confirm or reject signal{' — search web for latest news first' if use_web else ' based on your knowledge'}.
If theme has losing record, be more skeptical. Return ONLY:
<json>{{"confirm": true/false, "p_claude": 0.00, "reasoning": "one sentence", "confidence": 0.00}}</json>"""

    # Build compact track record
    track = ""
    if db:
        theme = signal.get("theme", "other")
        s = await db.get_win_loss_stats(theme)
        if s["theme_w"] + s["theme_l"] > 0:
            track = f"\n'{theme}': {s['theme_w']}W/{s['theme_l']}L."
            if s["theme_l"] > s["theme_w"]:
                track += " LOSING THEME."
        if s["total_w"] + s["total_l"] > 0:
            track += f" All: {s['total_w']}W/{s['total_l']}L."

    prompt = (
        f"{signal['question']}\n"
        f"{signal['side']} @ {signal['side_price']*100:.0f}¢ | "
        f"Our estimate: {signal['p_final']*100:.0f}% vs market: {signal['p_market']*100:.0f}%"
        f"{track}\nConfirm?"
    )
    try:
        call_args = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 200 if not use_web else 300,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        if use_web:
            call_args["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
        r = await client.messages.create(**call_args)
        final = r.content
        if use_web and r.stop_reason == "tool_use":
            tu = next(b for b in r.content if b.type == "tool_use")
            r2 = await client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=300, system=SYSTEM,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": r.content},
                    {"role": "user",      "content": [{"type":"tool_result","tool_use_id":tu.id,"content":"Return your answer as JSON in <json></json> tags."}]},
                ],
            )
            final = r2.content
        txt   = "".join(b.text for b in final if hasattr(b,"text"))
        match = re.search(r"<json>([\s\S]*?)</json>", txt)
        if match:
            return json.loads(match.group(1).strip())
    except Exception as e:
        log.warning(f"[CLAUDE] {e}")
    return {"confirm": False, "p_claude": signal["p_final"], "confidence": 0, "reasoning": "api_error"}

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

async def daily_ai_analysis(db: Database, config: dict) -> str:
    """Once-daily Sonnet analysis of trading performance with recommendations."""
    client = _get_claude_client(config)
    data = await db.get_analytics()
    stats = await db.get_stats()
    open_pos = await db.get_open_positions()
    start = float(os.getenv("BANKROLL", "1000"))

    summary = (
        f"=== QUANT ENGINE DAILY STATS ===\n"
        f"Bankroll: ${stats['bankroll']:.2f} (start: ${start:.0f}, ROI: {(stats['bankroll']-start)/start*100:+.1f}%)\n"
        f"P&L: ${stats['total_pnl']:+.2f} | WR: {stats['wins']}W/{stats['losses']}L\n"
        f"Open positions: {len(open_pos)} | Avg EV: {stats['avg_ev']*100:.1f}% | Avg Kelly: {stats['avg_kelly']*100:.1f}%\n\n"
        f"=== WIN RATE BY THEME ===\n"
    )
    for r in data["by_theme"]:
        wr = round(r['wins']/r['total']*100) if r['total'] > 0 else 0
        summary += f"  {r['theme']}: {r['wins']}/{r['total']} ({wr}%) avg_pnl={float(r['avg_pnl']):+.2f}\n"

    summary += f"\n=== WIN RATE BY SOURCE ===\n"
    for r in data["by_source"]:
        wr = round(r['wins']/r['total']*100) if r['total'] > 0 else 0
        summary += f"  {r['source']}: {r['wins']}/{r['total']} ({wr}%) avg_pnl={float(r['avg_pnl']):+.2f}\n"

    summary += f"\n=== WIN RATE BY SIDE ===\n"
    for r in data["by_side"]:
        wr = round(r['wins']/r['total']*100) if r['total'] > 0 else 0
        summary += f"  {r['side']}: {r['wins']}/{r['total']} ({wr}%) avg_pnl={float(r['avg_pnl']):+.2f}\n"

    summary += f"\n=== CALIBRATION ===\n"
    for r in data["calibration"]:
        summary += f"  {r['bucket']}: {r['total']} trades, predicted={float(r['avg_predicted'])*100:.1f}%, actual={float(r['actual_wr'])*100:.1f}%\n"

    summary += (
        f"\n=== EV ACCURACY ===\n"
        f"  Predicted EV: +{data['ev_predicted']*100:.1f}% | Actual return: {data['ev_actual']*100:+.1f}%\n"
        f"  Avg position lifetime: {data['avg_lifetime_hours']:.1f}h\n"
        f"\n=== CLOSE REASONS ===\n"
    )
    for r in data["by_reason"]:
        summary += f"  {r['reason']}: {r['total']} trades, avg_pnl={float(r['avg_pnl']):+.2f}\n"

    summary += (
        f"\n=== CONFIG ===\n"
        f"  MIN_EV={config['MIN_EV']} MIN_KL={config['MIN_KL']} MAX_KELLY_FRAC={config['MAX_KELLY_FRAC']}\n"
        f"  TAKE_PROFIT={config['TAKE_PROFIT_PCT']} STOP_LOSS={config['STOP_LOSS_PCT']}\n"
        f"  MAX_DRIFT=0.15 Kelly_fraction=0.15\n"
    )

    try:
        r = await client.messages.create(
            model="claude-sonnet-4-5", max_tokens=800,
            system="""You are a quantitative trading analyst reviewing a prediction market bot's performance.
Give specific, actionable recommendations. Be direct and concise.
Focus on: what's working, what's not, config changes to suggest (with specific numbers), and risks.
Reply in English, max 500 words. Use plain text (no markdown).""",
            messages=[{"role": "user", "content": summary}],
        )
        return "".join(b.text for b in r.content if hasattr(b, "text"))
    except Exception as e:
        log.warning(f"[ANALYSIS] Sonnet call failed: {e}")
        return ""

MAX_PER_THEME = 5  # no more than 5 positions in the same theme

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

async def _close_for_displacement(pos: dict, db: Database, telegram: TelegramBot):
    """Close a position to make room for a better signal."""
    price = pos.get("current_price", pos["side_price"])
    pnl_pct = (price - pos["side_price"]) / pos["side_price"] if pos["side_price"] > 0 else 0
    payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
    pnl = round(payout - pos["stake_amt"], 2)
    outcome = f"{pos['side']}@{price*100:.0f}¢"

    await db.close_position(pos["id"], outcome, payout, pnl)
    log.info(f"[DISPLACE] Closed {pos['id']} PnL:{pnl:+.2f} to make room")
    await telegram.send(
        f"🔄 <b>DISPLACEMENT</b> {'✅' if pnl > 0 else '❌'}\n\n"
        f"❓ {pos['question'][:120]}\n"
        f"💰 P&L:<b>{pnl:+.2f}$</b> → Freed slot for better signal"
    )

async def execute_signal(signal: dict, db: Database, telegram: TelegramBot, config: dict,
                         scanner: PolymarketScanner = None, math_eng: MathEngine = None):
    open_pos = await db.get_open_positions()
    if any(p["market_id"] == signal["market_id"] for p in open_pos): return False
    theme_count = sum(1 for p in open_pos if p.get("theme") == signal.get("theme"))
    if theme_count >= MAX_PER_THEME:
        log.info(f"[EXEC] Skipped: theme '{signal.get('theme')}' already has {theme_count} positions")
        return False

    # If full, try to displace worst position
    if len(open_pos) >= config["MAX_OPEN"]:
        if signal["ev"] < DISPLACE_MIN_EV or scanner is None:
            return False
        displace = await _find_displaceable(open_pos, signal, scanner)
        if not displace:
            return False
        await _close_for_displacement(displace, db, telegram)
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

    # Volatility-based SL: SL = max(MIN_SL, 2.5 × ATR / entry_price)
    # Low volatility → tight SL, high volatility → wider SL
    volatility = signal.get("volatility", 0)
    if volatility > 0 and signal["side_price"] > 0.10:  # skip vol SL for very cheap positions
        vol_sl = 2.5 * volatility / signal["side_price"]
        vol_sl = max(0.08, min(sl_pct, round(vol_sl, 3)))  # floor 8%, cap at default SL
        log.info(f"[EXEC] Vol SL: ATR={volatility:.5f} → SL:{vol_sl*100:.1f}% (default:{sl_pct*100:.0f}%)")
        sl_pct = vol_sl

    stake = math_eng.compute_stake(bankroll, kelly, signal.get("theme"), open_pos,
                                   signal.get("liquidity", 0))
    if stake < 1.0: return False
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
    await db.save_position(pos)
    src_emoji = {"math":"🔢","news":"📰","claude":"🧠"}.get(signal.get("source","math"),"🎯")
    await telegram.send(
        f"🎯 <b>СИГНАЛ [{mode}]</b> {src_emoji}\n\n"
        f"❓ {signal['question'][:150]}\n\n"
        f"{'✅ YES' if signal['side']=='YES' else '❌ NO'} по <b>{signal['side_price']*100:.1f}¢</b>\n\n"
        f"📊 EV:<b>+{signal['ev']*100:.1f}%</b> | KL:<b>{signal['kl']:.3f}</b> | Kelly:<b>{signal['kelly']*100:.1f}%</b>\n"
        f"p_true:<b>{signal['p_final']*100:.1f}%</b> vs рынок:<b>{signal['p_market']*100:.1f}%</b>\n"
        f"Edge:<b>{signal.get('edge',0)*100:.1f}%</b>\n\n"
        f"💵 Ставка:<b>${stake}</b>\n"
        f"🔗 <a href='{signal.get('url','')}'>Polymarket</a>"
    )
    return True

_last_db_price_update: dict = {}  # pos_id -> timestamp of last DB write
DB_PRICE_UPDATE_INTERVAL = 30  # update DB every 30 seconds, not every tick

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

    # 1. Market resolved
    is_resolved = is_closed or yes_price >= 0.95 or yes_price <= 0.05
    if is_resolved:
        outcome = "YES" if yes_price >= 0.50 else "NO"
        won = outcome == pos["side"]
        payout = pos["stake_amt"] * (1 / pos["side_price"]) if won else 0.0
        pnl = round(payout - pos["stake_amt"], 2)
        close_reason = "RESOLVED"

    # 2. Take profit
    elif pnl_pct >= tp_pct:
        payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
        pnl = round(payout - pos["stake_amt"], 2)
        close_reason = "TAKE_PROFIT"

    # 2b. Trailing take profit
    elif config.get("TRAILING_TP") and pnl_pct >= tp_pct * 0.5:
        prev_high = trailing_highs.get(pos["id"], 0)
        current_high = max(prev_high, pnl_pct)
        trailing_highs[pos["id"]] = current_high
        pullback = current_high - pnl_pct
        if pullback >= 0.05 and current_high >= tp_pct * 0.5:
            payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
            pnl = round(payout - pos["stake_amt"], 2)
            close_reason = "TRAILING_TP"
            log.info(f"[MONITOR] Trailing TP: peak {current_high*100:.1f}% → now {pnl_pct*100:.1f}% (pullback {pullback*100:.1f}%)")

    # 3. Stop loss
    elif pnl_pct <= -sl_pct:
        payout = round(pos["stake_amt"] * (1 + pnl_pct), 2)
        pnl = round(payout - pos["stake_amt"], 2)
        close_reason = "STOP_LOSS"

    if not close_reason:
        return False

    if close_reason != "RESOLVED":
        outcome = f"{pos['side']}@{price*100:.0f}¢"
    # Write final price to DB before closing
    await db.update_position_price(pos["id"], price, upnl)
    await db.close_position(pos["id"], outcome, payout, pnl)
    trailing_highs.pop(pos["id"], None)
    _last_db_price_update.pop(pos["id"], None)

    # Unsubscribe from WS after closing
    if ws:
        log.info(f"[WS] Unsubscribing {pos['market_id'][:8]} after {close_reason}: {pos['question'][:60]}")
        await ws.unsubscribe_market(pos["market_id"])

    stats = await db.get_stats()
    total = stats["wins"] + stats["losses"]
    wr = round(stats["wins"] / total * 100) if total > 0 else 0

    reason_emoji = {"RESOLVED": "🏁", "TAKE_PROFIT": "💰", "STOP_LOSS": "🛑", "TRAILING_TP": "📈"}[close_reason]
    won = pnl > 0
    log.info(f"[MONITOR] {reason_emoji} {close_reason} {'WIN' if won else 'LOSS'} P&L:{pnl:+.2f}")
    await telegram.send(
        f"{reason_emoji} <b>{close_reason}</b> {'✅' if won else '❌'}\n\n"
        f"❓ {pos['question'][:120]}\n\n"
        f"{pos['side']} @ {pos['side_price']*100:.1f}¢ → <b>{price*100:.1f}¢</b>\n"
        f"📈 Движение:<b>{pnl_pct*100:+.1f}%</b>\n"
        f"💰 P&L:<b>{pnl:+.2f}$</b> (ставка ${pos['stake_amt']:.2f})\n"
        f"📊 WR:{wr}% | Банкролл:${stats['bankroll']:.2f}"
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
        ws_price = ws.get_price(pos["market_id"]) if ws else 0
        if ws_price > 0:
            yes_price = ws_price
            no_price = 1 - ws_price
            m = m or {"id": pos["market_id"], "yes_price": yes_price, "no_price": no_price}
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
        price = yes_price if pos["side"] == "YES" else m.get("no_price", 1 - yes_price)
        await _check_position(pos, price, is_closed, yes_price, config, trailing_highs, db, telegram, ws)

async def main():
    log.info("🚀 QUANT ENGINE v3")
    log.info(f"💼 ${CONFIG['BANKROLL']} | {'SIM 🧪' if CONFIG['SIMULATION'] else 'REAL 💰'}")

    db         = Database()
    await db.init()
    telegram   = TelegramBot(CONFIG["TELEGRAM_TOKEN"], CONFIG["TELEGRAM_CHAT_ID"])
    scanner    = PolymarketScanner(CONFIG)
    news_mon   = NewsMonitor(db)
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
        price = yes_price if pos["side"] == "YES" else 1 - yes_price
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

    ws.set_callbacks(on_price_change=on_ws_price_change, on_trade=on_ws_trade)

    await telegram.send(
        f"🚀 <b>Quant Engine v3</b>\n"
        f"💼 ${CONFIG['BANKROLL']} | {'Симуляция 🧪' if CONFIG['SIMULATION'] else 'Реальный 💰'}\n"
        f"🔢 Math-first | Claude только EV>{CONFIG['CLAUDE_EV_THR']*100:.0f}%\n"
        f"📰 News Monitor | 🧠 Self-learning | ⚡ PostgreSQL\n"
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

    # Start WS in background
    ws_task = asyncio.create_task(ws.connect())

    last_news = last_history = last_metrics_save = 0
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
    claude_cache = {}
    CLAUDE_CACHE_TTL = 3600       # cache 1 hour (was 30 min)
    last_claude_call = 0
    CLAUDE_MIN_INTERVAL = 300     # max 1 call per 5 min (was 1/min)
    daily_report_sent = None
    peak_equity = CONFIG["BANKROLL"]  # track peak equity for drawdown
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

    while True:
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
                await telegram.send(
                    f"🚨 <b>TRADING HALTED — DRAWDOWN</b>\n\n"
                    f"Equity: <b>${equity:.2f}</b> (peak: ${peak_equity:.2f})\n"
                    f"Drawdown: <b>{drawdown*100:.1f}%</b> >= {MAX_DRAWDOWN*100:.0f}% limit\n"
                    f"Open positions: {len(open_pos)}\n\n"
                    f"No new trades until recovery + 30 min cooldown."
                )
            if trading_halted:
                # Recovery: drawdown recovered AND 30 min cooldown passed
                if drawdown < MAX_DRAWDOWN * 0.5 and (now - halt_time) >= HALT_COOLDOWN:
                    trading_halted = False
                    log.info(f"[RISK] TRADING RESUMED: equity=${equity:.2f} dd={drawdown*100:.1f}% recovered")
                    await telegram.send(
                        f"✅ <b>TRADING RESUMED</b>\n\n"
                        f"Equity: <b>${equity:.2f}</b> (peak: ${peak_equity:.2f})\n"
                        f"Drawdown: {drawdown*100:.1f}% < {MAX_DRAWDOWN*50:.0f}% recovery threshold"
                    )
                else:
                    # Still halted — monitor positions only
                    await monitor_positions(db, telegram, scanner, CONFIG, await scanner.fetch(), trailing_highs, ws)
                    await asyncio.sleep(CONFIG["SCAN_INTERVAL"])
                    continue

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

            news_signals = []
            if now - last_news >= CONFIG["NEWS_INTERVAL"]:
                last_news = now
                new_news = await news_mon.scan()
                for item in new_news:
                    relevant = await news_mon.find_relevant_markets(item, markets)
                    for m in relevant:
                        sig = math_eng.analyze(m)
                        if sig:
                            sig["source"] = "news"
                            sig["news_trigger"] = item["title"][:200]
                            news_signals.append(sig)

            news_market_ids = {s["market_id"] for s in news_signals}
            math_signals = []
            for m in markets:
                if m["id"] in news_market_ids:
                    continue
                sig = math_eng.analyze(m)
                if sig:
                    math_signals.append(sig)

            all_signals = {s["market_id"]: s for s in math_signals}
            for s in news_signals:
                all_signals[s["market_id"]] = s

            SIGNAL_COOLDOWN = 300
            _signal_cooldown = {k: v for k, v in _signal_cooldown.items() if now - v < SIGNAL_COOLDOWN}
            all_signals = {k: v for k, v in all_signals.items() if k not in _signal_cooldown}

            signals = sorted(all_signals.values(), key=lambda s: s["kelly"] * (1 - s.get("entropy", 0.5) * 0.3), reverse=True)

            if signals:
                log.info(f"[SCAN #{scan_count}] {len(markets)} рынков | {len(signals)} сигналов")

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
                        # Blend ML into p_final: 80% math + 20% ML
                        if p_ml is not None:
                            old_final = sig["p_final"]
                            sig["p_final"] = round(old_final * 0.8 + p_ml * 0.2, 4)
                            # Recalculate EV and Kelly with updated p_final
                            from agents.math_engine import expected_value, kelly_fraction
                            sig["ev"] = expected_value(
                                sig["p_final"] if sig["side"] == "YES" else 1 - sig["p_final"],
                                sig["side_price"])
                            sig["kelly"] = kelly_fraction(
                                sig["p_final"] if sig["side"] == "YES" else 1 - sig["p_final"],
                                sig["side_price"])
                            log.info(f"[ML] {sig['market_id'][:8]} p_ml={p_ml:.2f} p_final:{old_final:.2f}→{sig['p_final']:.2f} EV:{sig['ev']*100:+.1f}%")

            claude_cache = {k: v for k, v in claude_cache.items() if now - v[0] < CLAUDE_CACHE_TTL}

            confirmed = []
            CLAUDE_SKIP_THR = 0.20  # EV > 20% = strong enough, skip Claude
            can_call_claude = (now - last_claude_call) >= CLAUDE_MIN_INTERVAL
            for sig in signals[:3]:
                if sig["ev"] >= CLAUDE_SKIP_THR:
                    # Strong signal — pass without Claude
                    confirmed.append(sig)
                    log.info(f"[MATH] Auto-confirmed EV:{sig['ev']*100:.0f}% >= {CLAUDE_SKIP_THR*100:.0f}%: {sig['question'][:50]}")
                    continue
                if sig["ev"] >= CONFIG["CLAUDE_EV_THR"]:
                    cached = claude_cache.get(sig["market_id"])
                    if cached:
                        cache_time, result, cache_price = cached[0], cached[1], cached[2] if len(cached) > 2 else sig["p_market"]
                        # Invalidate if price moved >5% since cache
                        if abs(sig["p_market"] - cache_price) > 0.05:
                            cached = None
                            log.info(f"[CLAUDE] Cache invalidated for {sig['market_id'][:8]}: price moved {cache_price:.2f}→{sig['p_market']:.2f}")
                    if cached:
                        result = cached[1]
                        log.debug(f"[CLAUDE] Cache hit for {sig['market_id'][:8]}")
                    elif can_call_claude:
                        result = await claude_confirm(sig, CONFIG, db)
                        claude_cache[sig["market_id"]] = (now, result, sig["p_market"])
                        last_claude_call = now
                        can_call_claude = False
                    else:
                        confirmed.append(sig)
                        continue
                    if result.get("confirm"):
                        sig["p_claude"] = result.get("p_claude", sig["p_final"])
                        blended = sig["p_final"] * 0.6 + sig["p_claude"] * 0.4
                        max_drift = 0.15
                        sig["p_final"] = max(sig["p_market"] - max_drift, min(sig["p_market"] + max_drift, blended))
                        sig["source"] = "claude"
                        confirmed.append(sig)
                        log.info(f"[CLAUDE] ✅ {sig['question'][:50]}")
                    else:
                        log.info(f"[CLAUDE] ❌ {sig['question'][:50]}")
                else:
                    confirmed.append(sig)

            mmap = {m["id"]: m for m in markets}
            for sig in confirmed[:3]:
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
                    "p_claude":    sig.get("p_claude"),
                    "p_final":     sig["p_final"],
                    "ev":          sig["ev"],
                    "kl":          sig["kl"],
                    "kelly":       sig["kelly"],
                    "confidence":  sig.get("edge", 0),
                    "volume_ratio":sig.get("vol_signal", 1.0),
                    "news_trigger":sig.get("news_trigger"),
                    "source":      sig.get("source","math"),
                })
                _signal_cooldown[sig["market_id"]] = now
                await db.mark_signal_cooldown(sig["market_id"])
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

            # Refresh WS position map and run REST fallback check
            await _refresh_ws_positions()
            await monitor_positions(db, telegram, scanner, CONFIG, markets, trailing_highs, ws)

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

            utc = datetime.now(timezone.utc)
            if utc.hour >= 8 and daily_report_sent != utc.date():
                daily_report_sent = utc.date()
                await telegram.send(await db.build_report())

        except Exception as e:
            log.error(f"[MAIN] {e}", exc_info=True)

        await asyncio.sleep(CONFIG["SCAN_INTERVAL"])

async def shutdown(db, telegram, scanner, ws=None):
    log.info("🛑 Shutting down...")
    if ws:
        ws.stop()
    await scanner.close()
    await telegram.close()
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
