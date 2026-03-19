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
from agents.news_monitor  import NewsMonitor
from agents.math_engine   import MathEngine
from agents.history_agent import HistoryAgent
from ml.calibrator        import Calibrator
from dashboard.app        import start_dashboard

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
    "SCAN_INTERVAL":    int(os.getenv("SCAN_INTERVAL", "10")),
    "NEWS_INTERVAL":    int(os.getenv("NEWS_INTERVAL", "30")),
    "HISTORY_INTERVAL": int(os.getenv("HISTORY_INTERVAL", "14400")),
    "MIN_EV":           float(os.getenv("MIN_EV", "0.08")),
    "MIN_KL":           float(os.getenv("MIN_KL", "0.08")),
    "MIN_KELLY_FRAC":   float(os.getenv("MIN_KELLY_FRAC", "0.01")),
    "MAX_KELLY_FRAC":   float(os.getenv("MAX_KELLY_FRAC", "0.15")),
    "MAX_OPEN":         int(os.getenv("MAX_OPEN", "5")),
    "MIN_VOLUME":       float(os.getenv("MIN_VOLUME", "50000")),
    "CLAUDE_EV_THR":    float(os.getenv("CLAUDE_EV_THR", "0.15")),
}

async def claude_confirm(signal: dict, config: dict) -> dict:
    import re, json
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=config["ANTHROPIC_KEY"])
    SYSTEM = """You are a prediction market analyst.
Given a market signal found by mathematical analysis, confirm or reject it.
Return ONLY JSON in <json></json> tags:
<json>{"confirm": true/false, "p_claude": 0.00, "reasoning": "one sentence", "confidence": 0.00}</json>"""
    try:
        r = await client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=300, system=SYSTEM,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content":
                f"Market: {signal['question']}\n"
                f"Side: {signal['side']} @ {signal['side_price']*100:.1f}¢\n"
                f"Math: p_true={signal['p_final']*100:.1f}% vs market {signal['p_market']*100:.1f}%\n"
                f"EV:+{signal['ev']*100:.1f}% KL:{signal['kl']:.3f}\nConfirm?"
            }],
        )
        final = r.content
        if r.stop_reason == "tool_use":
            tu = next(b for b in r.content if b.type == "tool_use")
            r2 = await client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=300, system=SYSTEM,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[
                    {"role": "user",      "content":
                        f"Market: {signal['question']}\n"
                        f"Side: {signal['side']} @ {signal['side_price']*100:.1f}¢\n"
                        f"Math: p_true={signal['p_final']*100:.1f}% vs market {signal['p_market']*100:.1f}%\n"
                        f"EV:+{signal['ev']*100:.1f}% KL:{signal['kl']:.3f}\nConfirm?"
                    },
                    {"role": "assistant", "content": r.content},
                    {"role": "user",      "content": [{"type":"tool_result","tool_use_id":tu.id,"content":"Done. Now return your answer as JSON in <json></json> tags."}]},
                ],
            )
            final = r2.content
        txt   = "".join(b.text for b in final if hasattr(b,"text"))
        match = re.search(r"<json>([\s\S]*?)</json>", txt)
        if match:
            return json.loads(match.group(1).strip())
    except Exception as e:
        log.warning(f"[CLAUDE] {e}")
    return {"confirm": True, "p_claude": signal["p_final"], "confidence": 0.5, "reasoning": "fallback"}

async def execute_signal(signal: dict, db: Database, telegram: TelegramBot, config: dict):
    open_pos = await db.get_open_positions()
    if len(open_pos) >= config["MAX_OPEN"]: return False
    if any(p["market_id"] == signal["market_id"] for p in open_pos): return False
    stats    = await db.get_stats()
    bankroll = stats.get("bankroll", config["BANKROLL"])
    math_eng = MathEngine(config, db)
    stake    = math_eng.compute_stake(bankroll, signal["kelly"])
    if stake < 1.0: return False
    mode = "🧪 SIM" if config["SIMULATION"] else "💰 REAL"
    log.info(f"[EXEC] {mode} {signal['side']} '{signal['question'][:50]}' | ${stake} EV:{signal['ev']*100:.1f}%")
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
        "kelly":      signal["kelly"],
        "stake_amt":  stake,
        "url":        signal.get("url",""),
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

async def monitor_positions(db: Database, telegram: TelegramBot, scanner: PolymarketScanner):
    open_pos = await db.get_open_positions()
    if not open_pos: return
    markets  = await scanner.fetch()
    mmap     = {m["id"]: m for m in markets}
    for pos in open_pos:
        m = mmap.get(pos["market_id"])
        if not m: continue
        price = m["yes_price"] if pos["side"] == "YES" else m.get("no_price", 1-m["yes_price"])
        upnl  = (price / pos["side_price"] - 1) * pos["stake_amt"]
        await db.update_position_price(pos["id"], price, upnl)
        is_closed = m.get("yes_price", 0.5) >= 0.98 or m.get("yes_price", 0.5) <= 0.02
        if is_closed:
            outcome = "YES" if m["yes_price"] >= 0.98 else "NO"
            won     = outcome == pos["side"]
            payout  = pos["stake_amt"] * (1 / pos["side_price"]) if won else 0.0
            pnl     = round(payout - pos["stake_amt"], 2)
            await db.close_position(pos["id"], outcome, payout, pnl)
            stats = await db.get_stats()
            total = stats["wins"] + stats["losses"]
            wr    = round(stats["wins"]/total*100) if total > 0 else 0
            log.info(f"[MONITOR] {'WIN ✅' if won else 'LOSS ❌'} P&L:{pnl:+.2f}")
            await telegram.send(
                f"{'✅ WIN' if won else '❌ LOSS'}\n\n"
                f"❓ {pos['question'][:120]}\n\n"
                f"{pos['side']} @ {pos['side_price']*100:.1f}¢ → <b>{outcome}</b>\n"
                f"💰 P&L:<b>{pnl:+.2f}$</b>\n"
                f"📊 WR:{wr}% | Банкролл:${stats['bankroll']:.2f}"
            )

async def main():
    log.info("🚀 QUANT ENGINE v3")
    log.info(f"💼 ${CONFIG['BANKROLL']} | {'SIM 🧪' if CONFIG['SIMULATION'] else 'REAL 💰'}")

    db         = Database()
    await db.init()
    telegram   = TelegramBot(CONFIG["TELEGRAM_TOKEN"], CONFIG["TELEGRAM_CHAT_ID"])
    scanner    = PolymarketScanner(CONFIG)
    news_mon   = NewsMonitor(db)
    calibrator = Calibrator(db)
    math_eng   = MathEngine(CONFIG, db)
    history    = HistoryAgent(db, calibrator)

    asyncio.create_task(start_dashboard(db, CONFIG))

    await telegram.send(
        f"🚀 <b>Quant Engine v3</b>\n"
        f"💼 ${CONFIG['BANKROLL']} | {'Симуляция 🧪' if CONFIG['SIMULATION'] else 'Реальный 💰'}\n"
        f"🔢 Math-first | Claude только EV>{CONFIG['CLAUDE_EV_THR']*100:.0f}%\n"
        f"📰 News Monitor | 🧠 Self-learning | ⚡ PostgreSQL"
    )

    await history.analyze()
    await math_eng.load_patterns()

    last_news = last_history = 0
    scan_count = 0

    loop = asyncio.get_event_loop()
    for sig_name in ("SIGTERM", "SIGINT"):
        try:
            import signal as _signal
            loop.add_signal_handler(
                getattr(_signal, sig_name),
                lambda: asyncio.create_task(shutdown(db, telegram, scanner)),
            )
        except (NotImplementedError, AttributeError):
            pass

    while True:
        try:
            now = time.time()
            scan_count += 1

            markets = await scanner.fetch()
            for m in markets:
                await db.upsert_market(m)
                await db.save_snapshot(m["id"], m["yes_price"], m["volume"], m.get("volume_24h", 0))

            news_signals = []
            if now - last_news >= CONFIG["NEWS_INTERVAL"]:
                last_news = now
                new_news  = await news_mon.scan()
                for item in new_news:
                    relevant = await news_mon.find_relevant_markets(item, markets)
                    for m in relevant:
                        sig = math_eng.analyze(m)
                        if sig:
                            sig["source"]       = "news"
                            sig["news_trigger"] = item["title"][:200]
                            news_signals.append(sig)

            math_signals = []
            for m in markets:
                sig = math_eng.analyze(m)
                if sig:
                    math_signals.append(sig)

            all_signals = {s["market_id"]: s for s in math_signals}
            for s in news_signals:
                all_signals[s["market_id"]] = s

            signals = sorted(all_signals.values(), key=lambda s: s["ev"] * s["kelly"], reverse=True)

            if signals:
                log.info(f"[SCAN #{scan_count}] {len(markets)} рынков | {len(signals)} сигналов")

            confirmed = []
            for sig in signals[:5]:
                if sig["ev"] >= CONFIG["CLAUDE_EV_THR"]:
                    result = await claude_confirm(sig, CONFIG)
                    if result.get("confirm"):
                        sig["p_claude"] = result.get("p_claude", sig["p_final"])
                        sig["p_final"]  = sig["p_final"] * 0.6 + sig["p_claude"] * 0.4
                        sig["source"]   = "claude"
                        confirmed.append(sig)
                        log.info(f"[CLAUDE] ✅ {sig['question'][:50]}")
                    else:
                        log.info(f"[CLAUDE] ❌ {sig['question'][:50]}")
                else:
                    confirmed.append(sig)

            for sig in confirmed[:3]:
                await db.save_signal({
                    "id":          f"sig_{sig['market_id'][:8]}_{int(now)}",
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
                await execute_signal(sig, db, telegram, CONFIG)

            await monitor_positions(db, telegram, scanner)

            if now - last_history >= CONFIG["HISTORY_INTERVAL"]:
                last_history = now
                await history.analyze()
                await math_eng.load_patterns()

            utc = datetime.now(timezone.utc)
            if utc.hour == 8 and utc.minute < 1:
                await telegram.send(await db.build_report())

        except Exception as e:
            log.error(f"[MAIN] {e}", exc_info=True)

        await asyncio.sleep(CONFIG["SCAN_INTERVAL"])

async def shutdown(db, telegram, scanner):
    log.info("🛑 Shutting down...")
    await scanner.close()
    await telegram.close()
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
