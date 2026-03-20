import os
import logging
import asyncpg
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("db")

class Database:
    def __init__(self):
        self.url  = os.getenv("DATABASE_URL")
        self.pool: Optional[asyncpg.Pool] = None

    async def init(self):
        self.pool = await asyncpg.create_pool(self.url, min_size=2, max_size=10, command_timeout=30)
        await self._create_schema()
        await self._migrate_positions_tp_sl()
        await self._backfill_executed_signals()
        await self._init_stats()
        log.info("[DB] PostgreSQL подключён")

    async def _create_schema(self):
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                CREATE TABLE IF NOT EXISTS markets (
                    id TEXT PRIMARY KEY, slug TEXT, question TEXT NOT NULL,
                    theme TEXT DEFAULT 'other', yes_price REAL, no_price REAL,
                    volume REAL DEFAULT 0, volume_24h REAL DEFAULT 0,
                    liquidity REAL DEFAULT 0, end_date TIMESTAMPTZ,
                    outcome TEXT, resolved_at TIMESTAMPTZ,
                    is_active BOOLEAN DEFAULT TRUE, url TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    id BIGSERIAL PRIMARY KEY, market_id TEXT REFERENCES markets(id),
                    yes_price REAL, volume REAL, volume_delta REAL DEFAULT 0,
                    snapshot_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS news (
                    id BIGSERIAL PRIMARY KEY, source TEXT, title TEXT,
                    url TEXT UNIQUE, keywords TEXT[], theme TEXT,
                    sentiment TEXT DEFAULT 'neutral', published_at TIMESTAMPTZ,
                    processed BOOLEAN DEFAULT FALSE, created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY, market_id TEXT REFERENCES markets(id),
                    question TEXT, side TEXT, side_price REAL,
                    p_market REAL, p_math REAL, p_history REAL, p_claude REAL, p_final REAL,
                    ev REAL, kl REAL, kelly REAL, confidence REAL,
                    volume_ratio REAL DEFAULT 1.0, news_trigger TEXT,
                    source TEXT DEFAULT 'math', executed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY, market_id TEXT REFERENCES markets(id),
                    signal_id TEXT, question TEXT, theme TEXT, side TEXT,
                    side_price REAL, p_final REAL, ev REAL, kl REAL,
                    kelly REAL, stake_amt REAL, current_price REAL,
                    unrealized_pnl REAL DEFAULT 0, outcome TEXT,
                    payout REAL DEFAULT 0, pnl REAL DEFAULT 0,
                    result TEXT, status TEXT DEFAULT 'open', url TEXT,
                    tp_pct REAL DEFAULT 0.20, sl_pct REAL DEFAULT 0.50,
                    opened_at TIMESTAMPTZ DEFAULT NOW(), closed_at TIMESTAMPTZ
                );
                CREATE TABLE IF NOT EXISTS patterns (
                    id BIGSERIAL PRIMARY KEY, category TEXT UNIQUE,
                    base_rate REAL, sample_size INTEGER DEFAULT 0,
                    avg_volume REAL DEFAULT 0, volume_signal REAL DEFAULT 1.0,
                    time_decay_coef REAL DEFAULT 0.05, prospect_factor REAL DEFAULT 1.0,
                    win_rate REAL DEFAULT 0.5, avg_ev REAL DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS calibration (
                    id BIGSERIAL PRIMARY KEY, agent TEXT,
                    brier_score REAL, bias REAL DEFAULT 0, factor REAL DEFAULT 1.0,
                    n_samples INTEGER DEFAULT 0, created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS stats (
                    id INTEGER PRIMARY KEY DEFAULT 1, bankroll REAL NOT NULL,
                    total_pnl REAL DEFAULT 0, total_bets INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
                    avg_ev REAL DEFAULT 0, avg_kelly REAL DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS config_history (
                    tag TEXT PRIMARY KEY,
                    params JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_market ON price_snapshots(market_id, snapshot_at DESC);
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
                CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_news_processed ON news(processed, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(is_active, volume DESC);
            """)
            log.info("[DB] Schema created")
        except Exception as e:
            log.error(f"[DB] Schema creation failed: {e}")
            raise

    async def _init_stats(self):
        bankroll = float(os.getenv("BANKROLL", "1000"))
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO stats (id, bankroll) VALUES (1, $1) ON CONFLICT (id) DO NOTHING",
                bankroll
            )

    async def _migrate_positions_tp_sl(self):
        """Add tp_pct/sl_pct columns to positions if missing (existing DBs)."""
        async with self.pool.acquire() as conn:
            cols = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name='positions'"
            )
            col_names = {r["column_name"] for r in cols}
            if "tp_pct" not in col_names:
                await conn.execute("ALTER TABLE positions ADD COLUMN tp_pct REAL DEFAULT 0.20")
                log.info("[DB] Added tp_pct column to positions")
            if "sl_pct" not in col_names:
                await conn.execute("ALTER TABLE positions ADD COLUMN sl_pct REAL DEFAULT 0.50")
                log.info("[DB] Added sl_pct column to positions")
            if "config_tag" not in col_names:
                await conn.execute("ALTER TABLE positions ADD COLUMN config_tag TEXT DEFAULT 'v0'")
                log.info("[DB] Added config_tag column to positions")

    async def _backfill_executed_signals(self):
        """One-time: mark signals as executed if they have a matching position."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE signals SET executed = TRUE
                    WHERE id IN (SELECT signal_id FROM positions WHERE signal_id IS NOT NULL)
                    AND executed = FALSE
                """)
                count = int(result.split()[-1])
                if count > 0:
                    log.info(f"[DB] Backfilled {count} signals as executed")
        except Exception as e:
            log.warning(f"[DB] Backfill executed signals: {e}")

    async def get_price_history(self, market_id: str, minutes: int = 60) -> list:
        """Returns recent price snapshots [{yes_price, volume, snapshot_at}] ordered ASC."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT yes_price, volume, snapshot_at
                FROM price_snapshots
                WHERE market_id = $1 AND snapshot_at >= NOW() - ($2 || ' minutes')::INTERVAL
                ORDER BY snapshot_at ASC
            """, market_id, str(minutes))
            return [dict(r) for r in rows]

    async def upsert_market(self, m: dict):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO markets (id,slug,question,theme,yes_price,no_price,volume,volume_24h,liquidity,end_date,url,updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,NOW())
                ON CONFLICT (id) DO UPDATE SET
                    yes_price=EXCLUDED.yes_price, no_price=EXCLUDED.no_price,
                    volume=EXCLUDED.volume, volume_24h=EXCLUDED.volume_24h,
                    liquidity=EXCLUDED.liquidity, updated_at=NOW()
            """, m["id"], m.get("slug",""), m["question"], m.get("theme","other"),
                m["yes_price"], m.get("no_price", 1-m["yes_price"]),
                m["volume"], m.get("volume_24h",0), m.get("liquidity",0),
                m.get("end_date"), m.get("url",""))

    async def save_snapshot(self, market_id: str, price: float, volume: float, delta: float):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO price_snapshots (market_id,yes_price,volume,volume_delta) VALUES ($1,$2,$3,$4)",
                market_id, price, volume, delta
            )

    async def get_active_markets(self) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM markets WHERE is_active=TRUE ORDER BY volume DESC LIMIT 500")
            return [dict(r) for r in rows]

    async def get_closed_markets(self, limit: int = 500) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM markets WHERE outcome IS NOT NULL ORDER BY resolved_at DESC LIMIT $1", limit)
            return [dict(r) for r in rows]

    async def save_news(self, item: dict) -> bool:
        async with self.pool.acquire() as conn:
            try:
                await conn.execute("""
                    INSERT INTO news (source,title,url,keywords,theme,sentiment,published_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                """, item["source"], item["title"], item["url"],
                    item.get("keywords",[]), item.get("theme","other"),
                    item.get("sentiment","neutral"), item.get("published_at"))
                return True
            except asyncpg.UniqueViolationError:
                return False

    async def get_unprocessed_news(self) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM news WHERE processed=FALSE ORDER BY created_at DESC LIMIT 20")
            return [dict(r) for r in rows]

    async def mark_news_processed(self, news_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE news SET processed=TRUE WHERE id=$1", news_id)

    async def save_signal(self, sig: dict):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO signals (id,market_id,question,side,side_price,p_market,p_math,p_history,p_claude,p_final,ev,kl,kelly,confidence,volume_ratio,news_trigger,source)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
                ON CONFLICT (id) DO NOTHING
            """, sig["id"], sig["market_id"], sig["question"], sig["side"], sig["side_price"],
                sig["p_market"], sig.get("p_math"), sig.get("p_history"), sig.get("p_claude"), sig["p_final"],
                sig["ev"], sig["kl"], sig["kelly"], sig.get("confidence",0),
                sig.get("volume_ratio",1.0), sig.get("news_trigger"), sig.get("source","math"))

    async def mark_signal_executed(self, signal_id: str):
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE signals SET executed = TRUE WHERE id = $1", signal_id)

    async def get_recent_signals(self, limit: int = 20) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM signals ORDER BY created_at DESC LIMIT $1", limit)
            return [dict(r) for r in rows]

    async def save_position(self, pos: dict):
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("""
                        INSERT INTO positions (id,market_id,signal_id,question,theme,side,side_price,p_final,ev,kl,kelly,stake_amt,current_price,url,tp_pct,sl_pct,config_tag)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
                    """, pos["id"], pos["market_id"], pos.get("signal_id"),
                        pos["question"], pos.get("theme","other"), pos["side"],
                        pos["side_price"], pos["p_final"], pos["ev"], pos["kl"],
                        pos["kelly"], pos["stake_amt"], pos["side_price"], pos.get("url",""),
                        pos.get("tp_pct", 0.20), pos.get("sl_pct", 0.50),
                        pos.get("config_tag", "v0"))
                    await conn.execute("UPDATE stats SET bankroll=bankroll+$1, updated_at=NOW() WHERE id=1", -pos["stake_amt"])
                    await conn.execute("""
                        UPDATE stats SET
                            total_bets=total_bets+1,
                            avg_ev=(avg_ev*total_bets+$1)/(total_bets+1),
                            avg_kelly=(avg_kelly*total_bets+$2)/(total_bets+1),
                            updated_at=NOW()
                        WHERE id=1
                    """, pos["ev"], pos["kelly"])
            log.info(f"[DB] Position opened: {pos['id']} {pos['side']} ${pos['stake_amt']:.2f} on {pos['question'][:50]}")
        except Exception as e:
            log.error(f"[DB] save_position failed: {e}")
            raise

    async def update_position_price(self, pos_id: str, price: float, upnl: float):
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE positions SET current_price=$1, unrealized_pnl=$2 WHERE id=$3", price, upnl, pos_id)

    async def close_position(self, pos_id: str, outcome: str, payout: float, pnl: float):
        result = "WIN" if pnl > 0 else "LOSS"
        won = result == "WIN"
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("""
                        UPDATE positions SET outcome=$1,payout=$2,pnl=$3,result=$4,status='closed',closed_at=NOW() WHERE id=$5
                    """, outcome, payout, pnl, result, pos_id)
                    await conn.execute("UPDATE stats SET bankroll=bankroll+$1, updated_at=NOW() WHERE id=1", payout)
                    await conn.execute("""
                        UPDATE stats SET total_pnl=total_pnl+$1, wins=wins+$2, losses=losses+$3, updated_at=NOW() WHERE id=1
                    """, pnl, 1 if won else 0, 0 if won else 1)
            log.info(f"[DB] Position closed: {pos_id} {result} pnl={pnl:+.2f} payout={payout:.2f}")
        except Exception as e:
            log.error(f"[DB] close_position failed: {e}")
            raise

    async def get_open_positions(self) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM positions WHERE status='open' ORDER BY opened_at DESC")
            return [dict(r) for r in rows]

    async def get_closed_positions(self, limit: int = 100) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM positions WHERE status='closed' ORDER BY closed_at DESC LIMIT $1", limit)
            return [dict(r) for r in rows]

    async def upsert_pattern(self, category: str, data: dict):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO patterns (category,base_rate,sample_size,avg_volume,volume_signal,time_decay_coef,prospect_factor,win_rate,avg_ev,updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NOW())
                ON CONFLICT (category) DO UPDATE SET
                    base_rate=EXCLUDED.base_rate, sample_size=EXCLUDED.sample_size,
                    avg_volume=EXCLUDED.avg_volume, volume_signal=EXCLUDED.volume_signal,
                    time_decay_coef=EXCLUDED.time_decay_coef, prospect_factor=EXCLUDED.prospect_factor,
                    win_rate=EXCLUDED.win_rate, avg_ev=EXCLUDED.avg_ev, updated_at=NOW()
            """, category, data["base_rate"], data["sample_size"],
                data.get("avg_volume",0), data.get("volume_signal",1.0),
                data.get("time_decay_coef",0.05), data.get("prospect_factor",1.0),
                data.get("win_rate",0.5), data.get("avg_ev",0))
        log.debug(f"[DB] Pattern upserted: {category} base_rate={data['base_rate']:.4f} n={data['sample_size']}")

    async def get_patterns(self) -> dict:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM patterns")
            return {r["category"]: dict(r) for r in rows}

    async def save_calibration(self, agent: str, brier: float, bias: float, factor: float, n: int):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO calibration (agent,brier_score,bias,factor,n_samples) VALUES ($1,$2,$3,$4,$5)",
                agent, brier, bias, factor, n
            )

    async def get_latest_calibration(self, agent: str) -> dict:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM calibration WHERE agent=$1 ORDER BY created_at DESC LIMIT 1", agent)
            return dict(row) if row else {"brier_score":0.25,"bias":0,"factor":1.0,"n_samples":0}

    async def get_stats(self) -> dict:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM stats WHERE id=1")
            return dict(row) if row else {"bankroll": float(os.getenv("BANKROLL","1000")), "total_pnl":0,"total_bets":0,"wins":0,"losses":0,"avg_ev":0,"avg_kelly":0}

    async def update_bankroll(self, delta: float):
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE stats SET bankroll=bankroll+$1, updated_at=NOW() WHERE id=1", delta)

    async def _update_stats_on_open(self, pos: dict):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE stats SET
                    total_bets=total_bets+1,
                    avg_ev=(avg_ev*total_bets+$1)/(total_bets+1),
                    avg_kelly=(avg_kelly*total_bets+$2)/(total_bets+1),
                    updated_at=NOW()
                WHERE id=1
            """, pos["ev"], pos["kelly"])

    async def _update_stats_on_close(self, pnl: float, result: str):
        won = result == "WIN"
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE stats SET total_pnl=total_pnl+$1, wins=wins+$2, losses=losses+$3, updated_at=NOW() WHERE id=1
            """, pnl, 1 if won else 0, 0 if won else 1)

    async def build_report(self) -> str:
        stats = await self.get_stats()
        open_ = await self.get_open_positions()
        start = float(os.getenv("BANKROLL","1000"))
        roi   = (stats["bankroll"] - start) / start * 100
        total = stats["wins"] + stats["losses"]
        wr    = round(stats["wins"]/total*100) if total > 0 else 0
        return (
            f"📊 <b>ЕЖЕДНЕВНЫЙ ОТЧЁТ</b>\n\n"
            f"💼 Банкролл: <b>${stats['bankroll']:.2f}</b>\n"
            f"📈 ROI: <b>{roi:+.2f}%</b>\n"
            f"💰 P&L: <b>{stats['total_pnl']:+.2f}$</b>\n\n"
            f"🎯 WR:{wr}% | ✅{stats['wins']} / ❌{stats['losses']}\n"
            f"📋 Ставок:{stats['total_bets']} | Открытых:{len(open_)}\n"
            f"⚡ Avg EV:+{stats['avg_ev']*100:.1f}%"
        )

    async def save_config_snapshot(self, tag: str, config: dict):
        """Save config params for A/B tracking. Skips if tag already exists."""
        import json
        # Only save trading-relevant params
        params = {k: v for k, v in config.items() if k not in ("ANTHROPIC_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID")}
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO config_history (tag, params) VALUES ($1, $2)
                ON CONFLICT (tag) DO NOTHING
            """, tag, json.dumps(params))

    async def get_config_history(self) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM config_history ORDER BY created_at DESC")
            return [dict(r) for r in rows]

    async def get_signal_outcomes(self, limit: int = 200) -> list:
        """Check what happened to signals after they were generated.
        Uses market price if available, otherwise position outcome."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT s.id, s.question, s.side, s.side_price, s.p_market, s.p_final,
                    s.ev, s.kelly, s.source, s.executed, s.created_at,
                    COALESCE(m.yes_price, CASE WHEN p.result='WIN' THEN
                        CASE WHEN s.side='YES' THEN 0.95 ELSE 0.05 END
                        ELSE CASE WHEN s.side='YES' THEN 0.05 ELSE 0.95 END
                    END) as current_price,
                    COALESCE(m.is_active, FALSE) as is_active,
                    CASE WHEN m.id IS NOT NULL THEN
                        CASE WHEN s.side = 'YES' THEN m.yes_price - s.side_price
                             ELSE (1 - m.yes_price) - s.side_price END
                    WHEN p.id IS NOT NULL THEN
                        CASE WHEN p.result = 'WIN' THEN ABS(s.side_price)
                             ELSE -s.side_price END
                    END as price_move
                FROM signals s
                LEFT JOIN markets m ON s.market_id = m.id
                LEFT JOIN positions p ON s.id = p.signal_id
                ORDER BY s.created_at DESC
                LIMIT $1
            """, limit)
            return [dict(r) for r in rows]

    async def get_cumulative_pnl(self) -> list:
        """Returns cumulative PnL over time for charting."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT closed_at, pnl,
                    SUM(pnl) OVER (ORDER BY closed_at) as cumulative
                FROM positions
                WHERE status='closed' AND closed_at IS NOT NULL
                ORDER BY closed_at ASC
            """)
            return [{"t": r["closed_at"].isoformat(), "pnl": float(r["pnl"]), "cum": float(r["cumulative"])} for r in rows]

    async def get_analytics(self) -> dict:
        """Compute analytics for dashboard: win rates, calibration, timing."""
        async with self.pool.acquire() as conn:
            # Win rate by theme
            by_theme = await conn.fetch("""
                SELECT theme, COUNT(*) as total,
                    SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                    ROUND(AVG(pnl)::numeric, 2) as avg_pnl
                FROM positions WHERE status='closed' AND theme IS NOT NULL
                GROUP BY theme ORDER BY total DESC
            """)

            # Win rate by source (from signal_id → signals.source, fallback to 'math')
            by_source = await conn.fetch("""
                SELECT COALESCE(s.source, 'math') as source, COUNT(*) as total,
                    SUM(CASE WHEN p.result='WIN' THEN 1 ELSE 0 END) as wins,
                    ROUND(AVG(p.pnl)::numeric, 2) as avg_pnl
                FROM positions p
                LEFT JOIN signals s ON p.signal_id = s.id
                WHERE p.status='closed'
                GROUP BY COALESCE(s.source, 'math') ORDER BY total DESC
            """)

            # Win rate by side
            by_side = await conn.fetch("""
                SELECT side, COUNT(*) as total,
                    SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                    ROUND(AVG(pnl)::numeric, 2) as avg_pnl
                FROM positions WHERE status='closed'
                GROUP BY side
            """)

            # Close reason breakdown
            by_reason = await conn.fetch("""
                SELECT
                    CASE
                        WHEN outcome LIKE '%@%' AND pnl > 0 THEN 'TAKE_PROFIT'
                        WHEN outcome LIKE '%@%' AND pnl <= 0 THEN 'STOP_LOSS'
                        ELSE 'RESOLVED'
                    END as reason,
                    COUNT(*) as total,
                    ROUND(AVG(pnl)::numeric, 2) as avg_pnl
                FROM positions WHERE status='closed'
                GROUP BY reason ORDER BY total DESC
            """)

            # Win rate by config tag (A/B testing)
            by_config = await conn.fetch("""
                SELECT COALESCE(config_tag, 'v0') as config_tag, COUNT(*) as total,
                    SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                    ROUND(SUM(pnl)::numeric, 2) as total_pnl,
                    ROUND(AVG(pnl)::numeric, 2) as avg_pnl,
                    ROUND(AVG(ev)::numeric, 4) as avg_ev,
                    ROUND(AVG(stake_amt)::numeric, 2) as avg_stake
                FROM positions WHERE status='closed'
                GROUP BY COALESCE(config_tag, 'v0') ORDER BY config_tag
            """)

            # Calibration buckets: predicted p_final vs actual outcome
            calibration = await conn.fetch("""
                SELECT
                    CASE
                        WHEN p_final < 0.3 THEN '0-30%'
                        WHEN p_final < 0.5 THEN '30-50%'
                        WHEN p_final < 0.7 THEN '50-70%'
                        ELSE '70-100%'
                    END as bucket,
                    COUNT(*) as total,
                    ROUND(AVG(p_final)::numeric, 3) as avg_predicted,
                    ROUND(AVG(CASE WHEN result='WIN' THEN 1.0 ELSE 0.0 END)::numeric, 3) as actual_wr
                FROM positions WHERE status='closed'
                GROUP BY bucket ORDER BY bucket
            """)

            # Average position lifetime (hours)
            avg_lifetime = await conn.fetchrow("""
                SELECT ROUND(AVG(EXTRACT(EPOCH FROM (closed_at - opened_at)) / 3600)::numeric, 1) as avg_hours
                FROM positions WHERE status='closed' AND closed_at IS NOT NULL
            """)

            # PnL over time (daily)
            daily_pnl = await conn.fetch("""
                SELECT DATE(closed_at) as day,
                    ROUND(SUM(pnl)::numeric, 2) as pnl,
                    COUNT(*) as trades,
                    SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins
                FROM positions WHERE status='closed' AND closed_at IS NOT NULL
                GROUP BY day ORDER BY day DESC LIMIT 14
            """)

            # EV accuracy: predicted EV vs actual return
            ev_accuracy = await conn.fetchrow("""
                SELECT
                    ROUND(AVG(ev)::numeric, 4) as avg_predicted_ev,
                    ROUND(AVG(pnl / NULLIF(stake_amt, 0))::numeric, 4) as avg_actual_return
                FROM positions WHERE status='closed' AND stake_amt > 0
            """)

        return {
            "by_config": [dict(r) for r in by_config],
            "by_theme": [dict(r) for r in by_theme],
            "by_source": [dict(r) for r in by_source],
            "by_side": [dict(r) for r in by_side],
            "by_reason": [dict(r) for r in by_reason],
            "calibration": [dict(r) for r in calibration],
            "avg_lifetime_hours": float(avg_lifetime["avg_hours"] or 0) if avg_lifetime else 0,
            "daily_pnl": [dict(r) for r in daily_pnl],
            "ev_predicted": float(ev_accuracy["avg_predicted_ev"] or 0) if ev_accuracy else 0,
            "ev_actual": float(ev_accuracy["avg_actual_return"] or 0) if ev_accuracy else 0,
        }

    async def cleanup(self, snap_days: int = 1, sig_days: int = 7, news_days: int = 5):
        """Delete old data to prevent DB bloat.
        KEEP FOREVER: positions (analytics/calibration), markets (backtest), executed signals.
        DELETE: price_snapshots, unexecuted signals, processed news."""
        try:
            async with self.pool.acquire() as conn:
                r1 = await conn.execute(
                    "DELETE FROM price_snapshots WHERE snapshot_at < NOW() - ($1 || ' days')::INTERVAL",
                    str(snap_days)
                )
                r2 = await conn.execute(
                    "DELETE FROM signals WHERE created_at < NOW() - ($1 || ' days')::INTERVAL AND executed = FALSE",
                    str(sig_days)
                )
                r3 = await conn.execute(
                    "DELETE FROM news WHERE created_at < NOW() - ($1 || ' days')::INTERVAL AND processed = TRUE",
                    str(news_days)
                )
                # reclaim disk space
                await conn.execute("VACUUM")
                log.info(f"[DB] Cleanup: snapshots={r1}, signals={r2}, news={r3}")
        except Exception as e:
            log.error(f"[DB] Cleanup failed: {e}")

    async def close(self):
        if self.pool:
            await self.pool.close()
            log.info("[DB] Connection pool closed")
