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
        self.pool = await asyncpg.create_pool(self.url, min_size=2, max_size=15, command_timeout=30)
        await self._create_schema()
        await self._migrate_positions_tp_sl()
        await self._migrate_clv()
        await self._migrate_patterns_theme_perf()
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
                CREATE TABLE IF NOT EXISTS market_metrics (
                    market_id TEXT PRIMARY KEY REFERENCES markets(id),
                    volatility REAL DEFAULT 0,
                    momentum REAL DEFAULT 0,
                    vol_ratio REAL DEFAULT 1.0,
                    vol_direction TEXT DEFAULT 'neutral',
                    long_prices REAL[] DEFAULT '{}',
                    short_prices REAL[] DEFAULT '{}',
                    last_signal_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS trade_log (
                    id BIGSERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    market_id TEXT,
                    position_id TEXT,
                    signal_id TEXT,
                    question TEXT,
                    theme TEXT,
                    side TEXT,
                    side_price REAL,
                    yes_price REAL,
                    p_market REAL,
                    p_final REAL,
                    p_prospect REAL,
                    p_history REAL,
                    p_claude REAL,
                    p_ml REAL,
                    ev REAL,
                    kl REAL,
                    kelly REAL,
                    edge REAL,
                    entropy REAL,
                    stake_amt REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    payout REAL,
                    tp_pct REAL,
                    sl_pct REAL,
                    bankroll REAL,
                    equity REAL,
                    open_positions INTEGER,
                    drawdown_pct REAL,
                    peak_equity REAL,
                    is_contrarian BOOLEAN DEFAULT FALSE,
                    is_simulation BOOLEAN DEFAULT TRUE,
                    config_tag TEXT,
                    details JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_snapshots_market ON price_snapshots(market_id, snapshot_at DESC);
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
                CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_news_processed ON news(processed, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(is_active, volume DESC);
                CREATE TABLE IF NOT EXISTS trader_commands (
                    id BIGSERIAL PRIMARY KEY,
                    command TEXT NOT NULL,
                    position_id TEXT,
                    params JSONB DEFAULT '{}',
                    status TEXT DEFAULT 'pending',
                    result JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    executed_at TIMESTAMPTZ
                );
                CREATE INDEX IF NOT EXISTS idx_trade_log_event ON trade_log(event_type, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_trade_log_market ON trade_log(market_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_trade_log_created ON trade_log(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_trader_commands_status ON trader_commands(status) WHERE status='pending';
                CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_open_market ON positions(market_id) WHERE status='open';
                CREATE TABLE IF NOT EXISTS dma_weights (
                    source TEXT PRIMARY KEY,
                    weight REAL NOT NULL DEFAULT 1.0,
                    hits INTEGER DEFAULT 0,
                    misses INTEGER DEFAULT 0,
                    avg_likelihood REAL DEFAULT 0.5,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            log.info("[DB] Schema created")
        except Exception as e:
            log.error(f"[DB] Schema creation failed: {e}")
            raise

    async def log_event(self, event_type: str, **kw):
        """Append a row to trade_log. Fire-and-forget — never raises."""
        try:
            import json as _json
            details = kw.pop("details", None)
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trade_log (
                        event_type, market_id, position_id, signal_id, question, theme, side,
                        side_price, yes_price, p_market, p_final, p_prospect, p_history, p_claude, p_ml,
                        ev, kl, kelly, edge, entropy, stake_amt, pnl, pnl_pct, payout, tp_pct, sl_pct,
                        bankroll, equity, open_positions, drawdown_pct, peak_equity,
                        is_contrarian, is_simulation, config_tag, details
                    ) VALUES (
                        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
                        $21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35
                    )
                """,
                    event_type,
                    kw.get("market_id"), kw.get("position_id"), kw.get("signal_id"),
                    kw.get("question"), kw.get("theme"), kw.get("side"),
                    kw.get("side_price"), kw.get("yes_price"),
                    kw.get("p_market"), kw.get("p_final"), kw.get("p_prospect"),
                    kw.get("p_history"), kw.get("p_claude"), kw.get("p_ml"),
                    kw.get("ev"), kw.get("kl"), kw.get("kelly"), kw.get("edge"), kw.get("entropy"),
                    kw.get("stake_amt"), kw.get("pnl"), kw.get("pnl_pct"), kw.get("payout"),
                    kw.get("tp_pct"), kw.get("sl_pct"),
                    kw.get("bankroll"), kw.get("equity"), kw.get("open_positions"),
                    kw.get("drawdown_pct"), kw.get("peak_equity"),
                    kw.get("is_contrarian", False), kw.get("is_simulation", True),
                    kw.get("config_tag"),
                    _json.dumps(details) if details else None,
                )
        except Exception as e:
            log.warning(f"[DB] log_event({event_type}) failed: {e}")

    async def get_trade_log(self, limit=200, event_type=None, market_id=None) -> list:
        """Read trade_log entries for dashboard."""
        async with self.pool.acquire() as conn:
            q = "SELECT * FROM trade_log WHERE 1=1"
            args = []
            i = 1
            if event_type:
                q += f" AND event_type = ${i}"; args.append(event_type); i += 1
            if market_id:
                q += f" AND market_id = ${i}"; args.append(market_id); i += 1
            q += f" ORDER BY created_at DESC LIMIT ${i}"; args.append(limit)
            rows = await conn.fetch(q, *args)
            return [dict(r) for r in rows]

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

    async def _migrate_clv(self):
        """Add CLV (Closing Line Value) columns to positions for entry quality tracking."""
        async with self.pool.acquire() as conn:
            cols = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name='positions'"
            )
            col_names = {r["column_name"] for r in cols}
            new_cols = {
                "clv_1h":         "REAL",   # market price 1h after entry
                "clv_4h":         "REAL",   # market price 4h after entry
                "clv_24h":        "REAL",   # market price 24h after entry
                "clv_close":      "REAL",   # market price at resolution/close
            }
            for col, typ in new_cols.items():
                if col not in col_names:
                    await conn.execute(f"ALTER TABLE positions ADD COLUMN {col} {typ}")
                    log.info(f"[DB] Added {col} column to positions (CLV tracking)")

    async def _migrate_patterns_theme_perf(self):
        """Add per-theme performance columns to patterns for Bayesian theme calibration."""
        async with self.pool.acquire() as conn:
            cols = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name='patterns'"
            )
            col_names = {r["column_name"] for r in cols}
            new_cols = {
                "trade_n":    "INTEGER DEFAULT 0",
                "trade_wr":   "REAL DEFAULT 0.5",
                "trade_roi":  "REAL DEFAULT 0",
                "kelly_mult": "REAL DEFAULT 1.0",
                "ev_mult":    "REAL DEFAULT 1.0",
            }
            for col, typedef in new_cols.items():
                if col not in col_names:
                    await conn.execute(f"ALTER TABLE patterns ADD COLUMN {col} {typedef}")
                    log.info(f"[DB] Added {col} column to patterns")

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

    async def save_market_metrics(self, market_id: str, metrics: dict):
        """Upsert market metrics (volatility, momentum, caches)."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO market_metrics (market_id, volatility, momentum, vol_ratio, vol_direction,
                    long_prices, short_prices, last_signal_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (market_id) DO UPDATE SET
                    volatility = $2, momentum = $3, vol_ratio = $4, vol_direction = $5,
                    long_prices = $6, short_prices = $7,
                    last_signal_at = COALESCE($8, market_metrics.last_signal_at),
                    updated_at = NOW()
            """, market_id, metrics.get("volatility", 0), metrics.get("momentum", 0),
                metrics.get("vol_ratio", 1.0), metrics.get("vol_direction", "neutral"),
                metrics.get("long_prices", []), metrics.get("short_prices", []),
                metrics.get("last_signal_at"))

    async def mark_signal_cooldown(self, market_id: str):
        """Update last_signal_at for cooldown tracking."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO market_metrics (market_id, last_signal_at, updated_at)
                VALUES ($1, NOW(), NOW())
                ON CONFLICT (market_id) DO UPDATE SET last_signal_at = NOW(), updated_at = NOW()
            """, market_id)

    async def get_all_market_metrics(self) -> list:
        """Load all market metrics for warm restart and analytics."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT mm.*, m.question, m.yes_price, m.theme
                FROM market_metrics mm
                JOIN markets m ON mm.market_id = m.id
                WHERE m.is_active = TRUE
                ORDER BY mm.updated_at DESC
            """)
            return [dict(r) for r in rows]

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

    async def upsert_markets_batch(self, markets: list):
        """Batch upsert markets in a single transaction."""
        if not markets:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO markets (id,slug,question,theme,yes_price,no_price,volume,volume_24h,liquidity,end_date,url,updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,NOW())
                ON CONFLICT (id) DO UPDATE SET
                    yes_price=EXCLUDED.yes_price, no_price=EXCLUDED.no_price,
                    volume=EXCLUDED.volume, volume_24h=EXCLUDED.volume_24h,
                    liquidity=EXCLUDED.liquidity, updated_at=NOW()
            """, [(m["id"], m.get("slug",""), m["question"], m.get("theme","other"),
                   m["yes_price"], m.get("no_price", 1-m["yes_price"]),
                   m["volume"], m.get("volume_24h",0), m.get("liquidity",0),
                   m.get("end_date"), m.get("url","")) for m in markets])

    async def save_snapshot(self, market_id: str, price: float, volume: float, delta: float):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO price_snapshots (market_id,yes_price,volume,volume_delta) VALUES ($1,$2,$3,$4)",
                market_id, price, volume, delta
            )

    async def save_snapshots_batch(self, snapshots: list):
        """Batch insert snapshots. snapshots = [(market_id, price, volume, delta), ...]"""
        if not snapshots:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO price_snapshots (market_id,yes_price,volume,volume_delta) VALUES ($1,$2,$3,$4)",
                snapshots
            )

    async def get_active_markets(self) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM markets WHERE is_active=TRUE ORDER BY volume DESC LIMIT 500")
            return [dict(r) for r in rows]

    async def get_closed_markets(self, limit: int = 500) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM markets WHERE outcome IS NOT NULL ORDER BY resolved_at DESC LIMIT $1", limit)
            return [dict(r) for r in rows]

    async def get_signal_prices_by_theme(self) -> dict:
        """Get average p_market at signal time per theme (from signals + markets join)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT m.theme, AVG(s.p_market) as avg_p_market, COUNT(*) as n
                FROM signals s
                JOIN markets m ON s.market_id = m.id
                WHERE s.p_market IS NOT NULL AND s.p_market > 0.01 AND s.p_market < 0.99
                GROUP BY m.theme
                HAVING COUNT(*) >= 5
            """)
            return {r["theme"]: float(r["avg_p_market"]) for r in rows}

    async def get_win_loss_stats(self, theme: str = None) -> dict:
        """Get W/L counts for Claude context. Returns {theme_w, theme_l, total_w, total_l}."""
        async with self.pool.acquire() as conn:
            total = await conn.fetchrow("""
                SELECT COUNT(*) FILTER (WHERE result='WIN') as w,
                       COUNT(*) FILTER (WHERE result='LOSS') as l
                FROM positions WHERE status='closed' AND result IS NOT NULL
            """)
            theme_row = None
            if theme:
                theme_row = await conn.fetchrow("""
                    SELECT COUNT(*) FILTER (WHERE result='WIN') as w,
                           COUNT(*) FILTER (WHERE result='LOSS') as l
                    FROM positions WHERE status='closed' AND result IS NOT NULL AND theme=$1
                """, theme)
            return {
                "total_w": total["w"], "total_l": total["l"],
                "theme_w": theme_row["w"] if theme_row else 0,
                "theme_l": theme_row["l"] if theme_row else 0,
            }

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

    async def get_rejected_signal_outcomes(self, limit: int = 200) -> list:
        """Get rejected (unexecuted) signals whose markets have since resolved.
        Used for calibration to fix confirmation bias."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT s.p_final, s.p_market, s.market_id, m.outcome
                FROM signals s
                JOIN markets m ON s.market_id = m.id
                WHERE s.executed = FALSE
                  AND m.outcome IS NOT NULL
                ORDER BY s.created_at DESC
                LIMIT $1
            """, limit)
            return [dict(r) for r in rows]

    async def save_position(self, pos: dict) -> bool:
        """Save position. Returns False if duplicate open position on same market (unique index guard)."""
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
            return True
        except asyncpg.UniqueViolationError:
            log.warning(f"[DB] Duplicate position blocked: market {pos['market_id']} already has open position")
            return False
        except Exception as e:
            log.error(f"[DB] save_position failed: {e}")
            raise

    async def update_position_price(self, pos_id: str, price: float, upnl: float):
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE positions SET current_price=$1, unrealized_pnl=$2 WHERE id=$3", price, upnl, pos_id)

    async def close_position(self, pos_id: str, outcome: str, payout: float, pnl: float) -> bool:
        """Close a position. Returns False if already closed (race protection)."""
        result = "WIN" if pnl >= 0 else "LOSS"
        won = result == "WIN"
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Only close if still open — prevents double-close from WS + REST race
                    row = await conn.fetchrow("""
                        UPDATE positions SET outcome=$1,payout=$2,pnl=$3,result=$4,status='closed',closed_at=NOW()
                        WHERE id=$5 AND status='open' RETURNING id
                    """, outcome, payout, pnl, result, pos_id)
                    if not row:
                        log.warning(f"[DB] close_position skipped (already closed): {pos_id}")
                        return False
                    await conn.execute("UPDATE stats SET bankroll=bankroll+$1, updated_at=NOW() WHERE id=1", payout)
                    await conn.execute("""
                        UPDATE stats SET total_pnl=total_pnl+$1, wins=wins+$2, losses=losses+$3, updated_at=NOW() WHERE id=1
                    """, pnl, 1 if won else 0, 0 if won else 1)
            log.info(f"[DB] Position closed: {pos_id} {result} pnl={pnl:+.2f} payout={payout:.2f}")
            return True
        except Exception as e:
            log.error(f"[DB] close_position failed: {e}")
            raise

    # ── DMA (Dynamic Model Averaging) ──

    async def get_dma_weights(self) -> dict:
        """Get current DMA weights for all sources. Returns {source: weight}."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT source, weight FROM dma_weights")
            return {r["source"]: float(r["weight"]) for r in rows}

    async def save_dma_weights(self, weights: dict):
        """Upsert all DMA weights."""
        async with self.pool.acquire() as conn:
            for source, data in weights.items():
                await conn.execute("""
                    INSERT INTO dma_weights (source, weight, hits, misses, avg_likelihood, updated_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    ON CONFLICT (source) DO UPDATE SET
                        weight = $2, hits = $3, misses = $4, avg_likelihood = $5, updated_at = NOW()
                """, source, data["weight"], data["hits"], data["misses"], data["avg_likelihood"])

    async def get_closed_positions_with_signals(self, limit: int = 100) -> list:
        """Get recently closed positions with their signal source probabilities from trade_log."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT p.id, p.market_id, p.side, p.side_price, p.result, p.pnl,
                       p.p_final, p.theme, p.closed_at,
                       tl.details
                FROM positions p
                LEFT JOIN trade_log tl ON tl.signal_id = p.signal_id AND tl.event_type = 'SIGNAL_GENERATED'
                WHERE p.status = 'closed' AND p.result IS NOT NULL
                ORDER BY p.closed_at DESC
                LIMIT $1
            """, limit)
            return [dict(r) for r in rows]

    # ── CLV (Closing Line Value) ──

    async def update_clv(self, pos_id: str, column: str, price: float):
        """Update a CLV column (clv_1h, clv_4h, clv_24h, clv_close) if not already set."""
        if column not in ("clv_1h", "clv_4h", "clv_24h", "clv_close"):
            return
        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE positions SET {column} = $1
                WHERE id = $2 AND {column} IS NULL AND status = 'open'
            """, price, pos_id)

    async def update_clv_batch(self, updates: list):
        """Batch CLV updates. updates = [(pos_id, column, price), ...].
        Groups by column and runs one query per column."""
        if not updates:
            return
        by_col = {}
        for pos_id, column, price in updates:
            if column not in ("clv_1h", "clv_4h", "clv_24h", "clv_close"):
                continue
            by_col.setdefault(column, []).append((pos_id, price))
        async with self.pool.acquire() as conn:
            for column, pairs in by_col.items():
                await conn.executemany(f"""
                    UPDATE positions SET {column} = $2
                    WHERE id = $1 AND {column} IS NULL AND status = 'open'
                """, pairs)

    async def update_clv_close(self, pos_id: str, price: float):
        """Set CLV at close time (always overwrite)."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE positions SET clv_close = $1 WHERE id = $2",
                price, pos_id,
            )

    async def get_clv_analytics(self) -> dict:
        """Compute CLV stats: do we consistently enter at better prices than market moves to?"""
        async with self.pool.acquire() as conn:
            # CLV = (entry_price - clv_price) / entry_price for our side
            # Positive CLV = we bought cheaper than later price (good entry)
            # For YES: positive if price went UP after we bought
            # For NO: positive if price went DOWN after we bought (= NO price went up)
            rows = await conn.fetch("""
                SELECT side, side_price, clv_1h, clv_4h, clv_24h, clv_close,
                       result, theme, config_tag
                FROM positions
                WHERE status = 'closed' AND side_price > 0
                ORDER BY closed_at DESC LIMIT 500
            """)
            if not rows:
                return {"avg_clv_1h": 0, "avg_clv_4h": 0, "avg_clv_24h": 0, "avg_clv_close": 0,
                        "total": 0, "positive_clv_pct": 0, "by_theme": [], "by_tag": []}

            def clv_val(row, col):
                v = row.get(col)
                if v is None:
                    return None
                entry = row["side_price"]
                if row["side"] == "YES":
                    return (v - entry) / entry  # price went up = good entry
                else:
                    return (entry - v) / entry  # YES price went down = NO price up = good entry

            clvs = {"1h": [], "4h": [], "24h": [], "close": []}
            by_theme = {}
            by_tag = {}

            for r in rows:
                for label, col in [("1h", "clv_1h"), ("4h", "clv_4h"), ("24h", "clv_24h"), ("close", "clv_close")]:
                    v = clv_val(r, col)
                    if v is not None:
                        clvs[label].append(v)

                # Per-theme CLV (use clv_close as primary)
                cv = clv_val(r, "clv_close")
                if cv is not None:
                    theme = r.get("theme") or "other"
                    by_theme.setdefault(theme, []).append(cv)
                    tag = r.get("config_tag") or "?"
                    by_tag.setdefault(tag, []).append(cv)

            def avg(lst):
                return round(sum(lst) / len(lst) * 100, 2) if lst else 0

            def pos_pct(lst):
                return round(sum(1 for v in lst if v > 0) / len(lst) * 100, 1) if lst else 0

            theme_stats = [{"theme": t, "avg_clv": avg(vs), "positive_pct": pos_pct(vs), "n": len(vs)}
                           for t, vs in sorted(by_theme.items(), key=lambda x: -len(x[1]))]
            tag_stats = [{"tag": t, "avg_clv": avg(vs), "positive_pct": pos_pct(vs), "n": len(vs)}
                         for t, vs in sorted(by_tag.items(), key=lambda x: -len(x[1]))]

            return {
                "avg_clv_1h": avg(clvs["1h"]),
                "avg_clv_4h": avg(clvs["4h"]),
                "avg_clv_24h": avg(clvs["24h"]),
                "avg_clv_close": avg(clvs["close"]),
                "positive_clv_pct": pos_pct(clvs["close"]),
                "total": len(rows),
                "by_theme": theme_stats,
                "by_tag": tag_stats,
            }

    # ── Trader Commands ──

    async def fetch_pending_commands(self) -> list:
        """Fetch all pending commands, atomically marking them as 'processing'."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                UPDATE trader_commands SET status='processing'
                WHERE status='pending'
                RETURNING id, command, position_id, params, created_at
            """)
            return [dict(r) for r in rows]

    async def complete_command(self, cmd_id: int, result: dict):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE trader_commands SET status='done', result=$1::jsonb, executed_at=NOW()
                WHERE id=$2
            """, __import__('json').dumps(result), cmd_id)

    async def fail_command(self, cmd_id: int, error: str):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE trader_commands SET status='error', result=$1::jsonb, executed_at=NOW()
                WHERE id=$2
            """, __import__('json').dumps({"error": error}), cmd_id)

    async def setup_listen(self, conn):
        """Start LISTEN on trader_commands channel."""
        await conn.execute("LISTEN trader_commands")

    async def get_open_positions(self) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT p.*, m.end_date as end_date
                FROM positions p
                LEFT JOIN markets m ON p.market_id = m.id
                WHERE p.status='open' ORDER BY p.opened_at DESC
            """)
            return [dict(r) for r in rows]

    async def get_closed_positions(self, limit: int = 100) -> list:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM positions WHERE status='closed' ORDER BY closed_at DESC LIMIT $1", limit)
            return [dict(r) for r in rows]

    async def upsert_pattern(self, category: str, data: dict):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO patterns (category,base_rate,sample_size,avg_volume,volume_signal,
                    time_decay_coef,prospect_factor,win_rate,avg_ev,
                    trade_n,trade_wr,trade_roi,kelly_mult,ev_mult,updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,NOW())
                ON CONFLICT (category) DO UPDATE SET
                    base_rate=EXCLUDED.base_rate, sample_size=EXCLUDED.sample_size,
                    avg_volume=EXCLUDED.avg_volume, volume_signal=EXCLUDED.volume_signal,
                    time_decay_coef=EXCLUDED.time_decay_coef, prospect_factor=EXCLUDED.prospect_factor,
                    win_rate=EXCLUDED.win_rate, avg_ev=EXCLUDED.avg_ev,
                    trade_n=EXCLUDED.trade_n, trade_wr=EXCLUDED.trade_wr,
                    trade_roi=EXCLUDED.trade_roi, kelly_mult=EXCLUDED.kelly_mult,
                    ev_mult=EXCLUDED.ev_mult, updated_at=NOW()
            """, category, data["base_rate"], data["sample_size"],
                data.get("avg_volume",0), data.get("volume_signal",1.0),
                data.get("time_decay_coef",0.05), data.get("prospect_factor",1.0),
                data.get("win_rate",0.5), data.get("avg_ev",0),
                data.get("trade_n",0), data.get("trade_wr",0.5),
                data.get("trade_roi",0), data.get("kelly_mult",1.0),
                data.get("ev_mult",1.0))
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
        data = await self.get_analytics()
        start = float(os.getenv("BANKROLL","1000"))
        roi   = (stats["bankroll"] - start) / start * 100
        total = stats["wins"] + stats["losses"]
        wr    = round(stats["wins"]/total*100) if total > 0 else 0

        # Portfolio value
        pos_value = sum((p.get("stake_amt",0) + (p.get("unrealized_pnl",0) or 0)) for p in open_)
        equity = stats["bankroll"] + pos_value
        upnl_total = sum((p.get("unrealized_pnl",0) or 0) for p in open_)

        report = (
            f"📊 <b>ЕЖЕДНЕВНЫЙ ОТЧЁТ</b>\n\n"
            f"💼 Банкролл: <b>${stats['bankroll']:.2f}</b>\n"
            f"💎 Equity: <b>${equity:.2f}</b>\n"
            f"📈 ROI: <b>{roi:+.2f}%</b>\n"
            f"💰 P&L: <b>{stats['total_pnl']:+.2f}$</b>\n"
            f"📊 Unrealized: <b>{upnl_total:+.2f}$</b>\n\n"
            f"🎯 WR:{wr}% | ✅{stats['wins']} / ❌{stats['losses']}\n"
            f"📋 Ставок:{stats['total_bets']} | Открытых:{len(open_)}\n"
            f"⚡ Avg EV:+{stats['avg_ev']*100:.1f}% | Avg Kelly:{stats['avg_kelly']*100:.1f}%\n"
        )

        # Win rate by theme
        if data["by_theme"]:
            report += "\n<b>📂 По темам:</b>\n"
            for r in data["by_theme"][:8]:
                t_wr = round(r['wins']/r['total']*100) if r['total'] > 0 else 0
                report += f"  {r['theme']}: {r['wins']}/{r['total']} ({t_wr}%) pnl:{float(r['avg_pnl']):+.2f}\n"

        # Win rate by side
        if data["by_side"]:
            report += "\n<b>📐 По сторонам:</b>\n"
            for r in data["by_side"]:
                s_wr = round(r['wins']/r['total']*100) if r['total'] > 0 else 0
                report += f"  {r['side']}: {r['wins']}/{r['total']} ({s_wr}%) pnl:{float(r['avg_pnl']):+.2f}\n"

        # Close reasons
        if data["by_reason"]:
            report += "\n<b>🔒 Причины закрытия:</b>\n"
            for r in data["by_reason"]:
                report += f"  {r['reason']}: {r['total']} | pnl:{float(r['avg_pnl']):+.2f}\n"

        # Calibration
        if data["calibration"]:
            report += "\n<b>🎯 Калибровка:</b>\n"
            for r in data["calibration"]:
                report += f"  {r['bucket']}: {r['total']}шт pred:{float(r['avg_predicted'])*100:.0f}% act:{float(r['actual_wr'])*100:.0f}%\n"

        # EV accuracy
        report += (
            f"\n<b>📈 Точность:</b>\n"
            f"  EV pred:+{data['ev_predicted']*100:.1f}% | actual:{data['ev_actual']*100:+.1f}%\n"
            f"  Avg lifetime: {data['avg_lifetime_hours']:.1f}h\n"
        )

        # Daily PnL (last 5 days)
        if data["daily_pnl"]:
            report += "\n<b>📅 По дням:</b>\n"
            for r in data["daily_pnl"][:5]:
                d_wr = round(r['wins']/r['trades']*100) if r['trades'] > 0 else 0
                report += f"  {r['day']}: {float(r['pnl']):+.2f}$ ({r['trades']}шт, {d_wr}%WR)\n"

        # Top open positions by unrealized PnL
        if open_:
            sorted_pos = sorted(open_, key=lambda p: p.get("unrealized_pnl",0) or 0)
            best = sorted_pos[-1] if sorted_pos else None
            worst = sorted_pos[0] if sorted_pos else None
            if best and best.get("unrealized_pnl",0):
                upnl_b = best["unrealized_pnl"]
                report += f"\n🟢 Best: {best['question'][:60]} <b>{upnl_b:+.2f}$</b>\n"
            if worst and worst.get("unrealized_pnl",0) and worst != best:
                upnl_w = worst["unrealized_pnl"]
                report += f"🔴 Worst: {worst['question'][:60]} <b>{upnl_w:+.2f}$</b>\n"

        return report

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

            # Calibration buckets: predicted p_final (YES probability) vs actual YES outcome
            # p_final = probability of YES. Compare against actual YES outcome, not bet result.
            # YES outcome: side=YES+WIN or side=NO+LOSS means YES happened
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
                    ROUND(AVG(CASE
                        WHEN (side='YES' AND result='WIN') OR (side='NO' AND result='LOSS')
                        THEN 1.0 ELSE 0.0
                    END)::numeric, 3) as actual_wr
                FROM positions WHERE status='closed' AND outcome IN ('YES', 'NO')
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
                log.info(f"[DB] Cleanup: snapshots={r1}, signals={r2}, news={r3}")
            # VACUUM must run outside a transaction block
            conn = await self.pool.acquire()
            try:
                await conn.execute("VACUUM (ANALYZE) price_snapshots")
                await conn.execute("VACUUM (ANALYZE) signals")
                await conn.execute("VACUUM (ANALYZE) news")
                log.info("[DB] VACUUM done")
            except Exception as e:
                log.warning(f"[DB] VACUUM failed: {e}")
            finally:
                await self.pool.release(conn)
        except Exception as e:
            log.error(f"[DB] Cleanup failed: {e}")

    async def close(self):
        if self.pool:
            await self.pool.close()
            log.info("[DB] Connection pool closed")
