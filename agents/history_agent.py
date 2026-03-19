import math
import logging
import numpy as np
from collections import defaultdict

log = logging.getLogger("history")

class HistoryAgent:
    def __init__(self, db, calibrator):
        self.db         = db
        self.calibrator = calibrator

    async def analyze(self):
        log.info("[HISTORY] Анализируем исторические данные...")
        closed_markets   = await self.db.get_closed_markets(limit=1000)
        closed_positions = await self.db.get_closed_positions(limit=500)

        if not closed_markets:
            log.info("[HISTORY] Нет данных пока")
            return

        await self._compute_base_rates(closed_markets)
        await self._compute_volume_patterns(closed_markets)

        if len(closed_positions) >= 10:
            await self._calibrate_system(closed_positions)

        log.info("[HISTORY] ✅ Анализ завершён")

    async def _compute_base_rates(self, markets: list):
        by_theme = defaultdict(list)
        for m in markets:
            if m.get("outcome"):
                by_theme[m["theme"]].append(1 if m["outcome"]=="YES" else 0)

        for theme, outcomes in by_theme.items():
            if len(outcomes) < 5: continue
            arr       = np.array(outcomes)
            base_rate = float(np.mean(arr))

            market_prices = [
                m["yes_price"] for m in markets
                if m["theme"]==theme and m.get("outcome") and m.get("yes_price")
            ]
            avg_mp         = float(np.mean(market_prices)) if market_prices else 0.5
            prospect_factor = max(0.3, min(3.0, base_rate/avg_mp if avg_mp>0 else 1.0))

            await self.db.upsert_pattern(theme, {
                "base_rate":       round(base_rate, 4),
                "sample_size":     len(outcomes),
                "prospect_factor": round(prospect_factor, 4),
                "win_rate":        round(base_rate, 4),
            })
            log.info(f"[HISTORY] {theme}: base_rate={base_rate:.3f} n={len(outcomes)} pf={prospect_factor:.3f}")

    async def _compute_volume_patterns(self, markets: list):
        volumes = [m["volume"] for m in markets if m.get("volume")]
        if not volumes: return
        median_vol  = float(np.median(volumes))
        high_vol_out, low_vol_out = [], []

        for m in markets:
            if not m.get("outcome") or not m.get("volume"): continue
            outcome = 1 if m["outcome"]=="YES" else 0
            if m["volume"] > median_vol*2: high_vol_out.append(outcome)
            else: low_vol_out.append(outcome)

        if high_vol_out and low_vol_out:
            high_rate  = float(np.mean(high_vol_out))
            low_rate   = float(np.mean(low_vol_out))
            vol_signal = high_rate/low_rate if low_rate>0 else 1.0
            log.info(f"[HISTORY] Volume: high={high_rate:.3f} low={low_rate:.3f} ratio={vol_signal:.3f}")
            await self.db.upsert_pattern("volume_signal", {
                "base_rate": round(high_rate,4), "sample_size": len(high_vol_out),
                "volume_signal": round(vol_signal,4), "win_rate": round(high_rate,4),
            })

    async def _calibrate_system(self, positions: list):
        preds    = np.array([float(p.get("p_final",0.5)) for p in positions])
        outcomes = np.array([1.0 if p.get("result")=="WIN" else 0.0 for p in positions])

        if len(preds) < 5: return

        brier  = float(np.mean((preds-outcomes)**2))
        bias   = float(np.mean(preds-outcomes))
        factor = max(0.7, min(1.3, 1.0-bias*0.5))

        await self.db.save_calibration("final", brier, bias, factor, len(preds))
        self.calibrator.update_from_history("final", brier, bias, factor)
        log.info(f"[HISTORY] Calibration: Brier={brier:.4f} bias={bias:+.4f} factor={factor:.4f}")
