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

        # Get entry prices from our signals (p_market at time of signal, not final price)
        signal_prices = await self.db.get_signal_prices_by_theme()

        for theme, outcomes in by_theme.items():
            if len(outcomes) < 5: continue
            arr       = np.array(outcomes)
            base_rate = float(np.mean(arr))

            # prospect_factor: how much does the market misprice this theme?
            # base_rate = actual P(YES) for theme, avg_entry = what market was pricing at signal time
            # pf > 1 = market underprices YES, pf < 1 = market overprices YES
            avg_entry = signal_prices.get(theme)
            if avg_entry and avg_entry > 0:
                prospect_factor = max(0.3, min(3.0, base_rate / avg_entry))
            else:
                prospect_factor = 1.0  # no signal data yet, neutral

            await self.db.upsert_pattern(theme, {
                "base_rate":       round(base_rate, 4),
                "sample_size":     len(outcomes),
                "prospect_factor": round(prospect_factor, 4),
                "win_rate":        round(base_rate, 4),
            })
            log.info(f"[HISTORY] {theme}: base_rate={base_rate:.3f} n={len(outcomes)} pf={prospect_factor:.3f} avg_entry={avg_entry or 'N/A'}")

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
        # Use actual market outcome (YES/NO), not bet result (WIN/LOSS)
        # p_final is our predicted P(YES), outcome is whether YES happened
        preds, actuals = [], []
        for p in positions:
            outcome = p.get("outcome", "")
            if not outcome:
                continue
            p_final = float(p.get("p_final", 0.5))
            # Only use RESOLVED positions for calibration
            # TP/SL outcomes like "YES@65¢" don't tell us the true outcome
            if outcome == "YES":
                actual = 1.0
            elif outcome == "NO":
                actual = 0.0
            else:
                continue  # skip TP/SL — we don't know the true outcome
            preds.append(p_final)
            actuals.append(actual)

        if len(preds) < 5: return

        preds   = np.array(preds)
        actuals = np.array(actuals)
        brier  = float(np.mean((preds - actuals)**2))
        bias   = float(np.mean(preds - actuals))
        factor = max(0.7, min(1.3, 1.0 - bias * 0.5))

        await self.db.save_calibration("final", brier, bias, factor, len(preds))
        self.calibrator.update_from_history("final", brier, bias, factor)
        log.info(f"[HISTORY] Calibration: Brier={brier:.4f} bias={bias:+.4f} factor={factor:.4f}")
