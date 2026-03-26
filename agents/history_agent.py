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
        await self._compute_theme_performance(closed_positions)

        if len(closed_positions) >= 10:
            await self._calibrate_system(closed_positions)

        await self._update_dma_weights()

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

    async def _compute_theme_performance(self, positions: list):
        """Bayesian shrinkage per-theme: compute kelly_mult and ev_mult from trade history."""
        if len(positions) < 10:
            return

        SHRINKAGE_K = 20  # strength of pull toward global mean

        # Global stats
        global_wins = sum(1 for p in positions if p.get("result") == "WIN")
        global_wr = global_wins / len(positions) if positions else 0.5
        global_staked = sum(float(p.get("stake_amt", 0)) for p in positions)
        global_pnl = sum(float(p.get("pnl", 0)) for p in positions)
        global_roi = global_pnl / global_staked if global_staked > 0 else 0

        # Per-theme stats
        by_theme = defaultdict(list)
        for p in positions:
            theme = p.get("theme", "other")
            by_theme[theme].append(p)

        for theme, trades in by_theme.items():
            n = len(trades)
            wins = sum(1 for t in trades if t.get("result") == "WIN")
            raw_wr = wins / n if n > 0 else 0.5
            staked = sum(float(t.get("stake_amt", 0)) for t in trades)
            pnl = sum(float(t.get("pnl", 0)) for t in trades)
            roi = pnl / staked if staked > 0 else 0

            # Bayesian shrinkage: w_adj = (n * w_theme + k * w_global) / (n + k)
            # Few trades → trust global. Many trades → trust theme.
            wr_adj = (n * raw_wr + SHRINKAGE_K * global_wr) / (n + SHRINKAGE_K)
            roi_adj = (n * roi + SHRINKAGE_K * global_roi) / (n + SHRINKAGE_K)

            # Kelly multiplier: theme outperforms → bigger bets
            # Based on adjusted WR relative to global
            kelly_mult = wr_adj / global_wr if global_wr > 0 else 1.0
            kelly_mult = round(max(0.3, min(2.0, kelly_mult)), 3)

            # EV threshold multiplier: bad themes need higher EV to enter
            # Based on combined WR and ROI signal
            roi_factor = 1.0
            if roi_adj < -0.02:  # theme is losing money
                roi_factor = 1.0 + abs(roi_adj) * 3  # e.g. -5% ROI → 1.15x EV required
            elif roi_adj > 0.02:  # theme is making money
                roi_factor = max(0.7, 1.0 - roi_adj * 2)  # e.g. +10% ROI → 0.8x EV required

            ev_mult = round(max(0.7, min(2.0, (global_wr / wr_adj) * roi_factor)) if wr_adj > 0 else 1.5, 3)

            # Update pattern with trade performance
            existing = await self.db.get_patterns()
            pat = existing.get(theme, {})
            await self.db.upsert_pattern(theme, {
                "base_rate":       pat.get("base_rate", 0.5),
                "sample_size":     pat.get("sample_size", 0),
                "prospect_factor": pat.get("prospect_factor", 1.0),
                "win_rate":        pat.get("win_rate", 0.5),
                "trade_n":         n,
                "trade_wr":        round(wr_adj, 4),
                "trade_roi":       round(roi_adj, 4),
                "kelly_mult":      kelly_mult,
                "ev_mult":         ev_mult,
            })
            log.info(
                f"[HISTORY] Theme perf: {theme} | {n} trades WR:{raw_wr*100:.0f}%→{wr_adj*100:.0f}%(adj) "
                f"ROI:{roi*100:+.1f}% | kelly_mult:{kelly_mult:.2f} ev_mult:{ev_mult:.2f}"
            )

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

    # ── DMA: Dynamic Model Averaging ──

    # Sources tracked: map from trade_log details key → DMA source name
    DMA_SOURCES = {
        "p_history":    "history",
        "p_momentum":   "momentum",
        "p_long_mom":   "long_momentum",
        "p_contrarian": "contrarian",
        "p_vol_trend":  "volume",
        "p_arb":        "arb",
        "p_book":       "book",     # stored in details as p_book (from SIGNAL_GENERATED)
        "p_flb":        "crowd",    # FLB is part of crowd_combined
    }
    FORGETTING_FACTOR = 0.97

    async def _update_dma_weights(self):
        """Recalculate DMA weights from closed positions + their signal source probabilities."""
        try:
            positions = await self.db.get_closed_positions_with_signals(limit=200)
        except Exception as e:
            log.warning(f"[DMA] Failed to fetch positions: {e}")
            return

        if len(positions) < 20:
            log.info(f"[DMA] Not enough closed positions ({len(positions)}), need 20+")
            return

        # Load current weights or init
        current = await self.db.get_dma_weights()
        weights = {}
        for src in self.DMA_SOURCES.values():
            weights[src] = {
                "weight": current.get(src, 1.0),
                "hits": 0,
                "misses": 0,
                "likelihoods": [],
            }

        # Process each closed position
        processed = 0
        for pos in positions:
            details = pos.get("details") or {}
            if isinstance(details, str):
                import json
                try:
                    details = json.loads(details)
                except Exception:
                    continue

            side = pos.get("side", "YES")
            result = pos.get("result")
            if result not in ("WIN", "LOSS"):
                continue

            won = result == "WIN"

            for detail_key, src_name in self.DMA_SOURCES.items():
                p_val = details.get(detail_key)
                if p_val is None:
                    continue
                p_val = float(p_val)
                if p_val <= 0 or p_val >= 1:
                    continue

                # Did this source predict the correct direction?
                # For YES side: p > 0.5 means source predicted YES → correct if won
                # For NO side: p < 0.5 means source predicted NO → correct if won
                if side == "YES":
                    source_said_yes = p_val > 0.5
                else:
                    source_said_yes = p_val < 0.5  # low p_yes = predicted NO

                correct = (source_said_yes and won) or (not source_said_yes and not won)

                # Likelihood: how confident was the source in the correct direction
                if correct:
                    likelihood = abs(p_val - 0.5) * 2  # 0.5→0, 1.0→1.0
                    weights[src_name]["hits"] += 1
                else:
                    likelihood = 1.0 - abs(p_val - 0.5) * 2
                    weights[src_name]["misses"] += 1

                weights[src_name]["likelihoods"].append(max(0.01, likelihood))

            processed += 1

        if processed < 10:
            return

        # Apply forgetting factor + likelihood update
        alpha = self.FORGETTING_FACTOR
        for src_name, data in weights.items():
            if not data["likelihoods"]:
                continue
            avg_l = sum(data["likelihoods"]) / len(data["likelihoods"])
            # DMA update: w_new = w_old^α × avg_likelihood
            old_w = data["weight"]
            data["weight"] = (old_w ** alpha) * (avg_l + 0.5)  # +0.5 to keep weights centered around 1.0
            data["avg_likelihood"] = round(avg_l, 4)

        # Normalize: mean weight = 1.0
        all_weights = [d["weight"] for d in weights.values() if d["weight"] > 0]
        if all_weights:
            mean_w = sum(all_weights) / len(all_weights)
            if mean_w > 0:
                for data in weights.values():
                    data["weight"] = round(max(0.3, min(2.0, data["weight"] / mean_w)), 4)

        # Save to DB
        await self.db.save_dma_weights(weights)

        # Log
        for src_name, data in sorted(weights.items(), key=lambda x: -x[1]["weight"]):
            total = data["hits"] + data["misses"]
            wr = data["hits"] / total * 100 if total > 0 else 0
            log.info(
                f"[DMA] {src_name:12s} w={data['weight']:.2f} | "
                f"{data['hits']}H/{data['misses']}M ({wr:.0f}%) | L={data.get('avg_likelihood', 0):.3f}"
            )
