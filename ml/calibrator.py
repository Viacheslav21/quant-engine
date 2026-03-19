import math
import logging
import numpy as np

log = logging.getLogger("calibrator")

class Calibrator:
    def __init__(self, db, window: int = 100):
        self.db      = db
        self.window  = window
        self.factor  = 1.0
        self.bias    = 0.0
        self._brier  = 0.25
        self._min_bets = 10
        self._agent_factors: dict = {}

    async def update(self):
        closed = await self.db.get_closed_positions(limit=self.window)
        if len(closed) < self._min_bets:
            log.info(f"[CALIBRATOR] Skipped: {len(closed)} positions < {self._min_bets} minimum")
            return
        preds    = np.array([float(p.get("p_final",0.5)) for p in closed])
        outcomes = np.array([1.0 if p.get("result")=="WIN" else 0.0 for p in closed])
        self._brier = float(np.mean((preds-outcomes)**2))
        self.bias   = float(np.mean(preds-outcomes))
        if abs(self.bias) > 0.05:
            self.factor = max(0.7, min(1.3, 1.0-self.bias*0.5))
        else:
            self.factor = 1.0
        log.info(f"[CALIBRATOR] Brier:{self._brier:.4f} Bias:{self.bias:+.4f} Factor:{self.factor:.4f} [{self.quality()}]")

    def update_from_history(self, agent: str, brier: float, bias: float, factor: float):
        log.debug(f"[CALIBRATOR] Updated from history: agent={agent} brier={brier:.4f} bias={bias:+.4f} factor={factor:.4f}")
        self._agent_factors[agent] = factor
        if agent == "final":
            self.factor = factor
            self.bias   = bias
            self._brier = brier

    def adjust(self, p_raw: float) -> float:
        if self._brier >= 0.25:
            return p_raw
        try:
            logit     = math.log(max(p_raw,0.001) / (1-min(p_raw,0.999)+1e-9))
            logit_adj = logit * self.factor - self.bias
            return max(0.02, min(0.98, 1/(1+math.exp(-logit_adj))))
        except Exception as e:
            log.warning(f"[CALIBRATOR] adjust() failed for p_raw={p_raw}: {e}")
            return p_raw

    def get_score(self) -> float:
        return self._brier

    def quality(self) -> str:
        if self._brier < 0.10: return "EXCELLENT 🟢"
        if self._brier < 0.15: return "GOOD 🟡"
        if self._brier < 0.20: return "FAIR 🟠"
        return "POOR 🔴"
