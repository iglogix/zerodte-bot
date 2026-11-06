"""Markov regime entry gate — causal day-level pre-check for entry engines.

Labels the underlying's trailing `window`-session cumulative return as
BULL (>= bull_threshold) / BEAR (<= bear_threshold) / SIDEWAYS, both at the
prior close and one window earlier, and blocks the session on the
WF-CV-validated rule (results/markov_gate_wf_cv_2026-06-11.txt):

  excl_stale_sw  block iff SIDEWAYS now AND SIDEWAYS one window ago
                 ("stale sideways" — the regime where Power Hour bleeds)
  non_sw         block iff SIDEWAYS now (stricter; kept for experiments)

Inputs are prior-session daily closes only, so the decision is identical at
every entry window of the day and fully causal. On insufficient data the gate
fails OPEN (consistent with the LLM gates): a missing feed must degrade to
baseline behaviour, never to a silent permanent skip.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

SIDEWAYS, BULL, BEAR = "SIDEWAYS", "BULL", "BEAR"


class MarkovGate:

    def __init__(self, config: Dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.window = int(config.get("window", 20))
        self.bull_thr = float(config.get("bull_threshold", 0.05))
        self.bear_thr = float(config.get("bear_threshold", -0.05))
        self.rule = config.get("rule", "excl_stale_sw")
        if self.rule not in ("excl_stale_sw", "non_sw"):
            raise ValueError(f"unknown markov_gate rule: {self.rule}")

    @property
    def min_closes(self) -> int:
        return 2 * self.window + 1

    def _state(self, ret: float) -> str:
        if ret >= self.bull_thr:
            return BULL
        if ret <= self.bear_thr:
            return BEAR
        return SIDEWAYS

    def evaluate(self, closes: Sequence[float]) -> Dict[str, Any]:
        """Decide from daily closes (oldest -> newest, last = prior session).

        Returns {"allow": bool, "reason": str, ...} with the returns and
        states the decision was made from, for the decision log.
        """
        closes = [c for c in closes if c and c > 0]
        if len(closes) < self.min_closes:
            return {
                "allow": True,
                "reason": (f"insufficient_data: {len(closes)} closes < "
                           f"{self.min_closes} required — failing open"),
            }
        w = self.window
        ret_now = closes[-1] / closes[-1 - w] - 1.0
        ret_prev = closes[-1 - w] / closes[-1 - 2 * w] - 1.0
        state_now, state_prev = self._state(ret_now), self._state(ret_prev)

        if self.rule == "non_sw":
            allow = state_now != SIDEWAYS
            why = f"state={state_now}"
        else:
            allow = not (state_now == SIDEWAYS and state_prev == SIDEWAYS)
            why = f"stale_sideways={not allow}"

        return {
            "allow": allow,
            "reason": (f"{why} (now={state_now} {ret_now:+.2%}, "
                       f"prev={state_prev} {ret_prev:+.2%}, window={w})"),
            "state_now": state_now,
            "state_prev": state_prev,
            "ret_now": ret_now,
            "ret_prev": ret_prev,
        }
