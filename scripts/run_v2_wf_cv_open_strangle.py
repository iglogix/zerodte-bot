"""Walk-forward cross-validation for the SPX 0DTE open strangle.

Sweep: strike_mode × (target_short_delta | em_mult) × tp_pct × tp_pct_end.
SL is per-trade-matched from data/ic_sl_dollars_by_day.csv (not swept).

Usage:
  python scripts/run_v2_wf_cv_open_strangle.py \\
      --config config/config_spx_0dte_open_strangle.yaml \\
      --ratio 5 --folds 4 --top-k 5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the IC runner's main() but swap in the strangle PARAM_GRID first.
import scripts.run_v2_wf_cv_open_ic as base_runner

EM_MULT_GRID = {
    "options.strike_mode": ["em_mult"],
    "options.em_mult": [0.20, 0.30, 0.40, 0.50, 0.75, 1.00],
    "options.tp_pct_of_credit": [0.25, 0.50, 0.75],
    "options.tp_pct_of_credit_end": [None, 0.75],
    "strategy.skip_fomc_next_day": [True],
    "strategy.skip_weekdays": [["Fri"]],
}
# 6 em_mults × 3 tp × 2 tp_end = 36 combos.

DELTA_GRID = {
    "options.strike_mode": ["delta"],
    "options.target_short_delta": [0.10, 0.15, 0.20, 0.25, 0.30],
    "options.tp_pct_of_credit": [0.25, 0.50, 0.75],
    "options.tp_pct_of_credit_end": [None, 0.75],
    "strategy.skip_fomc_next_day": [True],
    "strategy.skip_weekdays": [["Fri"]],
}
# 5 deltas × 3 tp × 2 tp_end = 30 combos.

# Total across both modes = 66, matching spec §4.


def main():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        choices=["em_mult", "delta"],
        default="em_mult",
        help="Strike-selection mode to sweep (run once per mode).",
    )
    # Parse only --mode here; let base_runner.main() handle the rest.
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    base_runner.PARAM_GRID = EM_MULT_GRID if args.mode == "em_mult" else DELTA_GRID
    print(
        f"Strangle WF-CV: sweeping {args.mode} mode "
        f"({len(base_runner.PARAM_GRID['options.' + ('em_mult' if args.mode == 'em_mult' else 'target_short_delta')])} strike configs)"
    )
    base_runner.main()


if __name__ == "__main__":
    main()
