"""
Apples-to-apples comparison: whole-IC vs per-side management on the same
multi-anchor CBBO data, same anchors, same configs.

Both exit at the 4 intraday anchors (11:00, 13:00, 15:30, 15:59) + expiry.
The only difference is whole-IC (combined mark) vs per-side (each vertical
managed independently — the "leg-out" model).
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analytical_ic_eval import _load_data, _evaluate


def main():
    anchors = ["09:35", "11:00", "13:00", "15:30", "15:59"]
    by_date, _ = _load_data(
        Path("data/cbbo_cache_5anchor"), "data/spy_1m.csv",
        date(2022, 4, 18), date(2026, 2, 28), anchors,
    )
    print(f"Days: {len(by_date)}  (same data for both modes)\n")

    # Test configs spanning the sweet spot
    configs = [
        (0.20, 25, 0.50, 2.0),
        (0.20, 30, 0.50, 2.0),
        (0.25, 20, 0.50, 2.0),
        (0.25, 25, 0.50, 2.0),
        (0.30, 25, 0.50, 2.0),
        # Also test a tighter SL (where per-side stop bites harder)
        (0.25, 25, 0.50, 1.5),
        (0.20, 30, 0.50, 1.5),
    ]

    for manage in ("total", "per_side"):
        print(f"═══ Management: {manage} ═══")
        print(f"{'em':>5} {'wing':>5} {'tp':>5} {'sl':>5} {'n':>4} {'WR%':>6} "
              f"{'total':>11} {'Sharpe':>7} {'maxL':>5} {'maxDD':>9} {'avgW':>7} {'avgL':>7}")
        print("─" * 95)
        for em, w, tp, sl in configs:
            r = _evaluate(em, w, tp, sl, anchors, by_date, manage=manage)
            if r["n"] == 0:
                continue
            print(f"{em:>5.2f} {w:>5.0f} {tp:>5.2f} {sl:>5.1f} {r['n']:>4d} "
                  f"{r['wr']:>5.1f}% ${r['total']:>+10,.0f} {r['sharpe_ann']:>+7.2f} "
                  f"{r['max_consec_L']:>5d} ${r['max_dd']:>+8,.0f} "
                  f"${r['avg_win']:>+5.0f} ${r['avg_loss']:>+5.0f}")
        print()


if __name__ == "__main__":
    main()
