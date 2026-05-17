"""Sweep asymmetric IC: vary call-side em_mult while keeping put-side fixed."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analytical_ic_eval import _load_data, _evaluate


def main():
    by_date, _ = _load_data(
        Path("data/cbbo_cache_5anchor"), "data/spy_1m.csv",
        date(2022, 4, 18), date(2026, 2, 28),
        anchors=["09:35", "11:00", "13:00", "15:30", "15:59"],
    )
    print(f"Days: {len(by_date)}\n")
    anchors = ["09:35", "11:00", "13:00", "15:30", "15:59"]

    wing, tp, sl = 30, 0.50, 2.0
    print(f"Config: wing={wing} TP={tp}×credit SL={sl}×credit  (no extra filters)\n")
    print(f"{'put_em':>7} {'call_em':>8} {'n':>4} {'WR%':>6} {'total':>10} "
          f"{'Sharpe':>7} {'max_L':>5} {'max_DD':>8} {'avg_W':>7} {'avg_L':>7}")
    print("─" * 90)

    # Sweep symmetric reference + call-skew variants
    configs = [
        (0.20, 0.20),  # symmetric reference
        (0.20, 0.25),
        (0.20, 0.30),
        (0.20, 0.35),
        (0.20, 0.40),
        (0.20, 0.50),
        (0.20, 0.60),
        (0.20, 0.80),
        # Also try put-skew (in case put losses dominate in another regime)
        (0.25, 0.20),
        (0.30, 0.20),
        # Pure symmetric variations for context
        (0.25, 0.25),
        (0.30, 0.30),
    ]
    for put_em, call_em in configs:
        r = _evaluate(
            em_mult=put_em, wing_width=wing, tp_pct=tp, sl_mult=sl,
            anchors=anchors, by_date=by_date,
            em_mult_put=put_em, em_mult_call=call_em,
        )
        if r["n"] == 0:
            print(f"{put_em:>7.2f} {call_em:>8.2f}  NO TRADES")
            continue
        print(f"{put_em:>7.2f} {call_em:>8.2f} {r['n']:>4d} {r['wr']:>5.1f}% "
              f"${r['total']:>+9,.0f} {r['sharpe_ann']:>+7.2f} "
              f"{r['max_consec_L']:>5d} ${r['max_dd']:>+7,.0f} "
              f"${r['avg_win']:>+5.0f} ${r['avg_loss']:>+5.0f}")


if __name__ == "__main__":
    main()
