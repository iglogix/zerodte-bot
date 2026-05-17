"""Test individual + combined filters on the analytical IC evaluator."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.vix_loader import load_vix_series
from scripts.analytical_ic_eval import _load_data, _evaluate


def main():
    by_date, miss = _load_data(
        Path("data/cbbo_cache_5anchor"), "data/spy_1m.csv",
        date(2022, 4, 18), date(2026, 2, 28),
        anchors=["09:35", "11:00", "13:00", "15:30", "15:59"],
    )
    print(f"Days: {len(by_date)}")
    vix = load_vix_series("data/vix9d_daily.csv")

    em, w, tp, sl = 0.20, 30, 0.50, 2.0
    anchors = ["09:35", "11:00", "13:00", "15:30", "15:59"]

    configs = [
        ("Baseline (no filters)", {}),
        ("+ skip_cpi_next_day", {"skip_cpi_next_day": True}),
        ("+ skip_fomc_next_day", {"skip_fomc_next_day": True}),
        ("+ skip both cpi/fomc next", {"skip_cpi_next_day": True, "skip_fomc_next_day": True}),
        ("+ trend skip σ>1.5", {"skip_trend_sigma": 1.5}),
        ("+ trend skip σ>2.0", {"skip_trend_sigma": 2.0}),
        ("+ low_em (today<80% of 5d avg)", {"low_em_pct": 0.80}),
        ("+ low_em (today<70% of 5d avg)", {"low_em_pct": 0.70}),
        ("+ vix9d > 25", {"vix9d_max": 25, "vix_series": vix}),
        ("+ vix9d > 30", {"vix9d_max": 30, "vix_series": vix}),
        ("ALL filters (cpi+fomc+trend1.5+lowem0.8+vix30)", {
            "skip_cpi_next_day": True, "skip_fomc_next_day": True,
            "skip_trend_sigma": 1.5, "low_em_pct": 0.80,
            "vix9d_max": 30, "vix_series": vix,
        }),
        ("ALL aggressive (cpi+fomc+trend2.0+lowem0.7+vix25)", {
            "skip_cpi_next_day": True, "skip_fomc_next_day": True,
            "skip_trend_sigma": 2.0, "low_em_pct": 0.70,
            "vix9d_max": 25, "vix_series": vix,
        }),
    ]

    print(f"\nConfig: em={em} wing={w} TP={tp}×credit SL={sl}×credit anchors=hourly\n")
    print(f"{'Filter set':<48} {'n':>4} {'WR%':>6} {'total':>10} "
          f"{'Sharpe':>7} {'max_L':>5} {'max_DD':>8} {'skipped'}")
    print("─" * 110)
    for label, filt in configs:
        r = _evaluate(em, w, tp, sl, anchors, by_date, filters=filt)
        if r["n"] == 0:
            print(f"{label:<48}  NO TRADES")
            continue
        print(f"{label:<48} {r['n']:>4d} {r['wr']:>5.1f}% "
              f"${r['total']:>+9,.0f} {r['sharpe_ann']:>+7.2f} "
              f"{r['max_consec_L']:>5d} ${r['max_dd']:>+7,.0f} {r.get('skips', {})}")


if __name__ == "__main__":
    main()
