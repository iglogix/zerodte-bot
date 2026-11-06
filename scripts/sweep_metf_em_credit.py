"""
METF EM% × credit-band parameter sweep on real cbbo_cache pricing.

For each (em_fixed_pct, credit_band) combo, runs the METF backtester end-to-end
and reports trades, win-rate, P&L, profit factor, skip-reason breakdown.

Bars are loaded once and re-used; cbbo_cache is loaded per-day inside METFBacktester.run().
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

from agentic_ic.strategies.metf.metf_backtester import (
    METFBacktestConfig,
    METFBacktester,
    METFResultsReporter,
    load_bars_via_bar_loader,
)


def run_combo(bars, em_pct: float, credit_min: float, credit_max: float,
              ema_min_sep: float, widths: List[int]) -> dict:
    cfg = METFBacktestConfig(
        use_cbbo=True,
        ema_min_separation_pct=ema_min_sep,
        em_method="fixed_pct",
        em_fixed_pct=em_pct,
        min_credit=credit_min,
        max_credit=credit_max,
        spread_widths=widths,
    )
    bt = METFBacktester(cfg)
    res = bt.run(bars)

    # By-entry breakdown
    entry_pnl = {n: 0.0 for n in range(1, 7)}
    entry_count = {n: 0 for n in range(1, 7)}
    for d in res.daily_results:
        for t in d.trades:
            entry_pnl[t.entry_number] += t.pnl
            entry_count[t.entry_number] += 1

    return {
        "em_pct": em_pct,
        "credit_min": credit_min,
        "credit_max": credit_max,
        "trades": res.total_trades,
        "win_rate": res.win_rate,
        "total_pnl": res.total_pnl,
        "profit_factor": res.profit_factor,
        "sharpe": res.sharpe_ratio,
        "max_dd": res.max_drawdown,
        "skip_credit_band": res.aggregate_skip_reasons.get("credit_band", 0),
        "skip_no_long_strike": res.aggregate_skip_reasons.get("no_long_strike", 0),
        "skip_no_short_quote": res.aggregate_skip_reasons.get("no_short_quote", 0),
        "skip_no_long_quote": res.aggregate_skip_reasons.get("no_long_quote", 0),
        "skip_no_strikes": res.aggregate_skip_reasons.get("no_strikes", 0),
        "skipped_days_no_cbbo": len(res.skipped_days_no_cbbo),
        "entry_pnl_1230": entry_pnl[1], "entry_n_1230": entry_count[1],
        "entry_pnl_1300": entry_pnl[2], "entry_n_1300": entry_count[2],
        "entry_pnl_1330": entry_pnl[3], "entry_n_1330": entry_count[3],
        "entry_pnl_1400": entry_pnl[4], "entry_n_1400": entry_count[4],
        "entry_pnl_1430": entry_pnl[5], "entry_n_1430": entry_count[5],
        "entry_pnl_1445": entry_pnl[6], "entry_n_1445": entry_count[6],
    }


def main():
    print("Loading bars once...")
    bars = load_bars_via_bar_loader(
        "data/spy_1m.csv",
        start_date="2025-09-01",
        end_date="2026-03-20",
        price_scale=10.0,
        symbol="SPX",
    )
    print(f"Bars loaded: {len(bars)} ({bars.index.normalize().nunique()} days)")

    # EMA min-separation: use 0.0003 (0.03%) so we have signal flow above noise
    EMA_MIN_SEP = 0.0003

    em_pcts = [0.003, 0.005, 0.0075, 0.010, 0.0125, 0.015]
    credit_bands = [
        (0.25, 5.00),   # very loose
        (0.50, 5.00),   # loose
        (0.75, 3.00),   # moderate
        (1.00, 3.00),   # tighter
        (1.25, 2.50),   # METF default
    ]
    widths_variants = [
        [20, 30, 35],   # default
        [10, 15, 20],   # tighter
    ]

    rows = []
    total = len(em_pcts) * len(credit_bands) * len(widths_variants)
    i = 0
    for widths in widths_variants:
        for em in em_pcts:
            for (cmin, cmax) in credit_bands:
                i += 1
                row = run_combo(bars, em, cmin, cmax, EMA_MIN_SEP, widths)
                row["widths"] = "/".join(str(w) for w in widths)
                rows.append(row)
                print(f"[{i:3d}/{total}] em={em*100:.2f}%  band=[{cmin:.2f},{cmax:.2f}]  "
                      f"widths={row['widths']:8s} -> trades={row['trades']:3d} "
                      f"wr={row['win_rate']*100:5.1f}%  pnl=${row['total_pnl']:>8.0f}  "
                      f"PF={row['profit_factor']:.2f}")

    # Sort: most trades first, then by P&L
    rows.sort(key=lambda r: (-r["trades"], -r["total_pnl"]))

    out = Path("outputs/metf_em_credit_sweep.csv")
    out.parent.mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} combos)")

    # Print top-10 by P&L (among combos with >=10 trades for meaningful stats)
    print("\nTop combos by P&L (n>=10 trades):")
    qualifying = [r for r in rows if r["trades"] >= 10]
    qualifying.sort(key=lambda r: -r["total_pnl"])
    print(f"{'EM%':>6} {'band':>14} {'widths':>10} {'n':>4} {'WR':>6} {'PF':>6} {'$PnL':>10} {'$MDD':>8}")
    for r in qualifying[:10]:
        print(f"{r['em_pct']*100:5.2f}%  [{r['credit_min']:.2f},{r['credit_max']:.2f}]  "
              f"{r['widths']:>10}  {r['trades']:>4d}  {r['win_rate']*100:5.1f}%  "
              f"{r['profit_factor']:5.2f}  ${r['total_pnl']:>8.0f}  ${r['max_dd']:>6.0f}")


if __name__ == "__main__":
    main()
