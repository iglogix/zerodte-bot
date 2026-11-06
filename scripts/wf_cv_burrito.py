"""Walk-forward CV over the Burrito param grid, maximizing OOS Sharpe.

Splits the 2020-2026 history into rolling folds: train+validation years -> next
(test) year. Selects the param cell by VALIDATION Sharpe (min-trade gated), then
reports TEST (OOS) metrics per fold — never selecting on test. Also runs a
slippage stress grid on the most-recently-selected cell.

Usage: .venv/bin/python scripts/wf_cv_burrito.py --folds 4
"""
import argparse
import itertools
import json
import os
import time

from zerodte.backtest.iron_duck import DuckStore
from zerodte.backtest.burrito_core import BurritoConfig, run_history, summarize_burrito

# Focused grid — the levers that matter. Wider stop/lock bands than the deck's
# tight 5-10% because EOD marks are noisy; hold-to-expiry is the wide limit.
# 'both' dropped from the sweep: the structural sweep showed the directional call
# barely moves WR (hull 18.3% vs both 18.0%) and it doubles runtime.
GRID = {
    "entry_dte": [1, 3],
    "body_offset": [0.5, 0.75],
    "direction_signal": ["hull", "momentum"],
    "lock_trigger_pct": [0.25, 0.5],
    "stop_pct": [0.5, 1.0],
    "pt_pct": [0.5],
    "defense_policy": [(), ("clawback",)],
}
MIN_TRADES = 30


def cells():
    keys = list(GRID)
    for combo in itertools.product(*[GRID[k] for k in keys]):
        yield dict(zip(keys, combo))


def evaluate(store, params, start, end, slippage=1.0):
    cfg = BurritoConfig(slippage_k=slippage, gates_on=True, **params)
    return summarize_burrito(run_history(store, cfg, start, end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/iron_duck_eod/SPXW")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out", default="results/burrito/wfcv.json")
    a = ap.parse_args()
    store = DuckStore(a.cache)
    qd = store.quote_dates()
    years = sorted({d // 10000 for d in qd})
    t0 = time.time()

    grid = list(cells())
    print(f"grid cells: {len(grid)}, years: {years}")
    results = []
    # rolling: need >=2 train/val years before the first test year
    start_k = max(2, len(years) - a.folds)
    for k in range(start_k, len(years)):
        tv_years = years[:k]
        test_y = years[k]
        # split train/val 70/30 by time; validate on the most recent val years
        split_idx = max(1, int(len(tv_years) * 0.7))
        val_start = tv_years[split_idx] * 10000 + 101
        val_end = tv_years[-1] * 10000 + 1231
        best, best_sharpe, best_val = None, -1e9, None
        for params in grid:
            v = evaluate(store, params, val_start, val_end)
            s = v.get("sharpe", -1e9) if v.get("trades", 0) >= MIN_TRADES else -1e9
            if s > best_sharpe:
                best_sharpe, best, best_val = s, params, v
        oos = evaluate(store, best, test_y * 10000 + 101, test_y * 10000 + 1231)
        results.append({"test_year": test_y, "selected": best,
                        "val_sharpe": round(best_sharpe, 3), "val": best_val, "oos": oos})
        print(f"  fold test={test_y}: val_sharpe={best_sharpe:.2f} "
              f"OOS sharpe={oos.get('sharpe')} OOS trades={oos.get('trades')} sel={best}",
              flush=True)

    stress = {}
    if results:
        sel = results[-1]["selected"]
        for sk in (1.0, 1.25, 1.5, 2.0):
            stress[str(sk)] = evaluate(store, sel, years[0] * 10000 + 101,
                                       years[-1] * 10000 + 1231, sk)
            print(f"  slippage {sk}: sharpe={stress[str(sk)].get('sharpe')} "
                  f"net={stress[str(sk)].get('net_pnl')}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"grid_size": len(grid), "folds": results,
                   "slippage_stress": stress}, f, indent=2, default=str)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
