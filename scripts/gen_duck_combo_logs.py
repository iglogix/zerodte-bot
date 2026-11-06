"""Pre-simulate Iron Duck parameter combos for the WF METHOD validation.

Writes one CSV per combo under results/iron_duck_wf/ in the format
run_wf_method_strategy.py consumes (status, session_date, realized_pnl_usd).
P&L is attributed to the exit date.

Grid (finalize per as-taught results before running the WF):
    put_short_delta x put_long_delta x stop_frac x call_short_delta

Usage:
    .venv/bin/python scripts/gen_duck_combo_logs.py --dte 7 --fill-k 1.0
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.iron_duck import (  # noqa: E402
    DuckParams,
    DuckStore,
    IronDuckEngine,
)

CACHE = "data/iron_duck_eod/SPXW"
OUT = Path("results/iron_duck_wf")


def iso(d: int) -> str:
    return dt.date(d // 10000, d // 100 % 100, d % 100).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--dtes", default="7,10,14")
    ap.add_argument("--fill-k", type=float, default=0.25)
    ap.add_argument("--exit-fill-k", type=float, default=1.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99999999)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--call-deltas", default="0.60,0.65,0.70")
    ap.add_argument("--put-shorts", default="0.10,0.15,0.20")
    ap.add_argument("--put-longs", default="0.05,0.10")
    ap.add_argument("--stops", default="1.0,2.0,none")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = DuckStore(args.cache)

    dtes = [int(x) for x in args.dtes.split(",")]
    csd = [float(x) for x in args.call_deltas.split(",")]
    psd = [float(x) for x in args.put_shorts.split(",")]
    pld = [float(x) for x in args.put_longs.split(",")]
    stops = [None if x == "none" else float(x) for x in args.stops.split(",")]

    grid = [
        (dte, c, ps, pl, st)
        for dte, c, ps, pl, st in itertools.product(dtes, csd, psd, pld, stops)
        if pl < ps
    ]
    print(f"{len(grid)} combos, dtes={dtes}, fill_k={args.fill_k}", flush=True)
    t0 = time.time()
    for i, (dte, c, ps, pl, st) in enumerate(grid, 1):
        st_tag = "none" if st is None else f"{st:g}"
        tag = f"d{dte}_c{c:g}_ps{ps:g}_pl{pl:g}_st{st_tag}"
        path = out_dir / f"combo_{tag}.csv"
        if path.exists():
            continue
        p = DuckParams(
            dte_target=dte,
            dte_min=max(1, dte - 3),
            dte_max=dte + 3,
            call_short_delta=c,
            put_short_delta=ps,
            put_long_delta=pl,
            stop_frac=st,
            fill_k=args.fill_k,
            exit_fill_k=args.exit_fill_k,
        )
        eng = IronDuckEngine(store, p)
        eng.run(args.start, args.end)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["status", "session_date", "realized_pnl_usd"])
            for t in eng.trades:
                if t.pnl_usd is None:
                    continue
                w.writerow(["closed", iso(t.exit_date), round(t.pnl_usd, 2)])
        el = (time.time() - t0) / 60
        print(
            f"[{i}/{len(grid)}] {tag} n={len(eng.trades)} "
            f"net=${sum(t.pnl_usd or 0 for t in eng.trades):,.0f} ({el:.1f}m)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
