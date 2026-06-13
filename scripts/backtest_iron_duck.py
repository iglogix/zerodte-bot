"""Run the Iron Duck backtest over the SPXW EOD cache.

As-taught reproduction (instructor's Option Omega config) and variations:

    # the instructor's OO config at 7 DTE, mid fills
    .venv/bin/python scripts/backtest_iron_duck.py --dte 7

    # full as-taught sweep table (DTE x put-long-delta x fill model)
    .venv/bin/python scripts/backtest_iron_duck.py --sweep as_taught

    # custom single run
    .venv/bin/python scripts/backtest_iron_duck.py --dte 14 --put-long-delta 0.10 \
        --fill-k 1.0 --stop 1.0 --constraints

Outputs a summary line per config and writes per-trade CSVs under
results/iron_duck_spx/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.iron_duck import (  # noqa: E402
    DuckParams,
    DuckStore,
    IronDuckEngine,
    summarize,
)

CACHE = "data/iron_duck_eod/SPXW"
OUT = Path("results/iron_duck_spx")


def run_one(
    store: DuckStore, params: DuckParams, tag: str, start: int, end: int, write_csv: bool = True
) -> dict:
    eng = IronDuckEngine(store, params)
    eng.run(start, end)
    s = summarize(eng.trades, eng.daily_equity)
    s["tag"] = tag
    s["skips"] = len(eng.skips)
    if write_csv and eng.trades:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"trades_{tag}.csv"
        cols = [k for k in asdict(eng.trades[0]) if k != "daily_marks"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for t in eng.trades:
                w.writerow({k: v for k, v in asdict(t).items() if k != "daily_marks"})
    return s


def fmt(s: dict) -> str:
    if s.get("trades", 0) == 0:
        return f"{s.get('tag','?'):42s}  NO TRADES (skips={s.get('skips')})"
    return (
        f"{s['tag']:42s} n={s['trades']:4d} net=${s['net_pnl']:>10,.0f} "
        f"avg=${s['avg_pnl']:>7,.0f} win={s['win_rate']*100:4.1f}% "
        f"aw=${s['avg_win']:>6,.0f} al=${s['avg_loss']:>8,.0f} "
        f"dd=${s['max_dd_usd']:>9,.0f} margin=${s['max_margin_usd']:>8,.0f} "
        f"roc/yr={(s['roc_ann'] or 0)*100:6.1f}% "
        f"cred={s['avg_credit']:5.2f} pop={s['avg_pop']*100:4.1f}% "
        f"exits={s['exit_reasons']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99999999)
    ap.add_argument("--dte", type=int, default=7)
    ap.add_argument("--put-long-delta", type=float, default=0.05)
    ap.add_argument("--put-short-delta", type=float, default=0.15)
    ap.add_argument("--call-short-delta", type=float, default=0.65)
    ap.add_argument("--call-long-mode", default="next_strike", choices=["next_strike", "delta"])
    ap.add_argument("--fill-k", type=float, default=0.0)
    ap.add_argument(
        "--exit-fill-k",
        type=float,
        default=None,
        help="fill_k for stop/beak exits (default: same as --fill-k)",
    )
    ap.add_argument("--stop", type=float, default=1.0, help="stop fraction of credit; -1 = no stop")
    ap.add_argument("--beak-exit", type=float, default=None)
    ap.add_argument(
        "--constraints",
        action="store_true",
        help="enforce cheat-sheet gates (POP>=85, no upside risk, BE loss)",
    )
    ap.add_argument("--sweep", default=None, choices=[None, "as_taught", "fills"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    store = DuckStore(args.cache)

    def mk(
        dte: int,
        pld: float,
        fill_k: float,
        stop: float | None,
        constraints: bool,
        clm: str = "next_strike",
    ) -> DuckParams:
        return DuckParams(
            dte_target=dte,
            dte_min=max(1, dte - 3),
            dte_max=dte + 3,
            call_short_delta=args.call_short_delta,
            call_long_mode=clm,
            put_short_delta=args.put_short_delta,
            put_long_delta=pld,
            stop_frac=stop,
            beak_exit_prob=args.beak_exit,
            fill_k=fill_k,
            exit_fill_k=args.exit_fill_k,
            require_no_upside_risk=constraints,
            pop_min=0.85 if constraints else None,
            require_be_loss_lt_credit=constraints,
        )

    results = []
    if args.sweep == "as_taught":
        # instructor's OO table: DTE x {5d OO put, 10d cheat-sheet put} x fill model
        for fill_k, fk_tag in ((0.0, "mid"), (1.0, "nat")):
            for pld in (0.05, 0.10):
                for dte in (5, 7, 10, 14, 21):
                    tag = f"dte{dte}_pl{int(pld*100)}d_{fk_tag}"
                    p = mk(dte, pld, fill_k, 1.0, False)
                    results.append(run_one(store, p, tag, args.start, args.end))
                    print(fmt(results[-1]), flush=True)
    elif args.sweep == "fills":
        for fill_k in (0.0, 0.25, 0.5, 0.75, 1.0):
            tag = f"dte{args.dte}_fill{fill_k}"
            p = mk(
                args.dte,
                args.put_long_delta,
                fill_k,
                None if args.stop < 0 else args.stop,
                args.constraints,
            )
            results.append(run_one(store, p, tag, args.start, args.end))
            print(fmt(results[-1]), flush=True)
    else:
        stop = None if args.stop < 0 else args.stop
        tag = f"dte{args.dte}_pl{int(args.put_long_delta*100)}d" f"_f{args.fill_k}_s{args.stop}" + (
            "_con" if args.constraints else ""
        ) + (f"_beak{args.beak_exit}" if args.beak_exit else "")
        p = mk(
            args.dte, args.put_long_delta, args.fill_k, stop, args.constraints, args.call_long_mode
        )
        results.append(run_one(store, p, tag, args.start, args.end))
        print(fmt(results[-1]), flush=True)

    if args.json:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
