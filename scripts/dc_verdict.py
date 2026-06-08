"""DC Time Machine — verdict report generator.

Reads a multi-expiration cache and runs the simulator across the two axes that
decide whether the "risk-free transform" is actually profitable:

  * delta  — higher delta ⇒ richer front verticals ⇒ the DC banks the wing with
             less appreciation (more transforms), BUT a narrower risk-free tent
             ⇒ lower chance of max profit (price stays inside).
  * fill_k — 0 = mid (optimistic), 1 = full bid/ask cross (conservative). High-vol
             spreads are wide, so the edge can live entirely in this knob.

For each (delta, k) it reports transform%, max-profit% (of settled transforms),
win%, profit factor, and NET P&L — the number that integrates the tradeoff.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/dc_verdict.py --cache data/dc_cache_full \
        --extra data/dc_cache_hivol --md docs/.../verdict.md
"""
from __future__ import annotations

import argparse
import glob
from datetime import date
from typing import Dict, List, Optional

from src_v2.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from src_v2.backtest.dc_time_machine.simulator import run_session
from src_v2.backtest.dc_time_machine.strategy import DcParams


def _sessions(cache: str) -> List[date]:
    return sorted(date.fromisoformat(p.split("/")[-1][:-4]) for p in glob.glob(f"{cache}/*.pkl"))


def summarize(rows) -> Dict:
    real = [r for r in rows if r is not None and r.outcome != "no_entry"]
    tr = [r for r in real if r.outcome == "transformed"]
    wins = [r.pnl for r in real if r.pnl > 0]
    losses = [r.pnl for r in real if r.pnl < 0]
    gw, gl = sum(wins), abs(sum(losses))
    settled = [r for r in tr if r.settlement is not None]
    # "max profit" = landed inside the tent ⇒ terminal pnl == banked (C-D)*100
    maxp = [r for r in settled if abs(r.pnl - r.bank * 100) < 1.0]
    return {
        "n": len(real), "transformed": len(tr),
        "tr_rate": len(tr) / len(real) if real else 0.0,
        "win_rate": len(wins) / len(real) if real else 0.0,
        "total": sum(r.pnl for r in real),
        "pf": gw / gl if gl else float("inf"),
        "maxp_rate": len(maxp) / len(settled) if settled else 0.0,
        "settled": len(settled),
    }


def run_grid(store, sessions, lo=None, hi=None, front_dte=9,
             deltas=(0.30, 0.35, 0.40, 0.45), ks=(0.0, 0.5, 1.0), filt="always"):
    sub = [s for s in sessions if (lo is None or s >= lo) and (hi is None or s <= hi)]
    out = {}
    for dlt in deltas:
        for k in ks:
            p = DcParams(front_dte=front_dte, fill_k=k, target_delta=dlt, entry_filter=filt)
            out[(dlt, k)] = summarize([run_session(store, p, s) for s in sub])
    return out, len(sub)


def _fmt(s: Dict) -> str:
    pf = "inf " if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"n={s['n']:3d} tr%={s['tr_rate']*100:3.0f} maxP%={s['maxp_rate']*100:3.0f} "
            f"win%={s['win_rate']*100:3.0f} PF={pf:>5} NET=${s['total']:8.0f}")


def _table(grid, deltas, ks, P):
    P("```")
    P(f"{'':12s} " + "  ".join(f"k={k:<28.2f}" for k in ks))
    for dlt in deltas:
        for i, k in enumerate(ks):
            tag = f"delta={dlt:.2f}" if i == 0 else " " * 11
            P(f"{tag}  {_fmt(grid[(dlt, k)])}")
        P("")
    P("```")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--md", default=None)
    ap.add_argument("--front-dte", type=int, default=9)
    args = ap.parse_args()

    merged: Dict[date, dict] = {}
    for c in [args.cache] + args.extra:
        st = MultiExpiryStore(cache_dir=c)
        for s in _sessions(c):
            st.load_session(s)
            merged.setdefault(s, st._panel)
    store = MultiExpiryStore(preloaded=merged)
    sessions = sorted(merged)
    if not sessions:
        print("No sessions.")
        return

    lines: List[str] = []
    def P(x=""):
        print(x); lines.append(x)

    deltas = (0.30, 0.35, 0.40, 0.45)
    ks = (0.0, 0.5, 1.0)

    P("# DC Time Machine — backtest verdict\n")
    P(f"Cache(s): {', '.join([args.cache] + args.extra)}")
    P(f"Sessions: {len(sessions)}  [{sessions[0]} .. {sessions[-1]}], front_dte={args.front_dte}\n")
    P("Each cell: n trades · tr%=transform rate · maxP%=max-profit hit (of settled "
      "transforms) · win% · PF · NET $ total. k=0 mid → k=1 full spread cross.\n")

    P("## All sessions — delta × fill_k")
    grid, n = run_grid(store, sessions, front_dte=args.front_dte, deltas=deltas, ks=ks)
    _table(grid, deltas, ks, P)

    lo, hi = date(2026, 2, 1), date(2026, 4, 30)
    if any(lo <= s <= hi for s in sessions):
        nwin = sum(1 for s in sessions if lo <= s <= hi)
        P(f"## His-numbers cross-check — Feb–Apr 2026 ({nwin} sessions)")
        P("Author published (3 months): ~106 trades, ~63% win, ~25% max-profit-hit.\n")
        cc, _ = run_grid(store, sessions, lo, hi, front_dte=args.front_dte, deltas=deltas, ks=ks)
        _table(cc, deltas, ks, P)

    if args.md:
        import os
        os.makedirs(os.path.dirname(args.md), exist_ok=True)
        with open(args.md, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[wrote {args.md}]")


if __name__ == "__main__":
    main()
