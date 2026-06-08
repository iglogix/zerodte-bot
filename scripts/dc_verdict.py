"""DC Time Machine — verdict report generator.

Reads a multi-expiration cache, runs the simulator across a parameter grid and a
fill-aggressiveness (k) sweep, and prints the headline question for this strategy:
does the "risk-free transform" survive realistic execution? Also emits the Feb–Apr
2026 cross-check vs the author's published numbers, and a per-regime breakdown.

Usage:
    .venv/bin/python scripts/dc_verdict.py --cache data/dc_cache_full [--md OUT.md]
    .venv/bin/python scripts/dc_verdict.py --cache data/dc_cache_full --extra data/dc_cache_smoke data/dc_cache_hivol
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
    maxprofit = [r for r in settled if abs(r.pnl - r.bank * 100) < 1e-6]  # landed inside tent
    return {
        "n": len(real), "transformed": len(tr),
        "transform_rate": len(tr) / len(real) if real else 0.0,
        "win_rate": len(wins) / len(real) if real else 0.0,
        "total": sum(r.pnl for r in real),
        "pf": gw / gl if gl else float("inf"),
        "avg_win": gw / len(wins) if wins else 0.0,
        "avg_loss": -gl / len(losses) if losses else 0.0,
        "maxprofit_rate": len(maxprofit) / len(settled) if settled else 0.0,
    }


def run_grid(store: MultiExpiryStore, sessions: List[date],
             lo: Optional[date] = None, hi: Optional[date] = None,
             fdtes=(3, 5, 7, 9), ks=(0.0, 0.5, 1.0), filt="always") -> Dict:
    sub = [s for s in sessions if (lo is None or s >= lo) and (hi is None or s <= hi)]
    out = {}
    for fdte in fdtes:
        for k in ks:
            p = DcParams(front_dte=fdte, fill_k=k, entry_filter=filt)
            rows = [run_session(store, p, s) for s in sub]
            out[(fdte, k)] = summarize(rows)
    return out, len(sub)


def _fmt(s: Dict) -> str:
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"n={s['n']:4d} tr%={s['transform_rate']*100:4.0f} win%={s['win_rate']*100:4.0f} "
            f"PF={pf:>5} tot=${s['total']:8.0f} avgW=${s['avg_win']:6.0f} avgL=${s['avg_loss']:6.0f} "
            f"maxP%={s['maxprofit_rate']*100:3.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    caches = [args.cache] + args.extra
    # Merge sessions from all caches into one store view by preloading dicts.
    merged: Dict[date, dict] = {}
    for c in caches:
        store = MultiExpiryStore(cache_dir=c)
        for s in _sessions(c):
            store.load_session(s)
            merged.setdefault(s, store._panel)
    store = MultiExpiryStore(preloaded=merged)
    sessions = sorted(merged)
    if not sessions:
        print("No sessions in cache(s).")
        return

    lines = []
    def P(x=""):
        print(x); lines.append(x)

    P(f"# DC Time Machine — verdict\n")
    P(f"Cache(s): {', '.join(caches)}")
    P(f"Sessions: {len(sessions)}  [{sessions[0]} .. {sessions[-1]}]\n")

    P("## Fill-aggressiveness sweep (favorite config family), ALL sessions")
    P("k=0 is mid (optimistic); k=1 is full bid/ask cross (conservative). The verdict")
    P("lives in how the edge changes across k.\n")
    grid, n = run_grid(store, sessions)
    P("```")
    for fdte in (3, 5, 7, 9):
        for k in (0.0, 0.5, 1.0):
            P(f"front_dte={fdte} k={k:.2f}: {_fmt(grid[(fdte, k)])}")
        P("")
    P("```")

    # Cross-check window
    lo, hi = date(2026, 2, 1), date(2026, 4, 30)
    if any(lo <= s <= hi for s in sessions):
        P("## His-numbers cross-check (Feb–Apr 2026)")
        P("Author published: ~106 trades, ~63% win, ~25% max-profit-hit (3 months).\n")
        cc, ncc = run_grid(store, sessions, lo, hi, fdtes=(7, 9), ks=(0.0, 0.5, 1.0))
        P("```")
        for fdte in (7, 9):
            for k in (0.0, 0.5, 1.0):
                P(f"front_dte={fdte} k={k:.2f}: {_fmt(cc[(fdte, k)])}")
            P("")
        P("```")

    if args.md:
        with open(args.md, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[wrote {args.md}]")


if __name__ == "__main__":
    main()
