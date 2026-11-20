# zerodte/backtest/dc_time_machine/run_backtest.py
"""CLI for the DC Time Machine backtest.

Subcommands:
  backfill       Fetch multi-expiry full-day cbbo-1m into a cache dir (needs Databento).
  study-spreads  Print the empirical half-spread table from a cache dir.
  simulate       Run a config over a cache dir → trades.csv + summary.
  scan           Reconstructed Flux scan over a cache dir.
  wf-cv          Walk-forward CV over a param grid.
  crosscheck     Simulate Feb–Apr 2026 and print aggregate stats vs his published numbers.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from zerodte.backtest.dc_time_machine.dc_backfill import backfill_session
from zerodte.backtest.dc_time_machine.multi_expiry_store import MultiExpiryStore
from zerodte.backtest.dc_time_machine.simulator import TradeResult, run_session
from zerodte.backtest.dc_time_machine.strategy import DcParams
from zerodte.backtest.dc_time_machine.trading_calendar import (
    nth_trading_day_after,
    sessions_between,
)

_CSV_FIELDS = ["session", "front", "back", "k_put", "k_call", "outcome", "entry_debit",
               "credit", "bank", "transform_minute", "settlement", "pnl"]


def session_expiries(session: date, params: DcParams, pad: int = 1) -> List[date]:
    """Expiry dates to fetch for one session: trading days [front_dte-pad .. front_dte+gap+pad].

    The pad tolerates SPXW dates that aren't actual expiries, so resolve_expiries always
    finds a cached expiry within tolerance of its front/back targets.
    """
    lo = max(1, params.front_dte - pad)
    hi = params.front_dte + params.gap + pad
    out: List[date] = []
    for n in range(lo, hi + 1):
        d = nth_trading_day_after(session, n)
        if d is not None and d not in out:
            out.append(d)
    return out


def atm_from_spy(spy_csv: str | Path = "data/spy_daily.csv"):
    """Build a session→ATM-strike lookup from SPY daily closes (SPY×10 ≈ SPX, step-aligned)."""
    import csv as _csv
    m: Dict[date, float] = {}
    with open(spy_csv) as f:
        for row in _csv.DictReader(f):
            d = date.fromisoformat(row["Date"])
            m[d] = round(float(row["Close"]) * 10 / 5) * 5
    return lambda s: m.get(s)


def backfill_range(client, sessions: List[date], atm_for, params: DcParams,
                   out_dir: str | Path, pad: int = 1, half_width: int = 200,
                   source: str = "databento", workers: int = 1,
                   skip_existing: bool = True) -> List[Path]:
    """Fetch each session's front/back expiry panels into <out_dir>/<session>.pkl.

    source="theta" centers the window on the data-derived forward (no external ATM
    needed) and is flat-rate; "databento" uses the provided client and requires an
    ATM reference. `workers` parallelizes across sessions (I/O-bound HTTP); each
    pickle is written atomically per session so the run is resumable via
    `skip_existing`. Returns the written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _one(s: date) -> Optional[Path]:
        dest = out_dir / f"{s.isoformat()}.pkl"
        if skip_existing and dest.exists():
            return dest
        atm = atm_for(s) if atm_for else None
        if source != "theta" and atm is None:
            return None
        exps = session_expiries(s, params, pad=pad)
        if not exps:
            return None
        # Prepend the session's own 0DTE panel so settlement() can resolve the front
        # IC at expiry (parity from the expiry's own last-minute quotes).
        fetch_exps = [s] + exps
        try:
            if source == "theta":
                from zerodte.backtest.dc_time_machine.theta_backfill import backfill_session_theta
                return backfill_session_theta(s, fetch_exps, atm if atm is not None else 0.0,
                                              out_dir, half_width_pts=half_width)
            return backfill_session(client, s, fetch_exps, atm, out_dir, half_width_pts=half_width)
        except Exception as e:   # one bad session must not abort a multi-hour run
            print(f"  [skip {s}: {type(e).__name__}: {e}]", flush=True)
            return None

    if workers <= 1:
        return [p for p in (_one(s) for s in sessions) if p is not None]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return [p for p in ex.map(_one, sessions) if p is not None]


def write_trades_csv(trades: List[TradeResult], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for t in trades:
            w.writerow({k: getattr(t, k) for k in _CSV_FIELDS})


def summarize(trades: List[TradeResult]) -> Dict[str, float]:
    real = [t for t in trades if t.outcome != "no_entry"]
    wins = [t.pnl for t in real if t.pnl > 0]
    losses = [t.pnl for t in real if t.pnl < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "n": len(real),
        "win_rate": (len(wins) / len(real)) if real else 0.0,
        "total_pnl": sum(t.pnl for t in real),
        "profit_factor": (gross_win / gross_loss) if gross_loss else float("inf"),
        "n_transformed": sum(1 for t in real if t.outcome == "transformed"),
    }


def _sessions_in_cache(cache_dir: Path) -> List[date]:
    return sorted(date.fromisoformat(p.stem) for p in cache_dir.glob("*.pkl"))


def _simulate_range(cache_dir: Path, params: DcParams,
                    lo: date | None = None, hi: date | None = None) -> List[TradeResult]:
    store = MultiExpiryStore(cache_dir=cache_dir)
    out: List[TradeResult] = []
    for s in _sessions_in_cache(cache_dir):
        if (lo and s < lo) or (hi and s > hi):
            continue
        res = run_session(store, params, s)
        if res is not None:
            out.append(res)
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dc-backtest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    bf = sub.add_parser("backfill")
    bf.add_argument("--start", required=True)
    bf.add_argument("--end", required=True)
    bf.add_argument("--out", default="data/dc_cache")
    bf.add_argument("--pad", type=int, default=1)
    bf.add_argument("--half-width", type=int, default=200)
    bf.add_argument("--source", choices=["theta", "databento"], default="theta")
    bf.add_argument("--workers", type=int, default=4)
    bf.add_argument("--oldest-first", action="store_true",
                    help="default is recent-first so the cross-check window lands fast")

    sim = sub.add_parser("simulate")
    sim.add_argument("--cache", required=True)
    sim.add_argument("--out", default="results/dc_time_machine/trades.csv")
    sim.add_argument("--fill-k", type=float, default=1.0)

    cc = sub.add_parser("crosscheck")
    cc.add_argument("--cache", required=True)
    cc.add_argument("--fill-k", type=float, default=1.0)

    args = ap.parse_args(argv)

    if args.cmd == "backfill":
        client = None
        if args.source == "databento":
            import databento as dbn  # local import: only needed for the paid path
            client = dbn.Historical()  # reads DATABENTO_API_KEY from env
        sessions = sessions_between(date.fromisoformat(args.start), date.fromisoformat(args.end))
        if not args.oldest_first:
            sessions = sessions[::-1]   # recent-first: warm/fast data + cross-check window land first
        atm_for = atm_from_spy()
        params = DcParams()
        paths = backfill_range(client, sessions, atm_for, params, Path(args.out),
                               pad=args.pad, half_width=args.half_width, source=args.source,
                               workers=args.workers)
        print(f"backfilled {len(paths)}/{len(sessions)} sessions → {args.out} (source={args.source})")
        return 0

    if args.cmd == "simulate":
        params = DcParams(fill_k=args.fill_k)
        trades = _simulate_range(Path(args.cache), params)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_trades_csv(trades, out)
        print(summarize(trades))
        return 0

    if args.cmd == "crosscheck":
        params = DcParams(fill_k=args.fill_k)
        trades = _simulate_range(Path(args.cache), params, date(2026, 2, 1), date(2026, 4, 30))
        s = summarize(trades)
        print("Feb–Apr 2026 cross-check:", s)
        print("His published: ~106 trades, ~63% win, ~25% max-profit-hit")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
