"""Extract per-day effective $-SL from the winning open-IC WF-CV trade log.

Outputs data/ic_sl_dollars_by_day.csv with columns:
    trade_date (YYYY-MM-DD), ic_total_credit, sl_mult, ic_sl_dollars

Also emits a companion JSON file data/ic_sl_dollars_by_day.meta.json with:
    {"p75": <float>, "median": <float>, "mean": <float>, "n_days": <int>, ...}

The date column is auto-detected from: `trade_date`, `session_date`, `entry_date`, `date`, `day`.

The script handles three trade-log formats:

  1. Single-row-per-day with a ``total_credit`` column.
  2. Single-row-per-day with ``put_credit`` and ``call_credit`` columns.
  3. Two-rows-per-day (one per leg) with a ``credit_received`` column
     (the format produced by the open-IC WF-CV runner).  Rows are grouped
     by the date column and summed.

Usage:
    python scripts/extract_ic_sl_dollars.py \\
        --trades results/spx_0dte_open_ic/<run>/trades.csv \\
        --sl-mult 2.0 \\
        --out data/ic_sl_dollars_by_day.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ("trade_date", "session_date", "entry_date", "date", "day"):
        if c in df.columns:
            return c
    raise SystemExit(
        f"Trade log has no recognizable date column.  Columns present: {list(df.columns)}"
    )


def _compute_total_credit(df: pd.DataFrame, date_col: str) -> pd.Series:
    """Return a Series of total IC credit indexed by trade_date string."""
    if "total_credit" in df.columns:
        # Format 1: single row per day, pre-summed credit.
        out = df[[date_col, "total_credit"]].copy()
        out[date_col] = pd.to_datetime(out[date_col]).dt.date.astype(str)
        n_dups = out.duplicated(subset=date_col).sum()
        if n_dups:
            print(
                f"Warning: {n_dups} duplicate dates in trade log; keeping first occurrence per date.",
                file=sys.stderr,
            )
        return (
            out.drop_duplicates(subset=date_col).set_index(date_col)["total_credit"].astype(float)
        )

    if {"put_credit", "call_credit"}.issubset(df.columns):
        # Format 2: single row per day with separate put/call columns.
        dates = pd.to_datetime(df[date_col]).dt.date.astype(str)
        credit = df["put_credit"].astype(float) + df["call_credit"].astype(float)
        tmp = pd.DataFrame({"date": dates, "credit": credit})
        n_dups = tmp.duplicated(subset="date").sum()
        if n_dups:
            print(
                f"Warning: {n_dups} duplicate dates in trade log; keeping first occurrence per date.",
                file=sys.stderr,
            )
        return tmp.drop_duplicates(subset="date").set_index("date")["credit"]

    if "credit_received" in df.columns:
        # Format 3: two rows per day (one per leg); sum within each date.
        dates = pd.to_datetime(df[date_col]).dt.date.astype(str)
        tmp = pd.DataFrame({"date": dates, "credit": df["credit_received"].astype(float)})
        return tmp.groupby("date")["credit"].sum()

    raise SystemExit(
        f"Trade log has no credit column (total_credit, put_credit/call_credit, "
        f"or credit_received).  Columns present: {list(df.columns)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--trades",
        required=True,
        help="Path to IC trades.csv (or .parquet) from the WF-CV run",
    )
    ap.add_argument(
        "--sl-mult",
        type=float,
        default=2.0,
        help="The IC's sl_mult_of_credit (default 2.0)",
    )
    ap.add_argument("--out", default="data/ic_sl_dollars_by_day.csv")
    ap.add_argument(
        "--multiplier",
        type=int,
        default=100,
        help="Contract multiplier (SPX = 100)",
    )
    args = ap.parse_args()

    p = Path(args.trades)
    if not p.exists():
        raise SystemExit(f"Trades file not found: {p}")

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    date_col = _detect_date_col(df)
    credit_by_day = _compute_total_credit(df, date_col)

    out = pd.DataFrame(
        {
            "trade_date": credit_by_day.index,
            "ic_total_credit": credit_by_day.values.round(2),
            "sl_mult": args.sl_mult,
            "ic_sl_dollars": (credit_by_day.values * args.sl_mult * args.multiplier).round(2),
        }
    ).sort_values("trade_date")

    if len(out) == 0:
        raise SystemExit(
            f"Trade log {p} produced zero rows after grouping/dedup — nothing to write."
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    arr = out["ic_sl_dollars"].to_numpy()
    meta = {
        "p75": float(np.percentile(arr, 75)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_days": int(len(arr)),
        "source_trades": str(p),
        "sl_mult": args.sl_mult,
    }
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {len(out)} days to {args.out}")
    print(f"Wrote meta to {meta_path}")
    print(
        f"  p75=${meta['p75']:.2f}  median=${meta['median']:.2f}  "
        f"mean=${meta['mean']:.2f}  range=[${meta['min']:.2f},${meta['max']:.2f}]"
    )


if __name__ == "__main__":
    main()
