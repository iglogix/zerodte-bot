"""Build a manifest covering every trading day at 09:35 ET for IC entry-only CBBO.

Output JSONL is a drop-in input for scripts/databento_backfill.py — used to
estimate (via --dry-run) and fetch the entry-minute CBBO across the full strike
chain at 09:35.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodte.backtest.data_loader import BarLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-04-18")
    p.add_argument("--end",   default="2026-02-28")
    p.add_argument("--bars",  default="data/spy_1m.csv")
    # Comma-separated anchor minutes; one entry per (day, anchor) pair.
    p.add_argument("--anchors", default="09:35,11:00,13:00,15:30,15:59",
                   help="comma-separated HH:MM list — one manifest entry per (day, anchor)")
    p.add_argument("--out", default="data/signal_manifest_multi_anchor.jsonl")
    args = p.parse_args()

    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    loader = BarLoader(args.bars, price_scale=10.0, symbol="SPX")
    days = list(loader.iter_days(start_date=start, end_date=end))

    out_path = Path(args.out)
    n_written = 0
    n_skipped_days = 0
    n_skipped_anchors = 0
    from datetime import time as _t

    with open(out_path, "w") as f:
        for d in days:
            if not d.bars:
                n_skipped_days += 1
                continue
            for anchor in anchors:
                hh, mm = int(anchor[:2]), int(anchor[3:5])
                target = _t(hh, mm)
                anchor_bar = next((b for b in d.bars if b["ts_et"].time() == target), None)
                if anchor_bar is None:
                    n_skipped_anchors += 1
                    continue
                entry = {
                    "date": d.session_date.isoformat(),
                    "signal_minute_et": anchor,
                    # ES proxy from spot/10 — basis correction is small for our purposes
                    "es_price_at_signal": round(anchor_bar["open"] / 10.0, 2),
                }
                f.write(json.dumps(entry) + "\n")
                n_written += 1

    print(f"Wrote {n_written} entries → {out_path}")
    print(f"Anchors: {anchors}")
    print(f"Skipped {n_skipped_days} days (no bars), {n_skipped_anchors} (day, anchor) pairs (missing minute bar)")


if __name__ == "__main__":
    main()
