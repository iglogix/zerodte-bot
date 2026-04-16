#!/usr/bin/env python3
"""Enumerate historical ORB signal events from ES OHLCV bars.

Reads existing GLBX MDP3 ES OHLCV-1m CSVs via DatabentoLoader, replays the ORB
signal detector across all days in the range, and writes one JSON line per
signal event to a manifest file.

Usage:
    python scripts/enumerate_signals.py \
        --es-csv glbx-mdp3-20210227-20260226.ohlcv-1m.csv \
        --config config/config_orb_dir_breakout_spx.yaml \
        --start 2021-03-01 --end 2026-02-26 \
        --out data/signal_manifest_orb_breakout.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_8am.backtest.data_loader import DatabentoLoader
from src_v2.backtest.signal_detector import detect_signals_for_day


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--es-csv", required=True, help="GLBX MDP3 ES OHLCV-1m CSV (.csv or .csv.zst)")
    p.add_argument("--config", required=True, help="Strategy config YAML")
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--out", required=True, help="Output manifest path (.jsonl)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    strat = cfg["strategy"]
    orb_start = strat.get("orb_window_start", "08:00")
    orb_end = strat.get("orb_window_end", "08:15")
    market_open = strat.get("market_open", "09:30")
    eod = strat.get("eod_force_close", "15:50")
    retest_mode = strat.get("retest_mode", "midline")
    midline_tol = strat.get("midline_tolerance_pts", 1.0)
    rfilter = strat.get("orb_range_filter", {})
    min_rng = rfilter.get("min_pts", 2.0)
    max_rng = rfilter.get("max_pts", 30.0)

    loader = DatabentoLoader(args.es_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate by date, keeping the symbol with highest volume per date
    days_by_date = {}
    for day in loader.iter_days(start_date=args.start, end_date=args.end, front_month_only=True):
        vol = sum(b["volume"] for b in day.bars)
        if day.session_date not in days_by_date or vol > sum(b["volume"] for b in days_by_date[day.session_date].bars):
            days_by_date[day.session_date] = day

    n_days = 0
    n_signals = 0
    with open(out_path, "w") as f:
        for session_date in sorted(days_by_date.keys()):
            day = days_by_date[session_date]
            n_days += 1
            sigs = detect_signals_for_day(
                bars=day.bars,
                session_date=day.session_date,
                orb_start=orb_start, orb_end=orb_end,
                market_open=market_open, eod=eod,
                retest_mode=retest_mode,
                midline_tol_pts=midline_tol,
                min_range_pts=min_rng, max_range_pts=max_rng,
            )
            for s in sigs:
                f.write(json.dumps({
                    "date": s.session_date.isoformat(),
                    "bias": s.bias,
                    "signal_minute_et": s.signal_minute_et,
                    "es_price_at_signal": s.es_price_at_signal,
                    "orb_high": s.orb_high,
                    "orb_low": s.orb_low,
                }) + "\n")
                n_signals += 1

    print(f"Scanned {n_days} days, emitted {n_signals} signals → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
