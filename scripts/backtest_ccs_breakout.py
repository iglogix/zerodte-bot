#!/usr/bin/env python3
"""
Backtest: Sell call credit spread on bearish ORB breakout at market open.

No midline retest needed — enters at bar N after 09:30 when ORB broke low.
Uses real CBBO quotes for entry/exit pricing.

Optimizes:
  1. Entry bar (0=09:30, 1=09:45, 2=10:00, ...)
  2. EM multiplier (distance from spot in straddle units)
  3. Spread width
  4. TP percentage

Usage:
    python scripts/backtest_ccs_breakout.py
    python scripts/backtest_ccs_breakout.py --config config/config_8am_spx_options_em.yaml
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import defaultdict
from datetime import date, time, timedelta
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_8am.backtest.cbbo_store import CbboStore
from src_8am.backtest.data_loader import DatabentoLoader


def main():
    parser = argparse.ArgumentParser(description="CCS breakout backtest")
    parser.add_argument("--config", default="config/config_8am_spx_options_em.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data
    mem = CbboStore.preload_cache(config.get("options", {}).get("cbbo_cache_dir", "data/cbbo_cache"))
    cbbo_dates = set(mem.keys())

    loader = DatabentoLoader(config.get("backtest", {}).get("data_file", "glbx-mdp3-20210227-20260226.ohlcv-1m.csv"))
    all_chunks = [d for d in loader.iter_days(start_date=date(2025, 9, 22), end_date=date(2026, 2, 26))]

    by_date = defaultdict(list)
    for d in all_chunks:
        if " " not in d.symbol:  # futures only
            by_date[d.session_date].append(d)

    strategy = config.get("strategy", {})
    orb_start = time(8, 0)
    orb_end = time(8, 15)
    eod = time(15, 50)
    min_range = strategy.get("orb_range_filter", {}).get("min_pts", 2.0)
    max_range = strategy.get("orb_range_filter", {}).get("max_pts", 30.0)
    strike_step = config.get("options", {}).get("strike_step_pts", 5.0)
    min_credit = config.get("options", {}).get("min_credit_per_spread", 0.05)
    multiplier = config.get("options", {}).get("contract_multiplier", 100)

    # Collect short-bias days
    short_days = []
    for session_date in sorted(by_date.keys()):
        if session_date not in cbbo_dates:
            continue
        candidates = by_date[session_date]
        if not candidates:
            continue
        best = max(candidates, key=lambda d: sum(b.get("volume", 0) for b in d.bars))
        bars = best.bars

        orb_bars = [b for b in bars if orb_start <= b["ts_et"].time() < orb_end]
        if not orb_bars:
            continue
        orb_high = max(b["high"] for b in orb_bars)
        orb_low = min(b["low"] for b in orb_bars)
        orb_range = orb_high - orb_low
        if orb_range < min_range or orb_range > max_range:
            continue

        # Check if short bias: first bar at/after 09:30 below ORB low
        post_open = [b for b in bars if b["ts_et"].time() >= time(9, 30)]
        if not post_open:
            continue

        # Detect bearish breakout in first few bars
        bias = None
        for bar in post_open[:10]:  # check first 10 bars (~10 min)
            if bar["close"] < orb_low:
                bias = "short"
                break
            if bar["close"] > orb_high:
                bias = "long"
                break

        if bias != "short":
            continue

        short_days.append({
            "date": session_date,
            "bars": bars,
            "post_open": post_open,
            "orb_high": orb_high,
            "orb_low": orb_low,
            "quotes": mem[session_date],
        })

    print(f"Short-bias days with CBBO: {len(short_days)}")
    if not short_days:
        print("No short-bias days found")
        return

    # Grid
    grid = {
        "entry_bar": [1, 2, 3, 4],           # 0=09:30, 1=09:45, 2=10:00, 3=10:15, 4=10:30
        "em_mult": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "width": [5, 10, 15, 25],
        "tp_pct": [0.25, 0.50, 0.75],
    }
    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    print(f"Grid: {len(combos)} combos × {len(short_days)} days\n")

    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        entry_bar_num = p["entry_bar"]
        em_mult = p["em_mult"]
        width = p["width"]
        tp_pct = p["tp_pct"]

        trades = []
        for day in short_days:
            bars = day["bars"]
            post_open = day["post_open"]
            quotes = day["quotes"]

            # Get entry bar (15-min bars: bar 1 = 09:45)
            total_mins = 9 * 60 + 30 + entry_bar_num * 15
            entry_minute = time(total_mins // 60, total_mins % 60)
            if entry_minute.hour >= 16:
                continue

            # Find the 1-min bar closest to entry time
            entry_bar = None
            for b in post_open:
                if b["ts_et"].time() >= entry_minute:
                    entry_bar = b
                    break
            if entry_bar is None:
                continue

            spot = entry_bar["close"]
            minute_key = entry_bar["ts_et"].strftime("%H:%M")

            # ATM straddle for EM
            strikes = set(k[1] for k in quotes.keys())
            if not strikes:
                continue
            atm_k = min(strikes, key=lambda k: abs(k - spot))

            pq = cq = None
            for offset in range(0, 6):
                hh, mm = int(minute_key[:2]), int(minute_key[3:5])
                mm2 = mm - offset
                hh2 = hh
                if mm2 < 0:
                    mm2 += 60
                    hh2 -= 1
                mk = f"{hh2:02d}:{mm2:02d}"
                pq = pq or quotes.get((mk, atm_k, False))
                cq = cq or quotes.get((mk, atm_k, True))
                if pq and cq:
                    break
            if not pq or not cq:
                continue

            em = (pq[0] + pq[1]) / 2 + (cq[0] + cq[1]) / 2
            if em <= 0:
                continue

            # Call credit spread strikes
            short_k = math.ceil((spot + em_mult * em) / strike_step) * strike_step
            long_k = short_k + width

            # Get CBBO credit at entry
            sq = quotes.get((minute_key, short_k, True))
            lq = quotes.get((minute_key, long_k, True))
            if not sq or not lq:
                continue
            credit = (sq[0] + sq[1]) / 2 - (lq[0] + lq[1]) / 2
            if credit < min_credit:
                continue

            tp_debit = round(credit * (1 - tp_pct), 2)

            # Simulate exit: walk bars after entry
            entry_idx = post_open.index(entry_bar)
            exit_reason = "expired_worthless"
            debit = 0.0

            for bar in post_open[entry_idx + 1:]:
                ts = bar["ts_et"]
                if ts.time() >= eod:
                    break

                # Breach check
                if bar["high"] >= short_k:
                    exit_reason = "short_breached"
                    debit = width  # max loss
                    break

                # CBBO mark
                mk = ts.strftime("%H:%M")
                sq2 = quotes.get((mk, short_k, True))
                lq2 = quotes.get((mk, long_k, True))
                if sq2 and lq2:
                    mark = (sq2[0] + sq2[1]) / 2 - (lq2[0] + lq2[1]) / 2
                    if mark <= tp_debit:
                        exit_reason = "tp"
                        debit = tp_debit
                        break

            pnl = (credit - debit) * multiplier
            trades.append({
                "date": day["date"],
                "credit": credit,
                "pnl": pnl,
                "exit": exit_reason,
                "em": em,
                "short_k": short_k,
                "spot": spot,
            })

        if not trades:
            continue

        total_pnl = sum(t["pnl"] for t in trades)
        wins = len([t for t in trades if t["pnl"] > 0])
        wr = wins / len(trades)
        pnls = np.array([t["pnl"] for t in trades])
        sharpe = float(np.mean(pnls) / max(np.std(pnls), 0.01) * np.sqrt(252))
        running = 0
        peak = 0
        dd = 0
        for t in trades:
            running += t["pnl"]
            peak = max(peak, running)
            dd = max(dd, peak - running)

        breaches = len([t for t in trades if t["exit"] == "short_breached"])

        results.append({
            **p, "n": len(trades), "wr": wr, "pnl": total_pnl,
            "avg_pnl": total_pnl / len(trades), "sharpe": sharpe,
            "dd": dd, "breaches": breaches,
            "avg_credit": np.mean([t["credit"] for t in trades]),
        })

    print(f"{len(results)} valid combos\n")

    hdr = (f'{"Bar":>4} {"EM":>5} {"W":>3} {"TP":>4} | '
           f'{"N":>3} {"WR":>5} {"Br":>3} | '
           f'{"PnL":>7} {"AvgPnL":>7} {"AvgCr":>6} | '
           f'{"Shrp":>6} {"DD":>6}')

    results.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"Top 20 by PnL:\n{hdr}")
    for r in results[:20]:
        print(f'{r["entry_bar"]:>4} {r["em_mult"]:>5} {r["width"]:>3} {r["tp_pct"]:>4} | '
              f'{r["n"]:>3} {r["wr"]:>4.0%} {r["breaches"]:>3} | '
              f'{r["pnl"]:>7.0f} {r["avg_pnl"]:>7.1f} {r["avg_credit"]:>6.2f} | '
              f'{r["sharpe"]:>6.1f} {r["dd"]:>6.0f}')

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    print(f"\nTop 20 by Sharpe:\n{hdr}")
    for r in results[:20]:
        print(f'{r["entry_bar"]:>4} {r["em_mult"]:>5} {r["width"]:>3} {r["tp_pct"]:>4} | '
              f'{r["n"]:>3} {r["wr"]:>5.0%} {r["breaches"]:>3} | '
              f'{r["pnl"]:>7.0f} {r["avg_pnl"]:>7.1f} {r["avg_credit"]:>6.2f} | '
              f'{r["sharpe"]:>6.1f} {r["dd"]:>6.0f}')

    # Show per-trade detail for best combo
    if results:
        best = max(results, key=lambda x: x["sharpe"])
        print(f"\n=== Best combo detail: bar={best['entry_bar']} em={best['em_mult']} "
              f"w={best['width']} tp={best['tp_pct']} ===")
        # Re-run best combo to get trade details
        p = best
        for day in short_days:
            bars = day["bars"]
            post_open = day["post_open"]
            quotes = day["quotes"]
            entry_minute = time(9, 30 + p["entry_bar"] * 15)
            entry_bar = next((b for b in post_open if b["ts_et"].time() >= entry_minute), None)
            if not entry_bar:
                continue
            spot = entry_bar["close"]
            minute_key = entry_bar["ts_et"].strftime("%H:%M")
            strikes = set(k[1] for k in quotes.keys())
            if not strikes:
                continue
            atm_k = min(strikes, key=lambda k: abs(k - spot))
            pq = cq = None
            for offset in range(0, 6):
                hh, mm = int(minute_key[:2]), int(minute_key[3:5])
                mm2 = mm - offset; hh2 = hh
                if mm2 < 0: mm2 += 60; hh2 -= 1
                mk = f"{hh2:02d}:{mm2:02d}"
                pq = pq or quotes.get((mk, atm_k, False))
                cq = cq or quotes.get((mk, atm_k, True))
                if pq and cq: break
            if not pq or not cq: continue
            em = (pq[0]+pq[1])/2 + (cq[0]+cq[1])/2
            if em <= 0: continue
            short_k = math.ceil((spot + p["em_mult"] * em) / strike_step) * strike_step
            long_k = short_k + p["width"]
            sq = quotes.get((minute_key, short_k, True))
            lq = quotes.get((minute_key, long_k, True))
            if not sq or not lq: continue
            credit = (sq[0]+sq[1])/2 - (lq[0]+lq[1])/2
            if credit < min_credit: continue

            # Quick exit scan
            entry_idx = post_open.index(entry_bar)
            exit_r = "eod"
            for bar in post_open[entry_idx+1:]:
                if bar["ts_et"].time() >= eod: break
                if bar["high"] >= short_k: exit_r = "breach"; break
                mk = bar["ts_et"].strftime("%H:%M")
                sq2 = quotes.get((mk, short_k, True))
                lq2 = quotes.get((mk, long_k, True))
                if sq2 and lq2:
                    mark = (sq2[0]+sq2[1])/2 - (lq2[0]+lq2[1])/2
                    if mark <= round(credit*(1-p["tp_pct"]),2):
                        exit_r = "tp"; break

            pnl_val = credit * multiplier if exit_r != "breach" else -p["width"] * multiplier
            if exit_r == "tp":
                pnl_val = credit * p["tp_pct"] * multiplier
            print(f"  {day['date']} spot={spot:.0f} em={em:.1f} short={short_k:.0f} "
                  f"credit={credit:.2f} exit={exit_r} pnl=${pnl_val:.0f}")


if __name__ == "__main__":
    main()
