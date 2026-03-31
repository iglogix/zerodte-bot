"""
Analyze losing trades for patterns: day of week, time of day, economic events,
ORB characteristics, bias direction, volatility regime, etc.

Usage:
  python -m scripts.analyze_losses
  python -m scripts.analyze_losses --delta 0.25
  python -m scripts.analyze_losses --delta 0.10
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_8am.backtest.data_loader import DatabentoLoader
from src_8am.backtest.options_engine import OrbOptionsBacktestEngine, OptionsBacktestResult
from src_8am.backtest.options_report import generate_options_report

# ── Major US economic event dates (FOMC announcements, CPI, NFP) ──────────

# FOMC announcement dates (2nd day of 2-day meetings)
FOMC_DATES = {
    # 2021
    date(2021,1,27), date(2021,3,17), date(2021,4,28), date(2021,6,16),
    date(2021,7,28), date(2021,9,22), date(2021,11,3), date(2021,12,15),
    # 2022
    date(2022,1,26), date(2022,3,16), date(2022,5,4), date(2022,6,15),
    date(2022,7,27), date(2022,9,21), date(2022,11,2), date(2022,12,14),
    # 2023
    date(2023,2,1), date(2023,3,22), date(2023,5,3), date(2023,6,14),
    date(2023,7,26), date(2023,9,20), date(2023,11,1), date(2023,12,13),
    # 2024
    date(2024,1,31), date(2024,3,20), date(2024,5,1), date(2024,6,12),
    date(2024,7,31), date(2024,9,18), date(2024,11,7), date(2024,12,18),
    # 2025
    date(2025,1,29), date(2025,3,19), date(2025,5,7), date(2025,6,18),
    date(2025,7,30), date(2025,9,17), date(2025,10,29), date(2025,12,10),
    # 2026
    date(2026,1,28), date(2026,3,18), date(2026,4,29), date(2026,6,17),
    date(2026,7,29), date(2026,9,16), date(2026,10,28), date(2026,12,9),
}

# CPI release dates (typically 2nd or 3rd Tuesday/Wednesday of the month, 8:30 AM ET)
CPI_DATES = {
    # 2021
    date(2021,1,13), date(2021,2,10), date(2021,3,10), date(2021,4,13),
    date(2021,5,12), date(2021,6,10), date(2021,7,13), date(2021,8,11),
    date(2021,9,14), date(2021,10,13), date(2021,11,10), date(2021,12,10),
    # 2022
    date(2022,1,12), date(2022,2,10), date(2022,3,10), date(2022,4,12),
    date(2022,5,11), date(2022,6,10), date(2022,7,13), date(2022,8,10),
    date(2022,9,13), date(2022,10,13), date(2022,11,10), date(2022,12,13),
    # 2023
    date(2023,1,12), date(2023,2,14), date(2023,3,14), date(2023,4,12),
    date(2023,5,10), date(2023,6,13), date(2023,7,12), date(2023,8,10),
    date(2023,9,13), date(2023,10,12), date(2023,11,14), date(2023,12,12),
    # 2024
    date(2024,1,11), date(2024,2,13), date(2024,3,12), date(2024,4,10),
    date(2024,5,15), date(2024,6,12), date(2024,7,11), date(2024,8,14),
    date(2024,9,11), date(2024,10,10), date(2024,11,13), date(2024,12,11),
    # 2025
    date(2025,1,15), date(2025,2,12), date(2025,3,12), date(2025,4,10),
    date(2025,5,13), date(2025,6,11), date(2025,7,15), date(2025,8,12),
    date(2025,9,10), date(2025,10,14), date(2025,11,12), date(2025,12,10),
    # 2026
    date(2026,1,14), date(2026,2,11),
}

# NFP: first Friday of each month (8:30 AM ET) — already skipped by skip_nfp_fridays
# but including for analysis
def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 4:  # Friday
        d = d.replace(day=d.day + 1)
    return d

NFP_DATES = {_first_friday(y, m) for y in range(2021, 2027) for m in range(1, 13) if not (y == 2026 and m > 2)}

# PPI release dates (typically day after or near CPI)
# GDP advance/preliminary/final releases (end of month)
# For simplicity, mark the day before and after FOMC/CPI as "adjacent"

ALL_EVENTS = {}
for d in FOMC_DATES:
    ALL_EVENTS[d] = "FOMC"
for d in CPI_DATES:
    ALL_EVENTS[d] = "CPI"
for d in NFP_DATES:
    if d not in ALL_EVENTS:
        ALL_EVENTS[d] = "NFP"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def analyze(results: list[OptionsBacktestResult], label: str):
    closed = [r for r in results if r.status == "closed"]
    losses = [r for r in closed if (r.realized_pnl_usd or 0) < 0]
    wins = [r for r in closed if (r.realized_pnl_usd or 0) >= 0]

    if not losses:
        print(f"\n{'='*70}")
        print(f"  {label}: NO LOSSES to analyze ({len(wins)} wins)")
        print(f"{'='*70}")
        return

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {len(losses)} losses / {len(closed)} trades ({len(losses)/len(closed)*100:.1f}% loss rate)")
    print(f"{'='*70}")

    # ── 1. Day of Week ─────────────────────────────────────────────────
    print("\n── Day of Week ──")
    loss_dow = Counter(r.session_date.weekday() for r in losses)
    all_dow = Counter(r.session_date.weekday() for r in closed)
    for dow in range(5):
        n_loss = loss_dow.get(dow, 0)
        n_all = all_dow.get(dow, 0)
        rate = n_loss / n_all * 100 if n_all else 0
        bar = "█" * n_loss
        print(f"  {DAY_NAMES[dow]:>3}: {n_loss:>3} losses / {n_all:>3} trades ({rate:5.1f}%)  {bar}")

    # ── 2. Entry Time of Day ───────────────────────────────────────────
    print("\n── Entry Time (hour bucket) ──")
    def time_bucket(dt: datetime | None) -> str:
        if dt is None:
            return "unknown"
        h = dt.hour
        if h < 10:
            return "08:00-09:59 (pre-open)"
        elif h < 12:
            return "10:00-11:59 (morning)"
        elif h < 14:
            return "12:00-13:59 (midday)"
        else:
            return "14:00-15:59 (afternoon)"

    loss_time = Counter(time_bucket(r.entry_time) for r in losses)
    all_time = Counter(time_bucket(r.entry_time) for r in closed)
    for bucket in sorted(set(list(loss_time.keys()) + list(all_time.keys()))):
        n_loss = loss_time.get(bucket, 0)
        n_all = all_time.get(bucket, 0)
        rate = n_loss / n_all * 100 if n_all else 0
        print(f"  {bucket:>28}: {n_loss:>3} losses / {n_all:>3} trades ({rate:5.1f}%)")

    # ── 3. Economic Events ─────────────────────────────────────────────
    print("\n── Economic Calendar Events ──")
    event_losses = defaultdict(list)
    event_all = defaultdict(int)
    non_event_losses = 0
    non_event_all = 0

    for r in closed:
        ev = ALL_EVENTS.get(r.session_date)
        is_loss = (r.realized_pnl_usd or 0) < 0
        if ev:
            event_all[ev] += 1
            if is_loss:
                event_losses[ev].append(r)
        else:
            non_event_all += 1
            if is_loss:
                non_event_losses += 1

    for ev in ["FOMC", "CPI", "NFP"]:
        n_loss = len(event_losses.get(ev, []))
        n_all = event_all.get(ev, 0)
        rate = n_loss / n_all * 100 if n_all else 0
        pnl = sum(r.realized_pnl_usd or 0 for r in event_losses.get(ev, []))
        print(f"  {ev:>5}: {n_loss:>3} losses / {n_all:>3} trades ({rate:5.1f}%)  PnL: ${pnl:>10,.0f}")
    rate = non_event_losses / non_event_all * 100 if non_event_all else 0
    print(f"  {'Other':>5}: {non_event_losses:>3} losses / {non_event_all:>3} trades ({rate:5.1f}%)")

    # ── 4. ORB Range at Entry ──────────────────────────────────────────
    print("\n── ORB Range (pts) ──")
    range_buckets = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
    for lo, hi in range_buckets:
        n_loss = sum(1 for r in losses if lo <= r.orb_range_pts < hi)
        n_all = sum(1 for r in closed if lo <= r.orb_range_pts < hi)
        rate = n_loss / n_all * 100 if n_all else 0
        print(f"  {lo:>2}-{hi:>2} pts: {n_loss:>3} losses / {n_all:>3} trades ({rate:5.1f}%)")

    # ── 5. Bias Direction ──────────────────────────────────────────────
    print("\n── Bias Direction ──")
    for bias in ["long", "short"]:
        n_loss = sum(1 for r in losses if r.bias == bias)
        n_all = sum(1 for r in closed if r.bias == bias)
        rate = n_loss / n_all * 100 if n_all else 0
        pnl = sum(r.realized_pnl_usd or 0 for r in losses if r.bias == bias)
        print(f"  {bias:>5}: {n_loss:>3} losses / {n_all:>3} trades ({rate:5.1f}%)  Loss PnL: ${pnl:>10,.0f}")

    # ── 6. Volatility Regime (IV at entry) ─────────────────────────────
    print("\n── IV at Entry ──")
    iv_buckets = [(0, 0.12), (0.12, 0.18), (0.18, 0.25), (0.25, 0.40), (0.40, 1.0)]
    for lo, hi in iv_buckets:
        n_loss = sum(1 for r in losses if lo <= (r.iv_assumed or 0.15) < hi)
        n_all = sum(1 for r in closed if lo <= (r.iv_assumed or 0.15) < hi)
        rate = n_loss / n_all * 100 if n_all else 0
        label = f"{lo*100:.0f}-{hi*100:.0f}%"
        print(f"  IV {label:>9}: {n_loss:>3} losses / {n_all:>3} trades ({rate:5.1f}%)")

    # ── 7. Month of Year ───────────────────────────────────────────────
    print("\n── Month ──")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in range(1, 13):
        n_loss = sum(1 for r in losses if r.session_date.month == m)
        n_all = sum(1 for r in closed if r.session_date.month == m)
        rate = n_loss / n_all * 100 if n_all else 0
        if n_all:
            print(f"  {month_names[m-1]:>3}: {n_loss:>3} losses / {n_all:>3} trades ({rate:5.1f}%)")

    # ── 8. Year ────────────────────────────────────────────────────────
    print("\n── Year ──")
    for y in sorted(set(r.session_date.year for r in closed)):
        n_loss = sum(1 for r in losses if r.session_date.year == y)
        n_all = sum(1 for r in closed if r.session_date.year == y)
        rate = n_loss / n_all * 100 if n_all else 0
        pnl = sum(r.realized_pnl_usd or 0 for r in losses if r.session_date.year == y)
        print(f"  {y}: {n_loss:>3} losses / {n_all:>3} trades ({rate:5.1f}%)  Loss PnL: ${pnl:>10,.0f}")

    # ── 9. Consecutive losses ──────────────────────────────────────────
    print("\n── Consecutive Loss Streaks ──")
    streak = 0
    streaks = []
    for r in sorted(closed, key=lambda x: x.session_date):
        if (r.realized_pnl_usd or 0) < 0:
            streak += 1
        else:
            if streak > 0:
                streaks.append(streak)
            streak = 0
    if streak > 0:
        streaks.append(streak)
    if streaks:
        print(f"  Max streak: {max(streaks)}  Avg: {sum(streaks)/len(streaks):.1f}  Count: {len(streaks)}")
    else:
        print(f"  No consecutive losses")

    # ── 10. Worst individual losses ────────────────────────────────────
    print("\n── Top 10 Worst Losses ──")
    worst = sorted(losses, key=lambda r: r.realized_pnl_usd or 0)[:10]
    for r in worst:
        ev = ALL_EVENTS.get(r.session_date, "")
        dow = DAY_NAMES[r.session_date.weekday()]
        entry_t = r.entry_time.strftime("%H:%M") if r.entry_time else "?"
        print(f"  {r.session_date} ({dow}) {entry_t}  "
              f"bias={r.bias:>5}  ORB={r.orb_range_pts:5.1f}pts  "
              f"credit={r.credit_received or 0:.3f}  "
              f"PnL=${r.realized_pnl_usd or 0:>10,.0f}  "
              f"IV={r.iv_assumed:.2f}  {ev}")


def main():
    parser = argparse.ArgumentParser(description="Analyze losing trades for patterns")
    parser.add_argument("--config", default="config/config_8am_es_options.yaml")
    parser.add_argument("--delta", type=float, nargs="+", default=[0.10, 0.25],
                        help="Delta values to test (default: 0.10 0.25)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader = DatabentoLoader(config["backtest"]["data_file"])
    print("Loading all bars...")
    all_days = list(loader.iter_days(start_date=date(2021, 3, 1), end_date=date(2026, 2, 26)))
    print(f"  {len(all_days)} day-symbol chunks\n")

    for delta in args.delta:
        cfg = yaml.safe_load(yaml.dump(config))  # deep copy
        cfg["options"]["target_short_delta"] = delta
        cfg["options"]["spread_width_pts"] = 20
        cfg["options"]["tp_pct_of_credit"] = 0.75
        cfg["options"]["quantity"] = 50
        cfg["options"]["contract_multiplier"] = 50
        cfg["strategy"]["midline_tolerance_pts"] = 0.5
        cfg["strategy"]["orb_range_filter"]["min_pts"] = 5.0
        cfg["risk"]["max_loss_per_trade_usd"] = 50000
        cfg["risk"]["max_daily_loss_usd"] = 50000

        engine = OrbOptionsBacktestEngine(config=cfg)
        results = engine.run_on_days(all_days)
        report = generate_options_report(results)

        label = (f"Delta={delta:.2f}  |  "
                 f"Trades={report.get('trades_placed', 0)}  "
                 f"WR={report.get('win_rate', 0)*100:.1f}%  "
                 f"PnL=${report.get('total_pnl_usd', 0):,.0f}")
        analyze(results, label)


if __name__ == "__main__":
    main()
