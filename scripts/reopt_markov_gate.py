#!/usr/bin/env python3
"""Quarterly Markov-gate re-optimization for Power Hour (the validated METHOD).

Validated by results/markov_gate_wf_method_2026-06-12.txt + the full-span
T/cadence grid: re-optimize the gate on the trailing ~130 trade days every
65-130 trade days; every cell of that box beat no-optimization over 593 OOS
days. This script is one re-optimization cycle, end to end:

  1. refresh SPX daily closes (Yahoo chart API, keyless) -> /tmp/gspc.json
  2. incrementally extend data/spy_1m.csv + both CBBO caches (databento,
     ~$0.01/day, hard cap --max-cost-usd)
  3. regenerate the Version A trade log through yesterday and verify the
     overlap region reproduces the previous log exactly
  4. grid-search the gate (rule x window x threshold) on the trailing
     --train-days trade days by account-level Sharpe
  5. rewrite the markov_gate block in config/config_power_hour_live.yaml
     (winner 'none' -> enabled: false) unless --no-config-update
  6. append to results/markov_gate_reopt_history.csv and write a report to
     logs/markov_gate_reopt/<date>.md

Designed to run headless from a systemd timer (markov-gate-reopt.timer) and
to be portable to a VPS: only needs the repo, .venv, and .env (DATABENTO key).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

PY = str(ROOT / ".venv/bin/python")
CONFIG = ROOT / "config/config_power_hour_live.yaml"
TRADES = ROOT / "results/power_hour_spx/trades_A_triggerplus_ext.csv"
HISTORY = ROOT / "results/markov_gate_reopt_history.csv"


def log(msg: str) -> None:
    print(f"[reopt {datetime.now():%H:%M:%S}] {msg}", flush=True)


def refresh_gspc() -> None:
    now = int(datetime.now().timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
           f"?period1=1450000000&period2={now}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        Path("/tmp/gspc.json").write_bytes(r.read())
    log("SPX daily closes refreshed")


def extend_spy_bars(end: date) -> None:
    last = pd.read_csv(ROOT / "data/spy_1m.csv", usecols=["ts_event"]).ts_event.iloc[-1]
    last_day = pd.Timestamp(last).date()
    if last_day >= end:
        log(f"spy_1m.csv current ({last_day}) — skip")
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import databento as db
    data = db.Historical().timeseries.get_range(
        dataset="EQUS.MINI", symbols=["SPY"], schema="ohlcv-1m",
        start=str(last_day + timedelta(days=1)), end=str(end + timedelta(days=1)))
    df = data.to_df().reset_index()
    out = pd.DataFrame({
        "ts_event": df["ts_event"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f000Z"),
        "symbol": "SPY", "open": df["open"], "high": df["high"],
        "low": df["low"], "close": df["close"], "volume": df["volume"]})
    out.to_csv(ROOT / "data/spy_1m.csv", mode="a", header=False, index=False)
    log(f"spy_1m.csv extended {last_day} -> {end} ({len(out)} rows)")


def backfill_caches(end: date, max_cost: float) -> None:
    for script, outdir in (("backfill_butterfly_cbbo.py", None),
                           ("backfill_meic_cbbo.py", "data/cbbo_cache_meic")):
        cache = ROOT / (outdir or "data/cbbo_cache_fly_10min")
        days = sorted(cache.glob("2*.pkl"))
        start = (date.fromisoformat(days[-1].stem) + timedelta(days=1)) if days else end
        if start > end:
            log(f"{cache.name} current — skip")
            continue
        cmd = [PY, str(ROOT / "scripts" / script), "--start", str(start),
               "--end", str(end), "--max-cost-usd", str(max_cost)]
        if outdir:  # backfill_meic_cbbo.py defaults --out to the WRONG dir
            cmd += ["--out", str(ROOT / outdir)]
        log(" ".join(cmd[1:]))
        subprocess.run(cmd, check=True, cwd=ROOT)


def regenerate_trades(end: date) -> None:
    prev = pd.read_csv(TRADES) if TRADES.exists() else None
    subprocess.run([PY, str(ROOT / "scripts/backtest_power_hour.py"),
                    "--config", str(ROOT / "config/config_power_hour_spx.yaml"),
                    "--start", "2022-05-11", "--end", str(end),
                    "--out", str(TRADES)], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    new = pd.read_csv(TRADES)
    if prev is not None:  # overlap must reproduce exactly
        for df in (prev, new):
            df.query("status=='closed'", inplace=True)
        o = prev.groupby("session_date").realized_pnl_usd.sum()
        e = new.groupby("session_date").realized_pnl_usd.sum()
        common = o.index.intersection(e.index)
        bad = int(((o[common] - e[common]).abs() > 0.01).sum())
        if bad:
            sys.exit(f"ABORT: {bad} overlap days diverge from previous trade log")
    log(f"trade log regenerated through {end} (overlap verified)")


def reoptimize(train_days: int):
    from run_wf_method_gate import load_spx_states, gate_masks, sharpe
    spx_dates, closes = load_spx_states()
    df = pd.read_csv(TRADES)
    df = df[df.status == "closed"]
    df["session_date"] = pd.to_datetime(df.session_date).dt.date
    daily = df.groupby("session_date").realized_pnl_usd.sum().sort_index()
    dates = np.array(daily.index)
    pnl = daily.to_numpy(float)
    masks = gate_masks(dates, spx_dates, closes)
    t = slice(len(pnl) - train_days, len(pnl))
    scores = {k: sharpe(np.where(m[t], pnl[t], 0.0)) for k, m in masks.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    win = ranked[0]
    base = scores[("none", 0, 0.0)]
    return win, scores[win], base, dates[t][0], dates[-1], ranked[:5], scores


def score_previous_deployment() -> None:
    """Close the books on the outgoing gate: score it on the period since it
    was deployed (its OOS window) vs baseline, and append to the live OOS
    record. Every deployment period is an OOS test — this file is the proof."""
    if not HISTORY.exists():
        return
    import ast
    from run_wf_method_gate import load_spx_states, gate_masks, sharpe
    prev = pd.read_csv(HISTORY).iloc[-1]
    deployed = date.fromisoformat(prev.run_date)
    rule, window, thr = ast.literal_eval(prev.winner)

    df = pd.read_csv(TRADES)
    df = df[df.status == "closed"]
    df["session_date"] = pd.to_datetime(df.session_date).dt.date
    daily = df.groupby("session_date").realized_pnl_usd.sum().sort_index()
    oos = daily[daily.index > deployed]
    if len(oos) == 0:
        log("previous deployment has no elapsed OOS days yet — nothing to score")
        return
    spx_dates, closes = load_spx_states()
    dates = np.array(oos.index)
    pnl = oos.to_numpy(float)
    key = ("none", 0, 0.0) if rule == "none" else (rule, window, thr)
    m = gate_masks(dates, spx_dates, closes)[key]
    ser = np.where(m, pnl, 0.0)
    rec = ROOT / "results/markov_gate_oos_record.csv"
    pd.DataFrame([{
        "deployed": str(deployed), "oos_start": str(dates[0]),
        "oos_end": str(dates[-1]), "gate": prev.winner,
        "days_traded": int(m.sum()), "days_total": len(pnl),
        "gate_pnl": round(float(ser.sum()), 2),
        "baseline_pnl": round(float(pnl.sum()), 2),
        "gate_sharpe": round(sharpe(ser), 3),
        "baseline_sharpe": round(sharpe(pnl), 3),
    }]).to_csv(rec, mode="a", header=not rec.exists(), index=False)
    log(f"OOS record closed for {prev.winner}: gate ${ser.sum():,.0f} vs "
        f"baseline ${pnl.sum():,.0f} over {len(pnl)}d")


def update_config(win) -> str:
    rule, window, thr = win
    text = CONFIG.read_text()
    block = re.search(r"(  markov_gate:\n(?:    .+\n)+)", text)
    if not block:
        sys.exit("ABORT: markov_gate block not found in config")
    if rule == "none":
        new = ("  markov_gate:\n    enabled: false\n"
               f"    # re-opt {date.today()}: 'none' won the sweep — gate disabled\n")
    else:
        new = ("  markov_gate:\n"
               "    enabled: true\n"
               f"    window: {window}\n"
               f"    bull_threshold: {thr}\n"
               f"    bear_threshold: -{thr}\n"
               f"    rule: {rule}\n"
               f"    # re-opt {date.today()} (quarterly method, trailing-130d sweep)\n")
    CONFIG.write_text(text.replace(block.group(1), new))
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=130)
    ap.add_argument("--max-cost-usd", type=float, default=5.0)
    ap.add_argument("--no-config-update", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run even if the last re-opt was < 45 days ago")
    args = ap.parse_args()
    end = date.today() - timedelta(days=1)

    # cadence guard: the timer fires on a fixed calendar (and Persistent=true
    # replays missed fires); the validated cadence is 65-130 trade days, so
    # anything inside 45 calendar days is a duplicate, not a re-optimization
    if HISTORY.exists() and not args.force:
        last = date.fromisoformat(pd.read_csv(HISTORY).run_date.iloc[-1])
        if (date.today() - last).days < 45:
            log(f"last re-opt {last} is < 45 days ago — skipping (use --force)")
            return

    refresh_gspc()
    extend_spy_bars(end)
    backfill_caches(end, args.max_cost_usd)
    regenerate_trades(end)
    score_previous_deployment()
    win, win_sh, base_sh, t0, t1, top5, scores = reoptimize(args.train_days)
    log(f"winner {win}  Sharpe {win_sh:.2f} (baseline {base_sh:.2f}) "
        f"on {t0}..{t1}")

    cfg_note = "config NOT updated (--no-config-update)"
    if not args.no_config_update:
        update_config(win)
        cfg_note = f"config updated -> {win}"
        log(cfg_note)

    HISTORY.parent.mkdir(exist_ok=True)
    pd.DataFrame([{"run_date": str(date.today()), "train_start": str(t0),
                   "train_end": str(t1), "winner": str(win),
                   "winner_sharpe": round(win_sh, 3),
                   "baseline_sharpe": round(base_sh, 3)}]).to_csv(
        HISTORY, mode="a", header=not HISTORY.exists(), index=False)

    rep_dir = ROOT / "logs/markov_gate_reopt"
    rep_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(f"- {k}: Sharpe {scores[k]:.2f}" for k in top5)
    (rep_dir / f"{date.today()}.md").write_text(
        f"# Markov gate re-optimization — {date.today()}\n\n"
        f"Train window: {t0} .. {t1} ({args.train_days} trade days)\n\n"
        f"Winner: **{win}** (Sharpe {win_sh:.2f}; ungated baseline {base_sh:.2f})\n\n"
        f"Top 5:\n{lines}\n\n{cfg_note}\n\n"
        "Method provenance: results/markov_gate_wf_method_2026-06-12.txt; "
        "next re-optimization due in ~65-130 trade days.\n")
    log(f"report: logs/markov_gate_reopt/{date.today()}.md")


if __name__ == "__main__":
    main()
