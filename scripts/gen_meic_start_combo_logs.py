#!/usr/bin/env python3
"""Phase 1 of the MEIC START-TIME method WF: simulate each ladder-START variant
ONCE over the full span (0DTE day-independence -> folds are slices).

Question (Phase 2, run_wf_method_strategy.py): does periodically re-optimizing
the entry-ladder START time beat always starting at 11:00 (the deployed start),
OOS? All other strategy params are PINNED at the validated deployed values
(hold_to_expiry, short_em_mult 1.0, wing_em_frac 0.5, stop_pct 0.70) and sizing
is UNIFORM static (default_tranche_qty 1, no weight map) so the ONLY moving part
is the start time. This mirrors the View-2 start-shift sweep in
scripts/sweep_meic_start.py that produced the 11:30 signal.

Grid (4 combos): fixed-length 13-slot, 20-min-spaced ladder, capped at 15:40,
start in {11:00, 11:30, 12:00, 12:30}. Baseline tag = s1100.

Data: cbbo_cache_meic (11:00+ quotes, FULL SPAN 2022-2026 — every candidate is
>= 11:00 so no early-morning backfill is needed). Under hold_to_expiry there are
no stops, so the relevant execution stress is ENTRY slippage: --fill nominal
(slippage_per_spread 0.10) vs --fill stress (0.25). Results go to separate dirs
so the WF picks among starts within ONE fill regime, not across fills.

Memory design copied verbatim from gen_meic_combo_logs.py (day-chunked, lazy
per-chunk CbboStore, resumable parts + merge). 4 combos << the 27 there, so this
is several x faster.
"""

import argparse
import copy
import os
import sys
from datetime import date
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WORKERS = 6
CHUNK_DAYS = 64
START, END = date(2022, 5, 11), date(2026, 6, 11)

# Pinned validated strategy params (deployed MEIC config).
PINNED = dict(exit_mode="hold_to_expiry", short_em_mult=1.0,
              wing_em_frac=0.5, stop_pct_of_total_credit=0.70)
STARTS = ("11:00", "11:30", "12:00", "12:30")
FILL_SLIP = {"nominal": 0.10, "stress": 0.25}

_G = {}


def ladder(start_hhmm, n=13, step_min=20, last="15:40"):
    h, m = int(start_hhmm[:2]), int(start_hhmm[3:])
    lh, lm = int(last[:2]), int(last[3:])
    out = []
    for _ in range(n):
        if (h, m) > (lh, lm):
            break
        out.append(f"{h:02d}:{m:02d}")
        m += step_min
        while m >= 60:
            m -= 60
            h += 1
    return out


def combos():
    """[(tag, schedule_list)] — one per start time."""
    return [(f"s{s.replace(':', '')}", ladder(s)) for s in STARTS]


def _init_parent(slip):
    from src_v2.backtest.data_loader import BarLoader
    base = yaml.safe_load((ROOT / "config/config_meic_spx.yaml").read_text())
    # pin params + uniform static sizing so start time is the ONLY variable
    st = base["strategy"]
    st.update(PINNED)
    st["sizing_mode"] = "static"
    st["tranche_sizing"] = {}
    st["default_tranche_qty"] = 1
    base["options"]["slippage_per_spread"] = slip
    base["backtest"]["end_date"] = str(END)
    loader = BarLoader(base["backtest"]["data_file"],
                       price_scale=base["backtest"]["price_scale"], symbol="SPX")
    days = list(loader.iter_days(start_date=START, end_date=END))
    print(f"{len(days)} trading days, chunk={CHUNK_DAYS} "
          f"-> {(len(days) + CHUNK_DAYS - 1) // CHUNK_DAYS} chunks x "
          f"{len(combos())} combos, slip={slip}", flush=True)
    _G["base"] = base
    _G["days"] = days


def run_chunk(ci, parts_dir):
    days = _G["days"][ci * CHUNK_DAYS:(ci + 1) * CHUNK_DAYS]
    if not days:
        return f"SKIP chunk {ci:02d} (empty)"
    from src_v2.backtest.cbbo_store import CbboStore
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.strategies.meic import MeicStrategy
    from src_v2.backtest.report import export_trades_csv
    cbbo_dir = _G["base"]["backtest"]["cbbo_cache_dir"]
    mem = CbboStore.preload_cache(cbbo_dir, start=days[0].session_date,
                                  end=days[-1].session_date)
    cbbo = CbboStore(cbbo_dir, preloaded=mem)
    done = 0
    for tag, schedule in combos():
        part = parts_dir / tag / f"{ci:03d}.csv"
        if part.exists():
            continue
        part.parent.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(_G["base"])
        cfg["entry"]["schedule"] = schedule
        engine = BacktestEngine(cfg, MeicStrategy())
        engine._cbbo = cbbo
        results = engine.run(days)
        export_trades_csv(results, str(part))
        if not part.exists():  # no closed trades in chunk -> empty marker
            part.write_text("session_date,realized_pnl_usd,status\n")
        done += 1
    return f"OK   chunk {ci:02d} ({len(days)}d, {done} combos run, pid {os.getpid()})"


def merge(out_dir, parts_dir):
    for tag, _ in combos():
        parts = sorted((parts_dir / tag).glob("*.csv"))
        if not parts:
            continue
        pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) \
            .to_csv(out_dir / f"combo_{tag}.csv", index=False)
    print("merged", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", choices=list(FILL_SLIP), default="nominal")
    ap.add_argument("--start", default=str(START))
    ap.add_argument("--end", default=str(END))
    a = ap.parse_args()
    START = date.fromisoformat(a.start)
    END = date.fromisoformat(a.end)
    out_dir = ROOT / f"results/meic_wf_start_{a.fill}"
    parts_dir = out_dir / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    _init_parent(FILL_SLIP[a.fill])
    n_chunks = (len(_G["days"]) + CHUNK_DAYS - 1) // CHUNK_DAYS
    with Pool(WORKERS) as pool:
        for msg in pool.imap_unordered(partial(run_chunk, parts_dir=parts_dir),
                                       range(n_chunks)):
            print(msg, flush=True)
    merge(out_dir, parts_dir)
    print("done", flush=True)
