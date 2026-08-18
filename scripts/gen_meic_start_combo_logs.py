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

# Pinned validated strategy params (deployed MEIC config). exit_mode/tp set
# per-combo (see EXITS), so they are NOT pinned here.
PINNED = dict(short_em_mult=1.0, wing_em_frac=0.5, stop_pct_of_total_credit=0.70)
STARTS = ("11:00", "11:30", "12:00", "12:30")  # overridden by --starts
EXITS = ("hold",)  # overridden by --exits
CACHE_DIR = None  # overridden by --cache-dir
# Per-fill execution knobs. stop_execution=combo (realistic spread close) is set
# always; stop_slippage bites only stop_hold, entry/tp slippage the rest.
FILL = {
    "nominal": {"slippage_per_spread": 0.10, "stop_slippage_per_spread": 0.25},
    "stress": {"slippage_per_spread": 0.25, "stop_slippage_per_spread": 1.00},
}

_G = {}


def parse_exit(tag):
    """exit tag -> (exit_mode, strategy-override dict).
    'hold' | 'tpNN' (take-profit at NN% credit) | 'slNx' (stop at N x total credit)."""
    if tag == "hold":
        return ("hold_to_expiry", {})
    if tag.startswith("tp"):
        return ("take_profit", {"tp_pct_of_credit": int(tag[2:]) / 100.0})
    if tag.startswith("sl") and tag.endswith("x"):
        return ("stop_hold", {"stop_pct_of_total_credit": float(tag[2:-1])})
    raise ValueError(f"bad exit tag {tag!r} (use 'hold' | 'tpNN' | 'slNx')")


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
    """[(tag, schedule, exit_mode, overrides)] — start x exit grid."""
    out = []
    for s in STARTS:
        for ex in EXITS:
            mode, overrides = parse_exit(ex)
            out.append((f"s{s.replace(':', '')}_{ex}", ladder(s), mode, overrides))
    return out


def _init_parent(fill_name):
    from src_v2.backtest.data_loader import BarLoader

    base = yaml.safe_load((ROOT / "config/config_meic_spx.yaml").read_text())
    # pin params + uniform static sizing so start/exit is the ONLY variable
    st = base["strategy"]
    st.update(PINNED)
    st["sizing_mode"] = "static"
    st["tranche_sizing"] = {}
    st["default_tranche_qty"] = 1
    knobs = FILL[fill_name]
    base["options"]["slippage_per_spread"] = knobs["slippage_per_spread"]
    base["options"]["stop_slippage_per_spread"] = knobs["stop_slippage_per_spread"]
    base["options"]["stop_execution"] = "combo"  # realistic close, not optimistic trigger
    base["backtest"]["end_date"] = str(END)
    if CACHE_DIR:
        base["backtest"]["cbbo_cache_dir"] = CACHE_DIR
    loader = BarLoader(
        base["backtest"]["data_file"], price_scale=base["backtest"]["price_scale"], symbol="SPX"
    )
    days = list(loader.iter_days(start_date=START, end_date=END))
    print(
        f"{len(days)} trading days, chunk={CHUNK_DAYS} "
        f"-> {(len(days) + CHUNK_DAYS - 1) // CHUNK_DAYS} chunks x "
        f"{len(combos())} combos, fill={fill_name} {knobs}",
        flush=True,
    )
    _G["base"] = base
    _G["days"] = days


def run_chunk(ci, parts_dir):
    days = _G["days"][ci * CHUNK_DAYS : (ci + 1) * CHUNK_DAYS]
    if not days:
        return f"SKIP chunk {ci:02d} (empty)"
    from src_v2.backtest.cbbo_store import CbboStore
    from src_v2.backtest.engine import BacktestEngine
    from src_v2.backtest.report import export_trades_csv
    from src_v2.strategies.meic import MeicStrategy

    cbbo_dir = _G["base"]["backtest"]["cbbo_cache_dir"]
    mem = CbboStore.preload_cache(cbbo_dir, start=days[0].session_date, end=days[-1].session_date)
    cbbo = CbboStore(cbbo_dir, preloaded=mem)
    done = 0
    for tag, schedule, exit_mode, overrides in combos():
        part = parts_dir / tag / f"{ci:03d}.csv"
        if part.exists():
            continue
        part.parent.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(_G["base"])
        cfg["entry"]["schedule"] = schedule
        cfg["strategy"]["exit_mode"] = exit_mode
        cfg["strategy"].update(overrides)
        engine = BacktestEngine(cfg, MeicStrategy())
        engine._cbbo = cbbo
        results = engine.run(days)
        export_trades_csv(results, str(part))
        if not part.exists():  # no closed trades in chunk -> empty marker
            part.write_text("session_date,realized_pnl_usd,status\n")
        done += 1
    return f"OK   chunk {ci:02d} ({len(days)}d, {done} combos run, pid {os.getpid()})"


def merge(out_dir, parts_dir):
    for tag, *_ in combos():
        parts = sorted((parts_dir / tag).glob("*.csv"))
        if not parts:
            continue
        pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(
            out_dir / f"combo_{tag}.csv", index=False
        )
    print("merged", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", choices=list(FILL), default="nominal")
    ap.add_argument("--start", default=str(START))
    ap.add_argument("--end", default=str(END))
    ap.add_argument(
        "--starts",
        default=",".join(STARTS),
        help="comma-separated ladder start times, e.g. 09:40,10:00,10:30,11:00",
    )
    ap.add_argument(
        "--exits",
        default=",".join(EXITS),
        help="comma-separated exit tags, e.g. hold,tp40,tp55,tp70",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="override backtest.cbbo_cache_dir (e.g. data/cbbo_cache for pre-11:00)",
    )
    ap.add_argument("--label", default="start", help="output dir = results/meic_wf_<label>_<fill>")
    a = ap.parse_args()
    START = date.fromisoformat(a.start)
    END = date.fromisoformat(a.end)
    STARTS = tuple(s.strip() for s in a.starts.split(","))
    EXITS = tuple(e.strip() for e in a.exits.split(","))
    CACHE_DIR = a.cache_dir
    out_dir = ROOT / f"results/meic_wf_{a.label}_{a.fill}"
    parts_dir = out_dir / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    _init_parent(a.fill)
    n_chunks = (len(_G["days"]) + CHUNK_DAYS - 1) // CHUNK_DAYS
    with Pool(WORKERS) as pool:
        for msg in pool.imap_unordered(partial(run_chunk, parts_dir=parts_dir), range(n_chunks)):
            print(msg, flush=True)
    merge(out_dir, parts_dir)
    print("done", flush=True)
