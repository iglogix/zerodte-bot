#!/usr/bin/env python3
"""Phase 1 of the MEIC strategy-param method WF: simulate each combo ONCE over
the full span (0DTE day-independence -> folds are slices, no per-fold re-sim).

Grid (27 combos, baseline = deployed config eh_s1.0_w0.5_p0.7):
  exit_mode hold_to_expiry: short_em_mult {0.8,1.0,1.2} x wing_em_frac
      {0.4,0.5,0.6} (stop_pct ignored by the engine; pinned 0.70 in the tag)
  exit_mode managed:        same axes x stop_pct {0.65, 0.75}

sizing_mode is forced STATIC: fixed-fractional compounding makes daily P&L
depend on equity history, which would break fold-slicing AND day-chunking.

MEMORY DESIGN (v2 — the v1 full-preload + fork was OOM-killed): work is
chunked by DAYS. Each worker task = (chunk of ~64 days), runs ALL combos over
that chunk with its own small lazy CbboStore, writes per-combo PART files to
results/meic_wf_method/parts/<tag>/<chunk>.csv. Day independence makes the
final concatenation exact. Peak memory per worker ~ one chunk of quotes.
Resumable: completed chunk parts are skipped; merge runs at the end.
"""

import copy
import os
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/meic_wf_method"
PARTS = OUT / "parts"
WORKERS = 6
CHUNK_DAYS = 64
START, END = date(2022, 5, 11), date(2026, 6, 11)

_G = {}


def combos():
    out = []
    for mode in ("hold_to_expiry", "managed"):
        stops = (0.70,) if mode == "hold_to_expiry" else (0.65, 0.75)
        for sm in (0.8, 1.0, 1.2):
            for wf in (0.4, 0.5, 0.6):
                for sp in stops:
                    tag = f"e{'h' if mode == 'hold_to_expiry' else 'm'}_s{sm}_w{wf}_p{sp}"
                    out.append((tag, mode, sm, wf, sp))
    return out


def _init_parent():
    from zerodte.backtest.data_loader import BarLoader
    base = yaml.safe_load((ROOT / "config/config_meic_spx.yaml").read_text())
    loader = BarLoader(base["backtest"]["data_file"],
                       price_scale=base["backtest"]["price_scale"], symbol="SPX")
    days = list(loader.iter_days(start_date=START, end_date=END))
    print(f"{len(days)} trading days, chunk={CHUNK_DAYS} "
          f"-> {(len(days) + CHUNK_DAYS - 1) // CHUNK_DAYS} chunks x "
          f"{len(combos())} combos", flush=True)
    _G["base"] = base
    _G["days"] = days


def run_chunk(ci):
    days = _G["days"][ci * CHUNK_DAYS:(ci + 1) * CHUNK_DAYS]
    if not days:
        return f"SKIP chunk {ci:02d} (empty)"
    from zerodte.backtest.cbbo_store import CbboStore
    from zerodte.backtest.engine import BacktestEngine
    from zerodte.strategies.meic import MeicStrategy
    from zerodte.backtest.report import export_trades_csv
    # preload ONLY this chunk's date window (CbboStore(dir) alone is a trap:
    # the lazy fallback reads cache_dir, which defaults to None -> 0 quotes)
    cbbo_dir = _G["base"]["backtest"]["cbbo_cache_dir"]
    mem = CbboStore.preload_cache(cbbo_dir, start=days[0].session_date,
                                  end=days[-1].session_date)
    cbbo = CbboStore(cbbo_dir, preloaded=mem)
    done = 0
    for tag, mode, sm, wf, sp in combos():
        part = PARTS / tag / f"{ci:03d}.csv"
        if part.exists():
            continue
        part.parent.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(_G["base"])
        st = cfg["strategy"]
        st["exit_mode"] = mode
        st["short_em_mult"] = sm
        st["wing_em_frac"] = wf
        st["stop_pct_of_total_credit"] = sp
        st["sizing_mode"] = "static"
        cfg["backtest"]["end_date"] = str(END)
        engine = BacktestEngine(cfg, MeicStrategy())
        engine._cbbo = cbbo
        results = engine.run(days)
        export_trades_csv(results, str(part))
        if not part.exists():  # no closed trades in chunk -> write empty marker
            part.write_text("session_date,realized_pnl_usd,status\n")
        done += 1
    return f"OK   chunk {ci:02d} ({len(days)}d, {done} combos run, pid {os.getpid()})"


def merge():
    for tag, *_ in combos():
        parts = sorted((PARTS / tag).glob("*.csv"))
        if not parts:
            continue
        pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) \
            .to_csv(OUT / f"combo_{tag}.csv", index=False)
    print("merged", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    _init_parent()
    n_chunks = (len(_G["days"]) + CHUNK_DAYS - 1) // CHUNK_DAYS
    with Pool(WORKERS) as pool:   # fork: children inherit _G (bars only)
        for msg in pool.imap_unordered(run_chunk, range(n_chunks)):
            print(msg, flush=True)
    merge()
    print("done", flush=True)
