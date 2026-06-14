#!/usr/bin/env python3
"""Phase 1 of the strategy-param method WF: simulate each param combo ONCE
over the full span. 0DTE days are independent, so walk-forward folds can be
sliced from these logs without re-simulation (same trick as the gate study,
one level deeper).

Grid (3 effective knobs, respects the 10x data-points rule on 130d trains):
  credit_scale x stop_scale in {0.75, 1.0, 1.25, 1.5}^2, ratchet in {on, off}
  (1.0, 1.0, on) = the as-taught Version A baseline.

Writes results/power_hour_wf_method/combo_c<credit>_s<stop>_r<0|1>.csv;
existing logs are skipped, so the script is resumable.
"""

import copy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")
OUT = ROOT / "results/power_hour_wf_method"
SCALES = (0.75, 1.0, 1.25, 1.5)
WORKERS = 6

base = yaml.safe_load((ROOT / "config/config_power_hour_spx.yaml").read_text())
(OUT / "configs").mkdir(parents=True, exist_ok=True)

jobs = []
for cs in SCALES:
    for ss in SCALES:
        for ratchet in (True, False):
            tag = f"c{cs}_s{ss}_r{int(ratchet)}"
            log = OUT / f"combo_{tag}.csv"
            if log.exists():
                continue
            cfg = copy.deepcopy(base)
            for tr in cfg["strategy"]["tranches"]:
                tr["target_credit"] = round(tr["target_credit"] * cs, 2)
                tr["stop"] = round(tr["stop"] * ss, 2)
                if not ratchet:
                    tr["ratchet_add"] = None
            cfg["backtest"]["end_date"] = "2026-06-11"
            cfg_path = OUT / "configs" / f"{tag}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            jobs.append((tag, cfg_path, log))


def run(job):
    tag, cfg_path, log = job
    r = subprocess.run(
        [PY, str(ROOT / "scripts/backtest_power_hour.py"),
         "--config", str(cfg_path), "--start", "2022-05-11",
         "--end", "2026-06-11", "--out", str(log)],
        cwd=ROOT, capture_output=True, text=True)
    ok = r.returncode == 0 and log.exists()
    print(f"{'OK ' if ok else 'FAIL'} {tag}" + ("" if ok else f"\n{r.stderr[-400:]}"),
          flush=True)
    return ok


print(f"{len(jobs)} combos to simulate ({WORKERS} workers)")
with ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(run, jobs))
print(f"done: {sum(results)}/{len(jobs)} succeeded")
sys.exit(0 if all(results) else 1)
