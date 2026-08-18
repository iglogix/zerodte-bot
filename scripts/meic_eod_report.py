#!/usr/bin/env python3
"""MEIC end-of-day report orchestrator (run by meic-eod-mark.timer at 16:30 ET).

Three steps, in order, each isolated so a later failure never voids an earlier
artifact:
  1. settlement marker  — hold-to-expiry paper P&L vs SPX close (always runs)
  2. fetch real fills   — pull today's actual SPX option fills from the account
  3. reconcile          — real fills vs bot-assumed credits vs backtest model

Steps 2-3 hit the live broker API and are best-effort: if the network/auth is
down, the settlement record from step 1 still stands and the job exits 0.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")
DAY = sys.argv[1] if len(sys.argv) > 1 else str(date.today())


def run(script: str, critical: bool, date_flag: str = "") -> bool:
    # marker takes --date DAY; fetch/reconcile take a positional DAY
    date_args = [date_flag, DAY] if date_flag else [DAY]
    cmd = [PY, str(ROOT / "scripts" / script), *date_args]
    print(f"\n=== {script} {DAY} ===", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    ok = r.returncode == 0
    if not ok and critical:
        print(f"!! {script} failed (critical) — rc={r.returncode}", flush=True)
    elif not ok:
        print(f"-- {script} failed (best-effort, continuing) — rc={r.returncode}",
              flush=True)
    return ok


def main() -> int:
    settle_ok = run("mark_meic_eod.py", critical=True, date_flag="--date")
    fills_ok = run("fetch_meic_fills.py", critical=False)
    if fills_ok and (ROOT / f"results/meic_paper/fills_{DAY}.json").exists():
        run("reconcile_meic_fills.py", critical=False)
    else:
        print("\n(skipping reconcile — no fills fetched)", flush=True)
    # only the settlement marker is allowed to fail the job
    return 0 if settle_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
