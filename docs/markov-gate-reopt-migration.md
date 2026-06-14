# Migrating the quarterly Markov-gate re-optimization to a VPS / cloud host

The job is one systemd user timer driving one deterministic script — no Claude
session, no broker connection, no GUI. This documents exactly what it needs
and how to move it.

## What runs

| Piece | Path (on this machine) |
|---|---|
| Script (the whole pipeline) | `scripts/reopt_markov_gate.py` |
| Timer / service units | `~/.config/systemd/user/markov-gate-reopt.{timer,service}` |
| Config it rewrites | `config/config_power_hour_live.yaml` (`entry.markov_gate` block) |
| State it appends | `results/markov_gate_reopt_history.csv`, `logs/markov_gate_reopt/` |

Cycle (validated method, `results/markov_gate_wf_method_2026-06-12.txt`):
refresh SPX daily closes (Yahoo, keyless) → incrementally extend
`data/spy_1m.csv` + both CBBO caches (Databento, ~$0.01/day, `--max-cost-usd 5`
cap) → regenerate the Power Hour trade log and **abort if the overlap with the
previous log doesn't reproduce exactly** → sweep the gate grid on the trailing
130 trade days → rewrite the config block → log history + report.
A 45-day cadence guard makes duplicate fires no-ops (`--force` overrides).

## What it needs on the target host

1. **The repo** (this git repo; the script resolves all paths relative to it).
2. **Python venv** at `<repo>/.venv` — Python 3.13, with at least:
   `databento`, `pandas`, `numpy`, `pyyaml`, `python-dotenv`, plus whatever
   `scripts/backtest_power_hour.py` imports (it's all in `pip install -e ".[dev]"`).
3. **`.env`** in the repo root containing **only** `DATABENTO_API_KEY=...`.
   Do NOT copy the full local `.env` (it holds TastyTrade/Anthropic/SMTP
   secrets this job never uses — least privilege).
4. **Data directories** (rsync once; the script extends them incrementally):
   - `data/cbbo_cache_fly_10min/` — 2.7 GB
   - `data/cbbo_cache_meic/` — 2.5 GB
   - `data/spy_1m.csv` — 55 MB
   - `results/power_hour_spx/trades_A_triggerplus_ext.csv` — 0.5 MB
     (needed for the overlap-reproduction safety check)
5. Outbound HTTPS to `query1.finance.yahoo.com` and `hist.databento.com`.
   ~512 MB free RAM is plenty; one cycle is ~2 min CPU.

## Steps

```bash
# 0. On the VPS: clone + venv
git clone <repo-url> ~/Dev/0dte_bot && cd ~/Dev/0dte_bot
python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]"
printf 'DATABENTO_API_KEY=%s\n' '<key>' > .env && chmod 600 .env

# 1. Ship the data (from this machine)
rsync -avz data/cbbo_cache_fly_10min data/cbbo_cache_meic data/spy_1m.csv \
      <vps>:~/Dev/0dte_bot/data/
rsync -avz results/power_hour_spx/trades_A_triggerplus_ext.csv \
      results/markov_gate_reopt_history.csv \
      <vps>:~/Dev/0dte_bot/results/power_hour_spx/../   # keep relative layout

# 2. Ship the units (fix WorkingDirectory/ExecStart paths if $HOME differs)
rsync -avz ~/.config/systemd/user/markov-gate-reopt.* \
      <vps>:~/.config/systemd/user/

# 3. On the VPS: headless user timers need lingering, or the timer only
#    runs while you're SSH'd in
sudo loginctl enable-linger $USER
systemctl --user daemon-reload
systemctl --user enable --now markov-gate-reopt.timer

# 4. Verify schedule + do a manual dry cycle (guard will skip if <45d since
#    last run — that proves history travelled; --force + --no-config-update
#    exercises the full pipeline without touching the config)
systemctl --user list-timers markov-gate-reopt.timer
.venv/bin/python scripts/reopt_markov_gate.py --force --no-config-update

# 5. Disable the timer on THIS machine so the job runs in exactly one place
systemctl --user disable --now markov-gate-reopt.timer
```

## Gotchas

- **Timezone.** `OnCalendar=*-03,06,09,12-14 18:37:00` is host-local time and
  was written for `America/New_York`. On a UTC VPS either set the host TZ, or
  change the line to `*-03,06,09,12-14 22:37:00 UTC` (≈18:37 ET; exact hour
  drifts with DST — irrelevant for a quarterly evening job, the script uses
  "yesterday" for all data boundaries regardless of fire time).
- **The config it rewrites is the live bot's config.** If the Power Hour bot
  runs on a *different* host than the re-opt job, the new gate params won't
  reach it — either run both on the same host, or add a step that commits/
  pushes `config/config_power_hour_live.yaml` and have the bot host pull.
  Simplest: keep re-opt and bot together.
- **Persistent=true** replays a missed fire at next boot; the 45-day guard
  turns accidental replays into no-ops. Don't remove the guard.
- **Databento spend** is capped per run (`--max-cost-usd 5`); a quarterly run
  fetching ~65 days costs ≈ $0.75/cache. The cost ledger appends to
  `data/databento_cost_ledger.jsonl`.
- **Failure surface:** the script exits non-zero on overlap divergence or
  fetch failure and the config is then left untouched; check
  `logs/markov_gate_reopt/service.log` and re-run manually with `--force`.
  An untouched config is safe — the bot just keeps the previous quarter's
  gate, and the gate itself fails open in the engine if its data feed breaks.

## Cloud-agent alternative (instead of systemd)

If the VPS step is skipped in favor of Claude Code cloud (`/schedule`), the
routine's prompt is simply: "cd to the repo and run
`.venv/bin/python scripts/reopt_markov_gate.py`, then summarize
`logs/markov_gate_reopt/<today>.md`." All intelligence lives in the script by
design; the agent is only a launcher + reporter there.
