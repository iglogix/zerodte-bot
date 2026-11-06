Review today's (2026-06-11) Power Hour manual-mode bot run and write a report. This is a READ-ONLY review: do not modify code, configs, or the database; your only write is the report file.

Context: the bot ran `zerodte.main --manual --config config/config_power_hour_live.yaml` from a systemd timer (unit power-hour-manual) between 14:40 and ~16:10 ET. Manual mode = live TastyTrade data, orders generated but NOT submitted. Strategy: 3 iron-condor tranches at 14:55 ($4/side target), 15:15 ($3/side), 15:25 ($2/side), ~$0.05 wings, combined-strangle stops $4/$3/$3 with one-time ratchet (+$3/+$2/none) at 40% profit, forced close 15:55. Spec: docs/0DTE-Power-Hour-Strategy.md. Backtest reference: results/power_hour_spx/trades_A_triggerplus.csv and the engine in zerodte/backtest/power_hour.py.

Inspect:
1. logs/2026-06-11/power_hour.log — did the bot start, authenticate, fire all three entry cycles, and close by 15:55? Note any errors, exceptions, or skipped cycles (and why).
2. logs/decisions/2026-06-11/ (if present) — per-cycle decision records.
3. data/power_hour_bot.db (sqlite3) — inspect the schema first, then pull today's generated trades/orders: strikes, credits, quantities, stop levels, exit events/times.
4. logs/execution_comparison/ (if today's files exist) — quote comparator output.

Compare against the backtest's assumptions (the quote caches end 2026-02, so you cannot re-run the backtest for today; compare against its historical distribution instead):
- Entry credits per side vs the $4/$3/$2 targets, and the combined strangle credit vs the historical distribution in trades_A_triggerplus.csv (also note today's premium regime: above or below the ~$6-7 strangle floor where the backtest says the strategy loses money).
- Wing selection: asks near $0.05, widths within the historical 20-115pt range.
- Stop levels = entry strangle + 4/3/3, ratchet events if any, and whether the 15:55 forced close fired on time.
- Anything the live code did that the backtest engine would not have done (selection differences, skipped tranches, dislocation guards).

Write the report to logs/2026-06-11/power_hour_review.md (create it). Structure: TLDR verdict (did the run validate the backtest's entry assumptions? any bugs?), per-tranche table (entry time, strikes, credits, stop, exit, notes), discrepancies vs backtest assumptions, errors found, and recommended next steps. Be specific and quote log lines for anything anomalous. If the bot did not run or crashed, say so plainly in the TLDR and diagnose from the journal: `journalctl --user -u power-hour-manual.service --since today`.
