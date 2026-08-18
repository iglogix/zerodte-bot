#!/usr/bin/env python3
"""EOD settlement marker for MEIC manual/paper sessions.

MEIC's manual path logs assumed entries to logs/execution_comparison/<date>.jsonl
but (under hold_to_expiry) never persists a DB trade row, so nobody computes the
day's paper P&L. This script closes that gap: it marks each assumed entry at the
SPX cash-settlement print and computes hold-to-expiry P&L using the SAME formula
as the backtest engine (BacktestEngine._simulate_meic_side), so paper results are
directly comparable to the validated backtest numbers.

Per side (put spread / call spread), per the engine:
    credit_eff = assumed_credit - slippage_per_spread
    commission = 2 * qty * commission_per_contract_per_leg   # TT SPX: $1 open, $0 close
    intrinsic  = max(0, short-settle)  [put]  |  max(0, settle-short)  [call]
    loss       = min(intrinsic, |short_strike - long_strike|)
    side_pnl   = (credit_eff - loss) * 100 * qty - commission

Settlement: SPXW 0DTE is PM-settled at the 16:00 ET index print; we use the SPX
(^GSPC) daily close as the settlement proxy (documented approximation; the live
QuoteComparator separately captures actual-vs-assumed ENTRY fill quality).

Writes results/meic_paper/<date>.csv (per-trade) and appends a daily row to
results/meic_paper/daily_pnl.csv. Idempotent: re-running overwrites the day.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path

import yaml

def _meic_log(root, day):
    """execution_comparison/meic/<date>.jsonl, falling back to the pre-split
    flat path for files written before strategies were namespaced."""
    from pathlib import Path as _P
    ns = _P(root) / "logs/execution_comparison/meic" / f"{day}.jsonl"
    return ns if ns.exists() else _P(root) / "logs/execution_comparison" / f"{day}.jsonl"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/meic_paper"


def spx_settlement(d: date) -> float | None:
    """SPX (^GSPC) close for date d via Yahoo chart API (keyless)."""
    p1 = int(datetime(d.year, d.month, d.day).timestamp()) - 86400
    p2 = p1 + 4 * 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
           f"?period1={p1}&period2={p2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.load(r)["chart"]["result"][0]
    except Exception as exc:
        print(f"settlement fetch failed: {exc}", file=sys.stderr)
        return None
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    for t, c in zip(ts, closes):
        if c and datetime.fromtimestamp(t).date() == d:
            return float(c)
    # fall back to the last available close on/before d
    cands = [(t, c) for t, c in zip(ts, closes)
             if c and datetime.fromtimestamp(t).date() <= d]
    return float(cands[-1][1]) if cands else None


def side_pnl(leg: dict, settle: float, is_call: bool, qty: int,
             slip: float, comm_per_leg: float, mult: int) -> dict | None:
    short = leg.get("short_strike")
    long_ = leg.get("long_strike")
    credit = leg.get("assumed_credit")
    if short is None or long_ is None or credit is None:
        return None
    width = abs(short - long_)
    credit_eff = credit - slip
    commission = 2 * qty * comm_per_leg
    intrinsic = max(0.0, settle - short) if is_call else max(0.0, short - settle)
    loss = min(intrinsic, width)
    pnl = (credit_eff - loss) * mult * qty - commission
    return {"short_strike": short, "long_strike": long_, "width": width,
            "credit": round(credit, 2), "intrinsic": round(intrinsic, 2),
            "loss": round(loss, 2), "pnl": round(pnl, 2),
            "outcome": "max_loss" if loss >= width - 1e-9
                       else ("breached" if loss > 0 else "expired_otm")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--config", default="config/config_meic_spx.yaml")
    ap.add_argument("--settle", type=float, default=None,
                    help="override SPX settlement (else fetched from Yahoo)")
    args = ap.parse_args()
    d = date.fromisoformat(args.date)

    jsonl = _meic_log(ROOT, d.isoformat())
    if not jsonl.exists():
        print(f"no execution log for {d} — MEIC did not run or generated nothing")
        return 0
    entries = []
    for line in jsonl.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "assumed" and r.get("kind") == "entry" \
                and str(r.get("trade_id", "")).startswith("meic"):
            entries.append(r)
    if not entries:
        print(f"{d}: 0 MEIC assumed entries logged")
        return 0

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    opt = cfg["options"]
    slip = opt.get("slippage_per_spread", 0.10)
    comm = opt.get("commission_per_contract_per_leg", 1.0)
    mult = opt.get("contract_multiplier", 100)

    settle = args.settle if args.settle is not None else spx_settlement(d)
    if settle is None:
        print("ABORT: no settlement price", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    rows, day_pnl = [], 0.0
    for e in entries:
        qty = int(e.get("quantity") or 1)
        legs = e.get("legs", {})
        p = side_pnl(legs.get("short_put", {}), settle, False, qty, slip, comm, mult)
        c = side_pnl(legs.get("short_call", {}), settle, True, qty, slip, comm, mult)
        if p is None or c is None:
            print(f"  SKIP {e.get('trade_id')}: missing strikes (pre-fix entry?)")
            continue
        tot = round(p["pnl"] + c["pnl"], 2)
        day_pnl += tot
        # CapReq from the order if present (older entries: recompute from strikes)
        capreq = e.get("capreq_usd")
        if capreq is None:  # pre-capreq entry: recompute from exact total_credit
            wider = max(p["width"], c["width"])
            net = e.get("total_credit", p["credit"] + c["credit"])
            capreq = round((wider - net) * 100 * qty, 2)
        rows.append({"trade_id": e["trade_id"], "qty": qty, "settle": settle,
                     "capreq_usd": capreq,
                     "put_pnl": p["pnl"], "put_outcome": p["outcome"],
                     "call_pnl": c["pnl"], "call_outcome": c["outcome"],
                     "total_pnl": tot})

    per_day = OUT / f"{d.isoformat()}.csv"
    with per_day.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # hold_to_expiry keeps every tranche open to settlement, so peak BP = sum
    peak_capreq = round(sum(r["capreq_usd"] for r in rows), 2)

    daily = OUT / "daily_pnl.csv"
    cols = ["date", "trades", "settle", "peak_capreq_usd", "paper_pnl_usd", "roc_pct"]
    prior = [r for r in csv.DictReader(daily.open())] if daily.exists() else []
    prior = [r for r in prior if r["date"] != d.isoformat()]   # idempotent
    prior.append({"date": d.isoformat(), "trades": str(len(rows)),
                  "settle": f"{settle:.2f}", "peak_capreq_usd": f"{peak_capreq:.2f}",
                  "paper_pnl_usd": f"{day_pnl:.2f}",
                  "roc_pct": f"{(day_pnl / peak_capreq * 100) if peak_capreq else 0:.2f}"})
    prior.sort(key=lambda r: r["date"])
    with daily.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(prior)

    print(f"{d}: {len(rows)} MEIC ICs, settle {settle:.2f}, "
          f"peak CapReq ${peak_capreq:,.0f}, paper P&L ${day_pnl:,.2f} "
          f"(RoC {day_pnl / peak_capreq * 100 if peak_capreq else 0:+.1f}%) -> {per_day}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
