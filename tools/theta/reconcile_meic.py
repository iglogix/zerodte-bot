#!/usr/bin/env python3
"""Independent reconciliation of the MEIC backtest (results/meic_spx/trades.csv).

Two independent layers, by design:

  Layer A — ACCOUNTING IDENTITY (no quotes needed)
      Recompute realized_pnl_usd from the *recorded* credit/debit using the
      engine's documented cost model and confirm the engine's arithmetic:
          realized = (credit - slippage - debit) * 100 * qty  -  2*qty*commission
      (debit = 0 on expiry; slippage applied once at entry; commission upfront)
      This catches P&L accounting bugs. It does NOT touch market data.

  Layer B — QUOTE RE-PRICING (independent code path over the quote source)
      Re-derive each spread's ENTRY CREDIT directly from bid/ask quotes at the
      recorded entry minute and compare to the recorded credit_received. This
      catches look-ahead, wrong-minute/strike pricing, and (when run against a
      DIFFERENT vendor) data-vendor disagreement.

Quote source is pluggable:
  --source databento  -> reads the project's own pickle cache (same DATA, but a
                          fully independent reader/lookup). Validates the engine.
  --source theta      -> queries the local Theta Terminal REST API (independent
                          DATA + engine). Requires a paid tier for intraday quotes.

Usage:
  .venv/bin/python tools/theta/reconcile_meic.py --source databento
  .venv/bin/python tools/theta/reconcile_meic.py --source databento --limit 500
  .venv/bin/python tools/theta/reconcile_meic.py --source theta --sample 300
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

REPO = Path("~/Dev/0dte_bot")
TRADES_CSV = REPO / "results/meic_spx/trades.csv"
DBN_CACHE = REPO / "data/cbbo_cache_meic"
THETA = "http://127.0.0.1:25510"

# Cost model (from config/config_meic_spx.yaml -> options:)
SLIPPAGE = 0.10
COMMISSION_PER_LEG = 1.0
MULT = 100

Quote = Tuple[float, float]  # (bid, ask)


# --------------------------------------------------------------------------- #
# Quote sources
# --------------------------------------------------------------------------- #
class QuoteSource:
    def get(self, d: date, minute: str, strike: float, is_call: bool) -> Optional[Quote]:
        raise NotImplementedError


class DatabentoCache(QuoteSource):
    """Independent reader of the project's pickle cache (same data, fresh code)."""

    def __init__(self, cache_dir: Path = DBN_CACHE, max_stale: int = 5):
        self.cache_dir = cache_dir
        self.max_stale = max_stale
        self._day: Optional[date] = None
        self._q: dict = {}

    def _load(self, d: date) -> None:
        if d == self._day:
            return
        p = self.cache_dir / f"{d.isoformat()}.pkl"
        self._q = pickle.load(open(p, "rb")) if p.exists() else {}
        self._day = d

    def get(self, d, minute, strike, is_call):
        self._load(d)
        if not self._q:
            return None
        key = (minute, strike, is_call)
        if key in self._q:
            return self._q[key]
        hh, mm = int(minute[:2]), int(minute[3:5])
        for off in range(1, self.max_stale + 1):
            pm, ph = mm - off, hh
            while pm < 0:
                pm += 60
                ph -= 1
            if ph < 0:
                break
            k = (f"{ph:02d}:{pm:02d}", strike, is_call)
            if k in self._q:
                return self._q[k]
        return None


class ThetaTerminal(QuoteSource):
    """Local Theta Terminal REST. Caches the per-minute series per contract/day."""

    def __init__(self, base: str = THETA, max_stale: int = 5):
        self.base = base
        self.max_stale = max_stale
        self._cache: dict = {}  # (d, strike, is_call) -> {minute: (bid, ask)}

    def _series(self, d: date, strike: float, is_call: bool) -> dict:
        ck = (d, strike, is_call)
        if ck in self._cache:
            return self._cache[ck]
        ds = d.strftime("%Y%m%d")
        url = (f"{self.base}/v2/hist/option/quote?root=SPXW&exp={ds}"
               f"&strike={int(round(strike * 1000))}&right={'C' if is_call else 'P'}"
               f"&start_date={ds}&end_date={ds}&ivl=60000&rth=true")
        out: dict = {}
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            fmt = data["header"]["format"]
            i_ms, i_bid, i_ask = fmt.index("ms_of_day"), fmt.index("bid"), fmt.index("ask")
            for row in data["response"]:
                m = row[i_ms] // 60000
                out[f"{m // 60:02d}:{m % 60:02d}"] = (row[i_bid], row[i_ask])
        except Exception:
            out = {}
        self._cache[ck] = out
        return out

    def get(self, d, minute, strike, is_call):
        s = self._series(d, strike, is_call)
        if minute in s:
            return s[minute]
        hh, mm = int(minute[:2]), int(minute[3:5])
        for off in range(1, self.max_stale + 1):
            pm, ph = mm - off, hh
            while pm < 0:
                pm += 60
                ph -= 1
            if ph < 0:
                break
            k = f"{ph:02d}:{pm:02d}"
            if k in s:
                return s[k]
        return None


def mid(q: Optional[Quote]) -> Optional[float]:
    return None if q is None else (q[0] + q[1]) / 2


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def parse_trades(limit: Optional[int]):
    rows = []
    with open(TRADES_CSV) as f:
        for r in csv.DictReader(f):
            if not r["entry_time"]:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


# Exits whose P&L is fully reconstructable from the single recorded
# (credit, debit) pair. Partial/scaled exits book intermediate take-profit
# fills that the per-row CSV does not store, so they cannot be reconciled
# from the CSV alone (they need full re-simulation).
RECONSTRUCTABLE = ("expired_worthless", "short_stopped")


def expected_pnl(credit, debit, qty, exit_reason):
    """Engine's documented identity for single-exit trades.
    expired_worthless -> debit 0 (full credit kept).
    short_stopped     -> recorded debit (one closing fill).
    """
    d = 0.0 if exit_reason == "expired_worthless" else debit
    return round((credit - SLIPPAGE - d) * MULT * qty - 2 * qty * COMMISSION_PER_LEG, 2)


def run(source: QuoteSource, limit, sample_stride):
    rows = parse_trades(limit)
    n = len(rows)

    # ---- Layer A: accounting identity (segmented by exit type) -------------
    a_err = defaultdict(list)       # reason -> [|error|] for reconstructable
    a_bad = []                      # reconstructable rows off by > $1
    a_implied = defaultdict(list)   # non-reconstructable -> implied scale-out $
    for r in rows:
        credit = float(r["credit_received"])
        debit = float(r["debit_paid"]) if r["debit_paid"] else 0.0
        qty = int(r["qty"])
        rec = float(r["realized_pnl_usd"])
        reason = r["exit_reason"]
        if reason in RECONSTRUCTABLE:
            exp = expected_pnl(credit, debit, qty, reason)
            e = round(rec - exp, 2)
            a_err[reason].append(abs(e))
            if abs(e) > 1.0:
                a_bad.append((r["trade_id"], reason, rec, exp, e))
        else:
            # full-credit-kept baseline; gap = cash given up to scale-outs
            naive = round((credit - SLIPPAGE) * MULT * qty - 2 * qty * COMMISSION_PER_LEG, 2)
            a_implied[reason].append(rec - naive)

    # ---- Layer B: entry-credit re-pricing from quotes ----------------------
    b_diffs = []
    b_missing = 0
    b_total = 0
    b_by_reason = defaultdict(list)
    worst = []
    for i, r in enumerate(rows):
        if sample_stride and (i % sample_stride):
            continue
        b_total += 1
        d = date.fromisoformat(r["session_date"])
        minute = r["entry_time"][11:16]  # "HH:MM" (offset is ET already)
        is_call = r["spread_type"] == "call"
        sq = source.get(d, minute, float(r["short_strike"]), is_call)
        lq = source.get(d, minute, float(r["long_strike"]), is_call)
        sm, lm = mid(sq), mid(lq)
        if sm is None or lm is None:
            b_missing += 1
            continue
        recomputed = sm - lm
        recorded = float(r["credit_received"])
        diff = round(recomputed - recorded, 4)
        b_diffs.append(diff)
        b_by_reason[r["exit_reason"]].append(diff)
        worst.append((abs(diff), r["trade_id"], minute, r["spread_type"],
                      r["short_strike"], r["long_strike"], recorded, round(recomputed, 3),
                      sq, lq))

    # ---- Report -------------------------------------------------------------
    print("=" * 78)
    print(f"MEIC RECONCILIATION  |  source={source.__class__.__name__}  |  trades={n}")
    print("=" * 78)

    print("\n[Layer A] Accounting identity  realized == (credit-0.10-debit)*100*qty - 2*qty*$1")
    print("  Reconstructable exits (single fill) — should hold to rounding:")
    for reason in RECONSTRUCTABLE:
        errs = a_err.get(reason, [])
        if errs:
            print(f"     {reason:20s} n={len(errs):5d}  mean|err|=${statistics.mean(errs):.3f}"
                  f"  max|err|=${max(errs):.2f}")
    print(f"  reconstructable rows > $1 off : {len(a_bad)}")
    for tid, er, rec, exp, e in a_bad[:8]:
        print(f"     {tid:28s} {er:20s} rec={rec:>9.2f} exp={exp:>9.2f} d={e:+.2f}")
    print("  Non-reconstructable exits (scale-outs not stored in CSV) — needs re-sim:")
    for reason, vals in sorted(a_implied.items()):
        print(f"     {reason:20s} n={len(vals):5d}  implied scale-out cash:"
              f" mean=${statistics.mean(vals):+.1f}  total=${sum(vals):+,.0f}")

    print("\n[Layer B] Entry-credit re-priced from quotes vs recorded credit_received")
    print(f"  spreads re-priced : {b_total}  (stride={sample_stride or 1})")
    print(f"  quotes found      : {len(b_diffs)}   missing/stale: {b_missing}"
          f"  ({100*b_missing/max(b_total,1):.1f}%)")
    if b_diffs:
        ad = [abs(x) for x in b_diffs]
        print(f"  mean diff (recomp - recorded) : {statistics.mean(b_diffs):+.4f} pts")
        print(f"  mean |diff|                   : {statistics.mean(ad):.4f} pts")
        print(f"  median |diff|                 : {statistics.median(ad):.4f} pts")
        print(f"  p90  |diff|                   : {sorted(ad)[int(0.9*len(ad))-1]:.4f} pts")
        print(f"  max  |diff|                   : {max(ad):.4f} pts")
        within = sum(1 for x in ad if x <= 0.05)
        print(f"  within $0.05                  : {within}/{len(ad)} "
              f"({100*within/len(ad):.1f}%)")
        print("  by exit_reason (mean signed diff):")
        for k in sorted(b_by_reason):
            v = b_by_reason[k]
            print(f"     {k:22s} n={len(v):5d}  mean={statistics.mean(v):+.4f}"
                  f"  mean|.|={statistics.mean([abs(x) for x in v]):.4f}")
        print("  worst 8 entry-credit mismatches:")
        for ad_, tid, mn, st, ss, ls, rc, rp, sq, lq in sorted(worst, reverse=True)[:8]:
            print(f"     {tid:26s} {mn} {st:4s} K{ss}/{ls}"
                  f"  recorded={rc:6.2f} recomp={rp:6.3f}  short={sq} long={lq}")
    else:
        print("  No quotes returned — (Theta free tier blocks intraday; needs VALUE tier.)")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["databento", "theta"], default="databento")
    ap.add_argument("--limit", type=int, default=None, help="cap rows parsed (debug)")
    ap.add_argument("--sample", type=int, default=None,
                    help="re-price every Nth spread in Layer B (default: all)")
    args = ap.parse_args()
    src = DatabentoCache() if args.source == "databento" else ThetaTerminal()
    run(src, args.limit, args.sample)
