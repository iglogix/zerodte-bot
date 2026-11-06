#!/usr/bin/env python3
"""Diagnostic sweep to reconcile our 0DTE Jade Lizard backtest with the tastylive
study slides ("SPX 0DTE: Perfecting The Jade Lizard Setup").

Study facts (from the slides):
  - 3yr 0DTE SPX, sampled every 10 min, all fills at MID, times are Chicago (CT).
  - 0DTE variants: $20p/$5c placed at 0.25x EM; $30p/$10c placed ATM.
  - "Closed @X%" = capture X% of max profit (credit). In our code that is an exit
    when combined_mark <= (1 - X) * total_credit, i.e. tp_pct = 1 - X/100.
  - Whole-position management (close the structure when it has captured X%).

Validation target ($30p/$10c ATM):
  capture:  10%   25%   50%   75%   none
  WR%  :    91    85    82    81    79
  P/L  :   +60  +109  +162  +141  +111
  time :  2h05  4h03  5h55  6h22  6h30

Entry = 09:31 ET (the 09:30:00-09:31:00 opening interval; cbbo-1m stamps interval end).
"""
from __future__ import annotations
import argparse
import math
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analytical_jl_eval import load_by_date
from zerodte.backtest import jl_core as jl

ENTRY = "09:31"
ANCHORS = [ENTRY] + [f"{h:02d}:{m:02d}" for h in range(9, 16)
                     for m in range(0, 60, 10) if f"{h:02d}:{m:02d}" > ENTRY] + ["15:59"]
STEP = 5.0
CAPTURES = [0.10, 0.25, 0.50, 0.75, None]  # None = no management (hold to final mid)


def _entry_minutes(anchor: str) -> int:
    return (int(anchor[:2]) * 60 + int(anchor[3:5])) - (9 * 60 + 30)


def eval_config(by_date, put_w, call_w, placement, k, capture,
                require_credit_ge_call=False, sl_mult=0.0, slip=0.0, comm=0.0):
    """Whole-position. placement in {atm, em, callatm_putem, credit_otm, callatm_put_credit}.
    capture None => hold to final available mid. sl_mult>0 => close whole structure when
    mark >= sl_mult*credit. slip (per spread, entry+exit) and comm (per leg) optional.
    Returns dict or None."""
    tp_pct = None if capture is None else (1.0 - capture)
    pnls, times, creds, skips = [], [], [], 0
    for d in sorted(by_date):
        q = by_date[d]["quotes"]
        atm = jl.atm_from_chain(q, ENTRY)
        if atm is None:
            skips += 1; continue
        em = jl.compute_em(q, ENTRY, atm)
        fwd = jl.forward_from_chain(q, ENTRY, atm)
        if not em or em <= 0 or fwd is None:
            skips += 1; continue

        def _credit(sp_, sc_):
            spm_ = jl.quote_mid(q, ENTRY, sp_, False); lpm_ = jl.quote_mid(q, ENTRY, sp_ - put_w, False)
            scm_ = jl.quote_mid(q, ENTRY, sc_, True);  lcm_ = jl.quote_mid(q, ENTRY, sc_ + call_w, True)
            if None in (spm_, lpm_, scm_, lcm_):
                return None
            pc_, cc_ = spm_ - lpm_, scm_ - lcm_
            if pc_ <= 0 or cc_ <= 0:
                return None
            return pc_ + cc_

        if placement == "atm":
            sp = sc = atm
        elif placement == "callatm_putem":
            sc = atm
            sp = math.floor((fwd - k * em) / STEP) * STEP
        elif placement == "credit_otm":
            # Furthest-OTM (symmetric, 5pt grid) where total credit still >= call width.
            chosen = None
            for delta in range(5, 200, 5):
                sp_ = math.floor((fwd - delta) / STEP) * STEP
                sc_ = math.ceil((fwd + delta) / STEP) * STEP
                tot_ = _credit(sp_, sc_)
                if tot_ is None:
                    continue
                if tot_ >= call_w:
                    chosen = (sp_, sc_)   # still no upside risk; keep pushing further out
                else:
                    break                  # credit fell below width; further is worse
            if chosen is None:
                skips += 1; continue
            sp, sc = chosen
        elif placement == "callatm_put_credit":
            # Call spread at ATM (k=0) or k*EM OTM; put placed as FAR OTM as possible
            # while total credit still >= call width (description's "synthetic naked put").
            sc = math.ceil((fwd + (k or 0.0) * em) / STEP) * STEP
            chosen = None
            for dp in range(150, 0, -5):                 # far -> near
                sp_ = math.floor((fwd - dp) / STEP) * STEP
                tot_ = _credit(sp_, sc)
                if tot_ is not None and tot_ >= call_w:
                    chosen = sp_; break                   # furthest OTM put meeting the credit gate
            if chosen is None:
                skips += 1; continue
            sp = chosen
        elif placement == "callatm_put_nogate":
            # NO gate: trade EVERY day. Call at ATM; put as far OTM as still gives
            # credit >= width; if no day reaches that (low-vol days), fall back to the
            # closest put (max credit) and accept sub-width credit / some upside risk.
            sc = math.ceil((fwd + (k or 0.0) * em) / STEP) * STEP
            chosen = None
            fallback = None
            best_c = -1.0
            for dp in range(150, 0, -5):
                sp_ = math.floor((fwd - dp) / STEP) * STEP
                tot_ = _credit(sp_, sc)
                if tot_ is None:
                    continue
                if tot_ >= call_w:
                    chosen = sp_; break
                if tot_ > best_c:
                    best_c = tot_; fallback = sp_
            sp = chosen if chosen is not None else fallback
            if sp is None:
                skips += 1; continue
        elif placement == "atm_straddle":
            # short strikes straddle spot (the literal "ATM" reading); trade every day.
            sp = math.floor(fwd / STEP) * STEP
            sc = math.ceil(fwd / STEP) * STEP
        else:  # "em"
            sp = math.floor((fwd - k * em) / STEP) * STEP
            sc = math.ceil((fwd + k * em) / STEP) * STEP
        lp, lc = sp - put_w, sc + call_w
        spm = jl.quote_mid(q, ENTRY, sp, False); lpm = jl.quote_mid(q, ENTRY, lp, False)
        scm = jl.quote_mid(q, ENTRY, sc, True);  lcm = jl.quote_mid(q, ENTRY, lc, True)
        if None in (spm, lpm, scm, lcm):
            skips += 1; continue
        put_credit, call_credit = spm - lpm, scm - lcm
        total = put_credit + call_credit
        if put_credit <= 0 or call_credit <= 0 or total <= 0:
            skips += 1; continue
        if require_credit_ge_call and total < call_w:
            skips += 1; continue
        creds.append(total)
        strikes = (sp, lp, sc, lc)

        def _capped_mark(a):
            spm_ = jl.quote_mid(q, a, sp, False); lpm_ = jl.quote_mid(q, a, lp, False)
            scm_ = jl.quote_mid(q, a, sc, True);  lcm_ = jl.quote_mid(q, a, lc, True)
            if None in (spm_, lpm_, scm_, lcm_):
                return None
            # a vertical's value is bounded [0, width]; deep-ITM mids overshoot.
            put_mark = min(max(spm_ - lpm_, 0.0), put_w)
            call_mark = min(max(scm_ - lcm_, 0.0), call_w)
            return put_mark + call_mark

        exit_mark = exit_t = last = None
        active = False   # True => closed via TP or stop (a real exit trade); False => held to expiry
        for a in ANCHORS[1:]:
            m = _capped_mark(a)
            if m is None:
                continue
            last = (a, m)
            if sl_mult and m >= sl_mult * total:
                exit_mark, exit_t, active = m, a, True; break
            if tp_pct is not None and m <= tp_pct * total:
                exit_mark, exit_t, active = m, a, True; break
        if exit_mark is None:
            if last is None:
                skips += 1; continue
            exit_t, exit_mark = last  # held to expiry: cash-settles, no exit trade
        gross = (total - exit_mark) * 100
        # entry: slip on 2 spreads + commission on 4 legs (open).
        # exit: slip on 2 spreads ONLY if actively closed (TP/stop); expiry settles free.
        entry_cost = slip * 2 * 100 + comm * 4
        exit_cost = (slip * 2 * 100) if active else 0.0
        pnls.append(gross - (entry_cost + exit_cost))
        times.append(_entry_minutes(exit_t))
    if not pnls:
        return None
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    mt = sum(times) / len(times)
    return dict(n=len(pnls), wr=100 * len(wins) / len(pnls),
                mean=sum(pnls) / len(pnls), total=sum(pnls),
                avg_win=sum(wins) / len(wins) if wins else 0.0,
                avg_loss=sum(losses) / len(losses) if losses else 0.0,
                mean_time_min=mt, avg_credit=sum(creds) / len(creds) if creds else 0.0,
                skips=skips)


def print_row(label, by_date, put_w, call_w, placement, k, gate):
    cells = []
    for cap in CAPTURES:
        r = eval_config(by_date, put_w, call_w, placement, k, cap, gate)
        if r is None:
            cells.append("   n/a   "); continue
        cells.append(f"{r['wr']:4.0f}% {r['mean']:+5.0f} {int(r['mean_time_min']//60)}h{int(r['mean_time_min']%60):02d}")
    n = eval_config(by_date, put_w, call_w, placement, k, 0.50, gate)
    print(f"{label:<34} n={n['n'] if n else 0:>3} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-05-11")
    ap.add_argument("--end", default="2026-02-27")
    ap.add_argument("--cache", default="data/cbbo_cache_jl_10min")
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()
    bd = load_by_date(Path(args.cache), "data/spy_1m.csv",
                      date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"days={len(bd)}  anchors={len(ANCHORS)}  range {args.start}..{args.end}\n")

    # Grid: widths x placement x capture x stop-loss. Whole-position, mid, width-capped.
    WIDTHS = [(10, 5), (20, 5), (20, 10), (30, 10), (10, 10)]
    PLACEMENTS = [("callATM/put-far", "callatm_put_credit", 0.0),
                  ("call0.1EM/put-far", "callatm_put_credit", 0.10),
                  ("sym0.50EM", "em", 0.50), ("sym0.75EM", "em", 0.75), ("sym1.0EM", "em", 1.00)]
    CAPS = [0.25, 0.50, 0.75, None]
    SLS = [0.0, 1.5, 2.0]

    results = []
    for (pw, cw) in WIDTHS:
        for (plabel, pl, k) in PLACEMENTS:
            for cap in CAPS:
                for sl in SLS:
                    r = eval_config(bd, pw, cw, pl, k, cap, sl_mult=sl)
                    if r is None or r["n"] < 200:
                        continue
                    capn = "none" if cap is None else f"{int(cap*100)}%"
                    r["label"] = f"${pw}p/${cw}c {plabel:<16} cap{capn:<4} sl{sl}"
                    results.append(r)

    results.sort(key=lambda r: r["mean"], reverse=True)
    print("TOP configs by mean P/L per trade (mid, zero-cost, width-capped marks):")
    print(f"{'config':<48} {'n':>4} {'WR%':>5} {'mean$':>6} {'total$':>9} {'avgW':>6} {'avgL':>7} {'cred':>5}")
    print("-" * 100)
    for r in results[:args.top]:
        print(f"{r['label']:<48} {r['n']:>4} {r['wr']:>5.1f} {r['mean']:>+6.0f} {r['total']:>+9.0f} "
              f"{r['avg_win']:>+6.0f} {r['avg_loss']:>+7.0f} {r['avg_credit']:>5.1f}")

    # Re-price the top 6 WITH realistic costs (slip 0.10/spread entry+exit, $1/leg) + honest gut-check.
    print("\nTop 6 re-priced WITH costs (slip 0.10/spread, $1/leg commission):")
    print(f"{'config':<48} {'mid_mean$':>9} {'cost_mean$':>10} {'cost_total$':>11} {'WR%':>5}")
    print("-" * 90)
    seen = set()
    for r in results:
        key = r["label"]
        if key in seen:
            continue
        seen.add(key)
        # parse back the config from the stored fields by re-running with costs
        # (simplest: recompute via the same params captured in closure below)
        if len(seen) > 6:
            break
    # Recompute top-6 with costs by re-deriving params from label is fragile; instead
    # re-loop the grid and match the top-6 labels.
    top_labels = [r["label"] for r in results[:6]]
    for (pw, cw) in WIDTHS:
        for (plabel, pl, k) in PLACEMENTS:
            for cap in CAPS:
                for sl in SLS:
                    capn = "none" if cap is None else f"{int(cap*100)}%"
                    lbl = f"${pw}p/${cw}c {plabel:<16} cap{capn:<4} sl{sl}"
                    if lbl not in top_labels:
                        continue
                    rc = eval_config(bd, pw, cw, pl, k, cap, sl_mult=sl, slip=0.10, comm=1.0)
                    rm = eval_config(bd, pw, cw, pl, k, cap, sl_mult=sl)
                    print(f"{lbl:<48} {rm['mean']:>+9.0f} {rc['mean']:>+10.0f} {rc['total']:>+11.0f} {rc['wr']:>5.1f}")


if __name__ == "__main__":
    main()
