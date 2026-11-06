#!/usr/bin/env python3
"""Data-quality audit + per-trade table for the corrected 0DTE Jade Lizard.

Config audited (TT-faithful $30p/$10c): short call ~ATM, short put placed as far
OTM as possible while total credit >= call width; whole-position; close winners at
50% of max profit (tp_pct=0.50); marks capped at spread width; mid fills, zero cost.

Outputs:
  results/jade_lizard/trades_detail.csv   (every trade, rich columns)
  results/jade_lizard/trades_detail.md    (summary + worst/best trades to eyeball)
and prints data-quality diagnostics:
  - SPY*10 (1-min OHLC) vs chain-implied SPX forward, at entry and close
  - where OHLC enters the pipeline (and proof it's not in option pricing)
  - option bid/ask spread widths on the legs (entry + exit), esp. on losers
  - exact-minute quote availability vs +/-5min fallback usage
  - put-call parity residuals (internal chain consistency)
"""
from __future__ import annotations
import csv
import math
import statistics
import sys
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zerodte.backtest.data_loader import BarLoader
from zerodte.backtest import jl_core as jl

ENTRY = "09:31"
ANCHORS = [ENTRY] + [f"{h:02d}:{m:02d}" for h in range(9, 16)
                     for m in range(0, 60, 10) if f"{h:02d}:{m:02d}" > ENTRY] + ["15:59"]
STEP, PUT_W, CALL_W, TP_PCT = 5.0, 30.0, 10.0, 0.50
CACHE = Path("data/cbbo_cache_jl_10min")
OUT = Path("results/jade_lizard")


def q_raw(q, minute, strike, is_call):
    """Exact-minute (bid,ask) or None."""
    return q.get((minute, strike, is_call))


def q_fb(q, minute, strike, is_call, fb=5):
    """(mid, bid, ask, used_minute) using +/-fb fallback; used_minute=='' if missing."""
    v = q.get((minute, strike, is_call))
    used = minute
    if v is None:
        hh, mm = int(minute[:2]), int(minute[3:5])
        for off in [o for d in range(1, fb + 1) for o in (-d, d)]:
            nmm, nhh = mm + off, hh
            while nmm < 0:
                nmm += 60; nhh -= 1
            while nmm >= 60:
                nmm -= 60; nhh += 1
            if 0 <= nhh <= 23:
                v = q.get((f"{nhh:02d}:{nmm:02d}", strike, is_call))
                if v is not None:
                    used = f"{nhh:02d}:{nmm:02d}"; break
    if v is None:
        return None, None, None, ""
    return jl._mid(*v), v[0], v[1], used


def rel_spread(bid, ask):
    if bid is None or ask is None or (bid + ask) <= 0:
        return None
    return (ask - bid) / ((ask + bid) / 2)


def parity_residual(q, minute, atm, fwd):
    """Mean |(call_mid-put_mid) - (fwd-K)| over strikes +/-15 of ATM (chain consistency)."""
    res = []
    for K in [atm - 15, atm - 10, atm - 5, atm + 5, atm + 10, atm + 15]:
        cm = jl.quote_mid(q, minute, K, True, 0)
        pm = jl.quote_mid(q, minute, K, False, 0)
        if cm is None or pm is None:
            continue
        res.append(abs((cm - pm) - (fwd - K)))
    return statistics.mean(res) if res else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loader = BarLoader("data/spy_1m.csv", price_scale=10.0, symbol="SPX")
    rows = []
    gaps_open, gaps_close, exit_spreads, entry_spreads = [], [], [], []
    fb_entry = fb_exit = n_fb_entry = n_fb_exit = 0
    parity = []
    skip_reasons = {"no_atm": 0, "no_em": 0, "no_window_put": 0, "no_entry_quote": 0}

    for day in loader.iter_days(start_date=date(2022, 5, 11), end_date=date(2026, 2, 27)):
        d = day.session_date
        pkl = CACHE / f"{d.isoformat()}.pkl"
        if not pkl.exists():
            continue
        import pickle
        q = pickle.load(open(pkl, "rb"))
        b_open = next((b for b in day.bars if b["ts_et"].time() == time(9, 30)), None)
        b_eod = next((b for b in reversed(day.bars) if b["ts_et"].time() >= time(15, 55)), None)
        spy_open = b_open["open"] if b_open else None
        spy_close = b_eod["close"] if b_eod else None

        atm = jl.atm_from_chain(q, ENTRY)
        if atm is None:
            skip_reasons["no_atm"] += 1; continue
        em = jl.compute_em(q, ENTRY, atm)
        fwd = jl.forward_from_chain(q, ENTRY, atm)
        if not em or em <= 0 or fwd is None:
            skip_reasons["no_em"] += 1; continue
        pr = parity_residual(q, ENTRY, atm, fwd)
        if pr is not None:
            parity.append(pr)
        if spy_open is not None:
            gaps_open.append(spy_open - fwd)

        # NO GATE, trade every day: call ~ATM; put as far OTM as still gives credit >=
        # call width; if no day reaches that (low-vol), fall back to the closest put
        # (max credit) and accept sub-width credit.
        sc = math.ceil(fwd / STEP) * STEP
        sp = None
        fb_put = None
        best_c = -1.0
        for dp in range(150, 0, -5):
            cand = math.floor((fwd - dp) / STEP) * STEP
            spm_, _, _, _ = q_fb(q, ENTRY, cand, False)
            lpm_, _, _, _ = q_fb(q, ENTRY, cand - PUT_W, False)
            scm_, _, _, _ = q_fb(q, ENTRY, sc, True)
            lcm_, _, _, _ = q_fb(q, ENTRY, sc + CALL_W, True)
            if None in (spm_, lpm_, scm_, lcm_):
                continue
            tot_ = (spm_ - lpm_) + (scm_ - lcm_)
            if (spm_ - lpm_) <= 0 or (scm_ - lcm_) <= 0:
                continue
            if tot_ >= CALL_W:
                sp = cand; break
            if tot_ > best_c:
                best_c = tot_; fb_put = cand
        if sp is None:
            sp = fb_put
        if sp is None:
            skip_reasons["no_window_put"] += 1; continue
        lp, lc = sp - PUT_W, sc + CALL_W

        # entry leg quotes
        spm, spb, spa, su = q_fb(q, ENTRY, sp, False)
        lpm, lpb, lpa, lu = q_fb(q, ENTRY, lp, False)
        scm, scb, sca, cu = q_fb(q, ENTRY, sc, True)
        lcm, lcb, lca, lcu = q_fb(q, ENTRY, lc, True)
        if None in (spm, lpm, scm, lcm):
            skip_reasons["no_entry_quote"] += 1; continue
        put_credit, call_credit = spm - lpm, scm - lcm
        total = put_credit + call_credit
        for (b, a) in [(spb, spa), (lpb, lpa), (scb, sca), (lcb, lca)]:
            rs = rel_spread(b, a)
            if rs is not None:
                entry_spreads.append(rs)
        fb_entry += sum(1 for u in (su, lu, cu, lcu) if u != ENTRY)
        n_fb_entry += 4

        # walk anchors @50% capped marks
        def capped_mark(a):
            spm_, _, _, _ = q_fb(q, a, sp, False); lpm_, _, _, _ = q_fb(q, a, lp, False)
            scm_, _, _, _ = q_fb(q, a, sc, True);  lcm_, _, _, _ = q_fb(q, a, lc, True)
            if None in (spm_, lpm_, scm_, lcm_):
                return None
            return min(max(spm_ - lpm_, 0.0), PUT_W) + min(max(scm_ - lcm_, 0.0), CALL_W)

        exit_mark = exit_t = last = None
        for a in ANCHORS[1:]:
            m = capped_mark(a)
            if m is None:
                continue
            last = (a, m)
            if m <= TP_PCT * total:
                exit_mark, exit_t = m, a; break
        if exit_mark is None:
            if last is None:
                continue
            exit_t, exit_mark = last
        kind = "tp" if exit_mark <= TP_PCT * total else "expiry"
        pnl = (total - exit_mark) * 100

        # exit leg quotes (for spread audit)
        xspm, xspb, xspa, xsu = q_fb(q, exit_t, sp, False)
        xlpm, xlpb, xlpa, xlu = q_fb(q, exit_t, lp, False)
        xscm, xscb, xsca, xcu = q_fb(q, exit_t, sc, True)
        xlcm, xlcb, xlca, xlcu = q_fb(q, exit_t, lc, True)
        for (b, a) in [(xspb, xspa), (xlpb, xlpa), (xscb, xsca), (xlcb, xlca)]:
            rs = rel_spread(b, a)
            if rs is not None:
                exit_spreads.append(rs)
        fb_exit += sum(1 for u in (xsu, xlu, xcu, xlcu) if u != exit_t)
        n_fb_exit += 4

        fwd_close = jl.forward_from_chain(q, "15:59", jl.atm_from_chain(q, "15:59") or atm)
        if spy_close is not None and fwd_close is not None:
            gaps_close.append(spy_close - fwd_close)

        rows.append(dict(
            date=d.isoformat(), em=round(em, 1), fwd_open=round(fwd, 1),
            spy_open=round(spy_open, 1) if spy_open else "",
            gap_open=round(spy_open - fwd, 1) if spy_open else "",
            short_put=sp, long_put=lp, short_call=sc, long_call=lc,
            put_dist=round(fwd - sp, 1), call_dist=round(sc - fwd, 1),
            put_credit=round(put_credit, 2), call_credit=round(call_credit, 2),
            total_credit=round(total, 2), kind=kind, exit_anchor=exit_t,
            exit_mark=round(exit_mark, 2), pnl=round(pnl, 1),
            fwd_close=round(fwd_close, 1) if fwd_close else "",
            put_breach=round(sp - fwd_close, 1) if fwd_close else "",
            # widest leg bid/ask at the EXIT anchor (data-quality flag)
            exit_max_relspread=round(max([rel_spread(b, a) for (b, a) in
                                     [(xspb, xspa), (xlpb, xlpa), (xscb, xsca), (xlcb, xlca)]
                                     if rel_spread(b, a) is not None] or [0]), 3),
            sp_exit_bidask=f"{xspb}/{xspa}", lp_exit_bidask=f"{xlpb}/{xlpa}",
        ))

    # ---- write CSV ----
    cols = list(rows[0].keys())
    with open(OUT / "trades_detail.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    pnls = [r["pnl"] for r in rows]
    wins = [p for p in pnls if p > 0]
    losers = sorted(rows, key=lambda r: r["pnl"])[:40]
    best = sorted(rows, key=lambda r: -r["pnl"])[:10]

    def md_table(rs):
        head = "| date | EM | fwd_open | sp | lp | sc | lc | put_d | cred | kind | exit | mark | pnl | fwd_close | breach | exitSprd |\n"
        head += "|" + "---|" * 16 + "\n"
        body = ""
        for r in rs:
            body += (f"| {r['date']} | {r['em']} | {r['fwd_open']} | {r['short_put']:.0f} | {r['long_put']:.0f} | "
                     f"{r['short_call']:.0f} | {r['long_call']:.0f} | {r['put_dist']} | {r['total_credit']} | "
                     f"{r['kind']} | {r['exit_anchor']} | {r['exit_mark']} | {r['pnl']:+.0f} | {r['fwd_close']} | "
                     f"{r['put_breach']} | {r['exit_max_relspread']} |\n")
        return head + body

    with open(OUT / "trades_detail.md", "w") as f:
        f.write("# 0DTE Jade Lizard — per-trade audit ($30p/$10c, call-ATM/put-far, @50%, mid)\n\n")
        f.write(f"- Trades: **{len(rows)}**  | WR **{100*len(wins)/len(rows):.1f}%**  | "
                f"total **${sum(pnls):+,.0f}**  | mean **${sum(pnls)/len(pnls):+.1f}/trade**\n")
        f.write(f"- avg credit ${statistics.mean([r['total_credit'] for r in rows]):.2f}  | "
                f"avg win ${statistics.mean(wins):+.0f}  | "
                f"avg loss ${statistics.mean([p for p in pnls if p<0]):+.0f}\n")
        f.write(f"- **Exit-anchor bid/ask rel-spread**: median "
                f"{statistics.median(exit_spreads):.1%}, 90th pct "
                f"{sorted(exit_spreads)[int(0.9*len(exit_spreads))]:.1%}, max {max(exit_spreads):.1%}\n")
        f.write(f"- **+/-5min fallback used**: entry {fb_entry}/{n_fb_entry} legs "
                f"({100*fb_entry/n_fb_entry:.1f}%), exit {fb_exit}/{n_fb_exit} legs ({100*fb_exit/n_fb_exit:.1f}%)\n")
        f.write(f"- **Put-call parity residual** (chain consistency): median "
                f"${statistics.median(parity):.2f}, 90th pct ${sorted(parity)[int(0.9*len(parity))]:.2f}\n")
        f.write(f"- **SPY×10 vs chain forward** — open gap median ${statistics.median(gaps_open):+.1f}, "
                f"close gap median ${statistics.median(gaps_close):+.1f} (the dividend/tracking basis; "
                f"NOT used for pricing)\n")
        f.write(f"- Skips: {skip_reasons}\n\n")
        f.write("Full table in `trades_detail.csv`. `put_d` = pts short-put is OTM; "
                "`breach` = pts close settled below short put (>0 ITM); "
                "`exitSprd` = widest leg bid/ask rel-spread at exit (data-quality flag).\n\n")
        f.write("## 40 worst trades (are the losses real, or wide-quote artifacts?)\n\n")
        f.write(md_table(losers))
        f.write("\n## 10 best trades\n\n")
        f.write(md_table(best))

    # ---- print diagnostics ----
    print(f"Trades: {len(rows)}  WR {100*len(wins)/len(rows):.1f}%  total ${sum(pnls):+,.0f}  "
          f"mean ${sum(pnls)/len(pnls):+.1f}")
    print(f"Skips: {skip_reasons}")
    print(f"\n1-MIN OHLC (SPY*10) AUDIT:")
    print(f"  open  gap SPY*10 - chain_fwd: median {statistics.median(gaps_open):+.1f}  "
          f"mean {statistics.mean(gaps_open):+.1f}  stdev {statistics.pstdev(gaps_open):.1f}")
    print(f"  close gap SPY*10 - chain_fwd: median {statistics.median(gaps_close):+.1f}  "
          f"mean {statistics.mean(gaps_close):+.1f}  stdev {statistics.pstdev(gaps_close):.1f}")
    print(f"  -> OHLC used ONLY to center the backfill window + settlement fallback; "
          f"NOT in EM/forward/strike/mark (all chain-derived).")
    print(f"\nOPTION QUOTE QUALITY (cbbo-1m):")
    print(f"  entry leg rel-spread: median {statistics.median(entry_spreads):.1%}  "
          f"90pct {sorted(entry_spreads)[int(0.9*len(entry_spreads))]:.1%}")
    print(f"  exit  leg rel-spread: median {statistics.median(exit_spreads):.1%}  "
          f"90pct {sorted(exit_spreads)[int(0.9*len(exit_spreads))]:.1%}  max {max(exit_spreads):.1%}")
    print(f"  +/-5min fallback: entry {100*fb_entry/n_fb_entry:.1f}%  exit {100*fb_exit/n_fb_exit:.1f}%")
    print(f"  put-call parity residual: median ${statistics.median(parity):.2f}  "
          f"90pct ${sorted(parity)[int(0.9*len(parity))]:.2f}  (large => inconsistent quotes)")
    print(f"\nWrote {OUT}/trades_detail.csv and trades_detail.md")


if __name__ == "__main__":
    main()
