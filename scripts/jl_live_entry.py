#!/usr/bin/env python3
"""Live SPX 0DTE Jade Lizard entry generator (manual mode = live prod quotes, NO orders).

Run at ~09:30 ET. Prints, for manual entry into the TastyTrade GUI and side-by-side
comparison with TT's displayed values:
  - SPX spot, ATM strike
  - EM two ways: ATM straddle (our backtest convention) AND the 16-delta +/-1 SD
    (what TT's chain shows as the expected move) -> so we can see which TT uses
  - The canonical study setups with per-leg bid/ask + total credit:
       A) $30p/$10c, shorts ATM            (the slide's headline variant)
       B) $20p/$5c,  shorts at 0.25x EM
       C) $30p/$10c, shorts ~1x straddle   (our most cost-robust variant)
  - An ATM chain snapshot (strike | put bid/ask/delta | call bid/ask/delta)

Output goes to stdout AND logs/jl_live_entry/<date>.txt.

Usage (manual):   .venv/bin/python -m scripts.jl_live_entry
Usage (cron 9:30 ET, machine TZ = America/New_York):
   30 9 * * 1-5 cd ~/Dev/0dte_bot && .venv/bin/python -m scripts.jl_live_entry >> logs/jl_live_entry/cron.log 2>&1
"""
from __future__ import annotations

import asyncio
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv
load_dotenv(_REPO / ".env")   # absolute, so it works under cron regardless of CWD

from agentic_ic.services.tastytrade_client import TastyTradeClient

ET = ZoneInfo("America/New_York")
STEP = 5.0


def _mid(o):
    b, a = o.get("bid", 0) or 0, o.get("ask", 0) or 0
    if b <= 0 and a <= 0:
        return None
    if b <= 0:
        return a
    if a <= 0:
        return b
    return (b + a) / 2


def _round_strike(x, mode="near"):
    if mode == "floor":
        return math.floor(x / STEP) * STEP
    if mode == "ceil":
        return math.ceil(x / STEP) * STEP
    return round(x / STEP) * STEP


def _fmt_leg(label, by, strike, is_call):
    o = by.get((float(strike), "call" if is_call else "put"))
    if o is None:
        return f"  {label:<11} {strike:>7.0f} {'(call)' if is_call else '(put)':>6}   [NO QUOTE in enriched window]", None
    m = _mid(o)
    d = o.get("delta", 0) or 0
    return (f"  {label:<11} {strike:>7.0f} {'C' if is_call else 'P'}   "
            f"bid {o.get('bid',0):>6.2f} / ask {o.get('ask',0):>6.2f}  mid {m if m else 0:>6.2f}  Δ{d:+.2f}", m)


def _spread_credit(by, short_k, long_k, is_call):
    so = by.get((float(short_k), "call" if is_call else "put"))
    lo = by.get((float(long_k), "call" if is_call else "put"))
    if so is None or lo is None:
        return None
    sm, lm = _mid(so), _mid(lo)
    if sm is None or lm is None:
        return None
    return sm - lm


def _variant(out, title, by, sp, lp, sc, lc, call_w):
    out.append(f"\n{title}")
    for lbl, k, ic in [("SHORT PUT", sp, False), ("LONG PUT", lp, False),
                       ("SHORT CALL", sc, True), ("LONG CALL", lc, True)]:
        line, _ = _fmt_leg(lbl, by, k, ic)
        out.append(line)
    pc = _spread_credit(by, sp, lp, False)
    cc = _spread_credit(by, sc, lc, True)
    if pc is None or cc is None:
        out.append("  -> credit: UNAVAILABLE (a leg fell outside the live-enriched window)")
        return
    tot = pc + cc
    risk = "YES (no upside risk)" if tot >= call_w else f"NO (upside risk: credit {tot:.2f} < call width {call_w:.0f})"
    out.append(f"  put credit {pc:>5.2f} | call credit {cc:>5.2f} | TOTAL CREDIT {tot:>6.2f}   credit>=call_width? {risk}")
    out.append(f"  --> enter as a LIMIT CREDIT of ~{tot:.2f}")


async def main():
    now = datetime.now(ET)
    out = [f"{'='*78}",
           f"SPX 0DTE JADE LIZARD — LIVE ENTRY   {now:%Y-%m-%d %H:%M:%S} ET",
           f"{'='*78}"]

    tt = TastyTradeClient(mode="manual")
    await tt.connect()
    try:
        # spot
        spot = None
        try:
            q = await tt.get_quote("SPX")
            spot = q.get("price") or q.get("last") or (
                ((q.get("bid", 0) or 0) + (q.get("ask", 0) or 0)) / 2 or None)
        except Exception as e:
            out.append(f"(SPX index quote failed: {type(e).__name__}; will infer ATM from deltas)")

        chain = await tt.get_option_chain("SPX")
        exp = chain[0]["expiration"] if chain else "?"
        by = {(float(o["strike"]), o["option_type"]): o for o in chain}
        enr = [o for o in chain if (o.get("bid", 0) or 0) > 0 and o.get("delta") is not None]

        # ATM via delta ~0.50 (fallback: nearest strike to spot)
        calls = [o for o in enr if o["option_type"] == "call"]
        puts = [o for o in enr if o["option_type"] == "put"]
        if calls:
            atm_o = min(calls, key=lambda o: abs(abs(o.get("delta", 0)) - 0.50))
            atm = _round_strike(atm_o["strike"])
        elif spot:
            atm = _round_strike(spot)
        else:
            out.append("ERROR: no enriched calls and no spot; cannot determine ATM."); print("\n".join(out)); return
        if spot is None:
            spot = float(atm)

        # straddle EM
        c_atm = by.get((float(atm), "call")); p_atm = by.get((float(atm), "put"))
        cm, pm = (_mid(c_atm) if c_atm else None), (_mid(p_atm) if p_atm else None)
        straddle = (cm + pm) if (cm and pm) else None

        # 16-delta strikes -> TT-style +/-1 SD
        put16 = min(puts, key=lambda o: abs(abs(o.get("delta", 0)) - 0.16)) if puts else None
        call16 = min(calls, key=lambda o: abs(abs(o.get("delta", 0)) - 0.16)) if calls else None
        sd_up = (call16["strike"] - spot) if call16 else None
        sd_dn = (spot - put16["strike"]) if put16 else None
        em_1sd = None
        if sd_up and sd_dn:
            em_1sd = (sd_up + sd_dn) / 2

        out.append(f"\nexpiration: {exp}   SPX spot: {spot:.2f}   ATM strike: {atm:.0f}   "
                   f"(enriched strikes: {len(enr)})")
        out.append(f"ATM straddle (our EM)         : {straddle:.2f} pts" if straddle else "ATM straddle: UNAVAILABLE")
        if em_1sd:
            out.append(f"16-delta +/-1 SD (TT-style EM): ~{em_1sd:.1f} pts   "
                       f"[put16 {put16['strike']:.0f} (Δ{put16.get('delta',0):+.2f}), "
                       f"call16 {call16['strike']:.0f} (Δ{call16.get('delta',0):+.2f})]")
            if straddle:
                out.append(f"  -> 1SD / straddle ratio = {em_1sd/straddle:.2f}x   "
                           f"(compare 'expected move' to the TT GUI value)")
        em = straddle or (em_1sd or 0)

        # Placements use TT-style EM = 16-delta +/-1 SD (fallback: straddle). NO ATM.
        emU = em_1sd if em_1sd else (straddle or em)

        def place(title, pw, cw, mult):
            sp = _round_strike(spot - mult * emU, "floor")
            sc = _round_strike(spot + mult * emU, "ceil")
            _variant(out, title, by, sp, sp - pw, sc, sc + cw, cw)

        out.append(f"\n[EM used for placement = 16Δ ±1 SD ≈ {emU:.1f} pts]")
        place(f"VARIANT A — $30p/$10c at 0.25x EM  (offset ~{0.25*emU:.0f}pt) [slide says this; sweep: weak]:", 30, 10, 0.25)
        place(f"VARIANT B — $20p/$5c  at 0.25x EM  (offset ~{0.25*emU:.0f}pt) [the slide's $20/$5 variant]:", 20, 5, 0.25)
        place(f"VARIANT C — $30p/$10c at 1.0x EM   (offset ~{emU:.0f}pt) [our swept optimum, hold-to-expiry]:", 30, 10, 1.0)

        # ATM chain snapshot
        out.append(f"\nATM CHAIN SNAPSHOT (compare bid/ask/delta to the TT GUI):")
        out.append(f"  {'strike':>7} | {'put bid/ask':>15} {'Δ':>6} | {'call bid/ask':>15} {'Δ':>6}")
        lo, hi = atm - 40, atm + 40
        strikes = sorted({float(o['strike']) for o in chain if lo <= o['strike'] <= hi})
        for k in strikes:
            po = by.get((k, "put")); co = by.get((k, "call"))
            ps = f"{po.get('bid',0):.2f}/{po.get('ask',0):.2f}" if po else "-"
            cs = f"{co.get('bid',0):.2f}/{co.get('ask',0):.2f}" if co else "-"
            pd = f"{po.get('delta',0):+.2f}" if po and po.get('delta') is not None else "-"
            cd = f"{co.get('delta',0):+.2f}" if co and co.get('delta') is not None else "-"
            mark = " <-ATM" if k == atm else ""
            out.append(f"  {k:>7.0f} | {ps:>15} {pd:>6} | {cs:>15} {cd:>6}{mark}")
    finally:
        try:
            await tt.disconnect()
        except Exception:
            pass

    text = "\n".join(out)
    print(text)
    logdir = _REPO / "logs" / "jl_live_entry"; logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"{now:%Y-%m-%d}.txt").write_text(text + "\n")
    print(f"\n[saved to {logdir / f'{now:%Y-%m-%d}.txt'}]")


if __name__ == "__main__":
    asyncio.run(main())
