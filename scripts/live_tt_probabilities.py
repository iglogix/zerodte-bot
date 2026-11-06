#!/usr/bin/env python3
"""
Fetch live ES 0DTE option chain from TastyTrade, compute %ITM and %Touch,
then take a screenshot for visual comparison.
"""
from dotenv import load_dotenv; load_dotenv()

import asyncio
import json
import os
import sys
from datetime import datetime, date, time
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_ic.services.tastytrade_client import TastyTradeClient
from es_8am.utils.contract_utils import get_dxlink_symbol

ET = ZoneInfo("America/New_York")


# ── Black-Scholes helpers ──────────────────────────────────────────────

def bs_d(S, K, T, sigma, r=0.0):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2

def bs_call(S, K, T, sigma, r=0.0):
    d1, d2 = bs_d(S, K, T, sigma, r)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, sigma, r=0.0):
    d1, d2 = bs_d(S, K, T, sigma, r)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(S, K, T, price, is_call=True, r=0.0):
    func = bs_call if is_call else bs_put
    try:
        return brentq(lambda s: func(S, K, T, s, r) - price, 0.01, 10.0, xtol=1e-8)
    except (ValueError, RuntimeError):
        return None

def pct_itm_call(S, K, T, sigma, r=0.0):
    _, d2 = bs_d(S, K, T, sigma, r)
    return norm.cdf(d2) * 100

def pct_itm_put(S, K, T, sigma, r=0.0):
    _, d2 = bs_d(S, K, T, sigma, r)
    return norm.cdf(-d2) * 100

def pct_touch(S, K, T, sigma, r=0.0):
    mu = r - 0.5 * sigma**2
    sqrtT = np.sqrt(T)
    b = np.log(K / S)
    if b > 0:
        t1 = norm.cdf((-b + mu * T) / (sigma * sqrtT))
        t2 = np.exp(2 * mu * b / sigma**2) * norm.cdf((-b - mu * T) / (sigma * sqrtT))
    else:
        b_abs = -b
        t1 = norm.cdf((-b_abs - mu * T) / (sigma * sqrtT))
        t2 = np.exp(-2 * mu * b_abs / sigma**2) * norm.cdf((-b_abs + mu * T) / (sigma * sqrtT))
    return min((t1 + t2) * 100, 100.0)


# ── Fetch live data ────────────────────────────────────────────────────

async def fetch_chain():
    """Fetch ES 0DTE option chain with quotes and greeks."""
    from tastytrade.instruments import NestedFutureOptionChain
    from tastytrade import DXLinkStreamer
    from tastytrade.dxfeed import Quote, Greeks

    tt = TastyTradeClient(mode="manual")
    await tt.connect()

    # Get ES price
    es_sym = get_dxlink_symbol()
    q = await tt.get_quote(es_sym)
    spot = q["price"]
    print(f"ES symbol: {es_sym}  |  Spot: {spot:.2f}")

    # Get chain structure
    session = tt._market_sdk_session or tt._sdk_session
    chain = await NestedFutureOptionChain.get(session, "ES")

    today_str = date.today().strftime("%Y-%m-%d")
    options = []
    for subchain in chain.option_chains:
        for exp in subchain.expirations:
            if str(exp.expiration_date) != today_str:
                continue
            for strike in exp.strikes:
                k = float(strike.strike_price)
                if abs(k - spot) > 120:
                    continue
                for side in ["call", "put"]:
                    ssym = getattr(strike, f"{side}_streamer_symbol", "")
                    if ssym:
                        options.append({
                            "streamer_symbol": ssym,
                            "strike": k,
                            "option_type": side,
                        })

    print(f"Chain: {len(options)} options (±120 pts of {spot:.2f})")

    # Stream quotes + greeks
    sym_map = {o["streamer_symbol"]: o for o in options}
    syms = list(sym_map.keys())

    quotes = {}
    greeks_data = {}

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, syms)
        await streamer.subscribe(Greeks, syms)
        print("Streaming quotes + greeks...")

        deadline = asyncio.get_event_loop().time() + 12  # 12 sec
        while asyncio.get_event_loop().time() < deadline:
            try:
                ev = await asyncio.wait_for(streamer.get_event(Quote), timeout=1)
                quotes[ev.event_symbol] = {
                    "bid": float(ev.bid_price or 0),
                    "ask": float(ev.ask_price or 0),
                }
            except asyncio.TimeoutError:
                pass
            try:
                ev = await asyncio.wait_for(streamer.get_event(Greeks), timeout=0.5)
                greeks_data[ev.event_symbol] = {
                    "iv": float(ev.volatility or 0),
                    "delta": float(ev.delta or 0),
                }
            except asyncio.TimeoutError:
                pass

    print(f"Got quotes for {len(quotes)} syms, greeks for {len(greeks_data)} syms")

    # Merge into rows per strike
    strikes = {}
    for sym, opt in sym_map.items():
        k = opt["strike"]
        side = opt["option_type"]
        if k not in strikes:
            strikes[k] = {"strike": k}
        q = quotes.get(sym, {})
        g = greeks_data.get(sym, {})
        prefix = "c" if side == "call" else "p"
        strikes[k][f"{prefix}_bid"] = q.get("bid", 0)
        strikes[k][f"{prefix}_ask"] = q.get("ask", 0)
        strikes[k][f"{prefix}_iv"] = g.get("iv", 0)
        strikes[k][f"{prefix}_delta"] = g.get("delta", 0)

    await tt.disconnect()
    return spot, sorted(strikes.values(), key=lambda x: x["strike"])


# ── Compute and display ───────────────────────────────────────────────

def compute_and_print(spot, rows):
    # Estimate T from market hours remaining
    now_et = datetime.now(ET)
    # ES options expire at 16:00 ET
    close = datetime.combine(now_et.date(), time(16, 0), tzinfo=ET)
    hours_left = max((close - now_et).total_seconds() / 3600, 0.1)
    T = hours_left / (252 * 6.5)  # fraction of year (trading hours basis)
    print(f"\nTime: {now_et.strftime('%H:%M:%S ET')}  |  Hours to expiry: {hours_left:.2f}  |  T = {T:.6f} years")
    print(f"Spot: {spot:.2f}\n")

    header = (
        f"{'Strike':>6}  {'IV':>6}  │ "
        f"{'Delta':>6}  │ "
        f"{'%ITM':>5} {'calc':>5} {'Δ':>5}  │ "
        f"{'%Touch':>6} {'calc':>5} {'Δ':>5}  │ Side"
    )
    print(header)
    print("─" * 80)

    results = []
    for row in rows:
        K = row["strike"]
        c_bid = row.get("c_bid", 0)
        c_ask = row.get("c_ask", 0)
        p_bid = row.get("p_bid", 0)
        p_ask = row.get("p_ask", 0)
        c_delta = row.get("c_delta", 0)
        p_delta = row.get("p_delta", 0)

        # Use OTM side
        if K >= spot:
            mid = (c_bid + c_ask) / 2
            if mid <= 0:
                continue
            iv = implied_vol(spot, K, T, mid, is_call=True)
            if iv is None:
                continue
            itm_calc = pct_itm_call(spot, K, T, iv)
            touch_calc = pct_touch(spot, K, T, iv)
            # TT's %ITM ≈ |delta| * 100 (approx, since they use N(d2) not N(d1))
            # We'll compare our N(d2) against TT's displayed value
            # For now, show delta from DXLink as reference
            delta_show = c_delta
            side = "OTM C"
        else:
            mid = (p_bid + p_ask) / 2
            if mid <= 0:
                continue
            iv = implied_vol(spot, K, T, mid, is_call=False)
            if iv is None:
                continue
            itm_calc = pct_itm_put(spot, K, T, iv)
            touch_calc = pct_touch(spot, K, T, iv)
            delta_show = p_delta
            side = "OTM P"

        # TT %ITM is N(d2), which we compute. We don't have TT's displayed %ITM
        # from API, so we just show our computed values for screenshot comparison.
        results.append({
            "strike": K,
            "iv": iv,
            "delta": delta_show,
            "itm": itm_calc,
            "touch": touch_calc,
            "side": side,
            "mid": mid,
        })

        print(
            f"{K:>6.0f}  {iv*100:>5.1f}%  │ "
            f"{delta_show:>+6.2f}  │ "
            f"{'':>5} {itm_calc:>5.1f}  {'':>5}  │ "
            f"{'':>6} {touch_calc:>5.1f}  {'':>5}  │ {side}"
        )

    print("─" * 80)
    print("\nCompare these values with the TT screenshot to verify.")
    print("ITM% should match TT's 'ITM %' column; Touch should match 'Touch %'.\n")

    # Save for easy comparison
    out = {
        "timestamp": datetime.now(ET).isoformat(),
        "spot": spot,
        "T_years": T,
        "hours_left": hours_left,
        "results": results,
    }
    with open("/tmp/tt_prob_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("Results saved to /tmp/tt_prob_results.json")


async def main():
    spot, rows = await fetch_chain()
    compute_and_print(spot, rows)


if __name__ == "__main__":
    asyncio.run(main())
