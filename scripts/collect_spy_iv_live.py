#!/usr/bin/env python3
"""
Live SPY 0DTE IV collector — streams real quotes + greeks via DXLink.

Uses SPY equity options (which work via DXLink, unlike ES futures options)
to collect live market data and compare against ANN IV predictions.

Records per strike: bid, ask, mid, delta, gamma, theta, vega, market_iv,
ann_iv, flat_iv, BS theoretical prices from each IV source, and errors.

Output: data/iv_comparison/YYYY-MM-DD_spy_live.jsonl

Usage:
    python scripts/collect_spy_iv_live.py
    python scripts/collect_spy_iv_live.py --interval 120   # poll every 2 min
    python scripts/collect_spy_iv_live.py --atm-width 50   # ±50 pts from ATM
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ET = ZoneInfo("America/New_York")


def load_ann_table(path: str = "models/iv_ann_table.json"):
    """Load precomputed ANN IV lookup table."""
    if not Path(path).exists():
        print(f"[WARN] ANN table not found at {path} — will collect market data only")
        return None, []
    import numpy as np

    with open(path) as f:
        data = json.load(f)
    table = data.get("compact", {})
    fine = data.get("tte_fine_range", [0.0001, 0.05])
    fine_step = data.get("tte_fine_step", 0.001)
    coarse = data.get("tte_coarse_range", [0.05, 4.0])
    coarse_step = data.get("tte_coarse_step", 0.05)
    tte_all = (
        list(np.arange(fine[0], fine[1], fine_step))
        + list(np.arange(coarse[0], coarse[1] + coarse_step, coarse_step))
    )
    print(f"Loaded ANN IV table: {sum(len(v) for v in table.values())} entries")
    return table, tte_all


def ann_lookup(table, tte_all, moneyness: float, tte: float, is_call: int):
    """Look up IV from ANN table."""
    if not table:
        return None
    opt_key = "call" if is_call else "put"
    sub = table.get(opt_key, {})
    if not sub:
        return None
    m_snap = round(max(0.900, min(1.100, moneyness)), 3)
    best_tte = tte_all[0] if tte_all else 0.001
    best_dist = abs(best_tte - tte)
    for tv in tte_all:
        d = abs(tv - tte)
        if d < best_dist:
            best_dist = d
            best_tte = tv
    key = f"{m_snap:.3f}_{best_tte:.4f}"
    iv = sub.get(key)
    if iv is not None:
        return iv
    for offset in [0.001, -0.001]:
        alt_key = f"{m_snap + offset:.3f}_{best_tte:.4f}"
        iv = sub.get(alt_key)
        if iv is not None:
            return iv
    return None


def bs_price(s, k, t, iv, opt_type):
    """BS theoretical price."""
    if t <= 0 or iv is None or iv <= 0:
        return None

    def _ncdf(x):
        a = abs(x)
        t2 = 1.0 / (1.0 + 0.2316419 * a)
        poly = t2 * (0.319381530 + t2 * (-0.356563782 + t2 * (
            1.781477937 + t2 * (-1.821255978 + t2 * 1.330274429))))
        cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a) * poly
        return cdf if x >= 0 else 1.0 - cdf

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    if opt_type == "call":
        return s * _ncdf(d1) - k * _ncdf(d2)
    else:
        return k * _ncdf(-d2) - s * _ncdf(-d1)


async def _enrich_near_atm(session, options, timeout_s=10):
    """Stream quotes + greeks for a small set of options via DXLink."""
    import asyncio
    from tastytrade import DXLinkStreamer
    from tastytrade.dxfeed import Quote, Greeks

    syms = [o["streamer_symbol"] for o in options if o.get("streamer_symbol")]
    if not syms:
        return options

    quotes_map = {}
    greeks_map = {}

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, syms)
        await streamer.subscribe(Greeks, syms)

        needed_q = set(syms)
        needed_g = set(syms)
        deadline = asyncio.get_event_loop().time() + timeout_s

        while (needed_q or needed_g) and asyncio.get_event_loop().time() < deadline:
            left = deadline - asyncio.get_event_loop().time()
            if left <= 0:
                break
            try:
                q = await asyncio.wait_for(streamer.get_event(Quote), timeout=min(1.0, left))
                quotes_map[q.event_symbol] = q
                needed_q.discard(q.event_symbol)
            except asyncio.TimeoutError:
                pass
            try:
                g = await asyncio.wait_for(streamer.get_event(Greeks), timeout=min(1.0, left))
                greeks_map[g.event_symbol] = g
                needed_g.discard(g.event_symbol)
            except asyncio.TimeoutError:
                pass

    for opt in options:
        ssym = opt.get("streamer_symbol", "")
        q = quotes_map.get(ssym)
        g = greeks_map.get(ssym)
        if q:
            opt["bid"] = float(q.bid_price or 0)
            opt["ask"] = float(q.ask_price or 0)
        if g:
            opt["delta"] = float(g.delta or 0)
            opt["gamma"] = float(g.gamma or 0)
            opt["theta"] = float(g.theta or 0)
            opt["vega"] = float(g.vega or 0)
            opt["iv"] = float(g.volatility or 0)

    return options


async def collect_one_snapshot(tt_client, ann_table, tte_all, atm_width: int):
    """Fetch SPY chain via REST, enrich near-ATM via DXLink, compare to ANN."""
    now_et = datetime.now(ET)

    # Get SPY price
    try:
        q = await tt_client.get_quote("SPY")
        spy_price = q.get("price", 0)
    except Exception as exc:
        print(f"  Failed to get SPY quote: {exc}")
        return []

    if spy_price <= 0:
        print("  SPY price not available")
        return []

    # Compute TTE
    close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    tte_hours = max(0.01, (close_dt - now_et).total_seconds() / 3600)
    tte_years = tte_hours / (252 * 6.5)

    # Get option chain structure (REST only — no DXLink enrichment)
    try:
        chain = await tt_client.get_option_chain("SPY")
    except Exception as exc:
        print(f"  Failed to get SPY option chain: {exc}")
        return []

    if not chain:
        print("  Empty SPY option chain")
        return []

    # Filter to near-ATM before DXLink enrichment (key fix: small batch)
    near_atm = [o for o in chain if abs(o["strike"] - spy_price) <= atm_width]
    print(f"  {len(near_atm)} near-ATM options (±{atm_width} from {spy_price:.2f})")

    # Enrich only near-ATM strikes via DXLink
    session = tt_client._market_sdk_session or tt_client._sdk_session
    near_atm = await _enrich_near_atm(session, near_atm)
    enriched = sum(1 for o in near_atm if o.get("bid", 0) > 0)
    print(f"  {enriched}/{len(near_atm)} enriched with live quotes")

    chain = near_atm

    records = []

    # ATM snapshot
    records.append({
        "timestamp": now_et.isoformat(),
        "type": "atm_snapshot",
        "underlying": round(spy_price, 2),
        "tte_hours": round(tte_hours, 2),
        "tte_years": round(tte_years, 6),
    })

    # Per-strike comparison
    for opt in chain:
        strike = opt.get("strike", 0)
        if strike <= 0:
            continue
        if abs(strike - spy_price) > atm_width:
            continue

        bid = opt.get("bid", 0)
        ask = opt.get("ask", 0)
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2
        if mid < 0.01:
            continue

        opt_type = opt.get("option_type", "")
        delta = opt.get("delta", 0)
        gamma = opt.get("gamma", 0)
        theta = opt.get("theta", 0)
        vega = opt.get("vega", 0)
        market_iv = opt.get("iv", 0)

        moneyness = spy_price / strike

        # ANN prediction
        is_call = 1 if opt_type == "call" else 0
        ann_iv = ann_lookup(ann_table, tte_all, moneyness, tte_years, is_call)

        # Flat IV baseline
        flat_iv = 0.15

        # Model-predicted prices
        price_flat = bs_price(spy_price, strike, tte_years, flat_iv, opt_type)
        price_ann = bs_price(spy_price, strike, tte_years, ann_iv, opt_type) if ann_iv else None
        price_market_iv = bs_price(spy_price, strike, tte_years, market_iv, opt_type) if market_iv else None

        records.append({
            "timestamp": now_et.isoformat(),
            "type": "strike",
            "underlying": round(spy_price, 2),
            "strike": strike,
            "option_type": opt_type,
            "moneyness": round(moneyness, 4),
            "tte_years": round(tte_years, 6),
            "tte_hours": round(tte_hours, 2),
            # Market data (live from DXLink)
            "bid": bid,
            "ask": ask,
            "mid": round(mid, 4),
            # Normalized to base=100: spot_norm = 100, mid_norm = mid * (100/spot)
            "norm_spot": 100.0,
            "norm_mid": round(mid * 100 / spy_price, 4),
            "norm_strike": round(strike * 100 / spy_price, 4),
            # Greeks (live from DXLink)
            "delta": round(delta, 4) if delta else None,
            "gamma": round(gamma, 6) if gamma else None,
            "theta": round(theta, 4) if theta else None,
            "vega": round(vega, 4) if vega else None,
            # IV: market vs models
            "market_iv": round(market_iv, 5) if market_iv else None,
            "ann_iv": round(ann_iv, 5) if ann_iv else None,
            "flat_iv": round(flat_iv, 5),
            # Predicted prices
            "price_market_iv_bs": round(price_market_iv, 4) if price_market_iv is not None else None,
            "price_ann_bs": round(price_ann, 4) if price_ann is not None else None,
            "price_flat_bs": round(price_flat, 4) if price_flat is not None else None,
            # Errors (ANN vs market)
            "ann_iv_err": round(ann_iv - market_iv, 5) if (ann_iv and market_iv) else None,
            "ann_price_err": round(price_ann - mid, 4) if price_ann is not None else None,
            "market_iv_bs_err": round(price_market_iv - mid, 4) if price_market_iv is not None else None,
        })

    return records


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds (default 5 min)")
    parser.add_argument("--atm-width", type=int, default=30, help="Strike distance from ATM to include (default ±30)")
    parser.add_argument("--ann-table", default="models/iv_ann_table.json")
    args = parser.parse_args()

    # Authenticate
    from src.services.tastytrade_client import TastyTradeClient

    print("Authenticating to TastyTrade (production for live market data)...")
    tt = TastyTradeClient(mode="manual")  # live data, no orders
    await tt.connect()
    print("Authenticated OK")

    ann_table, tte_all = load_ann_table(args.ann_table)

    out_dir = Path("data/iv_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today().isoformat()}_spy_live.jsonl"
    print(f"Output: {out_path}")
    print(f"Polling every {args.interval}s, ATM width ±{args.atm_width}")

    poll_count = 0
    while True:
        try:
            now_et = datetime.now(ET)
            # Only run during market hours (9:30-16:15 ET)
            if now_et.time() < time(9, 30):
                wait = (datetime.combine(now_et.date(), time(9, 30), tzinfo=ET) - now_et).seconds
                print(f"Pre-market — waiting {wait}s until 09:30 ET")
                await asyncio.sleep(min(wait, 60))
                continue
            if now_et.time() > time(16, 15):
                print("Market closed — stopping")
                break

            print(f"\n[{now_et.strftime('%H:%M:%S')}] Collecting SPY 0DTE snapshot...")
            records = await collect_one_snapshot(tt, ann_table, tte_all, args.atm_width)

            if records:
                with open(out_path, "a") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")

                n_strikes = sum(1 for r in records if r["type"] == "strike")
                n_with_ann = sum(1 for r in records if r.get("ann_iv") is not None)
                n_with_mkt_iv = sum(1 for r in records if r.get("market_iv") is not None)

                # Compute average errors for strikes with both ANN and market IV
                ann_errs = [r["ann_iv_err"] for r in records if r.get("ann_iv_err") is not None]
                avg_ann_err = sum(ann_errs) / len(ann_errs) if ann_errs else 0

                poll_count += 1
                print(
                    f"  Poll #{poll_count}: {n_strikes} strikes, "
                    f"{n_with_mkt_iv} with market IV, {n_with_ann} with ANN IV, "
                    f"avg ANN IV err={avg_ann_err:+.4f}"
                )
            else:
                print("  No data collected")

        except KeyboardInterrupt:
            print("\nStopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(args.interval)

    await tt.disconnect()
    print(f"\nDone. {poll_count} polls saved to {out_path}")


def main():
    loop = asyncio.new_event_loop()

    def _suppress_ssl(loop, context):
        msg = context.get("message", "")
        if "SSL" in msg or "Bad file descriptor" in msg:
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_suppress_ssl)
    try:
        loop.run_until_complete(main_async())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
