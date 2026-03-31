#!/usr/bin/env python3
"""
Multi-symbol 0DTE option data collector — streams quotes + greeks via DXLink.

Collects normalized option prices across non-correlated underlyings for
training a universal 0DTE pricing ANN.

All prices are normalized to base=100 so different symbols are comparable:
  norm_strike = strike * 100 / spot
  norm_mid    = mid    * 100 / spot

Symbols chosen for low cross-correlation:
  SPY  — S&P 500 (large-cap US equities)
  QQQ  — Nasdaq 100 (tech-heavy)
  IWM  — Russell 2000 (small-cap, higher vol)
  GLD  — Gold ETF (commodity, uncorrelated to equities)
  TLT  — 20+ Year Treasury (bonds, often inverse to equities)
  XLE  — Energy Select (oil/gas sector)

Output: data/iv_collection/YYYY-MM-DD.jsonl

Usage:
    python scripts/collect_0dte_live.py
    python scripts/collect_0dte_live.py --interval 120
    python scripts/collect_0dte_live.py --symbols SPY,QQQ,IWM
    python scripts/collect_0dte_live.py --atm-width 10
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

# Default symbols: non-correlated asset classes
DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]


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


async def collect_symbol_snapshot(
    symbol: str, tt_client, session, ann_table, tte_all, atm_width: int,
):
    """Collect one snapshot for a single symbol."""
    now_et = datetime.now(ET)

    # Get underlying price
    try:
        q = await tt_client.get_quote(symbol)
        spot = q.get("price", 0)
    except Exception as exc:
        print(f"  [{symbol}] Quote failed: {exc}")
        return []

    if spot <= 0:
        print(f"  [{symbol}] No price")
        return []

    # Compute TTE
    close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    tte_hours = max(0.01, (close_dt - now_et).total_seconds() / 3600)
    tte_years = tte_hours / (252 * 6.5)

    # ATM width scales with price: ±atm_width in normalized (base=100) terms
    # e.g. atm_width=5 means ±5% of spot → for SPY@670: ±33.5
    raw_width = atm_width * spot / 100

    # Get option chain structure (REST)
    try:
        chain = await tt_client.get_option_chain(symbol)
    except Exception as exc:
        print(f"  [{symbol}] Chain failed: {exc}")
        return []

    if not chain:
        print(f"  [{symbol}] Empty chain")
        return []

    # Filter to near-ATM
    near_atm = [o for o in chain if abs(o["strike"] - spot) <= raw_width]
    if not near_atm:
        print(f"  [{symbol}] No near-ATM options (spot={spot:.2f}, width={raw_width:.1f})")
        return []

    # Enrich via DXLink
    near_atm = await _enrich_near_atm(session, near_atm)
    enriched = sum(1 for o in near_atm if o.get("bid", 0) > 0)
    print(f"  [{symbol}] spot={spot:.2f}, {enriched}/{len(near_atm)} enriched")

    records = []
    for opt in near_atm:
        strike = opt.get("strike", 0)
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
        moneyness = spot / strike

        is_call = 1 if opt_type == "call" else 0
        ann_iv = ann_lookup(ann_table, tte_all, moneyness, tte_years, is_call)

        records.append({
            "timestamp": now_et.isoformat(),
            "symbol": symbol,
            "strike": strike,
            "option_type": opt_type,
            "spot": round(spot, 2),
            "moneyness": round(moneyness, 4),
            "tte_years": round(tte_years, 6),
            "tte_hours": round(tte_hours, 2),
            # Raw market data
            "bid": bid,
            "ask": ask,
            "mid": round(mid, 4),
            # Normalized to base=100
            "norm_spot": 100.0,
            "norm_strike": round(strike * 100 / spot, 4),
            "norm_mid": round(mid * 100 / spot, 4),
            # Greeks
            "delta": round(delta, 4) if delta else None,
            "gamma": round(gamma, 6) if gamma else None,
            "theta": round(theta, 4) if theta else None,
            "vega": round(vega, 4) if vega else None,
            # IV
            "market_iv": round(market_iv, 5) if market_iv else None,
            "ann_iv": round(ann_iv, 5) if ann_iv else None,
            "ann_iv_err": round(ann_iv - market_iv, 5) if (ann_iv and market_iv) else None,
        })

    return records


async def main_async():
    parser = argparse.ArgumentParser(
        description="Multi-symbol 0DTE option data collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interval", type=int, default=300,
                        help="Poll interval in seconds (default 5 min)")
    parser.add_argument("--atm-width", type=int, default=5,
                        help="ATM width in normalized points (base=100, default ±5)")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help=f"Comma-separated symbols (default: {','.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--ann-table", default="models/iv_ann_table.json")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    from src.services.tastytrade_client import TastyTradeClient

    print("Authenticating to TastyTrade...")
    tt = TastyTradeClient(mode="manual")
    await tt.connect()
    session = tt._market_sdk_session or tt._sdk_session
    print("Authenticated OK")

    ann_table, tte_all = load_ann_table(args.ann_table)

    out_dir = Path("data/iv_collection")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today().isoformat()}.jsonl"
    print(f"Output: {out_path}")
    print(f"Symbols: {symbols}")
    print(f"Polling every {args.interval}s, ATM width ±{args.atm_width} (normalized)")

    # Quick check: which symbols have 0DTE chains today
    print("\nChecking 0DTE availability...")
    available = []
    for sym in symbols:
        try:
            chain = await tt.get_option_chain(sym)
            if chain:
                available.append(sym)
                print(f"  {sym}: {len(chain)} options")
            else:
                print(f"  {sym}: no 0DTE chain today")
        except Exception as exc:
            print(f"  {sym}: error — {exc}")

    if not available:
        print("No symbols have 0DTE options today — exiting")
        await tt.disconnect()
        return

    symbols = available
    print(f"\nCollecting: {symbols}")

    poll_count = 0
    reauth_interval = 600  # re-authenticate every 10 min to avoid token expiry
    last_auth = asyncio.get_event_loop().time()

    while True:
        try:
            now_et = datetime.now(ET)
            if now_et.time() < time(9, 30):
                wait = (datetime.combine(now_et.date(), time(9, 30), tzinfo=ET) - now_et).seconds
                print(f"Pre-market — waiting {wait}s until 09:30 ET")
                await asyncio.sleep(min(wait, 60))
                continue
            if now_et.time() > time(16, 15):
                print("Market closed — stopping")
                break

            # Re-authenticate before token expires (15 min lifetime)
            if asyncio.get_event_loop().time() - last_auth > reauth_interval:
                print("  Re-authenticating...")
                try:
                    await tt.disconnect()
                except Exception:
                    pass
                tt = TastyTradeClient(mode="manual")
                await tt.connect()
                session = tt._market_sdk_session or tt._sdk_session
                last_auth = asyncio.get_event_loop().time()
                print("  Re-authenticated OK")

            print(f"\n[{now_et.strftime('%H:%M:%S')}] Collecting 0DTE snapshots...")
            all_records = []

            for sym in symbols:
                try:
                    records = await collect_symbol_snapshot(
                        sym, tt, session, ann_table, tte_all, args.atm_width,
                    )
                    all_records.extend(records)
                except Exception as exc:
                    print(f"  [{sym}] Error: {exc}")

            if all_records:
                with open(out_path, "a") as f:
                    for r in all_records:
                        f.write(json.dumps(r) + "\n")

                poll_count += 1
                by_sym = {}
                for r in all_records:
                    s = r.get("symbol", "?")
                    by_sym[s] = by_sym.get(s, 0) + 1
                summary = ", ".join(f"{s}={n}" for s, n in sorted(by_sym.items()))
                print(f"  Poll #{poll_count}: {len(all_records)} total — {summary}")
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
