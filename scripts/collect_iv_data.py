#!/usr/bin/env python3
"""
REST-only IV data collector — runs without DXLink.

Polls TastyTrade REST API every 60s and records:
  - ES ATM IV from market-metrics (per-expiration)
  - 0DTE option chain structure (strikes, symbols)
  - ANN IV predictions for each strike
  - BS theoretical prices from ANN IV

Output: data/iv_comparison/YYYY-MM-DD_rest.jsonl

Usage:
    python scripts/collect_iv_data.py
    python scripts/collect_iv_data.py --interval 30   # poll every 30s
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time as time_mod
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ET = ZoneInfo("America/New_York")


def load_ann_table(path: str = "models/iv_ann_table.json"):
    """Load precomputed ANN IV lookup table."""
    import numpy as np

    if not Path(path).exists():
        print(f"ANN table not found at {path}")
        return None, []
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--ann-table", default="models/iv_ann_table.json")
    args = parser.parse_args()

    from tastytrade import Session
    from tastytrade.instruments import NestedFutureOptionChain

    secret = os.environ.get("TASTYTRADE_CLIENT_SECRET", "")
    token = os.environ.get("TASTYTRADE_REFRESH_TOKEN", "")
    if not secret or not token:
        print("Set TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN")
        sys.exit(1)

    print("Authenticating...")
    session = Session(secret, token)
    print("Authenticated OK")

    ann_table, tte_all = load_ann_table(args.ann_table)

    out_dir = Path("data/iv_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today().isoformat()}_rest.jsonl"
    print(f"Output: {out_path}")

    today_str = date.today().isoformat()

    # Get option chain structure once (REST — doesn't need DXLink)
    print("Fetching futures option chain structure...")
    chain = NestedFutureOptionChain.get(session, "ES")
    today_options = []
    for sc in chain.option_chains:
        for exp in sc.expirations:
            if str(exp.expiration_date) == today_str:
                for strike in exp.strikes:
                    sp = float(strike.strike_price)
                    for side in ["call", "put"]:
                        sym = getattr(strike, side, None)
                        ssym = getattr(strike, f"{side}_streamer_symbol", "")
                        if sym:
                            today_options.append({
                                "symbol": str(sym),
                                "streamer_symbol": str(ssym),
                                "strike": sp,
                                "option_type": side,
                            })

    print(f"Found {len(today_options)} 0DTE options for {today_str}")
    if not today_options:
        print("No 0DTE options found — exiting")
        sys.exit(0)

    poll_count = 0
    while True:
        try:
            now_et = datetime.now(ET)
            # Stop after 4:15 PM ET
            if now_et.time() > time(16, 15):
                print("Market closed — stopping")
                break

            # Get ATM IV from market-metrics (REST)
            resp = session.sync_client.get("/market-metrics", params={"symbols": "/ES"})
            if resp.status_code != 200:
                print(f"market-metrics failed: {resp.status_code}")
                time_mod.sleep(args.interval)
                continue

            metrics = resp.json()["data"]["items"][0]
            iv_index = float(metrics.get("implied-volatility-index", 0))

            # Find today's expiry IV
            atm_iv_0dte = None
            for exp_iv in metrics.get("option-expiration-implied-volatilities", []):
                if exp_iv["expiration-date"] == today_str:
                    atm_iv_0dte = float(exp_iv["implied-volatility"])
                    break

            # Compute TTE
            close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            tte_hours = max(0.01, (close_dt - now_et).total_seconds() / 3600)
            tte_years = tte_hours / (252 * 6.5)

            # Estimate underlying price from market-metrics
            # (We don't have a direct REST quote, use the liquidity value as proxy
            # or we can infer from ATM strike)
            # For now, find ATM from chain structure using the IV index
            # Actually, market-metrics doesn't give us the price directly.
            # Let's try to get it from a different REST endpoint.
            price = None

            # Try accounts endpoint for buying power (as indirect price source)
            # Or just use the strikes to bracket ATM
            if not price:
                # Use last known price or estimate from chain
                # The ATM strike should be roughly the current price
                strikes = sorted(set(o["strike"] for o in today_options))
                mid_idx = len(strikes) // 2
                price = strikes[mid_idx] if strikes else 5700.0

            records = []
            # Record ATM IV snapshot
            snapshot = {
                "timestamp": now_et.isoformat(),
                "type": "atm_snapshot",
                "underlying_est": price,
                "tte_hours": round(tte_hours, 2),
                "tte_years": round(tte_years, 6),
                "iv_index": round(iv_index, 6),
                "atm_iv_0dte": round(atm_iv_0dte, 6) if atm_iv_0dte else None,
            }
            records.append(snapshot)

            # Record ANN predictions for near-ATM strikes
            for opt in today_options:
                strike = opt["strike"]
                moneyness = price / strike if strike > 0 else 1.0
                if moneyness < 0.97 or moneyness > 1.03:
                    continue  # only near-ATM

                is_call = 1 if opt["option_type"] == "call" else 0
                ann_iv = ann_lookup(ann_table, tte_all, moneyness, tte_years, is_call)

                # BS price from ANN IV
                p_ann = bs_price(price, strike, tte_years, ann_iv, opt["option_type"]) if ann_iv else None
                # BS price from TT ATM IV (flat)
                p_atm = bs_price(price, strike, tte_years, atm_iv_0dte, opt["option_type"]) if atm_iv_0dte else None

                records.append({
                    "timestamp": now_et.isoformat(),
                    "type": "strike_prediction",
                    "underlying_est": price,
                    "strike": strike,
                    "option_type": opt["option_type"],
                    "symbol": opt["symbol"],
                    "moneyness": round(moneyness, 4),
                    "tte_years": round(tte_years, 6),
                    "tte_hours": round(tte_hours, 2),
                    "ann_iv": round(ann_iv, 5) if ann_iv else None,
                    "tt_atm_iv": round(atm_iv_0dte, 5) if atm_iv_0dte else None,
                    "price_ann": round(p_ann, 4) if p_ann else None,
                    "price_atm_flat": round(p_atm, 4) if p_atm else None,
                })

            # Write records
            with open(out_path, "a") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            poll_count += 1
            n_strikes = sum(1 for r in records if r["type"] == "strike_prediction")
            print(
                f"[{now_et.strftime('%H:%M:%S')}] Poll #{poll_count}: "
                f"ATM IV={atm_iv_0dte:.4f}, TTE={tte_hours:.1f}h, "
                f"{n_strikes} strikes logged"
            )

        except KeyboardInterrupt:
            print("\nStopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")

        time_mod.sleep(args.interval)

    print(f"\nDone. {poll_count} polls saved to {out_path}")


if __name__ == "__main__":
    main()
