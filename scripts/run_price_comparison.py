"""One-shot price comparison: Market vs BS vs ANN for current ES 0DTE options."""
import asyncio
import json
import math
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv; load_dotenv()

from agentic_ic.services.tastytrade_client import TastyTradeClient
from scripts.train_price_ann import PriceANN

ET = ZoneInfo("America/New_York")


def bs_price(s, k, t, iv, opt_type):
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
    return k * _ncdf(-d2) - s * _ncdf(-d1)


def bs_iv(market_price, s, k, t, opt_type, tol=1e-6, max_iter=50):
    if t <= 0 or market_price <= 0:
        return None
    intrinsic = max(s - k, 0.0) if opt_type == "call" else max(k - s, 0.0)
    if market_price < intrinsic - 0.01:
        return None
    def _ncdf(x):
        a = abs(x)
        t2 = 1.0 / (1.0 + 0.2316419 * a)
        poly = t2 * (0.319381530 + t2 * (-0.356563782 + t2 * (
            1.781477937 + t2 * (-1.821255978 + t2 * 1.330274429))))
        cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * a * a) * poly
        return cdf if x >= 0 else 1.0 - cdf
    iv = 0.20
    diff = 0.0
    for _ in range(max_iter):
        sqrt_t = math.sqrt(t)
        d1 = (math.log(s / k) + 0.5 * iv * iv * t) / (iv * sqrt_t)
        d2 = d1 - iv * sqrt_t
        if opt_type == "call":
            price = s * _ncdf(d1) - k * _ncdf(d2)
        else:
            price = k * _ncdf(-d2) - s * _ncdf(-d1)
        vega = s * sqrt_t * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        if vega < 1e-12:
            return None
        diff = price - market_price
        iv -= diff / vega
        if iv <= 0.001:
            iv = 0.001
        if abs(diff) < tol:
            return iv
    return iv if abs(diff) < tol * 10 else None


async def main():
    # Connect to TT
    tt = TastyTradeClient(mode="manual")
    await tt.connect()

    # Get underlying price
    q = await tt.get_quote("/ESM26:XCME")
    spot = q["price"]
    print(f"ES spot: {spot:.2f}")

    # TTE
    now_et = datetime.now(ET)
    close_dt = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    tte_hours = max(0.01, (close_dt - now_et).total_seconds() / 3600)
    tte_years = tte_hours / (252 * 6.5)
    print(f"TTE: {tte_hours:.2f}h ({tte_years:.6f}y)")

    # Load ANN
    ann = PriceANN.load("models/price_ann_v3.pt")
    print("PriceANN v3 loaded\n")

    # Get option chain
    chain = await tt.get_future_option_chain("ES", underlying_price=spot)
    print(f"Chain: {len(chain)} options\n")

    # Header
    print(f"{'Strike':>7} {'Type':>4} {'Delta':>6} │ {'Mkt Bid':>7} {'Mkt Ask':>7} {'Mkt Mid':>7} │ "
          f"{'BS Theo':>7} {'BS Err':>6} │ {'ANN Bid':>7} {'ANN Ask':>7} {'ANN Mid':>7} {'ANN Err':>7}")
    print("─" * 115)

    results = []
    for opt in sorted(chain, key=lambda x: (x["option_type"], x["strike"])):
        strike = opt["strike"]
        bid = opt.get("bid", 0)
        ask = opt.get("ask", 0)
        opt_type = opt["option_type"]
        delta = opt.get("delta", 0)

        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2
        if mid < 0.05:
            continue
        moneyness = spot / strike if strike > 0 else 1.0
        if moneyness < 0.96 or moneyness > 1.04:
            continue

        # BS
        miv = bs_iv(mid, spot, strike, tte_years, opt_type)
        bs_p = bs_price(spot, strike, tte_years, miv, opt_type) if miv else None
        bs_err = (bs_p - mid) if bs_p is not None else None

        # ANN
        is_call = opt_type == "call"
        ann_bid, ann_ask = ann.predict_bid_ask("ES", spot, strike, tte_hours, is_call)
        ann_mid = (ann_bid + ann_ask) / 2
        ann_err = ann_mid - mid

        print(f"{strike:7.0f} {opt_type:>4} {delta:>6.3f} │ "
              f"{bid:7.2f} {ask:7.2f} {mid:7.2f} │ "
              f"{bs_p:7.2f} {bs_err:>+6.2f} │ " if bs_p else f"{strike:7.0f} {opt_type:>4} {delta:>6.3f} │ "
              f"{bid:7.2f} {ask:7.2f} {mid:7.2f} │ "
              f"{'N/A':>7} {'N/A':>6} │ ",
              end="")
        print(f"{ann_bid:7.2f} {ann_ask:7.2f} {ann_mid:7.2f} {ann_err:>+7.2f}")

        results.append({"bs_err": bs_err, "ann_err": ann_err, "mid": mid})

    # Summary
    ann_errs = [abs(r["ann_err"]) for r in results]
    bs_errs = [abs(r["bs_err"]) for r in results if r["bs_err"] is not None]
    ann_pct = [abs(r["ann_err"]) / r["mid"] * 100 for r in results if r["mid"] > 0]
    bs_pct = [abs(r["bs_err"]) / r["mid"] * 100 for r in results if r["bs_err"] is not None and r["mid"] > 0]

    print("─" * 115)
    print(f"Options: {len(results)}")
    print(f"BS  avg err: {sum(bs_errs)/len(bs_errs):.2f} pts ({sum(bs_pct)/len(bs_pct):.1f}%)" if bs_errs else "BS: N/A")
    print(f"ANN avg err: {sum(ann_errs)/len(ann_errs):.2f} pts ({sum(ann_pct)/len(ann_pct):.1f}%)" if ann_errs else "ANN: N/A")

    # Save to jsonl
    out = Path("data/iv_comparison") / f"{date.today().isoformat()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved to {out}")

    await tt.close()


if __name__ == "__main__":
    asyncio.run(main())
