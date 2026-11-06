"""
Post-breach price recovery analysis for ES and SPX strategies.

For each breached trade, replays raw bar data from breach time to EOD to answer:
- Does price return OTM after breaching the short strike?
- How long does it take?
- How deep does it go before recovering?
- What's the EOD price relative to the short strike?

This informs whether taking an ATM directional short after breach could recover losses.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date, time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from es_8am.backtest.data_loader import DatabentoLoader
from es_8am.backtest.bar_loader import BarLoader
from es_8am.backtest.options_engine import OrbOptionsBacktestEngine


def run_backtest(config_path, params, start, end):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for k, v in params.items():
        config["options"][k] = v

    width = params["spread_width_pts"]
    mult = config["options"].get("contract_multiplier", 50)
    qty = params["quantity"]
    needed = width * mult * qty
    config.setdefault("risk", {})["max_loss_per_trade_usd"] = needed
    config["risk"]["max_daily_loss_usd"] = needed

    data_file = config["backtest"]["data_file"]
    data_source = config.get("backtest", {}).get("data_source", "databento")

    if data_source == "spy":
        price_scale = config.get("backtest", {}).get("price_scale", 10.0)
        loader = BarLoader(data_file, price_scale=price_scale, symbol="SPX")
    else:
        loader = DatabentoLoader(data_file)

    print(f"Loading bars from {data_file}...")
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} day-symbol chunks")

    # Keep raw day data for post-breach replay
    days_by_date = {}
    for day in days:
        days_by_date[day.session_date] = day.bars

    print("Running backtest...")
    engine = OrbOptionsBacktestEngine(config=config)
    results = engine.run_on_days(days)
    trades = [r for r in results if r.status == "closed"]
    print(f"  {len(trades)} trades\n")

    return trades, days_by_date, config


def analyze_post_breach(trades, days_by_date, config, label):
    eod_time = time(15, 50)

    breached = [t for t in trades if t.exit_reason == "short_breached"]
    print("=" * 70)
    print(f"  POST-BREACH PRICE RECOVERY — {label}")
    print("=" * 70)
    print(f"\nTotal trades: {len(trades)}")
    print(f"Breached: {len(breached)} ({len(breached)/len(trades)*100:.1f}%)")

    if not breached:
        print("No breaches to analyze.")
        return

    # Track outcomes
    returned_otm = 0          # price went back OTM at some point
    stayed_otm_eod = 0        # price was OTM at EOD
    deeper_itm_eod = 0        # price ended deeper ITM than breach
    oscillated = 0            # crossed strike multiple times

    recovery_minutes = []     # time to first OTM return
    max_adverse = []          # max ITM depth (pts past strike)
    eod_distance = []         # EOD price distance from strike (+ = OTM, - = ITM)
    breach_details = []

    for t in breached:
        bars = days_by_date.get(t.session_date)
        if not bars:
            continue

        short_k = t.short_strike
        spread_type = t.spread_type
        breach_time = t.exit_time

        # Walk bars from breach onward
        post_breach = [b for b in bars if b["ts_et"] >= breach_time]
        if not post_breach:
            continue

        first_otm_time = None
        max_depth = 0.0
        crossings = 0
        was_itm = True  # starts ITM (just breached)

        for bar in post_breach:
            if bar["ts_et"].time() >= eod_time:
                break

            if spread_type == "put_credit_spread":
                # PUT: breached means price went below strike
                # OTM = price above strike
                depth = short_k - bar["low"]   # how far below strike
                is_otm_now = bar["close"] > short_k
            else:
                # CALL: breached means price went above strike
                # OTM = price below strike
                depth = bar["high"] - short_k  # how far above strike
                is_otm_now = bar["close"] < short_k

            max_depth = max(max_depth, depth)

            if is_otm_now and was_itm:
                crossings += 1
                if first_otm_time is None:
                    first_otm_time = bar["ts_et"]
            elif not is_otm_now and not was_itm:
                crossings += 1

            was_itm = not is_otm_now

        # EOD state
        eod_bars = [b for b in bars if b["ts_et"].time() < eod_time]
        eod_bar = eod_bars[-1] if eod_bars else post_breach[-1]

        if spread_type == "put_credit_spread":
            eod_dist = eod_bar["close"] - short_k  # + = OTM (above), - = ITM (below)
        else:
            eod_dist = short_k - eod_bar["close"]  # + = OTM (below), - = ITM (above)

        eod_is_otm = eod_dist > 0

        if first_otm_time is not None:
            returned_otm += 1
            mins = (first_otm_time - breach_time).total_seconds() / 60
            recovery_minutes.append(mins)

        if eod_is_otm:
            stayed_otm_eod += 1
        else:
            deeper_itm_eod += 1

        if crossings > 1:
            oscillated += 1

        max_adverse.append(max_depth)
        eod_distance.append(eod_dist)

        breach_details.append({
            "date": t.session_date,
            "bias": t.bias,
            "spread": spread_type,
            "strike": short_k,
            "max_depth": max_depth,
            "eod_dist": eod_dist,
            "returned_otm": first_otm_time is not None,
            "eod_otm": eod_is_otm,
            "crossings": crossings,
            "recovery_mins": (first_otm_time - breach_time).total_seconds() / 60 if first_otm_time else None,
        })

    n = len(breached)
    print(f"\n--- Post-breach price behavior ---")
    print(f"  Price returned OTM at some point:  {returned_otm}/{n} = {returned_otm/n*100:.1f}%")
    print(f"  Price OTM at EOD:                  {stayed_otm_eod}/{n} = {stayed_otm_eod/n*100:.1f}%")
    print(f"  Price deeper ITM at EOD:           {deeper_itm_eod}/{n} = {deeper_itm_eod/n*100:.1f}%")
    print(f"  Oscillated (crossed strike 2+ times): {oscillated}/{n} = {oscillated/n*100:.1f}%")

    if recovery_minutes:
        recovery_minutes.sort()
        print(f"\n--- Time to first OTM return (of {len(recovery_minutes)} that recovered) ---")
        print(f"  Fastest:  {recovery_minutes[0]:.0f} min")
        print(f"  Median:   {recovery_minutes[len(recovery_minutes)//2]:.0f} min")
        print(f"  Slowest:  {recovery_minutes[-1]:.0f} min")

    if max_adverse:
        max_adverse.sort()
        print(f"\n--- Max ITM depth after breach (pts past strike) ---")
        print(f"  Shallowest: {max_adverse[0]:.1f} pts")
        print(f"  Median:     {max_adverse[len(max_adverse)//2]:.1f} pts")
        print(f"  Deepest:    {max_adverse[-1]:.1f} pts")

    if eod_distance:
        eod_distance.sort()
        print(f"\n--- EOD distance from strike (+ = OTM, - = ITM) ---")
        print(f"  Most ITM:   {eod_distance[0]:+.1f} pts")
        print(f"  Median:     {eod_distance[len(eod_distance)//2]:+.1f} pts")
        print(f"  Most OTM:   {eod_distance[-1]:+.1f} pts")

    # By bias
    print(f"\n--- By bias ---")
    for bias in ("long", "short"):
        b = [d for d in breach_details if d["bias"] == bias]
        if not b:
            continue
        ret = sum(1 for d in b if d["returned_otm"])
        eod_otm = sum(1 for d in b if d["eod_otm"])
        avg_depth = sum(d["max_depth"] for d in b) / len(b)
        spread_label = "PUT" if bias == "long" else "CALL"
        print(f"  {spread_label} ({bias} bias): {len(b)} breaches, "
              f"returned OTM={ret} ({ret/len(b)*100:.0f}%), "
              f"OTM at EOD={eod_otm} ({eod_otm/len(b)*100:.0f}%), "
              f"avg max depth={avg_depth:.1f}pts")

    # By day of week
    print(f"\n--- By day of week ---")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for dow in range(5):
        b = [d for d in breach_details if d["date"].weekday() == dow]
        if not b:
            continue
        ret = sum(1 for d in b if d["returned_otm"])
        eod_otm = sum(1 for d in b if d["eod_otm"])
        print(f"  {dow_names[dow]}: {len(b)} breaches, "
              f"returned OTM={ret} ({ret/len(b)*100:.0f}%), "
              f"OTM@EOD={eod_otm} ({eod_otm/len(b)*100:.0f}%)")

    # Implication for ATM directional trade after breach
    print(f"\n--- Implication for post-breach directional recovery trade ---")
    # If price stays ITM at EOD, a directional short (ATM put for long bias breach,
    # ATM call for short bias breach) would profit.
    itm_eod_pct = deeper_itm_eod / n * 100
    otm_eod_pct = stayed_otm_eod / n * 100
    print(f"  Price stays ITM through EOD: {itm_eod_pct:.0f}% — directional trade wins")
    print(f"  Price returns OTM by EOD:    {otm_eod_pct:.0f}% — directional trade loses")
    if itm_eod_pct > 60:
        print(f"  -> Favorable: {itm_eod_pct:.0f}% of breaches continue in breach direction")
    elif itm_eod_pct > 45:
        print(f"  -> Marginal: roughly coin flip")
    else:
        print(f"  -> Unfavorable: price tends to recover, directional trade would lose")


def main():
    # ES analysis
    print("\n" + "#" * 70)
    print("#  ES 8AM ORB — delta=0.16, width=20, qty=25")
    print("#" * 70 + "\n")

    es_params = {
        "target_short_delta": 0.16,
        "spread_width_pts": 20,
        "tp_pct_of_credit": 0.75,
        "sl_mult_of_credit": 0,
        "slippage_per_spread": 0.12,
        "quantity": 25,
    }
    es_trades, es_days, es_config = run_backtest(
        "config/config_8am_es_options.yaml", es_params,
        date(2021, 3, 1), date(2026, 2, 26)
    )
    analyze_post_breach(es_trades, es_days, es_config, "ES (delta=0.16, width=20)")

    # SPX analysis
    print("\n\n" + "#" * 70)
    print("#  SPX 0DTE — delta=0.25, width=5, qty=50")
    print("#" * 70 + "\n")

    spx_params = {
        "target_short_delta": 0.25,
        "spread_width_pts": 5,
        "tp_pct_of_credit": 0.75,
        "sl_mult_of_credit": 0,
        "slippage_per_spread": 0.10,
        "quantity": 50,
    }
    spx_trades, spx_days, spx_config = run_backtest(
        "config/config_8am_spx_options.yaml", spx_params,
        date(2022, 5, 11), date(2026, 2, 28)
    )
    analyze_post_breach(spx_trades, spx_days, spx_config, "SPX (delta=0.25, width=5)")


if __name__ == "__main__":
    main()
