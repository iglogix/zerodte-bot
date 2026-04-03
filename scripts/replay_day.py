#!/usr/bin/env python3
"""
Replay a historical trading day through the full bot pipeline.

Uses real ES bars for ORB/price and real SPX CBBO quotes for options.
Validates the complete entry flow: ORB → breakout → midline touch → spread selection.

Usage:
    python scripts/replay_day.py 2025-10-13
    python scripts/replay_day.py 2025-10-13 --config config/config_8am_spx_options_em.yaml
    python scripts/replay_day.py 2025-10-13 --step  # pause at each event
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_8am.backtest.cbbo_store import CbboStore
from src_8am.backtest.data_loader import DatabentoLoader
from src_8am.services.replay_client import ReplayTTClient

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Quiet noisy loggers
    logging.getLogger("src.services").setLevel(logging.WARNING)


async def replay(trading_date: date, config: dict, step_mode: bool = False):
    """Replay one day through the bot pipeline."""

    # Load ES bars
    data_file = config.get("backtest", {}).get("data_file", "glbx-mdp3-20210227-20260226.ohlcv-1m.csv")
    loader = DatabentoLoader(data_file)
    day_chunks = [d for d in loader.iter_days(start_date=trading_date, end_date=trading_date)]
    if not day_chunks:
        # Try continuation file
        data_file_2 = config.get("backtest", {}).get("data_file_2", "")
        if data_file_2 and Path(data_file_2).exists():
            loader = DatabentoLoader(data_file_2)
            day_chunks = [d for d in loader.iter_days(start_date=trading_date, end_date=trading_date)]
    if not day_chunks:
        print(f"No ES bars for {trading_date}")
        return

    # Select highest-volume futures contract (not options)
    futures = [d for d in day_chunks if " " not in d.symbol]  # filter out "ESH6 C6500" etc.
    if not futures:
        print(f"No ES futures bars for {trading_date} (only options found)")
        return
    best = max(futures, key=lambda d: sum(b.get("volume", 0) for b in d.bars))
    bars = best.bars
    print(f"ES bars: {best.symbol}, {len(bars)} bars")

    # Load CBBO
    cbbo_cache_dir = config.get("options", {}).get("cbbo_cache_dir", "data/cbbo_cache")
    cbbo = {}
    pkl_path = Path(cbbo_cache_dir) / f"{trading_date.isoformat()}.pkl"
    if pkl_path.exists():
        import pickle
        with open(pkl_path, "rb") as f:
            cbbo = pickle.load(f)
        print(f"CBBO quotes: {len(cbbo)} entries")
    else:
        print(f"WARNING: No CBBO cache for {trading_date}")

    # Create replay client
    client = ReplayTTClient(bars=bars, cbbo_quotes=cbbo, config=config)

    # Import bot components
    from src_8am.services.orb_options_service import OrbOptionsService
    from src_8am.agent.options_tool_handlers import OrbOptionsToolHandlers

    opts_service = OrbOptionsService(client, config)
    market_data = type("MD", (), {
        "get_current_price": lambda self: asyncio.coroutine(lambda: client._current_price())(),
    })()

    # Manual replay: step through the day
    strategy = config.get("strategy", {})
    orb_start = time(*[int(x) for x in strategy.get("orb_window_start", "08:00").split(":")])
    orb_end = time(*[int(x) for x in strategy.get("orb_window_end", "08:15").split(":")])
    market_open = time(*[int(x) for x in strategy.get("market_open", "09:30").split(":")])
    eod = time(*[int(x) for x in strategy.get("eod_force_close", "15:50").split(":")])
    tolerance = strategy.get("midline_tolerance_pts", 0.25)
    min_range = strategy.get("orb_range_filter", {}).get("min_pts", 2.0)
    max_range = strategy.get("orb_range_filter", {}).get("max_pts", 30.0)

    # Phase A: ORB
    orb_bars = [b for b in bars if orb_start <= b["ts_et"].time() < orb_end]
    if not orb_bars:
        print("No ORB bars — skipping day")
        return

    orb_high = max(b["high"] for b in orb_bars)
    orb_low = min(b["low"] for b in orb_bars)
    orb_mid = (orb_high + orb_low) / 2
    orb_range = orb_high - orb_low

    print(f"\n=== Phase A: ORB ===")
    print(f"  High={orb_high:.2f}  Low={orb_low:.2f}  Mid={orb_mid:.2f}  Range={orb_range:.2f}")

    if orb_range < min_range:
        print(f"  SKIP: range {orb_range:.2f} < min {min_range}")
        return
    if orb_range > max_range:
        print(f"  SKIP: range {orb_range:.2f} > max {max_range}")
        return

    # Phase B: Entry watch
    print(f"\n=== Phase B: Entry Watch ===")
    post_open = [b for b in bars if b["ts_et"].time() >= market_open]

    breakout_bias = None
    midline_touched = False
    entry_bar = None

    for bar in post_open:
        ts = bar["ts_et"]
        price = bar["close"]
        client.set_time(ts)

        if ts.time() >= eod:
            print(f"  {ts.strftime('%H:%M')} EOD — no entry")
            break

        # Breakout detection
        if breakout_bias is None:
            if price > orb_high:
                breakout_bias = "long"
                print(f"  {ts.strftime('%H:%M')} BREAKOUT long @ {price:.2f} (above {orb_high:.2f})")
            elif price < orb_low:
                breakout_bias = "short"
                print(f"  {ts.strftime('%H:%M')} BREAKOUT short @ {price:.2f} (below {orb_low:.2f})")
            continue

        # Midline touch
        if abs(price - orb_mid) <= tolerance:
            print(f"  {ts.strftime('%H:%M')} MIDLINE TOUCH @ {price:.2f} (mid={orb_mid:.2f}, tol={tolerance})")
            entry_bar = bar
            midline_touched = True
            break

    if not midline_touched:
        print("  No midline touch — no entry")
        return

    # Phase B: Spread selection
    print(f"\n=== Phase B: Spread Selection ===")
    print(f"  Bias: {breakout_bias}  Spot: {entry_bar['close']:.2f}")

    spot = entry_bar["close"]
    chain = await client.get_option_chain("SPX") if config.get("options", {}).get("option_root") == "SPXW" \
        else await client.get_future_option_chain("ES", spot)

    puts_with_bid = [o for o in chain if o["option_type"] == "put" and o["bid"] > 0]
    calls_with_bid = [o for o in chain if o["option_type"] == "call" and o["bid"] > 0]
    print(f"  Chain: {len(chain)} options ({len(puts_with_bid)} puts, {len(calls_with_bid)} calls with bid)")

    # ATM straddle
    straddle = opts_service.atm_straddle(chain, spot)
    print(f"  ATM Straddle (EM): {straddle:.2f}")

    if opts_service._use_em:
        em_wt = opts_service._em_with_trend
        em_at = opts_service._em_against_trend
        w_wt = opts_service._width_with_trend
        w_at = opts_service._width_against_trend

        for leg_type in ["put_credit_spread", "call_credit_spread"]:
            spread = opts_service.select_em_spread(chain, spot, leg_type, breakout_bias)
            if spread:
                print(f"\n  {spread['spread_type']}:")
                print(f"    Short={spread['short_strike']}  Long={spread['long_strike']}  Width={spread['spread_width']}")
                print(f"    Credit={spread['credit']:.2f}  TP={spread['tp_debit']:.2f}  SL={spread['sl_debit']:.2f}")
                print(f"    EM={spread.get('atm_straddle', 0):.1f} × {spread.get('em_multiplier', 0):.2f} = {spread.get('em_distance', 0):.0f} pts ({spread.get('trend_type', '')})")
                print(f"    Short bid={spread['short_bid']:.2f} ask={spread['short_ask']:.2f}")
                print(f"    Long  bid={spread['long_bid']:.2f} ask={spread['long_ask']:.2f}")
            else:
                print(f"\n  {leg_type}: NO VALID SPREAD (credit < min)")
    else:
        # Delta-based
        if breakout_bias == "long":
            spread = opts_service.select_put_spread(chain, orb_low, spot)
        else:
            spread = opts_service.select_call_spread(chain, orb_high, spot)
        if spread:
            print(f"\n  {spread['spread_type']}:")
            print(f"    Short={spread['short_strike']}  Long={spread['long_strike']}  Credit={spread['credit']:.2f}")
        else:
            print("  NO VALID SPREAD")

    if step_mode:
        input("\nPress Enter to continue to Phase C simulation...")

    # Phase C: Simulate position monitoring
    if not entry_bar or not midline_touched:
        return

    print(f"\n=== Phase C: Position Monitor (post-entry bars) ===")
    entry_idx = bars.index(entry_bar)
    post_entry = bars[entry_idx + 1:]
    opt_cfg = config.get("options", {})
    hold_put = opt_cfg.get("hold_put_breach", False)

    for leg_type in ["put_credit_spread", "call_credit_spread"]:
        spread = opts_service.select_em_spread(chain, spot, leg_type, breakout_bias) if opts_service._use_em else None
        if not spread:
            continue

        credit = spread["credit"]
        tp_pct = opt_cfg.get("tp_pct_of_credit", 0.50)
        tp_debit = round(credit * (1 - tp_pct), 2)
        short_k = spread["short_strike"]
        is_put = "put" in leg_type

        print(f"\n  Monitoring {leg_type} short={short_k} credit={credit:.2f} tp_debit={tp_debit:.2f}")

        for bar in post_entry:
            ts = bar["ts_et"]
            if ts.time() >= eod:
                print(f"    {ts.strftime('%H:%M')} EOD → expired worthless | PnL=${credit * 100:.0f}")
                break

            client.set_time(ts)
            minute = ts.strftime("%H:%M")

            # Check breach
            if is_put and bar["low"] <= short_k:
                if hold_put:
                    pass  # hold through
                else:
                    print(f"    {ts.strftime('%H:%M')} PUT BREACH @ {bar['low']:.2f} <= {short_k}")
                    break
            elif not is_put and bar["high"] >= short_k:
                print(f"    {ts.strftime('%H:%M')} CALL BREACH @ {bar['high']:.2f} >= {short_k}")
                break

            # Check spread mark from CBBO
            short_q = cbbo.get((minute, short_k, not is_put))
            long_k = spread["long_strike"]
            long_q = cbbo.get((minute, long_k, not is_put))
            if short_q and long_q:
                mark = (short_q[0] + short_q[1]) / 2 - (long_q[0] + long_q[1]) / 2
                if mark <= tp_debit:
                    pnl = (credit - tp_debit) * 100
                    print(f"    {ts.strftime('%H:%M')} TP HIT mark={mark:.2f} <= {tp_debit:.2f} | PnL=${pnl:.0f}")
                    break
        else:
            print(f"    EOD → expired worthless | PnL=${credit * 100:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Replay a historical trading day")
    parser.add_argument("date", help="Trading date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/config_8am_spx_options_em.yaml")
    parser.add_argument("--step", action="store_true", help="Pause at each phase")
    args = parser.parse_args()

    setup_logging()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trading_date = date.fromisoformat(args.date)
    print(f"Replaying {trading_date} with {args.config}")
    print(f"Strategy: use_straddle_em={config.get('options', {}).get('use_straddle_em', False)} "
          f"spread_mode={config.get('options', {}).get('spread_mode', 'directional')}")

    asyncio.run(replay(trading_date, config, step_mode=args.step))


if __name__ == "__main__":
    main()
