"""
Touch & Turn Scalper — Interactive trade chart with intraday candles,
15-min opening range, Fibonacci levels, TP/SL lines, and entry/exit markers.

Uses TradingView's lightweight-charts.

Usage:
  python -m touch_turn.run_chart
  python -m touch_turn.run_chart --config touch_turn/config_nvda.yaml
  python -m touch_turn.run_chart --start 2024-01-01 --end 2024-06-30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from es_8am.backtest.data_loader import DatabentoLoader, DayBars
from touch_turn.backtest.engine import TouchTurnBacktestEngine, TradeResult
from touch_turn.backtest.equity_loader import EquityLoader

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_5min(bars: List[Dict]) -> List[Dict]:
    """Aggregate 1-min bars into 5-min OHLCV for chart display."""
    if not bars:
        return []
    out = []
    bucket = None
    for b in bars:
        ts: datetime = b["ts_et"]
        bucket_ts = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
        if bucket is None or bucket["time"] != bucket_ts:
            if bucket is not None:
                out.append(bucket)
            bucket = {
                "time": bucket_ts,
                "open": b["open"],
                "high": b["high"],
                "low": b["low"],
                "close": b["close"],
                "volume": b.get("volume", 0),
            }
        else:
            bucket["high"] = max(bucket["high"], b["high"])
            bucket["low"] = min(bucket["low"], b["low"])
            bucket["close"] = b["close"]
            bucket["volume"] += b.get("volume", 0)
    if bucket is not None:
        out.append(bucket)
    return out


def _ts_to_unix(dt: datetime) -> int:
    """Convert datetime to Unix timestamp, displaying ET wall-clock time."""
    ts = int(dt.timestamp())
    offset = dt.utcoffset()
    if offset is not None:
        ts += int(offset.total_seconds())
    return ts


def _snap_5min(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def _time_to_unix(d: date, t: time) -> int:
    dt = datetime.combine(d, t, tzinfo=ET)
    return _ts_to_unix(dt)


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def _compute_daily_atr(daily_ohlc: List[Dict], period: int = 14) -> Optional[float]:
    """Compute ATR from a list of daily OHLC dicts using true range."""
    if len(daily_ohlc) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(daily_ohlc)):
        prev_c = daily_ohlc[i - 1]["close"]
        curr = daily_ohlc[i]
        tr = max(curr["high"] - curr["low"],
                 abs(curr["high"] - prev_c),
                 abs(curr["low"] - prev_c))
        true_ranges.append(tr)
    recent = true_ranges[-period:]
    return sum(recent) / len(recent)


def process_data(days: List[DayBars], trades: List[TradeResult], config: dict):
    """Build chart-ready JSON from day bars + trade results.

    Shows ORB high/low and fib 0.382/0.618 lines on ALL qualifying ORB days,
    with day-break gaps (no cross-day connections). TP/SL lines only on
    filled trade days. ATR annotation on each ORB day.
    """

    strat = config.get("strategy", {})
    orb_start = time(9, 30)
    orb_end_str = strat.get("orb_candle_end", "09:45")
    orb_end = time(int(orb_end_str.split(":")[0]), int(orb_end_str.split(":")[1]))
    display_end = time(16, 0)
    atr_period = strat.get("atr_period", 14)
    atr_min_pct = strat.get("atr_min_pct", 0.25)

    # Index trades by session_date
    trades_by_date: Dict[date, TradeResult] = {}
    for t in trades:
        trades_by_date[t.session_date] = t

    # Best volume per date (dedup multi-contract)
    by_date: Dict[date, DayBars] = {}
    for d in days:
        sd = d.session_date
        if sd not in by_date:
            by_date[sd] = d
        else:
            rth_count_new = sum(1 for b in d.bars if time(9, 30) <= b["ts_et"].time() < time(16, 0))
            rth_count_old = sum(1 for b in by_date[sd].bars if time(9, 30) <= b["ts_et"].time() < time(16, 0))
            if rth_count_new > rth_count_old:
                by_date[sd] = d

    candles = []
    level_lines: List[Dict] = []   # per-day line segments [{c, s, d:[{time,value}]}]
    markers = []
    trade_details = {}
    equity = []
    running_pnl = 0.0

    # Build rolling daily OHLC for independent ATR calculation
    daily_ohlc_history: List[Dict] = []

    for sd in sorted(by_date.keys()):
        day = by_date[sd]
        bars_5m = _aggregate_5min(day.bars)
        trade = trades_by_date.get(sd)
        is_trade_day = trade is not None and trade.exit_reason not in ("expired", None)
        is_loss = is_trade_day and trade.pnl_usd < 0

        # --- Compute daily OHLC (RTH only) for ATR ---
        rth_1m = [b for b in day.bars if time(9, 30) <= b["ts_et"].time() < time(16, 0)]
        if rth_1m:
            daily_ohlc_history.append({
                "open": rth_1m[0]["open"],
                "high": max(b["high"] for b in rth_1m),
                "low": min(b["low"] for b in rth_1m),
                "close": rth_1m[-1]["close"],
            })

        # --- Compute ORB independently for ALL days ---
        orb_1m = [b for b in day.bars if orb_start <= b["ts_et"].time() < orb_end]
        has_orb = len(orb_1m) >= 5
        orb_high = max(b["high"] for b in orb_1m) if has_orb else None
        orb_low = min(b["low"] for b in orb_1m) if has_orb else None
        orb_range = (orb_high - orb_low) if has_orb else None

        # ATR for this day
        atr = _compute_daily_atr(daily_ohlc_history, atr_period)

        # Determine if this is a qualifying ORB day (has ORB and passes ATR filter)
        qualifying_orb = (has_orb and orb_range and orb_range > 0
                          and atr is not None and orb_range >= atr_min_pct * atr)

        # Fib levels
        fib_382 = orb_low + 0.382 * orb_range if qualifying_orb else None
        fib_618 = orb_low + 0.618 * orb_range if qualifying_orb else None

        first_bar_ts = None

        # Determine ORB candle bias color (bearish=red, bullish=green)
        orb_open = orb_1m[0]["open"] if has_orb else None
        orb_close = orb_1m[-1]["close"] if has_orb else None
        if has_orb and orb_close < orb_open:
            orb_color = "#ef5350"      # red = bearish candle → long bias
            orb_border = "#ef5350"
        elif has_orb and orb_close > orb_open:
            orb_color = "#26a69a"      # green = bullish candle → short bias
            orb_border = "#26a69a"
        else:
            orb_color = "#787b86"      # gray = doji
            orb_border = "#787b86"

        # Collect per-day time points for level lines
        day_times: List[int] = []

        for b in bars_5m:
            bar_time = b["time"].time()
            ts = _ts_to_unix(b["time"])
            if first_bar_ts is None:
                first_bar_ts = ts

            candle: Dict[str, Any] = {
                "time": ts,
                "open": round(b["open"], 2),
                "high": round(b["high"], 2),
                "low": round(b["low"], 2),
                "close": round(b["close"], 2),
            }

            # Tint ORB-window candles with bias color
            if orb_start <= bar_time < orb_end:
                candle["color"] = orb_color
                candle["borderColor"] = orb_border
                candle["wickColor"] = orb_border
            elif is_loss:
                candle["color"] = "#5c2020"
                candle["borderColor"] = "#ef5350"
                candle["wickColor"] = "#ef5350"

            candles.append(candle)

            # Collect timestamps for level lines (after ORB close, during RTH)
            if qualifying_orb and orb_end <= bar_time <= display_end:
                day_times.append(ts)

        # Create per-day line segments (each is an independent series → no cross-day)
        if qualifying_orb and day_times:
            def _mk(val: float, ts_list: List[int]) -> List[Dict]:
                return [{"time": t, "value": round(val, 2)} for t in ts_list]

            # ORB high/low + fib on ALL qualifying days
            level_lines.append({"c": "rgba(38,166,154,0.6)",  "s": 0, "d": _mk(orb_high, day_times)})
            level_lines.append({"c": "rgba(239,83,80,0.6)",   "s": 0, "d": _mk(orb_low, day_times)})
            level_lines.append({"c": "rgba(171,71,188,0.6)",  "s": 2, "d": _mk(fib_382, day_times)})
            level_lines.append({"c": "rgba(0,188,212,0.6)",   "s": 2, "d": _mk(fib_618, day_times)})

            # TP/SL/entry only on filled trade days
            if is_trade_day:
                level_lines.append({"c": "rgba(255,183,77,0.7)",  "s": 2, "d": _mk(trade.entry_price, day_times)})
                level_lines.append({"c": "rgba(0,230,118,0.7)",   "s": 2, "d": _mk(trade.tp_price, day_times)})
                level_lines.append({"c": "rgba(255,23,68,0.7)",   "s": 2, "d": _mk(trade.sl_price, day_times)})

        # ATR annotation marker at ORB close candle
        if qualifying_orb and atr is not None:
            orb_close_ts = _time_to_unix(sd, orb_end)
            markers.append({
                "time": orb_close_ts,
                "position": "aboveBar",
                "color": "#787b86",
                "shape": "circle",
                "text": f"ATR {atr:.1f}",
            })

        # Trade markers + equity
        if is_trade_day:
            pnl = trade.pnl_usd
            running_pnl += pnl
            is_win = pnl >= 0

            if trade.entry_time:
                entry_ts = _ts_to_unix(_snap_5min(trade.entry_time))
                markers.append({
                    "time": entry_ts,
                    "position": "belowBar" if trade.direction == "long" else "aboveBar",
                    "color": "#26a69a" if is_win else "#ef5350",
                    "shape": "arrowUp" if trade.direction == "long" else "arrowDown",
                    "text": f"{'W' if is_win else 'L'} ${pnl:+.0f}",
                })

                trade_details[str(entry_ts)] = {
                    "date": sd.isoformat(),
                    "direction": trade.direction.upper(),
                    "entry": round(trade.entry_price, 2),
                    "fill": round(trade.fill_price, 2),
                    "tp": round(trade.tp_price, 2),
                    "sl": round(trade.sl_price, 2),
                    "exit": round(trade.exit_price, 2),
                    "pnl": round(pnl, 2),
                    "exit_reason": trade.exit_reason,
                    "orb_high": round(trade.orb_high, 2),
                    "orb_low": round(trade.orb_low, 2),
                    "orb_range": round(trade.orb_range, 2),
                    "atr": round(trade.daily_atr, 2),
                    "entry_time": trade.entry_time.strftime("%H:%M") if trade.entry_time else "",
                    "exit_time": trade.exit_time.strftime("%H:%M") if trade.exit_time else "",
                }

            if trade.exit_time:
                exit_ts = _ts_to_unix(_snap_5min(trade.exit_time))
                markers.append({
                    "time": exit_ts,
                    "position": "aboveBar" if trade.direction == "long" else "belowBar",
                    "color": "#26a69a" if is_win else "#ef5350",
                    "shape": "circle",
                    "text": f"{trade.exit_reason} ${pnl:+.0f}",
                })

        if first_bar_ts is not None:
            equity.append({"time": first_bar_ts, "value": round(running_pnl, 2)})

    return candles, level_lines, markers, trade_details, equity


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(candles, level_lines, markers, trade_details, equity,
                  summary, config, out_path):
    symbol = config.get("bot", {}).get("symbol", "ES")
    qty = config.get("risk", {}).get("position_size", 1)
    mult = config.get("risk", {}).get("contract_multiplier", 50)

    strat = config.get("strategy", {})
    params_str = (
        f"fib_long={strat.get('fib_tp_long', 0.382)}  "
        f"fib_short={strat.get('fib_tp_short', 0.618)}  "
        f"atr_min={strat.get('atr_min_pct', 0.25)}  "
        f"sl_ratio={strat.get('sl_ratio', 0.5)}"
    )

    title = (
        f"Touch &amp; Turn Scalper ({symbol}) — "
        f"{summary['trades']} trades, "
        f"WR {summary['win_rate']*100:.1f}%, "
        f"Sharpe {summary['sharpe']:.2f}, "
        f"PnL ${summary['total_pnl_usd']:,.0f} | {params_str}"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #131722; color: #d1d4dc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  #header {{ padding: 8px 16px; font-size: 13px; border-bottom: 1px solid #2a2e39; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  #header h1 {{ font-size: 14px; font-weight: 600; color: #e0e3eb; white-space: nowrap; }}
  #nav {{ display: flex; align-items: center; gap: 6px; }}
  #nav select, #nav button {{
    background: #1e222d; color: #d1d4dc; border: 1px solid #2a2e39;
    border-radius: 4px; padding: 4px 8px; font-size: 12px; cursor: pointer;
  }}
  #nav button:hover {{ background: #2a2e39; }}
  #nav button.active {{ background: #1976d2; color: #fff; border-color: #1976d2; }}
  #candle-chart {{ width: 100%; height: calc(100vh - 42px); }}
  #tooltip {{
    position: fixed; top: 60px; right: 16px;
    background: rgba(19,23,34,0.95); border: 1px solid #2a2e39;
    border-radius: 6px; padding: 10px 14px;
    font-size: 12px; line-height: 1.6;
    min-width: 220px; pointer-events: none;
    display: none; z-index: 100;
  }}
  #tooltip .label {{ color: #787b86; }}
  #tooltip .win {{ color: #26a69a; font-weight: 600; }}
  #tooltip .loss {{ color: #ef5350; font-weight: 600; }}
  #tooltip .val {{ color: #e0e3eb; }}
  .legend {{ position: absolute; top: 8px; left: 16px; font-size: 11px; color: #787b86; z-index: 10; }}
  .legend span {{ margin-right: 12px; }}
  #helpOverlay {{
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7);
    z-index: 200; justify-content: center; align-items: center;
  }}
  #helpOverlay.show {{ display: flex; }}
  #helpBox {{
    background: #1e222d; border: 1px solid #2a2e39; border-radius: 8px;
    padding: 24px 32px; max-width: 520px; color: #d1d4dc; font-size: 13px; line-height: 1.8;
  }}
  #helpBox h2 {{ font-size: 15px; color: #e0e3eb; margin-bottom: 12px; }}
  #helpBox h3 {{ font-size: 13px; color: #e0e3eb; margin-top: 14px; margin-bottom: 4px; }}
  #helpBox .legend-item {{ display: flex; align-items: center; gap: 8px; }}
  #helpBox .swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 2px; }}
</style>
</head>
<body>
<div id="header">
  <h1>{title}</h1>
  <div id="nav">
    <select id="yearSelect">
      <option value="all">All Years</option>
      <option value="2022">2022</option>
      <option value="2023">2023</option>
      <option value="2024">2024</option>
      <option value="2025">2025</option>
    </select>
    <button id="btnZoomOut" title="Zoom out">&minus;</button>
    <button id="btnZoomIn" title="Zoom in">+</button>
    <button id="btnLeft" title="Pan left">&#9664;</button>
    <button id="btnRight" title="Pan right">&#9654;</button>
    <button id="btn1M">1M</button>
    <button id="btn3M">3M</button>
    <button id="btn1Y">1Y</button>
    <button id="btnAll">All</button>
    <button id="btnHelp" style="margin-left:8px; font-weight:700;">?</button>
  </div>
  <span style="font-size:11px; color:#787b86; margin-left:auto;">{qty} contract(s) x ${mult}/pt</span>
</div>
<div id="candle-chart" style="position:relative;">
  <div class="legend">
    <span style="color:#ef5350;">&block;</span><span style="color:#26a69a;">&block;</span> ORB (red=bearish green=bullish)
    <span style="color:#26a69a;">&mdash; ORB High</span>
    <span style="color:#ef5350;">&mdash; ORB Low</span>
    <span style="color:#ab47bc;">&mdash; Fib 0.382</span>
    <span style="color:#00bcd4;">&mdash; Fib 0.618</span>
    <span style="color:#ffb74d;">&mdash; Entry Limit</span>
    <span style="color:#00e676;">&mdash; TP</span>
    <span style="color:#ff1744;">&mdash; SL</span>
    <span style="color:#787b86;">&#9679; ATR</span>
    <span style="color:#1976d2;">&mdash; P&amp;L</span>
    <span style="color:#5c2020;">&block; <span style="color:#ef5350;">Loss day</span></span>
  </div>
</div>
<div id="tooltip"></div>

<div id="helpOverlay">
<div id="helpBox">
  <h2>Touch &amp; Turn Chart</h2>
  <h3>Chart Elements</h3>
  <div class="legend-item"><span class="swatch" style="background:#ef5350;"></span><span class="swatch" style="background:#26a69a;"></span> ORB candles: red = bearish (long bias), green = bullish (short bias)</div>
  <div class="legend-item"><span style="color:#26a69a;">&mdash;</span> ORB High &nbsp; <span style="color:#ef5350;">&mdash;</span> ORB Low &nbsp;(all qualifying days)</div>
  <div class="legend-item"><span style="color:#ab47bc;">&mdash;</span> Fib 0.382 &nbsp; <span style="color:#00bcd4;">&mdash;</span> Fib 0.618 &nbsp;(all qualifying days)</div>
  <div class="legend-item"><span style="color:#787b86;">&#9679;</span> ATR value annotation at ORB close</div>
  <div class="legend-item"><span style="color:#ffb74d;">&mdash;</span> Entry limit order level (trade days only)</div>
  <div class="legend-item"><span style="color:#00e676;">&mdash;</span> Take Profit (fib level)</div>
  <div class="legend-item"><span style="color:#ff1744;">&mdash;</span> Stop Loss</div>
  <div class="legend-item"><span style="color:#26a69a;">&#9650;</span><span style="color:#ef5350;">&#9660;</span> Entry markers &nbsp; &#9679; Exit markers</div>
  <div class="legend-item"><span style="color:#1976d2;">&mdash;</span> Cumulative P&amp;L</div>
  <h3>Navigation</h3>
  <b>Arrow keys</b> pan &bull; <b>+/-</b> zoom &bull; Drag to pan &bull; Scroll to zoom<br>
  <br><button onclick="document.getElementById('helpOverlay').classList.remove('show')" style="background:#2a2e39; color:#d1d4dc; border:1px solid #363a45; border-radius:4px; padding:6px 16px; cursor:pointer;">Close</button>
</div>
</div>

<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const candleData = {json.dumps(candles)};
const levelLines = {json.dumps(level_lines)};
const markerData = {json.dumps(markers)};
const tradeDetails = {json.dumps(trade_details)};
const equityData = {json.dumps(equity)};

const chartEl = document.getElementById('candle-chart');
const chart = LightweightCharts.createChart(chartEl, {{
  layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
  grid: {{ vertLines: {{ color: '#1e222d' }}, horzLines: {{ color: '#1e222d' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  timeScale: {{ timeVisible: true, secondsVisible: false, borderColor: '#2a2e39', minBarSpacing: 0.001 }},
  rightPriceScale: {{ borderColor: '#2a2e39' }},
  leftPriceScale: {{ visible: true, borderColor: '#2a2e39', scaleMargins: {{ top: 0.7, bottom: 0.02 }} }},
}});

const candleSeries = chart.addCandlestickSeries({{
  upColor: '#26a69a', downColor: '#ef5350',
  borderUpColor: '#26a69a', borderDownColor: '#ef5350',
  wickUpColor: '#26a69a', wickDownColor: '#ef5350',
}});
candleSeries.setData(candleData);

// Per-day level lines (ORB high/low, fib, TP/SL/entry — each day is its own series)
levelLines.forEach(line => {{
  chart.addLineSeries({{
    color: line.c, lineWidth: 1, lineStyle: line.s,
    lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
  }}).setData(line.d);
}});

// Trade markers
candleSeries.setMarkers(markerData.sort((a, b) => a.time - b.time));

// P&L equity overlay
chart.addAreaSeries({{
  topColor: 'rgba(25,118,210,0.15)', bottomColor: 'rgba(25,118,210,0.02)',
  lineColor: '#1976d2', lineWidth: 2,
  lastValueVisible: true, priceLineVisible: false, priceScaleId: 'left',
}}).setData(equityData);

// Tooltip
const tooltip = document.getElementById('tooltip');
chart.subscribeCrosshairMove((param) => {{
  if (!param.time || !param.seriesData) {{ tooltip.style.display = 'none'; return; }}
  const ts = String(param.time);
  let detail = tradeDetails[ts];
  if (!detail) {{
    for (const [key, val] of Object.entries(tradeDetails)) {{
      if (Math.abs(parseInt(key) - param.time) < 300) {{ detail = val; break; }}
    }}
  }}
  if (!detail) {{ tooltip.style.display = 'none'; return; }}
  const cls = detail.pnl >= 0 ? 'win' : 'loss';
  tooltip.innerHTML = `
    <div><span class="label">Date:</span> <span class="val">${{detail.date}}</span></div>
    <div><span class="label">Dir:</span> <span class="val">${{detail.direction}}</span></div>
    <div><span class="label">Entry limit:</span> <span class="val">${{detail.entry}}</span></div>
    <div><span class="label">Fill:</span> <span class="val">${{detail.fill}}</span></div>
    <div><span class="label">TP:</span> <span class="val">${{detail.tp}}</span></div>
    <div><span class="label">SL:</span> <span class="val">${{detail.sl}}</span></div>
    <div><span class="label">Exit:</span> <span class="val">${{detail.exit}} (${{detail.exit_reason}})</span></div>
    <div><span class="label">P&L:</span> <span class="${{cls}}">$${{detail.pnl.toFixed(0)}}</span></div>
    <div><span class="label">ORB:</span> <span class="val">${{detail.orb_high}} / ${{detail.orb_low}} (range ${{detail.orb_range}})</span></div>
    <div><span class="label">ATR:</span> <span class="val">${{detail.atr}}</span></div>
    <div><span class="label">Time:</span> <span class="val">${{detail.entry_time}} → ${{detail.exit_time}}</span></div>
  `;
  tooltip.style.display = 'block';
}});

// Resize
new ResizeObserver(() => {{
  chart.applyOptions({{ width: chartEl.clientWidth, height: chartEl.clientHeight }});
}}).observe(chartEl);

// Navigation
const totalBars = candleData.length;
const BARS_1D = 78;  // ~78 five-min bars per RTH day
const BARS_1M = BARS_1D * 22;
const BARS_3M = BARS_1M * 3;
const BARS_1Y = BARS_1D * 252;

const yearStartIdx = {{}};
let prevYear = 0;
for (let i = 0; i < candleData.length; i++) {{
  const d = new Date(candleData[i].time * 1000);
  const y = d.getUTCFullYear();
  if (y !== prevYear) {{ yearStartIdx[y] = i; prevYear = y; }}
}}

function setLR(from, to) {{
  from = Math.max(0, Math.min(from, totalBars - 1));
  to = Math.max(from + 10, Math.min(to, totalBars + 50));
  chart.timeScale().setVisibleLogicalRange({{ from, to }});
}}
function getLR() {{ return chart.timeScale().getVisibleLogicalRange(); }}
function panBy(n) {{ const r = getLR(); if (r) setLR(r.from + n, r.to + n); }}
function zoomBy(f) {{
  const r = getLR(); if (!r) return;
  const mid = (r.from + r.to) / 2, half = (r.to - r.from) / 2 * f;
  setLR(mid - half, mid + half);
}}
function showNEnd(bars, btnId) {{
  const r = getLR(); const end = r ? Math.round(r.to) : totalBars + 20;
  setLR(end - bars, end);
  ['btn1M','btn3M','btn1Y','btnAll'].forEach(id =>
    document.getElementById(id).classList.toggle('active', id === btnId));
}}

document.getElementById('yearSelect').addEventListener('change', (e) => {{
  const v = e.target.value;
  if (v === 'all') {{ chart.timeScale().fitContent(); return; }}
  const y = parseInt(v);
  setLR(yearStartIdx[y] || 0, yearStartIdx[y + 1] || totalBars);
}});
document.getElementById('btnZoomIn').addEventListener('click', () => zoomBy(0.5));
document.getElementById('btnZoomOut').addEventListener('click', () => zoomBy(2.0));
document.getElementById('btnLeft').addEventListener('click', () => {{ const r = getLR(); if (r) panBy(-Math.round(r.to - r.from)); }});
document.getElementById('btnRight').addEventListener('click', () => {{ const r = getLR(); if (r) panBy(Math.round(r.to - r.from)); }});
document.getElementById('btn1M').addEventListener('click', () => showNEnd(BARS_1M, 'btn1M'));
document.getElementById('btn3M').addEventListener('click', () => showNEnd(BARS_3M, 'btn3M'));
document.getElementById('btn1Y').addEventListener('click', () => showNEnd(BARS_1Y, 'btn1Y'));
document.getElementById('btnAll').addEventListener('click', () => {{
  chart.timeScale().fitContent();
  ['btn1M','btn3M','btn1Y','btnAll'].forEach(id =>
    document.getElementById(id).classList.toggle('active', id === 'btnAll'));
}});
document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'SELECT') return;
  const r = getLR(); if (!r) return;
  const span = r.to - r.from;
  if (e.key === 'ArrowLeft') panBy(-Math.round(span));
  else if (e.key === 'ArrowRight') panBy(Math.round(span));
  else if (e.key === '+' || e.key === '=') zoomBy(0.5);
  else if (e.key === '-') zoomBy(2.0);
  else if (e.key === 'Escape') document.getElementById('helpOverlay').classList.remove('show');
}});
document.getElementById('btnHelp').addEventListener('click', () => {{
  document.getElementById('helpOverlay').classList.toggle('show');
}});

// Default: show last 3 months
setLR(totalBars - BARS_3M, totalBars + 20);
</script>
</body>
</html>"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Chart saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Touch & Turn trade chart")
    parser.add_argument("--config", default="touch_turn/config.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    stem = Path(args.config).stem.removeprefix("config_").strip("_") or "touch_turn"
    results_dir = Path(f"results/{stem}")
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        args.out = str(results_dir / "trade_chart.html")

    data_path = config["backtest"]["data_file"]
    data_source = config.get("backtest", {}).get("data_source", "databento")
    start = date.fromisoformat(args.start or config["backtest"]["start_date"])
    end = date.fromisoformat(args.end or config["backtest"]["end_date"])

    symbol = config.get("bot", {}).get("symbol", "ES")
    print(f"Loading {symbol} bars from {data_path}...")
    if data_source == "equity":
        loader = EquityLoader(data_path)
    else:
        loader = DatabentoLoader(data_path)

    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} days")

    print("Running backtest...")
    engine = TouchTurnBacktestEngine(config)
    report = engine.run(iter(days))

    trades = report.trades
    n = len(trades)
    if n == 0:
        print("No trades — nothing to chart.")
        return

    import math
    total = sum(t.pnl_usd for t in trades)
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gp = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    wr = len(wins) / n
    avg = total / n
    std = math.sqrt(sum((t.pnl_usd - avg)**2 for t in trades) / (n-1)) if n > 1 else 1
    sharpe = (avg / std) * math.sqrt(252) if std > 0 else 0

    summary = {
        "trades": n, "total_pnl_usd": total, "win_rate": wr,
        "sharpe": sharpe, "profit_factor": gp / gl if gl > 0 else 99,
    }

    print(f"  {n} trades, WR {wr*100:.1f}%, Sharpe {sharpe:.2f}, PnL ${total:,.0f}")

    print("Building chart...")
    candles, level_lines, markers, details, equity = process_data(days, trades, config)
    print(f"  {len(candles)} candles, {len(level_lines)} level segments, {len(markers)} markers")

    generate_html(candles, level_lines, markers, details, equity,
                  summary, config, args.out)


if __name__ == "__main__":
    main()
