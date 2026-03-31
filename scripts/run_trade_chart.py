"""
Interactive trade chart with intraday candles, ORB levels, and P&L.

Uses TradingView's lightweight-charts for smooth zoom/pan.

Usage:
  python -m scripts.run_trade_chart
  python -m scripts.run_trade_chart --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_8am.backtest.data_loader import DatabentoLoader
from src_8am.backtest.options_engine import OrbOptionsBacktestEngine, OptionsBacktestResult
from src_8am.backtest.options_report import generate_options_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_5min(bars: List[Dict]) -> List[Dict]:
    """Aggregate 1-min bars into 5-min OHLCV."""
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
                "volume": b["volume"],
            }
        else:
            bucket["high"] = max(bucket["high"], b["high"])
            bucket["low"] = min(bucket["low"], b["low"])
            bucket["close"] = b["close"]
            bucket["volume"] += b["volume"]
    if bucket is not None:
        out.append(bucket)
    return out


def _ts_to_unix(dt: datetime) -> int:
    """Convert datetime to Unix timestamp, displaying ET wall-clock time.

    lightweight-charts renders UTC — offset so the chart shows ET hours.
    """
    ts = int(dt.timestamp())
    offset = dt.utcoffset()
    if offset is not None:
        ts += int(offset.total_seconds())
    return ts


def _snap_5min(dt: datetime) -> datetime:
    """Snap datetime to start of its 5-min bucket (must match candle times)."""
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def _time_to_unix(d: date, t: time) -> int:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    dt = datetime.combine(d, t, tzinfo=ET)
    return _ts_to_unix(dt)


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def process_data(days, results):
    """Process raw days + backtest results into chart-ready JSON data."""

    # Index trades by session_date
    trades_by_date: Dict[date, OptionsBacktestResult] = {}
    for r in results:
        if r.status == "closed" and r.realized_pnl_usd is not None:
            trades_by_date[r.session_date] = r

    # Sort days by date, pick best volume per date
    by_date: Dict[date, Any] = {}
    for d in days:
        sd = d.session_date
        if sd not in by_date or len(d.bars) > len(by_date[sd].bars):
            by_date[sd] = d

    candles = []
    orb_high_pts: List[Dict] = []   # {time, value} per candle for horizontal lines
    orb_low_pts: List[Dict] = []
    orb_mid_pts: List[Dict] = []
    markers = []    # {time, position, color, shape, text}
    trade_details = {}  # unix_ts -> detail dict

    equity = []
    running_pnl = 0.0

    # ORB window: highlight candles within the opening range period
    orb_window_start = time(8, 0)
    orb_window_end = time(8, 15)
    # ORB lines display from end of ORB window to market close
    orb_line_start = time(8, 15)
    orb_line_end = time(16, 0)

    for sd in sorted(by_date.keys()):
        day = by_date[sd]
        bars_5m = _aggregate_5min(day.bars)
        trade = trades_by_date.get(sd)
        is_loss = trade is not None and (trade.realized_pnl_usd or 0.0) < 0

        # Use first bar of the day as the equity timestamp anchor
        first_bar_ts: Optional[int] = None

        for b in bars_5m:
            bar_time: time = b["time"].time()
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
            # Tint ORB-window candles blue
            if orb_window_start <= bar_time < orb_window_end:
                candle["color"] = "#1565c0"
                candle["borderColor"] = "#42a5f5"
                candle["wickColor"] = "#42a5f5"
            elif is_loss:
                candle["color"] = "#5c2020"
                candle["borderColor"] = "#ef5350"
                candle["wickColor"] = "#ef5350"
            candles.append(candle)

            # Emit ORB level data at each candle within display range
            if trade and orb_line_start <= bar_time <= orb_line_end:
                orb_high_pts.append({"time": ts, "value": round(trade.orb_high, 2)})
                orb_low_pts.append({"time": ts, "value": round(trade.orb_low, 2)})
                orb_mid_pts.append({"time": ts, "value": round(trade.orb_midline, 2)})

        # Update running PnL on trade days
        if trade is not None:
            pnl = trade.realized_pnl_usd or 0.0
            running_pnl += pnl
            is_win = pnl >= 0

            # Entry marker — snap to 5-min bucket so it aligns with candle time
            if trade.entry_time:
                entry_ts = _ts_to_unix(_snap_5min(trade.entry_time))
                bias_label = "Bull (put spread)" if trade.bias == "long" else "Bear (call spread)"
                markers.append({
                    "time": entry_ts,
                    "position": "belowBar" if trade.bias == "long" else "aboveBar",
                    "color": "#26a69a" if is_win else "#ef5350",
                    "shape": "arrowUp" if trade.bias == "long" else "arrowDown",
                    "text": f"{'W' if is_win else 'L'} ${pnl:+.0f}",
                })

                # Trade detail for tooltip lookup
                trade_details[str(entry_ts)] = {
                    "date": sd.isoformat(),
                    "bias": bias_label,
                    "spread": trade.spread_type or "",
                    "short_strike": trade.short_strike,
                    "long_strike": trade.long_strike,
                    "delta": round(trade.short_delta_at_entry or 0, 4),
                    "credit": round(trade.credit_received or 0, 3),
                    "debit": round(trade.debit_paid or 0, 3),
                    "pnl": round(pnl, 2),
                    "exit_reason": trade.exit_reason or "",
                    "es_entry": round(trade.underlying_at_entry or 0, 2),
                    "es_exit": round(trade.underlying_at_exit or 0, 2),
                    "orb_high": round(trade.orb_high, 2),
                    "orb_low": round(trade.orb_low, 2),
                    "orb_range": round(trade.orb_range_pts, 2),
                    "entry_time": trade.entry_time.strftime("%H:%M") if trade.entry_time else "",
                    "exit_time": trade.exit_time.strftime("%H:%M") if trade.exit_time else "",
                }

            # Exit marker — snap to 5-min bucket, show P&L
            if trade.exit_time:
                exit_ts = _ts_to_unix(_snap_5min(trade.exit_time))
                markers.append({
                    "time": exit_ts,
                    "position": "aboveBar" if trade.bias == "long" else "belowBar",
                    "color": "#26a69a" if is_win else "#ef5350",
                    "shape": "circle",
                    "text": f"{trade.exit_reason or ''} ${pnl:+.0f}",
                })

        # Equity curve — emit a point every day (carry forward on non-trade days)
        if first_bar_ts is not None:
            equity.append({
                "time": first_bar_ts,
                "value": round(running_pnl, 2),
            })

    return candles, orb_high_pts, orb_low_pts, orb_mid_pts, markers, trade_details, equity


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(candles, orb_high_pts, orb_low_pts, orb_mid_pts, markers, trade_details, equity, report, config, out_path):
    qty_label = config.get('options', {}).get('quantity', config.get('bot', {}).get('quantity', 1))
    mult_label = config.get('options', {}).get('contract_multiplier', 50)
    params_str = (
        f"delta={config['options']['target_short_delta']}  "
        f"width={config['options']['spread_width_pts']}  "
        f"tp={config['options']['tp_pct_of_credit']}  "
        f"min_range={config['strategy']['orb_range_filter']['min_pts']}"
    )
    title = (
        f"ORB Options Backtest — "
        f"{report['trades_placed']} trades, "
        f"WR {report['win_rate']*100:.1f}%, "
        f"Sharpe {report['sharpe_annualised']:.2f}, "
        f"PnL ${report['total_pnl_usd']:,.0f} | {params_str}"
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
  #helpBox kbd {{ background: #2a2e39; border: 1px solid #363a45; border-radius: 3px; padding: 1px 5px; font-size: 11px; }}
  #helpBox .legend-item {{ display: flex; align-items: center; gap: 8px; }}
  #helpBox .swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 2px; }}
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
  .legend .orb-h {{ color: #26a69a; }}
  .legend .orb-l {{ color: #ef5350; }}
  .legend .orb-m {{ color: #ffb74d; }}
</style>
</head>
<body>
<div id="header">
  <h1>{title}</h1>
  <div id="nav">
    <select id="yearSelect">
      <option value="all">All Years</option>
      <option value="2021">2021</option>
      <option value="2022">2022</option>
      <option value="2023">2023</option>
      <option value="2024">2024</option>
      <option value="2025">2025</option>
      <option value="2026">2026</option>
    </select>
    <button id="btnZoomOut" title="Zoom out">−</button>
    <button id="btnZoomIn" title="Zoom in">+</button>
    <button id="btnLeft" title="Pan left">◀</button>
    <button id="btnRight" title="Pan right">▶</button>
    <button id="btn1M" title="1 Month view">1M</button>
    <button id="btn3M" title="3 Month view">3M</button>
    <button id="btn1Y" title="1 Year view">1Y</button>
    <button id="btnAll" title="All data">All</button>
    <button id="btnHelp" title="Keyboard shortcuts & chart legend" style="margin-left:8px; font-weight:700;">?</button>
  </div>
  <span style="font-size:11px; color:#787b86; margin-left:auto;">P&amp;L: {qty_label} spread(s) x ${mult_label} multiplier</span>
</div>
<div id="candle-chart" style="position:relative;">
  <div class="legend">
    <span style="color:#42a5f5;">■ ORB Window</span>
    <span class="orb-h">— ORB High</span>
    <span class="orb-l">— ORB Low</span>
    <span class="orb-m">— ORB Mid</span>
    <span>▲▼ Trade Entry</span>
    <span>● Exit</span>
    <span style="color:#1976d2;">— P&L</span>
    <span style="color:#5c2020;">■ <span style="color:#ef5350;">Loss day</span></span>
  </div>
</div>
<div id="tooltip"></div>

<!-- Help overlay -->
<div id="helpOverlay">
<div id="helpBox">
  <h2>Chart Navigation</h2>
  <h3>Buttons</h3>
  <b>Year dropdown</b> — jump to a specific year<br>
  <b>1M / 3M / 1Y</b> — zoom to that period <i>ending at current view's right edge</i><br>
  <b>All</b> — fit all data<br>
  <b>&#9664; &#9654;</b> — scroll left/right by <b>one full screen</b><br>
  <b>+ / &minus;</b> — zoom in/out centered on current view

  <h3>Keyboard</h3>
  <kbd>&larr;</kbd> <kbd>&rarr;</kbd> pan left/right one screen &nbsp;
  <kbd>+</kbd> <kbd>&minus;</kbd> zoom in/out<br>
  <kbd>Esc</kbd> close this help

  <h3>Mouse</h3>
  Drag to pan &bull; Scroll wheel to zoom &bull; Hover trade markers for details

  <h3>Chart Elements</h3>
  <div class="legend-item"><span class="swatch" style="background:#1565c0;"></span> ORB window candles (08:00 &ndash; 08:15 ET)</div>
  <div class="legend-item"><span class="swatch" style="background:#5c2020; border:1px solid #ef5350;"></span> Losing trade day candles</div>
  <div class="legend-item"><span style="color:#26a69a;">&#9650;</span><span style="color:#ef5350;">&#9660;</span> Entry markers (green=win, red=loss)</div>
  <div class="legend-item"><span style="color:#26a69a;">&#9679;</span><span style="color:#ef5350;">&#9679;</span> Exit markers (green=win, red=loss)</div>
  <div class="legend-item"><span style="color:#26a69a;">&mdash;</span> ORB High &nbsp; <span style="color:#ef5350;">&mdash;</span> ORB Low &nbsp; <span style="color:#ffb74d;">&mdash;</span> ORB Mid</div>
  <div class="legend-item"><span style="color:#1976d2;">&mdash;</span> Cumulative P&amp;L (left axis, {qty_label} spread(s) &times; ${mult_label})</div>

  <br><button onclick="document.getElementById('helpOverlay').classList.remove('show')" style="background:#2a2e39; color:#d1d4dc; border:1px solid #363a45; border-radius:4px; padding:6px 16px; cursor:pointer;">Close</button>
</div>
</div>

<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
const candleData = {json.dumps(candles)};
const orbHighData = {json.dumps(orb_high_pts)};
const orbLowData = {json.dumps(orb_low_pts)};
const orbMidData = {json.dumps(orb_mid_pts)};
const markerData = {json.dumps(markers)};
const tradeDetails = {json.dumps(trade_details)};
const equityData = {json.dumps(equity)};

// ---- Main candle chart ----
const chartEl = document.getElementById('candle-chart');
const chart = LightweightCharts.createChart(chartEl, {{
  layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
  grid: {{
    vertLines: {{ color: '#1e222d' }},
    horzLines: {{ color: '#1e222d' }},
  }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  timeScale: {{
    timeVisible: true,
    secondsVisible: false,
    borderColor: '#2a2e39',
    minBarSpacing: 0.001,
  }},
  rightPriceScale: {{ borderColor: '#2a2e39' }},
  leftPriceScale: {{
    visible: true,
    borderColor: '#2a2e39',
    scaleMargins: {{ top: 0.7, bottom: 0.02 }},
  }},
}});

const candleSeries = chart.addCandlestickSeries({{
  upColor: '#26a69a', downColor: '#ef5350',
  borderUpColor: '#26a69a', borderDownColor: '#ef5350',
  wickUpColor: '#26a69a', wickDownColor: '#ef5350',
}});
candleSeries.setData(candleData);

// ---- ORB level lines (as separate line series) ----
const orbHighSeries = chart.addLineSeries({{
  color: 'rgba(38,166,154,0.6)', lineWidth: 1, lineStyle: 0,
  lastValueVisible: false, priceLineVisible: false,
  crosshairMarkerVisible: false,
}});
const orbLowSeries = chart.addLineSeries({{
  color: 'rgba(239,83,80,0.6)', lineWidth: 1, lineStyle: 0,
  lastValueVisible: false, priceLineVisible: false,
  crosshairMarkerVisible: false,
}});
const orbMidSeries = chart.addLineSeries({{
  color: 'rgba(255,183,77,0.5)', lineWidth: 1, lineStyle: 2,
  lastValueVisible: false, priceLineVisible: false,
  crosshairMarkerVisible: false,
}});

// ORB data is pre-built per candle timestamp (horizontal within each day)
orbHighSeries.setData(orbHighData);
orbLowSeries.setData(orbLowData);
orbMidSeries.setData(orbMidData);

// ---- Trade markers ----
candleSeries.setMarkers(
  markerData.sort((a, b) => a.time - b.time)
);

// ---- P&L equity overlay on left price scale ----
const equitySeries = chart.addAreaSeries({{
  topColor: 'rgba(25,118,210,0.15)',
  bottomColor: 'rgba(25,118,210,0.02)',
  lineColor: '#1976d2',
  lineWidth: 2,
  lastValueVisible: true,
  priceLineVisible: false,
  priceScaleId: 'left',
}});
equitySeries.setData(equityData);

// ---- Tooltip on crosshair move ----
const tooltip = document.getElementById('tooltip');
chart.subscribeCrosshairMove((param) => {{
  if (!param.time || !param.seriesData) {{
    tooltip.style.display = 'none';
    return;
  }}
  // Find nearest trade marker
  const ts = String(param.time);
  const detail = tradeDetails[ts];
  if (!detail) {{
    // Check nearby timestamps (within 5 min = 300s)
    let found = null;
    for (const [key, val] of Object.entries(tradeDetails)) {{
      if (Math.abs(parseInt(key) - param.time) < 300) {{
        found = val;
        break;
      }}
    }}
    if (!found) {{
      tooltip.style.display = 'none';
      return;
    }}
    showTooltip(found);
  }} else {{
    showTooltip(detail);
  }}
}});

function showTooltip(d) {{
  const pnlClass = d.pnl >= 0 ? 'win' : 'loss';
  tooltip.innerHTML = `
    <div><span class="label">Date:</span> <span class="val">${{d.date}}</span></div>
    <div><span class="label">Bias:</span> <span class="val">${{d.bias}}</span></div>
    <div><span class="label">Spread:</span> <span class="val">${{d.spread}}</span></div>
    <div><span class="label">Strikes:</span> <span class="val">${{d.short_strike}} / ${{d.long_strike}}</span></div>
    <div><span class="label">Delta:</span> <span class="val">${{d.delta}}</span></div>
    <div><span class="label">Credit:</span> <span class="val">$${{d.credit}}</span></div>
    <div><span class="label">Debit:</span> <span class="val">$${{d.debit}}</span></div>
    <div><span class="label">P&L:</span> <span class="${{pnlClass}}">$${{d.pnl.toFixed(0)}}</span></div>
    <div><span class="label">Exit:</span> <span class="val">${{d.exit_reason}}</span></div>
    <div><span class="label">Entry:</span> <span class="val">${{d.entry_time}} @ ${{d.es_entry}}</span></div>
    <div><span class="label">Exit:</span> <span class="val">${{d.exit_time}} @ ${{d.es_exit}}</span></div>
    <div><span class="label">ORB:</span> <span class="val">${{d.orb_high}} / ${{d.orb_low}} (range ${{d.orb_range}})</span></div>
  `;
  tooltip.style.display = 'block';
}}

// ---- Resize handling ----
const ro = new ResizeObserver(() => {{
  chart.applyOptions({{ width: chartEl.clientWidth, height: chartEl.clientHeight }});
}});
ro.observe(chartEl);

// ---- Navigation helpers (logical range = bar indices) ----
const totalBars = candleData.length;
// Approx bars per time period (5min bars, ~228/day, ~1140/week)
const BARS_1D = 228;
const BARS_1W = BARS_1D * 5;
const BARS_1M = BARS_1D * 22;
const BARS_3M = BARS_1M * 3;
const BARS_1Y = BARS_1D * 252;

// Build time->bar index lookup for year navigation
const yearStartIdx = {{}};
let prevYear = 0;
for (let i = 0; i < candleData.length; i++) {{
  const d = new Date(candleData[i].time * 1000);
  const y = d.getUTCFullYear();
  if (y !== prevYear) {{
    yearStartIdx[y] = i;
    prevYear = y;
  }}
}}

function setLogicalRange(from, to) {{
  from = Math.max(0, Math.min(from, totalBars - 1));
  to = Math.max(from + 10, Math.min(to, totalBars + 50));
  chart.timeScale().setVisibleLogicalRange({{ from, to }});
}}

function getLogicalRange() {{
  return chart.timeScale().getVisibleLogicalRange();
}}

function showLastN(bars) {{
  setLogicalRange(totalBars - bars, totalBars + 20);
}}

// Show N bars ending at current view's right edge (for period buttons after year select)
function showNFromViewEnd(bars, btnId) {{
  const r = getLogicalRange();
  const end = r ? Math.round(r.to) : totalBars + 20;
  setLogicalRange(end - bars, end);
  // Highlight active period button
  ['btn1M','btn3M','btn1Y','btnAll'].forEach(id =>
    document.getElementById(id).classList.toggle('active', id === btnId));
}}

function panByBars(n) {{
  const r = getLogicalRange();
  if (r) setLogicalRange(r.from + n, r.to + n);
}}

function zoomBy(factor) {{
  const r = getLogicalRange();
  if (!r) return;
  const mid = (r.from + r.to) / 2;
  const half = (r.to - r.from) / 2 * factor;
  setLogicalRange(mid - half, mid + half);
}}

// Year dropdown
document.getElementById('yearSelect').addEventListener('change', (e) => {{
  const v = e.target.value;
  if (v === 'all') {{ chart.timeScale().fitContent(); return; }}
  const y = parseInt(v);
  const startIdx = yearStartIdx[y] || 0;
  const endIdx = yearStartIdx[y + 1] || totalBars;
  setLogicalRange(startIdx, endIdx);
}});

// Zoom buttons
document.getElementById('btnZoomIn').addEventListener('click', () => zoomBy(0.5));
document.getElementById('btnZoomOut').addEventListener('click', () => zoomBy(2.0));

// Pan buttons (scroll by one full screen)
document.getElementById('btnLeft').addEventListener('click', () => {{
  const r = getLogicalRange();
  if (r) panByBars(-Math.round(r.to - r.from));
}});
document.getElementById('btnRight').addEventListener('click', () => {{
  const r = getLogicalRange();
  if (r) panByBars(Math.round(r.to - r.from));
}});

// Period buttons — zoom relative to current view's right edge
document.getElementById('btn1M').addEventListener('click', () => showNFromViewEnd(BARS_1M, 'btn1M'));
document.getElementById('btn3M').addEventListener('click', () => showNFromViewEnd(BARS_3M, 'btn3M'));
document.getElementById('btn1Y').addEventListener('click', () => showNFromViewEnd(BARS_1Y, 'btn1Y'));
document.getElementById('btnAll').addEventListener('click', () => {{
  chart.timeScale().fitContent();
  ['btn1M','btn3M','btn1Y','btnAll'].forEach(id =>
    document.getElementById(id).classList.toggle('active', id === 'btnAll'));
}});

// Keyboard nav: arrow keys to pan, +/- to zoom
document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'SELECT') return;
  const r = getLogicalRange();
  if (!r) return;
  const span = r.to - r.from;
  if (e.key === 'ArrowLeft') panByBars(-Math.round(span));
  else if (e.key === 'ArrowRight') panByBars(Math.round(span));
  else if (e.key === '+' || e.key === '=') zoomBy(0.5);
  else if (e.key === '-') zoomBy(2.0);
  else if (e.key === 'Escape') document.getElementById('helpOverlay').classList.remove('show');
}});

// Help button
document.getElementById('btnHelp').addEventListener('click', () => {{
  document.getElementById('helpOverlay').classList.toggle('show');
}});

// Default: show last 3 months
showLastN(BARS_3M);
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

def _strategy_name_from_config(config_path: str) -> str:
    """Derive strategy subdirectory name from config filename."""
    stem = Path(config_path).stem
    name = stem.removeprefix("config_").strip("_")
    return name or "default"


def main():
    parser = argparse.ArgumentParser(description="Trade chart with candles + ORB levels")
    parser.add_argument("--config", default="config/config_8am_es_options.yaml")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="Output HTML path (default: results/<strategy>/trade_chart.html)")
    args = parser.parse_args()

    strategy = _strategy_name_from_config(args.config)
    results_dir = Path(f"results/{strategy}")
    if args.out is None:
        args.out = str(results_dir / "trade_chart.html")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply optimized params if available
    opt_path = results_dir / "optimized_params.yaml"
    if opt_path.exists():
        with open(opt_path) as f:
            opt = yaml.safe_load(f)
        if "options" in opt:
            config["options"].update(opt["options"])
        if "strategy" in opt:
            for k, v in opt["strategy"].items():
                if isinstance(v, dict):
                    config["strategy"].setdefault(k, {}).update(v)
                else:
                    config["strategy"][k] = v
        if "risk" in opt:
            config.setdefault("risk", {}).update(opt["risk"])
        # Scale risk limit to accommodate optimized position size
        opt_qty = config["options"].get("quantity", 1)
        opt_width = config["options"].get("spread_width_pts", 5)
        opt_mult = config["options"].get("contract_multiplier", 50)
        needed = opt_width * opt_mult * opt_qty
        cur_limit = config.get("risk", {}).get("max_loss_per_trade_usd", 0)
        if needed > cur_limit:
            config.setdefault("risk", {})["max_loss_per_trade_usd"] = needed
            config["risk"]["max_daily_loss_usd"] = max(
                config["risk"].get("max_daily_loss_usd", 0), needed
            )
        print(f"Applied optimized params from {opt_path}")

    data_file = config["backtest"]["data_file"]
    # Default to full data range (ignore config's backtest window)
    start = (date.fromisoformat(args.start) if args.start
             else date(2021, 3, 1))
    end = (date.fromisoformat(args.end) if args.end
           else date(2026, 2, 26))

    print(f"Loading bars from {data_file}...")
    loader = DatabentoLoader(data_file)
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} day-symbol chunks")

    print("Running backtest...")
    engine = OrbOptionsBacktestEngine(config=config)
    results = engine.run_on_days(days)
    report = generate_options_report(results)

    print(f"\n  Trades: {report.get('trades_placed', 0)}  "
          f"WR: {report.get('win_rate', 0)*100:.1f}%  "
          f"Sharpe: {report.get('sharpe_annualised', 0):.2f}  "
          f"PnL: ${report.get('total_pnl_usd', 0):,.0f}")

    print("Building chart...")
    candles, orb_high_pts, orb_low_pts, orb_mid_pts, markers, trade_details, equity = process_data(days, results)
    print(f"  {len(candles)} candles, {len(orb_high_pts)} ORB pts, "
          f"{len(markers)} markers, {len(equity)} equity points")

    generate_html(candles, orb_high_pts, orb_low_pts, orb_mid_pts, markers, trade_details, equity, report, config, args.out)


if __name__ == "__main__":
    main()
