"""
Run backtest with current config and generate an equity + trade chart.

Usage:
  python -m scripts.run_backtest_chart
  python -m scripts.run_backtest_chart --config config/config_8am_es_options.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_8am.backtest.data_loader import DatabentoLoader
from src_8am.backtest.spy_loader import SpyLoader
from src_8am.backtest.options_engine import OrbOptionsBacktestEngine
from src_8am.backtest.options_report import generate_options_report


def _strategy_name_from_config(config_path: str) -> str:
    """Derive strategy subdirectory name from config filename."""
    stem = Path(config_path).stem
    name = stem.removeprefix("config_").strip("_")
    return name or "default"


def main():
    parser = argparse.ArgumentParser(description="Backtest + chart")
    parser.add_argument("--config", default="config/config_8am_es_options.yaml")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--out", default=None,
                        help="Output chart path (default: results/<strategy>/backtest_chart.html)")
    args = parser.parse_args()

    strategy = _strategy_name_from_config(args.config)
    if args.out is None:
        results_dir = Path(f"results/{strategy}")
        results_dir.mkdir(parents=True, exist_ok=True)
        args.out = str(results_dir / "backtest_chart.html")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_file = config["backtest"]["data_file"]
    start = (date.fromisoformat(args.start) if args.start
             else date.fromisoformat(config["backtest"].get("start_date", "2022-04-25")))
    end = (date.fromisoformat(args.end) if args.end
           else date.fromisoformat(config["backtest"].get("end_date", "2024-12-31")))

    data_source = config.get("backtest", {}).get("data_source", "databento")
    price_scale = config.get("backtest", {}).get("price_scale", 1.0)
    print(f"Loading bars from {data_file} (source={data_source})...")
    if data_source == "spy":
        equity_symbol = config.get("bot", {}).get("symbol", "SPX")
        loader = SpyLoader(data_file, price_scale=price_scale, symbol=equity_symbol)
    else:
        loader = DatabentoLoader(data_file)
    days = list(loader.iter_days(start_date=start, end_date=end))
    print(f"  {len(days)} day-symbol chunks")

    print("Running backtest...")
    engine = OrbOptionsBacktestEngine(config=config)
    results = engine.run_on_days(days)
    report = generate_options_report(results)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Backtest: {start} -> {end}")
    print(f"  Params: delta={config['options']['target_short_delta']}  "
          f"width={config['options']['spread_width_pts']}  "
          f"tp={config['options']['tp_pct_of_credit']}  "
          f"min_range={config['strategy']['orb_range_filter']['min_pts']}")
    print(f"{'='*60}")
    n_trades = report.get('trades_placed', 0)
    if n_trades == 0:
        print(f"  Trades: 0  (no trades passed filters)")
        print(f"{'='*60}\n")
        return
    print(f"  Trades: {n_trades}  "
          f"WR: {report.get('win_rate', 0)*100:.1f}%  "
          f"Sharpe: {report.get('sharpe_annualised', 0):.2f}")
    print(f"  Total PnL: ${report.get('total_pnl_usd', 0):,.0f}  "
          f"Avg: ${report.get('avg_pnl_per_trade_usd', 0):.2f}  "
          f"PF: {report.get('profit_factor', 0)}")
    print(f"  Max DD: ${report.get('max_drawdown_usd', 0):,.0f}")
    print(f"  Exit reasons: {report.get('by_exit_reason', {})}")
    print(f"{'='*60}\n")

    # Build chart
    build_chart(results, report, config, args.out)


def build_chart(results, report, config, out_path):
    """Generate interactive Plotly chart with equity curve + trade markers."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed. Run: pip install plotly")
        return

    # Filter to closed trades only
    trades = [r for r in results if r.status == "closed" and r.realized_pnl_usd is not None]
    if not trades:
        print("No trades to chart.")
        return

    # Sort by date
    trades.sort(key=lambda t: t.session_date)

    dates = [t.session_date for t in trades]
    pnls = [t.realized_pnl_usd for t in trades]

    # Cumulative equity
    equity = []
    running = 0.0
    for pnl in pnls:
        running += pnl
        equity.append(running)

    # Color by win/loss
    colors = ["#26a69a" if p >= 0 else "#ef5350" for p in pnls]

    # Hover text
    hovers = []
    for t in trades:
        h = (
            f"Date: {t.session_date}<br>"
            f"Bias: {t.bias}<br>"
            f"Spread: {t.spread_type}<br>"
            f"Short: {t.short_strike}  Long: {t.long_strike}<br>"
            f"Delta: {t.short_delta_at_entry:.4f}<br>"
            f"Credit: ${t.credit_received:.2f}<br>"
            f"Debit: ${t.debit_paid:.2f}<br>"
            f"P&L: ${t.realized_pnl_usd:+.0f}<br>"
            f"Exit: {t.exit_reason}<br>"
            f"ES entry: {t.underlying_at_entry:.2f}"
        )
        hovers.append(h)

    # Create subplots: equity curve + per-trade P&L bars
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=["Cumulative Equity ($)", "Per-Trade P&L ($)"],
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=dates, y=equity,
            mode="lines",
            name="Equity",
            line=dict(color="#1976d2", width=2),
            hovertext=hovers,
            hoverinfo="text",
        ),
        row=1, col=1,
    )

    # Drawdown shading
    peak = 0.0
    dd_values = []
    for eq in equity:
        peak = max(peak, eq)
        dd_values.append(eq - peak)

    fig.add_trace(
        go.Scatter(
            x=dates, y=dd_values,
            mode="lines",
            name="Drawdown",
            line=dict(color="#ef5350", width=1),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.15)",
        ),
        row=1, col=1,
    )

    # Per-trade P&L bars
    fig.add_trace(
        go.Bar(
            x=dates, y=pnls,
            name="Trade P&L",
            marker_color=colors,
            hovertext=hovers,
            hoverinfo="text",
        ),
        row=2, col=1,
    )

    # Layout
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
        f"PnL ${report['total_pnl_usd']:,.0f}<br>"
        f"<sub>{params_str}</sub>"
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=700,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="$", row=2, col=1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    print(f"Chart saved -> {out_path}")


if __name__ == "__main__":
    main()
