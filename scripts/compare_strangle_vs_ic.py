"""Compare WF-CV winners: open strangle vs. open IC.

Loads each strategy's WINNER trade log and emits a markdown table covering
the spec's acceptance criteria + a per-fold breakdown.

Usage:
  python scripts/compare_strangle_vs_ic.py \\
      --ic results/spx_0dte_open_ic/<RUN_DIR>/trades.csv \\
      --strangle results/spx_0dte_open_strangle/<RUN_DIR>/trades.csv \\
      --ic-meta data/ic_sl_dollars_by_day.meta.json

Column-name note: the IC trade log uses ``session_date`` and
``realized_pnl_usd``; the strangle log is expected to match the same schema.
The helper ``_per_day_pnl`` resolves whichever date / pnl column names are
present, so both formats are accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Match the IC findings doc (docs/spx_0dte_open_ic_findings.md).
IC_BASELINE: Dict[str, float] = {
    "trades": 172,
    "win_rate": 0.453,
    "total_pnl": 15014.0,
    "sharpe": 4.53,
    "worst_day": -826.0,
    "best_day": 793.0,
    "mean_win": 340.0,
    "mean_loss": -123.0,
}


def _load_trades(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _per_day_pnl(df: pd.DataFrame) -> pd.Series:
    date_col = next(
        c for c in ("trade_date", "session_date", "entry_date", "date") if c in df.columns
    )
    pnl_col = next(c for c in ("pnl", "realized_pnl_usd", "realized_pnl") if c in df.columns)
    return df.groupby(date_col)[pnl_col].sum()


def _metrics(df: pd.DataFrame, label: str) -> Dict[str, float]:
    daily = _per_day_pnl(df)
    daily_arr = daily.to_numpy(dtype=float)
    total = float(daily_arr.sum())
    wins = daily_arr[daily_arr > 0]
    losses = daily_arr[daily_arr < 0]
    wr = float(len(wins) / len(daily_arr)) if len(daily_arr) else 0.0
    sharpe = (
        float(daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0.0
    )
    return {
        "label": label,
        "trades": int(len(df)),
        "trade_days": int(len(daily_arr)),
        "win_rate": wr,
        "total_pnl": total,
        "sharpe": sharpe,
        "worst_day": float(daily_arr.min()) if len(daily_arr) else 0.0,
        "best_day": float(daily_arr.max()) if len(daily_arr) else 0.0,
        "mean_win": float(wins.mean()) if len(wins) else 0.0,
        "mean_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ic",
        required=False,
        help="IC trade log (optional — defaults to baseline numbers)",
    )
    ap.add_argument("--strangle", required=True)
    ap.add_argument("--ic-meta", required=False)
    args = ap.parse_args()

    if args.ic:
        ic_df = _load_trades(args.ic)
        ic = _metrics(ic_df, "Open IC")
    else:
        ic = {
            "label": "Open IC (findings baseline)",
            **IC_BASELINE,
            "trade_days": IC_BASELINE["trades"],
        }

    str_df = _load_trades(args.strangle)
    st = _metrics(str_df, "Open Strangle")

    # Acceptance gate (per spec §1)
    passes = (
        st["sharpe"] >= IC_BASELINE["sharpe"]
        and st["worst_day"] >= IC_BASELINE["worst_day"]
        and st["total_pnl"] > IC_BASELINE["total_pnl"]
    )

    print()
    print("| Metric | IC | Strangle | Δ | Strangle better? |")
    print("|---|---|---|---|---|")
    for k, label, sign, fmt in [
        ("trade_days", "Trade days", +1, "{:.0f}"),
        ("win_rate", "Win rate", +1, "{:.1%}"),
        ("total_pnl", "Total PnL", +1, "${:+,.0f}"),
        ("sharpe", "Sharpe", +1, "{:.2f}"),
        ("worst_day", "Worst day", +1, "${:+,.0f}"),
        ("best_day", "Best day", +1, "${:+,.0f}"),
        ("mean_win", "Mean win", +1, "${:+,.0f}"),
        ("mean_loss", "Mean loss", +1, "${:+,.0f}"),
    ]:
        ic_v = ic.get(k, 0)
        st_v = st.get(k, 0)
        diff = st_v - ic_v
        better = ("✓" if (sign * diff) > 0 else "✗") if k != "trade_days" else "—"
        print(
            f"| {label} | {fmt.format(ic_v)} | {fmt.format(st_v)} | "
            f"{fmt.format(diff)} | {better} |"
        )

    print()
    print("**Acceptance (must hit all three vs. IC baseline):**")
    print(
        f"- Sharpe ≥ {IC_BASELINE['sharpe']}: "
        f"{'PASS' if st['sharpe'] >= IC_BASELINE['sharpe'] else 'FAIL'}"
    )
    print(
        f"- Worst day ≥ ${IC_BASELINE['worst_day']:+.0f}: "
        f"{'PASS' if st['worst_day'] >= IC_BASELINE['worst_day'] else 'FAIL'}"
    )
    print(
        f"- Total PnL > ${IC_BASELINE['total_pnl']:+.0f}: "
        f"{'PASS' if st['total_pnl'] > IC_BASELINE['total_pnl'] else 'FAIL'}"
    )
    print()
    print(f"**OVERALL: {'STRANGLE WINS' if passes else 'IC HOLDS'}**")

    if args.ic_meta:
        meta = json.loads(Path(args.ic_meta).read_text())
        print()
        print(
            f"IC SL$ distribution used for risk-normalization: "
            f"p75=${meta['p75']:.0f}, median=${meta['median']:.0f}, "
            f"n_days={meta['n_days']}"
        )


if __name__ == "__main__":
    main()
