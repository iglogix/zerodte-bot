"""Analytical OTM 0DTE butterfly evaluator on a CBBO cache.

Phase 1 (single config, headline + with-costs):
    .venv/bin/python -m scripts.analytical_butterfly_eval \\
        --cache data/cbbo_cache_fly_10min --entry 09:30 \\
        --dir-mode hull --width-mode regime --exit hold \\
        --start 2022-05-11 --end 2026-03-30

Phase 2 (WF-CV sweep over the experimental axes; OOS-scored):
    .venv/bin/python -m scripts.analytical_butterfly_eval \\
        --cache data/cbbo_cache_fly_10min --sweep --start 2022-05-11 --end 2026-03-30
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import pickle
import sys
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_v2.backtest.data_loader import BarLoader
from src_v2.backtest import butterfly_core as bf
from src_v2.backtest.vix_loader import load_vix_series, lookup_prior_close


def load_inputs(cache_dir: Path, bars_path: str, start: date, end: date,
                vix_csv: str) -> Tuple[Dict, List[Tuple[date, float]]]:
    """Return ({date: {quotes, spot_close, vix}}, ordered [(date, daily_close)])."""
    loader = BarLoader(bars_path, price_scale=10.0, symbol="SPX")
    vix_series = load_vix_series(vix_csv)
    by_date: Dict = {}
    closes: List[Tuple[date, float]] = []
    for day in loader.iter_days(start_date=start, end_date=end):
        d = day.session_date
        eod = next((b for b in reversed(day.bars)
                    if b["ts_et"].time() >= time(15, 55)), None)
        if eod is not None:
            closes.append((d, eod["close"]))
        pkl = Path(cache_dir) / f"{d.isoformat()}.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            quotes = pickle.load(f)
        by_date[d] = {
            "quotes": quotes,
            "spot_close": eod["close"] if eod else None,
            "vix": lookup_prior_close(vix_series, date.fromordinal(d.toordinal() - 1)),
        }
    return by_date, closes


def closes_before(closes: List[Tuple[date, float]], d: date) -> List[float]:
    return [c for (cd, c) in closes if cd < d]


def precompute_trend(by_date: Dict, closes: List[Tuple[date, float]], n: int = 21
                     ) -> Dict[date, int]:
    """Hull-MA trend sign per date, computed ONCE from the full causal HMA series.

    `_hma_series` is causal (hma[j] depends only on closes[:j+1]), so the trend at
    date d (index i) = sign(hma[i-1]-hma[i-2]) exactly equals hull_trend(closes<d).
    This turns the per-combo O(N^3) recompute into a single O(N^2) pass — no lookahead.
    """
    ordered = sorted(closes)
    vals = [c for _, c in ordered]
    hma = bf._hma_series(vals, n)
    pos = {d: i for i, (d, _) in enumerate(ordered)}
    trend: Dict[date, int] = {}
    for d in by_date:
        i = pos.get(d)
        if i is None or i < 2:
            trend[d] = 0
            continue
        a, b = hma[i - 1], hma[i - 2]
        trend[d] = 0 if (math.isnan(a) or math.isnan(b)) else (1 if a > b else (-1 if a < b else 0))
    return trend


def _side_from_trend(dir_mode: str, d: date, t: int) -> str:
    if dir_mode == "call":
        return "call"
    if dir_mode == "put":
        return "put"
    if dir_mode == "coin":
        return bf.coin_flip_side(d)
    if dir_mode == "anti":
        return "put" if t >= 0 else "call"
    return "call" if t >= 0 else "put"   # hull


def run_config(by_date: Dict, closes: List[Tuple[date, float]], base: bf.ButterflyConfig,
               dir_mode: str, width_mode: str, fixed_width: float,
               trend_by_date: Optional[Dict[date, int]] = None) -> Dict:
    """Build per-day cfgs and aggregate (one trade/day).

    If `trend_by_date` is supplied, the Hull trend is read from it (fast path);
    otherwise it is computed per-day via build_day_cfg (slower, used for one-off runs).
    """
    results: List[Dict] = []
    skips: Dict[str, int] = {}
    for d in sorted(by_date.keys()):
        info = by_date[d]
        if trend_by_date is not None:
            side = _side_from_trend(dir_mode, d, trend_by_date.get(d, 0))
            width = bf.regime_width(info.get("vix")) if width_mode == "regime" else fixed_width
            cfg = dataclasses.replace(base, side=side, width=width)
        else:
            cfg = bf.build_day_cfg(base, dir_mode, width_mode, d,
                                   closes_before(closes, d), info.get("vix"), fixed_width)
        r = bf.evaluate_day(info["quotes"], cfg, info.get("spot_close"))
        if r.get("skip"):
            skips[r["skip"]] = skips.get(r["skip"], 0) + 1
            continue
        r["date"] = d
        r["vix"] = info.get("vix")
        results.append(r)
    agg = _aggregate(results)
    agg["skips"] = skips
    return agg


def _aggregate(results: List[Dict]) -> Dict:
    pnls = [r["pnl"] for r in results]
    if not pnls:
        return {"n": 0, "results": []}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / max(1, len(pnls) - 1)
    std = math.sqrt(var)
    gl = -sum(losses)
    return {
        "n": len(pnls), "wr": 100 * len(wins) / len(pnls), "total": sum(pnls),
        "mean": mean, "sharpe_ann": math.sqrt(252) * mean / std if std > 0 else 0.0,
        "profit_factor": (sum(wins) / gl) if gl > 0 else float("inf"),
        "max_dd": bf._max_drawdown(pnls),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "results": results,
    }


def _print(label: str, res: Dict) -> None:
    if res.get("n", 0) == 0:
        print(f"\n[{label}] NO TRADES  skips={res.get('skips')}")
        return
    print(f"\n[{label}]  n={res['n']}  WR={res['wr']:.1f}%  total=${res['total']:+,.0f}  "
          f"mean=${res['mean']:+.1f}  Sharpe={res['sharpe_ann']:.2f}  "
          f"PF={res['profit_factor']:.2f}  maxDD=${res['max_dd']:+,.0f}  skips={res.get('skips')}")


def _regime_breakdown(res: Dict) -> None:
    buckets = {"Zombieland(<17)": [], "Goldilocks(17-32)": [], "Chaos(>=32)": []}
    for r in res["results"]:
        v = r.get("vix")
        if v is not None and v >= 32:
            key = "Chaos(>=32)"
        elif v is None or v >= 17:
            key = "Goldilocks(17-32)"
        else:
            key = "Zombieland(<17)"
        buckets[key].append(r["pnl"])
    print("  per-regime:")
    for k, pnls in buckets.items():
        if pnls:
            print(f"    {k:<18} n={len(pnls):>4}  total=${sum(pnls):+,.0f}  "
                  f"mean=${sum(pnls)/len(pnls):+.1f}")
        else:
            print(f"    {k:<18} n=0")


def _write_trades(out_dir: Path, res: Dict, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"trades_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "side", "kind", "exit_anchor", "vix", "em", "forward",
                    "k_low", "k_body", "k_high", "width", "debit", "exit_mark",
                    "r2r_theo", "pnl"])
        for r in res["results"]:
            kl, kb, kh = r["strikes"]
            w.writerow([r["date"], r["side"], r["kind"], r["exit_anchor"],
                        f"{(r.get('vix') or 0):.2f}", f"{r['em']:.2f}", f"{r['forward']:.2f}",
                        kl, kb, kh, r["width"], f"{r['debit']:.2f}", f"{r['exit_mark']:.2f}",
                        f"{r['r2r_theo']:.2f}", f"{r['pnl']:.2f}"])
    summary = {k: v for k, v in res.items() if k != "results"}
    with open(out_dir / f"summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def sweep(by_date: Dict, closes: List[Tuple[date, float]], entry: str, anchors: List[str],
          out_dir: Path, slippage: float, commission: float, min_train_n: int = 30) -> None:
    """Cost-aware WF-CV sweep over direction x exit x width x gates.

    Selection is on NET (with-cost) train Sharpe so the winner is the best REALISTIC
    config, not the best mid-price config. Each fold's pick is scored OOS on the val
    fold both net (primary) and at mid (reference). Combos with < min_train_n train
    trades are excluded so the gates-on degenerate cases (~2 trades) can't win on a
    fluke Sharpe. The Hull trend is precomputed once (fast)."""
    dir_modes = ["hull", "call", "put", "coin", "anti"]
    exit_models = ["hold", "target", "trail"]
    width_specs = [("regime", 0.0), ("fixed", 25.0), ("fixed", 40.0), ("fixed", 55.0)]
    gates_opts = [True, False]
    combos = list(itertools.product(dir_modes, exit_models, width_specs, gates_opts))

    trend = precompute_trend(by_date, closes)

    def cfg(xm: str, gates_on: bool, costs: bool) -> bf.ButterflyConfig:
        return bf.ButterflyConfig(
            entry_anchor=entry, anchors=anchors, exit_model=xm, gates_on=gates_on,
            slippage_per_leg=(slippage if costs else 0.0),
            commission_per_leg=(commission if costs else 0.0))

    dates = sorted(by_date.keys())
    folds = bf.make_wf_folds(dates, ratio=5, folds=4)
    print(f"Sweep: {len(combos)} combos x {len(folds)} folds "
          f"(select on NET train Sharpe, min_train_n={min_train_n})")

    fold_rows: List[Dict] = []
    for fi, (train, val) in enumerate(folds):
        train_by = {d: by_date[d] for d in train if d in by_date}
        val_by = {d: by_date[d] for d in val if d in by_date}
        scored = []
        for (dm, xm, (wmode, fw), g) in combos:
            tr = run_config(train_by, closes, cfg(xm, g, True), dm, wmode, fw, trend)
            if tr.get("n", 0) >= min_train_n:
                scored.append(((dm, xm, wmode, fw, g), tr.get("sharpe_ann", 0.0)))
        if not scored:
            print(f"  fold {fi}: no combo had >= {min_train_n} train trades")
            continue
        scored.sort(key=lambda x: x[1], reverse=True)
        (dm, xm, wmode, fw, g), tr_sharpe = scored[0]
        vr = run_config(val_by, closes, cfg(xm, g, True), dm, wmode, fw, trend)
        vr_mid = run_config(val_by, closes, cfg(xm, g, False), dm, wmode, fw, trend)
        gtag = "gOn" if g else "gOff"
        row = {"fold": fi, "pick": f"{dm}/{xm}/{wmode}:{fw}/{gtag}",
               "train_net_sharpe": round(tr_sharpe, 2),
               "val_n": vr.get("n", 0),
               "val_net_mean": round(vr.get("mean", 0.0), 1),
               "val_net_sharpe": round(vr.get("sharpe_ann", 0.0), 2),
               "val_mid_mean": round(vr_mid.get("mean", 0.0), 1)}
        fold_rows.append(row)
        print(f"  fold {fi}: pick={row['pick']:<26} train_netSharpe={row['train_net_sharpe']:<6} "
              f"-> OOS n={row['val_n']} net_mean=${row['val_net_mean']:+.1f} "
              f"net_Sharpe={row['val_net_sharpe']} (mid_mean=${row['val_mid_mean']:+.1f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wf_folds.json", "w") as f:
        json.dump(fold_rows, f, indent=2)
    tot_n = sum(r["val_n"] for r in fold_rows)
    if tot_n:
        oos_net = sum(r["val_net_mean"] * r["val_n"] for r in fold_rows) / tot_n
        oos_mid = sum(r["val_mid_mean"] * r["val_n"] for r in fold_rows) / tot_n
        print(f"\nPooled OOS NET mean/trade = ${oos_net:+.1f}   (mid = ${oos_mid:+.1f})")
        print("Primary success test: NET OOS mean > 0 with positive skew.")
    else:
        print("\nNo OOS trades selected.")


def main() -> int:
    p = argparse.ArgumentParser(description="Analytical OTM butterfly evaluator")
    p.add_argument("--cache", default="data/cbbo_cache_fly_10min")
    p.add_argument("--bars", default="data/spy_1m.csv")
    # vix9d covers the full 2021-03..2026 range; vix1d only starts 2023-04.
    p.add_argument("--vix", default="data/vix9d_daily.csv")
    p.add_argument("--start", default="2022-05-11")
    p.add_argument("--end", default="2026-03-30")
    # cbbo-1m bars are stamped at interval END, so the 09:30->09:31 bar is
    # labeled "09:31" — that is the real first tradable anchor (not "09:30").
    p.add_argument("--entry", default="09:31")
    p.add_argument("--anchors", default="")
    p.add_argument("--dir-mode", default="hull", choices=["hull", "call", "put", "coin", "anti"])
    p.add_argument("--width-mode", default="regime", choices=["regime", "fixed"])
    p.add_argument("--fixed-width", type=float, default=40.0)
    p.add_argument("--exit", default="hold", choices=["hold", "target", "trail"])
    p.add_argument("--gates-off", action="store_true",
                   help="disable the 10%%-debit / 9:1 entry gate (trade every day)")
    p.add_argument("--slippage", type=float, default=0.05)
    p.add_argument("--commission", type=float, default=1.0)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--out", default="results/butterfly_otm")
    args = p.parse_args()

    anchors = ([a.strip() for a in args.anchors.split(",")] if args.anchors
               else bf.make_anchors(args.entry, "16:00", 10))
    by_date, closes = load_inputs(Path(args.cache), args.bars,
                                  date.fromisoformat(args.start), date.fromisoformat(args.end),
                                  args.vix)
    print(f"Days with cache: {len(by_date)}  entry={args.entry}  anchors={len(anchors)}")

    if args.sweep:
        sweep(by_date, closes, args.entry, anchors, Path(args.out), args.slippage, args.commission)
        return 0

    trend = precompute_trend(by_date, closes)
    gates_on = not args.gates_off
    gtag = "gates-off" if args.gates_off else "gates-on"
    base = bf.ButterflyConfig(entry_anchor=args.entry, anchors=anchors, exit_model=args.exit,
                              gates_on=gates_on)
    headline = run_config(by_date, closes, base, args.dir_mode, args.width_mode, args.fixed_width, trend)
    _print(f"HEADLINE {args.dir_mode}/{args.exit}/{args.width_mode}/{gtag} (mid, zero-cost)", headline)
    if headline.get("n"):
        _regime_breakdown(headline)

    base_c = bf.ButterflyConfig(entry_anchor=args.entry, anchors=anchors, exit_model=args.exit,
                                gates_on=gates_on,
                                slippage_per_leg=args.slippage, commission_per_leg=args.commission)
    costs = run_config(by_date, closes, base_c, args.dir_mode, args.width_mode, args.fixed_width, trend)
    _print(f"WITH COSTS (slip=${args.slippage}/leg, comm=${args.commission}/leg)", costs)

    _write_trades(Path(args.out), headline, "headline")
    print(f"\nPer-trade rows -> {args.out}/trades_headline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
