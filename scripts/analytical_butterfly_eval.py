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
import itertools
import json
import math
import pickle
import sys
from datetime import date, time
from pathlib import Path
from typing import Dict, List, Tuple

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


def run_config(by_date: Dict, closes: List[Tuple[date, float]], base: bf.ButterflyConfig,
               dir_mode: str, width_mode: str, fixed_width: float) -> Dict:
    """Build per-day cfgs and aggregate (one trade/day)."""
    results: List[Dict] = []
    skips: Dict[str, int] = {}
    for d in sorted(by_date.keys()):
        info = by_date[d]
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
          out_dir: Path) -> None:
    """WF-CV sweep over direction x exit x width; select per fold on train Sharpe,
    score on OOS val. Coin-flip direction is always reported as a baseline."""
    dir_modes = ["hull", "call", "put", "coin", "anti"]
    exit_models = ["hold", "target", "trail"]
    width_specs = [("regime", 0.0), ("fixed", 25.0), ("fixed", 40.0), ("fixed", 55.0)]
    combos = list(itertools.product(dir_modes, exit_models, width_specs))

    dates = sorted(by_date.keys())
    folds = bf.make_wf_folds(dates, ratio=5, folds=4)
    print(f"Sweep: {len(combos)} combos x {len(folds)} folds")

    fold_rows: List[Dict] = []
    for fi, (train, val) in enumerate(folds):
        train_set, val_set = set(train), set(val)
        scored = []
        for (dm, xm, (wmode, fw)) in combos:
            base = bf.ButterflyConfig(entry_anchor=entry, anchors=anchors, exit_model=xm)
            tr = run_config({d: by_date[d] for d in by_date if d in train_set},
                            closes, base, dm, wmode, fw)
            scored.append(((dm, xm, wmode, fw), tr.get("sharpe_ann", 0.0), tr.get("n", 0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        (dm, xm, wmode, fw), tr_sharpe, tr_n = scored[0]
        base = bf.ButterflyConfig(entry_anchor=entry, anchors=anchors, exit_model=xm)
        vr = run_config({d: by_date[d] for d in by_date if d in val_set},
                        closes, base, dm, wmode, fw)
        row = {"fold": fi, "pick": f"{dm}/{xm}/{wmode}:{fw}",
               "train_sharpe": round(tr_sharpe, 2), "train_n": tr_n,
               "val_n": vr.get("n", 0), "val_total": round(vr.get("total", 0.0), 0),
               "val_mean": round(vr.get("mean", 0.0), 1),
               "val_sharpe": round(vr.get("sharpe_ann", 0.0), 2)}
        fold_rows.append(row)
        print(f"  fold {fi}: pick={row['pick']:<28} train_Sharpe={row['train_sharpe']:<6} "
              f"-> OOS n={row['val_n']} mean=${row['val_mean']:+.1f} Sharpe={row['val_sharpe']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wf_folds.json", "w") as f:
        json.dump(fold_rows, f, indent=2)
    oos_mean = sum(r["val_mean"] * r["val_n"] for r in fold_rows) / max(
        1, sum(r["val_n"] for r in fold_rows))
    print(f"\nPooled OOS mean/trade = ${oos_mean:+.1f}  (>0 with positive skew = primary success test)")


def main() -> int:
    p = argparse.ArgumentParser(description="Analytical OTM butterfly evaluator")
    p.add_argument("--cache", default="data/cbbo_cache_fly_10min")
    p.add_argument("--bars", default="data/spy_1m.csv")
    p.add_argument("--vix", default="data/vix1d_daily.csv")
    p.add_argument("--start", default="2022-05-11")
    p.add_argument("--end", default="2026-03-30")
    p.add_argument("--entry", default="09:30")
    p.add_argument("--anchors", default="")
    p.add_argument("--dir-mode", default="hull", choices=["hull", "call", "put", "coin", "anti"])
    p.add_argument("--width-mode", default="regime", choices=["regime", "fixed"])
    p.add_argument("--fixed-width", type=float, default=40.0)
    p.add_argument("--exit", default="hold", choices=["hold", "target", "trail"])
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
        sweep(by_date, closes, args.entry, anchors, Path(args.out))
        return 0

    base = bf.ButterflyConfig(entry_anchor=args.entry, anchors=anchors, exit_model=args.exit)
    headline = run_config(by_date, closes, base, args.dir_mode, args.width_mode, args.fixed_width)
    _print(f"HEADLINE {args.dir_mode}/{args.exit}/{args.width_mode} (mid, zero-cost)", headline)
    if headline.get("n"):
        _regime_breakdown(headline)

    base_c = bf.ButterflyConfig(entry_anchor=args.entry, anchors=anchors, exit_model=args.exit,
                                slippage_per_leg=args.slippage, commission_per_leg=args.commission)
    costs = run_config(by_date, closes, base_c, args.dir_mode, args.width_mode, args.fixed_width)
    _print(f"WITH COSTS (slip=${args.slippage}/leg, comm=${args.commission}/leg)", costs)

    _write_trades(Path(args.out), headline, "headline")
    print(f"\nPer-trade rows -> {args.out}/trades_headline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
