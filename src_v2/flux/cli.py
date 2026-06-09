"""`python -m src_v2.flux serve|scan`."""
from __future__ import annotations

import argparse
from datetime import date


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="flux")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--intraday", choices=["theta", "off"], default="theta",
                   help="live intraday source (theta = ThetaData stand-in for TT)")
    s.add_argument("--mode", choices=["manual", "sandbox", "live"], default="manual",
                   help="manual (default) builds tickets only; sandbox/live can SUBMIT to TastyTrade")
    sc = sub.add_parser("scan")
    sc.add_argument("--on-date", default=date.today().isoformat())
    sc.add_argument("--min-dte", type=int, default=0)
    sc.add_argument("--max-dte", type=int, default=30)
    sc.add_argument("--gaps", default="1")
    sc.add_argument("--ref-delta", type=float, default=0.35)
    sc.add_argument("--by", choices=["drop", "rise"], default="drop")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "serve":
        from src_v2.flux.app import create_app
        intraday_source = None
        if args.intraday == "theta":
            from src_v2.flux.sources.theta_intraday import ThetaIntradaySource
            intraday_source = ThetaIntradaySource()
        submitter = None
        if args.mode != "manual":          # only connect a broker when explicitly asked to submit
            from src_v2.flux.submit import make_tt_submitter
            try:
                submitter = make_tt_submitter(args.mode)
                print(f"[flux] order submission ENABLED — mode={args.mode}")
            except Exception as e:
                print(f"[flux] submission DISABLED (broker connect failed: {e}); serving manual")
        create_app(intraday_source=intraday_source, submitter=submitter).run(
            host=args.host, port=args.port, debug=False)
        return 0
    if args.cmd == "scan":
        from src_v2.flux.scanner import scan
        from src_v2.flux.sources.marketdata import MarketDataSource
        rows = scan(MarketDataSource(), date.fromisoformat(args.on_date), args.min_dte,
                    args.max_dte, [int(x) for x in args.gaps.split(",")], args.ref_delta, args.by)
        for r in rows[:25]:
            star = "★" if r.flag else " "
            print(f"{star} {r.front_expiry}->{r.back_expiry} gap{r.gap_td} "
                  f"ratio={r.ratio:.3f} drop={r.drop:+.3f} z={r.zscore:+.2f}")
        return 0
    return 1
