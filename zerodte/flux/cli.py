"""`python -m zerodte.flux serve|scan`."""
from __future__ import annotations

import argparse
from datetime import date


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="flux")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--intraday", choices=["theta", "tt", "off"], default="tt",
                   help="live intraday source: tt (default; TastyTrade broker prices — needs "
                        "--mode sandbox|live, else falls back to Theta; Theta seeds the morning), "
                        "theta (ThetaData), off")
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
    from dotenv import load_dotenv
    load_dotenv()                    # so --mode sandbox|live can read TT credentials from .env
    args = build_parser().parse_args(argv)
    if args.cmd == "serve":
        from zerodte.flux.app import create_app
        submitter = None
        if args.mode != "manual":          # only connect a broker when explicitly asked to submit
            from zerodte.flux.submit import make_tt_submitter
            try:
                submitter = make_tt_submitter(args.mode)
                print(f"[flux] order submission ENABLED — mode={args.mode}")
            except Exception as e:
                print(f"[flux] submission DISABLED (broker connect failed: {e}); serving manual")

        intraday_source = None
        if args.intraday == "theta":
            from zerodte.flux.sources.theta_intraday import ThetaIntradaySource
            intraday_source = ThetaIntradaySource()
        elif args.intraday == "tt":
            from zerodte.flux.sources.theta_intraday import ThetaIntradaySource
            client = getattr(submitter, "client", None)
            loop = getattr(submitter, "loop", None)
            if client is not None and loop is not None:
                import asyncio
                from zerodte.flux.sources.tastytrade_intraday import TastyTradeIntradaySource

                def _chain_fn(exp):       # run the broker chain fetch on the client's loop
                    return asyncio.run_coroutine_threadsafe(
                        client.get_option_chain("SPX", exp), loop).result(timeout=60)

                # 1-min Theta morning seed (full granularity). The seed is backgrounded, so its
                # compute time doesn't block first paint (~3s live point); the 1-min morning
                # streams in off-thread. Live tail is 1-min from TT.
                intraday_source = TastyTradeIntradaySource(
                    chain_fn=_chain_fn, seed=ThetaIntradaySource())
                print("[flux] intraday = TastyTrade live (broker prices) + Theta 1-min morning seed")
            else:
                intraday_source = ThetaIntradaySource()
                print("[flux] --intraday tt needs --mode sandbox|live; falling back to Theta")

        create_app(intraday_source=intraday_source, submitter=submitter).run(
            host=args.host, port=args.port, debug=False)
        return 0
    if args.cmd == "scan":
        from zerodte.flux.scanner import scan
        from zerodte.flux.sources.marketdata import MarketDataSource
        rows = scan(MarketDataSource(), date.fromisoformat(args.on_date), args.min_dte,
                    args.max_dte, [int(x) for x in args.gaps.split(",")], args.ref_delta, args.by)
        for r in rows[:25]:
            star = "★" if r.flag else " "
            print(f"{star} {r.front_expiry}->{r.back_expiry} gap{r.gap_td} "
                  f"ratio={r.ratio:.3f} drop={r.drop:+.3f} z={r.zscore:+.2f}")
        return 0
    return 1
