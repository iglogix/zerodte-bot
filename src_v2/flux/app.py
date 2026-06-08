# src_v2/flux/app.py
"""Flask shell: JSON endpoints over the Flux compute layer + the dashboard page."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src_v2.flux.engine import atm_iv, ratio_point
from src_v2.flux.history import daily_ratio_series
from src_v2.flux.scanner import scan
from src_v2.flux.sources.base import FluxSource
from src_v2.flux.sources.marketdata import MarketDataSource
from src_v2.backtest.dc_time_machine.trading_calendar import sessions_between, trading_days_after

_STATIC = Path(__file__).parent / "static"


def _last_eod_session() -> date:
    """Most recent NYSE session strictly before today — today's EOD isn't settled yet."""
    today = date.today()
    s = sessions_between(today - timedelta(days=12), today - timedelta(days=1))
    return s[-1] if s else today - timedelta(days=1)


def _rp_json(p):
    return {"as_of": p.as_of.isoformat(), "front_iv": p.front_iv,
            "back_iv": p.back_iv, "ratio": p.ratio,
            "front_expiry": p.front_expiry.isoformat(), "back_expiry": p.back_expiry.isoformat()}


def _sr_json(s):
    return {"front_expiry": s.front_expiry.isoformat(), "back_expiry": s.back_expiry.isoformat(),
            "gap_td": s.gap_td, "ratio": round(s.ratio, 4), "ratio_prev": round(s.ratio_prev, 4),
            "drop": round(s.drop, 4), "zscore": round(s.zscore, 2), "flag": s.flag}


def create_app(source: FluxSource | None = None, live_source: FluxSource | None = None) -> Flask:
    if source is None:
        from src_v2.flux.cache import ChainCache
        source = MarketDataSource(cache=ChainCache())   # cache EOD fetches → fewer credits
    src = source
    live = live_source                                   # TastyTradeLiveSource (None = live off)
    app = Flask(__name__)

    @app.get("/api/live_ratio")
    def live_ratio():
        if live is None:
            return jsonify({"error": "live source not configured (start with --live)"}), 200
        front = date.fromisoformat(request.args["front"])
        back = date.fromisoformat(request.args["back"])
        rd = float(request.args.get("ref_delta", 0.35))
        try:
            rp = ratio_point(live.chain(front), live.chain(back), rd)
        except Exception as e:
            return jsonify({"error": str(e)}), 200
        if rp is None:
            return jsonify({"error": "no live quotes for this pair"}), 200
        return jsonify(_rp_json(rp))

    @app.get("/api/expirations_iv")
    def expirations_iv():
        if live is None:
            return jsonify({"error": "live source not configured", "expirations": []}), 200
        ref = date.fromisoformat(request.args.get("ref", date.today().isoformat()))
        rd = float(request.args.get("ref_delta", 0.35))
        lo, hi = int(request.args.get("min_dte", 1)), int(request.args.get("max_dte", 15))
        out = []
        for e in live.expirations(ref, lo, hi):
            try:
                iv = atm_iv(live.chain(e), rd)
            except Exception:
                iv = None
            out.append({"expiry": e.isoformat(), "dte": trading_days_after(ref, e),
                        "atm_iv": round(iv, 4) if iv is not None else None})
        return jsonify({"expirations": out})

    @app.get("/api/ratio")
    def ratio():
        fd = int(request.args.get("front_dte", 9))
        gap = int(request.args.get("gap", 1))
        rd = float(request.args.get("ref_delta", 0.35))
        view = request.args.get("view", "5day")
        end = date.fromisoformat(request.args.get("end_date", date.today().isoformat()))
        n = {"5day": 5, "20day": 20}.get(view, 5)
        try:
            pts = daily_ratio_series(src, fd, gap, n, rd, end)
        except Exception as e:
            return jsonify({"error": str(e), "points": []}), 200
        meta = {"front_expiry": pts[-1].front_expiry.isoformat() if pts else None,
                "back_expiry": pts[-1].back_expiry.isoformat() if pts else None, "view": view}
        return jsonify({"points": [_rp_json(p) for p in pts], "meta": meta})

    @app.get("/api/scan")
    def do_scan():
        on = date.fromisoformat(request.args.get("on_date", _last_eod_session().isoformat()))
        rd = float(request.args.get("ref_delta", 0.35))
        gaps = [int(x) for x in request.args.get("gaps", "1").split(",")]
        try:
            rows = scan(src, on, int(request.args.get("min_dte", 0)),
                        int(request.args.get("max_dte", 30)), gaps, rd,
                        by=request.args.get("by", "drop"))
        except Exception as e:
            return jsonify({"error": str(e), "rows": []}), 200
        return jsonify({"rows": [_sr_json(s) for s in rows]})

    @app.get("/api/expirations")
    def expirations():
        ref = date.fromisoformat(request.args.get("ref", date.today().isoformat()))
        exps = src.expirations(ref, int(request.args.get("min_dte", 0)),
                               int(request.args.get("max_dte", 30)))
        return jsonify({"expirations": [e.isoformat() for e in exps]})

    @app.get("/")
    def index():
        return send_from_directory(_STATIC, "flux.html")

    return app
