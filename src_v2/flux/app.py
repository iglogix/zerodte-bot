# src_v2/flux/app.py
"""Flask shell: JSON endpoints over the Flux compute layer + the dashboard page."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src_v2.flux.history import daily_ratio_series
from src_v2.flux.scanner import scan
from src_v2.flux.sources.base import FluxSource
from src_v2.flux.sources.marketdata import MarketDataSource

_STATIC = Path(__file__).parent / "static"


def _rp_json(p):
    return {"as_of": p.as_of.isoformat(), "front_iv": p.front_iv,
            "back_iv": p.back_iv, "ratio": p.ratio,
            "front_expiry": p.front_expiry.isoformat(), "back_expiry": p.back_expiry.isoformat()}


def _sr_json(s):
    return {"front_expiry": s.front_expiry.isoformat(), "back_expiry": s.back_expiry.isoformat(),
            "gap_td": s.gap_td, "ratio": round(s.ratio, 4), "ratio_prev": round(s.ratio_prev, 4),
            "drop": round(s.drop, 4), "zscore": round(s.zscore, 2), "flag": s.flag}


def create_app(source: FluxSource | None = None) -> Flask:
    if source is None:
        from src_v2.flux.cache import ChainCache
        source = MarketDataSource(cache=ChainCache())   # cache EOD fetches → fewer credits
    src = source
    app = Flask(__name__)

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
        on = date.fromisoformat(request.args.get("on_date", date.today().isoformat()))
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
