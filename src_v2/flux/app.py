# src_v2/flux/app.py
"""Flask shell: JSON endpoints over the Flux compute layer + the dashboard page."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src_v2.flux.engine import atm_iv, ratio_point
from src_v2.flux.history import daily_ratio_series
from src_v2.flux.orders import build_dc_order, select_dc_strikes
from src_v2.flux.scanner import scan
from src_v2.flux.sources.base import FluxSource
from src_v2.flux.sources.marketdata import MarketDataSource
from src_v2.backtest.dc_time_machine.trading_calendar import (
    nth_trading_day_after, sessions_between, trading_days_after,
)

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


def _ticket_json(t):
    return {"kind": t.kind, "price": t.price, "price_effect": t.price_effect,
            "est_cost": t.est_cost, "note": t.note,
            "legs": [{"osi": l.osi, "expiry": l.expiry.isoformat(), "strike": l.strike,
                      "is_call": l.is_call, "action": l.action, "qty": l.quantity}
                     for l in t.legs]}


def _sr_json(s):
    return {"front_expiry": s.front_expiry.isoformat(), "back_expiry": s.back_expiry.isoformat(),
            "gap_td": s.gap_td, "ratio": round(s.ratio, 4), "ratio_prev": round(s.ratio_prev, 4),
            "drop": round(s.drop, 4), "zscore": round(s.zscore, 2), "flag": s.flag}


def create_app(source: FluxSource | None = None, live_source: FluxSource | None = None,
               intraday_source=None, submitter=None) -> Flask:
    if source is None:
        from src_v2.flux.cache import ChainCache
        source = MarketDataSource(cache=ChainCache())   # cache EOD fetches → fewer credits
    src = source
    live = live_source                                   # TastyTradeLiveSource (None = live off)
    intraday = intraday_source                            # ThetaIntradaySource (None = off)
    sub = submitter                                       # OrderSubmitter (None/manual = no submit)
    submit_mode = sub.mode if sub is not None else "manual"
    submit_proxy = getattr(sub, "proxy", None)            # "spy" → cert proxy (no SPXW)
    app = Flask(__name__)

    @app.get("/api/intraday")
    def intraday_ep():
        if intraday is None:
            return jsonify({"error": "intraday source not configured", "points": []}), 200
        # "Today" = today's live (delayed) intraday; ?session=YYYY-MM-DD replays a past day.
        session = date.fromisoformat(request.args.get("session", date.today().isoformat()))
        # accept explicit expiry dates, or resolve from front_dte/gap relative to the session
        if request.args.get("front") and request.args.get("back"):
            front = date.fromisoformat(request.args["front"])
            back = date.fromisoformat(request.args["back"])
        else:
            fd, gap = int(request.args.get("front_dte", 9)), int(request.args.get("gap", 1))
            front = nth_trading_day_after(session, fd)
            back = nth_trading_day_after(session, fd + gap)
            if front is None or back is None:
                return jsonify({"error": "could not resolve expiries", "points": []}), 200
        rd = float(request.args.get("ref_delta", 0.35))
        atm = request.args.get("atm")
        atm = float(atm) if atm else getattr(intraday, "last_spot", 7400.0)
        try:
            pts = intraday.intraday(session, front, back, rd, atm)
        except Exception as e:
            return jsonify({"error": str(e), "points": []}), 200
        return jsonify({"points": [_rp_json(p) for p in pts],
                        "meta": {"session": session.isoformat(), "atm": atm}})

    def _make_dc_ticket():
        """Build the DC ticket from request args → (ticket, error_str)."""
        from dataclasses import replace
        if intraday is None:
            return None, "intraday source not configured"
        session = date.fromisoformat(request.args.get("session", date.today().isoformat()))
        if request.args.get("front") and request.args.get("back"):
            front = date.fromisoformat(request.args["front"])
            back = date.fromisoformat(request.args["back"])
        else:
            fd, gap = int(request.args.get("front_dte", 9)), int(request.args.get("gap", 1))
            front = nth_trading_day_after(session, fd)
            back = nth_trading_day_after(session, fd + gap)
        if front is None or back is None:
            return None, "could not resolve expiries"
        td = float(request.args.get("target_delta", 0.35))
        qty = max(1, int(request.args.get("qty", 1)))
        try:
            fchain = intraday.latest_chain(session, front)
            bchain = intraday.latest_chain(session, back)
        except Exception as e:
            return None, str(e)
        kp, kc = request.args.get("k_put"), request.args.get("k_call")
        kp, kc = (float(kp), float(kc)) if (kp and kc) else select_dc_strikes(fchain, td)
        if kp is None or kc is None:
            return None, "no strikes (no live chain)"
        ticket = build_dc_order(fchain, bchain, kp, kc, qty)
        if ticket is None:
            return None, "could not price DC (missing leg quotes)"
        lim = request.args.get("limit")          # optional user-edited net limit
        if lim:
            ticket = replace(ticket, price=float(lim), est_cost=round(float(lim) * 100 * qty, 2))
        return ticket, None

    @app.get("/api/dc_order")
    def dc_order():
        ticket, err = _make_dc_ticket()
        if err:
            return jsonify({"error": err}), 200
        out = {"ticket": _ticket_json(ticket), "mode": submit_mode, "proxy": submit_proxy}
        if submit_proxy == "spy":                          # show what cert will actually receive
            from src_v2.flux.orders import spy_proxy
            out["proxy_ticket"] = _ticket_json(spy_proxy(ticket))
        return jsonify(out)

    @app.post("/api/dc_submit")
    def dc_submit():
        if sub is None or sub.mode == "manual":
            return jsonify({"ok": False, "detail":
                            "submission disabled — manual mode (start serve with --mode sandbox|live)"}), 200
        ticket, err = _make_dc_ticket()
        if err:
            return jsonify({"ok": False, "detail": err}), 200
        execution = request.args.get("execution", "combo")
        try:
            res = sub.submit(ticket, execution)
        except Exception as e:
            return jsonify({"ok": False, "detail": f"submit error: {e}"}), 200
        return jsonify({"ok": res.ok, "order_ids": res.order_ids,
                        "detail": res.detail, "mode": sub.mode, "execution": execution,
                        "ticket": _ticket_json(ticket)})

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
