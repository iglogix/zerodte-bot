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


def _latest_marketdata_session(src) -> date:
    """Most recent session that actually has settled EOD data. marketdata.app lags ~1
    session, so 'yesterday' is often still empty — walk back until a near chain has rows.
    (The probe chain is cached, so the subsequent scan reuses it.)"""
    today = date.today()
    sess = sessions_between(today - timedelta(days=16), today - timedelta(days=1))
    for s in reversed(sess):                       # newest first
        exp = nth_trading_day_after(s, 2)
        if exp is None:
            continue
        try:
            ch = src.chain(exp, on_date=s)
        except Exception:
            continue
        if ch and ch.rows:
            return s
    return sess[-1] if sess else today - timedelta(days=1)


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
        rd = float(request.args.get("ref_delta", 0.35))
        atm = request.args.get("atm")
        atm = float(atm) if atm else getattr(intraday, "last_spot", 7400.0)
        explicit = request.args.get("front") and request.args.get("back")
        fd, gap = int(request.args.get("front_dte", 9)), int(request.args.get("gap", 1))

        def _expiries(sess):
            if explicit:
                return date.fromisoformat(request.args["front"]), date.fromisoformat(request.args["back"])
            return nth_trading_day_after(sess, fd), nth_trading_day_after(sess, fd + gap)

        def _series(sess):
            front, back = _expiries(sess)
            if front is None or back is None:
                return None
            return intraday.intraday(sess, front, back, rd, atm)

        # "Today" = today's live (delayed) intraday; ?session=YYYY-MM-DD replays a past day.
        session_arg = request.args.get("session")
        session = date.fromisoformat(session_arg) if session_arg else date.today()
        try:
            pts = _series(session) or []
        except Exception as e:
            return jsonify({"error": str(e), "points": []}), 200
        fell_back = False
        # Market closed / no trades yet → fall back to the most recent session that has data
        # (only when the caller didn't pin an explicit session).
        if not pts and not session_arg:
            for s in reversed(sessions_between(date.today() - timedelta(days=10),
                                               date.today() - timedelta(days=1))):
                try:
                    cand = _series(s)
                except Exception:
                    continue
                if cand:
                    session, pts, fell_back = s, cand, True
                    break
        fe, be = _expiries(session)        # the expiry pair the chart actually compares
        # Resolve the pair's front-DTE / gap (trading days) so the UI dropdowns can mirror the chart.
        fdte = trading_days_after(session, fe) if fe else None
        bdte = trading_days_after(session, be) if be else None
        gap = (bdte - fdte) if (fdte is not None and bdte is not None) else None
        return jsonify({"points": [_rp_json(p) for p in pts],
                        "meta": {"session": session.isoformat(), "atm": atm,
                                 "fell_back": fell_back,
                                 "front_expiry": fe.isoformat() if fe else None,
                                 "back_expiry": be.isoformat() if be else None,
                                 "front_dte": fdte, "gap": gap}})

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

    def _order_json(ticket):
        """Ticket + mode, plus the snapped SPY-proxy preview when sandbox uses one."""
        out = {"ticket": _ticket_json(ticket), "mode": submit_mode, "proxy": submit_proxy}
        if submit_proxy == "spy" and sub is not None:      # show what cert will actually receive
            pt = sub.preview_proxy(ticket)                 # snapped to listed SPY contracts
            if pt is not None:
                out["proxy_ticket"] = _ticket_json(pt)
        return out

    def _submit_response(make_ticket):
        if sub is None or sub.mode == "manual":
            return jsonify({"ok": False, "detail":
                            "submission disabled — manual mode (start serve with --mode sandbox|live)"}), 200
        ticket, err = make_ticket()
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

    @app.get("/api/dc_order")
    def dc_order():
        ticket, err = _make_dc_ticket()
        if err:
            return jsonify({"error": err}), 200
        return jsonify(_order_json(ticket))

    @app.post("/api/dc_submit")
    def dc_submit():
        return _submit_response(_make_dc_ticket)

    def _make_transform_ticket():
        """Build the transform ticket from request args → (ticket, error_str)."""
        from src_v2.flux.positions import DoubleCalendar, build_transform_order
        if intraday is None:
            return None, "intraday source not configured"
        try:
            front = date.fromisoformat(request.args["front"])
            back = date.fromisoformat(request.args["back"])
            k_put = float(request.args["k_put"])
            k_call = float(request.args["k_call"])
        except (KeyError, ValueError):
            return None, "transform needs front, back, k_put, k_call"
        root = request.args.get("root", "SPXW")
        qty = max(1, int(request.args.get("qty", 1)))
        wing = float(request.args.get("wing", 5.0))
        entry_debit = float(request.args.get("entry_debit", 0.0))
        fill = request.args.get("fill", "cross")     # cross = honest limit; mid = optimistic
        session = date.fromisoformat(request.args.get("session", date.today().isoformat()))
        try:
            fchain = intraday.latest_chain(session, front)
            bchain = intraday.latest_chain(session, back)
        except Exception as e:
            return None, str(e)
        dc = DoubleCalendar(root, front, back, k_put, k_call, qty, entry_debit)
        ticket = build_transform_order(dc, wing, fchain, bchain, entry_debit, fill)
        if ticket is None:
            return None, "could not price transform (missing wing/back quotes)"
        return ticket, None

    @app.get("/api/positions")
    def positions_ep():
        if sub is None or not hasattr(sub, "open_double_calendars"):
            return jsonify({"error": "open positions need --mode sandbox|live", "dcs": []}), 200
        try:
            dcs = sub.open_double_calendars()
        except Exception as e:
            return jsonify({"error": str(e), "dcs": []}), 200
        return jsonify({"mode": submit_mode, "dcs": [
            {"root": d.root, "front_expiry": d.front_expiry.isoformat(),
             "back_expiry": d.back_expiry.isoformat(), "k_put": d.k_put,
             "k_call": d.k_call, "qty": d.qty, "entry_debit": d.entry_debit} for d in dcs]})

    @app.get("/api/transform_analysis")
    def transform_analysis():
        """Risk-free wing frontier for a DC: sweep listed wings → credit/banked/floor/ceiling."""
        from src_v2.flux.positions import DoubleCalendar, analyze_transform
        if intraday is None:
            return jsonify({"error": "intraday source not configured", "rows": []}), 200
        try:
            front = date.fromisoformat(request.args["front"])
            back = date.fromisoformat(request.args["back"])
            k_put = float(request.args["k_put"])
            k_call = float(request.args["k_call"])
        except (KeyError, ValueError):
            return jsonify({"error": "needs front, back, k_put, k_call", "rows": []}), 200
        root = request.args.get("root", "SPXW")
        qty = max(1, int(request.args.get("qty", 1)))
        entry_debit = float(request.args.get("entry_debit", 0.0))
        buffer = float(request.args.get("buffer", 0.0))
        max_wing = request.args.get("max_wing")
        max_wing = float(max_wing) if max_wing else None
        session = date.fromisoformat(request.args.get("session", date.today().isoformat()))
        try:
            fchain = intraday.latest_chain(session, front)
            bchain = intraday.latest_chain(session, back)
        except Exception as e:
            return jsonify({"error": str(e), "rows": []}), 200
        dc = DoubleCalendar(root, front, back, k_put, k_call, qty, entry_debit)
        rows, max_rf = analyze_transform(dc, entry_debit, fchain, bchain, max_wing, buffer)
        return jsonify({"entry_debit": entry_debit, "qty": qty, "buffer": buffer,
                        "max_risk_free_wing": max_rf,
                        "rows": [{"wing": r.wing, "credit": r.credit, "banked": r.banked,
                                  "floor": r.floor, "ceiling": r.ceiling,
                                  "risk_free": r.risk_free} for r in rows]})

    @app.get("/api/orders")
    def orders_ep():
        if sub is None or not hasattr(sub, "working_orders"):
            return jsonify({"error": "open orders need --mode sandbox|live", "orders": []}), 200
        try:
            orders = sub.working_orders()
        except Exception as e:
            return jsonify({"error": str(e), "orders": []}), 200
        return jsonify({"mode": submit_mode, "orders": orders})

    @app.get("/api/transform_order")
    def transform_order():
        ticket, err = _make_transform_ticket()
        if err:
            return jsonify({"error": err}), 200
        return jsonify(_order_json(ticket))

    @app.post("/api/transform_submit")
    def transform_submit():
        return _submit_response(_make_transform_ticket)

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
        # Default to the latest session that actually has settled EOD data (marketdata lags).
        on = (date.fromisoformat(request.args["on_date"]) if request.args.get("on_date")
              else _latest_marketdata_session(src))
        rd = float(request.args.get("ref_delta", 0.35))
        gaps = [int(x) for x in request.args.get("gaps", "1").split(",")]
        try:
            rows = scan(src, on, int(request.args.get("min_dte", 0)),
                        int(request.args.get("max_dte", 30)), gaps, rd,
                        by=request.args.get("by", "drop"))
        except Exception as e:
            return jsonify({"error": str(e), "rows": [], "session": on.isoformat()}), 200
        return jsonify({"rows": [_sr_json(s) for s in rows], "session": on.isoformat()})

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
