// Live P&L Cockpit controller.
(function () {
  "use strict";

  var POLL_MS = 2000;

  var state = {
    positions: [],
    selected: new Set(),     // selected trade_ids
    collapsed: new Set(),    // collapsed strategy names
    marks: {},               // symbol -> quote
    pnlByTrade: {},          // trade_id -> {open_pnl, max_profit}
    samples: [],             // [{ts, trade_id, strategy, open_pnl}] (history + live)
    underlying: null,
    sim: false,
    stale: false,
  };

  var $ = function (id) { return document.getElementById(id); };
  var fmtUsd = function (v) {
    if (v == null || isNaN(v)) return "—";
    var s = (v < 0 ? "-" : "+") + "$" + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
    return s;
  };
  var cls = function (v) { return v == null || isNaN(v) ? "" : v >= 0 ? "pos" : "neg"; };

  function tYears() {
    var now = new Date();
    var close = new Date(now);
    close.setHours(16, 0, 0, 0);
    var hours = Math.max((close - now) / 3.6e6, 0.5);
    return hours / (24 * 365);
  }

  function ivLookup(sym) {
    var q = state.marks[sym];
    return q && q.iv ? q.iv : 0.16;
  }

  // ---- data fetch --------------------------------------------------------
  async function loadPositions() {
    var r = await fetch("/api/positions");
    var data = await r.json();
    state.positions = data.positions || [];
    state.sim = !!data.sim;
    state.positions.forEach(function (p) { state.selected.add(p.trade_id); });
    var badge = $("mode-badge");
    badge.textContent = state.sim ? "SIM" : "LIVE";
    badge.className = "badge " + (state.sim ? "sim" : "live");
    buildTable();
  }

  async function loadHistory() {
    try {
      var r = await fetch("/api/pnl_history");
      var data = await r.json();
      state.samples = (data.samples || []).map(function (s) {
        return { ts: s.ts, trade_id: s.trade_id, strategy: s.strategy, open_pnl: s.open_pnl };
      });
    } catch (e) { /* no history yet */ }
  }

  async function poll() {
    try {
      var r = await fetch("/api/quotes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ positions: state.positions }),
      });
      if (!r.ok) throw new Error("HTTP " + r.status);
      var data = await r.json();
      state.marks = data.marks || {};
      state.underlying = data.underlying || state.underlying;
      state.pnlByTrade = {};
      (data.pnl || []).forEach(function (p) {
        state.pnlByTrade[p.trade_id] = p;
        if (p.open_pnl != null) {
          state.samples.push({ ts: data.as_of, trade_id: p.trade_id, strategy: p.strategy, open_pnl: p.open_pnl });
        }
      });
      setStale(false);
      if (data.sim && data.sim.clock) {
        var simDt = new Date(data.sim.clock);
        var hh = String(simDt.getHours()).padStart(2, "0");
        var mm = String(simDt.getMinutes()).padStart(2, "0");
        $("sim-clock").textContent = hh + ":" + mm + (data.sim.at_close ? " · CLOSE" : "");
        $("updated").textContent = "session " + hh + ":" + mm;
      } else {
        $("updated").textContent = new Date().toLocaleTimeString();
      }
    } catch (e) {
      setStale(true);
    }
    updateValues();
    recompute();
  }

  function setStale(on) {
    state.stale = on;
    var b = $("conn-badge");
    b.textContent = on ? "STALE" : "live";
    b.className = "badge " + (on ? "stale" : "ok");
  }

  // ---- table -------------------------------------------------------------
  function byStrategy() {
    var groups = {};
    state.positions.forEach(function (p) {
      (groups[p.strategy] = groups[p.strategy] || []).push(p);
    });
    return groups;
  }

  function legLabel(leg) {
    var side = leg.side === "short" ? "−" : "+";
    var rc = leg.right === "call" ? "C" : "P";
    return side + leg.qty + " " + rc + " " + (leg.strike != null ? leg.strike : "?");
  }

  function buildTable() {
    var body = $("pos-body");
    body.innerHTML = "";
    var groups = byStrategy();
    var names = Object.keys(groups).sort();
    if (!names.length) {
      body.innerHTML = '<tr><td colspan="5" class="empty">No open positions.</td></tr>';
      return;
    }
    names.forEach(function (name) {
      var trades = groups[name];
      // strategy header row
      var sr = document.createElement("tr");
      sr.className = "strat-row";
      sr.innerHTML =
        '<td><span class="caret">▾</span>' +
        '<input type="checkbox" class="strat-cb" data-strat="' + name + '"> ' +
        name + ' <span class="chip">' + trades.length + '</span></td>' +
        '<td></td><td></td>' +
        '<td class="num" data-strat-pnl="' + name + '">—</td>' +
        '<td class="num" data-strat-pct="' + name + '">—</td>';
      body.appendChild(sr);

      var caret = sr.querySelector(".caret");
      sr.addEventListener("click", function (e) {
        if (e.target.classList.contains("strat-cb")) return;
        if (state.collapsed.has(name)) state.collapsed.delete(name);
        else state.collapsed.add(name);
        caret.textContent = state.collapsed.has(name) ? "▸" : "▾";
        body.querySelectorAll('[data-group="' + cssEsc(name) + '"]').forEach(function (el) {
          el.style.display = state.collapsed.has(name) ? "none" : "";
        });
      });
      sr.querySelector(".strat-cb").addEventListener("change", function (e) {
        trades.forEach(function (t) {
          if (e.target.checked) state.selected.add(t.trade_id);
          else state.selected.delete(t.trade_id);
        });
        syncCheckboxes();
        recompute();
      });

      trades.forEach(function (t) {
        var tr = document.createElement("tr");
        tr.className = "trade-row";
        tr.setAttribute("data-group", name);
        tr.innerHTML =
          '<td><input type="checkbox" class="trade-cb" data-tid="' + t.trade_id + '"> ' +
          t.trade_id + '</td>' +
          '<td></td>' +
          '<td class="num">' + t.entry_credit.toFixed(2) + '</td>' +
          '<td class="num" data-tid-pnl="' + t.trade_id + '">—</td>' +
          '<td class="num" data-tid-pct="' + t.trade_id + '">—</td>';
        body.appendChild(tr);
        tr.querySelector(".trade-cb").addEventListener("change", function (e) {
          if (e.target.checked) state.selected.add(t.trade_id);
          else state.selected.delete(t.trade_id);
          syncCheckboxes();
          recompute();
        });

        (t.legs || []).forEach(function (leg) {
          var lr = document.createElement("tr");
          lr.className = "leg-row";
          lr.setAttribute("data-group", name);
          lr.innerHTML =
            '<td class="' + leg.side + '">' + legLabel(leg) + '</td>' +
            '<td class="num" data-sym="' + leg.symbol + '">—</td>' +
            '<td></td><td></td><td></td>';
          body.appendChild(lr);
        });
      });
    });
    syncCheckboxes();
  }

  function cssEsc(s) { return s.replace(/"/g, '\\"'); }

  function syncCheckboxes() {
    document.querySelectorAll(".trade-cb").forEach(function (cb) {
      cb.checked = state.selected.has(cb.getAttribute("data-tid"));
    });
    var groups = byStrategy();
    document.querySelectorAll(".strat-cb").forEach(function (cb) {
      var name = cb.getAttribute("data-strat");
      var trades = groups[name] || [];
      var all = trades.every(function (t) { return state.selected.has(t.trade_id); });
      var some = trades.some(function (t) { return state.selected.has(t.trade_id); });
      cb.checked = all;
      cb.indeterminate = !all && some;
    });
  }

  function updateValues() {
    // per-leg marks
    document.querySelectorAll("[data-sym]").forEach(function (td) {
      var q = state.marks[td.getAttribute("data-sym")];
      td.textContent = q && q.mark != null ? q.mark.toFixed(2) : (state.stale ? td.textContent : "—");
    });
    // per-trade pnl + % max
    state.positions.forEach(function (t) {
      var p = state.pnlByTrade[t.trade_id];
      var pnlCell = document.querySelector('[data-tid-pnl="' + t.trade_id + '"]');
      var pctCell = document.querySelector('[data-tid-pct="' + t.trade_id + '"]');
      if (pnlCell) {
        var v = p ? p.open_pnl : null;
        pnlCell.textContent = fmtUsd(v);
        pnlCell.className = "num " + cls(v);
      }
      if (pctCell) {
        var pct = p && p.open_pnl != null && t.max_profit ? (p.open_pnl / t.max_profit) * 100 : null;
        pctCell.textContent = pct == null ? "—" : pct.toFixed(0) + "%";
      }
    });
    // per-strategy aggregate
    var groups = byStrategy();
    Object.keys(groups).forEach(function (name) {
      var ids = groups[name].map(function (t) { return t.trade_id; });
      var agg = Pnl.aggregate(toPnlList(), ids);
      var pnlCell = document.querySelector('[data-strat-pnl="' + cssEsc(name) + '"]');
      var pctCell = document.querySelector('[data-strat-pct="' + cssEsc(name) + '"]');
      if (pnlCell) { pnlCell.textContent = fmtUsd(agg.total); pnlCell.className = "num " + cls(agg.total); }
      if (pctCell) { pctCell.textContent = agg.max_profit ? (agg.total / agg.max_profit * 100).toFixed(0) + "%" : "—"; }
    });
  }

  function toPnlList() {
    return state.positions.map(function (t) {
      var p = state.pnlByTrade[t.trade_id] || {};
      return { trade_id: t.trade_id, strategy: t.strategy, open_pnl: p.open_pnl, max_profit: t.max_profit };
    });
  }

  // ---- charts ------------------------------------------------------------
  // Fresh layout per call — Plotly mutates layout.xaxis (e.g. sets type:'date'),
  // so sharing one object would leak the date axis from the curve to the payoff.
  function darkLayout(opts) {
    var base = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#8b97a7", size: 11 },
      margin: { l: 56, r: 14, t: 10, b: 34 },
      showlegend: true,
      legend: { orientation: "h", y: 1.12, x: 0, font: { size: 10 } },
      xaxis: { gridcolor: "#222b36", zerolinecolor: "#2a3441" },
      yaxis: { gridcolor: "#222b36", zerolinecolor: "#2a3441", tickprefix: "$" },
    };
    return Object.assign(base, opts || {});
  }

  function selectedPositions() {
    return state.positions.filter(function (p) { return state.selected.has(p.trade_id); });
  }

  function recompute() {
    var sel = Array.from(state.selected);
    var agg = Pnl.aggregate(toPnlList(), sel);
    var tp = $("total-pnl");
    tp.textContent = fmtUsd(agg.total);
    tp.className = "value " + cls(agg.total);
    $("pnl-pct").textContent = agg.max_profit ? (agg.total / agg.max_profit * 100).toFixed(0) + "%" : "—";

    var g = Pnl.netGreeks(state.positions, state.marks, sel);
    $("g-delta").textContent = g.delta.toFixed(1);
    $("g-gamma").textContent = g.gamma.toFixed(2);
    $("g-theta").textContent = fmtUsd(g.theta);
    $("g-vega").textContent = fmtUsd(g.vega);

    if (state.underlying) {
      var px = state.underlying.last || state.underlying.mid;
      $("spx").textContent = px != null ? px.toFixed(2) : "—";
    }

    drawCurve(sel);
    drawPayoff();
  }

  function drawCurve(sel) {
    var c = Pnl.curveFromHistory(state.samples, sel);
    var x = c.x.map(function (t) { return new Date(t); });
    var color = c.y.length && c.y[c.y.length - 1] >= 0 ? "#2ec26a" : "#ff5d5d";
    Plotly.react("curve", [{
      x: x, y: c.y, type: "scatter", mode: "lines", name: "Open P&L",
      line: { color: color, width: 2 }, fill: "tozeroy",
      fillcolor: color === "#2ec26a" ? "rgba(46,194,106,0.12)" : "rgba(255,93,93,0.12)",
    }], darkLayout({
      showlegend: false,
      xaxis: { type: "date", gridcolor: "#222b36", zerolinecolor: "#2a3441" },
    }), { displayModeBar: false, responsive: true });
  }

  function drawPayoff() {
    var positions = selectedPositions();
    var center = state.underlying && (state.underlying.last || state.underlying.mid) || 6000;
    var spots = Payoff.spotGrid(center, 0.03, 81);
    var now = Payoff.sumPayoff(positions, spots, tYears(), "now", ivLookup);
    var expiry = Payoff.sumPayoff(positions, spots, 0, "expiry", ivLookup);
    var layout = darkLayout({
      xaxis: { type: "linear", gridcolor: "#222b36", zerolinecolor: "#2a3441",
               title: { text: "SPX", font: { size: 10 } } },
      shapes: [{
        type: "line", x0: center, x1: center, yref: "paper", y0: 0, y1: 1,
        line: { color: "#4c9aff", width: 1, dash: "dot" },
      }],
    });
    Plotly.react("payoff", [
      { x: spots, y: expiry, type: "scatter", mode: "lines", name: "At expiry",
        line: { color: "#8b97a7", width: 1.5 } },
      { x: spots, y: now, type: "scatter", mode: "lines", name: "Now",
        line: { color: "#4c9aff", width: 2 } },
    ], layout, { displayModeBar: false, responsive: true });
  }

  // ---- boot --------------------------------------------------------------
  async function boot() {
    await loadPositions();
    await loadHistory();
    await poll();
    setInterval(poll, POLL_MS);
  }

  boot();
})();
