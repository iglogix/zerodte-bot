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
    simInfo: null,           // latest {minute, total, clock, at_close} in sim mode
    stale: false,
  };

  // Per-strategy colors (avoid blue=Now/EM and amber=break-even). Assigned
  // stably by sorted strategy name so a strategy keeps its color across redraws.
  var STRAT_PALETTE = ["#2ec26a", "#b07cff", "#ff6fae", "#36d0c5", "#9aa0ff", "#e8b04b"];
  function stratColorMap() {
    var names = Array.from(new Set(state.positions.map(function (p) { return p.strategy; }))).sort();
    var m = {};
    names.forEach(function (n, i) { m[n] = STRAT_PALETTE[i % STRAT_PALETTE.length]; });
    return m;
  }

  var $ = function (id) { return document.getElementById(id); };
  var fmtUsd = function (v) {
    if (v == null || isNaN(v)) return "—";
    var s = (v < 0 ? "-" : "+") + "$" + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
    return s;
  };
  var cls = function (v) { return v == null || isNaN(v) ? "" : v >= 0 ? "pos" : "neg"; };

  function tYears() {
    // In sim mode, time-to-close comes from the simulated session clock so the
    // "now" payoff curve matches the server's marks (and decays toward expiry
    // as the day progresses). Otherwise fall back to the real wall clock.
    if (state.simInfo && typeof state.simInfo.minute === "number") {
      var left = Math.max(state.simInfo.total - state.simInfo.minute, 0);
      return Math.max(left / 525600, 1e-6); // minutes -> years
    }
    var now = new Date();
    var close = new Date(now);
    close.setHours(16, 0, 0, 0);
    var hours = Math.max((close - now) / 3.6e6, 1e-6);
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
    var resetBtn = $("reset-btn");
    if (state.sim && resetBtn) resetBtn.style.display = "";
    buildTable();
  }

  async function resetDay() {
    try {
      await fetch("/api/sim/reset", { method: "POST" });
    } catch (e) { /* ignore */ }
    state.samples = [];           // drop the old day's curve immediately
    Plotly.purge("curve");
    location.reload();            // reseed cleanly from 09:30
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
      state.simInfo = data.sim || null;
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
    // Rendering must never throw out of poll() — a single bad frame would
    // otherwise kill the interval and freeze the page.
    try {
      updateValues();
      recompute();
    } catch (e) {
      console.error("cockpit render error", e);
    }
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
    var colors = stratColorMap();
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
        '<span class="dot" style="background:' + colors[name] + '"></span>' +
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

  function avgIv() {
    var ivs = [];
    for (var k in state.marks) {
      var q = state.marks[k];
      if (q && q.iv) ivs.push(q.iv);
    }
    return ivs.length ? ivs.reduce(function (a, b) { return a + b; }, 0) / ivs.length : 0.16;
  }

  function drawPayoff() {
    var positions = selectedPositions();
    var center = (state.underlying && (state.underlying.last || state.underlying.mid)) || 6000;
    var spots = Payoff.spotGrid(center, 0.03, 81);
    var now = Payoff.sumPayoff(positions, spots, tYears(), "now", ivLookup);
    var expiry = Payoff.sumPayoff(positions, spots, 0, "expiry", ivLookup);

    // (2) break-evens: where the expiry payoff crosses zero (depends on selection)
    var bes = positions.length ? Payoff.zeroCrossings(spots, expiry) : [];
    // (3) expected move of the underlying: spot ± EM = ATM straddle over the
    // *remaining* time to close, so it shrinks through the day. Floor the
    // horizon at ~20 min so the last minutes don't degenerate to ~$1.
    var tEm = Math.max(tYears(), 20 / 525600);
    var em = Payoff.atmStraddle(center, tEm, avgIv());
    var emLo = center - em, emHi = center + em;

    var shapes = [
      // (3) EM band + edges
      { type: "rect", xref: "x", x0: emLo, x1: emHi, yref: "paper", y0: 0, y1: 1,
        fillcolor: "rgba(76,154,255,0.07)", line: { width: 0 }, layer: "below" },
      { type: "line", x0: emLo, x1: emLo, yref: "paper", y0: 0, y1: 1,
        line: { color: "#4c9aff", width: 1, dash: "dot" } },
      { type: "line", x0: emHi, x1: emHi, yref: "paper", y0: 0, y1: 1,
        line: { color: "#4c9aff", width: 1, dash: "dot" } },
      // current spot
      { type: "line", x0: center, x1: center, yref: "paper", y0: 0, y1: 1,
        line: { color: "#d7dee8", width: 1, dash: "dot" } },
    ];
    var annotations = [
      { x: emHi, y: 1, yref: "paper", yanchor: "bottom", xanchor: "left",
        text: "EM ±" + em.toFixed(0) + " (straddle)", showarrow: false,
        font: { size: 9, color: "#4c9aff" } },
    ];
    // (2) break-even vertical lines + labels
    bes.forEach(function (be) {
      shapes.push({ type: "line", x0: be, x1: be, yref: "paper", y0: 0, y1: 1,
        line: { color: "#f0a93b", width: 1, dash: "dash" } });
      annotations.push({ x: be, y: 0, yanchor: "bottom", xanchor: "center",
        text: "BE " + be.toFixed(0), showarrow: false,
        font: { size: 9, color: "#f0a93b" } });
    });

    var layout = darkLayout({
      xaxis: { type: "linear", gridcolor: "#222b36", zerolinecolor: "#2a3441",
               title: { text: "SPX", font: { size: 10 } } },
      // (1) highlight the zero-profit line
      yaxis: { gridcolor: "#222b36", tickprefix: "$", zeroline: true,
               zerolinecolor: "#7c8a9a", zerolinewidth: 2 },
      shapes: shapes,
      annotations: annotations,
    });

    var traces = [
      { x: spots, y: expiry, type: "scatter", mode: "lines", name: "At expiry",
        line: { color: "#8b97a7", width: 1.5 } },
      { x: spots, y: now, type: "scatter", mode: "lines", name: "Now",
        line: { color: "#4c9aff", width: 2 } },
    ];
    // (4) positions on the chart: markers per leg, colored by strategy. One
    // trace per strategy so each gets its own legend entry. Short = filled
    // diamond, long = open diamond; call/put shown via the up/down triangle.
    if (positions.length) {
      var colors = stratColorMap();
      var byStrat = {};
      positions.forEach(function (p) {
        (p.legs || []).forEach(function (leg) {
          if (leg.strike == null) return;
          var g = byStrat[p.strategy] || (byStrat[p.strategy] = { x: [], y: [], t: [], sym: [] });
          g.x.push(leg.strike);
          g.y.push(0);
          g.t.push((leg.side === "short" ? "Short " : "Long ") + leg.qty + " " +
                   (leg.right === "call" ? "C" : "P") + " " + leg.strike +
                   "  [" + p.trade_id + "]");
          var base = leg.right === "call" ? "triangle-up" : "triangle-down";
          g.sym.push(leg.side === "short" ? base : base + "-open");
        });
      });
      Object.keys(byStrat).sort().forEach(function (name) {
        var g = byStrat[name];
        traces.push({
          x: g.x, y: g.y, type: "scatter", mode: "markers", name: name,
          marker: { symbol: g.sym, color: colors[name], size: 11,
                    line: { color: "#0d1117", width: 1 } },
          text: g.t, hoverinfo: "text",
        });
      });
    }

    Plotly.react("payoff", traces, layout, { displayModeBar: false, responsive: true });
  }

  // ---- boot --------------------------------------------------------------
  async function boot() {
    var resetBtn = $("reset-btn");
    if (resetBtn) resetBtn.addEventListener("click", resetDay);
    try {
      await loadPositions();
      await loadHistory();
    } catch (e) {
      console.error("cockpit init error", e);
    }
    // Start the recurring poll FIRST and unconditionally, so nothing in the
    // initial render can prevent the loop from running.
    setInterval(poll, POLL_MS);
    poll();
  }

  boot();
})();
