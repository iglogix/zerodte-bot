// P&L aggregation + greeks — pure functions (no DOM, no I/O). On window.Pnl.
(function (global) {
  "use strict";

  // Sum open P&L over the selected trades, and break it out by strategy.
  // pnlList: [{trade_id, strategy, open_pnl, max_profit}]
  function aggregate(pnlList, selected) {
    var sel = selected ? new Set(selected) : null;
    var total = 0;
    var maxProfit = 0;
    var byStrategy = {};
    (pnlList || []).forEach(function (p) {
      if (sel && !sel.has(p.trade_id)) return;
      if (p.open_pnl == null) return;
      total += p.open_pnl;
      if (p.max_profit != null) maxProfit += p.max_profit;
      var g = byStrategy[p.strategy] || { open_pnl: 0, max_profit: 0 };
      g.open_pnl += p.open_pnl;
      g.max_profit += p.max_profit || 0;
      byStrategy[p.strategy] = g;
    });
    return { total: total, max_profit: maxProfit, byStrategy: byStrategy };
  }

  // Build the cumulative intraday curve from persisted samples, summing the
  // selected trades at each timestamp.
  // samples: [{ts, trade_id, strategy, open_pnl}] (ordered by ts)
  function curveFromHistory(samples, selected) {
    var sel = selected ? new Set(selected) : null;
    var byTs = new Map();
    (samples || []).forEach(function (s) {
      if (sel && !sel.has(s.trade_id)) return;
      if (s.open_pnl == null) return;
      byTs.set(s.ts, (byTs.get(s.ts) || 0) + s.open_pnl);
    });
    var xs = Array.from(byTs.keys()).sort();
    return { x: xs, y: xs.map(function (t) { return byTs.get(t); }) };
  }

  // Net position greeks over selected trades. Short legs flip the sign.
  // marks: {symbol: {delta, gamma, theta, vega, multiplier}}
  function netGreeks(positions, marks, selected) {
    var sel = selected ? new Set(selected) : null;
    var net = { delta: 0, gamma: 0, theta: 0, vega: 0 };
    (positions || []).forEach(function (p) {
      if (sel && !sel.has(p.trade_id)) return;
      (p.legs || []).forEach(function (leg) {
        var q = marks && marks[leg.symbol];
        if (!q) return;
        var sign = leg.side === "short" ? -1 : 1;
        var scale = sign * (leg.qty || 1) * (leg.multiplier || 100);
        net.delta += (q.delta || 0) * scale;
        net.gamma += (q.gamma || 0) * scale;
        net.theta += (q.theta || 0) * scale;
        net.vega += (q.vega || 0) * scale;
      });
    });
    return net;
  }

  global.Pnl = {
    aggregate: aggregate,
    curveFromHistory: curveFromHistory,
    netGreeks: netGreeks,
  };
})(typeof window !== "undefined" ? window : this);
