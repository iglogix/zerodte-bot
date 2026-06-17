// Payoff / risk-graph math — pure functions (no DOM, no I/O).
// Black-Scholes (r=0) for the "now" curve; intrinsic value for the "expiry" curve.
// Exposed on window.Payoff so tests.html and app.js can use them.
(function (global) {
  "use strict";

  function normCdf(x) {
    // Abramowitz & Stegun 7.1.26 approximation of the error function.
    var t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
    var y =
      1 -
      (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
        0.284496736) *
        t +
        0.254829592) *
        t *
        Math.exp((-x * x) / 2);
    var cdfAbs = 0.5 * (1 + y);
    return x >= 0 ? cdfAbs : 1 - cdfAbs;
  }

  // Black-Scholes price of one option (r = 0).
  function bsPrice(spot, strike, tYears, iv, isCall) {
    var t = Math.max(tYears, 1e-6);
    var sig = Math.max(iv, 1e-4);
    var sqrtT = Math.sqrt(t);
    var d1 = (Math.log(spot / strike) + 0.5 * sig * sig * t) / (sig * sqrtT);
    var d2 = d1 - sig * sqrtT;
    var price = isCall
      ? spot * normCdf(d1) - strike * normCdf(d2)
      : strike * normCdf(-d2) - spot * normCdf(-d1);
    return Math.max(price, 0);
  }

  function intrinsic(spot, strike, isCall) {
    return isCall ? Math.max(spot - strike, 0) : Math.max(strike - spot, 0);
  }

  // Value of one leg at a hypothetical spot, under "now" (BS) or "expiry" (intrinsic).
  function legValueAtSpot(leg, spot, tYears, mode, ivLookup) {
    var isCall = leg.right === "call";
    if (mode === "expiry") return intrinsic(spot, leg.strike, isCall);
    var iv = (ivLookup && ivLookup(leg.symbol)) || 0.16;
    return bsPrice(spot, leg.strike, tYears, iv, isCall);
  }

  // P&L of one position at a hypothetical spot (consistent with open_pnl_for_position).
  // pnl = sum over spreads of (credit - (shortVal - longVal)) * qty * multiplier.
  function positionPayoffAt(position, spot, tYears, mode, ivLookup) {
    var total = 0;
    (position.spreads || []).forEach(function (sp) {
      var mult = (sp.short || sp.long || {}).multiplier || 100;
      var shortVal = sp.short ? legValueAtSpot(sp.short, spot, tYears, mode, ivLookup) : 0;
      var longVal = sp.long ? legValueAtSpot(sp.long, spot, tYears, mode, ivLookup) : 0;
      total += (sp.credit - (shortVal - longVal)) * sp.qty * mult;
    });
    return total;
  }

  // Summed payoff curve across selected positions over a spot grid.
  function sumPayoff(positions, spots, tYears, mode, ivLookup) {
    return spots.map(function (spot) {
      var y = 0;
      positions.forEach(function (p) {
        y += positionPayoffAt(p, spot, tYears, mode, ivLookup);
      });
      return y;
    });
  }

  // Build a centered spot grid: center ± pct, n points.
  function spotGrid(center, pct, n) {
    pct = pct || 0.03;
    n = n || 81;
    var lo = center * (1 - pct);
    var hi = center * (1 + pct);
    var step = (hi - lo) / (n - 1);
    var out = [];
    for (var i = 0; i < n; i++) out.push(lo + i * step);
    return out;
  }

  global.Payoff = {
    normCdf: normCdf,
    bsPrice: bsPrice,
    intrinsic: intrinsic,
    legValueAtSpot: legValueAtSpot,
    positionPayoffAt: positionPayoffAt,
    sumPayoff: sumPayoff,
    spotGrid: spotGrid,
  };
})(typeof window !== "undefined" ? window : this);
