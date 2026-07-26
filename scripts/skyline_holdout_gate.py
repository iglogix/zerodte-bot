#!/usr/bin/env python3
"""Sanity gate for the untouched holdout cache (same checks as the main study)."""
import pickle, sqlite3, statistics as st, sys
from datetime import date
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from backtest_skyline import lookup, parity_spot  # noqa

CACHE = ROOT / "data" / "cbbo_cache_skyline_holdout"
con = sqlite3.connect(ROOT / "backtest_data" / "skyline.db")
g = {d: c for d, c in con.execute("SELECT d, close FROM gspc_daily")}
con.close()

errs, spreads, n = [], [], 0
mins_all, span = None, []
for f in sorted(CACHE.glob("*.pkl")):
    d = date.fromisoformat(f.stem); q = pickle.load(open(f, "rb"))
    ks = sorted({k[1] for k in q}); mins = sorted({k[0] for k in q})
    mins_all = (mins[0], mins[-1]); span.append(ks[-1] - ks[0])
    s45 = parity_spot(q, "09:45", ks)
    last = max(m for m in mins)
    s_last = parity_spot(q, last, ks)
    if s45 is None or s_last is None: continue
    n += 1
    errs.append(s_last - g[d.isoformat()])
    atm = round(s45 / 5) * 5
    for off in (-80, -45, -20, 0, 25, 50, 75):
        for c in (True, False):
            v = lookup(q, "09:45", atm + off, c)
            if v and v[1] > 0: spreads.append(v[1] - v[0])
print(f"holdout: {n} sessions, minutes {mins_all}, strike span {min(span):.0f}-{max(span):.0f}pt")
print(f"parity(last min) - ^GSPC close: mean {st.mean(errs):+.2f}  median {st.median(errs):+.2f}  "
      f"sd {st.pstdev(errs):.2f}  max|e| {max(abs(x) for x in errs):.2f}")
print(f"per-leg quoted spread: median {st.median(spreads):.2f}pt  "
      f"p95 {sorted(spreads)[19*len(spreads)//20]:.2f}pt  (main study median 0.10pt)")
