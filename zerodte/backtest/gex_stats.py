# zerodte/backtest/gex_stats.py
"""De-confound statistics for the GEX regime study (statsmodels not available)."""
from __future__ import annotations

from typing import Dict
import numpy as np
from scipy import stats


def ols(X: np.ndarray, y: np.ndarray):
    """Ordinary least squares with classical standard errors.
    X includes its own intercept column. Returns (beta, se, t, p).

    Uses pseudo-inverse for the covariance matrix so rank-deficient designs
    (e.g. perfect collinearity between regressors) return finite t-stats of 0
    rather than NaN.
    """
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    effective_rank = int(rank) if rank is not None else k
    dof = max(n - effective_rank, 1)
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(sigma2 * XtX_inv), 0.0))
    t = np.where(se > 0, beta / se, 0.0)
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, t, p


def tercile_buckets(em: np.ndarray) -> np.ndarray:
    """0/1/2 by EM tercile (rank-based, balanced)."""
    order = np.argsort(np.argsort(em))
    n = len(em)
    return np.minimum((order * 3) // n, 2)


def partial_effect(y: np.ndarray, regime: np.ndarray, em: np.ndarray) -> Dict:
    """Regress y ~ 1 + regime + em + regime*em; report the partial regime effect
    (coefficient, t-stat, p) holding EM fixed."""
    y = np.asarray(y, float); regime = np.asarray(regime, float); em = np.asarray(em, float)
    X = np.column_stack([np.ones_like(y), regime, em, regime * em])
    beta, se, t, p = ols(X, y)
    return {"regime_coef": float(beta[1]), "regime_se": float(se[1]),
            "regime_t": float(t[1]), "regime_p": float(p[1]),
            "interaction_coef": float(beta[3]), "interaction_t": float(t[3]),
            "n": int(len(y))}


def _sequential_folds(n: int, n_folds: int):
    """Contiguous, non-overlapping OOS blocks in time order (rows assumed date-sorted)."""
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(n_folds)]


def oos_partial_effect(y, regime, em, n_folds: int = 4) -> Dict:
    """Run partial_effect within each sequential OOS block. Rows MUST be date-sorted."""
    y = np.asarray(y, float); regime = np.asarray(regime, float); em = np.asarray(em, float)
    folds = []
    for k, (lo, hi) in enumerate(_sequential_folds(len(y), n_folds)):
        sl = slice(lo, hi)
        if hi - lo < 20:
            folds.append({"fold": k, "n": hi - lo, "regime_coef": float("nan"),
                          "regime_t": float("nan"), "regime_p": float("nan"), "thin": True})
            continue
        pe = partial_effect(y[sl], regime[sl], em[sl])
        pe["fold"] = k
        folds.append(pe)
    valid = [f for f in folds if not f.get("thin")]
    n_pos_sig = sum(1 for f in valid if f["regime_coef"] > 0 and f["regime_p"] < 0.1)
    n_neg_sig = sum(1 for f in valid if f["regime_coef"] < 0 and f["regime_p"] < 0.1)
    return {"folds": folds, "n_valid": len(valid),
            "n_pos_sig": n_pos_sig, "n_neg_sig": n_neg_sig,
            "consistent_majority": max(n_pos_sig, n_neg_sig) > len(valid) / 2}
