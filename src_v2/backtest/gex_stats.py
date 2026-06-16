# src_v2/backtest/gex_stats.py
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
