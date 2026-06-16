# tests/test_gex_stats.py
import numpy as np
from src_v2.backtest.gex_stats import ols, tercile_buckets, partial_effect

def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.1, size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    beta, se, t, p = ols(X, y)
    assert abs(beta[0] - 1.0) < 0.05
    assert abs(beta[1] - 2.0) < 0.05
    assert abs(beta[2] + 0.5) < 0.05
    assert p[1] < 1e-6   # x1 highly significant

def test_tercile_buckets_split_three_ways():
    em = np.arange(30.0)
    b = tercile_buckets(em)
    assert set(b) == {0, 1, 2}
    assert (b == 0).sum() == (b == 2).sum() == 10

def test_partial_effect_zero_when_regime_is_pure_vol_proxy():
    # regime perfectly collinear with EM -> partial effect ~0 after controlling EM
    rng = np.random.default_rng(1)
    n = 1500
    em = rng.uniform(20, 60, n)
    regime = (em - em.mean()) / em.std()          # regime IS vol, nothing more
    y = 0.3 * (em / em.mean()) + rng.normal(scale=0.05, size=n)
    res = partial_effect(y, regime, em)
    assert abs(res["regime_t"]) < 3.0             # not robustly significant

from src_v2.backtest.gex_stats import oos_partial_effect

def test_oos_partial_effect_reports_per_fold():
    rng = np.random.default_rng(2)
    n = 1000
    regime = rng.normal(size=n)
    em = rng.uniform(20, 60, n)
    y = 0.4 * regime + 0.2 * (em / em.mean()) + rng.normal(scale=0.05, size=n)  # real regime effect
    res = oos_partial_effect(y, regime, em, n_folds=4)
    assert len(res["folds"]) == 4
    # a genuine positive effect should appear in a majority of OOS folds
    assert sum(1 for f in res["folds"] if f["regime_coef"] > 0 and f["regime_p"] < 0.1) >= 3
