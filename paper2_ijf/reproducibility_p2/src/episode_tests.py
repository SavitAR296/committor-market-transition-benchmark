"""episode_tests -- reusable episode-level tests of whether the event rate on 'alarm' days changes across periods.

    from episode_tests import episode_tests
    out = episode_tests(y, score, years, h, threshold=0.99)

y: 0/1 outcomes (event within h steps) on scored days, in time order; score: forecast; years: period label per day;
h: forecast horizon in steps.  Alarm days (score > threshold) are grouped into episodes -- runs no more than h steps
apart, split at period boundaries -- and periods with fewer than `min_days` alarm days are dropped.  Returns the
day-level chi-square p-value (valid only for independent days, reported for comparison), the episode permutation
p-value (period labels permuted across whole episodes) and the quasi-binomial F p-value with its dispersion.
Same code as P2_F3_drift.py (paper 2, Section 5 and Table 2); see P2_S1_size_simulation.py for its size.
"""
import numpy as np
from scipy import stats
from p2_common import clusters_from_positions, cluster_permutation_homogeneity, quasibinomial_F


def episode_tests(y, score, years, h, threshold=0.99, min_days=5, R=999, seed=0):
    y, score, years = np.asarray(y, int), np.asarray(score, float), np.asarray(years)
    pos = np.flatnonzero(score > threshold); yr = years[pos]
    keep = [v for v in np.unique(yr) if (yr == v).sum() >= min_days]; pos = pos[np.isin(yr, keep)]; yr = years[pos]
    cid = clusters_from_positions(pos, h, yr)
    e = np.bincount(cid, y[pos]); n = np.bincount(cid); g = np.searchsorted(keep, np.array([yr[cid == c][0] for c in range(cid.max() + 1)]))
    E = np.bincount(g, e); N = np.bincount(g, n); p = E.sum() / N.sum()
    chi = float(stats.chi2.sf(np.sum((E - N * p) ** 2 / (N * p * (1 - p))), len(keep) - 1)) if 0 < p < 1 else 1.0
    perm = cluster_permutation_homogeneity(e, n, g, R, np.random.default_rng(seed))[1]
    F, pF, phi, _ = quasibinomial_F(e, n, g)
    return dict(days=int(n.sum()), episodes=len(e), periods=len(keep), p_day_chisq=chi, p_episode_permutation=perm,
                p_quasibinomial=pF, dispersion=phi)
