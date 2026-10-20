"""P2_S1 -- size and power of homogeneity tests for the saturated-day event rate when days come in episodes.

Design, fixed before the results (seed 20261027).  For each horizon h the simulation copies the headline sample of
Table 2: K model years with at least five saturated days, E episodes, D scored saturated days and a pooled event rate r
(read from P2_F3_saturation_exp.csv).  Every year has 1 + Poisson(E/K - 1) saturation episodes of geometric length
(mean calibrated so that the mean number of scored days per episode is D/E).  With probability pi_y an episode is
followed by an operational entry at a time tau drawn uniformly from its days 1, ..., len + h; a saturated day t is
labelled 1 if tau lies in (t, t + h], and days from tau on are not scored (the state is then in the stress set), as in
the paper.  Null: pi_y = pi for every year, pi calibrated to the pooled rate r.  Alternatives: logit pi_y = logit pi +
delta z_y, z_y ~ N(0, 1), delta = 0.5 and 1.  Tests as in P2_F3 (years with >= 5 saturated days): day-level Pearson
chi-square (days assumed independent), cluster permutation with episodes as units (R = 499) and the quasi-binomial F
test on episode counts.  Output: tables_p2/P2_S1_size_simulation.csv.  2000 replications per cell.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import TAB, cluster_permutation_homogeneity, quasibinomial_F, write_csv
SEED, NREP, R_PERM, DELTAS = 20261027, 2000, 499, (0.0, 0.5, 1.0)


def episodes(rng, Lmean, pi, h, n):
    L = rng.geometric(1 / Lmean, n); hit = rng.random(n) < pi; e = np.zeros(n, int); m = L.copy()
    tau = rng.integers(1, L + h + 1)
    m[hit] = np.minimum(L[hit], tau[hit])                        # days from tau on are in the set: not scored
    e[hit] = np.clip(np.minimum(tau[hit], L[hit]) - np.maximum(tau[hit] - h, 0), 0, None)
    return e, m


def calibrate(rng, Dbar, rate, h):
    Lm, lp = Dbar, logit(min(max(rate, 0.05), 0.95))
    for _ in range(40):
        e, m = episodes(rng, Lm, expit(lp), h, 40000)
        Lm *= Dbar / m.mean(); lp += logit(rate) - logit(e.sum() / m.sum())
    return Lm, expit(lp)


def chisq(e, n, g):
    K = g.max() + 1; E = np.bincount(g, e, K); N = np.bincount(g, n, K); p = E.sum() / N.sum()
    if p <= 0 or p >= 1:
        return 1.0
    return float(stats.chi2.sf(np.sum((E - N * p) ** 2 / (N * p * (1 - p))), K - 1))


def main():
    S = pd.read_csv(TAB / 'P2_F3_saturation_exp.csv'); rows = []
    for h in (5, 20):
        s = S[(S.horizon == h) & (S.days >= 5)]; K, E, D, V = len(s), int(s.episodes.sum()), int(s.days.sum()), int(s.events.sum())
        rng = np.random.default_rng([SEED, h]); Lm, pi = calibrate(rng, D / E, V / D, h)
        for delta in DELTAS:
            rej = {'day-level chi-square': [], 'episode permutation': [], 'quasi-binomial F': []}; units = []
            for _ in range(NREP):
                piy = expit(logit(pi) + delta * rng.standard_normal(K)); ee, nn, gg = [], [], []
                for y in range(K):
                    k = 1 + rng.poisson(max(E / K - 1, 0)); e, m = episodes(rng, Lm, piy[y], h, k)
                    ee.append(e); nn.append(m); gg.append(np.full(k, y))
                ee, nn, gg = np.concatenate(ee), np.concatenate(nn), np.concatenate(gg)
                tot = np.bincount(gg, nn, K); keep = np.flatnonzero(tot >= 5)
                if len(keep) < 2:
                    continue
                sel = np.isin(gg, keep) & (nn > 0); e, n = ee[sel], nn[sel]; g = np.searchsorted(keep, gg[sel])
                units.append((len(e), n.sum(), len(keep)))
                rej['day-level chi-square'].append(chisq(e, n, g) < 0.05)
                rej['episode permutation'].append(cluster_permutation_homogeneity(e, n, g, R_PERM, rng)[1] < 0.05)
                pF = quasibinomial_F(e, n, g)[1]
                rej['quasi-binomial F'].append(bool(pF < 0.05) if np.isfinite(pF) else False)
            u = np.array(units)
            for t, v in rej.items():
                v = np.array(v, float)
                rows.append(dict(horizon=h, delta=delta, test=t, rejection_rate=v.mean(), mc_se=np.sqrt(v.mean() * (1 - v.mean()) / len(v)),
                                 n_valid=len(v), mean_episodes=u[:, 0].mean(), mean_days=u[:, 1].mean(), mean_years=u[:, 2].mean(),
                                 target_episodes=E, target_days=D, target_years=K, pi=pi, mean_length=Lm))
            print(h, delta, {t: round(float(np.mean(v)), 3) for t, v in rej.items()}, flush=True)
    write_csv(pd.DataFrame(rows), 'P2_S1_size_simulation.csv')


if __name__ == '__main__':
    main()
