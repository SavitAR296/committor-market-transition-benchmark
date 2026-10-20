"""P2_B2 -- two checks requested in review.
(a) Stress-index baseline: the stress coordinate s (projection of the state on the learned direction) recalibrated by
    exactly the strict sequential logistic map of the committor score (seq-cal), expanding windows.  If the fitted OU
    dynamics add information beyond s, OU seq-cal should beat s seq-cal.
(b) OU seq-cal against MSM/AMC seq-cal at h = 20: the Wald test of the lagged loss differential alone in
    d_t = a + b d_{t-h} + e_t (Newey-West, lag h - 1), and the Giacomini-Rossi (2010) fluctuation test (rolling window
    of a share mu = 0.3 of the evaluation sample; two-sided 5% critical value simulated from the Brownian limit).
Inputs: P2_F1_anatomy_exp.csv, P2_F1_forecasts_exp.csv (and _roll5).  Outputs: P2_B2_stress_index.csv,
P2_B5_gw_split.csv, P2_B5_fluctuation.csv.  Seed 20261028.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT.parents[1] / 'paper1_pre' / 'reproducibility_pre' / 'src'))
from p2_common import TAB, bss, corp, mbb_index, boot_block, dm_tests, nw, write_csv
from pre_common import sequential_recal
SEED, MU = 20261028, 0.3
SAMPLES = {'2008-2026': 2008, '2009-2026': 2009}


def wide(F, h):
    g = F[(F.set == 'common7') & (F.horizon == h)]
    P = g.pivot_table(index='date', columns='method', values='p', aggfunc='first')
    return g.groupby('date').agg(y=('y', 'first'), cref=('clim', 'first'), year=('year', 'first')).join(P).sort_index()


def nw_ols(y, X, L):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b; Z = X * u[:, None]; S = Z.T @ Z
    for j in range(1, L + 1):
        G = Z[j:].T @ Z[:-j]; S += (1 - j / (L + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X); return b, Q @ S @ Q


def gr_critical(rng, mu, n_paths=20000, n_grid=1000):
    W = np.cumsum(rng.standard_normal((n_paths, n_grid)) / np.sqrt(n_grid), 1); m = int(round(mu * n_grid))
    W = np.c_[np.zeros(n_paths), W]; stat = np.abs(W[:, m:] - W[:, :-m]).max(1) / np.sqrt(mu)
    return float(np.quantile(stat, 0.95))


def main():
    F = pd.read_csv(TAB / 'P2_F1_forecasts_exp.csv', parse_dates=['date']); F = F[F.headline.astype(bool) & F.y.notna() & F.p.notna()]
    A = pd.read_csv(TAB / 'P2_F1_anatomy_exp.csv', parse_dates=['date'])
    rows, split, fluc = [], [], []
    rng = np.random.default_rng(SEED)
    for h in (5, 20):
        W = wide(F, h); a = A[A.horizon == h].set_index('date')
        W = W.join(a[['s']], how='left'); W = W[W.s.notna()]
        cal = []
        for Y in sorted(W.year.unique()):                         # strict sequential map, exactly as for the committor
            prior = W[W.year < Y]; cur = W[W.year == Y]
            cal.append(pd.Series(sequential_recal(prior.s.values, prior.y.values, cur.s.values, float(cur.cref.iloc[0])), index=cur.index))
        W['s_seqcal'] = pd.concat(cal)
        for sname, y0 in SAMPLES.items():
            V = W[W.year >= y0]; y = V.y.values; base = (y - V.cref.values) ** 2
            for m in ['ou_seqcal', 's_seqcal']:
                c = corp(y, V[m].values)
                rows.append(dict(horizon=h, sample=sname, method=m, n=len(y), BSS=bss(y, V[m].values, V.cref.values), MCB=c['MCB'], DSC=c['DSC']))
            d = (y - V.ou_seqcal.values) ** 2 - (y - V.s_seqcal.values) ** 2      # positive: s seq-cal better
            ix = mbb_index(len(y), boot_block(h), 1000, rng); v = d[ix].mean(1) / base[ix].mean(1)
            p = next(t['p_value'] for t in dm_tests(d, h) if t['variance'].startswith('EWC'))
            rows.append(dict(horizon=h, sample=sname, method='dBSS s_seqcal - ou_seqcal', n=len(y), BSS=d.mean() / base.mean(),
                             lo90=np.quantile(v, .05), hi90=np.quantile(v, .95), p_EWC=p))
    write_csv(pd.DataFrame(rows), 'P2_B2_stress_index.csv'); print(pd.DataFrame(rows).round(4).to_string(index=False))
    k95 = gr_critical(rng, MU)
    for tag in ['exp', 'roll5']:
        Ft = F if tag == 'exp' else pd.read_csv(TAB / 'P2_F1_forecasts_roll5.csv', parse_dates=['date'])
        Ft = Ft[Ft.headline.astype(bool) & Ft.y.notna() & Ft.p.notna()]; W = wide(Ft, 20)
        for sname, y0 in SAMPLES.items():
            V = W[W.year >= y0]; y = V.y.values
            for cm in ['msm_seqcal', 'amc_seqcal']:
                d = (y - V.ou_seqcal.values) ** 2 - (y - V[cm].values) ** 2; h = 20
                X = np.c_[np.ones(len(d) - h), d[:-h]]; b, Vb = nw_ols(d[h:], X, h - 1)
                t_b = b[1] / np.sqrt(Vb[1, 1]); t_a = b[0] / np.sqrt(Vb[0, 0])
                split.append(dict(window=tag, sample=sname, comparator=cm, slope=b[1], p_slope=float(2 * stats.norm.sf(abs(t_b))),
                                  const=b[0], p_const=float(2 * stats.norm.sf(abs(t_a)))))
                P = len(d); m = int(round(MU * P)); sig = np.sqrt(nw(d, h - 1)[0, 0])
                roll = np.convolve(d, np.ones(m), 'valid') / (sig * np.sqrt(m)); dates = V.index[m - 1:]
                out = np.abs(roll) > k95
                fluc.append(dict(window=tag, sample=sname, comparator=cm, mu=MU, window_days=m, k95=k95, max_abs=float(np.abs(roll).max()),
                                 reject=bool(out.any()), share_outside=float(out.mean()),
                                 first_outside=str(dates[out][0].date()) if out.any() else '', sign_at_max=float(np.sign(roll[np.abs(roll).argmax()]))))
    write_csv(pd.DataFrame(split), 'P2_B5_gw_split.csv'); write_csv(pd.DataFrame(fluc), 'P2_B5_fluctuation.csv')
    print(pd.DataFrame(split).round(4).to_string(index=False)); print(pd.DataFrame(fluc).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
