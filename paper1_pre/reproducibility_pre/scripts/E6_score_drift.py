"""E6 -- drift of the probability meaning of the raw hitting score S (common-7, E3 forecasts 2008-2026).
(i) r*_Y(S): event frequency within pooled deciles of S, by year;  (ii) days with S > 0.99: counts, events, rate and a
90% moving-block bootstrap interval (block 5 days, B = 2000, seed 20260916) per year, plus a chi-square test of
homogeneity of that rate across years with >= 5 such days;  (iii) whether an annually standardised logit(S)
(location/scale removed within each year, an ex-post transform) improves strict sequential recalibration.
Outputs: E6_score_drift.csv, E6_saturation.csv, E6_recal_tests.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from pre_common import wide, sequential_recal, bss, write_csv
SEED = 20260916


def main():
    E = pd.read_csv(ROOT / 'tables_pre' / 'E3_forecasts_2026.csv', parse_dates=['date'])
    E = E[E.headline.astype(bool) & E.y.notna() & E.p.notna()]
    rng = np.random.default_rng(SEED); dr, sat, rec, hom = [], [], [], []
    for h in [5, 20]:
        W = wide(E, 'common7', h); S = W.ou_raw.values; y = W.y.values
        edges = np.unique(np.quantile(S, np.linspace(0, 1, 11)))
        k = np.clip(np.searchsorted(edges[1:-1], S, side='right'), 0, len(edges) - 2)
        for Y in sorted(W.year.unique()):
            m = (W.year == Y).values
            for b in range(len(edges) - 1):
                mm = m & (k == b)
                if mm.sum():
                    dr.append(dict(horizon=h, year=Y, bin=b, S_lo=edges[b], S_hi=edges[b + 1], S_mean=S[mm].mean(),
                                   n=int(mm.sum()), events=int(y[mm].sum()), rate=y[mm].mean()))
            hi = m & (S > 0.99); yy = y[hi]; n = len(yy)
            if n:
                if n >= 5:
                    L = 5; kk = (n + L - 1) // L; st = rng.integers(0, max(1, n - L + 1), size=(2000, kk))
                    ix = np.minimum((st[:, :, None] + np.arange(L)).reshape(2000, -1)[:, :n], n - 1)
                    v = yy[ix].mean(1); lo, hi_ = np.quantile(v, [.05, .95])
                else:
                    lo = hi_ = np.nan
                sat.append(dict(horizon=h, year=Y, days=n, events=int(yy.sum()), rate=float(yy.mean()), lo90=lo, hi90=hi_))
        T = pd.DataFrame([r for r in sat if r['horizon'] == h and r['days'] >= 5])
        if len(T) >= 2:
            chi, p, dof, _ = chi2_contingency(np.c_[T.events, T.days - T.events] + 0.0)
            hom.append(dict(horizon=h, years=len(T), chi2=chi, dof=dof, p_value=p))
        lg = np.log(np.clip(S, 1e-4, 1 - 1e-4) / (1 - np.clip(S, 1e-4, 1 - 1e-4)))
        z = np.empty_like(lg)
        for Y in W.year.unique():
            m = (W.year == Y).values; z[m] = (lg[m] - lg[m].mean()) / (lg[m].std() + 1e-12)
        for kind, x in [('logistic on S (frozen)', S), ('logistic on annually standardised logit S', z)]:
            p = np.full(len(S), np.nan)
            for Y in sorted(W.year.unique()):
                pr = (W.year < Y).values; cu = (W.year == Y).values
                p[cu] = sequential_recal(x[pr], y[pr], x[cu], W.cref[cu].iloc[0])
            for smp, sel in [('2008-2026', np.ones(len(S), bool)), ('2009-2026', (W.year >= 2009).values),
                             ('2008-2020', (W.index <= pd.Timestamp('2020-04-17'))), ('2009-2020', (W.year >= 2009).values & (W.index <= pd.Timestamp('2020-04-17')))]:
                rec.append(dict(horizon=h, calibrator=kind, sample=smp, BSS=bss(y[sel], p[sel], W.cref.values[sel])))
    write_csv(pd.DataFrame(dr), 'E6_score_drift.csv'); write_csv(pd.DataFrame(sat), 'E6_saturation.csv')
    write_csv(pd.DataFrame(rec), 'E6_recal_tests.csv'); write_csv(pd.DataFrame(hom), 'E6_homogeneity.csv')
    print('E6 complete'); (ROOT / 'tables_pre' / '_cache' / 'DONE_E6_score_drift').touch()


if __name__ == '__main__':
    main()
