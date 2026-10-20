"""E5 -- Murphy decomposition (fixed 10 bins and decile bins), reliability curves, annual AUC, moving-block bootstrap
of Delta BSS (blocks 5-60 days, B = 500, seed 20260915 as in the frozen package) and Giacomini-White (2006) tests.
Samples: 2008-2020 and 2009-2020 (frozen forecast file), 2008-2026 and 2009-2026 (E3 extended forecasts).
GW: loss differential d_t = (y-p_OU)^2 - (y-p_cmp)^2 (positive = comparator better). Unconditional test = DM-type
t statistic with Newey-West HAC (lags h-1 and 20); conditional test with instruments (1, d_{t-h}) [the last loss
differential realised at the forecast origin], chi^2(2) with Newey-West lag h-1.  Days are taken in date order.
Outputs: E5_murphy.csv, E5_reliability.csv, E5_annual_auc.csv, E5_bootstrap.csv, E5_gw_tests.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2, norm
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from pre_common import load_forecasts, wide, murphy, reliability_curve, auc, bss, write_csv
SEED, B, BLOCKS = 20260915, 500, [5, 10, 20, 40, 60]
CMP = ['pc2_logistic', 'pc2_gbt', 'raw_logistic', 'msm_seqcal', 'amc_seqcal']
REL_METHODS = ['ou_raw', 'ou_seqcal', 'msm_seqcal', 'amc_seqcal', 'pc2_logistic', 'pc2_gbt', 'raw_logistic']


def samples():
    Fz = load_forecasts()
    E = pd.read_csv(ROOT / 'tables_pre' / 'E3_forecasts_2026.csv', parse_dates=['date'])
    E = E[E.headline.astype(bool) & E.y.notna() & E.p.notna()]
    return {'2008-2020': Fz, '2009-2020': Fz[Fz.year >= 2009], '2008-2026': E, '2009-2026': E[E.year >= 2009]}


def nw(Z, q):
    Z = np.atleast_2d(Z.T).T; Zc = Z - Z.mean(0); n = len(Z); S = Zc.T @ Zc / n
    for k in range(1, q + 1):
        G = Zc[k:].T @ Zc[:-k] / n; S += (1 - k / (q + 1)) * (G + G.T)
    return S


def mbb(n, L, rng):
    k = (n + L - 1) // L; st = rng.integers(0, max(1, n - L + 1), size=(B, k))
    return np.minimum((st[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n], n - 1)


def main():
    Sm = samples(); mur, rel, boot, gw = [], [], [], []
    rng = np.random.default_rng(SEED)
    for sname, F in Sm.items():
        for s in sorted(F.set.unique()):
            for h in [5, 20]:
                W = wide(F, s, h); meths = [m for m in W.columns if m not in ('y', 'cref', 'year')]
                for m in meths:
                    ok = W[m].notna()
                    for bins in ['fixed10', 'quantile10']:
                        d = murphy(W.y[ok].values, W[m][ok].values, bins=bins)
                        d.update(sample=sname, set=s, horizon=h, method=m, binning=bins,
                                 BSS=bss(W.y[ok], W[m][ok], W.cref[ok]), AUC=auc(W.y[ok], W[m][ok])); mur.append(d)
                if s != 'common7':
                    continue
                if sname in ('2008-2020', '2008-2026'):
                    for m in [x for x in REL_METHODS if x in W]:
                        r = reliability_curve(W.y.values, W[m].values, 10); r['sample'] = sname; r['horizon'] = h; r['method'] = m
                        rel.append(r)
                y = W.y.values; base = (y - W.cref.values) ** 2
                for cm in [c for c in CMP if c in W]:
                    dlt = (y - W.ou_seqcal.values) ** 2 - (y - W[cm].values) ** 2; n = len(dlt)
                    point = dlt.mean() / base.mean()
                    for L in BLOCKS:
                        ix = mbb(n, L, rng); v = dlt[ix].mean(1) / base[ix].mean(1)
                        boot.append(dict(sample=sname, horizon=h, comparator=cm, block=L, dBSS=point,
                                         lo95=np.quantile(v, .025), hi95=np.quantile(v, .975),
                                         p_sign=min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0)))))
                    for q in sorted({h - 1, 20}):
                        t = dlt.mean() / np.sqrt(nw(dlt[:, None], q)[0, 0] / n)
                        gw.append(dict(sample=sname, horizon=h, comparator=cm, test='unconditional', nw_lag=q, stat=float(t),
                                       df=1, p_value=float(2 * (1 - norm.cdf(abs(t)))), mean_dL=float(dlt.mean())))
                    Z = np.c_[dlt[h:], dlt[h:] * dlt[:-h]]; Zb = Z.mean(0); Om = nw(Z, h - 1)
                    stat = float(len(Z) * Zb @ np.linalg.solve(Om, Zb))
                    gw.append(dict(sample=sname, horizon=h, comparator=cm, test='conditional (1, d_{t-h})', nw_lag=h - 1,
                                   stat=stat, df=2, p_value=float(1 - chi2.cdf(stat, 2)), mean_dL=float(dlt.mean())))
    M = pd.DataFrame(mur)
    write_csv(M[['sample', 'set', 'horizon', 'method', 'binning', 'n', 'events', 'BS', 'REL', 'RES', 'UNC', 'WBV', 'WBC',
                 'identity_residual', 'BSS', 'AUC']], 'E5_murphy.csv')
    write_csv(pd.concat(rel, ignore_index=True), 'E5_reliability.csv')
    write_csv(pd.DataFrame(boot), 'E5_bootstrap.csv'); write_csv(pd.DataFrame(gw), 'E5_gw_tests.csv')
    E = Sm['2008-2026']; E = E[E.set == 'common7']; ann = []
    for (h, m, Y), g in E.groupby(['horizon', 'method', 'year']):
        ann.append(dict(horizon=h, method=m, year=Y, n=len(g), events=int(g.y.sum()), AUC=auc(g.y, g.p),
                        constant_forecast=bool(np.ptp(g.p.values) < 1e-12)))
    write_csv(pd.DataFrame(ann), 'E5_annual_auc.csv')
    assert M.identity_residual.abs().max() < 1e-6
    print('E5 complete'); (ROOT / 'tables_pre' / '_cache' / 'DONE_E5_murphy_inference').touch()


if __name__ == '__main__':
    main()
