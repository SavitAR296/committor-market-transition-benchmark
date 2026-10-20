"""P2_F3 -- does the probability meaning of the raw score S drift with the annual re-estimation?

(i)   Event frequency of the operational event within pooled deciles of S, by model year.
(ii)  Saturated days (S > 0.99): days, events and rate per year; 90% interval from a cluster bootstrap that resamples
      whole episodes within the year (an episode = run of saturated days no more than h trading days apart, split at
      year boundaries because the model changes there).
(iii) Homogeneity of the saturated-day event rate across years (years with >= 5 saturated days).  The previous
      chi-square test treated autocorrelated days as independent and is replaced by
        - a cluster permutation test: year labels are permuted across episodes (episodes per year fixed),
          R = 9999, statistic = Pearson dispersion of the year rates;
        - a cluster-level quasi-binomial F test (episodes as binomial observations with common dispersion);
        - sensitivity: episode gap 2h, and fixed 20-trading-day blocks as clusters.
      The day-level chi-square is written once with valid = False so the change is traceable; it is not used.
(iv)  Real-time recalibration of S (strict: each year uses only prior years' OOS scores and labels, with the same
      memory as the forecasts: all prior years for 'exp', the last five for 'roll5'): logistic on S (the frozen
      seq-cal, reproduced and checked), logistic on logit S, isotonic on S, logistic on the rank of S within the same
      model's training scores (u_S), logistic on the rank of the stress coordinate s within its training values (u_s);
      and, for completeness, logistic on an annually standardised logit S, which uses the test year's own mean and
      standard deviation (look-ahead; labelled ex post, never a forecast).
(v)   Drift penalty: CORP discrimination with one monotone map for all years (pooled) versus a separate map per year
      (by year); a year-constant forecast gives the part due to year base rates alone.
Inputs: P2_F1_forecasts_<tag>.csv, P2_F1_anatomy_<tag>.csv, features_SPX_r1.csv (trading calendar).
Outputs: P2_F3_score_drift_<tag>.csv, P2_F3_saturation_<tag>.csv, P2_F3_homogeneity_<tag>.csv,
         P2_F3_calibrators_<tag>.csv, P2_F3_drift_penalty_<tag>.csv.   Seed 20261022.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import (TAB, FULL, FROZEN_END, MAIN_METHODS, bss, corp, corp_stratified_dsc, clusters_from_positions,
                       cluster_permutation_homogeneity, quasibinomial_F, write_csv)
from pre_common import sequential_recal                                  # frozen

SEED, R_PERM, B_CL = 20261022, 9999, 2000
SAMPLES = {'2008-2026': lambda W: W.year >= 2008, '2009-2026': lambda W: W.year >= 2009,
           '2008-2020': lambda W: W.index <= FROZEN_END, '2009-2020': lambda W: (W.year >= 2009) & (W.index <= FROZEN_END)}


def wide(F, h):
    g = F[(F.set == 'common7') & (F.horizon == h)]
    P = g.pivot_table(index='date', columns='method', values='p', aggfunc='first')
    meta = g.groupby('date').agg(y=('y', 'first'), cref=('clim', 'first'), year=('year', 'first'))
    return meta.join(P).sort_index()


def homogeneity_rows(h, sat, pos, rng):
    """sat: saturated headline days (columns y, year) with trading-day positions pos."""
    keep = sat.groupby('year').size().pipe(lambda s: s[s >= 5].index)
    sel = sat.year.isin(keep).values; q = sat[sel]; pq = pos[sel]
    ymap = {Y: k for k, Y in enumerate(sorted(q.year.unique()))}
    ev = q.groupby('year').y.sum(); nd = q.groupby('year').size()
    chi, p, dof, _ = chi2_contingency(np.c_[ev.values, (nd - ev).values] + 0.0)
    rows = [dict(horizon=h, test='day-level chi-square (independence assumed)', clusters='days', years=len(ymap),
                 n_clusters=len(q), stat=chi, df=dof, p_value=p, dispersion=np.nan, valid=False)]
    blocks = pd.factorize(pd.Series(q.year.values).astype(str) + '_' + pd.Series(pq // 20).astype(str))[0]
    for label, cid in [('episodes, gap h', clusters_from_positions(pq, h, q.year.values)),
                       ('episodes, gap 2h', clusters_from_positions(pq, 2 * h, q.year.values)),
                       ('20-day blocks', blocks)]:
        g = pd.DataFrame(dict(c=cid, y=q.y.values, year=q.year.values)).groupby('c').agg(
            e=('y', 'sum'), n=('y', 'size'), year=('year', 'first'))
        gi = g.year.map(ymap).values
        stat, pp = cluster_permutation_homogeneity(g.e.values, g.n.values, gi, R_PERM, rng)
        rows.append(dict(horizon=h, test='cluster permutation', clusters=label, years=len(ymap), n_clusters=len(g),
                         stat=stat, df=np.nan, p_value=pp, dispersion=np.nan, valid=True))
        F_, pF, phi, dfr = quasibinomial_F(g.e.values, g.n.values, gi)
        rows.append(dict(horizon=h, test='quasi-binomial F', clusters=label, years=len(ymap), n_clusters=len(g),
                         stat=F_, df=dfr, p_value=pF, dispersion=phi, valid=True))
    return rows


def calibrate(x, y, year, cref, kind, memory=None):
    p = np.full(len(x), np.nan)
    for Y in sorted(np.unique(year)):
        pr = (year < Y) & (year >= (Y - memory if memory else -10 ** 9)); cu = year == Y
        p[cu] = sequential_recal(x[pr], y[pr], x[cu], cref[cu][0], kind=kind)
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='exp'); a = ap.parse_args()
    memory = int(a.tag[4:]) if a.tag.startswith('roll') else None
    F = pd.read_csv(TAB / f'P2_F1_forecasts_{a.tag}.csv', parse_dates=['date'])
    F = F[F.headline.astype(bool) & F.y.notna() & F.p.notna()]
    A = pd.read_csv(TAB / f'P2_F1_anatomy_{a.tag}.csv', parse_dates=['date'])
    cal = pd.read_csv(TAB / 'features_SPX_r1.csv', parse_dates=['date']).dropna(subset=FULL).reset_index(drop=True)
    posmap = pd.Series(np.arange(len(cal)), index=cal.date)
    rng = np.random.default_rng(SEED)
    dr, sat_rows, hom, calr, pen = [], [], [], [], []
    for h in [5, 20]:
        W = wide(F, h); S = W.ou_raw.values; y = W.y.values; yr = W.year.values; cref = W.cref.values
        edges = np.unique(np.quantile(S, np.linspace(0, 1, 11)))                       # (i)
        k = np.clip(np.searchsorted(edges[1:-1], S, side='right'), 0, len(edges) - 2)
        for Y in sorted(np.unique(yr)):
            for b in range(len(edges) - 1):
                m = (yr == Y) & (k == b)
                if m.any():
                    dr.append(dict(horizon=h, year=Y, bin=b, S_lo=edges[b], S_hi=edges[b + 1], n=int(m.sum()),
                                   events=int(y[m].sum()), rate=float(y[m].mean())))
        hi = S > 0.99                                                                   # (ii)
        sat = pd.DataFrame(dict(y=y[hi], year=yr[hi]), index=W.index[hi]); pos = posmap.loc[sat.index].values
        cid = clusters_from_positions(pos, h, sat.year.values)
        for Y in sorted(sat.year.unique()):
            m = (sat.year == Y).values; yy = sat.y.values[m]
            g = pd.DataFrame(dict(c=cid[m], y=yy)).groupby('c').agg(e=('y', 'sum'), n=('y', 'size'))
            lo = hi90 = np.nan
            if len(g) >= 2:
                draws = rng.integers(0, len(g), size=(B_CL, len(g)))
                v = g.e.values[draws].sum(1) / g.n.values[draws].sum(1); lo, hi90 = np.quantile(v, [.05, .95])
            sat_rows.append(dict(horizon=h, year=Y, days=int(m.sum()), episodes=len(g), events=int(yy.sum()),
                                 rate=float(yy.mean()), lo90_episode_bootstrap=lo, hi90_episode_bootstrap=hi90))
        hom += homogeneity_rows(h, sat, pos, rng)                                       # (iii)
        Ah = A[A.horizon == h].set_index('date').reindex(W.index)                       # (iv)
        lg = np.log(np.clip(S, 1e-4, 1 - 1e-4) / (1 - np.clip(S, 1e-4, 1 - 1e-4))); zst = np.empty_like(lg)
        for Y in np.unique(yr):
            m = yr == Y; zst[m] = (lg[m] - lg[m].mean()) / (lg[m].std() + 1e-12)
        CAL = {'logistic on S (frozen seq-cal)': (S, 'logistic', True),
               'logistic on logit S': (S, 'logit_logistic', True),
               'isotonic on S': (S, 'isotonic', True),
               'logistic on u_S (rank of S in own training scores)': (Ah.u_S.values, 'logistic', True),
               'logistic on u_s (rank of stress coordinate in training)': (Ah.u_s.values, 'logistic', True),
               'logistic on annually standardised logit S (ex post, look-ahead)': (zst, 'logistic', False)}
        for name, (x, kind, realtime) in CAL.items():
            p = calibrate(np.asarray(x, float), y, yr, cref, kind, memory)
            if name.startswith('logistic on S'):
                dev = float(np.max(np.abs(p - W.ou_seqcal.values)))
                assert dev < 1e-10, f'frozen seq-cal not reproduced ({dev})'
            for sname, sel in SAMPLES.items():
                m = np.asarray(sel(W)); c = corp(y[m], p[m])
                calr.append(dict(horizon=h, calibrator=name, real_time=realtime, sample=sname, BSS=bss(y[m], p[m], cref[m]),
                                 MCB=c['MCB'], DSC=c['DSC'], UNC=c['UNC']))
        for sname, sel in SAMPLES.items():                                              # (v)
            m = np.asarray(sel(W)); ybar = y[m].mean()
            yeff = float(sum((yr[m] == Y).sum() * (y[m][yr[m] == Y].mean() - ybar) ** 2 for Y in np.unique(yr[m])) / m.sum())
            for meth in [x for x in MAIN_METHODS if x in W] + ['clim']:
                p = W[meth].values[m]
                pen.append(dict(horizon=h, sample=sname, method=meth, DSC_pooled=corp(y[m], p)['DSC'],
                                DSC_by_year=corp_stratified_dsc(y[m], p, yr[m]), year_base_rate_part=yeff))
    D = pd.DataFrame(pen); D['penalty'] = D.DSC_by_year - D.DSC_pooled
    write_csv(pd.DataFrame(dr), f'P2_F3_score_drift_{a.tag}.csv'); write_csv(pd.DataFrame(sat_rows), f'P2_F3_saturation_{a.tag}.csv')
    write_csv(pd.DataFrame(hom), f'P2_F3_homogeneity_{a.tag}.csv'); write_csv(pd.DataFrame(calr), f'P2_F3_calibrators_{a.tag}.csv')
    write_csv(D, f'P2_F3_drift_penalty_{a.tag}.csv')
    print(pd.DataFrame(hom).round(4).to_string(index=False))
    C = pd.DataFrame(calr)
    print(C[C['sample'].isin(['2008-2026', '2009-2026'])].pivot_table(index='calibrator', columns=['horizon', 'sample'], values='BSS').round(3).to_string())
    print(D[D['sample'].isin(['2008-2026', '2009-2026'])].pivot_table(index='method', columns=['horizon', 'sample'], values='penalty').round(4).to_string())
    print('year base-rate part:', D.groupby(['horizon', 'sample']).year_base_rate_part.first().round(4).to_dict())
    print('F3', a.tag, 'complete')


if __name__ == '__main__':
    main()
