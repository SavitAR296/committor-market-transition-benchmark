"""P2_F4 -- matched comparisons of the committor forecast with the other estimators.

Reference: ou_seqcal (the committor score after strict sequential recalibration).  Comparators: msm_seqcal,
amc_seqcal, pc2_logistic, pc2_gbt, raw_logistic, raw_gbt.  Blocks: horizon 5 and 20 days x with and without 2008
(main: 2008-2026, 2009-2026; frozen period 2008-2020, 2009-2020 as a check) x expanding and rolling five-year windows.
Loss differential d_t = (y - p_ref)^2 - (y - p_cmp)^2 (positive: the comparator is better);
dBSS = mean(d) / mean((y - clim)^2) = BSS(cmp) - BSS(ref), 90% moving-block-bootstrap intervals (blocks 5-60, B = 1000).
Tests of E d = 0: Newey-West (lags h-1 and 20), EWC fixed-b (t_nu; primary), and the Giacomini-White conditional test
with instruments (1, d_{t-h}) (NW chi2 and EWC F).  GW asymptotics need a finite estimation window: the test is
formally valid for the rolling forecasts; on the expanding forecasts it is a conditional DM-type test of the forecasts.
Holm adjustment of the primary p-values over the six comparators within each family (window, horizon, sample).
Model confidence set (T_max, blocks of 2h days, B = 1000, alpha = 0.10) over reference, comparators and climatology.
Effective sample: event episodes (event days no more than h trading days apart).
Inputs: P2_F1_forecasts_<exp|roll5>.csv.  Outputs: P2_F4_tests.csv, P2_F4_dbss.csv, P2_F4_mcs.csv, P2_F4_summary.csv.
Seed 20261023.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import (TAB, FULL, FROZEN_END, dm_tests, gw_conditional, holm, mbb_index, boot_block, mcs, clusters_from_positions,
                       write_csv)

SEED, B, B_MCS, BLOCKS = 20261023, 1000, 1000, [5, 10, 20, 40, 60]
REF = 'ou_seqcal'
CMP = ['msm_seqcal', 'amc_seqcal', 'pc2_logistic', 'pc2_gbt', 'raw_logistic', 'raw_gbt']
SAMPLES = {'2008-2026': lambda W: W.year >= 2008, '2009-2026': lambda W: W.year >= 2009,
           '2008-2020': lambda W: W.index <= FROZEN_END, '2009-2020': lambda W: (W.year >= 2009) & (W.index <= FROZEN_END)}
MAIN = ('2008-2026', '2009-2026')


def wide(F, h):
    g = F[(F.set == 'common7') & (F.horizon == h)]
    P = g.pivot_table(index='date', columns='method', values='p', aggfunc='first')
    meta = g.groupby('date').agg(y=('y', 'first'), cref=('clim', 'first'), year=('year', 'first'))
    return meta.join(P).sort_index()


def main():
    cal = pd.read_csv(TAB / 'features_SPX_r1.csv', parse_dates=['date']).dropna(subset=FULL).reset_index(drop=True)
    posmap = pd.Series(np.arange(len(cal)), index=cal.date)
    rng = np.random.default_rng(SEED); T_rows, D_rows, M_rows, S_rows = [], [], [], []
    for tag in ['exp', 'roll5']:
        F = pd.read_csv(TAB / f'P2_F1_forecasts_{tag}.csv', parse_dates=['date'])
        F = F[F.headline.astype(bool) & F.y.notna() & F.p.notna()]
        for h in [5, 20]:
            W0 = wide(F, h)
            for sname, sel in SAMPLES.items():
                W = W0[np.asarray(sel(W0))]; y = W.y.values; base = (y - W.cref.values) ** 2; n = len(y)
                ep_pos = posmap.loc[W.index[y == 1]].values
                episodes = int(clusters_from_positions(ep_pos, h).max() + 1) if len(ep_pos) else 0
                d = {cm: (y - W[REF].values) ** 2 - (y - W[cm].values) ** 2 for cm in CMP}
                ci = {}
                for L in BLOCKS:
                    ix = mbb_index(n, L, B, rng); den = base[ix].mean(1)
                    for cm in CMP:
                        v = d[cm][ix].mean(1) / den; lo, hi = np.quantile(v, [.05, .95]); ci[(cm, L)] = (lo, hi)
                        D_rows.append(dict(window=tag, horizon=h, sample=sname, comparator=cm, block=L,
                                           dBSS=d[cm].mean() / base.mean(), lo90=lo, hi90=hi))
                fam = []
                for cm in CMP:
                    tests = dm_tests(d[cm], h) + gw_conditional(d[cm], h)
                    for t in tests:
                        T_rows.append(dict(window=tag, horizon=h, sample=sname, comparator=cm, mean_d=d[cm].mean(), **t))
                    pick = lambda key: next(t['p_value'] for t in tests if t['variance'].startswith(key[0]) and t['test'].startswith(key[1]))
                    fam.append(dict(window=tag, horizon=h, sample=sname, comparator=cm, n=n, events=int(y.sum()),
                                    episodes=episodes, dBSS=d[cm].mean() / base.mean(), lo90_b20=ci[(cm, 20)][0],
                                    hi90_b20=ci[(cm, 20)][1], lo90_2h=ci[(cm, boot_block(h))][0], hi90_2h=ci[(cm, boot_block(h))][1], p_EWC=pick(('EWC', 'uncond')), p_NW=pick((f'NW lag {h - 1}', 'uncond')),
                                    p_GW_NW=pick(('NW', 'cond')), p_GW_EWC=pick(('EWC', 'cond'))))
                adj = holm([f['p_EWC'] for f in fam])
                for f, a in zip(fam, adj):
                    f['p_EWC_holm'] = a
                if sname in MAIN:
                    names = [REF] + CMP + ['clim']
                    Mt = mcs(np.column_stack([(y - W[m].values) ** 2 for m in names]), names, boot_block(h), B_MCS, rng)
                    Mt['window'] = tag; Mt['horizon'] = h; Mt['sample'] = sname; M_rows.append(Mt)
                    mm = dict(zip(Mt.method, Mt.mcs_p))
                    for f in fam:
                        f['mcs_p_ref'] = mm[REF]; f['mcs_p_cmp'] = mm[f['comparator']]
                S_rows += fam
    write_csv(pd.DataFrame(T_rows), 'P2_F4_tests.csv'); write_csv(pd.DataFrame(D_rows), 'P2_F4_dbss.csv')
    write_csv(pd.concat(M_rows, ignore_index=True), 'P2_F4_mcs.csv')
    S = pd.DataFrame(S_rows); write_csv(S, 'P2_F4_summary.csv')
    show = S[S['sample'].isin(MAIN)]
    for col in ['dBSS', 'p_EWC_holm', 'p_GW_NW']:
        print(col); print(show.pivot_table(index='comparator', columns=['window', 'horizon', 'sample'], values=col).round(3).to_string())
    M = pd.concat(M_rows); print(M.pivot_table(index='method', columns=['window', 'horizon', 'sample'], values='mcs_p').round(2).to_string())
    print(show.groupby(['window', 'horizon', 'sample'])[['events', 'episodes']].first().to_string())
    print('F4 complete')


if __name__ == '__main__':
    main()
