"""P2_F7 -- sensitivity of the headline to a detail of the frozen label rule.

The lagged 1000-day RV quantile of the rule needs 500 rows, so the first 500 complete-case rows (2002-12-30 to
2004-12-23) carry structurally zero labels and enter every expanding training set.  P2_F1_forecasts.py
--labels-defined (tag exp_ld) drops them.  This script compares the two runs on what the paper reports: BSS, MCB and
DSC of the main methods; paired DSC intervals of raw S against seq-cal and PC2 logistic (blocks of 2h days, B = 2000); dBSS of
each comparator against seq-cal with EWC fixed-b p-values and Holm adjustment.  Both runs use identical bootstrap
indices (one generator per horizon and sample), so the comparison is paired.
Inputs: P2_F1_forecasts_exp.csv, P2_F1_forecasts_exp_ld.csv.  Output: P2_F7_label_rule.csv.  Seed 20261026.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import TAB, MAIN_METHODS, bss, corp, mbb_index, boot_block, dm_tests, holm, write_csv

SEED = 20261026
CMP = ['msm_seqcal', 'amc_seqcal', 'pc2_logistic', 'pc2_gbt', 'raw_logistic', 'raw_gbt']
SAMPLES = {'2008-2026': 2008, '2009-2026': 2009}


def wide(F, h):
    g = F[(F.set == 'common7') & (F.horizon == h)]
    P = g.pivot_table(index='date', columns='method', values='p', aggfunc='first')
    meta = g.groupby('date').agg(y=('y', 'first'), cref=('clim', 'first'), year=('year', 'first'))
    return meta.join(P).sort_index()


def main():
    rows = []
    for tag in ['exp', 'exp_ld']:
        F = pd.read_csv(TAB / f'P2_F1_forecasts_{tag}.csv', parse_dates=['date'])
        F = F[F.headline.astype(bool) & F.y.notna() & F.p.notna()]
        for h in [5, 20]:
            W0 = wide(F, h)
            for k, (sname, y0) in enumerate(SAMPLES.items()):
                W = W0[W0.year >= y0]; y = W.y.values; base = (y - W.cref.values) ** 2
                key = dict(run=tag, horizon=h, sample=sname)
                for m in MAIN_METHODS:
                    c = corp(y, W[m].values)
                    rows += [dict(**key, quantity='BSS', method=m, value=bss(y, W[m], W.cref)),
                             dict(**key, quantity='MCB', method=m, value=c['MCB']), dict(**key, quantity='DSC', method=m, value=c['DSC'])]
                rng = np.random.default_rng([SEED, h, k]); st = {m: [] for m in ['ou_raw', 'ou_seqcal', 'pc2_logistic']}
                for ix in mbb_index(len(y), boot_block(h), 2000, rng):
                    if y[ix].sum() == 0:
                        continue
                    for m in st:
                        st[m].append(corp(y[ix], W[m].values[ix])['DSC'])
                for m in ['ou_seqcal', 'pc2_logistic']:
                    dd = np.array(st['ou_raw']) - np.array(st[m])
                    rows += [dict(**key, quantity='dDSC rawS-m q05', method=m, value=np.quantile(dd, .05)),
                             dict(**key, quantity='dDSC rawS-m q95', method=m, value=np.quantile(dd, .95))]
                ps = []
                for cm in CMP:
                    d = (y - W.ou_seqcal.values) ** 2 - (y - W[cm].values) ** 2
                    p = next(t['p_value'] for t in dm_tests(d, h) if t['variance'].startswith('EWC'))
                    ps.append(p); rows.append(dict(**key, quantity='dBSS vs seq-cal', method=cm, value=d.mean() / base.mean()))
                for cm, a in zip(CMP, holm(ps)):
                    rows.append(dict(**key, quantity='p EWC Holm', method=cm, value=a))
    R = pd.DataFrame(rows); write_csv(R, 'P2_F7_label_rule.csv')
    P = R.pivot_table(index=['quantity', 'method'], columns=['horizon', 'sample', 'run'], values='value')
    show = ['BSS', 'DSC', 'dDSC rawS-m q05', 'dDSC rawS-m q95', 'dBSS vs seq-cal', 'p EWC Holm']
    print(P.loc[show].round(4).to_string())
    print('F7 complete')


if __name__ == '__main__':
    main()
