"""P2_F5 -- three episodes seen by the forecasts: August 2011, February-March 2020, March-April 2025.

For each window: daily S (h = 5, 20), seq-cal, PC2 logistic and PC2 GBT, the operational stress set and its entries.
Facts (every number the text may quote): entries in the window and their dates; for the first entry, the first of
the preceding 20 trading days with S5 > 0.99 and its lead in trading days; the maxima of seq-cal and PC2 GBT (h = 5)
over those 20 days and the year's training climatology; saturated days (S5 > 0.99, outside the stress set) in the
window and how many were followed by an entry within 5 days.
Inputs: P2_F1_forecasts_exp.csv, features_SPX_r1.csv.  Outputs: P2_F5_case_series.csv, P2_F5_case_facts.csv.
No randomness.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import TAB, FULL, write_csv
from data_daily import crisis_labels                                     # frozen

CASES = {'2011': ('2011-07-01', '2011-11-30'), '2020': ('2020-01-15', '2020-04-17'), '2025': ('2025-02-15', '2025-05-30')}
METH = ['ou_raw', 'ou_seqcal', 'pc2_logistic', 'pc2_gbt']


def main():
    F = pd.read_csv(TAB / 'P2_F1_forecasts_exp.csv', parse_dates=['date'])
    F = F[(F.set == 'common7') & F.method.isin(METH + ['clim'])]
    P = F.pivot_table(index='date', columns=['method', 'horizon'], values='p'); P.columns = [f'{m}_{h}' for m, h in P.columns]
    Y = F[F.method == 'ou_raw'].pivot_table(index='date', columns='horizon', values='y'); Y.columns = [f'y_{h}' for h in Y.columns]
    feat = pd.read_csv(TAB / 'features_SPX_r1.csv', parse_dates=['date']).dropna(subset=FULL).reset_index(drop=True)
    _, inB, entry = crisis_labels(feat, 5, rule='rv_dd'); feat['inB'] = inB; feat['entry'] = entry
    feat['rv_thr'] = feat.rv20.rolling(1000, min_periods=500).quantile(0.85).shift(1)
    G = feat[['date', 'close', 'dd20', 'rv20', 'rv_thr', 'inB', 'entry']].merge(P, on='date', how='left').merge(Y, on='date', how='left')
    rows, facts = [], []
    for case, (a, b) in CASES.items():
        Q = G[(G.date >= a) & (G.date <= b)].reset_index(drop=True); Q['case'] = case; rows.append(Q)
        ent = Q.index[Q.entry.astype(bool)]
        add = lambda k, v: facts.append(dict(case=case, fact=k, value=v))
        add('window', f'{a}..{b}'); add('trading days', len(Q)); add('entries', len(ent))
        add('entry dates', ';'.join(str(d.date()) for d in Q.date[ent]))
        sat = (Q.ou_raw_5 > 0.99) & ~Q.inB.astype(bool)
        add('saturated days S5>0.99 outside the set', int(sat.sum()))
        add('saturated days followed by an entry within 5 days', int((Q.y_5[sat] == 1).sum()))
        if len(ent):
            e = ent[0]; pre = Q.iloc[max(0, e - 20):e]; al = pre.index[pre.ou_raw_5 > 0.99]
            add('first entry', str(Q.date[e].date()))
            add('first S5>0.99 in the 20 days before', str(Q.date[al[0]].date()) if len(al) else 'none')
            add('lead of that alarm (trading days)', int(e - al[0]) if len(al) else 0)
            add('max seq-cal (h=5) in the 20 days before', round(float(pre.ou_seqcal_5.max()), 4))
            add('max PC2 GBT (h=5) in the 20 days before', round(float(pre.pc2_gbt_5.max()), 4))
            add('training climatology (h=5)', round(float(pre.clim_5.iloc[-1]), 4))
    write_csv(pd.concat(rows, ignore_index=True), 'P2_F5_case_series.csv'); write_csv(pd.DataFrame(facts), 'P2_F5_case_facts.csv')
    print(pd.DataFrame(facts).to_string(index=False)); print('F5 complete')


if __name__ == '__main__':
    main()
