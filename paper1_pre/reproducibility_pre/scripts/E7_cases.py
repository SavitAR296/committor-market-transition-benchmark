"""E7 -- three case studies (common-7 state, E3 forecasts; 2008-2020 identical to the frozen file).
Outputs: E7_cases.csv (time series for Fig. 6) and E7_case_facts.csv (every number quoted in the text)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; LEG = ROOT.parent / 'reproducibility'
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(LEG / 'src'))
from data_daily import crisis_labels
from pre_common import write_csv
CASES = {'2011': ('2011-07-01', '2011-11-30', '2011-08-29', '2011-10-10'),
         '2020': ('2020-01-15', '2020-04-17', '2020-02-18', '2020-02-28'),
         '2025': ('2025-02-15', '2025-05-30', '2025-03-24', '2025-04-22')}


def main():
    F = pd.read_csv(ROOT / 'tables_pre' / 'features_SPX_2026.csv', parse_dates=['date'])
    _, inB, entry = crisis_labels(F, 5, rule='rv_dd'); F['inB'] = inB; F['entry'] = entry
    F['rv_thr'] = F.rv20.rolling(1000, min_periods=500).quantile(0.85).shift(1)
    E = pd.read_csv(ROOT / 'tables_pre' / 'E3_forecasts_2026.csv', parse_dates=['date'])
    E = E[(E.set == 'common7') & E.method.isin(['ou_raw', 'ou_seqcal'])]
    P = E.pivot_table(index=['date', 'horizon'], columns='method', values='p').reset_index()
    Y = E[E.method == 'ou_raw'][['date', 'horizon', 'y', 'headline']]
    P = P.merge(Y, on=['date', 'horizon'])
    rows, facts = [], []
    for case, (a, b, fa, fb) in CASES.items():
        G = F[(F.date >= a) & (F.date <= b)][['date', 'close', 'dd20', 'rv20', 'rv_thr', 'inB', 'entry']]
        for h in [5, 20]:
            Q = G.merge(P[P.horizon == h], on='date', how='left'); Q['case'] = case; rows.append(Q)
        Q = G.merge(P[P.horizon == 5], on='date', how='left'); win = (Q.date >= fa) & (Q.date <= fb)
        facts += [dict(case=case, fact='window', value=f'{fa}..{fb}'),
                  dict(case=case, fact='trading days in window', value=int(win.sum())),
                  dict(case=case, fact='days with S>=0.9995 (h=5)', value=int((Q.ou_raw[win] >= 0.9995).sum())),
                  dict(case=case, fact='operational entries in window', value=int(Q.entry[win].sum())),
                  dict(case=case, fact='entry dates in window', value=';'.join(str(d.date()) for d in Q.date[win & Q.entry.astype(bool)])),
                  dict(case=case, fact='seq-cal min/max in window (h=5)', value=f'{Q.ou_seqcal[win].min():.4f}/{Q.ou_seqcal[win].max():.4f}'),
                  dict(case=case, fact='headline days in window with label 1 (h=5)', value=int((Q.y[win] == 1).sum()))]
        for d in Q.date[win]:
            r = Q[Q.date == d].iloc[0]
            facts.append(dict(case=case, fact=f'{d.date()} S/seqcal/y/inB', value=f'{r.ou_raw:.4f}/{r.ou_seqcal:.4f}/{r.y}/{bool(r.inB)}'))
    write_csv(pd.concat(rows, ignore_index=True), 'E7_cases.csv'); write_csv(pd.DataFrame(facts), 'E7_case_facts.csv')
    print('E7 complete'); (ROOT / 'tables_pre' / '_cache' / 'DONE_E7_cases').touch()


if __name__ == '__main__':
    main()
