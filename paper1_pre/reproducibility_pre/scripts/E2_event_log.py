"""E2 -- operational event log (frozen rule: RV20 > lagged 1000-day 85% quantile AND dd20 < -8%; event = first entry
within (t, t+h]).  Output: tables_pre/E2_event_log.csv (one row per entry), tables_pre/E2_clusters.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; LEG = ROOT.parent / 'reproducibility'
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(LEG / 'src'))
from data_daily import crisis_labels
FROZEN_END = pd.Timestamp('2020-04-17')


def main():
    F = pd.read_csv(ROOT / 'tables_pre' / 'features_SPX_2026.csv', parse_dates=['date'])
    _, inB, entry = crisis_labels(F, 20, rule='rv_dd')
    ent = np.where(entry)[0]; rows = []
    for k, i in enumerate(ent):
        j = i
        while j + 1 < len(inB) and inB[j + 1]:
            j += 1
        rows.append(dict(entry_date=F.date[i].date(), exit_date=F.date[j].date(), days_in_set=int(j - i + 1),
                         close=F.close[i], rv20=F.rv20[i], dd20=F.dd20[i], extension=bool(F.date[i] > FROZEN_END)))
    E = pd.DataFrame(rows)
    gap = np.r_[np.inf, np.diff(ent)]
    E['cluster'] = np.cumsum(gap > 21)
    E.to_csv(ROOT / 'tables_pre' / 'E2_event_log.csv', index=False)
    C = E.groupby('cluster').agg(first_entry=('entry_date', 'first'), last_exit=('exit_date', 'last'), entries=('entry_date', 'size'),
                                 days_in_set=('days_in_set', 'sum'), extension=('extension', 'all')).reset_index()
    C.to_csv(ROOT / 'tables_pre' / 'E2_clusters.csv', index=False)
    print(C[C.extension].to_string(index=False)); print('entries total', len(E), 'in extension', int(E.extension.sum()))


if __name__ == '__main__':
    main()
