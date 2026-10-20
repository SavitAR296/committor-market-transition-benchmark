"""E1e -- analysis data file for Paper 1: non-positive volumes are filled from a documented second source.

Input : data/spx_2000_2026.csv (E1b: frozen snapshot to 2020-04-17 + Yahoo chart-API download), and
        data/spx_volume_patches.csv (one row per patched day: second-vendor value, its source, and the values used
        to check that the vendor's series coincides with Yahoo's around that day).
Output: data/spx_2000_2026_r1.csv (identical except for the patched days) and tables_pre/E1e_repair_log.csv.
Yahoo's ^GSPC record has volume = 0 on 2023-05-24.  It is filled with 3,739,160,000 shares from Tonghuashun (SPX),
whose open/high/low/close for that day and whose volumes on 2023-05-23 and 2023-05-25 coincide with Yahoo's; the
script re-checks all of these on every run.  A non-positive volume without a patch row stops the build (no silent
forward fill): log(0) = -inf would make every surrogate of E4 share one volume path.
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026.csv')
    P = pd.read_csv(ROOT / 'data' / 'spx_volume_patches.csv', dtype={'date': str})
    vol = lambda day: float(d.loc[d.date == day, 'volume'].iloc[0])
    log = []
    for _, p in P.iterrows():
        i = d.index[d.date == p.date]
        assert len(i) == 1, f'{p.date} not in the data'
        i = i[0]
        assert p.field == 'volume' and float(d.at[i, 'volume']) == float(p.yahoo_value), f'{p.date}: raw value changed'
        for c in ['open', 'high', 'low', 'close']:
            assert abs(float(d.at[i, c]) - float(p[f'vendor_{c}'])) < 0.006, f'{p.date}: {c} differs from the vendor'
        assert vol(p.check_prev_date) == float(p.check_prev_volume), f'{p.check_prev_date}: vendor volume differs'
        assert vol(p.check_next_date) == float(p.check_next_volume), f'{p.check_next_date}: vendor volume differs'
        log.append(dict(date=p.date, volume_original=d.at[i, 'volume'], volume_repaired=int(p.patched_value),
                        source_short='second vendor', source=p.source,
                        check=f'OHLC equal; volumes on {p.check_prev_date} and {p.check_next_date} equal to Yahoo'))
        d.at[i, 'volume'] = int(p.patched_value)
    bad = d.date[~(d.volume > 0)].tolist()
    assert not bad, f'non-positive volume without a documented patch: {bad}'
    assert (d.close > 0).all()
    out = ROOT / 'data' / 'spx_2000_2026_r1.csv'
    d.to_csv(out, index=False)
    pd.DataFrame(log, columns=['date', 'volume_original', 'volume_repaired', 'source_short', 'source', 'check']).to_csv(
        ROOT / 'tables_pre' / 'E1e_repair_log.csv', index=False)
    print(f'{len(log)} day(s) patched; wrote {out.name} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
