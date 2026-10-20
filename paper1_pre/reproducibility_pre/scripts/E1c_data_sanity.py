"""E1c -- data sanity gate (run before every E-script; exits with status 1 on any hard failure).

Hard failures (gated on the analysis file spx_2000_2026_r1.csv): non-positive or missing close/volume, duplicated or non-increasing dates.
Warnings: |log return| > 0.15, volume jumps > 5x day-on-day, calendar gaps > 5 days.
Yahoo's ^GSPC record has volume = 0 on 2023-05-24 (in the October 2026 chart-API download and in the earlier
extension file alike); unrepaired, that single day makes every S&P surrogate of E4 share one volume path.  The raw
files are reported; only the repaired analysis file (E1e) is gated.
Output: tables_pre/E1c_data_sanity.csv (one row per finding).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {'spx_2000_2026': ROOT / 'data' / 'spx_2000_2026.csv',
         'spx_yahoo_download': ROOT / 'data' / 'spx_yahoo_chart_api_2019-01-02_2026-10-16.csv',
         'spx_extension_upload': ROOT / 'data' / 'sp500_2020-04-01_to_2026-10-16_upload.csv',
         'spx_2000_2026_r1': ROOT / 'data' / 'spx_2000_2026_r1.csv'}
GATED = 'spx_2000_2026_r1'      # the file every market script reads; only its failures stop the build


def check(name, path):
    d = pd.read_csv(path, encoding='utf-8-sig')
    d.columns = [c.strip().lower() for c in d.columns]
    d['date'] = pd.to_datetime(d['date'])
    out = []
    add = lambda kind, sev, row, val: out.append(dict(file=name, check=kind, severity=sev,
                                                      date=str(row.date.date()), value=val))
    for col in ['close', 'volume']:
        for _, row in d[~(d[col] > 0)].iterrows():
            add(f'non-positive or missing {col}', 'FAIL', row, row[col])
    for _, row in d[d.date.duplicated(keep=False)].iterrows():
        add('duplicated date', 'FAIL', row, '')
    for _, row in d[d.date.diff().dt.days <= 0].iterrows():
        add('non-increasing date', 'FAIL', row, '')
    lr = np.log(d.close).diff()
    for _, row in d[lr.abs() > 0.15].iterrows():
        add('|log return| > 0.15', 'WARN', row, float(lr[row.name]))
    jump = (d.volume / d.volume.shift(1)).replace([np.inf, -np.inf], np.nan)
    for _, row in d[(jump > 5) | (jump < 0.2)].iterrows():
        add('volume jump > 5x', 'WARN', row, float(jump[row.name]))
    for _, row in d[d.date.diff().dt.days > 5].iterrows():
        add('calendar gap > 5 days', 'WARN', row, int(d.date.diff().dt.days[row.name]))
    return out


def main():
    rows = []
    for name, path in FILES.items():
        if path.exists():
            rows += check(name, path)
    R = pd.DataFrame(rows, columns=['file', 'check', 'severity', 'date', 'value'])
    R.to_csv(ROOT / 'tables_pre' / 'E1c_data_sanity.csv', index=False)
    print(R.to_string(index=False) if len(R) else 'no findings')
    sys.exit(1 if ((R.severity == 'FAIL') & (R.file == GATED)).any() else 0)


if __name__ == '__main__':
    main()
