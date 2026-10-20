"""E1d -- independent check of the frozen 2000-2020 S&P 500 snapshot against a second Yahoo-derived archive.

Archive: github.com/vijinho/sp500 (csv/sp500.csv; Yahoo Finance ^GSPC daily OHLCV 1950-01-03..2018-12-21, repository
archived Jan 2025), stored as data/archive_vijinho_sp500_1950_2018.csv (SHA-256 in SHA256SUMS.txt).
Compared on the common dates with the frozen vega-datasets snapshot (../reproducibility/data/sp500-2000.csv).
The 2019-2026 part has no independent source reachable from the build environment (see README, task R1).
Output: tables_pre/E1d_archive_check.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    a = pd.read_csv(ROOT / 'data' / 'archive_vijinho_sp500_1950_2018.csv')
    a['date'] = pd.to_datetime(a['Date']).dt.normalize()
    f = pd.read_csv(ROOT.parent / 'reproducibility' / 'data' / 'sp500-2000.csv')
    f.columns = [c.lower() for c in f.columns]
    f['date'] = pd.to_datetime(f['date'])
    m = f.merge(a[['date', 'Close', 'Volume']], on='date', how='inner')
    lo, hi = m.date.min(), m.date.max()
    fd = set(f.date[(f.date >= lo) & (f.date <= hi)]); ad = set(a.date[(a.date >= lo) & (a.date <= hi)])
    rel = (m.close / m.Close - 1).abs(); vr = m.volume / m.Volume
    out = dict(first=str(lo.date()), last=str(hi.date()), common_days=len(m),
               frozen_only_days=len(fd - ad), archive_only_days=len(ad - fd),
               close_max_rel_diff=float(rel.max()), close_days_rel_diff_gt_1e6=int((rel > 1e-6).sum()),
               volume_ratio_median=float(vr.median()), volume_ratio_min=float(vr.min()),
               volume_ratio_max=float(vr.max()), volume_days_off_gt_1pct=int(((vr - 1).abs() > 0.01).sum()),
               zero_volume_frozen=int((f.volume <= 0).sum()), zero_volume_archive=int((a.Volume <= 0).sum()))
    pd.DataFrame([out]).to_csv(ROOT / 'tables_pre' / 'E1d_archive_check.csv', index=False, float_format='%.6g')
    for k, v in out.items():
        print(f'{k:28s} {v}')


if __name__ == '__main__':
    main()
