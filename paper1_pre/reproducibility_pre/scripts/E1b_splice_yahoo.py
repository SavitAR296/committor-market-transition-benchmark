"""E1b -- final S&P 500 daily file for Paper 1 (task R1).

Sources
  ../reproducibility/data/sp500-2000.csv            vega-datasets snapshot of Yahoo ^GSPC, 2000-01-03..2020-04-17 (frozen)
  data/spx_yahoo_chart_api_2019-01-02_2026-10-16.csv Yahoo Finance ^GSPC via the chart API
                                                    (query2.finance.yahoo.com/v8/finance/chart/^GSPC), downloaded by the
                                                    authors in October 2026; replaces the earlier author-supplied extension
  data/sp500_2020-04-01_to_2026-10-16_upload.csv    earlier extension (kept only for the comparison below)
Splice: frozen snapshot through 2020-04-17, Yahoo download from 2020-04-20 (so every 2000-2020 feature equals the
frozen feature table).  Overlaps are reported, not used.  Yahoo's record itself has volume = 0 on 2023-05-24
(present in both downloads); E1e repairs it.
Outputs: data/spx_2000_2026.csv, tables_pre/E1b_splice_report.csv
"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data'
YAHOO = D / 'spx_yahoo_chart_api_2019-01-02_2026-10-16.csv'


def load(path, **kw):
    d = pd.read_csv(path, encoding='utf-8-sig', **kw)
    d.columns = [c.strip().lower().replace(' ', '_') for c in d.columns]
    d['date'] = pd.to_datetime(d['date']).dt.normalize()
    return d


def compare(a, b, label):
    m = a.merge(b[['date', 'close', 'volume']], on='date', suffixes=('', '_b'))
    rel = (m.close / m.close_b - 1).abs(); ok = (m.volume_b > 0) & (m.volume > 0); vr = m.volume[ok] / m.volume_b[ok]
    lo, hi = m.date.min(), m.date.max()
    return dict(comparison=label, first=str(lo.date()), last=str(hi.date()), common_days=len(m),
                only_first=len(set(a.date[(a.date >= lo) & (a.date <= hi)]) - set(m.date)),
                only_second=len(set(b.date[(b.date >= lo) & (b.date <= hi)]) - set(m.date)),
                close_max_rel_diff=float(rel.max()), volume_ratio_median=float(vr.median()),
                volume_ratio_min=float(vr.min()), volume_ratio_max=float(vr.max()),
                volume_days_off_gt_1pct=int(((vr - 1).abs() > 0.01).sum()),
                zero_volume_days_second=int((b.volume <= 0).sum()))


def main():
    frozen = load(ROOT.parent / 'reproducibility' / 'data' / 'sp500-2000.csv')
    yahoo = load(YAHOO); old = load(D / 'sp500_2020-04-01_to_2026-10-16_upload.csv')
    assert not yahoo.date.duplicated().any() and yahoo.date.is_monotonic_increasing
    rows = [compare(frozen, yahoo, 'frozen snapshot vs Yahoo download'),
            compare(yahoo, old, 'Yahoo download vs earlier extension')]
    cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    spl = pd.concat([frozen[cols], yahoo[yahoo.date > frozen.date.max()][cols]], ignore_index=True)
    spl['date'] = spl.date.dt.strftime('%Y-%m-%d')
    out = D / 'spx_2000_2026.csv'; spl.to_csv(out, index=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    R = pd.DataFrame(rows); R['yahoo_sha256'] = sha(YAHOO); R['spliced_rows'] = len(spl); R['spliced_sha256'] = sha(out)
    R.to_csv(ROOT / 'tables_pre' / 'E1b_splice_report.csv', index=False, float_format='%.6g')
    print(R.drop(columns=['yahoo_sha256', 'spliced_sha256']).T.to_string(header=False))


if __name__ == '__main__':
    main()
