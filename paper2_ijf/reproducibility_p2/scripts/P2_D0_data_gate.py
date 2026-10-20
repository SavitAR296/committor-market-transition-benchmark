"""P2_D0 -- data gate for Paper 2 (run first; exit status 1 on any hard failure in an analysis input).

Inputs are read from the latest Paper 1 package (read-only) and checked against its SHA-256 manifests, so Paper 2 uses
byte-identical data: the gated S&P 500 file spx_2000_2026_r1.csv, VIX, Coin Metrics BTC/ETH, and the independent
Yahoo-derived archive (1950-1999 slice) for the pre-2000 era extension.  Frozen code modules are checked as well.
Patch check: r1 must equal the Yahoo splice spx_2000_2026.csv on every row except the days documented in
spx_volume_patches.csv (Paper 1: 2023-05-24, Yahoo volume 0 -> 3,739,160,000 shares from a second vendor whose OHLC
and adjacent-day volumes coincide with Yahoo's), and must carry the documented value on those days.
Cross-check (not an analysis input): a copy of the Yahoo download with the 2023-05-24 row deleted.  Paper 1 rejected
it because dropping a trading day merges two daily returns and shifts every trailing window; here it is only compared
with Paper 1's Yahoo file (every other day must be identical).
Hard failures: SHA mismatch, patch inconsistency, non-positive or missing close/volume, duplicated or non-increasing
dates.  Warnings: |log return| > 0.15, day-on-day volume ratio outside [0.2, 5], calendar gaps > 5 days (equity) /
> 1 day (crypto).  Outputs: tables_p2/P2_D0_data_gate.csv, P2_D0_inputs_sha256.csv, P2_D0_crosscheck.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import (LEG, PRE, SPX_R1, SPX_SPLICE, SPX_PATCHES, SPX_YAHOO, DROPDAY_UPLOAD, VIX, SPX_ARCHIVE, BTC, ETH,
                       FROZEN_FEATURES, sha256, write_csv)

MANIFEST = {   # path -> (manifest file, key in manifest)
    SPX_R1: (PRE / 'SHA256SUMS_P1.txt', 'data/spx_2000_2026_r1.csv'),
    SPX_SPLICE: (PRE / 'SHA256SUMS_P1.txt', 'data/spx_2000_2026.csv'),
    SPX_PATCHES: (PRE / 'SHA256SUMS_P1.txt', 'data/spx_volume_patches.csv'),
    VIX: (PRE / 'SHA256SUMS_P1.txt', 'data/vix_2000_2026.csv'),
    SPX_ARCHIVE: (PRE / 'SHA256SUMS_P1.txt', 'data/archive_vijinho_sp500_1950_2018.csv'),
    BTC: (LEG / 'SHA256SUMS.txt', './data/btc.csv'),
    ETH: (LEG / 'SHA256SUMS.txt', './data/eth.csv'),
    FROZEN_FEATURES: (LEG / 'SHA256SUMS.txt', './tables/features_SPX.csv'),
    LEG / 'src' / 'data_daily.py': (LEG / 'SHA256SUMS.txt', './src/data_daily.py'),
    LEG / 'src' / 'hawkes.py': (LEG / 'SHA256SUMS.txt', './src/hawkes.py'),
    LEG / 'src' / 'committor.py': (LEG / 'SHA256SUMS.txt', './src/committor.py'),
    LEG / 'experiments' / 'B_csf_additions.py': (LEG / 'SHA256SUMS.txt', './experiments/B_csf_additions.py'),
}


def manifest_value(mfile, key):
    for line in mfile.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip('./') == key.lstrip('./'):
            return parts[0]
    return None


def frame(name):
    if name == 'SPX r1 (2000-2026)':
        d = pd.read_csv(SPX_R1, parse_dates=['date']); return d[['date', 'close', 'volume']], 5
    if name == 'SPX archive (1950-1999)':
        d = pd.read_csv(SPX_ARCHIVE); d['date'] = pd.to_datetime(d['Date'])
        d = d[d.date < '2000-01-01'].rename(columns={'Close': 'close', 'Volume': 'volume'}); return d[['date', 'close', 'volume']], 5
    if name in ('BTC', 'ETH'):             # exactly the frozen loader's rows: date >= start and PriceUSD present
        d = pd.read_csv(BTC if name == 'BTC' else ETH, low_memory=False, parse_dates=['time'])
        d = d.rename(columns={'time': 'date', 'PriceUSD': 'close', 'volume_reported_spot_usd_1d': 'volume'})
        d = d[['date', 'close', 'volume']].dropna(subset=['close'])
        d = d[d.date >= ('2013-01-01' if name == 'BTC' else '2016-03-01')]; return d.reset_index(drop=True), 1
    if name == 'VIX':
        d = pd.read_csv(VIX, parse_dates=['DATE']).rename(columns={'DATE': 'date', 'CLOSE': 'close'})
        d['volume'] = 1.0; return d[['date', 'close', 'volume']], 5
    raise ValueError(name)


def check(name):
    d, maxgap = frame(name); out = []
    add = lambda kind, sev, date, val: out.append(dict(file=name, check=kind, severity=sev, date=str(pd.Timestamp(date).date()), value=val))
    for col in ['close', 'volume']:
        for i in np.where(~(d[col] > 0))[0]:
            add(f'non-positive or missing {col}', 'FAIL', d.date.iloc[i], d[col].iloc[i])
    for i in np.where(d.date.duplicated(keep=False))[0]:
        add('duplicated date', 'FAIL', d.date.iloc[i], '')
    for i in np.where(d.date.diff().dt.days <= 0)[0]:
        add('non-increasing date', 'FAIL', d.date.iloc[i], '')
    lr = np.log(d.close).diff()
    for i in np.where(lr.abs() > 0.15)[0]:
        add('|log return| > 0.15', 'WARN', d.date.iloc[i], float(lr.iloc[i]))
    if name != 'VIX':
        jump = (d.volume / d.volume.shift(1)).replace([np.inf, -np.inf], np.nan)
        for i in np.where((jump > 5) | (jump < 0.2))[0]:
            add('volume ratio outside [0.2, 5]', 'WARN', d.date.iloc[i], float(jump.iloc[i]))
    gaps = d.date.diff().dt.days
    for i in np.where(gaps > maxgap)[0]:
        add(f'calendar gap > {maxgap} days', 'WARN', d.date.iloc[i], int(gaps.iloc[i]))
    return out, dict(file=name, first=str(d.date.min().date()), last=str(d.date.max().date()), rows=len(d))


def patch_check():
    """r1 = splice except on documented patch days, where it must carry the documented value."""
    s = pd.read_csv(SPX_SPLICE, dtype={'date': str}); r = pd.read_csv(SPX_R1, dtype={'date': str})
    P = pd.read_csv(SPX_PATCHES, dtype={'date': str}); out = []
    if len(s) != len(r) or not (s.date.values == r.date.values).all():
        return [dict(file='SPX r1', check='r1 and splice have different dates', severity='FAIL', date='', value='')]
    diff = s.index[(s[['open', 'high', 'low', 'close', 'volume']].values != r[['open', 'high', 'low', 'close', 'volume']].values).any(1)]
    for i in diff:
        p = P[P.date == s.date[i]]
        ok = len(p) == 1 and float(r.volume[i]) == float(p.patched_value.iloc[0]) and float(s.volume[i]) == float(p.yahoo_value.iloc[0]) \
            and (s.loc[i, ['open', 'high', 'low', 'close']].values == r.loc[i, ['open', 'high', 'low', 'close']].values).all()
        out.append(dict(file='SPX r1', check='documented patch' if ok else 'undocumented difference to splice',
                        severity='INFO' if ok else 'FAIL', date=s.date[i], value=f'{s.volume[i]} -> {r.volume[i]}'))
    for d in set(P.date) - set(s.date[diff]):
        out.append(dict(file='SPX r1', check='patch listed but not applied', severity='FAIL', date=d, value=''))
    return out


def upload_crosscheck():
    """The uploaded drop-day copy versus Paper 1's Yahoo download: identical except for the deleted day."""
    if not DROPDAY_UPLOAD.exists():
        return pd.DataFrame()
    norm = lambda d: d.rename(columns={c: c.strip().lower().replace(' ', '_') for c in d.columns})
    a = norm(pd.read_csv(SPX_YAHOO, encoding='utf-8-sig')); b = norm(pd.read_csv(DROPDAY_UPLOAD, encoding='utf-8-sig'))
    m = a.merge(b, on='date', how='outer', suffixes=('_paper1', '_upload'), indicator=True); both = m[m._merge == 'both']
    return pd.DataFrame([dict(file=DROPDAY_UPLOAD.name, sha256=sha256(DROPDAY_UPLOAD), rows=len(b),
                              only_in_paper1_yahoo=';'.join(m.date[m._merge == 'left_only']),
                              only_in_upload=';'.join(m.date[m._merge == 'right_only']), common_days=len(both),
                              max_abs_close_diff=float((both.close_paper1 - both.close_upload).abs().max()),
                              volume_days_different=int((both.volume_paper1 != both.volume_upload).sum()),
                              analysis_input=False,
                              reason='Paper 1 (README_P1, E1e): dropping a day merges two returns and shifts trailing windows')])


def main():
    sha_rows, rows, summ = [], [], []
    for path, (mfile, key) in MANIFEST.items():
        got = sha256(path); want = manifest_value(mfile, key)
        sha_rows.append(dict(path=str(path.relative_to(path.parents[2])), sha256=got, manifest=mfile.name, match=got == want))
        if got != want:
            rows.append(dict(file=path.name, check='sha256 differs from manifest', severity='FAIL', date='', value=got))
    rows += patch_check()
    for name in ['SPX r1 (2000-2026)', 'SPX archive (1950-1999)', 'BTC', 'ETH', 'VIX']:
        o, s = check(name); rows += o; summ.append(s)
    R = pd.DataFrame(rows, columns=['file', 'check', 'severity', 'date', 'value'])
    X = upload_crosscheck()
    write_csv(R, 'P2_D0_data_gate.csv'); write_csv(pd.DataFrame(sha_rows), 'P2_D0_inputs_sha256.csv'); write_csv(X, 'P2_D0_crosscheck.csv')
    print(pd.DataFrame(summ).to_string(index=False))
    print(pd.DataFrame(sha_rows)[['path', 'match']].to_string(index=False))
    print(R[R.severity != 'WARN'].to_string(index=False))
    print(R[R.severity == 'WARN'].groupby(['file', 'check']).size().to_string())
    print(X.T.to_string(header=False))
    sys.exit(1 if (R.severity == 'FAIL').any() else 0)


if __name__ == '__main__':
    main()
