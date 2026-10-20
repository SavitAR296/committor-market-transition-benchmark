"""E1 -- extend the S&P 500 snapshot to 2026-10-16 and rebuild the feature table with the frozen feature code.

Inputs (read-only):
  ../reproducibility/data/sp500-2000.csv                vega-datasets Yahoo snapshot, 2000-01-03..2020-04-17
  data/sp500_2020-04-01_to_2026-10-16_upload.csv        author-supplied ^GSPC daily OHLCV (uploaded 2026-10-17)
  ../reproducibility/data/vix_daily.csv                  CBOE VIX via datahub, to 2026-09-04 (frozen)
  data/vix_daily_datahub_retrieved_2026-10-17.csv        datahub finance-vix retrieved 2026-10-17 (to 2026-09-22)
Splice rule: frozen snapshot through 2020-04-17, upload from 2020-04-20; VIX: frozen file, then datahub rows after
2026-09-04; the frozen loader forward-fills VIX beyond its last date (affects only full-11 features after 2026-09-22).
Outputs: data/spx_2000_2026.csv, data/vix_2000_2026.csv, tables_pre/features_SPX_2026.csv,
         tables_pre/E1_splice_overlap.csv, tables_pre/E1_feature_equality.csv, tables_pre/E1_splice_report.md
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; LEG = ROOT.parent / 'reproducibility'
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(LEG / 'src'))
from data_daily import build_features              # frozen feature code (unchanged)
from pre_common import sha256

D = ROOT / 'data'; T = ROOT / 'tables_pre'


def main():
    old = pd.read_csv(LEG / 'data' / 'sp500-2000.csv', parse_dates=['date'])
    new = pd.read_csv(D / 'sp500_2020-04-01_to_2026-10-16_upload.csv', encoding='utf-8-sig', parse_dates=['Date'])
    new = new.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    new['adjclose'] = new['close']
    ov = old.merge(new, on='date', suffixes=('_old', '_new'))
    ovr = pd.DataFrame(dict(date=ov.date, **{f'{c}_absdiff': (ov[f'{c}_old'] - ov[f'{c}_new']).abs() for c in ['open', 'high', 'low', 'close']},
                            volume_ratio=ov.volume_new / ov.volume_old))
    ovr.to_csv(T / 'E1_splice_overlap.csv', index=False)
    spl = pd.concat([old, new[new.date > old.date.max()][old.columns]], ignore_index=True).sort_values('date').reset_index(drop=True)
    assert spl.date.is_unique
    spl.to_csv(D / 'spx_2000_2026.csv', index=False)
    v0 = pd.read_csv(LEG / 'data' / 'vix_daily.csv', parse_dates=['DATE'])
    v1 = pd.read_csv(D / 'vix_daily_datahub_retrieved_2026-10-17.csv', parse_dates=['DATE'])
    vo = v0.merge(v1, on='DATE', suffixes=('_frozen', '_new'))
    vix_maxdiff = float((vo.CLOSE_frozen - vo.CLOSE_new).abs().max())
    vix = pd.concat([v0, v1[v1.DATE > v0.DATE.max()]], ignore_index=True)
    vix.to_csv(D / 'vix_2000_2026.csv', index=False)
    # identical to data_daily.load_market('SPX') but on the spliced files
    df = spl.merge(vix.rename(columns={'DATE': 'date', 'CLOSE': 'vix'})[['date', 'vix']], on='date', how='left')
    df['vix'] = df['vix'].ffill(); df['activity'] = df['volume'].astype(float); df['ann'] = 252.0
    df = df[['date', 'close', 'high', 'low', 'volume', 'activity', 'vix', 'ann']]
    F = build_features(df)
    F.to_csv(T / 'features_SPX_2026.csv', index=False)
    fz = pd.read_csv(LEG / 'tables' / 'features_SPX.csv', parse_dates=['date'])
    G = F[F.date <= fz.date.max()].reset_index(drop=True)
    rows = []
    for c in fz.columns:
        if c == 'date':
            rows.append(dict(column=c, n=len(fz), nan_pattern_equal=True, max_abs_diff=float((G.date != fz.date).sum()))); continue
        a = pd.to_numeric(G[c], errors='coerce').to_numpy(float); b = fz[c].to_numpy(float)
        na = np.isnan(a) == np.isnan(b); m = ~np.isnan(b) & ~np.isnan(a)
        rows.append(dict(column=c, n=int(m.sum()), nan_pattern_equal=bool(na.all()),
                         max_abs_diff=float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0))
    E = pd.DataFrame(rows); E.to_csv(T / 'E1_feature_equality.csv', index=False)
    vol_ratio_med = float(ovr.volume_ratio.median())
    rep = [
        '# E1 splice report', '',
        f'* Frozen snapshot: sp500-2000.csv ({old.date.min().date()} .. {old.date.max().date()}, {len(old)} rows), sha256 {sha256(LEG / "data" / "sp500-2000.csv")}',
        f'* Extension: author-supplied ^GSPC daily OHLCV, uploaded 2026-10-17 ({new.date.min().date()} .. {new.date.max().date()}, {len(new)} rows), sha256 {sha256(D / "sp500_2020-04-01_to_2026-10-16_upload.csv")}',
        f'* Spliced file data/spx_2000_2026.csv: {len(spl)} rows, {spl.date.min().date()} .. {spl.date.max().date()}, sha256 {sha256(D / "spx_2000_2026.csv")}',
        f'* Overlap available for comparison: {len(ov)} trading days ({ov.date.min().date()} .. {ov.date.max().date()}); the outline asked for 2019-01-02..2020-04-17, but the supplied extension starts on 2020-04-01.',
        f'* Overlap close: max |diff| = {ovr.close_absdiff.max():.3e} index points (max relative {float((ovr.close_absdiff / ov.close_old).max()):.2e}); open/high/low max |diff| = {ovr[["open_absdiff", "high_absdiff", "low_absdiff"]].to_numpy().max():.3e}.',
        f'* Overlap volume ratio new/old: median {vol_ratio_med:.4f}, range [{ovr.volume_ratio.min():.4f}, {ovr.volume_ratio.max():.4f}].',
        f'* VIX: frozen file to {v0.DATE.max().date()}, datahub retrieval to {v1.DATE.max().date()}; max |diff| on {len(vo)} common days = {vix_maxdiff:.3e}; values after {v1.DATE.max().date()} are forward-filled by the frozen loader (full-11 features only).',
        f'* Feature equality on {fz.date.min().date()} .. {fz.date.max().date()} (all {len(fz.columns) - 1} columns): max |diff| = {E[E.column != "date"].max_abs_diff.max():.3e}; NaN patterns equal: {bool(E.nan_pattern_equal.all())}.',
    ]
    (T / 'E1_splice_report.md').write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep)); print(E.sort_values('max_abs_diff', ascending=False).head(6).to_string(index=False))


if __name__ == '__main__':
    main()
