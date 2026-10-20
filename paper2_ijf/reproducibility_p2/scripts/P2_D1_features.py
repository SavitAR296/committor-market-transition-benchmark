"""P2_D1 -- S&P 500 feature table for Paper 2 from the gated file, with the frozen feature code (unchanged).

Input : spx_2000_2026_r1.csv (Paper 1, gated; 2023-05-24 volume repaired), vix_2000_2026.csv.
Code  : data_daily.build_features of the frozen CSF package (trailing windows, Hawkes branching ratio, VIX, VRP).
Checks: (i) 2000-01-03..2020-04-17 must equal the frozen feature table (all columns, NaN patterns);
        (ii) difference to the pre-repair table features_SPX_2026.csv that fed the old E3 forecasts, per column.
Outputs: tables_p2/features_SPX_r1.csv, P2_D1_feature_equality.csv, P2_D1_repair_effect.csv.  No randomness.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import SPX_R1, VIX, FROZEN_FEATURES, OLD_FEATURES, TAB, sha256, write_csv
from data_daily import build_features                    # frozen


def build():
    s = pd.read_csv(SPX_R1, parse_dates=['date'])
    v = pd.read_csv(VIX, parse_dates=['DATE']).rename(columns={'DATE': 'date', 'CLOSE': 'vix'})
    df = s.merge(v[['date', 'vix']], on='date', how='left')
    df['vix'] = df['vix'].ffill(); df['activity'] = df['volume'].astype(float); df['ann'] = 252.0
    return build_features(df[['date', 'close', 'high', 'low', 'volume', 'activity', 'vix', 'ann']])


def compare(A, B, label):
    rows = []
    for c in B.columns:
        if c == 'date':
            continue
        a = pd.to_numeric(A[c], errors='coerce').to_numpy(float); b = pd.to_numeric(B[c], errors='coerce').to_numpy(float)
        both = ~np.isnan(a) & ~np.isnan(b); d = np.abs(a[both] - b[both])
        rows.append(dict(comparison=label, column=c, n=int(both.sum()), nan_pattern_equal=bool((np.isnan(a) == np.isnan(b)).all()),
                         max_abs_diff=float(d.max()) if len(d) else 0.0, days_diff_gt_1e12=int((d > 1e-12).sum()),
                         first_day_diff=str(A.date[both][d > 1e-12].min().date()) if (d > 1e-12).any() else ''))
    return rows


def main():
    F = build(); F.to_csv(TAB / 'features_SPX_r1.csv', index=False)
    Z = pd.read_csv(FROZEN_FEATURES, parse_dates=['date'])
    G = F[F.date <= Z.date.max()].reset_index(drop=True)
    assert (G.date.values == Z.date.values).all()
    eq = pd.DataFrame(compare(G, Z, 'r1 vs frozen table, 2000-01-03..2020-04-17'))
    O = pd.read_csv(OLD_FEATURES, parse_dates=['date']); assert (F.date.values == O.date.values).all()
    rep = pd.DataFrame(compare(F, O, 'r1 vs pre-repair table, 2000-2026'))
    write_csv(eq, 'P2_D1_feature_equality.csv'); write_csv(rep, 'P2_D1_repair_effect.csv')
    print(f'features_SPX_r1.csv: {len(F)} rows, sha256 {sha256(TAB / "features_SPX_r1.csv")}')
    print('frozen-period equality: max |diff| =', eq.max_abs_diff.max(), '; NaN patterns equal:', bool(eq.nan_pattern_equal.all()))
    print(rep[rep.days_diff_gt_1e12 > 0][['column', 'max_abs_diff', 'days_diff_gt_1e12', 'first_day_diff']].to_string(index=False))
    assert eq.max_abs_diff.max() < 1e-12 and eq.nan_pattern_equal.all()


if __name__ == '__main__':
    main()
