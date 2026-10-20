"""P2_F2 -- resolution and reliability of every forecast (Murphy decomposition).

Primary: CORP decomposition BS = MCB - DSC + UNC with PAV recalibration (Dimitriadis, Gneiting and Jordan 2021),
which needs no binning choice and is exact.  Check: binned Murphy decomposition with the within-bin terms of
Stephenson et al. (2008), BS = REL - RES + UNC + WBV - WBC, fixed 10 bins and forecast deciles (frozen code).
Uncertainty: 90% moving-block-bootstrap intervals (blocks of 2h days, B = 2000) of MCB, DSC and BSS for the main methods
of the common-7 state.  AUC is reported only for raw scores and classifiers (see p2_common); the pooled AUC of a
sequentially recalibrated score or of climatology is not a discrimination measure and is not computed.
Samples: 2008-2026 and 2009-2026 (main), 2008-2020 and 2009-2020 (frozen period), 2021-2026.  Horizons 5 and 20.
Inputs: tables_p2/P2_F1_forecasts_<tag>.csv.  Outputs: P2_F2_decomposition_<tag>.csv, P2_F2_bootstrap_<tag>.csv,
P2_F2_reliability_<tag>.csv, P2_F2_annual_auc_<tag>.csv.  Seed 20261021 (bootstrap).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import (TAB, FROZEN_END, MAIN_METHODS, auc_reported, auc, bss, corp, pav, mbb_index, boot_block, write_csv)
from pre_common import murphy                                      # frozen (Stephenson within-bin terms)

SEED, B = 20261021, 2000
SAMPLES = {'2008-2026': lambda W: W.year >= 2008, '2009-2026': lambda W: W.year >= 2009,
           '2008-2020': lambda W: W.index <= FROZEN_END, '2009-2020': lambda W: (W.year >= 2009) & (W.index <= FROZEN_END),
           '2021-2026': lambda W: W.year >= 2021}


def wide(F, s, h):
    g = F[(F.set == s) & (F.horizon == h)]
    P = g.pivot_table(index='date', columns='method', values='p', aggfunc='first')
    meta = g.groupby('date').agg(y=('y', 'first'), cref=('clim', 'first'), year=('year', 'first'))
    return meta.join(P).sort_index()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='exp'); a = ap.parse_args()
    F = pd.read_csv(TAB / f'P2_F1_forecasts_{a.tag}.csv', parse_dates=['date'])
    F = F[F.headline.astype(bool) & F.y.notna() & F.p.notna()]
    rng = np.random.default_rng(SEED); dec, boot, rel, ann = [], [], [], []
    for s in ['common7', 'no_option9', 'full11']:
        for h in [5, 20]:
            W0 = wide(F, s, h); meths = [m for m in W0.columns if m not in ('y', 'cref', 'year')]
            for sname, sel in SAMPLES.items():
                W = W0[sel(W0)]; y = W.y.values
                for m in meths:
                    p = W[m].values; c = corp(y, p)
                    row = dict(set=s, horizon=h, sample=sname, method=m, n=c['n'], events=c['events'], BS=c['BS'],
                               BSS=bss(y, p, W.cref.values), MCB=c['MCB'], DSC=c['DSC'], UNC=c['UNC'],
                               corp_residual=c['identity_residual'],
                               AUC=auc(y, p) if auc_reported(m) else np.nan)
                    for bins in ['fixed10', 'quantile10']:
                        d = murphy(y, p, bins=bins)
                        for k in ['REL', 'RES', 'WBV', 'WBC', 'identity_residual']:
                            row[f'{k}_{bins}'] = d[k]
                    dec.append(row)
                if s != 'common7' or sname not in ('2008-2026', '2009-2026'):
                    continue
                idx = mbb_index(len(y), boot_block(h), B, rng)
                mm = [x for x in MAIN_METHODS if x in W]; st = {m: [] for m in mm}
                for ix in idx:                                 # every method on the same resamples (paired)
                    if y[ix].sum() == 0:
                        continue
                    for m in mm:
                        c = corp(y[ix], W[m].values[ix])
                        st[m].append((c['MCB'], c['DSC'], bss(y[ix], W[m].values[ix], W.cref.values[ix])))
                st = {m: np.array(v) for m, v in st.items()}
                for m in mm:
                    v = st[m]; dd = st['ou_raw'][:, 1] - v[:, 1]
                    boot.append(dict(horizon=h, sample=sname, method=m, B_used=len(v),
                                     MCB_q05=np.quantile(v[:, 0], .05), MCB_q95=np.quantile(v[:, 0], .95),
                                     DSC_q05=np.quantile(v[:, 1], .05), DSC_q95=np.quantile(v[:, 1], .95),
                                     BSS_q05=np.quantile(v[:, 2], .05), BSS_q95=np.quantile(v[:, 2], .95),
                                     dDSC_rawS_minus_m_q05=np.quantile(dd, .05), dDSC_rawS_minus_m_q95=np.quantile(dd, .95),
                                     share_dDSC_le_0=float(np.mean(dd <= 0))))
                for m in mm:
                    p = W[m].values
                    if sname == '2008-2026':                   # CORP reliability curve (PAV), on the unique forecasts
                        r = pav(y, p); o = np.argsort(p)
                        u = pd.DataFrame(dict(p=p[o], cep=r[o])).drop_duplicates()
                        u['horizon'] = h; u['method'] = m; rel.append(u)
            if s == 'common7':
                for m in meths:
                    if not auc_reported(m):
                        continue
                    for Y, g in W0.groupby('year'):
                        ann.append(dict(horizon=h, method=m, year=Y, n=len(g), events=int(g.y.sum()),
                                        AUC=auc(g.y, g[m]), constant=bool(np.ptp(g[m].values) < 1e-12)))
    D = pd.DataFrame(dec)
    assert D.corp_residual.abs().max() < 1e-10 and D[['identity_residual_fixed10', 'identity_residual_quantile10']].abs().max().max() < 1e-6
    write_csv(D, f'P2_F2_decomposition_{a.tag}.csv'); write_csv(pd.DataFrame(boot), f'P2_F2_bootstrap_{a.tag}.csv')
    write_csv(pd.concat(rel, ignore_index=True), f'P2_F2_reliability_{a.tag}.csv')
    write_csv(pd.DataFrame(ann), f'P2_F2_annual_auc_{a.tag}.csv')
    q = D[(D.set == 'common7') & D.method.isin(MAIN_METHODS) & D['sample'].isin(['2008-2026', '2009-2026'])]
    print(q.pivot_table(index='method', columns=['horizon', 'sample'], values=['MCB', 'DSC']).round(4).to_string())
    print('F2', a.tag, 'complete')


if __name__ == '__main__':
    main()
