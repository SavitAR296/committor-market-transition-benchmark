"""T1 -- Information-ledger check on the frozen forecasts (Proposition 1, standard).

Outputs (tables_pre/):
  T1_murphy_check.csv        Murphy decomposition, every set x horizon x method x sample x binning;
                             column identity_residual must be < 1e-6 in absolute value.
  T1_seqcal_reconstruction.csv  reconstruction of the frozen sequential calibration from frozen raw scores
                             (max abs deviation) and the alternative link functions (logit-S, isotonic).
  T1_annual_auc.csv          annual AUC of raw and seq-cal OU (shows the pooled seq-cal AUC is a pooling artefact).
  T1_score_saturation.csv    days with raw score S>0.99 and their events, by year and horizon.
  T1_pilot_verification.csv  every pilot number quoted in Sec. 1 of the rebuild outline, recomputed.
No randomness. Runtime < 1 min.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pre_common import (load_forecasts, wide, murphy, bss, auc, sequential_recal, write_csv)

METHODS = ['ou_raw', 'ou_seqcal', 'pc2_logistic', 'pc2_gbt', 'raw_logistic', 'raw_gbt', 'clim']


def main():
    F = load_forecasts()
    rows = []
    for s in ['common7', 'no_option9', 'full11']:
        for h in [5, 20]:
            W = wide(F, s, h)
            for sample, mask in [('all', np.ones(len(W), bool)), ('ex2008', (W.year != 2008).values)]:
                Ws = W[mask]
                for m in METHODS:
                    for bins in ['fixed10', 'quantile10']:
                        d = murphy(Ws.y.values, Ws[m].values, bins=bins)
                        d.update(set=s, horizon=h, sample=sample, method=m, binning=bins,
                                 BSS=bss(Ws.y.values, Ws[m].values, Ws.cref.values),
                                 AUC=auc(Ws.y.values, Ws[m].values))
                        rows.append(d)
    M = pd.DataFrame(rows)
    cols = ['set', 'horizon', 'sample', 'method', 'binning', 'n', 'events', 'BS', 'REL', 'RES', 'UNC', 'WBV', 'WBC',
            'identity_residual', 'nbins', 'BSS', 'AUC']
    M = M[cols]
    write_csv(M, 'T1_murphy_check.csv')
    assert M.identity_residual.abs().max() < 1e-6, 'Murphy identity violated'

    # ---- reconstruct the strict sequential calibration from the frozen raw scores
    rec = []
    alt_rows = []
    for s in ['common7', 'no_option9', 'full11']:
        for h in [5, 20]:
            W = wide(F, s, h)
            years = sorted(W.year.unique())
            out = {k: np.full(len(W), np.nan) for k in ['logistic', 'logit_logistic', 'isotonic']}
            for Y in years:
                prior = W[W.year < Y]; cur = (W.year == Y).values
                clim = W.cref[cur].iloc[0]
                for k in out:
                    out[k][cur] = sequential_recal(prior.ou_raw.values, prior.y.values, W.ou_raw[cur].values, clim, kind=k)
            dev = np.max(np.abs(out['logistic'] - W.ou_seqcal.values))
            rec.append(dict(set=s, horizon=h, max_abs_dev_vs_frozen=dev))
            for sample, mask in [('all', np.ones(len(W), bool)), ('ex2008', (W.year != 2008).values)]:
                for k, p in out.items():
                    alt_rows.append(dict(set=s, horizon=h, sample=sample, calibrator=k,
                                         BSS=bss(W.y.values[mask], p[mask], W.cref.values[mask])))
                alt_rows.append(dict(set=s, horizon=h, sample=sample, calibrator='(PC2 logistic, reference)',
                                     BSS=bss(W.y.values[mask], W.pc2_logistic.values[mask], W.cref.values[mask])))
    # eps sweep for the logit-S link (common-7 only)
    for h in [5, 20]:
        W = wide(F, 'common7', h)
        for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]:
            p = np.full(len(W), np.nan)
            for Y in sorted(W.year.unique()):
                prior = W[W.year < Y]; cur = (W.year == Y).values
                p[cur] = sequential_recal(prior.ou_raw.values, prior.y.values, W.ou_raw[cur].values,
                                          W.cref[cur].iloc[0], kind='logit_logistic', eps=eps)
            alt_rows.append(dict(set='common7', horizon=h, sample='all', calibrator=f'logit_logistic_eps={eps:g}',
                                 BSS=bss(W.y.values, p, W.cref.values)))
    R = pd.DataFrame(rec); A = pd.DataFrame(alt_rows)
    write_csv(R.merge(A, on=['set', 'horizon'], how='right'), 'T1_seqcal_reconstruction.csv')
    assert R.max_abs_dev_vs_frozen.max() < 1e-8, 'sequential calibration not reproduced'

    # ---- annual AUC and saturation table
    ann, sat = [], []
    for s in ['common7']:
        for h in [5, 20]:
            W = wide(F, s, h)
            for Y, g in W.groupby('year'):
                ann.append(dict(set=s, horizon=h, year=Y, n=len(g), events=int(g.y.sum()),
                                AUC_raw=auc(g.y, g.ou_raw), AUC_seqcal=auc(g.y, g.ou_seqcal),
                                seqcal_is_constant=bool(np.ptp(g.ou_seqcal.values) < 1e-12)))
                hi = g.ou_raw > 0.99
                sat.append(dict(set=s, horizon=h, year=Y, days_S_gt_099=int(hi.sum()), events_S_gt_099=int(g.y[hi].sum()),
                                event_rate_S_gt_099=float(g.y[hi].mean()) if hi.sum() else np.nan))
    write_csv(pd.DataFrame(ann), 'T1_annual_auc.csv')
    write_csv(pd.DataFrame(sat), 'T1_score_saturation.csv')

    # ---- pilot verification (Sec. 1 of the outline)
    def get(df, **kw):
        q = df
        for k, v in kw.items():
            q = q[q[k] == v]
        assert len(q) == 1, kw
        return q.iloc[0]
    pv = []
    def add(item, quantity, pilot, value):
        pv.append(dict(item=item, quantity=quantity, pilot=pilot, recomputed=value,
                       agrees=bool(np.isfinite(pilot) and abs(round(value, 3) - pilot) <= 0.0015)))
    for h, pairs in [(20, [('ou_raw', .0507, .0105), ('ou_seqcal', .0050, .0022), ('pc2_logistic', .0011, .0070),
                           ('pc2_gbt', .0021, .0072)]),
                     (5, [('ou_raw', .0554, .0048), ('ou_seqcal', .0001, .0020), ('pc2_gbt', .0004, .0053),
                          ('pc2_logistic', .0013, .0034)])]:
        for m, rel, res in pairs:
            r = get(M, set='common7', horizon=h, sample='all', method=m, binning='fixed10')
            add(f'1/2 Murphy h={h}', f'{m} REL', rel, r.REL); add(f'1/2 Murphy h={h}', f'{m} RES', res, r.RES)
        r = get(M, set='common7', horizon=h, sample='all', method='ou_raw', binning='fixed10')
        add(f'1/2 Murphy h={h}', 'UNC', {20: .0826, 5: .0322}[h], r.UNC)
    W5 = wide(F, 'common7', 5); W20 = wide(F, 'common7', 20)
    add('3 cold start', 'events 2008 h=5', 33, W5.y[W5.year == 2008].sum())
    add('3 cold start', 'events total h=5', 97, W5.y.sum())
    add('3 cold start', 'events 2008 h=20', 75, W20.y[W20.year == 2008].sum())
    add('3 cold start', 'events total h=20', 264, W20.y.sum())
    add('3 cold start', 'obs 2008', 187, (W5.year == 2008).sum())
    add('3 cold start', 'obs total h=5', 2910, len(W5))
    for h, vals in [(5, dict(ou_seqcal=.116, pc2_logistic=.067, raw_logistic=.053, pc2_gbt=.151)),
                    (20, dict(ou_seqcal=.007, pc2_logistic=.061, raw_logistic=.063))]:
        for m, v in vals.items():
            add('4 ex-2008 BSS', f'h={h} {m}', v, get(M, set='common7', horizon=h, sample='ex2008', method=m, binning='fixed10').BSS)
    AN = pd.DataFrame(ann)
    for Y, (a, b) in {2009: (.943, .944), 2010: (.967, .971), 2015: (.799, .799), 2016: (.974, .974),
                      2018: (.821, .821), 2020: (.806, .806)}.items():
        r = AN[(AN.horizon == 5) & (AN.year == Y)].iloc[0]
        add('5 annual AUC h=5', f'{Y} raw', a, r.AUC_raw); add('5 annual AUC h=5', f'{Y} seq-cal', b, r.AUC_seqcal)
    for h, vals in [(20, dict(logistic=.005, logit_logistic=-.006, isotonic=.025)),
                    (5, dict(logistic=.077, logit_logistic=.080, isotonic=.072))]:
        for k, v in vals.items():
            q = A[(A.set == 'common7') & (A.horizon == h) & (A['sample'] == 'all') & (A.calibrator == k)].iloc[0]
            add('6 link independence', f'h={h} {k}', v, q.BSS)
    SA = pd.DataFrame(sat)
    for Y, (d, e) in {2008: (105, 26), 2011: (29, 2), 2015: (12, 7), 2016: (12, 4), 2020: (7, 3)}.items():
        r = SA[(SA.horizon == 5) & (SA.year == Y)].iloc[0]
        add('7 S>0.99 h=5', f'{Y} days', d, r.days_S_gt_099); add('7 S>0.99 h=5', f'{Y} events', e, r.events_S_gt_099)
    PV = pd.DataFrame(pv)
    PV['agrees'] = [abs(r.recomputed - r.pilot) <= (0.5 if abs(r.pilot) >= 1 else 0.0015) for r in PV.itertuples()]
    write_csv(PV, 'T1_pilot_verification.csv')
    print(R.to_string(index=False))
    print(PV.to_string(index=False))
    print('max |identity residual| =', M.identity_residual.abs().max())


if __name__ == '__main__':
    main()
