"""P2_F1 -- matched forecasts on the gated S&P 500 feature table (frozen protocol of E3), plus score anatomy.

Usage   python scripts/P2_F1_forecasts.py            annual expanding windows 2008..2026 (headline; protocol of
                                                     E3_matched_estimators.py, unchanged)
        python scripts/P2_F1_forecasts.py --window 5 rolling five-year estimation windows and a five-year memory of
                                                     the sequential recalibration (finite-memory setting required
                                                     by the Giacomini-White asymptotics)
        python scripts/P2_F1_forecasts.py --labels-defined   F7 sensitivity (tag exp_ld): drop the first 500 complete
                                                     rows, on which the lagged RV quantile of the frozen label rule
                                                     is undefined and the labels are structurally zero
        python scripts/P2_F1_forecasts.py --check    reproduce the old E3 forecasts from the pre-repair feature
                                                     table (years 2008, 2016, 2026; every non-recalibrated method)
Protocol (frozen): complete-case sample on all 11 features; train-only standardisation, PCA(2), re-standardisation;
learned target B = upper training decile of s = z.w, w the PC direction most correlated with log RV20; S = P(hit B
within h days) in the fitted OU model (finite differences); operational label = first entry into the rule-based
stress set (RV20 above its lagged 1000-day 85% quantile and 20-day drawdown below -8%) within (t, t+h]; days already
in that set are not scored.  Methods: clim, pc2/raw logistic and GBT, ou_raw, msm_raw, amc_raw, GBT-CV, logistic C
sensitivity (common-7), and *_seqcal for the three dynamical scores.
Anatomy (new, common-7 only; nothing here changes a forecast): stress coordinate s and threshold thr of B, the two
PCs, the rank of S and of s within the same model's training-period values (u_S, u_s; mid-ranks, available at the
forecast origin), and the model's own target label hitB_h = 1{max_{k=1..h} s_{t+k} >= thr} computed with that
year's transforms.
Outputs (tag = exp | roll5): tables_p2/P2_F1_forecasts_<tag>.csv, P2_F1_anatomy_<tag>.csv, P2_F1_metrics_<tag>.csv,
P2_F1_gbt_cv_<tag>.csv, P2_F1_pca_loadings_<tag>.csv; --check writes P2_F1_reproduction.csv.
Randomness: scikit-learn random_state = 0 and deeptime KMeans fixed_seed = 0 (frozen); no other random numbers.
Units are cached per year in tables_p2/_cache (resumable); P2_BUDGET (seconds) limits the units started per call.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import (TAB, CACHE, OLD_FEATURES, OLD_E3, FULL, NO_OPTION, COMMON, FROZEN_END, auc_reported, auc, bss,
                       brier, write_csv)
from data_daily import crisis_labels                                           # frozen
from B_csf_additions import fit_ou_scores, safe_prob_logistic, safe_gbt         # frozen
from pre_common import sequential_recal                                        # frozen
from p2_e3_frozen import target_direction, msm_scores, amc_scores, gbt_cv       # verbatim copies (see module)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SETS = {'full11': FULL, 'no_option9': NO_OPTION, 'common7': COMMON}
YEARS = range(2008, 2027)
T0 = time.time(); BUDGET = float(os.environ.get('P2_BUDGET', '1e9'))


def load(path, labels_defined=False):
    F = pd.read_csv(path, parse_dates=['date'])
    F = F.dropna(subset=FULL).reset_index(drop=True)
    labels = {h: crisis_labels(F, h, rule='rv_dd')[0] for h in [5, 20]}
    _, inB, _ = crisis_labels(F, 20, rule='rv_dd'); F['inB'] = inB
    if labels_defined:   # F7 sensitivity: drop the rows on which the lagged RV quantile of the label rule is undefined
        keep = F.rv20.rolling(1000, min_periods=500).quantile(0.85).shift(1).notna().values
        F = F[keep].reset_index(drop=True); labels = {h: v[keep] for h, v in labels.items()}
    return F, labels


def midrank(train, x):
    s = np.sort(train)
    return (np.searchsorted(s, x, 'left') + np.searchsorted(s, x, 'right')) / (2.0 * len(s))


def year_unit(F, labels, Y, window=None, anatomy=True):
    lo = pd.Timestamp(f'{Y - window}-01-01') if window else pd.Timestamp('1900-01-01')
    trmask = (F.date < pd.Timestamp(f'{Y}-01-01')) & (F.date >= lo)
    temask = (F.date >= pd.Timestamp(f'{Y}-01-01')) & (F.date < pd.Timestamp(f'{Y + 1}-01-01'))
    tr = F[trmask]; te = F[temask]
    tridx = np.where(trmask)[0]; teidx = np.where(temask)[0]
    rows, cfg_rows, load_rows, anat = [], [], [], []
    for sname, cols in SETS.items():
        scaler = StandardScaler().fit(tr[cols]); Xtr = scaler.transform(tr[cols]); Xte = scaler.transform(te[cols])
        pca = PCA(n_components=2, random_state=0).fit(Xtr); Ztr = pca.transform(Xtr); Zte = pca.transform(Xte)
        zsc = StandardScaler().fit(Ztr); Ztr = zsc.transform(Ztr); Zte = zsc.transform(Zte)
        for j in range(2):
            for k, c in enumerate(cols):
                load_rows.append(dict(year=Y, set=sname, pc=j + 1, feature=c, loading=pca.components_[j, k],
                                      explained=pca.explained_variance_ratio_[j]))
        ou, _, _ = fit_ou_scores(Ztr, Zte, tr['log_rv20'].values)
        if sname == 'common7':
            w, thr = target_direction(Ztr, tr['log_rv20'].values)
            msm = msm_scores(Ztr, Zte, w, thr, (5, 20)); amc = amc_scores(Ztr, Zte, w, thr, (5, 20))
            if anatomy:                                   # same transforms applied to the test year and 20 days after
                ext = np.arange(teidx[0], min(teidx[-1] + 21, len(F)))
                Zx = zsc.transform(pca.transform(scaler.transform(F.iloc[ext][cols])))
                sx = Zx @ w; s_tr = Ztr @ w; pos = teidx - teidx[0]
        for h in [5, 20]:
            ytr = labels[h][tridx].copy(); yte = labels[h][teidx].copy()
            oktr = np.isfinite(ytr)
            if len(oktr) >= h:
                oktr[-h:] = False
            clim = float(np.nanmean(ytr[oktr])) if np.any(oktr) else 0.01
            yi = ytr[oktr].astype(int)
            M = {'clim': np.full(len(te), clim),
                 'pc2_logistic': safe_prob_logistic(Ztr[oktr], yi, Zte, C=1.0), 'pc2_gbt': safe_gbt(Ztr[oktr], yi, Zte),
                 'raw_logistic': safe_prob_logistic(Xtr[oktr], yi, Xte, C=0.2), 'raw_gbt': safe_gbt(Xtr[oktr], yi, Xte),
                 'ou_raw': ou[h][1]}
            if sname == 'common7':
                M['msm_raw'] = msm[h][1]; M['amc_raw'] = amc[h][1]
                M['pc2_gbt_cv'], c1 = gbt_cv(Ztr[oktr], yi, Zte, h); M['raw_gbt_cv'], c2 = gbt_cv(Xtr[oktr], yi, Xte, h)
                for Cc in [0.1, 1.0, 10.0]:
                    M[f'pc2_logistic_C{Cc:g}'] = safe_prob_logistic(Ztr[oktr], yi, Zte, C=Cc)
                    M[f'raw_logistic_C{Cc:g}'] = safe_prob_logistic(Xtr[oktr], yi, Xte, C=Cc)
                cfg_rows.append(dict(year=Y, horizon=h, gbt_cv_pc2=c1, gbt_cv_raw=c2))
                if anatomy:
                    hit = np.full(len(te), np.nan)
                    for i, p0 in enumerate(pos):
                        fut = sx[p0 + 1:p0 + h + 1]
                        if len(fut) == h:
                            hit[i] = float(fut.max() >= thr)
                    S_tr = ou[h][0]
                    anat.append(pd.DataFrame(dict(date=te.date.values, year=Y, horizon=h, S=ou[h][1], s=Zte @ w, thr=thr,
                                                  z1=Zte[:, 0], z2=Zte[:, 1], u_S=midrank(S_tr, ou[h][1]),
                                                  u_s=midrank(s_tr, Zte @ w), S_train_mean=float(S_tr.mean()),
                                                  share_train_S_ge_0999=float(np.mean(S_tr >= 0.999)),
                                                  hitB=hit, y=yte, inB=te.inB.values, w1=w[0], w2=w[1],
                                                  n_train=len(tr))))
            score_ok = np.isfinite(yte) & (~te['inB'].values.astype(bool))
            for m, p in M.items():
                rows.append(pd.DataFrame(dict(date=te.date.values, year=Y, set=sname, horizon=h, method=m, p=p, y=yte,
                                              headline=score_ok, inB=te.inB.values, clim=clim)))
    A = pd.concat(anat, ignore_index=True) if anat else pd.DataFrame()
    return pd.concat(rows, ignore_index=True), pd.DataFrame(cfg_rows), pd.DataFrame(load_rows), A


def seqcal(Fc, memory=None):
    """Strict sequential recalibration of the dynamical scores; history = prior years' headline OOS raw scores
    (all prior years, or the last `memory` years)."""
    out = []
    for (s, h), g in Fc.groupby(['set', 'horizon']):
        for dyn in [m for m in ['ou', 'msm', 'amc'] if f'{m}_raw' in set(g.method)]:
            gr = g[g.method == f'{dyn}_raw']; cal = []
            for Y in sorted(gr.year.unique()):
                lo = Y - memory if memory else -10 ** 9
                prior = gr[(gr.year < Y) & (gr.year >= lo) & gr.headline & gr.y.notna()]; cur = gr[gr.year == Y]
                p = sequential_recal(prior.p.values, prior.y.values, cur.p.values, float(cur.clim.iloc[0]))
                cal.append(cur.assign(method=f'{dyn}_seqcal', p=p))
            out.append(pd.concat(cal))
    return pd.concat(out, ignore_index=True)


SAMPLES = [('2008-2026', lambda g: g.year >= 2008), ('2009-2026', lambda g: g.year >= 2009),
           ('2008-2020', lambda g: g.date <= FROZEN_END), ('2009-2020', lambda g: (g.year >= 2009) & (g.date <= FROZEN_END)),
           ('2021-2026', lambda g: g.year >= 2021)]


def metrics(Fc):
    H = Fc[Fc.headline.astype(bool) & Fc.y.notna()]; met = []
    for (s, h, m), g in H.groupby(['set', 'horizon', 'method']):
        for sample, sel in SAMPLES:
            q = g[sel(g)]
            if len(q) < 50 or q.y.sum() < 3:
                continue
            met.append(dict(set=s, horizon=h, method=m, sample=sample, n=len(q), events=int(q.y.sum()),
                            BS=brier(q.y, q.p), BSS=bss(q.y, q.p, q.clim),
                            AUC=auc(q.y, q.p) if auc_reported(m) else np.nan))
    return pd.DataFrame(met)


def check():
    F, labels = load(OLD_FEATURES)
    old = pd.read_csv(OLD_E3, parse_dates=['date']); rows = []
    for Y in [2008, 2016, 2026]:
        new = year_unit(F, labels, Y, anatomy=False)[0]
        mg = old[old.year == Y].merge(new, on=['date', 'set', 'horizon', 'method'], suffixes=('_old', '_new'))
        for (s, h, m), d in mg.groupby(['set', 'horizon', 'method']):
            # the old file stores p with '%.10g'; round the new values the same way before comparing
            p_new10 = np.array([float(f'{x:.10g}') for x in d.p_new])
            rows.append(dict(year=Y, set=s, horizon=h, method=m, n=len(d),
                             max_abs_dp=float(np.nanmax(np.abs(d.p_old - d.p_new))),
                             max_abs_dp_at_stored_precision=float(np.nanmax(np.abs(d.p_old.values - p_new10))),
                             labels_equal=bool(((d.y_old == d.y_new) | (d.y_old.isna() & d.y_new.isna())).all())))
    R = pd.DataFrame(rows); write_csv(R, 'P2_F1_reproduction.csv')
    print(R.groupby('year').agg(units=('n', 'size'), max_abs_dp=('max_abs_dp', 'max'),
                                at_stored_precision=('max_abs_dp_at_stored_precision', 'max'),
                                labels_equal=('labels_equal', 'all')))
    assert R.max_abs_dp_at_stored_precision.max() < 1e-15 and R.labels_equal.all(), 'old E3 forecasts not reproduced'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--window', type=int, default=None); ap.add_argument('--check', action='store_true')
    ap.add_argument('--labels-defined', action='store_true')
    a = ap.parse_args()
    if a.check:
        return check()
    tag = (f'roll{a.window}' if a.window else 'exp') + ('_ld' if a.labels_defined else '')
    F, labels = load(TAB / 'features_SPX_r1.csv', labels_defined=a.labels_defined)
    for Y in YEARS:
        f = CACHE / f'F1_{tag}_{Y}.pkl'
        if f.exists():
            continue
        if time.time() - T0 > BUDGET:
            print('budget reached; rerun to continue'); return
        t = time.time(); pd.to_pickle(year_unit(F, labels, Y, window=a.window), f)
        print(tag, Y, f'{time.time() - t:.0f}s', flush=True)
    parts = [pd.read_pickle(CACHE / f'F1_{tag}_{Y}.pkl') for Y in YEARS]
    Fc = pd.concat([p[0] for p in parts], ignore_index=True)
    Fc = pd.concat([Fc, seqcal(Fc, memory=a.window)], ignore_index=True).sort_values(['set', 'horizon', 'method', 'date'])
    Fc.to_csv(TAB / f'P2_F1_forecasts_{tag}.csv', index=False, float_format='%.10g')
    write_csv(pd.concat([p[3] for p in parts], ignore_index=True), f'P2_F1_anatomy_{tag}.csv')
    write_csv(pd.concat([p[1] for p in parts], ignore_index=True), f'P2_F1_gbt_cv_{tag}.csv')
    write_csv(pd.concat([p[2] for p in parts], ignore_index=True), f'P2_F1_pca_loadings_{tag}.csv')
    M = metrics(Fc); write_csv(M, f'P2_F1_metrics_{tag}.csv')
    q = M[(M.set == 'common7') & M.method.isin(['ou_raw', 'ou_seqcal', 'pc2_logistic', 'pc2_gbt'])]
    print(q.pivot_table(index='method', columns=['horizon', 'sample'], values='BSS').round(3).to_string())
    print('F1', tag, 'complete')


if __name__ == '__main__':
    main()
