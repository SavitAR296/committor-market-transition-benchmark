"""E3 -- matched estimators on the extended S&P sample (annual expanding windows, 2008..2026).

Frozen protocol (B_csf_additions.py, unchanged): complete-case sample on all 11 features; train-only standardisation,
PCA(2) and re-standardisation; learned target B = upper training decile along the PC direction most correlated with
log RV20; operational label = first entry into the frozen stress set within (t, t+h]; days already in the set excluded.
Methods (all sets): ou_raw/ou_seqcal, pc2_logistic (C=1), pc2_gbt, raw_logistic (C=0.2), raw_gbt  [frozen settings]
Added on the headline common-7 state:
  msm_raw/msm_seqcal  k-means (deeptime KMeans, 100 centres, fixed_seed=0) on training PCs, lag 1 day,
                      non-reversible MLE P_ij = C_ij / sum_k C_ik (= deeptime MaximumLikelihoodMSM(reversible=False)),
                      B absorbing (centres inside B), v_h by h matrix-vector steps, test point -> nearest centre.
  amc_raw/amc_seqcal  analogue Markov chain (Lucente et al. 2022): from a state jump to the successor of one of its
                      K_AMC=20 nearest training analogues; v_h by backward recursion on the training points.
  pc2_gbt_cv/raw_gbt_cv  GradientBoostingClassifier tuned by 3-fold TimeSeriesSplit (gap = h) inside the training
                      period over GRID (learning_rate 0.03, min_samples_leaf 20, subsample 0.9, random_state 0).
  pc2_logistic_C*/raw_logistic_C*  L2 sensitivity C in {0.1, 1, 10}.
Sequential calibration of every dynamical score uses the frozen rule (<100 prior OOS obs, <5 events or <20 non-events
-> training climatology).  Per-year units are cached in tables_pre/_cache (resumable; 300-s command limit).
Outputs: tables_pre/E3_forecasts_2026.csv (long), E3_matched_estimators.csv (metrics), E3_frozen_reproduction.csv,
         E3_pca_loadings_2026.csv
"""
import sys, os, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; LEG = ROOT.parent / 'reproducibility'
for p in [ROOT / 'src', LEG / 'src', LEG / 'experiments']:
    sys.path.insert(0, str(p))
from data_daily import crisis_labels
from B_csf_additions import (fit_ou_scores, safe_prob_logistic, safe_gbt, OUModel, grid_from_train, interp_grid,
                             FULL, NO_OPTION, COMMON)
from pre_common import sequential_recal, bss, auc, brier, write_csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

SETS = {'full11': FULL, 'no_option9': NO_OPTION, 'common7': COMMON}
N_MSM, K_AMC = 100, 20
GRID = [(1, 120), (2, 60), (2, 120), (2, 240), (3, 120)]          # (max_depth, n_estimators)
CACHE = ROOT / 'tables_pre' / '_cache'; CACHE.mkdir(exist_ok=True)
T0 = time.time(); BUDGET = float(os.environ.get('PRE_BUDGET', '200'))
FROZEN_END = pd.Timestamp('2020-04-17')


def load():
    F = pd.read_csv(ROOT / 'tables_pre' / 'features_SPX_2026.csv', parse_dates=['date'])
    F = F.dropna(subset=FULL).reset_index(drop=True)
    labels = {h: crisis_labels(F, h, rule='rv_dd')[0] for h in [5, 20]}
    _, inB, _ = crisis_labels(F, 20, rule='rv_dd'); F['inB'] = inB
    return F, labels


def target_direction(Ztr, lrv):
    cors = np.array([np.corrcoef(Ztr[:, j], lrv)[0, 1] for j in range(2)])
    if not np.all(np.isfinite(cors)) or np.linalg.norm(cors) < 1e-9:
        cors = np.array([1., 0.])
    w = cors / np.linalg.norm(cors)
    return w, np.quantile(Ztr @ w, 0.90)


def msm_scores(Ztr, Zte, w, thr, hs):
    try:
        from deeptime.clustering import KMeans
        km = KMeans(n_clusters=N_MSM, init_strategy='kmeans++', fixed_seed=0, max_iter=500, n_jobs=1).fit(Ztr).fetch_model()
        centers = km.cluster_centers; dtr = km.transform(Ztr); dte = km.transform(Zte)
    except Exception:                                   # deterministic fallback (recorded)
        from sklearn.cluster import KMeans as SK
        km = SK(n_clusters=N_MSM, n_init=1, random_state=0).fit(Ztr); centers = km.cluster_centers_
        dtr = km.predict(Ztr); dte = km.predict(Zte)
    C = np.zeros((N_MSM, N_MSM)); np.add.at(C, (dtr[:-1], dtr[1:]), 1.0)
    rs = C.sum(1); P = np.where(rs[:, None] > 0, C / np.maximum(rs[:, None], 1), np.eye(N_MSM))
    inB = (centers @ w) >= thr
    out = {}; v = inB.astype(float)
    for k in range(1, max(hs) + 1):
        v = np.where(inB, 1.0, P @ v)
        if k in hs:
            out[k] = (v[dtr], v[dte])
    return out


def amc_scores(Ztr, Zte, w, thr, hs):
    n = len(Ztr); base = Ztr[:-1]                      # analogues must have a successor
    nn = NearestNeighbors(n_neighbors=K_AMC).fit(base)
    idx_tr = nn.kneighbors(Ztr, return_distance=False); idx_te = nn.kneighbors(Zte, return_distance=False)
    inB_tr = (Ztr @ w) >= thr; inB_te = (Zte @ w) >= thr
    u = inB_tr.astype(float); out = {}
    for k in range(1, max(hs) + 1):
        if k in hs:
            vte = np.where(inB_te, 1.0, u[idx_te + 1].mean(1)); vtr = np.where(inB_tr, 1.0, u[idx_tr + 1].mean(1))
            out[k] = (vtr, vte)
        u = np.where(inB_tr, 1.0, u[idx_tr + 1].mean(1))
    return out


def gbt_cv(Xtr, ytr, Xte, h):
    if np.sum(ytr == 1) < 3 or np.sum(ytr == 0) < 3:
        return np.full(len(Xte), np.mean(ytr)), None
    best, best_cfg = np.inf, GRID[2]
    for d, ne in GRID:
        losses = []
        for a, b in TimeSeriesSplit(n_splits=3, gap=h).split(Xtr):
            ya = ytr[a]
            if ya.min() == ya.max():
                p = np.full(len(b), ya.mean())
            else:
                m = GradientBoostingClassifier(n_estimators=ne, learning_rate=0.03, max_depth=d, min_samples_leaf=20,
                                               subsample=0.9, random_state=0).fit(Xtr[a], ya)
                p = m.predict_proba(Xtr[b])[:, 1]
            losses.append(np.mean((p - ytr[b]) ** 2))
        if np.mean(losses) < best:
            best, best_cfg = np.mean(losses), (d, ne)
    d, ne = best_cfg
    m = GradientBoostingClassifier(n_estimators=ne, learning_rate=0.03, max_depth=d, min_samples_leaf=20,
                                   subsample=0.9, random_state=0).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], f'{d}-{ne}'


def year_unit(F, labels, Y):
    trmask = F.date < pd.Timestamp(f'{Y}-01-01'); temask = (F.date >= pd.Timestamp(f'{Y}-01-01')) & (F.date < pd.Timestamp(f'{Y + 1}-01-01'))
    tr = F[trmask]; te = F[temask]
    tridx = np.where(trmask)[0]; teidx = np.where(temask)[0]
    rows, raw_rows, load_rows = [], [], []
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
        for h in [5, 20]:
            ytr = labels[h][tridx].copy(); yte = labels[h][teidx].copy()
            oktr = np.isfinite(ytr)
            if len(oktr) >= h: oktr[-h:] = False
            clim = float(np.nanmean(ytr[oktr])) if np.any(oktr) else 0.01
            yi = ytr[oktr].astype(int)
            M = {'clim': np.full(len(te), clim),
                 'pc2_logistic': safe_prob_logistic(Ztr[oktr], yi, Zte, C=1.0), 'pc2_gbt': safe_gbt(Ztr[oktr], yi, Zte),
                 'raw_logistic': safe_prob_logistic(Xtr[oktr], yi, Xte, C=0.2), 'raw_gbt': safe_gbt(Xtr[oktr], yi, Xte),
                 'ou_raw': ou[h][1]}
            if sname == 'common7':
                M['msm_raw'] = msm[h][1]; M['amc_raw'] = amc[h][1]
                M['pc2_gbt_cv'], cfg1 = gbt_cv(Ztr[oktr], yi, Zte, h); M['raw_gbt_cv'], cfg2 = gbt_cv(Xtr[oktr], yi, Xte, h)
                for Cc in [0.1, 1.0, 10.0]:
                    M[f'pc2_logistic_C{Cc:g}'] = safe_prob_logistic(Ztr[oktr], yi, Zte, C=Cc)
                    M[f'raw_logistic_C{Cc:g}'] = safe_prob_logistic(Xtr[oktr], yi, Xte, C=Cc)
                raw_rows.append(dict(year=Y, horizon=h, gbt_cv_pc2=cfg1, gbt_cv_raw=cfg2))
            score_ok = np.isfinite(yte) & (~te['inB'].values.astype(bool))
            for m, p in M.items():
                rows.append(pd.DataFrame(dict(date=te.date.values, year=Y, set=sname, horizon=h, method=m, p=p, y=yte,
                                              headline=score_ok, inB=te.inB.values, clim=clim)))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(raw_rows), pd.DataFrame(load_rows)


def assemble():
    parts = [pd.read_pickle(CACHE / f'E3_{Y}.pkl') for Y in range(2008, 2027)]
    Fc = pd.concat([p[0] for p in parts], ignore_index=True)
    # strict sequential calibration of dynamical scores (history = prior years' headline OOS raw scores)
    out = [Fc]
    for (s, h), g in Fc.groupby(['set', 'horizon']):
        for dyn in [m for m in ['ou', 'msm', 'amc'] if f'{m}_raw' in set(g.method)]:
            gr = g[g.method == f'{dyn}_raw']; cal = []
            for Y in sorted(gr.year.unique()):
                prior = gr[(gr.year < Y) & gr.headline & gr.y.notna()]; cur = gr[gr.year == Y]
                p = sequential_recal(prior.p.values, prior.y.values, cur.p.values, float(cur.clim.iloc[0]))
                cal.append(cur.assign(method=f'{dyn}_seqcal', p=p))
            out.append(pd.concat(cal))
    Fc = pd.concat(out, ignore_index=True).sort_values(['set', 'horizon', 'method', 'date']).reset_index(drop=True)
    Fc.to_csv(ROOT / 'tables_pre' / 'E3_forecasts_2026.csv', index=False, float_format='%.10g')
    write_csv(pd.concat([p[1] for p in parts], ignore_index=True), 'E3_gbt_cv_choices.csv')
    write_csv(pd.concat([p[2] for p in parts], ignore_index=True), 'E3_pca_loadings_2026.csv')
    # metrics on four samples
    H = Fc[Fc.headline.astype(bool) & Fc.y.notna()]
    met = []
    for (s, h, m), g in H.groupby(['set', 'horizon', 'method']):
        for sample, sel in [('2008-2020', g.date <= FROZEN_END), ('2009-2020', (g.date <= FROZEN_END) & (g.year >= 2009)),
                            ('2008-2026', g.year >= 2008), ('2009-2026', g.year >= 2009), ('2021-2026', g.year >= 2021)]:
            q = g[sel]
            if len(q) < 50 or q.y.sum() < 3:
                continue
            met.append(dict(set=s, horizon=h, method=m, sample=sample, n=len(q), events=int(q.y.sum()), AUC=auc(q.y, q.p),
                            Brier=brier(q.y, q.p), BSS=bss(q.y, q.p, q.clim)))
    write_csv(pd.DataFrame(met), 'E3_matched_estimators.csv')
    # reproduction of the frozen forecasts on common dates
    Z = pd.read_csv(LEG / 'tables_csf' / 'B_feature_attribution_forecasts.csv', parse_dates=['date'])
    Mg = Z.merge(Fc, on=['date', 'set', 'horizon', 'method'], suffixes=('_frozen', '_new'))
    rep = Mg.groupby(['set', 'horizon', 'method']).apply(lambda d: pd.Series(dict(
        n=len(d), max_abs_dp=float(np.nanmax(np.abs(d.p_frozen - d.p_new))),
        labels_equal=bool(((d.y_frozen == d.y_new) | d.y_frozen.isna()).all()),
        headline_equal_where_labelled=bool((d.headline_frozen == d.headline_new)[d.y_frozen.notna()].all())))).reset_index()
    write_csv(rep, 'E3_frozen_reproduction.csv')
    print(rep.groupby('method').max_abs_dp.max().to_string())


def main():
    F, labels = load()
    for Y in range(2008, 2027):
        f = CACHE / f'E3_{Y}.pkl'
        if f.exists():
            continue
        if time.time() - T0 > BUDGET:
            print('budget reached; rerun to continue'); return
        t = time.time(); pd.to_pickle(year_unit(F, labels, Y), f); print('year', Y, f'{time.time() - t:.0f}s', flush=True)
    assemble(); print('E3 complete')


if __name__ == '__main__':
    main()
