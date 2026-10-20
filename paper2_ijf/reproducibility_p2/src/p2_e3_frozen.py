"""Verbatim copies of four helpers of the frozen Paper-B script reproducibility_pre/scripts/E3_matched_estimators.py
(sha256 4527edeb2df0af8de706ddda934f59a464077e0385243014ce791eabbaaad167): target_direction, msm_scores, amc_scores, gbt_cv, with the constants they use.  Copied rather than
imported because importing E3 creates a cache directory inside the read-only Paper 1 tree.  P2_F1_forecasts.py --check
verifies that the forecasts on the pre-repair feature table reproduce the old E3 forecasts exactly at stored precision.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

N_MSM, K_AMC = 100, 20
GRID = [(1, 120), (2, 60), (2, 120), (2, 240), (3, 120)]          # (max_depth, n_estimators)


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
