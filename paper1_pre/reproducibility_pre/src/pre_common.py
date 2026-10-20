"""Shared utilities for the PRE reproducibility package (reproducibility_pre/).

Conventions
-----------
* Frozen inputs live in ../reproducibility/ and are opened read-only.
* Every script writes CSV to tables_pre/ and figures to figures_pre/ only.
* "raw" score    = OU finite-horizon hitting probability v_h^B (method 'ou_raw').
* "seq-cal"      = strictly sequential logistic recalibration ('ou_seqcal').
* "common-7"     = the seven-feature state shared by all feature sets (headline).
* BSS            = 1 - BS / BS(year-specific training climatology), as in the frozen paper.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PRE = Path(__file__).resolve().parents[1]           # reproducibility_pre/
LEGACY = PRE.parent / "reproducibility"               # frozen CSF package (read-only)
TABLES = PRE / "tables_pre"
FIGS = PRE / "figures_pre"
TABLES.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

FROZEN_FORECASTS = LEGACY / "tables_csf" / "B_feature_attribution_forecasts.csv"
FROZEN_SYNTH = LEGACY / "tables_csf" / "B_synthetic_phase.csv"
FROZEN_PCA = LEGACY / "tables_csf" / "B_pca_loadings.csv"
FROZEN_FEATURES = LEGACY / "tables" / "features_SPX.csv"

FULL = ['log_rv20', 'log_rv5', 'log_act_rel', 'dd20', 'ret20', 'one_minus_n', 'n_sd',
        'ews_var', 'ews_ar1', 'log_vix', 'vrp']
NO_OPTION = [c for c in FULL if c not in ['log_vix', 'vrp']]
COMMON = [c for c in NO_OPTION if c not in ['one_minus_n', 'n_sd']]

METHOD_LABELS = {
    'ou_raw': 'OU raw', 'ou_seqcal': 'OU seq-cal', 'pc2_logistic': 'PC2 logistic',
    'pc2_gbt': 'PC2 GBT', 'raw_logistic': 'raw logistic', 'raw_gbt': 'raw GBT', 'clim': 'climatology',
    'msm_raw': 'MSM raw', 'msm_seqcal': 'MSM seq-cal', 'amc_raw': 'AMC raw', 'amc_seqcal': 'AMC seq-cal',
    'pc2_gbt_cv': 'PC2 GBT (CV)', 'raw_gbt_cv': 'raw GBT (CV)'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_forecasts(path: Path = FROZEN_FORECASTS, headline_only: bool = True) -> pd.DataFrame:
    """Frozen annual expanding-window forecasts (long format)."""
    F = pd.read_csv(path, parse_dates=['date'])
    if headline_only:
        F = F[F.headline.astype(bool) & F.y.notna() & F.p.notna()].copy()
    return F


def wide(F: pd.DataFrame, s: str, h: int) -> pd.DataFrame:
    """One row per date: y, cref (row-wise training climatology used as BSS reference), year, one column per method."""
    g = F[(F.set == s) & (F.horizon == h)]
    P = g.pivot_table(index='date', columns='method', values='p', aggfunc='first')
    meta = g.groupby('date').agg(y=('y', 'first'), cref=('clim', 'first'), year=('year', 'first'))
    return meta.join(P).sort_index()


# ----------------------------------------------------------------------------- scores
def brier(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def bss(y, p, clim):
    """BSS against a (possibly row-wise) reference climatology forecast."""
    y = np.asarray(y, float)
    ref = np.mean((y - np.asarray(clim, float)) ** 2)
    return float(1.0 - brier(y, p) / ref) if ref > 0 else np.nan


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, p))


def murphy(y, p, bins='fixed10'):
    """Murphy (1973) decomposition with the Stephenson et al. (2008) within-bin terms, so that
       BS = REL - RES + UNC + WBV - WBC holds exactly.
    bins: 'fixed10' (10 equal-width bins on [0,1]) or 'quantile10' (deciles of the forecast; ties merged)."""
    y = np.asarray(y, float); p = np.asarray(p, float); n = len(y)
    if bins == 'fixed10':
        edges = np.linspace(0, 1, 11)
        k = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    elif bins == 'quantile10':
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        k = np.clip(np.searchsorted(edges[1:-1], p, side='right'), 0, max(len(edges) - 2, 0))
    else:
        raise ValueError(bins)
    obar = y.mean()
    rel = res = wbv = wbc = 0.0
    for b in np.unique(k):
        m = k == b; nk = m.sum()
        pk = p[m].mean(); ok = y[m].mean()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - obar) ** 2
        wbv += np.sum((p[m] - pk) ** 2)
        wbc += 2 * np.sum((y[m] - ok) * (p[m] - pk))
    rel /= n; res /= n; wbv /= n; wbc /= n
    unc = obar * (1 - obar)
    bs = brier(y, p)
    return dict(BS=bs, REL=rel, RES=res, UNC=unc, WBV=wbv, WBC=wbc,
                identity_residual=bs - (rel - res + unc + wbv - wbc), n=n, events=int(y.sum()),
                nbins=int(len(np.unique(k))))


def reliability_curve(y, p, nbins=10):
    edges = np.linspace(0, 1, nbins + 1)
    k = np.clip(np.digitize(p, edges[1:-1]), 0, nbins - 1)
    rows = []
    for b in range(nbins):
        m = k == b
        if m.sum() == 0:
            continue
        rows.append(dict(bin=b, p_mean=float(np.mean(p[m])), obs_freq=float(np.mean(y[m])), n=int(m.sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- calibration
def sequential_recal(prior_s, prior_y, s_te, clim, kind='logistic', C=1.0, eps=1e-4):
    """Strictly sequential recalibration (identical thresholds to the frozen B_csf_additions.py):
    fall back to the training climatology if <100 prior OOS observations, <5 events or <20 non-events.
    kind: 'logistic' (logit P = a + b S, frozen default), 'logit_logistic' (logit P = a + b logit S),
          'isotonic' (monotone step function, clipped).
    eps: clipping used before the logit transform in 'logit_logistic' (default 1e-4; see T1 eps sweep)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    prior_y = np.asarray(prior_y, int); s = np.asarray(prior_s, float); s_te = np.asarray(s_te, float)
    if len(prior_y) < 100 or np.sum(prior_y == 1) < 5 or np.sum(prior_y == 0) < 20:
        return np.full(len(s_te), clim)
    if np.nanstd(s) < 1e-12:
        return np.full(len(s_te), np.mean(prior_y))
    med = np.nanmedian(s)
    s = np.nan_to_num(s, nan=med); st = np.nan_to_num(s_te, nan=med)
    if kind == 'logistic':
        lr = LogisticRegression(C=C, max_iter=2000).fit(s[:, None], prior_y)
        return lr.predict_proba(st[:, None])[:, 1]
    if kind == 'logit_logistic':
        f = lambda v: np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps)))
        lr = LogisticRegression(C=C, max_iter=2000).fit(f(s)[:, None], prior_y)
        return lr.predict_proba(f(st)[:, None])[:, 1]
    if kind == 'isotonic':
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(s, prior_y)
        return ir.predict(st)
    raise ValueError(kind)


def write_csv(df: pd.DataFrame, name: str, float_format='%.10g'):
    path = TABLES / name
    df.to_csv(path, index=False, float_format=float_format)
    return path
