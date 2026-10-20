"""Shared utilities for the Paper 2 (IJF) reproducibility package.

Layout: paper2_ijf/reproducibility_p2/ sits next to paper1_pre/ in the same repository.  Frozen Paper-B code and the
gated data of Paper 1 are read from ../../paper1_pre/ and never modified (SHA-256 checked by P2_D0_data_gate.py).

Conventions
  S         raw committor-type score: finite-horizon hitting probability of the learned target set B in the fitted
            two-dimensional OU model of the PCA(2) state (method 'ou_raw').
  seq-cal   strictly sequential logistic recalibration of S on prior out-of-sample (OOS) scores ('ou_seqcal').
  headline  scored days: label observable and day not already in the operational stress set.
  AUC       reported only for scores whose ranking is the object (raw scores, classifiers).  It is NOT reported for
            sequentially recalibrated scores or for climatology: both are year-specific transforms of another
            forecast, and their pooled AUC measures the cross-year calibration levels, not discrimination.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

P2 = Path(__file__).resolve().parents[1]                 # reproducibility_p2/
REPO = P2.parents[1]                                      # contains paper1_pre/ and paper2_ijf/
P1 = REPO / 'paper1_pre'
LEG = P1 / 'reproducibility'                              # frozen CSF-era package (read-only)
PRE = P1 / 'reproducibility_pre'                          # Paper 1 package (read-only)
TAB = P2 / 'tables_p2'
FIG = P2 / 'figures_p2'
CACHE = TAB / '_cache'
for _d in (TAB, FIG, CACHE):
    _d.mkdir(exist_ok=True, parents=True)
for _p in (LEG / 'src', LEG / 'experiments', PRE / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SPX_R1 = PRE / 'data' / 'spx_2000_2026_r1.csv'           # gated S&P 500 file of Paper 1 (2023-05-24 volume patched)
SPX_SPLICE = PRE / 'data' / 'spx_2000_2026.csv'         # frozen snapshot + Yahoo chart-API download (before E1e)
SPX_PATCHES = PRE / 'data' / 'spx_volume_patches.csv'   # documented second-vendor values (Paper 1, E1e)
SPX_YAHOO = PRE / 'data' / 'spx_yahoo_chart_api_2019-01-02_2026-10-16.csv'
DROPDAY_UPLOAD = P2 / 'data' / 'spx_yahoo_2019_2026_without_2023-05-24_upload.csv'   # cross-check only
VIX = PRE / 'data' / 'vix_2000_2026.csv'
SPX_ARCHIVE = PRE / 'data' / 'archive_vijinho_sp500_1950_2018.csv'
BTC = LEG / 'data' / 'btc.csv'
ETH = LEG / 'data' / 'eth.csv'
FROZEN_FEATURES = LEG / 'tables' / 'features_SPX.csv'
OLD_FEATURES = PRE / 'tables_pre' / 'features_SPX_2026.csv'   # built before the volume repair (kept for the diff)
OLD_E3 = PRE / 'tables_pre' / 'E3_forecasts_2026.csv'

FULL = ['log_rv20', 'log_rv5', 'log_act_rel', 'dd20', 'ret20', 'one_minus_n', 'n_sd',
        'ews_var', 'ews_ar1', 'log_vix', 'vrp']
NO_OPTION = [c for c in FULL if c not in ['log_vix', 'vrp']]
COMMON = [c for c in NO_OPTION if c not in ['one_minus_n', 'n_sd']]
HORIZONS = (5, 20)
FROZEN_END = pd.Timestamp('2020-04-17')
DYN = ['ou', 'msm', 'amc']
MAIN_METHODS = ['ou_raw', 'ou_seqcal', 'msm_seqcal', 'amc_seqcal', 'pc2_logistic', 'pc2_gbt', 'raw_logistic', 'raw_gbt']
NO_AUC = ('clim',)                                       # plus every '*_seqcal' method (see module docstring)

LABELS = {'ou_raw': 'OU raw S', 'ou_seqcal': 'OU seq-cal', 'msm_raw': 'MSM raw', 'msm_seqcal': 'MSM seq-cal',
          'amc_raw': 'AMC raw', 'amc_seqcal': 'AMC seq-cal', 'pc2_logistic': 'PC2 logistic', 'pc2_gbt': 'PC2 GBT',
          'raw_logistic': 'raw logistic', 'raw_gbt': 'raw GBT', 'pc2_gbt_cv': 'PC2 GBT (CV)',
          'raw_gbt_cv': 'raw GBT (CV)', 'clim': 'climatology'}


def auc_reported(method: str) -> bool:
    return not (method in NO_AUC or method.endswith('_seqcal'))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, name: str, float_format='%.10g') -> Path:
    path = TAB / name
    df.to_csv(path, index=False, float_format=float_format)
    return path


# ----------------------------------------------------------------------------------------------------- scoring rules
def brier(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def bss(y, p, ref):
    """1 - BS(p)/BS(ref); ref may be row-wise (year-specific training climatology, as in the frozen protocol)."""
    y = np.asarray(y, float)
    b0 = float(np.mean((np.asarray(ref, float) - y) ** 2))
    return float(1 - brier(y, p) / b0) if b0 > 0 else np.nan


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, np.asarray(p, float)))


def pav(y, p):
    """Isotonic (PAV) recalibration of p against y; ties in p receive one value (CORP, Dimitriadis et al. 2021)."""
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds='clip').fit(p, y).predict(p)


def corp(y, p):
    """CORP decomposition of the mean Brier score:  BS = MCB - DSC + UNC  (exact).
    MCB = BS(p) - BS(PAV(p)) (miscalibration), DSC = BS(ybar) - BS(PAV(p)) (discrimination), UNC = ybar(1-ybar)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    r = pav(y, p); bs = brier(y, p); bs_r = brier(y, r); ybar = y.mean(); unc = ybar * (1 - ybar)
    return dict(BS=bs, MCB=bs - bs_r, DSC=unc - bs_r, UNC=unc, identity_residual=bs - ((bs - bs_r) - (unc - bs_r) + unc),
                n=len(y), events=int(y.sum()))


def corp_stratified_dsc(y, p, groups):
    """Discrimination when the recalibration map may differ by group (e.g. by model year): sum_g n_g DSC_g / n,
    each DSC_g computed against the pooled base rate.  Pooled DSC <= stratified DSC; the gap is the resolution that a
    single monotone map cannot recover when the meaning of the score drifts between groups."""
    y = np.asarray(y, float); p = np.asarray(p, float); g = np.asarray(groups)
    ybar = y.mean(); tot = 0.0
    for k in np.unique(g):
        m = g == k
        r = pav(y[m], p[m]); tot += np.sum((ybar - y[m]) ** 2) - np.sum((r - y[m]) ** 2)
    return float(tot / len(y))


# -------------------------------------------------------------------------------------- HAC and fixed-b variances
def nw(Z, q):
    """Newey-West long-run covariance (Bartlett weights, truncation q) of the columns of Z (demeaned inside)."""
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    Zc = Z - Z.mean(0); n = len(Z); S = Zc.T @ Zc / n
    for k in range(1, q + 1):
        G = Zc[k:].T @ Zc[:-k] / n
        S += (1 - k / (q + 1)) * (G + G.T)
    return S


def ewc(Z, nu=None):
    """Equal-weighted cosine (EWC) long-run covariance with nu = floor(0.4 T^(2/3)) cosine terms (Lazarus, Lewis,
    Stock and Watson 2018).  Returns (Omega, nu); fixed-b inference uses t_nu (scalar) or F(k, nu-k+1)."""
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    T = len(Z)
    if nu is None:
        nu = int(np.floor(0.4 * T ** (2 / 3)))
    Zc = Z - Z.mean(0); t = np.arange(1, T + 1)
    Om = np.zeros((Z.shape[1], Z.shape[1]))
    for j in range(1, nu + 1):
        L = np.sqrt(2 / T) * (np.cos(np.pi * j * (t - 0.5) / T) @ Zc)
        Om += np.outer(L, L)
    return Om / nu, nu


def dm_tests(d, h):
    """Tests of E[d]=0 for a loss differential d (positive = comparator better).  Returns a list of dicts."""
    d = np.asarray(d, float); T = len(d); out = []
    for q in sorted({h - 1, 20}):
        se = np.sqrt(nw(d, q)[0, 0] / T); t = d.mean() / se
        out.append(dict(test='unconditional', variance=f'NW lag {q}', stat=float(t), df=np.nan,
                        p_value=float(2 * stats.norm.sf(abs(t)))))
    Om, nu = ewc(d); t = d.mean() / np.sqrt(Om[0, 0] / T)
    out.append(dict(test='unconditional', variance=f'EWC nu={nu} (fixed-b, t_nu)', stat=float(t), df=nu,
                    p_value=float(2 * stats.t.sf(abs(t), nu))))
    return out


def gw_conditional(d, h):
    """Giacomini-White (2006) conditional test with instruments (1, d_{t-h}); chi2(2) with NW lag h-1, and a fixed-b
    EWC variant with the F(2, nu-1) reference distribution."""
    d = np.asarray(d, float)
    Z = np.c_[d[h:], d[h:] * d[:-h]]; n = len(Z); Zb = Z.mean(0)
    out = []
    S = nw(Z, max(h - 1, 0)); stat = float(n * Zb @ np.linalg.solve(S, Zb))
    out.append(dict(test='conditional (1, d_{t-h})', variance=f'NW lag {h - 1}', stat=stat, df=2,
                    p_value=float(stats.chi2.sf(stat, 2))))
    Om, nu = ewc(Z); W = float(n * Zb @ np.linalg.solve(Om, Zb)); k = 2
    F = W * (nu - k + 1) / (nu * k)
    out.append(dict(test='conditional (1, d_{t-h})', variance=f'EWC nu={nu} (fixed-b, F)', stat=F, df=nu,
                    p_value=float(stats.f.sf(F, k, nu - k + 1))))
    return out


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------------------------------------- resampling
def mbb_index(n, L, B, rng):
    """Moving-block bootstrap index matrix (B x n)."""
    k = (n + L - 1) // L
    st = rng.integers(0, max(1, n - L + 1), size=(B, k))
    return np.minimum((st[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n], n - 1)


def boot_block(h):
    """Block length of every moving-block bootstrap in the paper: twice the forecast horizon, so that one block spans
    two overlapping label windows (10 days at h = 5, 40 days at h = 20).  Table A3 reports blocks of 5 to 60 days."""
    return 2 * int(h)


def mcs(L, names, block, B, rng, alpha=0.10):
    """Model confidence set (Hansen, Lunde and Nason 2011), T_max statistic, moving-block bootstrap.
    L: T x M losses.  Returns DataFrame with the MCS p-value of every model and membership at level alpha."""
    L = np.asarray(L, float); T, M = L.shape
    idx = mbb_index(T, block, B, rng)
    alive = list(range(M)); pvals = {}; running = 0.0
    while len(alive) > 1:
        Ls = L[:, alive]; dbar = Ls.mean(0) - Ls.mean()          # d_i. = L_i - mean_j L_j  (time average)
        boot = np.stack([Ls[ix].mean(0) - Ls[ix].mean() for ix in idx])   # B x m
        var = ((boot - dbar) ** 2).mean(0)
        t = dbar / np.sqrt(var)
        Tmax = t.max(); Tb = ((boot - dbar) / np.sqrt(var)).max(1)
        p = float(np.mean(Tb >= Tmax)); running = max(running, p)
        worst = alive[int(np.argmax(t))]; pvals[worst] = running; alive.remove(worst)
    pvals[alive[0]] = 1.0
    return pd.DataFrame(dict(method=[names[i] for i in range(M)], mcs_p=[pvals[i] for i in range(M)],
                             in_mcs=[pvals[i] >= alpha for i in range(M)]))


# --------------------------------------------------------------------------------------------- cluster tools
def clusters_from_positions(pos, gap, groups=None):
    """Cluster ids for sorted integer positions (trading-day index): a new cluster starts when the distance to the
    previous selected day exceeds `gap` or when the group (model year) changes."""
    pos = np.asarray(pos); cid = np.zeros(len(pos), int)
    for i in range(1, len(pos)):
        new = pos[i] - pos[i - 1] > gap or (groups is not None and groups[i] != groups[i - 1])
        cid[i] = cid[i - 1] + int(new)
    return cid


def _pearson_groups(e, n, g, K):
    E = np.bincount(g, e, K); N = np.bincount(g, n, K); pb = E.sum() / N.sum()
    if pb <= 0 or pb >= 1:
        return 0.0
    return float(np.sum((E - N * pb) ** 2 / (N * pb * (1 - pb))))


def cluster_permutation_homogeneity(e, n, g, R, rng):
    """Randomization test of equal event rates across groups with whole clusters as the exchangeable units.
    e, n: events and days per cluster; g: group (0..K-1) per cluster.  The statistic is the Pearson dispersion of
    group rates; group labels are permuted across clusters (number of clusters per group fixed)."""
    e = np.asarray(e, float); n = np.asarray(n, float); g = np.asarray(g, int); K = g.max() + 1
    obs = _pearson_groups(e, n, g, K); cnt = 0
    for _ in range(R):
        cnt += _pearson_groups(e, n, rng.permutation(g), K) >= obs - 1e-12
    return obs, (1 + cnt) / (1 + R)


def quasibinomial_F(e, n, g):
    """Cluster-level quasi-binomial test of a group effect: clusters are binomial observations (e_c of n_c) with
    a common dispersion phi estimated by Pearson X^2/(G-K) from the group-rate model; F = (dDev/(K-1))/phi."""
    e = np.asarray(e, float); n = np.asarray(n, float); g = np.asarray(g, int); K = g.max() + 1; G = len(e)
    if G - K < 1:
        return np.nan, np.nan, np.nan, G - K
    pg = np.bincount(g, e, K) / np.bincount(g, n, K); p0 = e.sum() / n.sum()

    def dev(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        a = np.where(e > 0, e * np.log(np.maximum(e, 1e-300) / (n * p)), 0.0)
        b = np.where(n - e > 0, (n - e) * np.log(np.maximum(n - e, 1e-300) / (n * (1 - p))), 0.0)
        return 2 * np.sum(a + b)
    mu = pg[g]; ok = (mu > 0) & (mu < 1)
    phi = np.sum(((e - n * mu) ** 2 / np.where(ok, n * mu * (1 - mu), 1.0))[ok]) / (G - K)
    dD = dev(np.full(G, p0)) - dev(mu)
    if not phi > 0:
        return np.nan, np.nan, phi, G - K
    F = (dD / (K - 1)) / phi
    return float(F), float(stats.f.sf(F, K - 1, G - K)), float(phi), G - K
