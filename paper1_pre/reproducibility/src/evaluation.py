import numpy as np
from scipy.stats import norm, rankdata


def auc_delong(y, p):
    """AUC and DeLong (1988) standard error."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    pos = p[y == 1]; neg = p[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan
    # placements
    allp = np.concatenate([pos, neg])
    r = rankdata(allp)
    r_pos = rankdata(pos); r_neg = rankdata(neg)
    V10 = (r[:m] - r_pos) / n
    V01 = 1 - (r[m:] - r_neg) / m
    auc = V10.mean()
    s10 = V10.var(ddof=1) if m > 1 else 0.0
    s01 = V01.var(ddof=1) if n > 1 else 0.0
    se = np.sqrt(s10 / m + s01 / n)
    return auc, se


def brier(y, p):
    return np.mean((np.asarray(p) - np.asarray(y)) ** 2)


def bss(y, p, clim):
    b = brier(y, p); bc = brier(y, np.full(len(y), clim))
    return 1 - b / bc if bc > 0 else np.nan


def reliability(y, p, nbins=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    conf, acc, cnt = [], [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() == 0:
            conf.append(np.nan); acc.append(np.nan); cnt.append(0); continue
        conf.append(p[m].mean()); acc.append(y[m].mean()); cnt.append(m.sum())
    conf, acc, cnt = map(np.array, (conf, acc, cnt))
    ece = np.nansum(np.abs(acc - conf) * cnt) / cnt.sum()
    return conf, acc, cnt, ece


def lead_times(p, entries, fpr=0.10, lookback=60):
    """Lead time (days) between the first alarm and each crisis entry; alarm threshold set so that the
    false-alarm rate on non-event days equals `fpr`.  Returns array (0 = missed)."""
    p = np.asarray(p); entries = np.asarray(entries)
    T = len(p)
    ev_days = np.zeros(T, bool)
    for e in entries:
        ev_days[max(0, e - lookback):e + 1] = True
    thr = np.nanquantile(p[~ev_days], 1 - fpr) if (~ev_days).sum() > 10 else np.nanquantile(p, 1 - fpr)
    leads = []
    for e in entries:
        w = p[max(0, e - lookback):e]
        idx = np.where(w > thr)[0]
        leads.append(len(w) - idx[0] if len(idx) else 0)
    return np.array(leads), thr


def block_bootstrap_ci(y, p_a, p_b, clim, block=20, B=500, seed=0):
    """Moving-block bootstrap of BSS(a)-BSS(b) and AUC(a)-AUC(b)."""
    rng = np.random.default_rng(seed)
    T = len(y); nb = int(np.ceil(T / block))
    d_bss, d_auc = [], []
    for _ in range(B):
        starts = rng.integers(0, T - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        d_bss.append(bss(yy, p_a[idx], clim) - bss(yy, p_b[idx], clim))
        d_auc.append(auc_delong(yy, p_a[idx])[0] - auc_delong(yy, p_b[idx])[0])
    d_bss, d_auc = np.array(d_bss), np.array(d_auc)
    return np.quantile(d_bss, [0.05, 0.95]), np.quantile(d_auc, [0.05, 0.95])


def diebold_mariano(y, p_a, p_b, lag=20):
    """DM test on Brier loss differentials with Newey-West HAC variance. Negative stat = a better."""
    d = (np.asarray(p_a) - y) ** 2 - (np.asarray(p_b) - y) ** 2
    T = len(d); dbar = d.mean(); dc = d - dbar
    s = np.sum(dc * dc) / T
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * np.sum(dc[k:] * dc[:-k]) / T
    se = np.sqrt(max(s, 1e-16) / T)
    stat = dbar / se
    return stat, 2 * (1 - norm.cdf(abs(stat)))
