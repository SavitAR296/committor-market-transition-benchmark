"""Irreversibility estimators used identically on Gaussian-model, double-well and market states.

Definitions (written into Appendix D of the manuscript):
  * Fitted Markov model: least squares of the one-step increment on [z_t, 1] (the frozen OUModel), giving drift A
    (per step) and residual covariance D; interpreted, as in the forecasting pipeline, as the OU process
    dZ = A Z dt + G dW, GG^T = D, with stationary covariance Sigma solving A Sigma + Sigma A^T + D = 0.
  * Stationary current J(z) = f(z) rho(z) - (1/2) div(D rho) = Omega Sigma^{-1} z rho(z), Omega = (A Sigma - Sigma A^T)/2.
  * flux/drift ratio R = median_{z ~ rho} |J(z)| / (|f(z)| rho(z)), evaluated in standardized coordinates
    (each component scaled to unit sample variance) because, unlike e_p and the KLD, R is metric dependent.
  * e_p = 2 tr(Sigma^{-1} Omega^T D^{-1} Omega): entropy-production rate of the fitted OU (per step, k_B = 1).
  * d1: per-step KLD between forward and backward path measures of the Gaussian Markov projection
    (closed form in C(0), C(1)); invariant to invertible linear maps of the state.
  * Plug-in KLD (Roldan & Parrondo 2010, 2012): each component is symbolised into q equiprobable bins; D_k is the
    KLD between the empirical k-block distribution and that of the reversed blocks (blocks with an unobserved
    reverse are dropped, which biases D_k downward); the per-step rate estimate is D_2 (pairs) and D_3 - D_2.
"""
from __future__ import annotations

import numpy as np

from gaussian_observer import markov_projection, flux_drift_ratio


def standardize(Y):
    Y = np.asarray(Y, float)
    Y = Y[np.all(np.isfinite(Y), 1)]
    return (Y - Y.mean(0)) / Y.std(0)


def projection_stats(Y, segments=None):
    """Empirical Markov-projection statistics of a (T, m) state series.
    segments: optional array of segment ids; pairs (t, t+1) crossing a segment boundary are dropped."""
    Y = np.asarray(Y, float)
    ok = np.all(np.isfinite(Y), 1)
    mu = Y[ok].mean(0); sd = Y[ok].std(0)
    Z = (Y - mu) / sd
    pair = ok[:-1] & ok[1:]
    if segments is not None:
        segments = np.asarray(segments)
        pair &= segments[:-1] == segments[1:]
    Z0 = Z[:-1][pair]; Z1 = Z[1:][pair]
    m0 = Z0.mean(0); m1 = Z1.mean(0)
    # LS of increment on [z,1] (frozen OUModel) -> A, D; C0, C1 for the KLD use the same pairs
    X = np.c_[Z0, np.ones(len(Z0))]
    B, *_ = np.linalg.lstsq(X, Z1 - Z0, rcond=None)
    A = B[:-1].T
    resid = (Z1 - Z0) - X @ B
    D = np.cov(resid.T)
    C0 = 0.5 * (np.cov(Z0.T) + np.cov(Z1.T))
    C1 = ((Z1 - m1).T @ (Z0 - m0)) / (len(Z0) - 1)
    mp = markov_projection(C0, C1)
    from scipy import linalg
    try:
        Sig = linalg.solve_continuous_lyapunov(A, -D); Sig = 0.5 * (Sig + Sig.T)
        stable = bool(np.all(np.linalg.eigvals(A).real < 0) and np.all(np.linalg.eigvalsh(Sig) > 0))
    except Exception:
        Sig, stable = None, False
    if stable:
        Om = 0.5 * (A @ Sig - Sig @ A.T)
        ratio = flux_drift_ratio(A, D, Sig)
        ep = float(2 * np.trace(np.linalg.solve(Sig, Om.T) @ np.linalg.solve(D, Om)))
        omega = float(Om[0, 1]) if Om.shape[0] == 2 else np.nan
    else:
        ratio = ep = omega = np.nan
    return dict(ratio=ratio, ep=ep, d1=mp['d1'], omega_C1=float(mp['omega_C1'][0, 1]) if C0.shape[0] == 2 else np.nan,
                omega_fit=omega, stable=stable, n_pairs=int(pair.sum()))


def symbolize(Y, q):
    """Equiprobable symbolisation of each component; returns one integer symbol per row (NaN rows -> -1)."""
    Y = np.asarray(Y, float)
    ok = np.all(np.isfinite(Y), 1)
    sym = np.full(len(Y), -1, dtype=np.int64)
    code = np.zeros(ok.sum(), dtype=np.int64)
    for j in range(Y.shape[1]):
        v = Y[ok, j]
        edges = np.quantile(v, np.linspace(0, 1, q + 1)[1:-1])
        code = code * q + np.searchsorted(edges, v, side='right')
    sym[ok] = code
    return sym


def plugin_kld(sym, k, segments=None):
    """Plug-in D_k (nats) between k-block distribution and reversed k-blocks; rows with sym<0 break blocks."""
    sym = np.asarray(sym)
    T = len(sym)
    valid = sym >= 0
    if segments is not None:
        segments = np.asarray(segments)
    idx = np.arange(T - k + 1)
    ok = np.ones(len(idx), bool)
    for j in range(k):
        ok &= valid[idx + j]
        if segments is not None and j > 0:
            ok &= segments[idx + j] == segments[idx]
    idx = idx[ok]
    if len(idx) == 0:
        return np.nan
    base = int(sym.max()) + 1
    fwd = np.zeros(len(idx), dtype=np.int64); rev = np.zeros(len(idx), dtype=np.int64)
    for j in range(k):
        fwd = fwd * base + sym[idx + j]
        rev = rev * base + sym[idx + k - 1 - j]
    uf, cf = np.unique(fwd, return_counts=True)
    count = dict(zip(uf.tolist(), cf.tolist()))
    N = len(idx); D = 0.0
    for b, n in count.items():
        # reverse code of block b
        digits = []
        x = b
        for _ in range(k):
            digits.append(x % base); x //= base
        rb = 0
        for d in digits:           # digits are in reversed order already
            rb = rb * base + d
        nr = count.get(rb, 0)
        if nr > 0:
            D += (n / N) * np.log(n / nr)
    return float(D)


def plugin_rates(Y, qs=(3, 4, 5), segments=None):
    out = {}
    for q in qs:
        s = symbolize(Y, q)
        D2 = plugin_kld(s, 2, segments); D3 = plugin_kld(s, 3, segments)
        out[f'kld_q{q}_D2'] = D2; out[f'kld_q{q}_D3mD2'] = D3 - D2
    return out
