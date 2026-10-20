"""S4d -- why the local-curvature Gaussian theory overestimates the double-well leak, and a zero-parameter fix.

Symmetric double well (tau_c = inf, c = 0): dx = -(x^3 - x) dt + 0.72 dW, Euler step 0.05, clipped at +-2.2 (exactly
the frozen simulator); event: x >= 0.65 within 2 time units; scored days x < 0.65; observer: trailing mean of w steps.
1. The autocorrelation of x is a two-timescale function; its slow rate equals the Kramers switching rate
   kappa = 2k, k = sqrt(U''(-1)|U''(0)|)/(2 pi) exp(-dU/D), dU = 1/4, D = sigma^2/2.
2. The oracle variance splits into a between-well part (left well vs right-of-barrier, decorrelating at kappa)
   and a within-well part (decorrelating at the local curvature lambda = U''(-1) = 2).
3. Zero-parameter two-channel formula (normalised leak):
       ell_2ch(w) = [V_between s^2(kappa w) + V_within s^2(lambda w)] / Var p_op,
   s^2(rate, w) = posterior variance of a unit OU of that rate given its trailing mean (one window, as in Proposition 3).
4. Exact Bayes leak of the scalar observer: E[Var(p_op | ybar_w)], by 400 quantile bins of ybar on 2e6 steps.
Compared with the local-curvature Gaussian value s^2(2, w) used in the manuscript and with the frozen experiment's
GBT-based leak (S4_summary.csv, tau_c = inf).
Output: tables_pre/S4d_two_channel_leak.csv, S4d_acf.csv.  Seed 20261017.  Runtime ~ 1 min.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
SIG, DT, TGT, H = 0.72, 0.05, 0.65, 2.0
DIFF = SIG ** 2 / 2


def oracle():
    """p_op(x) = P(hit x >= 0.65 within H | x): implicit Euler on the backward equation, reflecting at -2.2."""
    xg = np.linspace(-2.2, TGT, 1200); h = xg[1] - xg[0]; n = len(xg); f = -(xg ** 3 - xg)
    up = np.maximum(f, 0) / h; dn = np.maximum(-f, 0) / h
    L = diags([-(up + dn) - 2 * DIFF / h ** 2, dn[1:] + DIFF / h ** 2, up[:-1] + DIFF / h ** 2], [0, -1, 1]).tolil()
    L[0, 1] = up[0] + dn[0] + 2 * DIFF / h ** 2
    dtp = 0.0025
    M = (identity(n) - dtp * L.tocsc()).tolil(); M[-1, :] = 0; M[-1, -1] = 1
    lu = splu(M.tocsc()); v = np.zeros(n); v[-1] = 1
    for _ in range(int(H / dtp)):
        b = v.copy(); b[-1] = 1; v = lu.solve(b)
    return lambda x: np.interp(x, xg, v)


def path(T, rng):
    x = np.empty(T); x[0] = -1.0; nz = rng.standard_normal(T) * SIG * np.sqrt(DT)
    for i in range(T - 1):
        xn = x[i] - (x[i] ** 3 - x[i]) * DT + nz[i]
        x[i + 1] = min(max(xn, -2.2), 2.2)
    return x


def s2_single(rate, w):
    """Posterior variance of x_t given its w-step trailing mean, unit OU with the given rate (per time unit)."""
    r = np.exp(-rate * DT * np.arange(w))
    V = sum(np.exp(-rate * DT * abs(a - b)) for a in range(w) for b in range(w)) / w ** 2
    return 1 - r.mean() ** 2 / V


def main(T=2_000_000):
    rng = np.random.default_rng(20261017)
    p_op = oracle(); x = path(T, rng); p = p_op(x); scored = x < TGT

    xc = x - x.mean(); L = 1200
    f = np.fft.rfft(xc, 2 * T); acf = np.fft.irfft(f * np.conj(f))[:L + 1] / T / xc.var()
    two = lambda s, a, l1, l2: a * np.exp(-l1 * s) + (1 - a) * np.exp(-l2 * s)
    (a, l1, l2), _ = curve_fit(two, np.arange(L + 1) * DT, acf, p0=(0.2, 2.0, 0.2), bounds=([0, 0.3, 0.01], [1, 10, 1]))
    kappa = 2 * np.sqrt(2.0 * 1.0) / (2 * np.pi) * np.exp(-0.25 / DIFF)
    print(f'ACF fit: slow weight {1 - a:.3f}, slow rate {l2:.4f};  Kramers 2k = {kappa:.4f}')
    pd.DataFrame(dict(lag_time=np.arange(L + 1) * DT, acf=acf, fit=two(np.arange(L + 1) * DT, a, l1, l2),
                      fast_weight=a, fast_rate=l1, slow_rate=l2, kramers=kappa)).iloc[::4].to_csv(
        ROOT / 'tables_pre' / 'S4d_acf.csv', index=False, float_format='%.6g')

    P = p[scored]; Vtot = P.var(); left = x[scored] < 0; piL = left.mean()
    Vb = piL * (1 - piL) * (P[left].mean() - P[~left].mean()) ** 2; Vw = Vtot - Vb
    print(f'Var p_op = {Vtot:.4f}: between-well {Vb / Vtot:.3f}, within-well {Vw / Vtot:.3f}')

    frozen = {}
    f4 = ROOT / 'tables_pre' / 'S4_summary.csv'
    if f4.exists():
        S = pd.read_csv(f4); S = S[S.tau_c.astype(str).isin(['inf', 'inf.0'])]
        frozen = dict(zip(S.w.astype(int), S.ell_meas))
    rows = []
    for w in [5, 10, 20, 40, 80]:
        yb = np.convolve(x, np.ones(w) / w, mode='full')[:T]
        ok = scored.copy(); ok[:w] = False
        q = np.quantile(yb[ok], np.linspace(0, 1, 401))
        k = np.clip(np.searchsorted(q, yb[ok]) - 1, 0, 399)
        pk = p[ok]; m = np.bincount(k, pk) / np.bincount(k)
        exact = np.mean((pk - m[k]) ** 2) / pk.var()
        rows.append(dict(w=w, lam_w=2.0 * w * DT, ell_exact_bayes=exact,
                         ell_two_channel=(Vb * s2_single(kappa, w) + Vw * s2_single(2.0, w)) / Vtot,
                         ell_local_gaussian_s2=s2_single(2.0, w), ell_frozen_gbt=frozen.get(w, np.nan)))
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / 'tables_pre' / 'S4d_two_channel_leak.csv', index=False, float_format='%.6g')
    print(D.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
