"""S4b -- why the local-curvature Gaussian limit fails in the double well (diagnostic for Proposition 4).
For each tau_c, a long path (1e6 steps, frozen dynamics, seed stream [20261003, 9, k]) gives the *measured* posterior
variance fraction of the Gaussian-copula coordinate g = Phi^{-1}(F(x)) given the rolling observation (linear
regression on the rolling means, scoring population x < 0.65): s2_eff(w) = 1 - R^2.  Inserting s2_eff into the exact
Mehler series of S4 tests the leak formula separately from the curvature map lam_bar -> s2.  lam_eff is the single
AR(1) rate whose single-window s2 best matches s2_eff (least squares over w).  Output: S4b_effective_variance.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import ndtri
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from S4_doublewell_collapse import gaussian_limit, leakG, TAUS, WS, DT, TARGET, SIG_X, SIG_C, tkey
from gaussian_observer import s2_discrete
from pre_common import write_csv


def path(tau, rng, n=1_000_000):
    zx = rng.normal(size=n) * SIG_X * np.sqrt(DT); zc = rng.normal(size=n) * SIG_C * np.sqrt(DT)
    x = np.empty(n); c = np.empty(n); xx, cc = -1.0, 0.0
    for i in range(n):
        x[i] = xx; c[i] = cc
        d = -(xx * xx * xx - xx - cc)
        if tau != np.inf:
            cc = cc - cc / tau * DT + zc[i]; cc = -0.45 if cc < -0.45 else (0.45 if cc > 0.45 else cc)
        xx = xx + d * DT + zx[i]; xx = -2.2 if xx < -2.2 else (2.2 if xx > 2.2 else xx)
    return x[5000:], c[5000:]


def main():
    a, _ = gaussian_limit(); pbar = a[0]; var_inf = np.sum(a[1:] ** 2); bs0 = pbar * (1 - pbar)
    S = pd.read_csv(ROOT / 'tables_pre' / 'S4_summary.csv'); S['tau_c'] = [tkey(float(t)) for t in S.tau_c]
    rows = []
    for k, tau in enumerate(TAUS):
        x, c = path(tau, np.random.default_rng([20261003, 9, k]))
        sel = x < TARGET; xs = np.sort(x[sel]); F = np.searchsorted(xs, x, side='right') / (len(xs) + 1.0)
        g = ndtri(np.clip(F, 1e-6, 1 - 1e-6))
        for w in WS:
            yx = pd.Series(x).rolling(w).mean().to_numpy(); yc = pd.Series(c).rolling(w).mean().to_numpy()
            ok = sel & np.isfinite(yx)
            X = np.c_[np.ones(ok.sum()), yx[ok]] if tau == np.inf else np.c_[np.ones(ok.sum()), yx[ok], yc[ok]]
            beta, *_ = np.linalg.lstsq(X, g[ok], rcond=None); res = g[ok] - X @ beta
            s2e = float(res.var() / g[ok].var())
            m = S[(S.tau_c == tkey(tau)) & (S.w == w)].iloc[0]
            rows.append(dict(tau_c=tkey(tau), w=w, lam_w=m.lam_w, s2_local=m.s2_G, s2_eff=s2e, dBSS_meas=m.dBSS_meas,
                             dBSS_G_local=m.dBSS_G, dBSS_G_eff=leakG(a, s2e) / bs0, ell_meas=m.ell_meas, ell_G_eff=leakG(a, s2e) / var_inf))
    D = pd.DataFrame(rows)
    lam_grid = np.exp(np.linspace(np.log(0.01), np.log(4), 400)); fits = []
    for t, g_ in D.groupby('tau_c'):
        sse = [np.sum([(s2_discrete(np.exp(-l * DT), [int(w)]) - s) ** 2 for w, s in zip(g_.w, g_.s2_eff)]) for l in lam_grid]
        fits.append((t, float(lam_grid[int(np.argmin(sse))])))
    D['lam_eff'] = D.tau_c.map(dict(fits)); D['dev_eff_BSS'] = D.dBSS_meas - D.dBSS_G_eff
    write_csv(D, 'S4b_effective_variance.csv')
    print(D.round(3).to_string(index=False)); print('max |dev| eff:', D.dev_eff_BSS.abs().max().round(3))


if __name__ == '__main__':
    main()
