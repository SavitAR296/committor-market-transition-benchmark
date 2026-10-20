"""S4e -- information loss in the tilted double well: the zero-parameter two-rate approximation (Eq. 13) against the
near-exact Bayes loss, for tau_c in {inf, 80, 20, 5} (tasks S1 and T3 of the Paper-1 plan).

Model, event and observer exactly as the frozen experiment (B_csf_additions.py): dx = -(x^3 - x - c)dt + 0.72 dW,
dc = -c/tau_c dt + 0.18 dW (clipped), Euler step 0.05, event x >= 0.65 within 2 time units, scored days x < 0.65,
oracle p(x, c) from the frozen 55 x 35 finite-difference grid, observer = trailing means (xbar_w, cbar_w).

Near-exact Bayes leak  L(w) = E[Var(p | xbar, cbar)]: a gradient-boosted regression of p on (xbar, cbar) is fitted on
the first 60% of a 1.2e6-step path and L is the mean squared residual on the last 40% (w = 1 checks the method).

Two-rate theory, conditional on c (c treated as known, accurate for w << tau_c).  In each of twelve quantile bins
of c the scored states split at the barrier x_0(c); by the law of total variance Var p = V_b + V_w, with
V_b = pi_L pi_R (pbar_R - pbar_L)^2 the between-side part.  V_b decorrelates at the Kramers switching rate
kappa(c) = k_LR + k_RL of the tilted well, V_w at the local curvature U''(x_L(c)); each is weighted by the
single-rate posterior variance s^2(rate, w) of a trailing mean (one window, as in Proposition 3):
    ell_two_rate(w) = sum_bins n_c [V_b s^2(kappa w) + V_w s^2(lambda w)] / (n Var p).   No fitted parameter.
'local' is the manuscript's single-rate Gaussian limit s^2(lambda_bar w).
Output: tables_pre/S4e_two_rate_leak.csv.  Seeds [20261020, tau index].  Runtime about 80 s on one core.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
LEG = ROOT.parent / 'reproducibility'
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(LEG / 'src')); sys.path.insert(0, str(LEG / 'experiments'))
from B_csf_additions import true_oracle_grid, interp_grid

DT, SX, SC, TGT, D = 0.05, 0.72, 0.18, 0.65, 0.72 ** 2 / 2
WS = [1, 5, 10, 20, 40, 80]
TAUS = [np.inf, 80.0, 20.0, 5.0]


def simulate(tau, T, rng):
    x = np.empty(T); c = np.zeros(T); x[0] = -1.0
    nx = rng.standard_normal(T) * SX * np.sqrt(DT); nc = rng.standard_normal(T) * SC * np.sqrt(DT)
    for i in range(T - 1):
        if np.isfinite(tau):
            c[i + 1] = min(max(c[i] - c[i] / tau * DT + nc[i], -0.45), 0.45)
        x[i + 1] = min(max(x[i] - (x[i] ** 3 - x[i] - c[i]) * DT + nx[i], -2.2), 2.2)
    return x, c


def stationary_points(c):
    r = np.roots([1.0, 0.0, -1.0, -c]); r = np.sort(r[np.abs(r.imag) < 1e-9].real)
    return r if len(r) == 3 else None


def U(x, c):
    return x ** 4 / 4 - x ** 2 / 2 - c * x


def kramers(c):
    sp = stationary_points(c)
    if sp is None:
        return None
    xl, x0, xr = sp; u2 = lambda z: 3 * z ** 2 - 1
    kl = np.sqrt(u2(xl) * abs(u2(x0))) / (2 * np.pi) * np.exp(-(U(x0, c) - U(xl, c)) / D)
    kr = np.sqrt(u2(xr) * abs(u2(x0))) / (2 * np.pi) * np.exp(-(U(x0, c) - U(xr, c)) / D)
    return dict(x0=x0, kappa=kl + kr, lam=u2(xl))


def window_stats(rate, w):
    a = np.arange(w)
    cbar = np.mean(np.exp(-rate * DT * a))
    V = np.mean(np.exp(-rate * DT * np.abs(a[:, None] - a[None, :])))
    return cbar, V


def theory(w, cbins):
    """Two-rate formula (S4d), conditional on c: the between-side variance decorrelates at the Kramers rate,
    the within-side variance at the local curvature; each enters with its single-rate posterior variance."""
    num = 0.0; n = 0
    for lo, hi, xs, ps in cbins:
        cm = 0.5 * (lo + hi); k = kramers(cm); n_c = len(ps); n += n_c
        if n_c < 100:
            continue
        if k is None:                                    # beyond the fold: one well, curvature channel only
            sp = np.roots([1.0, 0.0, -1.0, -cm]); xm = sp[np.abs(sp.imag) < 1e-9].real[0]
            cb, V = window_stats(3 * xm ** 2 - 1, w); num += n_c * ps.var() * (1 - cb ** 2 / V)
            continue
        left = xs < k['x0']; piL = left.mean()
        Vb = piL * (1 - piL) * (ps[left].mean() - ps[~left].mean()) ** 2 if 0 < piL < 1 else 0.0
        Vw = ps.var() - Vb
        cS, VS = window_stats(k['kappa'], w); cF, VF = window_stats(k['lam'], w)
        num += n_c * (Vb * (1 - cS ** 2 / VS) + Vw * (1 - cF ** 2 / VF))
    return num / n


def rolling(z, w):
    return np.convolve(z, np.ones(w) / w, mode='full')[:len(z)]


def main(T=1_200_000):
    rows = []
    for ti, tau in enumerate(TAUS):
        G, vgrid = true_oracle_grid(tau)
        x, c = simulate(tau, T, np.random.default_rng([20261020, ti]))
        p = interp_grid(G, vgrid, np.c_[x, c])
        sc = x < TGT
        # c-bins for the conditional theory (single bin for tau_c = inf)
        if np.isfinite(tau):
            edges = np.quantile(c[sc], np.linspace(0, 1, 13))
        else:
            edges = np.array([-1e-9, 1e-9])
        cbins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = sc & (c >= lo) & (c <= hi)
            cbins.append((lo, hi, x[m], p[m]))
        lam_bar = float(np.mean([3 * stationary_points(cc)[0] ** 2 - 1 if stationary_points(cc) is not None
                                 else 2.0 for cc in c[sc][::997]]))
        for w in WS:
            xb, cbw = rolling(x, w), rolling(c, w)
            ok = sc.copy(); ok[:max(w, 1)] = False
            idx = np.where(ok)[0]; cut = idx[int(0.6 * len(idx))]
            tr, te = idx[idx < cut], idx[idx >= cut]
            X = np.c_[xb, cbw] if np.isfinite(tau) else xb[:, None]
            reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                                min_samples_leaf=200, random_state=0).fit(X[tr], p[tr])
            L_exact = float(np.mean((p[te] - np.clip(reg.predict(X[te]), 0, 1)) ** 2) / p[te].var())
            th = theory(w, cbins) / float(p[sc].var())
            cb, V = window_stats(lam_bar, w)
            rows.append(dict(tau_c=tau, w=w, lam_bar=lam_bar, lam_w=lam_bar * w * DT, ell_exact=L_exact,
                             ell_two_rate=th, ell_local_gaussian=1 - cb ** 2 / V, var_p=float(p[sc].var())))
            print(f'tau {tau} w {w}: exact {L_exact:.3f} two-rate {th:.3f}', flush=True)
    D_ = pd.DataFrame(rows)
    D_.to_csv(ROOT / 'tables_pre' / 'S4e_two_rate_leak.csv', index=False, float_format='%.6g')
    D_['err_two_rate'] = D_.ell_two_rate - D_.ell_exact
    D_['err_local'] = D_.ell_local_gaussian - D_.ell_exact
    print(D_.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
