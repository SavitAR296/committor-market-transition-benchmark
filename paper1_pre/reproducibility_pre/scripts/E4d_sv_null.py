"""E4d -- generative null for the market irreversibility: asymmetric stochastic volatility through the same pipeline.

Model (daily):  r_t = exp(h_t / 2) eps_t,
                h_{t+1} = mu + phi (h_t - mu) + sig_h (rho eps_t + sqrt(1 - rho^2) xi_{t+1}),
                log v_t = const + kappa (h_t - mu)/2 + zeta_t,  zeta AR(1) with coefficient 0.97, sd 0.06.
rho = 0 is time-reversible at second order (no leverage); rho < 0 is the classical asymmetric-SV leverage.
Calibration: phi = 0.98 (as the lambda fit of E4), sig_h = 0.20 (matches sd of log RV20 = 0.50), unconditional
daily variance matched to the data.  Every replicate goes through the frozen common-7 feature code, PCA(2) and the
fitted OU of E4; we record e_p, the window-channel and leverage-channel antisymmetries, and the leverage sums
L+ = sum_{tau=1..20} <r_t r_{t+tau}^2>/<r^2>^2 and L- (reverse direction).
Expected: rho = 0 inside the reversible-surrogate band of E4c with the full window-channel current;
|rho| increasing -> leverage channel and PCA(2) e_p rise towards the observed values.
Output: tables_pre/E4d_sv_null.csv (+ printed summary).  Seed stream [20261017, k, rep].  Runtime ~ 10 s.
Round-2 pilot (ad hoc volume coupling kappa = 0.6); the manuscript uses E4f, which imports the helpers below.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from features_fast import common7, COMMON

RHOS = [0.0, -0.2, -0.4, -0.6, -0.75, -0.85, -0.95]


def standardise(X):
    return (X - X.mean(0)) / X.std(0)


def ou_parts(Z):
    Z0, Z1 = Z[:-1], Z[1:]
    X = np.c_[Z0, np.ones(len(Z0))]
    B, *_ = np.linalg.lstsq(X, Z1 - Z0, rcond=None)
    A = B[:-1].T
    D = np.cov(((Z1 - Z0) - X @ B).T)
    Sig = linalg.solve_continuous_lyapunov(A, -D); Sig = 0.5 * (Sig + Sig.T)
    Om = 0.5 * (A @ Sig - Sig @ A.T)
    zs = (Z - Z.mean(0)) / Z.std(0)
    om_direct = 0.5 * float(np.mean(zs[1:, 0] * zs[:-1, 1]) - np.mean(zs[1:, 1] * zs[:-1, 0]))
    return float(2 * np.trace(np.linalg.solve(Sig, Om.T) @ np.linalg.solve(D, Om))), om_direct


def pca2(F):
    X = standardise(F.to_numpy())
    w, V = np.linalg.eigh(np.cov(X.T)); V = V[:, ::-1][:, :2]
    return standardise(X @ V)


def leverage_sums(r, K=20):
    r = r - r.mean(); r2 = r ** 2; den = np.mean(r2) ** 2
    fwd = sum(np.mean(r[:-k] * r2[k:]) for k in range(1, K + 1)) / den
    bwd = sum(np.mean(r[k:] * r2[:-k]) for k in range(1, K + 1)) / den
    return fwd, bwd


def simulate(rho, T, r_sd, logv_mean, c0, v0, rng, phi=0.98, sig_h=0.20, kappa=0.6):
    eps = rng.standard_normal(T); xi = rng.standard_normal(T); e2 = rng.standard_normal(T)
    mu = 2 * np.log(r_sd) - 0.5 * sig_h ** 2 / (1 - phi ** 2)
    h = np.empty(T); h[0] = mu; zeta = np.zeros(T)
    for t in range(1, T):
        h[t] = mu + phi * (h[t - 1] - mu) + sig_h * (rho * eps[t - 1] + np.sqrt(1 - rho ** 2) * xi[t])
        zeta[t] = 0.97 * zeta[t - 1] + 0.06 * e2[t]
    r = np.exp(h / 2) * eps
    close = c0 * np.exp(np.r_[0.0, np.cumsum(r)])
    vol = np.r_[v0, np.exp(logv_mean + kappa * (h - mu) / 2 + zeta)]
    return close, vol, r


def summarise_data():
    d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026_r1.csv')        # gated file (E1e, E1c)
    c = d.close.to_numpy(float); v = d.volume.to_numpy(float)
    assert (v > 0).all() and (c > 0).all(), 'run E1e_analysis_data.py and E1c first'
    r = np.diff(np.log(c)); F = common7(c, v, 252.0).dropna()
    ep, _ = ou_parts(pca2(F))
    _, om_win = ou_parts(standardise(F[['log_rv5', 'log_rv20']].to_numpy()))
    _, om_lev = ou_parts(standardise(F[['ret20', 'log_rv20']].to_numpy()))
    return c, v, r, dict(ep=ep, om_win=om_win, om_lev=om_lev, sd_lrv20=F.log_rv20.std(),
                         lev_fwd=leverage_sums(r)[0], lev_bwd=leverage_sums(r)[1])


def main(n_rep=30):
    c, v, r, data = summarise_data()
    print('DATA:', {k: round(x, 4) for k, x in data.items()})
    rows = []
    for k, rho in enumerate(RHOS):
        for rep in range(n_rep):
            rng = np.random.default_rng([20261017, k, rep])
            cc, vv, rr = simulate(rho, len(r), r.std(), np.log(v).mean(), c[0], v[0], rng)
            F = common7(cc, vv, 252.0).dropna()
            ep, _ = ou_parts(pca2(F))
            _, om_win = ou_parts(standardise(F[['log_rv5', 'log_rv20']].to_numpy()))
            _, om_lev = ou_parts(standardise(F[['ret20', 'log_rv20']].to_numpy()))
            lf, lb = leverage_sums(rr)
            rows.append(dict(rho=rho, rep=rep, ep=ep, om_win=om_win, om_lev=om_lev,
                             sd_lrv20=F.log_rv20.std(), lev_fwd=lf, lev_bwd=lb))
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / 'tables_pre' / 'E4d_sv_null.csv', index=False, float_format='%.6g')
    S = D.groupby('rho').agg(ep_med=('ep', 'median'), ep_q05=('ep', lambda s: s.quantile(.05)),
                             ep_q95=('ep', lambda s: s.quantile(.95)), om_win=('om_win', 'median'),
                             om_lev=('om_lev', 'median'), lev_fwd=('lev_fwd', 'median'),
                             sd_lrv20=('sd_lrv20', 'median'))
    print(S.sort_index(ascending=False).round(4).to_string())


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:2]))
