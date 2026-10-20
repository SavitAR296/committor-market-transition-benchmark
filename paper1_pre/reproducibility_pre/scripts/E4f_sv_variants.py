"""E4f -- stochastic-volatility null, robustness variants (tasks E2 and E3 of the Paper-1 plan).

r_t = exp(h_t/2) eps_t, h_t = mu + sum_j h^(j)_t, h^(j)_t = phi_j h^(j)_{t-1} + s_j eta^(j)_t,
eta^(j)_t = rho eps_{t-1} + sqrt(1 - rho^2) xi^(j)_t   (leverage: return shock -> next volatility shocks).
log v_t = const + kappa (h_t - mu)/2 + zeta_t, zeta AR(1).
E3  kappa, phi_zeta, sd_zeta from the data: detrended log volume (minus its 250-day trailing mean) regressed on
    detrended log RV20; AR(1) fitted to the residual.
E2  variants: G1 Gaussian eps, one factor (phi = 0.98); T1 Student-t(5) eps (unit variance), one factor;
    G2/T2 two factors (phi_fast = 0.90, phi_slow and the variance share from a grid fit to the ACF of log RV20 at
    lags 20, 60, 120, 250).  Total log-volatility variance is scaled so that sd(log RV20) matches the data.
Every replicate goes through the frozen common-7 features, PCA(2) and the OU fit; recorded: PCA(2) e_p, window-
and leverage-channel antisymmetries, leverage sums L+ and L-.
Outputs: tables_pre/E4f_calibration.csv, E4f_sv_variants.csv.  Seeds [20261019, variant, k, rep].
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from features_fast import common7
from E4d_sv_null import standardise, ou_parts, pca2, leverage_sums

RHOS = [0.0, -0.4, -0.6, -0.75, -0.85, -0.95]
LAGS = [20, 60, 120, 250]


def acf(x, lags):
    x = x - x.mean(); d = np.dot(x, x)
    return np.array([np.dot(x[k:], x[:-k]) / d for k in lags])


def data_moments():
    d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026_r1.csv')
    c = d.close.to_numpy(float); v = d.volume.to_numpy(float)
    assert (v > 0).all()
    F = common7(c, v, 252.0)
    lrv = F.log_rv20; lv = pd.Series(np.log(v))
    y = (lv - lv.rolling(250).mean()); x = (lrv - lrv.rolling(250).mean())
    ok = y.notna() & x.notna()
    kappa = float(np.polyfit(x[ok], y[ok], 1)[0])
    e = (y[ok] - kappa * x[ok]).to_numpy(); e = e - e.mean()
    phz = float(np.dot(e[1:], e[:-1]) / np.dot(e[:-1], e[:-1])); sdz = float(np.std(e[1:] - phz * e[:-1]))
    r = np.diff(np.log(c))
    return c, v, r, dict(kappa=kappa, phi_zeta=phz, sd_zeta=sdz, sd_lrv20=float(lrv.std()),
                         **{f'acf{k}': a for k, a in zip(LAGS, acf(lrv.dropna().to_numpy(), LAGS))})


def simulate(T, rho, factors, dist, cal, r_sd, logv_mean, c0, v0, rng):
    """factors: list of (phi, s); returns close, volume, returns."""
    eps = rng.standard_normal(T) if dist == 'gauss' else rng.standard_t(5, T) / np.sqrt(5 / 3)
    h = np.zeros(T)
    for phi, s in factors:
        eta = rho * np.r_[0.0, eps[:-1]] + np.sqrt(1 - rho ** 2) * rng.standard_normal(T)
        h += lfilter([s], [1, -phi], eta)
    var_h = sum(s ** 2 / (1 - phi ** 2) for phi, s in factors)
    mu = 2 * np.log(r_sd) - 0.5 * var_h
    r = np.exp((mu + h) / 2) * eps
    zeta = lfilter([cal['sd_zeta']], [1, -cal['phi_zeta']], rng.standard_normal(T))
    vol = np.exp(logv_mean + cal['kappa'] * h / 2 + zeta)
    return c0 * np.exp(np.r_[0.0, np.cumsum(r)]), np.r_[v0, vol], r


def factor_sets(cal, r, c, v):
    """One-factor baseline (phi 0.98) and a two-factor fit to the ACF of log RV20; both scaled to sd(log RV20)."""
    T = len(r)

    def scaled(factors, seed):
        rng = np.random.default_rng([20261019, 99, seed])
        cc, vv, _ = simulate(3 * T, 0.0, factors, 'gauss', cal, r.std(), np.log(v).mean(), c[0], v[0], rng)
        lrv = common7(cc, vv, 252.0).log_rv20.dropna().to_numpy()
        return lrv.std(), acf(lrv, LAGS)

    one = [(0.98, 0.20)]
    sd1, _ = scaled(one, 0); one = [(0.98, 0.20 * cal['sd_lrv20'] / sd1)]
    best = None
    target = np.array([cal[f'acf{k}'] for k in LAGS])
    for ps in [0.99, 0.995, 0.998]:
        for share in [0.2, 0.35, 0.5, 0.65]:
            vf, vs = share, 1 - share                       # variance shares of the stationary log-vol
            fs = [(0.90, np.sqrt(vf * (1 - 0.90 ** 2))), (ps, np.sqrt(vs * (1 - ps ** 2)))]
            sd, a = scaled(fs, 1)
            err = float(np.sum((a - target) ** 2))
            if best is None or err < best[0]:
                best = (err, ps, share, fs, sd)
    err, ps, share, fs, sd = best
    two = [(phi, s * cal['sd_lrv20'] / sd) for phi, s in fs]
    return one, two, dict(phi_slow=ps, fast_share=share, acf_sse=err)


def main(n_rep=30):
    c, v, r, cal = data_moments()
    one, two, fit = factor_sets(cal, r, c, v)
    pd.DataFrame([dict(**cal, **fit, one_factor=str(one), two_factor=str(two))]).to_csv(
        ROOT / 'tables_pre' / 'E4f_calibration.csv', index=False)
    print('calibration:', {k: round(x, 4) if isinstance(x, float) else x for k, x in {**cal, **fit}.items()})
    rows = []
    for vi, (name, dist, factors) in enumerate([('G1', 'gauss', one), ('T1', 't5', one),
                                                 ('G2', 'gauss', two), ('T2', 't5', two)]):
        for k, rho in enumerate(RHOS):
            for rep in range(n_rep):
                rng = np.random.default_rng([20261019, vi, k, rep])
                cc, vv, rr = simulate(len(r), rho, factors, dist, cal, r.std(), np.log(v).mean(), c[0], v[0], rng)
                F = common7(cc, vv, 252.0).dropna()
                ep, _ = ou_parts(pca2(F))
                _, om_w = ou_parts(standardise(F[['log_rv5', 'log_rv20']].to_numpy()))
                _, om_l = ou_parts(standardise(F[['ret20', 'log_rv20']].to_numpy()))
                lf, lb = leverage_sums(rr)
                rows.append(dict(variant=name, rho=rho, rep=rep, ep=ep, om_win=om_w, om_lev=om_l,
                                 lev_fwd=lf, lev_bwd=lb, sd_lrv20=F.log_rv20.std()))
    D = pd.DataFrame(rows); D.to_csv(ROOT / 'tables_pre' / 'E4f_sv_variants.csv', index=False, float_format='%.6g')
    S = D.groupby(['variant', 'rho']).agg(ep=('ep', 'median'), ep05=('ep', lambda s: s.quantile(.05)),
                                          ep95=('ep', lambda s: s.quantile(.95)), om_win=('om_win', 'median'),
                                          om_lev=('om_lev', 'median'), lev_fwd=('lev_fwd', 'median'))
    print(S.round(4).to_string())


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:2]))
