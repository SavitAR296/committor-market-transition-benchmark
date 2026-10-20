"""E4c -- time-reversal decomposition of the irreversibility of the trailing-window market state.

Theorem (linear filters, any stationary multichannel input).  For y_i(t) = sum_a k_i(a) u^{p_i}_{t-a},
gamma_pq(tau) = Cov(u^p_{t+tau}, u^q_t) and the kernel cross-correlation K_ij(m) = sum_a k_i(a) k_j(a+m),
    C_ij(1) - C_ji(1) = sum_m K_ij(m) [gamma(m+1) - gamma(m-1)],   gamma = gamma_{p_i p_j}.
Splitting K and gamma into even (+) and odd (-) parts, the lag-one antisymmetry omega_ij = O_ij + P_ij with
    O = (1/2) sum_m K^-(m) [gamma^+(m+1) - gamma^+(m-1)]   observer term: needs kernels with different centres,
                                                          unchanged when the input is time-reversed;
    P = (1/2) sum_m K^+(m) [gamma^-(m+1) - gamma^-(m-1)]   physical term: needs an odd cross-covariance between
                                                          channels (impossible within one channel), flips sign
                                                          under time reversal, survives centred windows.
Model-free estimators: O = (omega_fwd + omega_rev)/2, P = (omega_fwd - omega_rev)/2.

Part A  exact check with linear windows of r and r^2 (theory from empirical cross-covariances vs measured).
Part B  the pipeline's own (nonlinear) features: window channel (log_rv5, log_rv20), leverage channels
        (ret20, log_rv20) and (dd20, log_rv20); observed, time-reversed and centred.
Part C  the paper's PCA(2) state: e_p = 2 omega^2 G factorisation for observed / reversed / centred states,
        200 reversible surrogates (seeds of E4) and 50 random-sign controls (seeds of E4b), plus omega in the
        common basis of the observed loadings (sign-meaningful; fixed orientation: PC1 +log_rv20, PC2 +ews_var).
Data guard: reads the gated analysis file data/spx_2000_2026_r1.csv (E1b splice, E1e repair, E1c gate).  Yahoo's
        ^GSPC record has volume = 0 on 2023-05-24; unrepaired, log(volume) = -inf makes every S&P surrogate share one
        volume path.  The number of repaired days is written to the Part-A output.
Outputs: tables_pre/E4c_linear_decomposition.csv, E4c_feature_pairs.csv, E4c_pca_factorization.csv
Seeds: 20261005 (as E4/E4b).  Runtime about 75 s.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from features_fast import common7, COMMON

SEED = 20261005
OUT = ROOT / 'tables_pre'


# ------------------------------------------------------------------------------------------------ data
def load_spx():
    d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026_r1.csv', parse_dates=['date'])   # gated file (E1e, E1c)
    c = d.close.to_numpy(float)
    v = d.volume.to_numpy(float)
    assert (v > 0).all() and (c > 0).all(), 'run E1e_analysis_data.py and E1c first'
    n_rep = len(pd.read_csv(ROOT / 'tables_pre' / 'E1e_repair_log.csv'))
    return c, v, n_rep


# ------------------------------------------------------------------------------- linear-window theory
def xcov(p, q, L):
    """gamma_pq(tau) = Cov(p_{t+tau}, q_t) for tau = -L..L (biased estimator, as in the sample covariance)."""
    p = p - p.mean(); q = q - q.mean(); n = len(p)
    return {t: (np.dot(p[t:], q[:n - t]) if t >= 0 else np.dot(p[:n + t], q[-t:])) / n for t in range(-L, L + 1)}


def kernel_xcorr(ki, kj):
    return {m: sum(ki[a] * kj[a + m] for a in range(len(ki)) if 0 <= a + m < len(kj))
            for m in range(-len(ki) + 1, len(kj))}


def omega_theory(ki, kj, g_ii, g_jj, g_ij):
    """Returns (omega, O, P) in standardised units from the theorem."""
    K = kernel_xcorr(ki, kj)
    M = max(abs(m) for m in K)
    ms = range(-M, M + 1)                                   # symmetric lag range: needed for the parity split
    Kp = {m: 0.5 * (K.get(m, 0) + K.get(-m, 0)) for m in ms}
    Km = {m: 0.5 * (K.get(m, 0) - K.get(-m, 0)) for m in ms}
    gp = {t: 0.5 * (g_ij[t] + g_ij[-t]) for t in g_ij}
    gm = {t: 0.5 * (g_ij[t] - g_ij[-t]) for t in g_ij}
    O = 0.5 * sum(Km[m] * (gp[m + 1] - gp[m - 1]) for m in ms)
    P = 0.5 * sum(Kp[m] * (gm[m + 1] - gm[m - 1]) for m in ms)
    vi = sum(kernel_xcorr(ki, ki)[m] * g_ii[m] for m in kernel_xcorr(ki, ki))
    vj = sum(kernel_xcorr(kj, kj)[m] * g_jj[m] for m in kernel_xcorr(kj, kj))
    s = np.sqrt(vi * vj)
    return (O + P) / s, O / s, P / s


def trailing_filter(x, k):
    return np.convolve(x, k, mode='full')[:len(x)][len(k) - 1:]


def omega_measured(x, y):
    n = min(len(x), len(y)); x = x[-n:]; y = y[-n:]
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    return 0.5 * (np.mean(x[1:] * y[:-1]) - np.mean(y[1:] * x[:-1]))


def part_a(close):
    rows = []
    pairs = {'window: mean r2 (5d) vs mean r2 (20d)': ('u', np.full(5, 1 / 5), 'u', np.full(20, 1 / 20)),
             'leverage: sum r (20d) vs mean r2 (20d)': ('r', np.ones(20), 'u', np.full(20, 1 / 20)),
             'mixed: sum r (5d) vs mean r2 (20d)': ('r', np.ones(5), 'u', np.full(20, 1 / 20))}
    for direction in ['forward', 'reversed']:
        r = np.diff(np.log(close))
        if direction == 'reversed':
            r = r[::-1].copy()
        r = r - r.mean(); ch = {'r': r, 'u': r ** 2}
        for name, (p, ki, q, kj) in pairs.items():
            g_ii = xcov(ch[p], ch[p], 80); g_jj = xcov(ch[q], ch[q], 80); g_ij = xcov(ch[p], ch[q], 80)
            om, O, P = omega_theory(ki, kj, g_ii, g_jj, g_ij)
            meas = omega_measured(trailing_filter(ch[p], ki), trailing_filter(ch[q], kj))
            rows.append(dict(direction=direction, pair=name, omega_measured=meas, omega_theory=om,
                             observer_term=O, physical_term=P))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- pipeline features (B, C)
def feats(c, v, kind='trailing'):
    return common7(c, v, 252.0, kind).dropna()


def ou_parts(Z):
    Z = np.asarray(Z, float)
    Z0, Z1 = Z[:-1], Z[1:]
    X = np.c_[Z0, np.ones(len(Z0))]
    B, *_ = np.linalg.lstsq(X, Z1 - Z0, rcond=None)
    A = B[:-1].T
    D = np.cov(((Z1 - Z0) - X @ B).T)
    Sig = linalg.solve_continuous_lyapunov(A, -D); Sig = 0.5 * (Sig + Sig.T)
    Om = 0.5 * (A @ Sig - Sig @ A.T)
    ep = float(2 * np.trace(np.linalg.solve(Sig, Om.T) @ np.linalg.solve(D, Om)))
    G = float(np.trace(np.linalg.solve(Sig, D)) / np.linalg.det(D))
    zs = (Z - Z.mean(0)) / Z.std(0)
    om_direct = 0.5 * float(np.mean(zs[1:, 0] * zs[:-1, 1]) - np.mean(zs[1:, 1] * zs[:-1, 0]))
    # omega: fitted-model Omega_12 (enters e_p = 2 omega^2 G); omega_direct: [C12(1)-C21(1)]/2 (Proposition 1)
    return dict(ep=ep, omega=float(Om[0, 1]), omega_direct=om_direct, G=G)


def standardise(X):
    return (X - X.mean(0)) / X.std(0)


def pca_basis(F):
    w, V = np.linalg.eigh(np.cov(standardise(F.to_numpy()).T))
    V = V[:, ::-1][:, :2].copy()
    if V[COMMON.index('log_rv20'), 0] < 0: V[:, 0] *= -1
    if V[COMMON.index('ews_var'), 1] < 0: V[:, 1] *= -1
    return V


def reversed_returns(c, v):
    r = np.diff(np.log(c))[::-1]
    return c[0] * np.exp(np.r_[0.0, np.cumsum(r)]), v[::-1].copy()


def iaaft(x, rng, n_iter=200):
    n = len(x); mu = x.mean(); amp = np.abs(np.fft.rfft(x - mu)); xs = np.sort(x); y = rng.permutation(x)
    for _ in range(n_iter):
        z = np.fft.irfft(amp * np.exp(1j * np.angle(np.fft.rfft(y - y.mean()))), n) + mu
        y_new = xs[np.argsort(np.argsort(z))]
        if np.array_equal(y_new, y):
            break
        y = y_new
    return y


def reversible_surrogate(c, v, rng):
    """E4 construction (unchanged), now fed with a strictly positive volume series."""
    r = np.diff(np.log(c)); a = np.abs(r); lv = np.log(v[1:])
    at = iaaft(a, rng)
    A = np.fft.rfft(a - a.mean()); Vf = np.fft.rfft(lv - lv.mean()); At = np.fft.rfft(at - at.mean())
    s = np.sign(np.real(A * np.conj(Vf))); s[s == 0] = 1.0
    vlin = np.fft.irfft(np.abs(Vf) * s * np.exp(1j * np.angle(At)), len(lv))
    vt = np.sort(lv)[np.argsort(np.argsort(vlin))]
    sign = np.where(rng.random(len(r)) < np.mean(r > 0), 1.0, -1.0)
    return c[0] * np.exp(np.r_[0.0, np.cumsum(sign * at)]), np.r_[v[0], np.exp(vt)]


def random_signs(c, v, sid):
    r = np.diff(np.log(c)); rng = np.random.default_rng([SEED, 9, sid])
    s = np.where(rng.random(len(r)) < np.mean(r > 0), 1.0, -1.0)
    return c[0] * np.exp(np.r_[0.0, np.cumsum(s * np.abs(r))]), v


PAIRS = {'window(log_rv5,log_rv20)': ('log_rv5', 'log_rv20'),
         'leverage(ret20,log_rv20)': ('ret20', 'log_rv20'),
         'leverage(dd20,log_rv20)': ('dd20', 'log_rv20')}


def main(n_sur=200, n_sign=50):
    c, v, n_bad = load_spx()
    A = part_a(c); A['n_volume_repaired'] = n_bad
    A.to_csv(OUT / 'E4c_linear_decomposition.csv', index=False, float_format='%.6g')
    print(A.round(4).to_string(index=False))

    F_obs = feats(c, v); F_rev = feats(*reversed_returns(c, v)); F_cen = feats(c, v, 'centered')
    rows = []
    for name, (a, b) in PAIRS.items():
        om = {k: ou_parts(standardise(F[[a, b]].to_numpy()))['omega_direct']
              for k, F in [('observed', F_obs), ('reversed', F_rev), ('centred', F_cen)]}
        rows.append(dict(pair=name, **{f'omega_{k}': x for k, x in om.items()},
                         observer_est=0.5 * (om['observed'] + om['reversed']),
                         physical_est=0.5 * (om['observed'] - om['reversed'])))
    B = pd.DataFrame(rows); B.to_csv(OUT / 'E4c_feature_pairs.csv', index=False, float_format='%.6g')
    print(B.round(4).to_string(index=False))

    V = pca_basis(F_obs); rows = []

    def add(label, sid, F):
        own = ou_parts(standardise(standardise(F.to_numpy()) @ pca_basis(F)))   # PCs re-standardised, as in E4
        com = ou_parts(standardise(standardise(F.to_numpy()) @ V))
        pr = {}
        for k, ab in PAIRS.items():
            o = ou_parts(standardise(F[list(ab)].to_numpy()))
            pr[f'omega_{k}'] = o['omega_direct']; pr[f'ep_{k}'] = o['ep']
        rows.append(dict(state=label, sid=sid, ep=own['ep'], abs_omega=abs(own['omega']), G=own['G'],
                         omega_common=com['omega_direct'], omega_common_fit=com['omega'], ep_common=com['ep'], **pr))

    for label, F in [('observed', F_obs), ('reversed_returns', F_rev), ('centred', F_cen)]:
        add(label, -1, F)
    for sid in range(n_sur):
        rng = np.random.default_rng([SEED, 0, 0, sid])
        add('reversible_surrogate', sid, feats(*reversible_surrogate(c, v, rng)))
    for sid in range(n_sign):
        add('random_signs', sid, feats(*random_signs(c, v, sid)))
    C = pd.DataFrame(rows); C.to_csv(OUT / 'E4c_pca_factorization.csv', index=False, float_format='%.6g')
    q = lambda s: f'{s.median():.4f} [{s.quantile(.05):.4f}, {s.quantile(.95):.4f}]'
    for st in ['observed', 'reversed_returns', 'centred']:
        o = C[C.state == st].iloc[0]
        print(f'{st:18s} e_p={o.ep:.4f} |omega|={o.abs_omega:.4f} G={o.G:.1f} omega_common={o.omega_common:+.4f}')
    for st in ['reversible_surrogate', 'random_signs']:
        g = C[C.state == st]
        print(f'{st:20s} e_p {q(g.ep)} |omega| {q(g.abs_omega)} G {q(g.G)} omega_common {q(g.omega_common)}')


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
