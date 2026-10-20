"""S4 -- tilted double well: direct compression leak and the collapse test of Proposition 4.

The frozen phase diagram (B_synthetic_phase.csv) is re-generated with the frozen code and the frozen seed sequence
(20260914; the order is identified automatically) so that per-observation predictions are available; the reproduced
AUC/BSS are checked against the frozen file (S4_frozen_reproduction.csv).  New simulation (seed 20261003): long paths
of c_t for the stationary average lam_bar = E_c[3 x_a(c)^2 - 1] over c < c* (left well present).
Gaussian limit (tau_c -> inf): exact Mehler series L_G(s^2) = sum_n a_n^2 [1-(1-s^2)^n], a_n = Hermite coefficients
of the double-well oracle p_op(x) (c = 0, x < 0.65) in the Gaussian-copula coordinate, s^2 = posterior variance of a
single trailing window of w steps for an AR(1) with phi = exp(-lam_bar dt).  Collapse criterion (outline): max
|Delta BSS_measured - Delta BSS_Gaussian| < 0.03 over all (tau_c, w); otherwise two-parameter forms are fitted.
Outputs: S4_doublewell_collapse.csv, S4_summary.csv, S4_lambda_bar.csv, S4_fit.csv, S4_frozen_reproduction.csv
"""
import sys, os, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import ndtri, ndtr
ROOT = Path(__file__).resolve().parents[1]; LEG = ROOT.parent / 'reproducibility'
for p in [ROOT / 'src', LEG / 'src', LEG / 'experiments']:
    sys.path.insert(0, str(p))
from B_csf_additions import (simulate_doublewell, future_hit_labels, rolling_state, true_oracle_grid, interp_grid,
                             safe_gbt, fit_ou_scores)
from gaussian_observer import s2_discrete
from pre_common import write_csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

TAUS = [5.0, 20.0, 80.0, np.inf]; WS = [1, 5, 10, 20, 40, 80]
DT, H_STEPS, TARGET, SIG_X, SIG_C, NREP = 0.05, 40, 0.65, 0.72, 0.18, 6
FROZEN_SEED, NEW_SEED = 20260914, 20261003
C_STAR = 2 / (3 * np.sqrt(3))
CACHE = ROOT / 'tables_pre' / '_cache'; CACHE.mkdir(exist_ok=True)
T0 = time.time(); BUDGET = float(os.environ.get('PRE_BUDGET', '200'))
_OR = {}


def tkey(t):
    return 'inf' if np.isinf(t) else f'{t:g}'


def oracle(tau):
    if tau not in _OR:
        _OR[tau] = true_oracle_grid(tau, h=H_STEPS * DT, target=TARGET)
    return _OR[tau]


def frozen_seeds(reps_in_sequence):
    rng = np.random.default_rng(FROZEN_SEED); out = {}
    for tau in TAUS:
        for rep in range(reps_in_sequence):
            s = int(rng.integers(1, 2 ** 31 - 1))
            if rep < NREP:
                out[(tau, rep)] = s
    return out


def choose_seed_order(frozen):
    f = frozen[(frozen.tau_c == 20.0) & (frozen.rep == 0) & (frozen.w == 1) & (frozen.method == 'oracle_pde')].AUC.iloc[0]
    for reps in [6, 8]:
        seed = frozen_seeds(reps)[(20.0, 0)]
        G, v = oracle(20.0); x, c = simulate_doublewell(tau_c=20.0, seed=seed); y = future_hit_labels(x, H_STEPS, target=TARGET)
        te = np.isfinite(y) & (np.arange(len(x)) >= int(len(x) * 0.6)) & (x < TARGET)
        if abs(roc_auc_score(y[te], interp_grid(G, v, np.c_[x, c][te])) - f) < 1e-12:
            return reps
    raise RuntimeError('frozen seed order not reproduced')


def unit(tau, rep, seed):
    G, v = oracle(tau)
    x, c = simulate_doublewell(tau_c=tau, seed=seed); y = future_hit_labels(x, H_STEPS, target=TARGET)
    split = int(len(x) * 0.60); base = np.arange(len(x)); latent = np.c_[x, c]; rows = []
    for w in WS:
        Z = rolling_state(x, c, w); valid = np.isfinite(Z).all(1) & np.isfinite(y)
        tr = valid & (base < split); te = valid & (base >= split) & (x < TARGET)
        if y[te].sum() < 3:
            continue
        p_or = interp_grid(G, v, latent[te])
        p_lat = safe_gbt(latent[tr], y[tr].astype(int), latent[te]); p_rol = safe_gbt(Z[tr], y[tr].astype(int), Z[te])
        sc = StandardScaler().fit(Z[tr])
        try:
            ous, _, _ = fit_ou_scores(sc.transform(Z[tr]), sc.transform(Z[te]), Z[tr][:, 0], horizons=(H_STEPS,), grid_n=32)
            p_ou = ous[H_STEPS][1]
        except Exception:
            p_ou = np.full(int(te.sum()), np.nan)
        yy = y[te]; clim = float(np.mean(y[tr])); bs0 = float(np.mean((yy - clim) ** 2))
        row = dict(tau_c=tkey(tau), rep=rep, seed=seed, w=w, n=int(te.sum()), events=int(yy.sum()), bs0=bs0,
                   mean_oracle=float(p_or.mean()), var_oracle=float(p_or.var()),
                   L_lat=float(np.mean((p_lat - p_or) ** 2)), L_roll=float(np.mean((p_rol - p_or) ** 2)),
                   L_ou=float(np.nanmean((p_ou - p_or) ** 2)))
        for m, p in [('oracle_pde', p_or), ('latent_flex', p_lat), ('rolling_flex', p_rol), ('rolling_ou', p_ou)]:
            ok = np.isfinite(p)
            row[f'AUC_{m}'] = roc_auc_score(yy[ok], p[ok]) if ok.sum() >= 50 else np.nan
            row[f'BSS_{m}'] = 1 - brier_score_loss(yy[ok], p[ok]) / np.mean((yy[ok] - clim) ** 2) if ok.sum() >= 50 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def lambda_bar():
    cg = np.linspace(-0.45, 0.45, 1801)
    def lam_of_c(c):
        if c >= C_STAR:
            return np.nan                                   # left well absent beyond the fold
        r = np.roots([1, 0, -1, -c]); r = np.sort(r[np.abs(r.imag) < 1e-9].real)
        return 3 * r[0] ** 2 - 1
    lam_g = np.array([lam_of_c(c) for c in cg])
    rng = np.random.default_rng(NEW_SEED); rows = []
    for tau in TAUS:
        if np.isinf(tau):
            rows.append(dict(tau_c='inf', lam_bar=2.0, frac_c_beyond_fold=0.0, sd_c=0.0)); continue
        n = 1_000_000; xi = rng.normal(size=n) * SIG_C * np.sqrt(DT); c = np.empty(n); c[0] = 0.0; a = 1 - DT / tau
        cc = 0.0
        for i in range(n - 1):
            cc = a * cc + xi[i]; cc = -0.45 if cc < -0.45 else (0.45 if cc > 0.45 else cc); c[i + 1] = cc
        lam = lam_g[np.clip(np.rint((c + 0.45) / 0.0005).astype(int), 0, len(cg) - 1)]; ok = np.isfinite(lam)
        rows.append(dict(tau_c=tkey(tau), lam_bar=float(lam[ok].mean()), frac_c_beyond_fold=float(1 - ok.mean()), sd_c=float(c.std())))
    return pd.DataFrame(rows)


def gaussian_limit(nmax=150):
    G, v = oracle(np.inf)
    xs = np.linspace(-2.2, TARGET, 40001)[:-1]
    p = interp_grid(G, v, np.c_[xs, np.zeros_like(xs)])
    rho = np.exp(-2 * (xs ** 4 / 4 - xs ** 2 / 2) / SIG_X ** 2); rho /= np.trapezoid(rho, xs)
    F = np.r_[0.0, np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(xs))]; F /= F[-1]
    g = np.linspace(-10, 10, 40001); wq = np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi)
    pg = np.interp(np.interp(ndtr(g), F, xs), xs, p)         # p_op as a function of the Gaussian copula variable
    hp, h = np.ones_like(g), g.copy(); a = [np.trapezoid(pg * wq, g), np.trapezoid(pg * h * wq, g)]
    for n in range(1, nmax):
        hp, h = h, (g * h - np.sqrt(n) * hp) / np.sqrt(n + 1); a.append(np.trapezoid(pg * h * wq, g))
    a = np.array(a); var_direct = np.trapezoid(p ** 2 * rho, xs) - np.trapezoid(p * rho, xs) ** 2
    return a, var_direct


def leakG(a, s2):
    n = np.arange(len(a)); return float(np.sum(a[1:] ** 2 * (1 - (1 - s2) ** n[1:])))


def main():
    frozen = pd.read_csv(LEG / 'tables_csf' / 'B_synthetic_phase.csv')
    fo = CACHE / 'S4_seed_order.pkl'
    if not fo.exists():
        pd.to_pickle(choose_seed_order(frozen), fo)
    seeds = frozen_seeds(pd.read_pickle(fo)); parts = []
    for tau in TAUS:
        for rep in range(NREP):
            f = CACHE / f'S4_{tkey(tau)}_{rep}.pkl'
            if not f.exists():
                if time.time() - T0 > BUDGET:
                    print('budget reached; rerun to continue'); return
                t = time.time(); pd.to_pickle(unit(tau, rep, seeds[(tau, rep)]), f)
                print('S4 unit', tkey(tau), rep, f'{time.time() - t:.0f}s', flush=True)
            parts.append(pd.read_pickle(f))
    D = pd.concat(parts, ignore_index=True)
    rr = []
    for m in ['oracle_pde', 'latent_flex', 'rolling_flex', 'rolling_ou']:
        fz = frozen[frozen.method == m].copy(); fz['tau_c'] = fz.tau_c.map(lambda t: 'inf' if np.isinf(t) else f'{t:g}')
        mg = fz.merge(D[['tau_c', 'rep', 'w', f'AUC_{m}', f'BSS_{m}']], on=['tau_c', 'rep', 'w'])
        rr.append(dict(method=m, n_matched=len(mg), max_abs_dAUC=float((mg.AUC - mg[f'AUC_{m}']).abs().max()),
                       max_abs_dBSS=float((mg.BSS - mg[f'BSS_{m}']).abs().max())))
    write_csv(pd.DataFrame(rr), 'S4_frozen_reproduction.csv')
    LB = lambda_bar(); write_csv(LB, 'S4_lambda_bar.csv')
    a, var_direct = gaussian_limit(); p_bar = a[0]; var_inf = float(np.sum(a[1:] ** 2)); bs0_inf = p_bar * (1 - p_bar)
    lam = dict(zip(LB.tau_c, LB.lam_bar))
    D['comp_leak'] = D.L_roll - D.L_lat; D['dBSS_meas'] = D.comp_leak / D.bs0; D['ell_meas'] = D.comp_leak / D.var_oracle
    D['lam_bar'] = D.tau_c.map(lam); D['lam_w'] = D.lam_bar * D.w * DT
    D['s2_G'] = [s2_discrete(np.exp(-l * DT), [int(w)]) for l, w in zip(D.lam_bar, D.w)]
    D['leak_G'] = [leakG(a, s) for s in D.s2_G]; D['ell_G'] = D.leak_G / var_inf; D['dBSS_G'] = D.leak_G / bs0_inf
    write_csv(D, 'S4_doublewell_collapse.csv')
    S = D.groupby(['tau_c', 'w']).agg(lam_bar=('lam_bar', 'first'), lam_w=('lam_w', 'first'), s2_G=('s2_G', 'first'),
                                      dBSS_meas=('dBSS_meas', 'mean'), dBSS_meas_se=('dBSS_meas', lambda v: v.std() / np.sqrt(len(v))),
                                      ell_meas=('ell_meas', 'mean'), ell_G=('ell_G', 'first'), dBSS_G=('dBSS_G', 'first'),
                                      BSS_oracle=('BSS_oracle_pde', 'mean'), BSS_latent=('BSS_latent_flex', 'mean'),
                                      BSS_rolling=('BSS_rolling_flex', 'mean'), BSS_ou=('BSS_rolling_ou', 'mean'),
                                      L_lat=('L_lat', 'mean'), L_roll=('L_roll', 'mean'), L_ou=('L_ou', 'mean')).reset_index()
    S['dev_BSS'] = S.dBSS_meas - S.dBSS_G; write_csv(S, 'S4_summary.csv')
    fin = S[S.tau_c != 'inf'].reset_index(drop=True); taus = fin.tau_c.astype(float).values; y = fin.ell_meas.values
    sst = np.sum((y - y.mean()) ** 2)
    def stretch(b):
        return np.array([leakG(a, s2_discrete(np.exp(-l * (1 + b / (l * t)) * DT), [int(w)])) / var_inf
                         for l, t, w in zip(fin.lam_bar, taus, fin.w)])
    def amplitude(g):
        return fin.ell_G.values * (1 + g / (fin.lam_bar.values * taus))
    rows = [dict(form='Gaussian limit, no free parameter', param=np.nan, rmse_ell=float(np.sqrt(np.mean((y - fin.ell_G) ** 2))),
                 r2_ell=float(1 - np.sum((y - fin.ell_G) ** 2) / sst))]
    for name, fn, grid in [('stretch: lam_eff = lam_bar (1 + beta/(lam_bar tau_c))', stretch, np.linspace(0, 40, 161)),
                           ('amplitude: ell = ell_G (1 + gamma/(lam_bar tau_c))', amplitude, np.linspace(-5, 40, 451))]:
        sse = np.array([np.sum((fn(b) - y) ** 2) for b in grid]); b = grid[int(np.argmin(sse))]; res = y - fn(b)
        rows.append(dict(form=name, param=float(b), rmse_ell=float(np.sqrt(np.mean(res ** 2))), r2_ell=float(1 - np.sum(res ** 2) / sst)))
    Fit = pd.DataFrame(rows); Fit['max_abs_dev_BSS_gaussian'] = float(S.dev_BSS.abs().max())
    Fit['collapse_within_0p03'] = bool(S.dev_BSS.abs().max() < 0.03); Fit['var_pop_inf'] = var_inf
    Fit['var_pop_inf_direct'] = var_direct; Fit['p_bar_inf'] = p_bar
    write_csv(Fit, 'S4_fit.csv')
    print(pd.DataFrame(rr).to_string(index=False)); print(LB.to_string(index=False)); print(Fit.round(4).to_string(index=False))
    print('S4 complete'); (CACHE / 'DONE_S4_doublewell_collapse').touch()


if __name__ == '__main__':
    main()
