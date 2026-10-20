"""T2 -- Gaussian observer model (Proposition 2): closed forms and simulations S1 (leak) and S2 (irreversibility).

Model: unit-variance AR(1) latent x (phi = e^{-lam}), threshold event E_t = 1{max_{1<=k<=h} x_{t+k} >= b}, b = 2,
observer y_t = trailing means over (w, 4w) steps.  Seed 20261001.

Outputs (tables_pre/):
  T2_table1_closed_form.csv   Table I: s^2, antisymmetric lag-one covariance, flux/drift ratio, e_p, d1 vs lam*w.
  T2_cov_check.csv            closed-form C(0), C(1) vs 10 x 10^6-step simulations (relative error and MC s.e.).
  T2_leak_curves.csv          exact leak L(s^2) (quadrature and Mehler series), kappa = E[p'^2]/Var(p).
  T2_spectral_kld.csv         full-path KLD rate with observation noise eta^2 (divergence as eta -> 0).
  S1_gaussian_leak.csv        oracle / Bayes observer / latent GBT / window GBT / window OU, AUC, BSS, REL, RES,
                              direct leak mean[(p_hat - p_op)^2], theory leak.
  S2_gaussian_irrev.csv       ratio, e_p, d1: closed form vs simulation; plug-in symbolic KLD; trailing, centred,
                              current+20-day and a reversible two-component control.
Runtime about 20 min on one core; resumable (per-unit cache in tables_pre/_cache).
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'reproducibility' / 'src'))
sys.path.insert(0, str(ROOT.parent / 'reproducibility' / 'experiments'))
from gaussian_observer import (cov_discrete, cross_xy_discrete, cov_continuous, cross_xy_continuous, posterior_variance,
                               s2_discrete, s2_continuous, AR1Hitting, leak_quadrature, hermite_coefficients,
                               leak_hermite, markov_projection, spectral_kld_rate, simulate_ar1, apply_windows)
from irreversibility import projection_stats, plugin_rates
from pre_common import write_csv, murphy, auc, brier
from B_csf_additions import fit_ou_scores          # frozen OU hitting-score machinery (unchanged)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

SEED = 20261001
T_START = time.time()
BUDGET = float(__import__('os').environ.get('PRE_BUDGET', '150'))   # seconds; units started after this are deferred
PHIS = [0.9, 0.95, 0.98, 0.99]
WS = [1, 2, 5, 10, 20, 40]
HS = [5, 10, 20]
B_THR = 2.0
T_SIM = 300_000
GBT_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                  early_stopping=False, random_state=0)


def std_moments(C0, C1):
    S = np.diag(1 / np.sqrt(np.diag(C0)))
    return S @ C0 @ S, S @ C1 @ S


def table1():
    rows = []
    phi = 0.98; lam = -np.log(phi)
    for lw in [0.1, 0.2, 0.4, 1.0, 2.0, 4.0]:
        w = max(1, int(round(lw / lam)))
        ws = [w, 4 * w]
        C0 = cov_discrete(phi, ws, 0); C1 = cov_discrete(phi, ws, 1)
        mp = markov_projection(*std_moments(C0, C1)); mpr = markov_projection(C0, C1)
        H = AR1Hitting(phi, B_THR, 20); xg = np.linspace(-7, 7, 28001); ug = H(xg, 20)
        u = lambda z: np.interp(z, xg, ug)
        s2 = s2_discrete(phi, ws)
        L = leak_quadrature(u, [s2, 1.0])
        rows.append(dict(lam_w=lw, phi=phi, w=w, w2=4 * w, lam_w_exact=lam * w, s2_discrete=s2,
                         s2_continuous=s2_continuous(lam, ws), omega_C1=0.5 * (C1[0, 1] - C1[1, 0]),
                         ratio_std=mp['ratio_lyap'], ratio_raw=mpr['ratio_lyap'], ep=mp['ep_lyap'], d1=mp['d1'],
                         leak_h20=L[0], var_pop_h20=L[1], ell_h20=L[0] / L[1]))
    return pd.DataFrame(rows)


def cov_check(rng):
    phi, ws = 0.98, [5, 20]
    C = {s: cov_discrete(phi, ws, s) for s in [0, 1, 5]}
    est = {s: [] for s in C}
    for r in range(10):
        x = simulate_ar1(phi, 1_000_000, rng); Y = apply_windows(x, ws); Y = Y[np.all(np.isfinite(Y), 1)]
        Y = Y - Y.mean(0)
        for s in C:
            est[s].append((Y[s:].T @ Y[:len(Y) - s]) / (len(Y) - s))
    rows = []
    for s in C:
        E = np.array(est[s]); mean = E.mean(0); se = E.std(0, ddof=1) / np.sqrt(len(E))
        for i in range(2):
            for j in range(2):
                rows.append(dict(phi=phi, w1=ws[0], w2=ws[1], lag=s, i=i, j=j, closed_form=C[s][i, j],
                                 sim_mean=mean[i, j], sim_se=se[i, j],
                                 rel_err=abs(mean[i, j] - C[s][i, j]) / abs(C[s][i, j]),
                                 z_score=(mean[i, j] - C[s][i, j]) / se[i, j]))
    # continuous-time closed form vs discrete sums in the continuum limit (lam -> 0, lam*w fixed)
    for lw in [0.4, 2.0]:
        lam = 1e-3; w = int(round(lw / lam))
        Cd = cov_discrete(np.exp(-lam), [w, 4 * w], 1); Cc = cov_continuous(lam, [w, 4 * w], 1.0)
        rows.append(dict(phi=np.exp(-lam), w1=w, w2=4 * w, lag=1, i=-1, j=-1, closed_form=Cc[0, 1], sim_mean=Cd[0, 1],
                         sim_se=np.nan, rel_err=abs(Cc[0, 1] - Cd[0, 1]) / abs(Cc[0, 1]), z_score=np.nan))
    return pd.DataFrame(rows)


def leak_curves():
    rows = []
    s2grid = np.linspace(0, 1, 41)
    for phi in PHIS:
        H = AR1Hitting(phi, B_THR, max(HS)); xg = np.linspace(-7, 7, 28001)
        for h in HS:
            ug = H(xg, h)
            u = lambda z: np.interp(z, xg, ug)
            a = hermite_coefficients(u, nmax=200)
            Lq = leak_quadrature(u, s2grid); Lh = leak_hermite(a, s2grid)
            var = Lq[-1]; kappa = np.sum(np.arange(len(a)) * a ** 2) / np.sum(a[1:] ** 2)
            for s2, lq, lh in zip(s2grid, Lq, Lh):
                rows.append(dict(phi=phi, h=h, s2=s2, leak_quad=lq, leak_hermite=lh, var_pop=var, ell=lq / var,
                                 kappa=kappa))
    return pd.DataFrame(rows)


def spectral_table():
    rows = []
    for phi in [0.9, 0.98]:
        for kind in ['trailing', 'centered']:
            for eta2 in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                rows.append(dict(phi=phi, w1=5, w2=20, kind=kind, eta2=eta2,
                                 kld_rate=spectral_kld_rate(phi, [5, 20], eta2, kinds=[kind, kind])))
    return pd.DataFrame(rows)


def event_labels(x, h, b):
    E = sliding_window_view(x[1:], h).max(1) >= b
    y = np.full(len(x), np.nan); y[:len(E)] = E
    return y


def ou_score(Ytr, Yte, h):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ytr)
    out, _, _ = fit_ou_scores(sc.transform(Ytr), sc.transform(Yte), Ytr[:, 0], horizons=(h,), grid_n=40)
    return out[h]


def simulate_phi(phi, rng):
    s1, s2rows = [], []
    if True:
        t0 = time.time()
        lam = -np.log(phi)
        x = simulate_ar1(phi, T_SIM, rng)
        H = AR1Hitting(phi, B_THR, max(HS))
        xg = np.linspace(-7, 7, 28001)
        ug = {h: H(xg, h) for h in HS}
        split = int(0.6 * T_SIM)
        idx = np.arange(T_SIM)
        labels = {h: event_labels(x, h, B_THR) for h in HS}
        # latent GBT (1-D, current x) -- depends on (phi, h) only
        latent = {}
        for h in HS:
            y = labels[h]; ok = np.isfinite(y) & (idx >= 4 * max(WS))   # common burn-in = widest window (160 steps)
            tr = ok & (idx < split); te = ok & (idx >= split)
            g = HistGradientBoostingClassifier(**GBT_PARAMS).fit(x[tr, None], y[tr].astype(int))
            latent[h] = (tr, te, g.predict_proba(x[te, None])[:, 1])
        for w in WS:
            ws = [w, 4 * w]
            Y = apply_windows(x, ws)
            C0 = cov_discrete(phi, ws, 0); c = cross_xy_discrete(phi, ws)
            s2 = posterior_variance(C0, c); beta = np.linalg.solve(C0, c)
            mpost = Y @ beta                                   # posterior mean of x_t given y_t
            for h in HS:
                y = labels[h]; tr, te, p_lat = latent[h]
                p_or = np.interp(x[te], xg, ug[h])
                # exact Bayes observer r*(y) = E[u_h(X) | m], X ~ N(m, s2)
                zq, wq = np.polynomial.hermite_e.hermegauss(80); wq = wq / wq.sum()
                mg = np.linspace(-6, 6, 4001)
                rg = (np.interp(mg[:, None] + np.sqrt(s2) * zq[None, :], xg, ug[h]) * wq).sum(1)
                p_bayes = np.interp(mpost[te], mg, rg)
                g = HistGradientBoostingClassifier(**GBT_PARAMS).fit(Y[tr], y[tr].astype(int))
                p_win = g.predict_proba(Y[te])[:, 1]
                p_ou_tr, p_ou = ou_score(Y[tr], Y[te], h)
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(p_ou_tr[:, None], y[tr].astype(int))
                p_ou_cal = lr.predict_proba(p_ou[:, None])[:, 1]
                yy = y[te]; clim = float(np.mean(y[tr])); bs0 = np.mean((yy - clim) ** 2)
                u = lambda z: np.interp(z, xg, ug[h])
                Lth, var = leak_quadrature(u, [s2, 1.0])
                for name, p in [('oracle', p_or), ('bayes_observer', p_bayes), ('latent_gbt', p_lat),
                                ('window_gbt', p_win), ('window_ou_raw', p_ou), ('window_ou_cal', p_ou_cal)]:
                    md = murphy(yy, p)
                    s1.append(dict(phi=phi, lam=lam, w=w, w2=4 * w, lam_w=lam * w, h=h, lam_h=lam * h, method=name,
                                   n=len(yy), events=int(yy.sum()), AUC=auc(yy, p), BS=brier(yy, p),
                                   BSS=1 - brier(yy, p) / bs0, REL=md['REL'], RES=md['RES'], UNC=md['UNC'],
                                   leak_direct=float(np.mean((p - p_or) ** 2)), s2=s2, leak_theory=Lth,
                                   var_pop=var, ell_theory=Lth / var))
            # irreversibility of the observed state (S2)
            for kind, wpair in [('trailing', ws), ('centered', ws)]:
                Yk = Y if kind == 'trailing' else apply_windows(x, wpair, ['centered'] * 2)
                C0k = cov_discrete(phi, wpair, 0, [kind] * 2); C1k = cov_discrete(phi, wpair, 1, [kind] * 2)
                mp = markov_projection(*std_moments(C0k, C1k))
                emp = projection_stats(Yk)
                row = dict(phi=phi, lam=lam, w=w, w2=4 * w, lam_w=lam * w, kind=kind,
                           omega_C1_closed=0.5 * (C1k[0, 1] - C1k[1, 0]), ratio_closed=mp['ratio_lyap'],
                           ep_closed=mp['ep_lyap'], d1_closed=mp['d1'], ratio_sim=emp['ratio'], ep_sim=emp['ep'],
                           d1_sim=emp['d1'])
                row.update(plugin_rates(Yk[np.all(np.isfinite(Yk), 1)]))
                s2rows.append(row)
        # controls: current value + 20-day window; reversible two-component latent
        Yc = apply_windows(x, [1, 20])
        C0k = cov_discrete(phi, [1, 20], 0); C1k = cov_discrete(phi, [1, 20], 1)
        mp = markov_projection(*std_moments(C0k, C1k)); emp = projection_stats(Yc)
        row = dict(phi=phi, lam=lam, w=1, w2=20, lam_w=lam, kind='current+20', omega_C1_closed=0.5 * (C1k[0, 1] - C1k[1, 0]),
                   ratio_closed=mp['ratio_lyap'], ep_closed=mp['ep_lyap'], d1_closed=mp['d1'],
                   ratio_sim=emp['ratio'], ep_sim=emp['ep'], d1_sim=emp['d1'])
        row.update(plugin_rates(Yc[np.all(np.isfinite(Yc), 1)])); s2rows.append(row)
        x2 = simulate_ar1(phi ** 4, T_SIM, rng)        # independent, reversible second coordinate
        Yr = np.c_[x, x2]; emp = projection_stats(Yr)
        row = dict(phi=phi, lam=lam, w=1, w2=1, lam_w=np.nan, kind='reversible_2d_latent', omega_C1_closed=0.0,
                   ratio_closed=0.0, ep_closed=0.0, d1_closed=0.0, ratio_sim=emp['ratio'], ep_sim=emp['ep'], d1_sim=emp['d1'])
        row.update(plugin_rates(Yr)); s2rows.append(row)
        print(f'phi={phi} done in {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(s1), pd.DataFrame(s2rows)


def replicate_spread(rng):
    reps = []
    for r in range(5):
        x = simulate_ar1(0.98, T_SIM, rng); emp = projection_stats(apply_windows(x, [5, 20]))
        reps.append(dict(phi=0.98, w=5, w2=20, kind='trailing_replicate', replicate=r, ratio_sim=emp['ratio'],
                         ep_sim=emp['ep'], d1_sim=emp['d1']))
    return pd.DataFrame(reps)


def pilot_row9(rng):
    """Outline Sec. 1 row 9: phi=0.98, b=2, h=10; windows (1,5),(5,20),(10,40),(20,80)."""
    phi, h = 0.98, 10
    x = simulate_ar1(phi, T_SIM, rng); H = AR1Hitting(phi, B_THR, h)
    xg = np.linspace(-7, 7, 28001); ug = H(xg, h)
    y = event_labels(x, h, B_THR); idx = np.arange(T_SIM); split = int(0.6 * T_SIM)
    ok = np.isfinite(y) & (idx >= 80); tr = ok & (idx < split); te = ok & (idx >= split)
    rows = []
    fits = {'latent': HistGradientBoostingClassifier(**GBT_PARAMS).fit(x[tr, None], y[tr].astype(int))}
    for ws in [[1, 5], [5, 20], [10, 40], [20, 80]]:
        fits[tuple(ws)] = HistGradientBoostingClassifier(**GBT_PARAMS).fit(apply_windows(x, ws)[tr], y[tr].astype(int))
    # scoring 'all': every test day;  'first_passage': only test days with x_t < b (latent not yet in the target set)
    for scoring, sel in [('all', te), ('first_passage', te & (x < B_THR))]:
        yy = y[sel]; clim = float(np.mean(y[tr])); bs0 = np.mean((yy - clim) ** 2)
        for name, p in [('oracle', np.interp(x[sel], xg, ug)), ('latent_gbt', fits['latent'].predict_proba(x[sel, None])[:, 1])]:
            rows.append(dict(scoring=scoring, windows='latent', method=name, AUC=auc(yy, p), BSS=1 - brier(yy, p) / bs0))
        for ws in [[1, 5], [5, 20], [10, 40], [20, 80]]:
            p = fits[tuple(ws)].predict_proba(apply_windows(x, ws)[sel])[:, 1]
            rows.append(dict(scoring=scoring, windows=f'({ws[0]},{ws[1]})', method='window_gbt', AUC=auc(yy, p), BSS=1 - brier(yy, p) / bs0))
    return pd.DataFrame(rows)


def cached(name, fn):
    cache = ROOT / 'tables_pre' / '_cache'; cache.mkdir(exist_ok=True)
    f = cache / f'T2_{name}.pkl'
    if f.exists():
        return pd.read_pickle(f)
    t0 = time.time(); out = fn(); pd.to_pickle(out, f)
    print(f'{name}: {time.time() - t0:.0f}s', flush=True)
    return out


def main():
    """Every unit uses its own seed stream np.random.default_rng([SEED, k]) and is cached in tables_pre/_cache,
    so an interrupted run resumes without changing any number."""
    write_csv(cached('table1', table1), 'T2_table1_closed_form.csv')
    write_csv(cached('cov', lambda: cov_check(np.random.default_rng([SEED, 0]))), 'T2_cov_check.csv')
    write_csv(cached('leak', leak_curves), 'T2_leak_curves.csv')
    write_csv(cached('spectral', spectral_table), 'T2_spectral_kld.csv')
    write_csv(cached('pilot9', lambda: pilot_row9(np.random.default_rng([SEED, 3]))), 'T2_pilot_row9.csv')
    reps = cached('reps', lambda: replicate_spread(np.random.default_rng([SEED, 2])))
    s1s, s2s = [], []
    for k, phi in enumerate(PHIS):
        if time.time() - T_START > BUDGET and not (ROOT / 'tables_pre' / '_cache' / f'T2_phi{phi}.pkl').exists():
            print('time budget reached; rerun to continue'); return
        a, b = cached(f'phi{phi}', lambda: simulate_phi(phi, np.random.default_rng([SEED, 1, k])))
        s1s.append(a); s2s.append(b)
    write_csv(pd.concat(s1s, ignore_index=True), 'S1_gaussian_leak.csv')
    write_csv(pd.concat(s2s + [reps], ignore_index=True), 'S2_gaussian_irrev.csv')
    print('T2 complete')


if __name__ == '__main__':
    main()
