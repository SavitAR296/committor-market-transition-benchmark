"""S5 -- double well seen only through two trailing windows (w, 4w) of x; the slow variable c is invisible.
tau_c = inf: latent 1-D gradient diffusion, exactly reversible (zero current) -> any fitted current is observer-induced.
Finite tau_c: compared with the latent current of (x, c) measured by the same estimator, and with the linearised
Proposition-3 value (A = [[-lam_bar, 1], [0, -1/tau_c]], D = diag(sigma_x^2, sigma_c^2)).
Proposition-2 prediction: unit AR(1) with phi = exp(-lam_bar dt) observed through (w, 4w), closed form.
Dynamics as in the frozen synthetic experiment (dt 0.05, sigma_x 0.72, sigma_c 0.18, clipping), 6 replicates x
1e5 steps per tau_c, seed 20261004.  Outputs: S5_doublewell_twowindow.csv, S5_summary.csv
"""
import sys, os, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import linalg
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from gaussian_observer import apply_windows, cov_discrete, markov_projection
from irreversibility import projection_stats, plugin_rates
from pre_common import write_csv

SEED = 20261004; TAUS = [5.0, 20.0, 80.0, np.inf]; WS = [1, 5, 10, 20, 40, 80]
DT, SIG_X, SIG_C, NREP, T_STEPS, BURN = 0.05, 0.72, 0.18, 6, 100_000, 2000
CACHE = ROOT / 'tables_pre' / '_cache'


def simulate(tau, rng):
    x = np.full(NREP, -1.0); c = np.zeros(NREP); X = np.empty((T_STEPS, NREP)); C = np.empty((T_STEPS, NREP))
    sx, sc = SIG_X * np.sqrt(DT), SIG_C * np.sqrt(DT)
    for i in range(T_STEPS):
        X[i] = x; C[i] = c
        drift = -(x ** 3 - x - c)
        if not np.isinf(tau):
            c = np.clip(c - c / tau * DT + sc * rng.normal(size=NREP), -0.45, 0.45)
        x = np.clip(x + drift * DT + sx * rng.normal(size=NREP), -2.2, 2.2)
    return X[BURN:], C[BURN:]


def main():
    LB = pd.read_csv(ROOT / 'tables_pre' / 'S4_lambda_bar.csv')
    lam = {('inf' if np.isinf(float(t)) else f'{float(t):g}'): v for t, v in zip(LB.tau_c, LB.lam_bar)}
    rows = []
    for k, tau in enumerate(TAUS):
        key = 'inf' if np.isinf(tau) else f'{tau:g}'; lb = lam[key]
        X, C = simulate(tau, np.random.default_rng([SEED, k]))
        if np.isinf(tau):
            ep3 = 0.0
        else:
            A = np.array([[-lb, 1.0], [0.0, -1.0 / tau]]); Dm = np.diag([SIG_X ** 2, SIG_C ** 2])
            Sg = linalg.solve_continuous_lyapunov(A, -Dm); Om = 0.5 * (A @ Sg - Sg @ A.T)
            ep3 = float(2 * np.trace(np.linalg.solve(Sg, Om.T) @ np.linalg.solve(Dm, Om))) * DT
        for rep in range(NREP):
            lat = projection_stats(np.c_[X[:, rep], C[:, rep]]) if not np.isinf(tau) else dict(ratio=0.0, ep=0.0, d1=0.0)
            for w in WS:
                Y = apply_windows(X[:, rep], [w, 4 * w]); Yf = Y[np.all(np.isfinite(Y), 1)]
                ob = projection_stats(Y); kl = plugin_rates(Yf, qs=(4,))
                phi = np.exp(-lb * DT); C0 = cov_discrete(phi, [w, 4 * w], 0); C1 = cov_discrete(phi, [w, 4 * w], 1)
                Sd = np.diag(1 / np.sqrt(np.diag(C0))); pr = markov_projection(Sd @ C0 @ Sd, Sd @ C1 @ Sd)
                rows.append(dict(tau_c=key, rep=rep, w=w, w2=4 * w, lam_bar=lb, lam_w=lb * w * DT,
                                 obs_ratio=ob['ratio'], obs_ep=ob['ep'], obs_d1=ob['d1'], obs_kld_q4_D2=kl['kld_q4_D2'],
                                 lat_ratio=lat['ratio'], lat_ep=lat['ep'], lat_d1=lat['d1'], prop3_linear_ep=ep3,
                                 pred_ratio=pr['ratio_lyap'], pred_ep=pr['ep_lyap'], pred_d1=pr['d1']))
        print('S5 tau', key, flush=True)
    D = pd.DataFrame(rows); write_csv(D, 'S5_doublewell_twowindow.csv')
    S = D.groupby(['tau_c', 'w']).agg(lam_w=('lam_w', 'first'), obs_ratio=('obs_ratio', 'mean'), obs_ep=('obs_ep', 'mean'),
                                      obs_ep_sd=('obs_ep', 'std'), obs_d1=('obs_d1', 'mean'), obs_kld=('obs_kld_q4_D2', 'mean'),
                                      lat_ep=('lat_ep', 'mean'), lat_ratio=('lat_ratio', 'mean'), prop3_linear_ep=('prop3_linear_ep', 'first'),
                                      pred_ratio=('pred_ratio', 'first'), pred_ep=('pred_ep', 'first'), pred_d1=('pred_d1', 'first')).reset_index()
    S['obs_over_latent_ep'] = np.where(S.lat_ep > 0, S.obs_ep / S.lat_ep, np.inf)
    S['obs_over_pred_ep'] = S.obs_ep / S.pred_ep
    write_csv(S, 'S5_summary.csv'); print(S.round(4).to_string(index=False))
    print('S5 complete'); (CACHE / 'DONE_S5_doublewell_twowindow').touch()


if __name__ == '__main__':
    main()
