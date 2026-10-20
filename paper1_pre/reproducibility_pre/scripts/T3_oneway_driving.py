"""T3 -- Proposition 3: physical irreversibility of one-way slow driving (linear skeleton of B-tipping).

    dx = (-lam x + c) dt + sqrt(2 lam) dW1,   dc = -mu c dt + sigma_c dW2,   mu = 1/tau_c.
Closed form (Appendix B):  Sigma_cc = sigma_c^2/(2 mu),  Sigma_xc = Sigma_cc/(lam+mu),  Sigma_xx = 1 + Sigma_xc/lam,
    Omega = omega J,  omega = mu Sigma_xc = sigma_c^2 / (2 (lam+mu)),
    e_p = 2 omega^2 (Sigma_cc/sigma_c^2 + Sigma_xx/(2 lam)) / det Sigma   ->  sigma_c^2/(2 lam^2)  as mu -> 0.
Simulation: exact discretisation (step dt) and the Markov-projection estimator of Appendix D.  Seed 20261002.
Output: tables_pre/T3_oneway.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import linalg
from scipy.signal import lfilter
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from pre_common import write_csv
from irreversibility import projection_stats

SEED = 20261002
LAM, SIG_C, DT, N_STEPS = 2.0, 0.18, 0.05, 2_000_000


def closed_form(lam, sig_c, mu):
    if mu == 0:
        return dict(omega=sig_c ** 2 / (2 * lam), ep=sig_c ** 2 / (2 * lam ** 2))
    Scc = sig_c ** 2 / (2 * mu); Sxc = Scc / (lam + mu); Sxx = 1 + Sxc / lam
    det = Sxx * Scc - Sxc ** 2
    omega = mu * Sxc
    ep = 2 * omega ** 2 * (Scc / sig_c ** 2 + Sxx / (2 * lam)) / det
    # cross-check with the generic matrix formula
    A = np.array([[-lam, 1.0], [0.0, -mu]]); D = np.diag([2 * lam, sig_c ** 2])
    S = linalg.solve_continuous_lyapunov(A, -D); Om = 0.5 * (A @ S - S @ A.T)
    ep_m = 2 * np.trace(np.linalg.solve(S, Om.T) @ np.linalg.solve(D, Om))
    assert abs(ep - ep_m) < 1e-10 * max(1, ep) and abs(omega - Om[0, 1]) < 1e-12
    return dict(omega=omega, ep=ep, Sxx=Sxx, Sxc=Sxc, Scc=Scc)


def simulate(lam, sig_c, mu, rng, n=N_STEPS, dt=DT):
    """Exact discretisation of the 2-D OU process (A upper triangular)."""
    A = np.array([[-lam, 1.0], [0.0, -mu]]); D = np.diag([2 * lam, sig_c ** 2])
    E = linalg.expm(A * dt)
    S = linalg.solve_continuous_lyapunov(A, -D)
    Q = S - E @ S @ E.T; Lq = np.linalg.cholesky(0.5 * (Q + Q.T))
    xi = rng.normal(size=(n, 2)) @ Lq.T
    c = lfilter([1.0], [1.0, -E[1, 1]], xi[:, 1])
    drive = np.r_[0.0, E[0, 1] * c[:-1]] + xi[:, 0]
    x = lfilter([1.0], [1.0, -E[0, 0]], drive)
    return np.c_[x, c][n // 20:]


def main():
    rng = np.random.default_rng(SEED); rows = []
    for tau in [5.0, 20.0, 80.0, 320.0, np.inf]:
        mu = 0.0 if np.isinf(tau) else 1.0 / tau
        cf = closed_form(LAM, SIG_C, mu)
        row = dict(lam=LAM, sigma_c=SIG_C, tau_c=tau, omega_closed=cf['omega'], ep_closed=cf['ep'])
        if mu > 0:
            Z = simulate(LAM, SIG_C, mu, rng)
            st = projection_stats(Z)
            row.update(ep_sim=st['ep'] / DT, ratio_sim_std=st['ratio'], d1_sim_per_time=st['d1'] / DT,
                       rel_err=abs(st['ep'] / DT - cf['ep']) / cf['ep'], dt=DT, n_steps=len(Z))
        rows.append(row)
    rows.append(dict(lam=LAM, sigma_c=0.0, tau_c=np.inf, omega_closed=0.0, ep_closed=0.0))   # c == 0: reversible
    df = pd.DataFrame(rows); write_csv(df, 'T3_oneway.csv'); print(df.to_string(index=False))


if __name__ == '__main__':
    main()
