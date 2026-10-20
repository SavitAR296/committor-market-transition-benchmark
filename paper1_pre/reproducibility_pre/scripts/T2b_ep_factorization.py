"""T2b -- exact factorisation of the fitted entropy production (m = 2) and its sensitivity to observation noise.

For a two-dimensional OU model dZ = A Z dt + G dW (GG^T = D) with stationary covariance Sigma, every 2x2
antisymmetric matrix is omega*J, J = [[0,1],[-1,0]], and J^T D^{-1} J = D / det D.  Hence, exactly,

    e_p = 2 tr(Sigma^{-1} Omega^T D^{-1} Omega) = 2 omega^2 * G,   G = tr(Sigma^{-1} D) / det D,

i.e. (antisymmetry)^2 x (innovation geometry).  This script verifies the identity on the Gaussian observer
(AR(1) latent, trailing windows (w, 4w)) and shows that independent white observation noise of variance
eta^2 leaves omega almost unchanged while G (and hence e_p) collapses: the large noise-free e_p of Table I is a
statement about nearly collinear innovations, not about a large current.
Output: tables_pre/T2b_ep_factorization.csv          Deterministic (no random numbers).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gaussian_observer import cov_discrete  # exact C(s) for trailing windows of an AR(1)


def fitted_ou(C0, C1):
    """Population least-squares VAR(1)/OU fit from (C(0), C(1)) and its factorised entropy production."""
    Phi = C1 @ np.linalg.inv(C0)
    A = Phi - np.eye(2)
    D = C0 - Phi @ C0 @ Phi.T
    D = 0.5 * (D + D.T)
    Sig = linalg.solve_continuous_lyapunov(A, -D)
    Sig = 0.5 * (Sig + Sig.T)
    Om = 0.5 * (A @ Sig - Sig @ A.T)
    ep = float(2 * np.trace(np.linalg.solve(Sig, Om.T) @ np.linalg.solve(D, Om)))
    omega = float(Om[0, 1])
    G = float(np.trace(np.linalg.solve(Sig, D)) / np.linalg.det(D))
    Lc = np.linalg.cholesky(Sig)
    Dw = np.linalg.solve(Lc, np.linalg.solve(Lc, D).T)          # D in whitened coordinates
    dmin, dmax = np.sort(np.linalg.eigvalsh(0.5 * (Dw + Dw.T)))
    omega_C1 = 0.5 * (C1[0, 1] - C1[1, 0]) / np.sqrt(C0[0, 0] * C0[1, 1])
    return dict(ep=ep, omega=omega, omega_C1=omega_C1, G=G, d_min=dmin, d_max=dmax,
                identity_residual=ep - 2 * omega ** 2 * G)


def main():
    rows = []
    for phi in [0.95, 0.98]:
        lam = -np.log(phi)
        for w in [5, 10, 20, 49]:
            ws = [w, 4 * w]
            C0 = cov_discrete(phi, ws, 0)
            C1 = cov_discrete(phi, ws, 1)
            for frac in [0.0, 0.001, 0.01, 0.1]:          # eta^2 as a fraction of each window's variance
                out = fitted_ou(C0 + np.diag(frac * np.diag(C0)), C1)
                rows.append(dict(phi=phi, w=w, w2=4 * w, lam_w=lam * w, eta2_frac=frac, **out))
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / 'tables_pre' / 'T2b_ep_factorization.csv', index=False, float_format='%.8g')
    print('max |identity residual| =', float(D.identity_residual.abs().max()))
    show = D[D.phi == 0.98][['w', 'eta2_frac', 'ep', 'omega', 'G', 'd_min', 'd_max']]
    print(show.to_string(index=False, float_format=lambda x: f'{x:.5g}'))


if __name__ == '__main__':
    main()
