"""T2d -- Table I at the forecast horizon of Fig. 2(a) (h = 10) and the fitted-model antisymmetry omega_fit.
For each row of Table I (phi = 0.98, windows (w, 4w), standardized coordinates): omega = [C12(1)-C21(1)]/2 (Eq. 3),
omega_fit = Omega_12 of the fitted OU model, G, e_p = 2 omega_fit^2 G, the same with observation noise eta^2, and the
normalized compression loss ell of the 2-sigma threshold event at horizons 10 and 20 (exact, Lemma 1 / quadrature).
Deterministic.  Output: tables_pre/T2d_table1.csv"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from gaussian_observer import cov_discrete, s2_discrete, AR1Hitting, leak_quadrature
from T2b_ep_factorization import fitted_ou
PHI, B_THR = 0.98, 2.0


def oracle(h):
    xg = np.linspace(-7, 7, 28001); ug = AR1Hitting(PHI, B_THR, h)(xg, h)
    return lambda z: np.interp(z, xg, ug)


def main():
    lam = -np.log(PHI); U = {h: oracle(h) for h in (10, 20)}; rows = []
    for lw in [0.1, 0.2, 0.4, 1.0, 2.0, 4.0]:
        w = max(1, int(round(lw / lam))); ws = [w, 4 * w]
        C0 = cov_discrete(PHI, ws, 0); C1 = cov_discrete(PHI, ws, 1); Sd = np.diag(1 / np.sqrt(np.diag(C0)))
        C0s, C1s = Sd @ C0 @ Sd, Sd @ C1 @ Sd
        o0 = fitted_ou(C0s, C1s); o2 = fitted_ou(C0s + 1e-2 * np.eye(2), C1s); o1 = fitted_ou(C0s + 1e-1 * np.eye(2), C1s)
        s2 = s2_discrete(PHI, ws)
        r = dict(lam_w=lw, w=w, w2=4 * w, s2=s2, omega=o0['omega_C1'], omega_fit=o0['omega'], G=o0['G'], ep=o0['ep'],
                 check_2wf2G=2 * o0['omega'] ** 2 * o0['G'], ep_eta2_p01=o2['ep'], omega_eta2_p01=o2['omega_C1'],
                 ep_eta2_p10=o1['ep'], omega_eta2_p10=o1['omega_C1'])
        for h in (10, 20):
            L = leak_quadrature(U[h], [s2, 1.0]); r[f'ell_h{h}'] = L[0] / L[1]
        rows.append(r)
    D = pd.DataFrame(rows); D.to_csv(ROOT / 'tables_pre' / 'T2d_table1.csv', index=False, float_format='%.8g')
    old = pd.read_csv(ROOT / 'tables_pre' / 'T2_table1_closed_form.csv')
    assert np.allclose(D.ell_h20, old.ell_h20, rtol=1e-6), 'h=20 losses must reproduce T2_table1_closed_form.csv'
    print(D.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
