"""T2c -- closed-form curves for Fig. 2: lag-one antisymmetry, innovation geometry and entropy production of the
OU model fitted to two trailing windows (w, 4w) of a unit AR(1) (phi = 0.98), versus lambda*w, for independent white
observation noise eta^2 in {0, 1e-3, 1e-2, 1e-1} x (window variance).  Uses T2b.fitted_ou (exact, no sampling).
Output: tables_pre/T2c_observer_curves.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from gaussian_observer import cov_discrete, s2_discrete
from T2b_ep_factorization import fitted_ou


def main(phi=0.98):
    lam = -np.log(phi); rows = []
    for w in [1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 57, 80, 113, 160, 200]:
        ws = [w, 4 * w]; C0 = cov_discrete(phi, ws, 0); C1 = cov_discrete(phi, ws, 1)
        for frac in [0.0, 1e-3, 1e-2, 1e-1]:
            o = fitted_ou(C0 + np.diag(frac * np.diag(C0)), C1)
            rows.append(dict(phi=phi, w=w, lam_w=lam * w, eta2_frac=frac, s2=s2_discrete(phi, ws), **o))
    pd.DataFrame(rows).to_csv(ROOT / 'tables_pre' / 'T2c_observer_curves.csv', index=False, float_format='%.8g')
    print(len(rows), 'rows')


if __name__ == '__main__':
    main()
