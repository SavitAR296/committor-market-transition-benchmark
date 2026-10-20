"""S4c -- empirical two-parameter scaling of the double-well compression leak (outline S4 failure branch):
Delta BSS(lam_bar w, lam_bar tau_c) = a (lam_bar w dt)^b  [no tau_c dependence, as the collapse suggests], fitted by
least squares on all (tau_c, w>1) cells of S4_summary.csv; plus the spread across tau_c at fixed w relative to the
Monte-Carlo standard error.  Output: S4c_scaling_fit.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src'))
from pre_common import write_csv
S = pd.read_csv(ROOT / 'tables_pre' / 'S4_summary.csv'); S = S[S.w > 1]
f = lambda u, a, b: a * u ** b
(a, b), cov = curve_fit(f, S.lam_w.values, S.dBSS_meas.values, p0=[0.04, 0.9])
res = S.dBSS_meas.values - f(S.lam_w.values, a, b)
r2 = 1 - np.sum(res ** 2) / np.sum((S.dBSS_meas - S.dBSS_meas.mean()) ** 2)
inf = S[S.tau_c.astype(float) == np.inf].set_index('w').dBSS_meas
S['dev_from_tau_inf'] = S.dBSS_meas - S.w.map(inf)
spread = S.groupby('w').agg(spread=('dBSS_meas', lambda v: v.max() - v.min()), max_dev_from_inf=('dev_from_tau_inf', lambda v: v.abs().max()),
                             mean_se=('dBSS_meas_se', 'mean')).reset_index()
out = pd.DataFrame([dict(form='a (lam_bar w dt)^b', a=a, b=b, se_a=np.sqrt(cov[0, 0]), se_b=np.sqrt(cov[1, 1]),
                         r2=r2, rmse=np.sqrt(np.mean(res ** 2)), n_cells=len(S))])
write_csv(out, 'S4c_scaling_fit.csv'); write_csv(spread, 'S4c_tau_spread.csv')
print(out.round(4).to_string(index=False)); print(spread.round(4).to_string(index=False))
