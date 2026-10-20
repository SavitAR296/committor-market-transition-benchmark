"""E9 -- what the market's trailing windows cost in information.
Latent log variance: the one-factor stochastic-volatility model of Appendix E (phi = 0.98; innovation s.d. from
E4f_calibration.csv; no leverage), x = (h - mu)/sd(h) a unit AR(1).  Event: x exceeds 2 within the next 10 days;
oracle p_op(x) = AR1Hitting(0.98, 2, 10) (the threshold event of Table I and Fig. 2(a)).
(a) Exact (Lemma 1): noise-free trailing windows of x -- single windows of 5, 20, 60 and 250 days, and the pair (5, 20).
(b) Simulation, T = 10^6 days: observers that see what the market pipeline sees -- log realized volatility over 5 and
    20 days (windows of squared returns, a noisy proxy of the latent variance) and the seven-feature PCA(2) state --
    and, as a check of the estimator, the noise-free pair of (a).  L = out-of-sample mean of (g - p_op)^2 for a
    gradient-boosted regression g of p_op on the observation (first 60% for training), which exceeds the exact loss
    by the estimation error only.
Output: tables_pre/E9_market_information.csv.  Seed 20261029.
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gaussian_observer import s2_discrete, AR1Hitting, leak_quadrature
from features_fast import common7
PHI, B_THR, H, T, SEED = 0.98, 2.0, 10, 1_000_000, 20261029


def main():
    cal = pd.read_csv(ROOT / 'tables_pre' / 'E4f_calibration.csv').iloc[0]
    phi1, s1 = [float(x) for x in re.findall(r'\d*\.\d+(?:e[-+]?\d+)?', cal.one_factor)][:2]
    assert abs(phi1 - PHI) < 1e-12
    xg = np.linspace(-7, 7, 28001); ug = AR1Hitting(PHI, B_THR, H)(xg, H); U = lambda z: np.interp(z, xg, ug)
    rows = []
    for name, ws in [('window 5', [5]), ('window 20', [20]), ('window 60', [60]), ('window 250', [250]), ('windows (5,20)', [5, 20])]:
        s2 = s2_discrete(PHI, ws); L = leak_quadrature(U, [s2, 1.0])
        rows.append(dict(observer=name, method='exact (Lemma 1), noise-free windows of x', lam_w=-np.log(PHI) * ws[-1],
                         s2=s2, ell=L[0] / L[1]))
    d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026_r1.csv'); r_data = np.diff(np.log(d.close.to_numpy(float)))
    lv = np.log(d.volume.to_numpy(float)); rng = np.random.default_rng(SEED); burn = 5000; n = T + burn
    x = lfilter([np.sqrt(1 - PHI ** 2)], [1, -PHI], rng.standard_normal(n))[burn:]
    sd_h = s1 / np.sqrt(1 - PHI ** 2); mu = 2 * np.log(r_data.std()) - 0.5 * sd_h ** 2; h = mu + sd_h * x
    r = np.exp(h / 2) * rng.standard_normal(T)
    zeta = lfilter([cal.sd_zeta], [1, -cal.phi_zeta], rng.standard_normal(n))[burn:]
    close = d.close.iloc[0] * np.exp(np.r_[0.0, np.cumsum(r)])
    vol = np.exp(np.r_[lv.mean(), lv.mean() + cal.kappa * (h - mu) / 2 + zeta])
    F = common7(close, vol, 252.0)                              # features at price index t use r[:t], i.e. x[:t]
    xc = pd.Series(np.r_[np.nan, x])                            # latent value entering the features at index t
    p = pd.Series(np.r_[np.nan, U(x)])
    obs = {'windows (5,20)': pd.concat([xc.rolling(5).mean(), xc.rolling(20).mean()], axis=1),
           'log RV (5, 20 days)': F[['log_rv5', 'log_rv20']]}
    ok = F.notna().all(1) & p.notna(); i0 = int(np.flatnonzero(ok)[0]); idx = np.arange(i0, len(F)); ntr = int(0.6 * len(idx))
    Z = F.iloc[idx].to_numpy(); Zs = (Z - Z[:ntr].mean(0)) / Z[:ntr].std(0)
    w, Vv = np.linalg.eigh(np.cov(Zs[:ntr].T)); obs['PCA(2) state'] = pd.DataFrame(Zs @ Vv[:, ::-1][:, :2], index=idx)
    y = p.iloc[idx].to_numpy(); var_p = y[ntr:].var()
    for name, X in obs.items():
        X = X.loc[idx].to_numpy() if name != 'PCA(2) state' else X.to_numpy()
        g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100,
                                          early_stopping=False, random_state=0).fit(X[:ntr], y[:ntr])
        L = float(np.mean((g.predict(X[ntr:]) - y[ntr:]) ** 2))
        rows.append(dict(observer=name, method='simulation, out-of-sample regression', lam_w=np.nan, s2=np.nan,
                         ell=L / var_p, n_train=ntr, n_test=len(idx) - ntr))
    D = pd.DataFrame(rows); D['lam'] = -np.log(PHI); D['sd_h'] = sd_h
    D.to_csv(ROOT / 'tables_pre' / 'E9_market_information.csv', index=False, float_format='%.6g')
    print(D.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
