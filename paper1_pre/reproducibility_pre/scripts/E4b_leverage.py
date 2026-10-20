"""E4b -- where does the residual S&P irreversibility come from?  (comparison with the leverage effect)
(e) random-sign surrogates on the *actual* |r_t| and actual volume (x50, seed stream [20261005, 9, sid]): keeps the
real volatility path, removes the return-volatility coupling (leverage);
leverage correlation (Bouchaud, Matacz & Potters 2001): L(tau) = <r_t r_{t+tau}^2> / <r^2>^2 for tau = -20..20.
Outputs: E4b_leverage_surrogates.csv, E4b_leverage_function.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from E4_surrogates import market_inputs, state, stats
from pre_common import write_csv


def main():
    close, volume, ann = market_inputs('SPX'); r = np.diff(np.log(close)); rows = []
    for sid in range(50):
        rng = np.random.default_rng([20261005, 9, sid]); s = np.where(rng.random(len(r)) < np.mean(r > 0), 1.0, -1.0)
        c = close[0] * np.exp(np.r_[0.0, np.cumsum(s * np.abs(r))])
        rows.append(dict(variant='random_signs_actual_abs_r', sid=sid, **stats(state(c, volume, ann))))
    write_csv(pd.DataFrame(rows), 'E4b_leverage_surrogates.csv')
    rc = r - r.mean(); r2 = r ** 2; den = np.mean(r2) ** 2; L = []
    for tau in range(-20, 21):
        v = np.mean(rc[:len(r) - tau] * r2[tau:]) if tau >= 0 else np.mean(rc[-tau:] * r2[:len(r) + tau])
        L.append(dict(tau=tau, L=v / den))
    write_csv(pd.DataFrame(L), 'E4b_leverage_function.csv')
    D = pd.DataFrame(rows); print(D[['ratio', 'ep', 'd1', 'kld_q4_D2']].quantile([.05, .5, .95]).round(4).to_string())
    Lf = pd.DataFrame(L); print('sum L(1..20)=', Lf[Lf.tau > 0].L.sum().round(3), ' sum L(-20..-1)=', Lf[Lf.tau < 0].L.sum().round(3))


if __name__ == '__main__':
    main()
