"""E4e -- robustness of the time-reversal decomposition (tasks E1 and R3 of the Paper-1 plan).

(i)   BTC and ETH (Coin Metrics, 365-day annualisation): feature-pair antisymmetries forward / time-reversed /
      centred with O and P estimates, PCA(2) in the common basis, and 100 reversible surrogates each.
(ii)  S&P 500 2000-01-03..2020-04-17 (frozen snapshot only): the same decomposition.
(iii) Symmetric-sign reversible surrogates for the S&P 500 (signs +-1 with probability 1/2, no drift): strictly
      time-reversible input; compared with the E4 construction (empirical up-probability, which leaves a drift).
(iv)  Third-order time asymmetry of |r| (Ramsey-Rothman type, lags 1..20) against 200 IAAFT surrogates of |r|:
      a property of the input that, by Corollary 1, no second-order statistic of windows of |r| can see.
Outputs: tables_pre/E4e_pairs.csv, E4e_surrogate_bands.csv, E4e_symmetric_sign.csv, E4e_third_order.csv
Seeds: [20261018, k, sid].  Runtime about 2 min on one core.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT.parent / 'reproducibility' / 'src'))
from E4c_channel_decomposition import (feats, ou_parts, standardise, pca_basis, reversed_returns, iaaft,
                                       reversible_surrogate, PAIRS, load_spx)
from E4_surrogates import market_inputs

OUT = ROOT / 'tables_pre'
SEED = 20261018


def decompose(label, c, v, ann):
    """Pair-level and PCA(2) common-basis antisymmetries, forward / reversed / centred, with O and P estimates."""
    F = {'fwd': feats_ann(c, v, ann), 'rev': feats_ann(*reversed_returns(c, v), ann),
         'cen': feats_ann(c, v, ann, 'centered')}
    rows = []
    for name, (a, b) in PAIRS.items():
        om = {k: ou_parts(standardise(F[k][[a, b]].to_numpy()))['omega_direct'] for k in F}
        rows.append(dict(sample=label, channel=name, **{f'omega_{k}': x for k, x in om.items()},
                         O=0.5 * (om['fwd'] + om['rev']), P=0.5 * (om['fwd'] - om['rev'])))
    V = pca_basis(F['fwd'])
    om = {k: ou_parts(standardise(standardise(F[k].to_numpy()) @ V))['omega_direct'] for k in F}
    ep = ou_parts(standardise(standardise(F['fwd'].to_numpy()) @ V))['ep']
    rows.append(dict(sample=label, channel='PCA2_common_basis', **{f'omega_{k}': x for k, x in om.items()},
                     O=0.5 * (om['fwd'] + om['rev']), P=0.5 * (om['fwd'] - om['rev']), ep_fwd=ep))
    return rows, V


def feats_ann(c, v, ann, kind='trailing'):
    from features_fast import common7
    return common7(c, v, ann, kind).dropna()


def surrogate_band(label, c, v, ann, V, n, k, symmetric=False):
    rows = []
    for sid in range(n):
        rng = np.random.default_rng([SEED, k, sid])
        cs, vs = (symmetric_surrogate if symmetric else reversible_surrogate)(c, v, rng)
        F = feats_ann(cs, vs, ann)
        r = dict(sample=label, sid=sid, symmetric=symmetric)
        for name, (a, b) in PAIRS.items():
            r[f'omega_{name}'] = ou_parts(standardise(F[[a, b]].to_numpy()))['omega_direct']
        r['omega_PCA2_common_basis'] = ou_parts(standardise(standardise(F.to_numpy()) @ V))['omega_direct']
        r['ep_PCA2_own'] = ou_parts(standardise(standardise(F.to_numpy()) @ pca_basis(F)))['ep']
        rows.append(r)
    return rows


def symmetric_surrogate(c, v, rng):
    """As reversible_surrogate, but signs are +-1 with probability 1/2: zero drift, strictly reversible input."""
    cs, vs = reversible_surrogate(c, v, rng)
    a = np.abs(np.diff(np.log(cs)))
    s = np.where(rng.random(len(a)) < 0.5, 1.0, -1.0)
    return c[0] * np.exp(np.r_[0.0, np.cumsum(s * a)]), vs


def rr3(x, K=20):
    z = (x - x.mean()) / x.std()
    return sum(np.mean(z[:-k] ** 2 * z[k:]) - np.mean(z[:-k] * z[k:] ** 2) for k in range(1, K + 1))


def main(n_crypto=100, n_sym=200, n_iaaft=200):
    pairs, bands = [], []
    c, v, _ = load_spx()
    rows, Vs = decompose('SPX 2000-2026', c, v, 252.0); pairs += rows
    rows, _ = decompose('SPX 2000-2020', c[:5105], v[:5105], 252.0); pairs += rows
    for k, m in enumerate(['BTC', 'ETH'], start=1):
        cm, vm, ann = market_inputs(m)
        rows, Vm = decompose(m, cm, vm, ann); pairs += rows
        bands += surrogate_band(m, cm, vm, ann, Vm, n_crypto, k)
        print(m, 'done', flush=True)
    P = pd.DataFrame(pairs); P.to_csv(OUT / 'E4e_pairs.csv', index=False, float_format='%.6g')
    print(P.round(4).to_string(index=False))

    sym = surrogate_band('SPX 2000-2026', c, v, 252.0, Vs, n_sym, 9, symmetric=True)
    S = pd.DataFrame(sym); S.to_csv(OUT / 'E4e_symmetric_sign.csv', index=False, float_format='%.6g')
    B = pd.DataFrame(bands); B.to_csv(OUT / 'E4e_surrogate_bands.csv', index=False, float_format='%.6g')
    q = lambda s: f'{s.median():+.4f} [{s.quantile(.05):+.4f}, {s.quantile(.95):+.4f}]'
    for lab, G in [('SPX symmetric-sign', S)] + [(m, B[B['sample'] == m]) for m in ['BTC', 'ETH']]:
        print(lab, '| window', q(G['omega_window(log_rv5,log_rv20)']), '| leverage',
              q(G['omega_leverage(ret20,log_rv20)']), '| PCA2 common', q(G['omega_PCA2_common_basis']),
              '| e_p own PCA2', q(G['ep_PCA2_own']))

    a = np.abs(np.diff(np.log(c)))
    rng = np.random.default_rng([SEED, 20, 0])
    null = np.array([rr3(iaaft(a, rng, 100)) for _ in range(n_iaaft)])
    obs, rev = rr3(a), rr3(a[::-1].copy())
    T = pd.DataFrame([dict(statistic='RR3(|r|), lags 1-20', observed=obs, reversed=rev, null_median=np.median(null),
                           null_q025=np.quantile(null, .025), null_q975=np.quantile(null, .975),
                           p_two_sided=float(np.mean(np.abs(null - np.median(null)) >= abs(obs - np.median(null)))),
                           n_null=n_iaaft)])
    T.to_csv(OUT / 'E4e_third_order.csv', index=False, float_format='%.6g')
    print(T.round(4).to_string(index=False))


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:4]))
