"""E4g -- uncertainty of Table II and sensitivity to the one patched volume.
(i)  Moving-block bootstrap standard errors of omega, omega_rev, omega_cen, O and P for every row of Table II.  The
     lag-one pairs of the forward, time-reversed and centred feature series are aligned on the original time axis
     (a reversed pair (s, s+1) is the original pair (t-1, t), t = N-s) and resampled jointly in blocks of L days,
     L = 20 (Table II) and 60 (sensitivity), B = 1000; means and variances are recomputed in every replicate, and the
     PCA(2) state uses the fixed basis of the observed loadings, as in E4e.
(ii) S&P 500, 2000-2026: the vendor volume of 2023-05-24 replaced by the mean of the two adjacent days.
(iii) Loadings of the observed PCA(2) basis.
Outputs: tables_pre/E4g_bootstrap_se.csv, E4g_volume_sensitivity.csv, E4g_pca_loadings.csv.  Seeds [20261028, k, L].
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT.parent / 'reproducibility' / 'src'))
from E4c_channel_decomposition import standardise, pca_basis, reversed_returns, PAIRS, load_spx, ou_parts
from E4e_decomposition_extensions import feats_ann, decompose
from E4_surrogates import market_inputs
from features_fast import COMMON
SEED, B, BLOCKS, CHUNK = 20261028, 1000, (20, 60), 125
OUT = ROOT / 'tables_pre'


def mbb_index(n, L, B, rng):
    k = (n + L - 1) // L
    st = rng.integers(0, max(1, n - L + 1), size=(B, k))
    return np.minimum((st[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n], n - 1)


def pairs(Y, index, key_of):
    """Lag-one pairs (rows t, t+1 consecutive in the original index) keyed on the original time axis."""
    idx = np.asarray(index); ok = np.diff(idx) == 1
    X0, X1 = Y[:-1][ok], Y[1:][ok]
    return pd.DataFrame(np.c_[X0, X1], index=key_of(idx[:-1][ok], idx[1:][ok]))


def omega_rep(X0, X1):
    m = X0.mean(1); s = X0.std(1)
    c01 = ((X1[..., 0] - m[:, None, 0]) * (X0[..., 1] - m[:, None, 1])).mean(1)
    c10 = ((X1[..., 1] - m[:, None, 1]) * (X0[..., 0] - m[:, None, 0])).mean(1)
    return 0.5 * (c01 - c10) / (s[:, 0] * s[:, 1])


def sample_rows(label, k, c, v, ann):
    Np = len(c)
    F = {'fwd': feats_ann(c, v, ann), 'rev': feats_ann(*reversed_returns(c, v), ann),
         'cen': feats_ann(c, v, ann, 'centered')}
    V = pca_basis(F['fwd'])
    keyf = {'fwd': lambda a, b: a, 'cen': lambda a, b: a, 'rev': lambda a, b: (Np - 1) - b}
    chans = {name: (lambda Fk, ab=ab: standardise(Fk[list(ab)].to_numpy())) for name, ab in PAIRS.items()}
    chans['PCA2_common_basis'] = lambda Fk: standardise(standardise(Fk.to_numpy()) @ V)
    rows = []
    for name, f in chans.items():
        P = {d: pairs(f(F[d]), F[d].index, keyf[d]) for d in F}
        common = P['fwd'].index.intersection(P['rev'].index).intersection(P['cen'].index)
        A = {d: P[d].loc[common].to_numpy() for d in P}; n = len(common)
        for L in BLOCKS:
            rng = np.random.default_rng([SEED, k, L]); ix_all = mbb_index(n, L, B, rng); rep = {d: [] for d in A}
            for j in range(0, B, CHUNK):
                ix = ix_all[j:j + CHUNK]
                for d in A:
                    rep[d].append(omega_rep(A[d][ix][..., :2], A[d][ix][..., 2:]))
            rep = {d: np.concatenate(x) for d, x in rep.items()}
            O = 0.5 * (rep['fwd'] + rep['rev']); Pp = 0.5 * (rep['fwd'] - rep['rev'])
            rows.append(dict(sample=label, channel=name, block=L, n_pairs=n, se_fwd=rep['fwd'].std(ddof=1),
                             se_rev=rep['rev'].std(ddof=1), se_cen=rep['cen'].std(ddof=1), se_O=O.std(ddof=1),
                             se_P=Pp.std(ddof=1)))
    return rows, V


def main():
    c, v, _ = load_spx(); rows = []
    r, Vs = sample_rows('SPX 2000-2026', 0, c, v, 252.0); rows += r
    r, _ = sample_rows('SPX 2000-2020', 1, c[:5105], v[:5105], 252.0); rows += r
    for k, m in enumerate(['BTC', 'ETH'], start=2):
        cm, vm, ann = market_inputs(m); r, _ = sample_rows(m, k, cm, vm, ann); rows += r
    S = pd.DataFrame(rows); S.to_csv(OUT / 'E4g_bootstrap_se.csv', index=False, float_format='%.6g')
    w = S.pivot_table(index=['sample', 'channel'], columns='block', values=['se_fwd', 'se_P'])
    print(w.round(4).to_string())
    ratio = (S[S.block == 60].set_index(['sample', 'channel'])[['se_fwd', 'se_rev', 'se_cen', 'se_O', 'se_P']] /
             S[S.block == 20].set_index(['sample', 'channel'])[['se_fwd', 'se_rev', 'se_cen', 'se_O', 'se_P']])
    print('SE ratio block60/block20: min %.2f max %.2f' % (ratio.min().min(), ratio.max().max()))
    pd.DataFrame(Vs, index=COMMON, columns=['PC1', 'PC2']).to_csv(OUT / 'E4g_pca_loadings.csv', float_format='%.6g')
    print(pd.DataFrame(Vs, index=COMMON, columns=['PC1', 'PC2']).round(3).to_string())
    d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026_r1.csv'); i = int(np.flatnonzero(d.date == '2023-05-24')[0])
    v_alt = v.copy(); v_alt[i] = 0.5 * (v[i - 1] + v[i + 1])
    base, Vb = decompose('SPX 2000-2026', c, v, 252.0); alt, Va = decompose('SPX 2000-2026', c, v_alt, 252.0)
    out = []
    for a, b in zip(base, alt):
        for q in ['omega_fwd', 'omega_rev', 'omega_cen', 'O', 'P', 'ep_fwd']:
            if q in a:
                out.append(dict(channel=a['channel'], quantity=q, patched=a[q], adjacent_mean=b[q], diff=b[q] - a[q]))
    for lab, vv in [('patched', v), ('adjacent_mean', v_alt)]:
        F = feats_ann(c, vv, 252.0)
        out.append(dict(channel='PCA2_own_basis', quantity='ep', **{lab: ou_parts(standardise(standardise(F.to_numpy()) @ pca_basis(F)))['ep']}))
    Vd = pd.DataFrame(out)
    e = Vd[Vd.channel == 'PCA2_own_basis']; Vd = Vd[Vd.channel != 'PCA2_own_basis']
    Vd = pd.concat([Vd, pd.DataFrame([dict(channel='PCA2_own_basis', quantity='ep', patched=e.patched.dropna().iloc[0],
                                           adjacent_mean=e.adjacent_mean.dropna().iloc[0],
                                           diff=e.adjacent_mean.dropna().iloc[0] - e.patched.dropna().iloc[0])])])
    Vd['adjacent_volume'] = v_alt[i]; Vd.to_csv(OUT / 'E4g_volume_sensitivity.csv', index=False, float_format='%.6g')
    print(Vd.to_string(index=False)); print('max |diff| =', Vd['diff'].abs().max())


if __name__ == '__main__':
    main()
