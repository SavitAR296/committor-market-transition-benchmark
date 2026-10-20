"""E4 -- is the irreversibility of the trailing-window market state observer-induced?  (decisive test)

Identical pipeline for every state: common-7 features (frozen formulas, src/features_fast.py) -> full-sample
standardisation -> PCA(2) -> re-standardisation -> Markov-projection statistics (ratio, e_p, d1; Appendix D) and the
plug-in symbolic KLD (q = 4 bins per component, pairs D2 and triples D3-D2).
States:  observed      actual prices and volume, trailing windows
         (a) reversed  log-return sequence and volume sequence reversed in time, prices rebuilt by cumulation
         (a2) reversed price path (returns reversed and negated)                        [supplementary]
         (b) reversible surrogates x N_SUR: IAAFT of |r| (exact marginal, matched spectrum => volatility clustering kept),
             i.i.d. signs (leverage destroyed), log-volume with the same random phases and a symmetrised (real)
             cross-spectrum, rank-remapped to its marginal => a time-reversible input with matched second-order structure
         (b2) literal IAAFT of r and independent IAAFT of log-volume x N_SUR2             [supplementary]
         (c) centred windows on the actual data (time-reversal symmetric observer)
         (d) Proposition-2 prediction: latent AR(1), lambda fitted to the autocorrelation of log RV20 (lags 1..60),
             observed through trailing means over {5, 20, 60, 250} days; standardised PCA(2); closed form.
Decision rule (outline Sec. 6): observed inside the (b) 5-95% band AND (c) clearly lower -> observer-induced;
otherwise residual irreversibility.  Seed 20261005.  Resumable (300-s command limit).
Outputs: tables_pre/E4_irreversibility_surrogates.csv, E4_summary.csv, E4_lambda_fit.csv
"""
import sys, os, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; LEG = ROOT.parent / 'reproducibility'
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(LEG / 'src'))
from features_fast import common7
from irreversibility import projection_stats, plugin_rates
from gaussian_observer import cov_discrete, markov_projection
from data_daily import load_market

SEED = 20261005; N_SUR = 200; N_SUR2 = 100; BATCH = 50
CACHE = ROOT / 'tables_pre' / '_cache'; CACHE.mkdir(exist_ok=True)
T0 = time.time(); BUDGET = float(os.environ.get('PRE_BUDGET', '200'))


def market_inputs(name):
    if name == 'SPX':
        # gated analysis file (E1e); log(volume) must be finite or every surrogate shares one volume path
        d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026_r1.csv', parse_dates=['date']); ann = 252.0
        assert (d.volume > 0).all() and (d.close > 0).all(), 'run E1e_analysis_data.py and E1c first'
        return d.close.to_numpy(float), d.volume.to_numpy(float), ann
    d = load_market(name)
    d = d[np.isfinite(d.close) & np.isfinite(d.volume) & (d.volume > 0)]
    first = d.index[0]; d = d.loc[first:]
    assert np.isfinite(d.volume).all()
    return d.close.to_numpy(float), d.volume.to_numpy(float), 365.0


def state(close, volume, ann, kind='trailing'):
    X = common7(close, volume, ann, kind).dropna().to_numpy()
    X = (X - X.mean(0)) / X.std(0)
    w, V = np.linalg.eigh(np.cov(X.T)); V = V[:, ::-1][:, :2]
    Z = X @ V; return (Z - Z.mean(0)) / Z.std(0)


def stats(Z):
    s = projection_stats(Z); out = dict(ratio=s['ratio'], ep=s['ep'], d1=s['d1'], omega_C1=s['omega_C1'], n=s['n_pairs'])
    out.update({k: v for k, v in plugin_rates(Z, qs=(4,)).items()}); return out


def iaaft(x, rng, n_iter=200):
    n = len(x); mu = x.mean(); amp = np.abs(np.fft.rfft(x - mu)); xs = np.sort(x); y = rng.permutation(x)
    for _ in range(n_iter):
        Y = np.fft.rfft(y - y.mean())
        z = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n) + mu
        y_new = xs[np.argsort(np.argsort(z))]
        if np.array_equal(y_new, y):
            break
        y = y_new
    return y


def reversible_surrogate(close, volume, rng):
    r = np.diff(np.log(close)); a = np.abs(r); lv = np.log(volume[1:])
    at = iaaft(a, rng)
    A = np.fft.rfft(a - a.mean()); Vf = np.fft.rfft(lv - lv.mean()); At = np.fft.rfft(at - at.mean())
    s = np.sign(np.real(A * np.conj(Vf))); s[s == 0] = 1.0
    vlin = np.fft.irfft(np.abs(Vf) * s * np.exp(1j * np.angle(At)), len(lv))
    vt = np.sort(lv)[np.argsort(np.argsort(vlin))]
    sign = np.where(rng.random(len(r)) < np.mean(r > 0), 1.0, -1.0)
    c = close[0] * np.exp(np.r_[0.0, np.cumsum(sign * at)])
    return c, np.r_[volume[0], np.exp(vt)]


def literal_surrogate(close, volume, rng):
    r = np.diff(np.log(close)); rt = iaaft(r, rng); lv = iaaft(np.log(volume), rng)
    return close[0] * np.exp(np.r_[0.0, np.cumsum(rt)]), np.exp(lv)


def reversed_returns(close, volume):
    r = np.diff(np.log(close))[::-1]
    return close[0] * np.exp(np.r_[0.0, np.cumsum(r)]), volume[::-1]


def prop2_prediction(close, volume, ann):
    """Fit lambda so that the 20-day trailing mean of a unit AR(1) reproduces the ACF of log RV20 at lags 1..60."""
    F = common7(close, volume, ann).dropna(); x = F.log_rv20.to_numpy(); x = x - x.mean()
    acf = np.array([np.dot(x[k:], x[:len(x) - k]) / np.dot(x, x) for k in range(1, 61)])
    def model_acf(lam):
        phi = np.exp(-lam); c0 = cov_discrete(phi, [20], 0)[0, 0]
        return np.array([cov_discrete(phi, [20], k)[0, 0] / c0 for k in range(1, 61)])
    grid = np.exp(np.linspace(np.log(1e-3), np.log(1.0), 120))
    sse = [np.sum((model_acf(l) - acf) ** 2) for l in grid]; lam = grid[int(np.argmin(sse))]
    fine = np.exp(np.linspace(np.log(lam / 1.1), np.log(lam * 1.1), 41))
    sse_f = [np.sum((model_acf(l) - acf) ** 2) for l in fine]; lam = fine[int(np.argmin(sse_f))]
    phi = np.exp(-lam); ws = [5, 20, 60, 250]
    C0 = cov_discrete(phi, ws, 0); C1 = cov_discrete(phi, ws, 1)
    S = np.diag(1 / np.sqrt(np.diag(C0))); R0 = S @ C0 @ S; R1 = S @ C1 @ S
    ev, V = np.linalg.eigh(R0); V = V[:, ::-1][:, :2]
    Z0 = V.T @ R0 @ V; Z1 = V.T @ R1 @ V; Sz = np.diag(1 / np.sqrt(np.diag(Z0)))
    mp = markov_projection(Sz @ Z0 @ Sz, Sz @ Z1 @ Sz)
    return dict(lam=lam, corr_time_days=1 / lam, sse=float(min(sse_f)), ratio=mp['ratio_lyap'], ep=mp['ep_lyap'], d1=mp['d1'])


def unit(name, variant, b):
    f = CACHE / f'E4_{name}_{variant}_{b}.pkl'
    if f.exists():
        return pd.read_pickle(f)
    if time.time() - T0 > BUDGET:
        return None
    close, volume, ann = market_inputs(name); rows = []
    if variant == 'deterministic':
        for v, (c, vol, kind) in {'observed': (close, volume, 'trailing'), 'reversed_returns': (*reversed_returns(close, volume), 'trailing'),
                                  'reversed_price_path': (close[::-1], volume[::-1], 'trailing'),
                                  'centered': (close, volume, 'centered')}.items():
            rows.append(dict(market=name, variant=v, sid=-1, **stats(state(c, vol, ann, kind))))
        if name == 'SPX':
            rows.append(dict(market=name, variant='observed_2000_2020', sid=-1, **stats(state(close[:5105], volume[:5105], ann))))
            p = prop2_prediction(close, volume, ann)
            rows.append(dict(market=name, variant='prop2_prediction', sid=-1, ratio=p['ratio'], ep=p['ep'], d1=p['d1']))
            pd.DataFrame([p]).to_csv(ROOT / 'tables_pre' / 'E4_lambda_fit.csv', index=False)
    else:
        n = N_SUR if variant == 'reversible_surrogate' else N_SUR2
        for sid in range(b * BATCH, min((b + 1) * BATCH, n)):
            rng = np.random.default_rng([SEED, ['SPX', 'BTC', 'ETH'].index(name), int(variant == 'literal_iaaft'), sid])
            c, vol = (reversible_surrogate if variant == 'reversible_surrogate' else literal_surrogate)(close, volume, rng)
            rows.append(dict(market=name, variant=variant, sid=sid, **stats(state(c, vol, ann))))
    df = pd.DataFrame(rows); pd.to_pickle(df, f); print('unit', name, variant, b, f'{time.time() - T0:.0f}s', flush=True)
    return df


def main():
    plan = [(m, 'deterministic', 0) for m in ['SPX', 'BTC', 'ETH']]
    plan += [(m, 'reversible_surrogate', b) for m in ['SPX', 'BTC', 'ETH'] for b in range(N_SUR // BATCH)]
    plan += [('SPX', 'literal_iaaft', b) for b in range(N_SUR2 // BATCH)]
    parts = [unit(*u) for u in plan]
    if any(p is None for p in parts):
        print('budget reached; rerun to continue'); return
    D = pd.concat(parts, ignore_index=True)
    D.to_csv(ROOT / 'tables_pre' / 'E4_irreversibility_surrogates.csv', index=False, float_format='%.8g')
    rows = []
    for m, g in D.groupby('market'):
        obs = g[g.variant == 'observed'].iloc[0]
        for stat in ['ratio', 'ep', 'd1', 'kld_q4_D2', 'kld_q4_D3mD2']:
            for v in ['reversible_surrogate', 'literal_iaaft']:
                s = g[g.variant == v][stat].dropna()
                if len(s) == 0:
                    continue
                lo, med, hi = np.quantile(s, [0.05, 0.5, 0.95])
                rows.append(dict(market=m, statistic=stat, null=v, observed=obs[stat], q05=lo, median=med, q95=hi,
                                 percentile=float(np.mean(s <= obs[stat])), inside_band=bool(lo <= obs[stat] <= hi),
                                 reversed_returns=g[g.variant == 'reversed_returns'][stat].iloc[0],
                                 centered=g[g.variant == 'centered'][stat].iloc[0], n_null=len(s)))
    S = pd.DataFrame(rows); S.to_csv(ROOT / 'tables_pre' / 'E4_summary.csv', index=False, float_format='%.6g')
    print(S[S.null == 'reversible_surrogate'].round(4).to_string(index=False)); print('E4 complete'); (CACHE / 'DONE_E4_surrogates').touch()


if __name__ == '__main__':
    main()
