"""P1_numbers -- result numbers quoted in the text of Paper 1 as LaTeX macros (../manuscript/numbers.tex), each with
its source table and selection (tables_pre/P1_numbers_sources.csv).  Reads tables_pre/ only; no randomness."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; T = ROOT / 'tables_pre'; MS = ROOT.parent / 'manuscript'
WIN, LEV, DD, PCA = 'window(log_rv5,log_rv20)', 'leverage(ret20,log_rv20)', 'leverage(dd20,log_rv20)', 'PCA2_common_basis'
rows = []


def add(name, value, nd, source, sel, sign=False, pct=False, sci=False, int_=False):
    if sci:
        m, e = f'{value:.{nd}e}'.split('e'); txt = f'{m}\\times10^{{{int(e)}}}'
    elif int_:
        txt = f'{int(round(value)):d}'
    else:
        txt = f'{value:+.{nd}f}' if sign else f'{value:.{nd}f}'
    if pct:
        txt += '\\%'
    rows.append(dict(macro=name, latex=f'\\ensuremath{{{txt}}}', value=value, source=source, selection=sel))


def main():
    S = pd.read_csv(T / 'E4_irreversibility_surrogates.csv'); g = S[(S.market == 'SPX') & (S.variant == 'reversible_surrogate')].ep
    obs_ep = S[(S.market == 'SPX') & (S.variant == 'observed')].ep.iloc[0]; src = 'E4_irreversibility_surrogates.csv'
    add('EpPca', obs_ep, 3, src, 'SPX observed PCA(2) e_p'); add('EpSurMed', g.median(), 4, src, 'reversible surrogates, median')
    add('EpSurQ', g.quantile(.95), 4, src, 'reversible surrogates, 95% quantile'); add('EpSurMax', g.max(), 4, src, 'reversible surrogates, max')
    add('NSur', len(g), 0, src, 'number of reversible surrogates', int_=True); add('EpObsOverMax', obs_ep / g.max(), 1, src, 'observed / max')
    C = pd.read_csv(T / 'E4c_pca_factorization.csv'); o = C[C.state == 'observed'].iloc[0]; rv = C[C.state == 'reversed_returns'].iloc[0]
    su = C[C.state == 'reversible_surrogate']; src = 'E4c_pca_factorization.csv'
    add('EpWinObs', o[f'ep_{WIN}'], 2, src, 'observed, window pair e_p'); add('EpWinSurMed', su[f'ep_{WIN}'].median(), 2, src, 'surrogates, window pair e_p median')
    add('OmFitSqRatio', o.abs_omega ** 2 / np.median(su.abs_omega ** 2), 1, src, 'omega_fit^2 / surrogate median')
    add('GRatio', o.G / su.G.median(), 1, src, 'G / surrogate median'); add('GPca', o.G, 0, src, 'observed PCA(2) G', int_=True)
    add('OmFitPca', o.abs_omega, 3, src, 'observed |omega_fit|'); add('EpRevRefit', rv.ep, 3, src, 'reversed input, re-fitted PCA, e_p')
    add('OmFitRevRefit', rv.abs_omega, 3, src, 'reversed, re-fitted, |omega_fit|'); add('GRevRefit', rv.G, 0, src, 'reversed, re-fitted, G', int_=True)
    add('EpRevFixed', rv.ep_common, 3, src, 'reversed, fixed basis, e_p'); add('OmFitRevFixed', abs(rv.omega_common_fit), 3, src, 'reversed, fixed basis, |omega_fit|')
    P = pd.read_csv(T / 'E4e_pairs.csv'); src = 'E4e_pairs.csv'
    q = lambda s, ch: P[(P['sample'] == s) & (P.channel == ch)].iloc[0]; sp = 'SPX 2000-2026'
    add('OmWinFwd', q(sp, WIN).omega_fwd, 3, src, 'window, forward'); add('OmWinRev', q(sp, WIN).omega_rev, 3, src, 'window, reversed')
    add('OmLevFwd', q(sp, LEV).omega_fwd, 3, src, 'leverage, forward', sign=True); add('OmLevRev', q(sp, LEV).omega_rev, 3, src, 'leverage, reversed', sign=True)
    add('OmLevCen', q(sp, LEV).omega_cen, 3, src, 'leverage, centred', sign=True); add('OmLevData', q(sp, LEV).omega_fwd, 4, src, 'leverage, forward (4 d.p.)', sign=True)
    add('DDO', q(sp, DD).O, 4, src, 'drawdown O', sign=True); add('DDP', q(sp, DD).P, 4, src, 'drawdown P', sign=True)
    add('DDCen', q(sp, DD).omega_cen, 4, src, 'drawdown centred', sign=True)
    add('OmPcaFwd', q(sp, PCA).omega_fwd, 3, src, 'PCA(2) fwd', sign=True); add('OmPcaRev', q(sp, PCA).omega_rev, 3, src, 'PCA(2) rev', sign=True)
    add('OmPcaCen', q(sp, PCA).omega_cen, 3, src, 'PCA(2) cen', sign=True); add('OPcaRev', q(sp, PCA).O, 4, src, 'PCA(2) O by time reversal', sign=True)
    add('OPcaCen', q(sp, PCA).omega_fwd - q(sp, PCA).omega_cen, 4, src, 'PCA(2) omega - omega_cen', sign=True)
    cr = P[P['sample'].isin(['BTC', 'ETH']) & P.channel.isin([WIN, LEV, PCA])]; add('CryptoPmax', cr.P.abs().max(), 3, src, 'max |P|, BTC and ETH rows')
    E = pd.read_csv(T / 'E4g_bootstrap_se.csv'); src = 'E4g_bootstrap_se.csv'
    e = lambda s, ch, L=20: E[(E['sample'] == s) & (E.channel == ch) & (E.block == L)].iloc[0]
    add('SELevP', e(sp, LEV).se_P, 4, src, 'SE of P, leverage, block 20'); add('LevPz', q(sp, LEV).P / e(sp, LEV).se_P, 0, src, 'P / SE, leverage', int_=True)
    add('PcaPz', q(sp, PCA).P / e(sp, PCA).se_P, 0, src, 'P / SE, PCA(2)', int_=True)
    add('WinOz', abs(q(sp, WIN).O) / e(sp, WIN).se_O, 0, src, '|O| / SE, window', int_=True)
    cols = ['se_fwd', 'se_rev', 'se_cen', 'se_O', 'se_P']; a = E[E.block == 60].set_index(['sample', 'channel'])[cols]
    b = E[E.block == 20].set_index(['sample', 'channel'])[cols]; ratio = (a / b).to_numpy()
    add('SERatioLo', ratio.min(), 1, src, 'min SE(60)/SE(20)'); add('SERatioHi', ratio.max(), 1, src, 'max SE(60)/SE(20)')
    btc = [abs(q(m, LEV).P) / e(m, LEV).se_P for m in ['BTC', 'ETH']]; add('CryptoLevPz', max(btc), 1, src, 'max |P|/SE leverage BTC, ETH')
    Ld = pd.read_csv(T / 'E4g_pca_loadings.csv', index_col=0)
    add('DDLoadOne', Ld.loc['dd20', 'PC1'], 2, 'E4g_pca_loadings.csv', 'dd20 loading on PC1', sign=True)
    add('DDLoadTwo', Ld.loc['dd20', 'PC2'], 2, 'E4g_pca_loadings.csv', 'dd20 loading on PC2', sign=True)
    Vs = pd.read_csv(T / 'E4g_volume_sensitivity.csv'); add('VolMaxDiff', Vs['diff'].abs().max(), 1, 'E4g_volume_sensitivity.csv', 'max |change|', sci=True)
    add('VolAdj', Vs.adjacent_volume.iloc[0], 3, 'E4g_volume_sensitivity.csv', 'adjacent-day mean volume', sci=True)
    D = pd.read_csv(T / 'T2d_table1.csv'); r0 = D.iloc[0]; src = 'T2d_table1.csv'
    add('OmTabOne', r0.omega, 4, src, 'lam w = 0.1, omega'); add('OmFitTabOne', r0.omega_fit, 4, src, 'lam w = 0.1, omega_fit')
    add('GTabOne', r0.G, 1, src, 'lam w = 0.1, G', sci=True); add('EpTabOne', r0.ep, 1, src, 'lam w = 0.1, e_p')
    add('EpNoiseOne', r0['ep_eta2_p01'], 2, src, 'eta^2 = 0.01, e_p'); add('EpNoiseTen', r0['ep_eta2_p10'], 3, src, 'eta^2 = 0.1, e_p')
    add('OmNoiseOne', 100 * (1 - r0['omega_eta2_p01'] / r0.omega), 0, src, 'percent change of omega, eta^2 = 0.01', pct=True)
    add('OmNoiseTen', 100 * (1 - r0['omega_eta2_p10'] / r0.omega), 0, src, 'percent change of omega, eta^2 = 0.1', pct=True)
    dev = 100 * (1 - D.omega_fit / D.omega); add('OmFitDevLo', dev.min(), 0, src, 'min % omega_fit below omega', int_=True)
    add('OmFitDevHi', dev.max(), 0, src, 'max % omega_fit below omega', int_=True); add('EllTabOne', r0.ell_h10, 2, src, 'lam w = 0.1, ell (h = 10)')
    G2 = pd.read_csv(T / 'S2_gaussian_irrev.csv'); G2 = G2[G2.kind.str.contains('trailing|current')].dropna(subset=['ratio_sim'])
    G2 = G2[G2.replicate.isna()] if 'replicate' in G2 else G2
    err = max((G2[f'{c}_sim'] / G2[f'{c}_closed'] - 1).abs().max() for c in ['ratio', 'ep', 'd1'])
    add('SimMaxErr', 100 * err, 1, 'S2_gaussian_irrev.csv', 'max rel. error of R, e_p, d1 (non-centred)', pct=True)
    L1 = pd.read_csv(T / 'S1_gaussian_leak.csv'); L1 = L1[L1.method == 'bayes_observer']
    dv = (L1.leak_direct - L1.leak_theory) / L1.var_pop
    add('MCDevMax', 100 * dv[L1.phi <= 0.98].abs().max(), 1, 'S1_gaussian_leak.csv', 'max |MC - series|/Var p, phi <= 0.98', pct=True)
    add('MCDevNineNine', 100 * (-dv[L1.phi == 0.99]).max(), 0, 'S1_gaussian_leak.csv', 'max (series - MC)/Var p, phi = 0.99', pct=True)
    W = pd.read_csv(T / 'S4e_two_rate_leak.csv'); W = W[W.w > 1]; src = 'S4e_two_rate_leak.csv'
    se_ = W.ell_local_gaussian - W.ell_exact; te = W.ell_two_rate - W.ell_exact; fac = W.ell_local_gaussian / W.ell_exact
    add('SingleErrLo', se_.min(), 2, src, 'single-rate error, min', sign=True); add('SingleErrHi', se_.max(), 2, src, 'single-rate error, max', sign=True)
    add('TwoErrLo', te.min(), 2, src, 'two-rate error, min', sign=True); add('TwoErrHi', te.max(), 2, src, 'two-rate error, max', sign=True)
    add('TwoErrAbs', te.abs().max(), 2, src, 'two-rate |error|, max'); add('SingleFacLo', fac.min(), 1, src, 'single/exact, min'); add('SingleFacHi', fac.max(), 1, src, 'single/exact, max')
    M4 = pd.read_csv(T / 'S4_summary.csv'); M4['tau_c'] = M4.tau_c.astype(float); W['tau_c'] = W.tau_c.astype(float)
    add('ClfEst', M4[(M4.tau_c == 5) & (M4.w == 80)].ell_meas.iloc[0], 2, 'S4_summary.csv', 'tau_c = 5, w = 80, classifier-skill estimate')
    add('ClfExact', W[(W.tau_c == 5) & (W.w == 80)].ell_exact.iloc[0], 2, src, 'tau_c = 5, w = 80, exact')
    S5 = pd.read_csv(T / 'S5_summary.csv'); S5['tau_c'] = S5.tau_c.astype(float); inf = S5[np.isinf(S5.tau_c)]; fin = S5[~np.isinf(S5.tau_c)]
    src = 'S5_summary.csv'
    add('DWEpOne', inf[inf.w == 1].obs_ep.iloc[0], 1, src, 'tau_c = inf, w = 1, observed e_p'); add('DWEpEighty', inf[inf.w == 80].obs_ep.iloc[0], 1, src, 'tau_c = inf, w = 80')
    add('DWLatLo', 1e3 * fin.lat_ep.min(), 1, src, 'latent e_p min x 1e3'); add('DWLatHi', 1e3 * fin.lat_ep.max(), 1, src, 'latent e_p max x 1e3')
    add('DWPredOne', inf[inf.w == 1].pred_ep.iloc[0], 1, src, 'Gaussian-observer prediction, w = 1')
    V = pd.read_csv(T / 'E4f_sv_variants.csv'); src = 'E4f_sv_variants.csv'; target = q(sp, LEV).omega_fwd; match, tmax = [], None
    add('SVQnfMax', max(V[(V.variant == v) & (V.rho == 0)].ep.quantile(.95) for v in V.variant.unique()), 3, src, 'max over variants of 95% e_p at rho = 0')
    for v, gv in V.groupby('variant'):
        m = gv.groupby('rho')[['ep', 'om_lev']].median().sort_values('om_lev')
        if m.om_lev.max() >= target:
            match.append(float(np.interp(target, m.om_lev, m.ep)))
        else:
            tmax = (v, m.om_lev.max(), m.ep[m.om_lev.idxmax()])
    add('SVMatchLo', min(match), 3, src, 'e_p interpolated to the data omega_lev, min over variants')
    add('SVMatchHi', max(match), 3, src, 'e_p interpolated to the data omega_lev, max over variants')
    if tmax:
        add('TOneLevMax', tmax[1], 3, src, f'{tmax[0]}: largest omega_lev', sign=True); add('TOneEpAtMax', tmax[2], 3, src, f'{tmax[0]}: e_p there')
    R3 = pd.read_csv(T / 'E4e_third_order.csv').iloc[0]; src = 'E4e_third_order.csv'
    add('RRobs', R3.observed, 2, src, 'observed', sign=True); add('RRlo', R3.null_q025, 2, src, 'null 2.5%'); add('RRhi', R3.null_q975, 2, src, 'null 97.5%')
    add('RRp', R3.p_two_sided, 2, src, 'two-sided p')
    I9 = pd.read_csv(T / 'E9_market_information.csv').set_index(['observer', 'method']); src = 'E9_market_information.csv'
    ex = lambda o: I9.loc[(o, 'exact (Lemma 1), noise-free windows of x')].ell; sm = lambda o: I9.loc[(o, 'simulation, out-of-sample regression')].ell
    add('LamMkt', I9.lam.iloc[0], 3, src, 'lambda per day'); add('EllMktPair', ex('windows (5,20)'), 2, src, 'exact, pair (5,20)')
    for o, nm in [('window 5', 'EllWinFive'), ('window 20', 'EllWinTwenty'), ('window 60', 'EllWinSixty'), ('window 250', 'EllWinTwoFifty')]:
        add(nm, ex(o), 2, src, f'exact, {o}')
    add('EllMktRV', sm('log RV (5, 20 days)'), 2, src, 'simulation, log RV pair'); add('EllMktPca', sm('PCA(2) state'), 2, src, 'simulation, PCA(2) state')
    add('EllCheckDiff', abs(sm('windows (5,20)') - ex('windows (5,20)')), 3, src, '|simulated - exact|, noise-free pair')
    R = pd.DataFrame(rows); R.to_csv(T / 'P1_numbers_sources.csv', index=False)
    (MS / 'numbers.tex').write_text('% generated by reproducibility_pre/scripts/P1_numbers.py -- do not edit\n' +
                                    ''.join(f'\\newcommand{{\\{r.macro}}}{{{r.latex}\\xspace}}\n' for r in R.itertuples()))
    print(R[['macro', 'latex']].to_string(index=False))


if __name__ == '__main__':
    main()
