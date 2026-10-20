"""Figures 1-4 of Paper 1 (PRE).  Inputs: tables_pre/*.csv and data/spx_2000_2026_r1.csv only.
Outputs: figures_p1/fig{1..4}_*.pdf and .png (double-column width 17.8 cm, Okabe-Ito colours, >= 7 pt text).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'tables_pre'; OUT = ROOT / 'figures_p1'; OUT.mkdir(exist_ok=True)
OK = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442',
          blue='#0072B2', red='#D55E00', purple='#CC79A7', grey='#7f7f7f')
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'lines.linewidth': 1.2})
W2 = 17.8 / 2.54


def label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=8, va='bottom')


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight'); fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def kernels(w1, w2, kind):
    def k(w):
        if kind == 'trailing':
            return np.arange(w), np.full(w, 1 / w)
        h = (w - 1) // 2
        return np.arange(-h, h + 1), np.full(w, 1 / w)
    (a1, k1), (a2, k2) = k(w1), k(w2)
    ms = np.arange(-40, 41); K = np.zeros(len(ms))
    for i, m in enumerate(ms):        # K(m) = sum_a k1(a) k2(a+m); lag a means x_{t-a}
        K[i] = sum(k1[ia] * k2[np.where(a2 == a1[ia] + m)[0][0]] for ia in range(len(a1)) if (a1[ia] + m) in a2)
    Kr = K[::-1]
    return ms, 0.5 * (K + Kr), 0.5 * (K - Kr)


def fig1():
    d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026_r1.csv')
    r = np.diff(np.log(d.close.to_numpy(float))); r = r - r.mean(); u = r ** 2; u = u - u.mean()
    n = len(r); L = 20; taus = np.arange(-L, L + 1)
    xc = lambda p, q, t: (np.dot(p[t:], q[:n - t]) if t >= 0 else np.dot(p[:n + t], q[-t:])) / n
    guu = np.array([xc(u, u, t) for t in taus]) / xc(u, u, 0)
    gru = np.array([xc(r, u, t) for t in taus]) / np.sqrt(xc(r, r, 0) * xc(u, u, 0))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0))
    for a, (kind, w2, title) in zip(ax[:2], [('trailing', 20, 'trailing windows (5, 20)'),
                                              ('centered', 21, 'centered windows (5, 21)')]):
        ms, Kp, Km = kernels(5, w2, kind)
        a.bar(ms - 0.2, Kp, width=0.4, color=OK['blue'], label=r'$K^+(m)$ (even)')
        a.bar(ms + 0.2, Km, width=0.4, color=OK['red'], label=r'$K^-(m)$ (odd)')
        a.axhline(0, color='k', lw=0.5); a.set_xlim(-25, 25); a.set_xlabel('$m$ (days)'); a.set_title(title, fontsize=7.5)
    ax[0].set_ylabel('kernel cross-correlation')
    h, l = ax[0].get_legend_handles_labels(); ax[1].legend(h, l, frameon=False, loc='upper left')
    nz = taus != 0
    ax[2].plot(taus[nz], guu[nz], 'o-', ms=2.5, color=OK['blue'], label=r'$\gamma_{uu}(\tau)$, $u=r^2$ (even)')
    ax[2].plot(taus, gru, 's-', ms=2.5, color=OK['red'], label=r'$\gamma_{ru}(\tau)$ (leverage: odd part)')
    ax[2].axhline(0, color='k', lw=0.5); ax[2].axvline(0, color='k', lw=0.5)
    ax[2].set_xlabel(r'$\tau$ (days)'); ax[2].set_ylabel('correlation')
    ax[2].set_ylim(-0.14, 0.62); ax[2].legend(frameon=False, loc='upper center', ncol=1, fontsize=6)
    ax[2].set_title('S&P 500 inputs', fontsize=7.5)
    ax[2].text(-19.5, -0.135, r'$\gamma_{uu}(0)=1$ not shown', fontsize=5.8, color=OK['blue'])
    for a, s in zip(ax, 'abc'):
        label(a, f'({s})')
    fig.tight_layout(w_pad=1.2); save(fig, 'fig1_decomposition')


def fig2():
    S = pd.read_csv(T / 'S1_gaussian_leak.csv'); S = S[(S.method == 'bayes_observer') & (S.h == 10)]
    C = pd.read_csv(T / 'T2c_observer_curves.csv')
    G = pd.read_csv(T / 'S2_gaussian_irrev.csv'); G = G[(G.kind == 'trailing') & (np.isclose(G.phi, 0.98))]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1))
    cols = [OK['red'], OK['orange'], OK['blue'], OK['green']]
    for phi, col in zip([0.9, 0.95, 0.98, 0.99], cols):
        g = S[np.isclose(S.phi, phi)].sort_values('lam_w')
        ax[0].plot(g.lam_w, g.leak_theory / g.var_pop, '-', color=col, label=fr'$\phi={phi}$')
        ax[0].plot(g.lam_w, g.leak_direct / g.var_pop, 'o', ms=3, mfc='none', color=col)
    ax[0].set_xscale('log'); ax[0].set_xlabel(r'$\lambda w$'); ax[0].set_ylabel(r'normalized loss $L/\mathrm{Var}\,p_{\rm op}$')
    ax[0].legend(frameon=False, loc='upper left')
    styles = [('-', OK['black'], r'$\eta^2=0$'), ('--', OK['blue'], r'$10^{-3}$'), ('-.', OK['green'], r'$10^{-2}$'),
              (':', OK['red'], r'$10^{-1}$')]
    for (ls, col, lab), fr in zip(styles, [0.0, 1e-3, 1e-2, 1e-1]):
        g = C[np.isclose(C.eta2_frac, fr)].sort_values('lam_w')
        ax[1].plot(g.lam_w, -g.omega_C1, ls, color=col, label=lab)
        ax[2].plot(g.lam_w, g.ep, ls, color=col, label=lab)
    ax[2].plot(G.lam_w, G.ep_sim, 'o', ms=3, mfc='none', color=OK['black'], label='simulation')
    ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\lambda w$'); ax[1].set_ylabel(r'antisymmetry $-\omega$')
    ax[1].legend(frameon=False, title='noise', title_fontsize=6.5)
    ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel(r'$\lambda w$')
    ax[2].set_ylabel(r'$\dot e_p$ of fitted OU (per step)'); ax[2].legend(frameon=False, fontsize=6)
    for a, s in zip(ax, 'abc'):
        label(a, f'({s})')
    fig.tight_layout(w_pad=1.2); save(fig, 'fig2_gaussian_observer')


def fig3():
    A = pd.read_csv(T / 'S4d_acf.csv'); E = pd.read_csv(T / 'S4e_two_rate_leak.csv')
    C = pd.read_csv(T / 'S5_summary.csv')
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1))
    ax[0].semilogy(A.lag_time, A.acf, '.', ms=2, color=OK['grey'], label='simulation')
    ax[0].semilogy(A.lag_time, A.fit, '-', color=OK['black'], label='two-exponential fit')
    k = A.kramers.iloc[0]; sw = 1 - A.fast_weight.iloc[0]
    ax[0].semilogy(A.lag_time, sw * np.exp(-k * A.lag_time), '--', color=OK['red'], label=fr'Kramers, $\kappa={k:.3f}$')
    ax[0].semilogy(A.lag_time, np.exp(-2 * A.lag_time), ':', color=OK['blue'], label=r'curvature, $\lambda=2$')
    ax[0].set_ylim(1e-2, 1.05); ax[0].set_xlim(0, 30); ax[0].set_xlabel('lag (time units)'); ax[0].set_ylabel('autocorrelation of $x$')
    ax[0].legend(frameon=False, loc='upper right')
    cols = {np.inf: OK['black'], 80.0: OK['blue'], 20.0: OK['orange'], 5.0: OK['red']}
    mk = {np.inf: 'o', 80.0: 's', 20.0: '^', 5.0: 'v'}
    for tau, g in E[E.w > 1].groupby('tau_c'):
        g = g.sort_values('lam_w'); lab = r'$\tau_c=\infty$' if np.isinf(tau) else fr'$\tau_c={tau:g}$'
        ax[1].plot(g.lam_w, g.ell_exact, mk[tau], ms=3.5, color=cols[tau], label=lab)
        ax[1].plot(g.lam_w, g.ell_two_rate, '-', color=cols[tau], lw=1)
    g = E[(E.w > 1) & np.isinf(E.tau_c)].sort_values('lam_w')
    ax[1].plot(g.lam_w, g.ell_local_gaussian, '--', color=OK['grey'], label='single rate')
    ax[1].set_xscale('log'); ax[1].set_xlabel(r'$\bar\lambda w$'); ax[1].set_ylabel(r'normalized loss $\ell$')
    ax[1].legend(frameon=False, loc='upper left', fontsize=6)
    for tau, g in C.groupby('tau_c'):
        tau = float(tau); g = g.sort_values('lam_w')
        ax[2].plot(g.lam_w, g.obs_ep, mk[tau], ms=3.5, color=cols[tau])
        ax[2].plot(g.lam_w, g.lat_ep, ':', color=cols[tau], lw=1)
    g = C[C.tau_c.astype(float) == np.inf].sort_values('lam_w')
    ax[2].plot(g.lam_w, g.pred_ep, '-', color=OK['grey'], label='Gaussian observer')
    ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel(r'$\bar\lambda w$')
    ax[2].set_ylabel(r'$\dot e_p$ of fitted OU (per step)'); ax[2].legend(frameon=False, loc='center left')
    ax[2].text(0.12, 0.004, 'latent (x, c), finite $\\tau_c$', fontsize=6, color=OK['grey'])
    for a, s in zip(ax, 'abc'):
        label(a, f'({s})')
    fig.tight_layout(w_pad=1.2); save(fig, 'fig3_double_well')


def fig4():
    Cc = pd.read_csv(T / 'E4c_pca_factorization.csv'); P = pd.read_csv(T / 'E4e_pairs.csv')
    Bd = pd.read_csv(T / 'E4e_surrogate_bands.csv'); Sy = pd.read_csv(T / 'E4e_symmetric_sign.csv')
    V = pd.read_csv(T / 'E4f_sv_variants.csv')
    win, lev = 'window(log_rv5,log_rv20)', 'leverage(ret20,log_rv20)'
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))

    def rows_for(col_pair, pca=False):
        key = 'PCA2_common_basis' if pca else col_pair
        sp = P[(P['sample'] == 'SPX 2000-2026') & (P.channel == key)].iloc[0]
        surr = Cc[Cc.state == 'reversible_surrogate']['omega_common' if pca else f'omega_{col_pair}']
        sym = Sy['omega_PCA2_common_basis' if pca else f'omega_{col_pair}']
        out = [('S&P data', sp.omega_fwd, None), ('S&P reversed', sp.omega_rev, None),
               ('S&P centered', sp.omega_cen, None), ('rev. surrogates', surr.median(), surr.quantile([.05, .95])),
               ('sym.-sign surr.', sym.median(), sym.quantile([.05, .95]))]
        if not pca:
            sv = V[V.rho == 0.0]['om_win' if col_pair == win else 'om_lev']
            out.append(('SV, no leverage', sv.median(), sv.quantile([.05, .95])))
        for m in ['BTC', 'ETH']:
            pm = P[(P['sample'] == m) & (P.channel == key)].iloc[0]
            bm = Bd[Bd['sample'] == m]['omega_PCA2_common_basis' if pca else f'omega_{col_pair}']
            out += [(f'{m} data', pm.omega_fwd, None), (f'{m} surrogates', bm.median(), bm.quantile([.05, .95]))]
        return out

    for a, (title, rows) in zip(ax[:3], [('(a) window channel', rows_for(win)), ('(b) leverage channel', rows_for(lev)),
                                          ('(c) PCA(2) state, fixed basis', rows_for(None, pca=True))]):
        for i, (name, val, band) in enumerate(rows):
            y = len(rows) - i
            col = OK['red'] if 'data' in name or 'reversed' in name or 'centered' in name else OK['blue']
            if 'reversed' in name: col = OK['orange']
            if 'centered' in name: col = OK['green']
            if band is not None:
                a.plot([band.iloc[0], band.iloc[1]], [y, y], '-', color=OK['grey'], lw=2.5, alpha=0.6)
            a.plot(val, y, 'o', ms=4, color=col)
        a.axvline(0, color='k', lw=0.5); a.set_yticks(range(len(rows), 0, -1))
        a.set_yticklabels([r[0] for r in rows] if a is ax[0] else [r[0] for r in rows], fontsize=6)
        a.set_title(title, fontsize=7.5, loc='left'); a.set_xlabel(r'antisymmetry $\omega$')
    names = {'G1': 'Gaussian, 1 factor', 'T1': r'$t_5$, 1 factor', 'G2': 'Gaussian, 2 factors', 'T2': r'$t_5$, 2 factors'}
    for vi, (name, col) in enumerate([('G1', OK['black']), ('T1', OK['blue']), ('G2', OK['orange']), ('T2', OK['purple'])]):
        g = V[V.variant == name].groupby('rho').ep
        med, lo, hi = g.median(), g.quantile(0.05), g.quantile(0.95)
        ax[3].plot(-med.index, med.values, 'o-', ms=3, color=col, label=names[name])
        if name == 'G1':
            ax[3].fill_between(-med.index, lo.values, hi.values, color=col, alpha=0.12, lw=0)
    obs = Cc[Cc.state == 'observed'].ep.iloc[0]; band = Cc[Cc.state == 'reversible_surrogate'].ep.quantile(0.95)
    ax[3].axhline(obs, ls='--', color=OK['red'], lw=1); ax[3].text(0.02, obs * 1.06, 'S&P data', color=OK['red'], fontsize=6)
    ax[3].axhspan(0, band, color=OK['grey'], alpha=0.2, lw=0)
    ax[3].text(0.97, band * 0.45, 'reversible band', fontsize=6, color='k', ha='right')
    ax[3].set_xlabel(r'leverage strength $-\rho$'); ax[3].set_ylabel(r'$\dot e_p$ of PCA(2) state (per day)')
    ax[3].set_title('(d) stochastic-volatility model', fontsize=7.5, loc='left')
    ax[3].legend(frameon=False, loc='upper left', fontsize=5.8)
    fig.tight_layout(w_pad=0.6); save(fig, 'fig4_markets')


if __name__ == '__main__':
    for f in sys.argv[1:] or ['1', '2', '3', '4']:
        {'1': fig1, '2': fig2, '3': fig3, '4': fig4}[f]()
    print('figures written to', OUT)
