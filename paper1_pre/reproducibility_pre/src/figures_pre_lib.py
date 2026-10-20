"""Figure library for the PRE manuscript (one public function per figure; thin wrappers scripts/fig*.py call them).
Okabe-Ito palette, sizes 8.6 cm (single) / 17.8 cm (double), fonts >= 7 pt, vector PDF + PNG."""
import shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
PRE = Path(__file__).resolve().parents[1]; T = PRE / 'tables_pre'; FIG = PRE / 'figures_pre'; LEG = PRE.parent / 'reproducibility'
OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
W1, W2 = 8.6 / 2.54, 17.8 / 2.54
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'lines.linewidth': 1.1,
                     'savefig.dpi': 300, 'figure.dpi': 150, 'pdf.fonttype': 42})
TAUCOL = {5.0: OI[6], 20.0: OI[1], 80.0: OI[5], np.inf: OI[0]}
TAULAB = {5.0: r'$\tau_c=5$', 20.0: r'$\tau_c=20$', 80.0: r'$\tau_c=80$', np.inf: r'$\tau_c=\infty$'}
PHICOL = {0.9: OI[6], 0.95: OI[1], 0.98: OI[5], 0.99: OI[3]}


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); fig.savefig(FIG / f'{name}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def lab(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='right')


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(W2, 2.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis('off')
    boxes = [(1, 'latent Markov state\n$Z_t=(x_t,c_t)$'), (21, 'trailing windows\n$y_t=\\sum_a k(a)\\,x_{t-a}$'),
             (41, 'fitted Markov model\n(OU, MSM, AMC)'), (61, 'hitting score\n$S=v_h^B(y_t)$'), (81, 'calibration\n$g(S)\\to$ forecast')]
    for x0, t in boxes:
        ax.add_patch(FancyBboxPatch((x0, 11), 17, 8, boxstyle='round,pad=0.4', fc='#f2f2f2', ec='k', lw=0.8))
        ax.text(x0 + 8.5, 15, t, ha='center', va='center', fontsize=7)
    for x0 in [18.6, 38.6, 58.6, 78.6]:
        ax.annotate('', xy=(x0 + 2.6, 15), xytext=(x0, 15), arrowprops=dict(arrowstyle='->', lw=0.9))
    notes = [(9.5, 26, 'state-information ceiling\n$\\mathbb{E}[p_{\\rm op}(1-p_{\\rm op})]$  (slow variables not in $Z$)', OI[0]),
             (29.5, 4.0, 'compression leak $\\mathbb{E}[(r^*-p_{\\rm op})^2]$\nS1, S4 (exact for Gaussian $x$)', OI[5]),
             (49.5, 26, 'observer-induced current\n$J=\\Omega\\Sigma^{-1}y\\rho$, $\\Omega=\\frac{1}{2}[C(1)-C(1)^\\top]$\nS2, S5, E4', OI[6]),
             (80, 4.0, 'estimator/calibration leak $\\approx$ REL\nE3, E5, E6', OI[3])]
    for x, y, t, c in notes:
        ax.text(x, y, t, ha='center', va='center', fontsize=7, color=c)
    for (xa, ya, xb, yb, c) in [(9.5, 23.5, 9.5, 19.6, OI[0]), (29.5, 6.6, 29.5, 10.4, OI[5]), (49.5, 23.0, 49.5, 19.6, OI[6]), (80, 6.6, 80, 10.4, OI[3])]:
        ax.annotate('', xy=(xb, yb), xytext=(xa, ya), arrowprops=dict(arrowstyle='->', color=c, lw=0.9))
    save(fig, 'fig1_pipeline')


def fig2_gaussian_observer():
    S1 = pd.read_csv(T / 'S1_gaussian_leak.csv'); S2 = pd.read_csv(T / 'S2_gaussian_irrev.csv')
    fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0), gridspec_kw=dict(wspace=0.32, hspace=0.42)); axs = axs.ravel()
    ax = axs[0]
    for phi, g in S1[(S1.method == 'bayes_observer') & (S1.h == 10)].groupby('phi'):
        g = g.sort_values('lam_w')
        ax.plot(g.lam_w, g.ell_theory, '-', color=PHICOL[phi], label=fr'$\phi={phi}$')
        ax.plot(g.lam_w, g.leak_direct / g.var_pop, 'o', ms=2.6, mfc='none', color=PHICOL[phi])
    lw = np.logspace(-2.3, 0.7, 200)
    from gaussian_observer import s2_continuous
    ax.plot(lw, [s2_continuous(1.0, [l, 4 * l]) for l in lw], ':', color='0.5', lw=0.9, label=r'$s^2(\lambda w)$')
    ax.set_xscale('log'); ax.set_xlabel(r'$\lambda w$'); ax.set_ylabel(r'normalized leak $L/\mathrm{Var}\,p_{\rm op}$'); ax.set_ylim(-0.02, 1)
    ax.legend(loc='upper left', fontsize=7, handlelength=1.2); lab(ax, '(a)')
    t = S2[S2.kind == 'trailing']
    for k, (col, ylab, logy) in enumerate([('ep', r'$\dot e_p$ of fitted OU (per step)', True), ('ratio', r'flux/drift ratio $R$', False)]):
        ax = axs[1 + k]
        for phi, g in t.groupby('phi'):
            g = g.sort_values('lam_w'); ax.plot(g.lam_w, g[f'{col}_closed'], '-', color=PHICOL[phi])
            ax.plot(g.lam_w, g[f'{col}_sim'], 'o', ms=2.6, mfc='none', color=PHICOL[phi])
        c = S2[S2.kind == 'centered']; ax.plot(c.lam_w, np.maximum(c[f'{col}_sim'], 1e-4 if logy else 0), 'x', ms=2.6, color='0.5', label='centered windows')
        ax.set_xscale('log'); ax.set_xlabel(r'$\lambda w$'); ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale('log'); ax.set_ylim(1e-4, 20)
        else:
            ax.set_ylim(-0.05, 1.9)
        lab(ax, '(b)' if k == 0 else '(c)')
    axs[1].legend(loc='lower left', fontsize=7)
    ax = axs[3]; D = pd.read_csv(T / 'E4_irreversibility_surrogates.csv'); Lv = pd.read_csv(T / 'E4b_leverage_surrogates.csv')
    spx = D[D.market == 'SPX']; rows = [('reversible\nsurrogates', spx[spx.variant == 'reversible_surrogate'].ep),
                                       ('random signs\non actual $|r|$', Lv.ep), ('literal IAAFT', spx[spx.variant == 'literal_iaaft'].ep)]
    for i, (name, v) in enumerate(rows):
        v = np.maximum(v.values, 1e-5); q = np.quantile(v, [0.05, 0.5, 0.95])
        ax.plot(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), '.', ms=1.4, color='0.6')
        ax.plot([q[0], q[2]], [i, i], '-', color=OI[5], lw=2); ax.plot(q[1], i, '|', color=OI[5], ms=7)
    det = spx.set_index('variant').ep
    for v, c, l in [('observed', OI[6], 'observed'), ('centered', OI[3], 'centered windows'), ('reversed_returns', OI[1], 'reversed returns')]:
        ax.axvline(det[v], color=c, lw=1.1, label=l)
    ax.set_xscale('log'); ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=7); ax.set_xlabel(r'$\dot e_p$ per day (S&P state)')
    ax.set_ylim(-0.6, 3.4); ax.set_xlim(1e-5, 0.5); ax.legend(loc='upper left', fontsize=7, ncol=3, columnspacing=0.8, handlelength=1.2); lab(ax, '(d)')
    ax.text(0.98, 0.03, f"Prop. 2 (linear): {det['prop2_prediction']:.1f} (off scale)", transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color='0.3')
    save(fig, 'fig2_gaussian_observer')


def fig3_doublewell():
    fz = pd.read_csv(LEG / 'tables_csf' / 'B_synthetic_phase.csv'); m = fz.groupby(['tau_c', 'w', 'method']).BSS.mean().reset_index()
    S = pd.read_csv(T / 'S4_summary.csv'); Sb = pd.read_csv(T / 'S4b_effective_variance.csv'); fit = pd.read_csv(T / 'S4c_scaling_fit.csv').iloc[0]
    S5 = pd.read_csv(T / 'S5_summary.csv'); LB = pd.read_csv(T / 'S4_lambda_bar.csv'); lb = dict(zip(LB.tau_c.astype(float), LB.lam_bar))
    fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0), gridspec_kw=dict(wspace=0.32, hspace=0.42)); axs = axs.ravel()
    ax = axs[0]
    for tau in [5.0, 20.0, 80.0, np.inf]:
        g = m[(m.tau_c == tau) & (m.method == 'rolling_flex')].sort_values('w'); o = m[(m.tau_c == tau) & (m.method == 'oracle_pde')].sort_values('w')
        x = lb[tau] * g.w * 0.05
        ax.plot(x, g.BSS, 'o-', ms=2.5, color=TAUCOL[tau], label=TAULAB[tau]); ax.plot(x, o.BSS, '--', color=TAUCOL[tau], lw=0.7)
    ax.set_xscale('log'); ax.set_xlabel(r'$\bar\lambda w$'); ax.set_ylabel('BSS (rolling flexible learner)'); ax.legend(fontsize=7, loc='lower left'); lab(ax, '(a)')
    ax = axs[1]; sel = [(20.0, 1), (20.0, 20), (20.0, 80), (np.inf, 1), (np.inf, 20), (np.inf, 80)]
    for i, (tau, w) in enumerate(sel):
        r = S[(S.tau_c.astype(float) == tau) & (S.w == w)].iloc[0]
        ou = max(r.BSS_ou, 0.0); segs = [(ou, OI[0]), (r.BSS_rolling - ou, OI[3]), (r.BSS_latent - r.BSS_rolling, OI[5]), (r.BSS_oracle - r.BSS_latent, OI[1])]
        b = 0.0
        for hgt, c in segs:
            ax.bar(i, hgt, bottom=b, color=c, width=0.7, lw=0); b += hgt
        if r.BSS_ou < 0:
            ax.text(i, 0.01, 'OU<0', ha='center', fontsize=7, color='w', rotation=90, va='bottom')
    ax.set_xticks(range(len(sel))); ax.set_xticklabels([f"{'∞' if np.isinf(t) else int(t)}\n{w}" for t, w in sel], fontsize=7)
    ax.set_xlabel(r'$\tau_c$ / $w$'); ax.set_ylabel('BSS'); lab(ax, '(b)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=OI[1], label='finite sample'), Patch(color=OI[5], label='compression'), Patch(color=OI[3], label='estimator/calibration'),
                       Patch(color=OI[0], label='OU skill')], fontsize=7, loc='upper center', ncol=2, columnspacing=0.6, handlelength=1.0)
    ax.set_ylim(0, 0.82)
    ax = axs[2]
    for tau in [5.0, 20.0, 80.0, np.inf]:
        g = S[S.tau_c.astype(float) == tau].sort_values('w'); g = g[g.w > 1]
        ax.errorbar(g.lam_w, g.dBSS_meas, yerr=g.dBSS_meas_se, fmt='o', ms=2.6, color=TAUCOL[tau], capsize=1.5, lw=0.7, label=TAULAB[tau])
    gi = S[(S.tau_c.astype(float) == np.inf) & (S.w > 1)].sort_values('w'); ax.plot(gi.lam_w, gi.dBSS_G, '--', color='0.3', label='Gaussian limit ($\\bar\\lambda$)')
    gb = Sb[(Sb.tau_c.astype(float) == np.inf) & (Sb.w > 1)].sort_values('w'); ax.plot(gb.lam_w, gb.dBSS_G_eff, ':', color='0.3', label='Mehler, measured $s^2$')
    u = np.logspace(np.log10(0.4), np.log10(9), 50); ax.plot(u, fit.a * u ** fit.b, '-', color=OI[7], lw=0.9, label=fr"${fit.a:.3f}(\bar\lambda w)^{{{fit.b:.2f}}}$")
    ax.set_xscale('log'); ax.set_xlabel(r'$\bar\lambda w$'); ax.set_ylabel(r'compression leak $\Delta$BSS'); ax.set_ylim(0, 0.62); ax.legend(fontsize=7, loc='upper left', ncol=2, columnspacing=0.5, handlelength=1.2); lab(ax, '(c)')
    ax = axs[3]
    for tau in [5.0, 20.0, 80.0, np.inf]:
        g = S5[S5.tau_c.astype(float) == tau].sort_values('w'); ax.plot(g.lam_w, g.obs_ep, 'o', ms=2.6, color=TAUCOL[tau], label=TAULAB[tau])
        if not np.isinf(tau):
            ax.axhline(g.lat_ep.iloc[0], color=TAUCOL[tau], ls=':', lw=0.8)
    g = S5[S5.tau_c.astype(float) == np.inf].sort_values('w'); ax.plot(g.lam_w, g.pred_ep, '-', color='0.3', label='Prop. 2 ($\\bar\\lambda$)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\bar\lambda w$'); ax.set_ylabel(r'$\dot e_p$ per step')
    ax.text(0.05, 0.08, 'dotted: latent $(x,c)$', transform=ax.transAxes, fontsize=7); ax.legend(fontsize=7, loc='center left'); lab(ax, '(d)')
    save(fig, 'fig3_doublewell')


ML = {'ou_raw': ('OU raw', OI[0], 'o'), 'ou_seqcal': ('OU seq-cal', OI[6], 's'), 'msm_seqcal': ('MSM seq-cal', OI[7], 'D'),
      'pc2_logistic': ('PC2 logistic', OI[5], '^'), 'pc2_gbt': ('PC2 GBT', OI[3], 'v'), 'raw_logistic': ('raw logistic', OI[1], '<')}


def fig4_murphy():
    R = pd.read_csv(T / 'E5_reliability.csv'); M = pd.read_csv(T / 'E5_murphy.csv')
    M = M[(M.set == 'common7') & (M.binning == 'fixed10')]
    fig, axs = plt.subplots(2, 2, figsize=(W2, 5.4), gridspec_kw=dict(wspace=0.3, hspace=0.55)); axs = axs.ravel()
    for k, h in enumerate([5, 20]):
        ax = axs[k]; ax.plot([0, 1], [0, 1], '-', color='0.8', lw=0.8)
        for m, (l, c, mk) in ML.items():
            g = R[(R['sample'] == '2008-2026') & (R.horizon == h) & (R.method == m) & (R.n >= 10)]
            ax.plot(g.p_mean, g.obs_freq, '-', marker=mk, ms=2.5, color=c, label=l, lw=0.8)
        ax.set_xlabel('forecast probability'); ax.set_ylabel('observed frequency'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f'{h}-day, 2008–2026', fontsize=7); lab(ax, '(a)' if k == 0 else '(b)')
    axs[0].legend(fontsize=7, loc='upper left')
    meths = list(ML)
    for k, h in enumerate([5, 20]):
        ax = axs[2 + k]; x = np.arange(len(meths))
        for j, (smp, hatch) in enumerate([('2008-2026', ''), ('2009-2026', '///')]):
            q = M[(M['sample'] == smp) & (M.horizon == h)].set_index('method')
            ax.bar(x + (j - 0.5) * 0.38, [q.REL.get(m, np.nan) for m in meths], width=0.36, color=OI[6], hatch=hatch, alpha=0.85, lw=0)
            ax.bar(x + (j - 0.5) * 0.38, [-q.RES.get(m, np.nan) for m in meths], width=0.36, color=OI[5], hatch=hatch, alpha=0.85, lw=0)
        ax.axhline(0, color='k', lw=0.6); ax.set_xticks(x); ax.set_xticklabels([ML[m][0] for m in meths], rotation=60, ha='right', fontsize=7)
        ax.set_ylabel('REL (up) / $-$RES (down)'); ax.set_title(f'{h}-day', fontsize=7); lab(ax, '(c)' if k == 0 else '(d)')
    save(fig, 'fig4_murphy')


def fig5_score_drift():
    D = pd.read_csv(T / 'E6_score_drift.csv'); Sx = pd.read_csv(T / 'E6_saturation.csv')
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.7), gridspec_kw=dict(wspace=0.38))
    ax = axs[0]; years = [2008, 2009, 2010, 2011, 2015, 2016, 2018, 2020, 2022, 2025]
    cm = plt.get_cmap('viridis')
    for i, Y in enumerate(years):
        g = D[(D.horizon == 5) & (D.year == Y) & (D.n >= 5)].sort_values('bin')
        ax.plot(g.bin + 1, g.rate, '-o', ms=2, lw=0.8, color=cm(i / (len(years) - 1)), label=str(Y))
    ax.set_xlabel('decile of raw score $S$ (pooled)'); ax.set_ylabel('event frequency $r^*_Y(S)$'); ax.legend(fontsize=7, ncol=2, loc='upper left'); lab(ax, '(a)')
    ax = axs[1]
    for h, c, off in [(5, OI[5], -0.15), (20, OI[6], 0.15)]:
        g = Sx[(Sx.horizon == h) & (Sx.days >= 5)]
        ax.errorbar(g.year + off, g.rate, yerr=[g.rate - g.lo90, g.hi90 - g.rate], fmt='o', ms=2.8, color=c, capsize=1.5, lw=0.8, label=f'{h}-day')
    ax.set_xlabel('year'); ax.set_ylabel(r'event rate when $S>0.99$'); ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7); lab(ax, '(b)')
    save(fig, 'fig5_score_drift')


def fig6_cases():
    C = pd.read_csv(T / 'E7_cases.csv', parse_dates=['date']); C = C[C.horizon == 5]
    fig, axs = plt.subplots(3, 3, figsize=(W2, 4.8), sharex='col', gridspec_kw=dict(hspace=0.12, wspace=0.42))
    for j, (case, title) in enumerate([('2011', 'Aug–Oct 2011'), ('2020', 'Feb 2020'), ('2025', 'Apr 2025')]):
        g = C[C['case'].astype(str) == case].sort_values('date')
        ax = axs[0, j]; ax.plot(g.date, g.close, color='k', lw=0.9); ax.set_title(title, fontsize=7.5)
        a2 = ax.twinx(); a2.plot(g.date, 100 * g.dd20, color=OI[6], lw=0.8); a2.axhline(-8, color=OI[6], ls=':', lw=0.7)
        a2.spines['right'].set_visible(True); a2.tick_params(labelsize=6)
        if j == 0: ax.set_ylabel('S&P 500')
        if j == 2: a2.set_ylabel('dd20 (%)', color=OI[6])
        ax = axs[1, j]; ax.plot(g.date, g.rv20, color=OI[5], lw=0.9, label='RV20'); ax.plot(g.date, g.rv_thr, '--', color='0.4', lw=0.8, label='threshold')
        if j == 0: ax.set_ylabel('RV20'); ax.legend(fontsize=7, loc='upper left')
        ax = axs[2, j]; ax.plot(g.date, g.ou_raw, color='k', lw=0.9, label='raw $S$'); ax.plot(g.date, g.ou_seqcal, color=OI[3], lw=0.9, label='seq-cal')
        yy = g[g.y == 1]; ax.plot(yy.date, np.full(len(yy), 1.08), '|', color=OI[6], ms=4, label='label $E=1$')
        ib = g[g.inB.astype(bool)]; ax.plot(ib.date, np.full(len(ib), -0.08), 's', color=OI[1], ms=1.8, label='in stress set')
        ax.set_ylim(-0.15, 1.15)
        if j == 0: ax.set_ylabel('5-day score'); ax.legend(fontsize=7, loc='center left', ncol=1)
        for a in axs[:, j]:
            a.tick_params(axis='x', labelsize=6, rotation=30)
    for j, l in enumerate(['(a)', '(b)', '(c)']):
        lab(axs[0, j], l)
    save(fig, 'fig6_cases')


def figC1_solver():
    shutil.copy(LEG / 'figures_csf' / 'B_fig1_doublewell.pdf', FIG / 'figC1_solver_validation.pdf')


ALL = [fig1_pipeline, fig2_gaussian_observer, fig3_doublewell, fig4_murphy, fig5_score_drift, fig6_cases, figC1_solver]
