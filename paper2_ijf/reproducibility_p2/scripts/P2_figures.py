"""P2_figures -- draft Figures 1-6 of Paper 2 (vector PDF + PNG, Okabe-Ito colours) from the CSVs in tables_p2/.
Nothing new is estimated here except the PAV curves of Fig. 1(b,c), computed with the same estimator as F2 and F6.
Fig. 1 anatomy of S; Fig. 2 CORP decomposition; Fig. 3 saturated-day event rates by year; Fig. 4 matched comparisons;
Fig. 5 cases; Fig. 6 other markets.  Every caption ends with 'Data:' and the source CSVs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import TAB, FIG, LABELS, pav

OI = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'axes.titlesize': 8, 'legend.fontsize': 6.5})
MAIN = ['ou_raw', 'ou_seqcal', 'msm_seqcal', 'amc_seqcal', 'pc2_logistic', 'pc2_gbt', 'raw_logistic', 'raw_gbt']
CMP = ['msm_seqcal', 'amc_seqcal', 'pc2_logistic', 'pc2_gbt', 'raw_logistic', 'raw_gbt']
clip = lambda v: np.clip(np.asarray(v, float), 0, None)


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); fig.savefig(FIG / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def fig1():
    A = pd.read_csv(TAB / 'P2_F1_anatomy_exp.csv', parse_dates=['date'])
    F = pd.read_csv(TAB / 'P2_F1_forecasts_exp.csv', parse_dates=['date'])
    hl = F[(F.set == 'common7') & (F.method == 'ou_raw')][['date', 'horizon', 'headline']]
    A = A.merge(hl, on=['date', 'horizon']); A = A[A.headline.astype(bool) & A.y.notna()]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    g = A[(A.horizon == 5) & (A.year == 2022)]
    ax[0].scatter(g.s[g.y == 0], g.S[g.y == 0], s=5, c=OI[5], lw=0, label='no entry within 5 days')
    ax[0].scatter(g.s[g.y == 1], g.S[g.y == 1], s=9, c=OI[6], lw=0, label='entry within 5 days')
    ax[0].axvline(g.thr.iloc[0], c='grey', ls=':', lw=0.8)
    ax[0].set(xlabel='stress coordinate s', ylabel='raw score S (h = 5)', title='(a) model year 2022')
    ax[0].legend(frameon=False, loc='lower right')
    for k, h in enumerate([5, 20]):
        O = A[(A.horizon == h) & (A.s < A.thr) & A.hitB.notna()]
        o = np.argsort(O.S.values); p = O.S.values[o]
        for tgt, col, lab in [('hitB', OI[3], 'own target: enter learned set'), ('y', OI[6], 'operational event')]:
            ax[k + 1].plot(p, pav(O[tgt].values[o], p), c=col, lw=1.2, label=lab, drawstyle='steps-post')
        ax[k + 1].plot([0, 1], [0, 1], c='grey', lw=0.6, ls='--')
        ax[k + 1].set(xlabel='raw score S', ylabel='event frequency (PAV)', title=f'({"bc"[k]}) h = {h} days, 2008-2026')
    ax[1].legend(frameon=False, loc='upper left')
    fig.tight_layout(); save(fig, 'fig1_anatomy')


def fig2():
    D = pd.read_csv(TAB / 'P2_F2_decomposition_exp.csv'); Bt = pd.read_csv(TAB / 'P2_F2_bootstrap_exp.csv')
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.0), sharey=True)
    for j, h in enumerate([5, 20]):
        for i, smp in enumerate(['2008-2026', '2009-2026']):
            d = D[(D.set == 'common7') & (D.horizon == h) & (D['sample'] == smp)].set_index('method').loc[MAIN]
            b = Bt[(Bt.horizon == h) & (Bt['sample'] == smp)].set_index('method').loc[MAIN]; x = np.arange(len(MAIN))
            ax[i, j].bar(x - 0.2, d.DSC, 0.4, color=OI[5], label='discrimination DSC',
                         yerr=[clip(d.DSC - b.DSC_q05), clip(b.DSC_q95 - d.DSC)], error_kw=dict(lw=0.6, capsize=1.5))
            ax[i, j].bar(x + 0.2, d.MCB, 0.4, color=OI[6], label='miscalibration MCB')
            ax[i, j].set_yscale('log'); ax[i, j].set_xticks(x, [LABELS[m] for m in MAIN], rotation=45, ha='right')
            ax[i, j].set_title(f'h = {h} days, {smp}')
    ax[0, 0].legend(frameon=False); fig.tight_layout(); save(fig, 'fig2_corp')


def fig3():
    S = pd.read_csv(TAB / 'P2_F3_saturation_exp.csv'); H = pd.read_csv(TAB / 'P2_F3_homogeneity_exp.csv')
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3), sharey=True)
    for j, h in enumerate([5, 20]):
        s = S[(S.horizon == h) & (S.days >= 5)]
        ax[j].errorbar(s.year, s.rate, yerr=[clip(s.rate - s.lo90_episode_bootstrap), clip(s.hi90_episode_bootstrap - s.rate)],
                       fmt='o', ms=3, c=OI[5], capsize=2, lw=0.8)
        ax[j].axhline(s.events.sum() / s.days.sum(), c='grey', ls='--', lw=0.7)
        hh = H[H.horizon == h]
        p_chi = hh[~hh.valid.astype(bool)].p_value.iloc[0]
        p_per = hh[(hh.test == 'cluster permutation') & (hh.clusters == 'episodes, gap h')].p_value.iloc[0]
        ax[j].text(0.02, 0.97, f'day-level chi-square (invalid): p = {p_chi:.1g}\nepisode permutation: p = {p_per:.2f}',
                   transform=ax[j].transAxes, va='top', fontsize=6)
        ax[j].set(xlabel='model year', title=f'h = {h} days')
    ax[0].set_ylabel('event rate when S > 0.99'); fig.tight_layout(); save(fig, 'fig3_saturation')


def fig4():
    Sm = pd.read_csv(TAB / 'P2_F4_summary.csv')
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    blocks = [(5, '2008-2026'), (5, '2009-2026'), (20, '2008-2026'), (20, '2009-2026')]
    for j, tag in enumerate(['exp', 'roll5']):
        for k, (h, smp) in enumerate(blocks):
            q = Sm[(Sm.window == tag) & (Sm.horizon == h) & (Sm['sample'] == smp)].set_index('comparator').loc[CMP]
            yy = np.arange(len(CMP)) + (k - 1.5) * 0.17
            ax[j].errorbar(q.dBSS, yy, xerr=[clip(q.dBSS - q.lo90_2h), clip(q.hi90_2h - q.dBSS)], fmt='o', ms=2.5,
                           c=OI[[5, 2, 6, 1][k]], capsize=1.5, lw=0.8, label=f'h = {h}, {smp}')
        ax[j].axvline(0, c='grey', lw=0.6); ax[j].set_yticks(range(len(CMP)), [LABELS[c] for c in CMP])
        ax[j].set(xlabel='BSS(comparator) - BSS(committor, seq-cal)',
                  title='expanding windows' if tag == 'exp' else 'rolling five-year windows')
    ax[1].legend(frameon=False, loc='lower right'); fig.tight_layout(); save(fig, 'fig4_comparisons')


def fig5():
    C = pd.read_csv(TAB / 'P2_F5_case_series.csv', parse_dates=['date'])
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.0))
    for k, case in enumerate(['2011', '2020', '2025']):
        q = C[C.case.astype(str) == case]
        ax[k].fill_between(q.date, 0, 1.02, where=q.inB.astype(bool), color=OI[6], alpha=0.12, step='mid', lw=0, label='in stress set')
        ax[k].plot(q.date, q.ou_raw_5, c=OI[0], lw=1, label='raw S')
        ax[k].plot(q.date, q.ou_seqcal_5, c=OI[5], lw=1, label='committor, seq-cal')
        ax[k].plot(q.date, q.pc2_gbt_5, c=OI[1], lw=1, label='PC2 GBT')
        for d in q.date[q.entry.astype(bool)]:
            ax[k].axvline(d, c=OI[6], lw=0.9)
        ax[k].set(ylim=(0, 1.02), ylabel='h = 5 days', title=case)
    ax[0].legend(frameon=False, ncol=4, loc='upper right'); fig.tight_layout(); save(fig, 'fig5_cases')


def fig6():
    X = pd.read_csv(TAB / 'P2_X1_decomposition.csv'); X = X[~X['sample'].str.contains('without')]
    D = pd.read_csv(TAB / 'P2_F2_decomposition_exp.csv'); D = D[(D.set == 'common7') & (D['sample'] == '2008-2026')]
    meths = ['ou_raw', 'ou_seqcal', 'pc2_logistic', 'pc2_gbt']
    panels = [('S&P 500\n1957-1999', X[X.market == 'SPX1950']), ('S&P 500\n2008-2026', D),
              ('Bitcoin\n2018-2026', X[X.market == 'BTC']), ('Ether\n2021-2026', X[X.market == 'ETH'])]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for j, h in enumerate([5, 20]):
        for k, m in enumerate(meths):
            v = [float((g[(g.horizon == h) & (g.method == m)].DSC / g[(g.horizon == h) & (g.method == m)].UNC).iloc[0]) for _, g in panels]
            ax[j].bar(np.arange(len(panels)) + (k - 1.5) * 0.2, v, 0.2, color=OI[[0, 5, 2, 1][k]], label=LABELS[m])
        ax[j].set_xticks(range(len(panels)), [p for p, _ in panels]); ax[j].set_title(f'h = {h} days')
    ax[0].set_ylabel('DSC / UNC'); ax[0].legend(frameon=False); fig.tight_layout(); save(fig, 'fig6_markets')


if __name__ == '__main__':
    for f in [fig1, fig2, fig3, fig4, fig5, fig6]:
        f(); print(f.__name__, 'done')
