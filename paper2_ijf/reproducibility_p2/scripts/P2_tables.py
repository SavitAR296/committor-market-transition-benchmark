"""P2_tables -- LaTeX tables of the Paper 2 manuscript (../manuscript_p2/tables/) from tables_p2/.  No randomness.
Table 1 samples and episodes; 2 CORP decomposition (S&P 500); 3 dBSS against the recalibrated committor forecast;
4 tests (Holm-adjusted EWC, Giacomini-White conditional, model confidence set); 5 homogeneity of the saturated-day event
rate; 6 other markets; A1 label-rule sensitivity; A2 rolling windows.  AUC is never shown for seq-cal or climatology.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import P2, TAB, LABELS, MAIN_METHODS, auc_reported

OUT = P2.parent / 'manuscript_p2' / 'tables'; OUT.mkdir(parents=True, exist_ok=True)
CMP = ['msm_seqcal', 'amc_seqcal', 'pc2_logistic', 'pc2_gbt', 'raw_logistic', 'raw_gbt']
BLOCKS = [(5, '2008-2026'), (5, '2009-2026'), (20, '2008-2026'), (20, '2009-2026')]
f = lambda v, nd=3: '--' if pd.isna(v) else (f'${v:+.{nd}f}$' if nd and v < 0 else f'${v:.{nd}f}$')
pf = lambda v: '--' if pd.isna(v) else (f'${v:.0e}$'.replace('e-0', r'\times10^{-').replace('$', '', 0) if v < 1e-3 else f'{v:.3f}')


def p_fmt(v):
    if pd.isna(v):
        return '--'
    if v < 1e-3:
        m, e = f'{v:.0e}'.split('e'); return f'${m}\\times10^{{{int(e)}}}$'
    return f'{v:.3f}' if v < 0.1 else f'{v:.2f}'


def write(name, lines):
    (OUT / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def t1():
    F4 = pd.read_csv(TAB / 'P2_F4_summary.csv'); X = pd.read_csv(TAB / 'P2_X1_tests.csv')
    L = [r'\begin{tabular}{lrrrrrr}\toprule', r' & \multicolumn{3}{c}{$h=5$ days} & \multicolumn{3}{c}{$h=20$ days}\\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-7} sample & days & event days & event episodes & days & event days & event episodes\\\midrule']
    def row(label, q):
        a, b = q[q.horizon == 5].iloc[0], q[q.horizon == 20].iloc[0]
        return f'{label} & {int(a.n):,} & {int(a.events)} & {int(a.episodes)} & {int(b.n):,} & {int(b.events)} & {int(b.episodes)}\\\\'
    for s, lab in [('2008-2026', 'S\\&P 500, 2008--2026'), ('2009-2026', 'S\\&P 500, 2009--2026')]:
        L.append(row(lab, F4[(F4.window == 'exp') & (F4['sample'] == s) & (F4.comparator == 'pc2_gbt')]))
    for m, lab in [('SPX1950', 'S\\&P 500, 1957--1999'), ('BTC', 'Bitcoin, 2018--2026'), ('ETH', 'Ether, 2021--2026')]:
        L.append(row(lab, X[(X.market == m) & ~X['sample'].str.contains('without') & (X.comparator == 'pc2_gbt')]))
    write('tab1_samples', L + [r'\bottomrule\end{tabular}'])


def t2():
    D = pd.read_csv(TAB / 'P2_F2_decomposition_exp.csv'); D = D[D.set == 'common7']
    L = [r'\begin{tabular}{lrrrrrrrr}\toprule', r' & \multicolumn{4}{c}{$h=5$ days} & \multicolumn{4}{c}{$h=20$ days}\\',
         r'\cmidrule(lr){2-5}\cmidrule(lr){6-9} method & BSS & MCB & DSC & AUC & BSS & MCB & DSC & AUC\\\midrule']
    for smp, lab in [('2008-2026', '2008--2026'), ('2009-2026', '2009--2026 (without 2008)')]:
        L.append(rf'\multicolumn{{9}}{{l}}{{\emph{{{lab}}}}}\\')
        BS = pd.read_csv(TAB / 'P2_B2_stress_index.csv')
        for m in MAIN_METHODS:
            cells = []
            for h in (5, 20):
                r = D[(D.horizon == h) & (D['sample'] == smp) & (D.method == m)].iloc[0]
                cells += [f(r.BSS, 3), f(r.MCB, 4), f(r.DSC, 4), f(r.AUC, 3) if auc_reported(m) else '--']
            L.append(LABELS[m] + ' & ' + ' & '.join(cells) + r'\\')
            if m == 'ou_seqcal':
                cells = []
                for h in (5, 20):
                    r = BS[(BS.horizon == h) & (BS['sample'] == smp) & (BS.method == 's_seqcal')].iloc[0]
                    cells += [f(r.BSS, 3), f(r.MCB, 4), f(r.DSC, 4), '--']
                L.append(r'stress index $s$ seq-cal & ' + ' & '.join(cells) + r'\\')
    write('tab2_corp', L + [r'\bottomrule\end{tabular}'])


def t3(window='exp', name='tab3_dbss', with_holm=False):
    S = pd.read_csv(TAB / 'P2_F4_summary.csv'); S = S[S.window == window]
    head = ' & '.join(f'$h={h}$, {s.replace("-", "--")}' for h, s in BLOCKS)
    L = [r'\begin{tabular}{l' + 'r' * (len(BLOCKS) * (2 if with_holm else 1)) + r'}\toprule']
    if with_holm:
        L.append(' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{$h={h}$, {s.replace("-", "--")}}}' for h, s in BLOCKS) + r'\\')
        L.append('comparator' + ' & $\\Delta$BSS & Holm $p$' * len(BLOCKS) + r'\\\midrule')
    else:
        L.append('comparator & ' + head + r'\\\midrule')
    for c in CMP:
        r1, r2 = [LABELS[c]], ['']
        for h, s in BLOCKS:
            q = S[(S.horizon == h) & (S['sample'] == s) & (S.comparator == c)].iloc[0]
            if with_holm:
                r1 += [f(q.dBSS, 3) + f' {{\\scriptsize[{q.lo90_2h:+.3f}, {q.hi90_2h:+.3f}]}}', p_fmt(q.p_EWC_holm)]
            else:
                r1.append(f(q.dBSS, 3)); r2.append(f'{{\\scriptsize[{q.lo90_2h:+.3f}, {q.hi90_2h:+.3f}]}}')
        L.append(' & '.join(r1) + r'\\')
        if not with_holm:
            L.append(' & '.join(r2) + r'\\')
    if window == 'exp' and not with_holm:
        BS = pd.read_csv(TAB / 'P2_B2_stress_index.csv'); BS = BS[BS.method.str.startswith('dBSS')]
        r1, r2 = [r'stress index $s$ seq-cal'], ['']
        for h, s in BLOCKS:
            q = BS[(BS.horizon == h) & (BS['sample'] == s)].iloc[0]; r1.append(f(q.BSS, 3)); r2.append(f'{{\\scriptsize[{q.lo90:+.3f}, {q.hi90:+.3f}]}}')
        L += [' & '.join(r1) + r'\\', ' & '.join(r2) + r'\\']
    write(name, L + [r'\bottomrule\end{tabular}'])


def t4():
    S = pd.read_csv(TAB / 'P2_F4_summary.csv'); S = S[S.window == 'exp']
    L = [r'\begin{tabular}{lrrrrrr}\toprule']
    for h in (5, 20):
        L += [rf'\multicolumn{{7}}{{l}}{{\emph{{$h={h}$ days}}}}\\',
              r' & \multicolumn{3}{c}{2008--2026} & \multicolumn{3}{c}{2009--2026}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
              r'comparator & Holm $p$ & GW $p$ & MCS $p$ & Holm $p$ & GW $p$ & MCS $p$\\\midrule']
        for c in CMP:
            cells = []
            for s in ('2008-2026', '2009-2026'):
                q = S[(S.horizon == h) & (S['sample'] == s) & (S.comparator == c)].iloc[0]
                cells += [p_fmt(q.p_EWC_holm), p_fmt(q.p_GW_NW), p_fmt(q.mcs_p_cmp)]
            L.append(LABELS[c] + ' & ' + ' & '.join(cells) + r'\\')
        refs = [S[(S.horizon == h) & (S['sample'] == s)].mcs_p_ref.iloc[0] for s in ('2008-2026', '2009-2026')]
        L.append(rf'OU seq-cal (reference) & & & {p_fmt(refs[0])} & & & {p_fmt(refs[1])}\\' + (r'\midrule' if h == 5 else ''))
    write('tab4_tests', L + [r'\bottomrule\end{tabular}'])


def t5():
    E = pd.read_csv(TAB / 'P2_F3_homogeneity_exp.csv'); R = pd.read_csv(TAB / 'P2_F3_homogeneity_roll5.csv')
    L = [r'\begin{tabular}{llrrrr}\toprule', r' & & \multicolumn{2}{c}{expanding windows} & \multicolumn{2}{c}{rolling five-year windows}\\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6} test & units & $h=5$ & $h=20$ & $h=5$ & $h=20$\\\midrule']
    keys = [('day-level chi-square (independence assumed)', 'days'), ('cluster permutation', 'episodes, gap h'),
            ('cluster permutation', 'episodes, gap 2h'), ('cluster permutation', '20-day blocks'),
            ('quasi-binomial F', 'episodes, gap h'), ('quasi-binomial F', 'episodes, gap 2h'), ('quasi-binomial F', '20-day blocks')]
    for t, c in keys:
        cells = []
        for H in (E, R):
            for h in (5, 20):
                q = H[(H.horizon == h) & (H.test == t) & (H.clusters == c)].iloc[0]; cells.append(f'{p_fmt(q.p_value)} ({int(q.n_clusters)})')
        lab = 'chi-square (invalid)' if t.startswith('day') else t
        L.append(f'{lab} & {c.replace("gap h", "gap $h$").replace("gap 2h", "gap $2h$")} & ' + ' & '.join(cells) + r'\\')
    write('tab5_drift', L + [r'\bottomrule\end{tabular}'])


def t6():
    M = pd.read_csv(TAB / 'P2_X1_metrics.csv'); D = pd.read_csv(TAB / 'P2_X1_decomposition.csv'); H = pd.read_csv(TAB / 'P2_X1_homogeneity.csv')
    full = lambda X: X[~X['sample'].str.contains('without')]
    M, D = full(M), full(D)
    L = [r'\begin{tabular}{llrrrrrr}\toprule', r'market & $h$ & \multicolumn{4}{c}{BSS} & $\Delta$DSC raw $S-$PC2 logistic & homogeneity $p$\\',
         r'\cmidrule(lr){3-6} & & raw $S$ & OU seq-cal & PC2 logistic & PC2 GBT & [90\% interval] & perm. / quasi-bin.\\\midrule']
    for m, lab in [('SPX1950', 'S\\&P 500, 1957--1999'), ('BTC', 'Bitcoin, 2018--2026'), ('ETH', 'Ether, 2021--2026')]:
        for h in (5, 20):
            b = [M[(M.market == m) & (M.horizon == h) & (M.method == k)].BSS.iloc[0] for k in ['ou_raw', 'ou_seqcal', 'pc2_logistic', 'pc2_gbt']]
            d = D[(D.market == m) & (D.horizon == h) & (D.method == 'pc2_logistic')].iloc[0]
            hh = H[(H.market == m) & (H.horizon == h)]
            hs = f'{p_fmt(hh.p_cluster_permutation.iloc[0])} / {p_fmt(hh.p_quasibinomial.iloc[0])}' if len(hh) else '--'
            L.append(f'{lab if h == 5 else ""} & {h} & ' + ' & '.join(f(x, 3) for x in b) +
                     f' & [{d.dDSC_rawS_minus_m_q05:+.4f}, {d.dDSC_rawS_minus_m_q95:+.4f}] & {hs}\\\\')
    write('tab6_markets', L + [r'\bottomrule\end{tabular}'])


def tA1():
    R = pd.read_csv(TAB / 'P2_F7_label_rule.csv')
    g = lambda run, h, s, q, m: R[(R.run == run) & (R.horizon == h) & (R['sample'] == s) & (R.quantity == q) & (R.method == m)].value.iloc[0]
    L = [r'\begin{tabular}{l' + 'rr' * 4 + r'}\toprule',
         ' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{$h={h}$, {s.replace("-", "--")}}}' for h, s in BLOCKS) + r'\\',
         'quantity' + ' & headline & dropped' * 4 + r'\\\midrule']
    items = [('BSS, OU seq-cal', 'BSS', 'ou_seqcal', 3), ('BSS, PC2 GBT', 'BSS', 'pc2_gbt', 3), ('BSS, PC2 logistic', 'BSS', 'pc2_logistic', 3),
             ('DSC, raw $S$', 'DSC', 'ou_raw', 4)]
    for lab, q, m, nd in items:
        L.append(lab + ' & ' + ' & '.join(f(g(run, h, s, q, m), nd) for h, s in BLOCKS for run in ('exp', 'exp_ld')) + r'\\')
    B2 = pd.read_csv(TAB / 'P2_F2_bootstrap_exp.csv')       # headline column = the interval of Section 4 (outline A3)
    hq = lambda h, s: B2[(B2.horizon == h) & (B2['sample'] == s) & (B2.method == 'ou_seqcal')].dDSC_rawS_minus_m_q05.iloc[0]
    L.append(r'$\Delta$DSC raw $S-$seq-cal, q05 & ' + ' & '.join(f(hq(h, s) if run == 'exp' else g(run, h, s, 'dDSC rawS-m q05', 'ou_seqcal'), 4) for h, s in BLOCKS for run in ('exp', 'exp_ld')) + r'\\')
    hm = lambda run, h, s: R[(R.run == run) & (R.horizon == h) & (R['sample'] == s) & (R.quantity == 'p EWC Holm')].value.min()
    L.append('smallest Holm $p$ & ' + ' & '.join(p_fmt(hm(run, h, s)) for h, s in BLOCKS for run in ('exp', 'exp_ld')) + r'\\')
    write('tabA1_label_rule', L + [r'\bottomrule\end{tabular}'])


def tA3():
    """Block-length sensitivity of the dBSS intervals (outline A6)."""
    D = pd.read_csv(TAB / 'P2_F4_dbss.csv'); D = D[D.window == 'exp']; bl = sorted(D.block.unique())
    L = [r'\begin{tabular}{ll' + 'r' * len(bl) + r'}\toprule', r'comparator & $h$, sample & ' + ' & '.join(f'{b} days' for b in bl) + r'\\\midrule']
    for c in CMP:
        for h, s in BLOCKS:
            q = D[(D.comparator == c) & (D.horizon == h) & (D['sample'] == s)].set_index('block')
            L.append(f"{LABELS[c] if (h, s) == BLOCKS[0] else ''} & {h}, {s.replace('-', '--')} & " + ' & '.join(f'[{q.lo90[b]:+.3f}, {q.hi90[b]:+.3f}]' for b in bl) + r'\\')
    write('tabA3_blocks', L + [r'\bottomrule\end{tabular}'])


def tA4():
    """Model-confidence-set p-values in all eight families (outline A9)."""
    M = pd.read_csv(TAB / 'P2_F4_mcs.csv'); fam = [(w, h, s) for w in ('exp', 'roll5') for h in (5, 20) for s in ('2008-2026', '2009-2026')]
    L = [r'\begin{tabular}{l' + 'r' * len(fam) + r'}\toprule', r' & \multicolumn{4}{c}{expanding windows} & \multicolumn{4}{c}{rolling five-year windows}\\',
         r'\cmidrule(lr){2-5}\cmidrule(lr){6-9} $h$ & 5 & 5 & 20 & 20 & 5 & 5 & 20 & 20\\', r'method & ' + ' & '.join(s[2:4] + '--' + s[7:9] for w, h, s in fam) + r'\\\midrule']
    for m in ['ou_seqcal'] + CMP + ['clim']:
        L.append(LABELS.get(m, 'climatology') + ' & ' + ' & '.join(p_fmt(M[(M.method == m) & (M.window == w) & (M.horizon == h) & (M['sample'] == s)].mcs_p.iloc[0]) for w, h, s in fam) + r'\\')
    write('tabA4_mcs', L + [r'\bottomrule\end{tabular}'])


def tA5():
    """Within-year AUC (outline B3)."""
    A = pd.read_csv(TAB / 'P2_F2_annual_auc_exp.csv'); A = A[~A.constant.astype(bool) & A.AUC.notna()]
    meths = ['ou_raw', 'msm_raw', 'amc_raw', 'pc2_logistic', 'pc2_gbt', 'raw_logistic', 'raw_gbt']
    L = [r'\begin{tabular}{lrrrr}\toprule', r' & \multicolumn{2}{c}{$h=5$ days} & \multicolumn{2}{c}{$h=20$ days}\\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5} method & median [quartiles] & years & median [quartiles] & years\\\midrule']
    for m in meths:
        cells = []
        for h in (5, 20):
            v = A[(A.horizon == h) & (A.method == m)].AUC
            cells += [f'{v.median():.3f} [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]', str(len(v))]
        L.append(LABELS.get(m, m) + ' & ' + ' & '.join(cells) + r'\\')
    write('tabA5_auc', L + [r'\bottomrule\end{tabular}'])


def tA6():
    """Classifier settings (outline B6): BSS of the headline settings and of the alternatives."""
    D = pd.read_csv(TAB / 'P2_F2_decomposition_exp.csv'); D = D[D.set == 'common7']
    rows = [('pc2_logistic', 'PC2 logistic, $C=1$ (headline)'), ('pc2_logistic_C0.1', 'PC2 logistic, $C=0.1$'), ('pc2_logistic_C10', 'PC2 logistic, $C=10$'),
            ('raw_logistic', 'raw logistic, $C=0.2$ (headline)'), ('raw_logistic_C0.1', 'raw logistic, $C=0.1$'), ('raw_logistic_C1', 'raw logistic, $C=1$'),
            ('raw_logistic_C10', 'raw logistic, $C=10$'), ('pc2_gbt', 'PC2 GBT, depth 2, 120 trees (headline)'), ('pc2_gbt_cv', 'PC2 GBT, cross-validated'),
            ('raw_gbt', 'raw GBT, depth 2, 120 trees (headline)'), ('raw_gbt_cv', 'raw GBT, cross-validated')]
    L = [r'\begin{tabular}{lrrrr}\toprule', 'setting & ' + ' & '.join(f'$h={h}$, {s.replace("-", "--")}' for h, s in BLOCKS) + r'\\\midrule']
    for m, lab in rows:
        L.append(lab + ' & ' + ' & '.join(f(D[(D.horizon == h) & (D['sample'] == s) & (D.method == m)].BSS.iloc[0], 3) for h, s in BLOCKS) + r'\\')
    write('tabA6_settings', L + [r'\bottomrule\end{tabular}'])


def t_size():
    """Size and power of the homogeneity tests (outline B4)."""
    S = pd.read_csv(TAB / 'P2_S1_size_simulation.csv'); ds = sorted(S.delta.unique())
    L = [r'\begin{tabular}{l' + 'r' * (2 * len(ds)) + r'}\toprule', rf' & \multicolumn{{{len(ds)}}}{{c}}{{$h=5$ days}} & \multicolumn{{{len(ds)}}}{{c}}{{$h=20$ days}}\\',
         rf'\cmidrule(lr){{2-{1 + len(ds)}}}\cmidrule(lr){{{2 + len(ds)}-{1 + 2 * len(ds)}}} test & ' + ' & '.join(('null' if d == 0 else f'$\\delta={d:g}$') for d in ds * 2) + r'\\\midrule']
    for t in ['day-level chi-square', 'episode permutation', 'quasi-binomial F']:
        L.append(t + ' & ' + ' & '.join(f"{S[(S.horizon == h) & (S.test == t) & (S.delta == d)].rejection_rate.iloc[0]:.3f}" for h in (5, 20) for d in ds) + r'\\')
    write('tab_size', L + [r'\bottomrule\end{tabular}'])


if __name__ == '__main__':
    t1(); t2(); t3(); t4(); t5(); t6(); tA1(); t3('roll5', 'tabA2_rolling', with_holm=True); tA3(); tA4(); tA5(); tA6(); t_size()
    print(sorted(p.name for p in OUT.glob('*.tex')))
