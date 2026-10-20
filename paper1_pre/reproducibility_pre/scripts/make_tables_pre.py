"""Write every LaTeX table of the manuscript (Tables I-III) and of the Supplemental Material (S1-S12) from
tables_pre CSVs and frozen files; booktabs throughout; numbers are formatted, never edited.  Also writes
tables_pre/tab2_matched.csv and tab3_irreversibility.csv (the exact numbers in Tables II-III)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
PRE = Path(__file__).resolve().parents[1]; LEG = PRE.parent / 'reproducibility'; T = PRE / 'tables_pre'; OUT = PRE.parent / 'manuscript' / 'tables'
OUT.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(PRE / 'src'))
from pre_common import wide, murphy, bss, write_csv

def tex(s):
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

def tabular(df, colfmt, header, fname, long=False, caption=None):
    lines = [r'\begin{longtable}{' + colfmt + '}', r'\toprule', header + r'\\', r'\midrule', r'\endfirsthead',
             r'\toprule', header + r'\\', r'\midrule', r'\endhead'] if long else [r'\begin{tabular}{' + colfmt + '}', r'\toprule', header + r'\\', r'\midrule']
    for r in df.itertuples(index=False):
        lines.append(' & '.join(tex(v) if isinstance(v, str) else str(v) for v in r) + r'\\')
    lines += [r'\bottomrule', r'\end{longtable}' if long else r'\end{tabular}']
    (OUT / fname).write_text('\n'.join(lines) + '\n')

f3 = lambda v: '--' if pd.isna(v) else f'{v:+.3f}'
f4 = lambda v: '--' if pd.isna(v) else f'{v:.4f}'

# ---------------- Table I
d = pd.read_csv(T / 'T2_table1_closed_form.csv')
tabular(pd.DataFrame({'a': [f'{v:.1f}' for v in d.lam_w], 'b': [f'({w},{w2})' for w, w2 in zip(d.w, d.w2)], 'c': [f'{v:.3f}' for v in d.s2_discrete],
                      'd': [f'{v:+.4f}' for v in d.omega_C1], 'e': [f'{v:.3f}' for v in d.ratio_std], 'f': [f'{v:.2f}' for v in d.ep],
                      'g': [f'{v:.2f}' for v in d.d1], 'h': [f'{v:.3f}' for v in d.ell_h20]}),
        'rcccccccc'[:8], r'$\lambda w$ & windows & $s^2$ & $\omega$ & $R$ & $\dot e_p$ & $d_1$ & $L/\mathrm{Var}\,p_{\rm op}$', 'tab1_gaussian.tex')

# ---------------- Table II (common-7; three samples)
E = pd.read_csv(T / 'E3_forecasts_2026.csv', parse_dates=['date']); E = E[E.headline.astype(bool) & E.y.notna() & E.p.notna()]
rows, meths = [], [('ou_raw', 'OU raw'), ('ou_seqcal', 'OU seq-cal'), ('msm_seqcal', 'MSM seq-cal'), ('amc_seqcal', 'AMC seq-cal'),
                   ('pc2_logistic', 'PC2 logistic'), ('pc2_gbt', 'PC2 GBT'), ('raw_logistic', 'raw logistic')]
from pre_common import load_forecasts
_k = load_forecasts()[['date', 'horizon']].drop_duplicates().assign(frozen_headline=True)
E = E.merge(_k, on=['date', 'horizon'], how='left'); E['frozen_headline'] = E.frozen_headline.fillna(False).astype(bool)
blocks = [('2008--2020 (incl.\\ 2008; frozen headline sample)', E.frozen_headline), ('2009--2020 (excl.\\ 2008; frozen headline sample)', E.frozen_headline & (E.year >= 2009)),
          ('2008--2026 (extended)', E.year >= 2008), ('2009--2026 (extended, excl.\\ 2008)', E.year >= 2009)]
lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule', r' & \multicolumn{3}{c}{$h=5$} & \multicolumn{3}{c}{$h=20$}\\', r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
         r'Method & BSS & REL & RES & BSS & REL & RES\\', r'\midrule']
for bname, sel in blocks:
    Eb = E[sel.values]; lines.append(r'\multicolumn{7}{l}{\emph{' + bname + r'}}\\')
    Ws = {h: wide(Eb, 'common7', h) for h in [5, 20]}
    for m, lab in meths:
        vals = []
        for h in [5, 20]:
            W = Ws[h]; md = murphy(W.y.values, W[m].values); b = bss(W.y.values, W[m].values, W.cref.values)
            vals += [b, md['REL'], md['RES']]; rows.append(dict(block=bname, method=m, horizon=h, BSS=b, REL=md['REL'], RES=md['RES'], UNC=md['UNC'], n=md['n'], events=md['events']))
        lines.append(lab + ' & ' + ' & '.join([f3(vals[0]), f4(vals[1]), f4(vals[2]), f3(vals[3]), f4(vals[4]), f4(vals[5])]) + r'\\')
    u5 = murphy(Ws[5].y.values, Ws[5].ou_raw.values)['UNC']; u20 = murphy(Ws[20].y.values, Ws[20].ou_raw.values)['UNC']
    lines.append(r'\quad UNC (events) & \multicolumn{3}{c}{' + f'{u5:.4f} ({int(Ws[5].y.sum())})' + r'} & \multicolumn{3}{c}{' + f'{u20:.4f} ({int(Ws[20].y.sum())})' + r'}\\')
    lines.append(r'\addlinespace')
lines[-1] = r'\bottomrule'; lines.append(r'\end{tabular}'); (OUT / 'tab2_matched.tex').write_text('\n'.join(lines) + '\n')
write_csv(pd.DataFrame(rows), 'tab2_matched.csv')

# ---------------- Table III (irreversibility)
D = pd.read_csv(T / 'E4_irreversibility_surrogates.csv'); Lv = pd.read_csv(T / 'E4b_leverage_surrogates.csv')
def band(v):
    q = np.quantile(v, [.05, .5, .95]); return q
rows3 = []
spx = D[D.market == 'SPX']; det = spx[spx.sid < 0].set_index('variant')
def addrow(name, r):
    rows3.append(dict(row=name, **{k: r[k] for k in ['ratio', 'ep', 'd1', 'kld_q4_D2']}))
addrow('S&P observed state (trailing windows)', det.loc['observed'])
addrow('S&P, return sequence reversed', det.loc['reversed_returns'])
addrow('S&P, centred windows', det.loc['centered'])
for name, sub in [('S&P reversible surrogates (200), median [5\\%,95\\%]', spx[spx.variant == 'reversible_surrogate']),
                  ('S&P random signs on actual $|r|$ (50)', Lv)]:
    rows3.append(dict(row=name, **{k: tuple(band(sub[k].dropna())) for k in ['ratio', 'ep', 'd1', 'kld_q4_D2']}))
pr = det.loc['prop2_prediction']; rows3.append(dict(row='Proposition 2 prediction (windows 5,20,60,250)', ratio=pr.ratio, ep=pr.ep, d1=pr.d1, kld_q4_D2=np.nan))
for mk in ['BTC', 'ETH']:
    sub = D[D.market == mk]; o = sub[sub.variant == 'observed'].iloc[0]; s = sub[sub.variant == 'reversible_surrogate']
    rows3.append(dict(row=f'{mk} observed state', **{k: o[k] for k in ['ratio', 'ep', 'd1', 'kld_q4_D2']}))
    rows3.append(dict(row=f'{mk} reversible surrogates (200)', **{k: tuple(band(s[k].dropna())) for k in ['ratio', 'ep', 'd1', 'kld_q4_D2']}))
def fmt(v, nd):
    if isinstance(v, tuple):
        return f'{v[1]:.{nd}f} [{v[0]:.{nd}f}, {v[2]:.{nd}f}]'
    return '--' if pd.isna(v) else f'{v:.{nd}f}'
lines = [r'\begin{tabular}{lcccc}', r'\toprule', r'State & $R$ & $\dot e_p$ & $d_1$ & KLD$_{q=4}$\\', r'\midrule']
for r in rows3:
    lines.append(r['row'].replace('&', r'\&') + ' & ' + ' & '.join([fmt(r['ratio'], 3), fmt(r['ep'], 4), fmt(r['d1'], 4), fmt(r['kld_q4_D2'], 4)]) + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']; (OUT / 'tab3_irreversibility.tex').write_text('\n'.join(lines) + '\n')
write_csv(pd.DataFrame([{k: (str(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows3]), 'tab3_irreversibility.csv')

# ---------------- Supplemental tables
M = pd.read_csv(T / 'E5_murphy.csv'); M = M[M.binning == 'fixed10']
P = M.pivot_table(index=['set', 'method'], columns=['sample', 'horizon'], values='BSS')
P = P.reset_index(); cols = [c for c in P.columns if c[0] not in ('set', 'method')]
S1 = pd.DataFrame({'set': P[('set', '')], 'method': P[('method', '')], **{f'{c[0]} h={c[1]}': [f3(v) for v in P[c]] for c in cols}})
tabular(S1, 'll' + 'r' * len(cols), ' & '.join(['Set', 'Method'] + [f'{c[0]} $h={c[1]}$' for c in cols]), 'S1_feature_full.tex', long=True)
Z = pd.read_csv(LEG / 'tables_csf' / 'B_synthetic_phase.csv')
g = Z.groupby(['tau_c', 'w', 'method']).agg(AUC=('AUC', 'mean'), AUCsd=('AUC', 'std'), BSS=('BSS', 'mean'), BSSsd=('BSS', 'std')).reset_index()
S2 = pd.DataFrame({'tau': ['$\\infty$' if np.isinf(t) else f'{t:g}' for t in g.tau_c], 'w': g.w.astype(str), 'm': [str(m) for m in g.method],
                   'auc': [f'{a:.3f} $\\pm$ {b:.3f}' for a, b in zip(g.AUC, g.AUCsd)], 'bss': [f'{a:+.3f} $\\pm$ {b:.3f}' for a, b in zip(g.BSS, g.BSSsd)]})
lines = [r'\begin{longtable}{lllrr}', r'\toprule', r'$\tau_c$ & $w$ & Method & AUC & BSS\\', r'\midrule', r'\endfirsthead', r'\toprule', r'$\tau_c$ & $w$ & Method & AUC & BSS\\', r'\midrule', r'\endhead']
lines += [f"{r.tau} & {r.w} & {r.m.replace('_', chr(92) + '_')} & {r.auc} & {r.bss}" + r'\\' for r in S2.itertuples()] + [r'\bottomrule', r'\end{longtable}']
(OUT / 'S2_synthetic_full.tex').write_text('\n'.join(lines) + '\n')
B = pd.read_csv(T / 'E5_bootstrap.csv'); G = pd.read_csv(T / 'E5_gw_tests.csv')
B['ci'] = [f'{a:+.3f} [{b:+.3f}, {c:+.3f}]' for a, b, c in zip(B.dBSS, B.lo95, B.hi95)]
S3 = B.pivot_table(index=['sample', 'horizon', 'comparator'], columns='block', values='ci', aggfunc='first').reset_index()
gu = G[(G.test == 'unconditional') & (G.nw_lag == 20)].set_index(['sample', 'horizon', 'comparator']).p_value
gc = G[G.test != 'unconditional'].set_index(['sample', 'horizon', 'comparator']).p_value
S3['GWu'] = [f'{gu.get((a, b, c), np.nan):.3f}' for a, b, c in zip(S3['sample'], S3.horizon, S3.comparator)]
S3['GWc'] = [f'{gc.get((a, b, c), np.nan):.3f}' for a, b, c in zip(S3['sample'], S3.horizon, S3.comparator)]
S3 = S3[['sample', 'horizon', 'comparator', 5, 20, 60, 'GWu', 'GWc']].astype(str)
tabular(S3, 'llllllrr', r'Sample & $h$ & Comparator & MBB 5 & MBB 20 & MBB 60 & GW$_u$ & GW$_c$', 'S3_inference_all.tex', long=True)
TS = pd.read_csv(LEG / 'tables_csf' / 'B_target_sensitivity.csv').groupby(['target_q', 'horizon']).AUC.agg(['mean', 'std']).reset_index()
tabular(pd.DataFrame({'q': [f'{q:.2f}' for q in TS.target_q], 'h': TS.horizon.astype(str), 'auc': [f'{a:.3f} $\\pm$ {b:.3f}' for a, b in zip(TS['mean'], TS['std'])]}),
        'llr', r'Training quantile & $h$ & mean annual raw-OU AUC', 'S4_target.tex')
Ex = E[(E.set == 'common7') & (E.method == 'ou_raw')]; ch = []
for Y in range(2008, 2027):
    cur = Ex[Ex.year == Y]; pri = Ex[Ex.year < Y]
    ch.append(dict(Y=str(Y), n=str(int((cur.horizon == 5).sum())), e5=str(int(cur[cur.horizon == 5].y.sum())), e20=str(int(cur[cur.horizon == 20].y.sum())),
                   pn=str(int((pri.horizon == 5).sum())), pe5=str(int(pri[pri.horizon == 5].y.sum())), pe20=str(int(pri[pri.horizon == 20].y.sum())),
                   cal='fallback' if (pri.horizon == 5).sum() < 100 else 'logistic'))
tabular(pd.DataFrame(ch), 'lrrrrrrl', r'Year & Test $n$ & ev.\ $h{=}5$ & ev.\ $h{=}20$ & Prior $n$ & prior ev.\ 5 & prior ev.\ 20 & Calibration', 'S5_calibration_history.tex')
P8 = pd.read_csv(T / 'E8_pca_table.csv'); keep = ['year'] + [f'PC1_{c}' for c in ['log_rv20', 'log_rv5', 'dd20', 'ret20']] + ['PC1_explained'] + [f'PC2_{c}' for c in ['ews_var', 'ews_ar1', 'log_act_rel']] + ['PC2_explained']
tabular(pd.DataFrame({k: (P8[k].astype(int).astype(str) if k == 'year' else [f'{v:+.2f}' for v in P8[k]]) for k in keep}), 'l' + 'r' * (len(keep) - 1),
        r'Year & \multicolumn{4}{c}{PC1: lrv20, lrv5, dd20, ret20} & expl. & \multicolumn{3}{c}{PC2: ews\_var, ews\_ar1, act} & expl.', 'S6_pca.tex')
S3r = pd.read_csv(T / 'S3_ratio_recomputed.csv')
tabular(pd.DataFrame({'m': S3r.market, 'y': S3r.year.astype(str), 'a': [f4(v) if np.isfinite(v) else '--' for v in S3r.archived_ratio],
                      'b': [f'{v:.3f}' for v in S3r.ratio_analytic], 'c': [f'{v:.3f}' for v in S3r.ratio_grid], 'e': [f'{v:.4f}' for v in S3r.ep]}),
        'llrrrr', r'Market & Year & archived $R$ & $R$ (analytic) & $R$ (grid) & $\dot e_p$', 'S7_ratio_recomputed.tex', long=True)
EV = pd.read_csv(T / 'E2_event_log.csv'); EV = EV[EV.extension]
tabular(pd.DataFrame({'e': EV.entry_date, 'x': EV.exit_date, 'd': EV.days_in_set.astype(str), 'c': EV.cluster.astype(str)}), 'llrr',
        r'Entry & Exit & Days in set & Cluster', 'S8_events.tex')
c = pd.read_csv(T / 'T2_cov_check.csv'); c = c[c.i >= 0]
tabular(pd.DataFrame({'l': c.lag.astype(str), 'ij': [f'{i}{j}' for i, j in zip(c.i, c.j)], 'cf': [f'{v:.6f}' for v in c.closed_form],
                      'sm': [f'{v:.6f}' for v in c.sim_mean], 'se': [f'{v:.6f}' for v in c.sim_se], 're': [f'{100 * v:.3f}' for v in c.rel_err]}),
        'llrrrr', r'Lag & $ij$ & closed form & simulation & s.e. & rel.\ err.\ (\%)', 'S9_cov_check.tex')
s2 = pd.read_csv(T / 'S2_gaussian_irrev.csv'); s2 = s2[s2.kind.isin(['trailing', 'current+20'])]
tabular(pd.DataFrame({'phi': s2.phi.astype(str), 'k': s2.kind, 'w': [f'({a},{b})' for a, b in zip(s2.w, s2.w2)],
                      'R': [f'{a:.4f} / {b:.4f}' for a, b in zip(s2.ratio_closed, s2.ratio_sim)], 'ep': [f'{a:.3f} / {b:.3f}' for a, b in zip(s2.ep_closed, s2.ep_sim)],
                      'd1': [f'{a:.3f} / {b:.3f}' for a, b in zip(s2.d1_closed, s2.d1_sim)], 'k3': [f'{v:.3f}' for v in s2.kld_q3_D2], 'k5': [f'{v:.3f}' for v in s2.kld_q5_D2]}),
        'lllrrrrr', r'$\phi$ & windows & $(w_1,w_2)$ & $R$ closed/sim & $\dot e_p$ closed/sim & $d_1$ closed/sim & KLD$_3$ & KLD$_5$', 'S10_irrev_check.tex', long=True)
S5s = pd.read_csv(T / 'S5_summary.csv')
tabular(pd.DataFrame({'t': ['$\\infty$' if np.isinf(float(t)) else f'{float(t):g}' for t in S5s.tau_c], 'w': S5s.w.astype(str), 'o': [f'{v:.3f}' for v in S5s.obs_ep],
                      'p': [f'{v:.3f}' for v in S5s.pred_ep], 'l': [f'{v:.4f}' for v in S5s.lat_ep], 'r': [f'{v:.3f}' for v in S5s.obs_ratio]}),
        'llrrrr', r'$\tau_c$ & $w$ & $\dot e_p$ observed & Prop.~2 & latent $(x,c)$ & $R$ observed', 'S11_doublewell_current.tex', long=True)
pv = pd.read_csv(T / 'T1_pilot_verification.csv')
tabular(pd.DataFrame({'i': pv['item'], 'q': pv.quantity, 'p': [f'{v:g}' for v in pv.pilot], 'r': [f'{v:.4f}' for v in pv.recomputed], 'a': ['yes' if a else 'no' for a in pv.agrees]}),
        'llrrl', r'Item & Quantity & pilot & recomputed & agrees', 'S12_pilot_verification.tex', long=True)
print('tables written:', sorted(p.name for p in OUT.iterdir()))
