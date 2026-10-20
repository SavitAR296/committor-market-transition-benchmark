"""Tables of Paper 1 as LaTeX fragments in ../manuscript/tables/: main Tables I-III and Supplemental Tables S1-S12.
Inputs: tables_pre/*.csv only.  File names carry the final supplemental numbers; the order of the floats in
supplement.tex fixes the numbering, and check_table_numbers.py verifies it."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))

T = ROOT / 'tables_pre'
OUT = ROOT.parent / 'manuscript' / 'tables'; OUT.mkdir(parents=True, exist_ok=True)
WIN, LEV, DD = 'window(log_rv5,log_rv20)', 'leverage(ret20,log_rv20)', 'leverage(dd20,log_rv20)'


def f(x, d=4, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    if round(x, d) == 0:
        x = 0.0
    s = (f'{x:+.{d}f}' if x != 0 else f'{0:.{d}f}') if sign else f'{x:.{d}f}'
    return s.replace('-', '$-$').replace('+', '$+$') if sign else s


def band(s, d=4, sign=True):
    return f'{f(s.median(), d, sign)} [{f(s.quantile(.05), d, sign)}, {f(s.quantile(.95), d, sign)}]'


def write(name, body):
    (OUT / name).write_text(body)


def sci(x):
    return f'{x:.2e}'.replace('e+0', r'\times10^{').replace('e+', r'\times10^{') + '}'


def fse(x, se, d=4):
    """value with its bootstrap standard error in units of the last digit, e.g. -0.0203(19)"""
    return f(x, d) + f'({int(round(se * 10 ** d))})'


def tab1():
    D = pd.read_csv(T / 'T2d_table1.csv')
    rows = [f"{r.lam_w:.1f} & ({int(r.w)},{int(r.w2)}) & {r.s2:.3f} & {f(r.omega)} & {f(r.omega_fit)} & ${sci(r.G)}$ & "
            f"{r.ep:.2f} & {r.ep_eta2_p01:.3f} & {r.ell_h10:.3f}\\\\" for r in D.itertuples()]
    write('tab1_gaussian.tex', "\\begin{ruledtabular}\n\\begin{tabular}{rcccccrrc}\n"
          "$\\lambda w$ & windows & $s^2$ & $\\omega$ & $\\omega_{\\rm fit}$ & $\\mathcal G$ & $\\dot e_p$ & "
          "$\\dot e_p^{\\,\\eta}$ & $\\ell$\\\\\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n\\end{ruledtabular}\n")


def tab2():
    P = pd.read_csv(T / 'E4e_pairs.csv'); C = pd.read_csv(T / 'E4c_pca_factorization.csv')
    B = pd.read_csv(T / 'E4e_surrogate_bands.csv'); SE = pd.read_csv(T / 'E4g_bootstrap_se.csv')
    SE = SE[SE.block == 20].set_index(['sample', 'channel'])
    sur = C[C.state == 'reversible_surrogate']
    names = {WIN: 'window', LEV: 'leverage, return', DD: 'leverage, drawdown', 'PCA2_common_basis': 'PCA(2) state'}
    body = ["\\begin{ruledtabular}\n\\begin{tabular}{llrrrrrl}",
            " & channel & $\\omega$ & $\\omega^{\\rm rev}$ & $\\omega^{\\rm cen}$ & $O$ & $P$ & surrogates\\\\\n\\hline"]
    for sample, tag in [('SPX 2000-2026', 'S\\&P 500'), ('SPX 2000-2020', 'S\\&P, to 2020'), ('BTC', 'Bitcoin'), ('ETH', 'Ether')]:
        for j, (key, nm) in enumerate(names.items()):
            if sample != 'SPX 2000-2026' and key == DD:
                continue
            r = P[(P['sample'] == sample) & (P.channel == key)].iloc[0]
            if sample == 'SPX 2000-2026':
                s = sur['omega_common' if key == 'PCA2_common_basis' else f'omega_{key}']; sb = band(s, 3)
            elif sample in ('BTC', 'ETH'):
                s = B[B['sample'] == sample]['omega_PCA2_common_basis' if key == 'PCA2_common_basis' else f'omega_{key}']
                sb = band(s, 3) if key != DD else '--'
            else:
                sb = '--'
            e = SE.loc[(sample, key)]
            body.append(f"{tag if j == 0 else ''} & {nm} & {fse(r.omega_fwd, e.se_fwd)} & {fse(r.omega_rev, e.se_rev)} & "
                        f"{fse(r.omega_cen, e.se_cen)} & {fse(r.O, e.se_O)} & {fse(r.P, e.se_P)} & {sb}\\\\")
        body.append('\\hline' if sample != 'ETH' else '')
    write('tab2_markets.tex', "\n".join(body) + "\n\\end{tabular}\n\\end{ruledtabular}\n")


def tab3():
    V = pd.read_csv(T / 'E4f_sv_variants.csv'); C = pd.read_csv(T / 'E4c_pca_factorization.csv')
    P = pd.read_csv(T / 'E4e_pairs.csv')
    body = ["\\begin{ruledtabular}\n\\begin{tabular}{rrrrrrr}",
            " & \\multicolumn{3}{c}{Gaussian, one factor} & \\multicolumn{3}{c}{Student-$t$, two factors}\\\\",
            "$\\rho$ & $\\dot e_p$ & $\\omega_{\\rm lev}$ & $L_+$ & $\\dot e_p$ & $\\omega_{\\rm lev}$ & $L_+$\\\\\n\\hline"]
    for rho in [0.0, -0.4, -0.6, -0.75, -0.85, -0.95]:
        cells = []
        for v in ['G1', 'T2']:
            g = V[(V.variant == v) & np.isclose(V.rho, rho)]
            cells += [f(g.ep.median(), 4, False), f(g.om_lev.median()), f'{g.lev_fwd.median():.0f}'.replace('-', '$-$')]
        body.append(f"{rho:.2f}".replace('-', '$-$') + ' & ' + ' & '.join(cells) + '\\\\')
    obs = C[C.state == 'observed'].iloc[0]
    lev = P[(P['sample'] == 'SPX 2000-2026') & (P.channel == LEV)].iloc[0].omega_fwd
    d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026_r1.csv'); r = np.diff(np.log(d.close.to_numpy(float)))
    r = r - r.mean(); r2 = r ** 2; Lp = sum(np.mean(r[:-k] * r2[k:]) for k in range(1, 21)) / np.mean(r2) ** 2
    body.append('\\hline\ndata & ' + f(obs.ep, 4, False) + ' & ' + f(lev) + ' & ' + f'{Lp:.0f}'.replace('-', '$-$')
                + ' & \\multicolumn{3}{c}{(same)}\\\\')
    write('tab3_sv.tex', "\n".join(body) + "\n\\end{tabular}\n\\end{ruledtabular}\n")


def supp():
    A = pd.read_csv(T / 'E1d_archive_check.csv').iloc[0]; R = pd.read_csv(T / 'E1e_repair_log.csv')
    s1 = ["\\begin{tabular}{p{7.2cm}p{9.3cm}}\\toprule quantity & value\\\\\\midrule",
          f"common days with the independent archive & {A.common_days} ({A['first']} to {A['last']})\\\\",
          f"dates present in only one file & {A.frozen_only_days} / {A.archive_only_days}\\\\",
          f"largest relative close difference & ${A.close_max_rel_diff:.1e}$\\\\".replace('e-0', r'\times10^{-') .replace('$', '$', 1),
          f"volume ratio, median [min, max] & {A.volume_ratio_median:.5f} [{A.volume_ratio_min:.5f}, {A.volume_ratio_max:.5f}]\\\\",
          "patched days in the 2000--2026 file & " + '; '.join(
              f"{r.date}: volume {r.volume_original:.0f} $\\to$ {r.volume_repaired:,.0f} ({r.source_short})" for r in R.itertuples()) + "\\\\",
          "\\bottomrule\\end{tabular}"]
    s1[3] = f"largest relative close difference & {A.close_max_rel_diff:.1e}\\\\"
    Y = pd.read_csv(T / 'E1b_splice_report.csv')
    for _, y in Y.iterrows():
        cmp = y.comparison.replace('frozen snapshot', 'vega-datasets snapshot').replace('earlier extension', 'second copy of the Yahoo series')
        s1.insert(-2, f"{cmp}: common days ({y['first']} to {y['last']}) & {y.common_days}\\\\")
        s1.insert(-2, f"\\quad close max.\\ rel.\\ diff.; volume ratio median [min, max]; days off $>1\\%$ & "
                      f"{y.close_max_rel_diff:.1e}; {y.volume_ratio_median:.4f} [{y.volume_ratio_min:.4f}, {y.volume_ratio_max:.4f}]; {y.volume_days_off_gt_1pct}\\\\")
    write('S1_data.tex', "\n".join(s1) + "\n")

    L = pd.read_csv(T / 'E4c_linear_decomposition.csv')
    s2 = ["\\begin{tabular}{llrrrr}\\toprule input & pair & measured & theory & $O$ & $P$\\\\\\midrule"]
    for _, r in L.iterrows():
        s2.append(f"{r.direction} & {r.pair.replace('r2', '$r^2$')} & {f(r.omega_measured)} & {f(r.omega_theory)} & "
                  f"{f(r.observer_term)} & {f(r.physical_term)}\\\\")
    write('S2_prop1_linear.tex', "\n".join(s2) + "\n\\bottomrule\\end{tabular}\n")

    F = pd.read_csv(T / 'T2b_ep_factorization.csv'); F = F[np.isclose(F.phi, 0.98)]
    s4 = ["\\begin{tabular}{rrrrrrr}\\toprule $w$ & $\\eta^2$ & $\\dot e_p$ & $\\omega_{\\rm fit}$ & $\\mathcal G$ & $d_{\\min}$ & residual\\\\\\midrule"]
    for _, r in F.iterrows():
        s4.append(f"{int(r.w)} & {r.eta2_frac:g} & {r.ep:.4g} & {f(r.omega)} & {r.G:.4g} & {r.d_min:.3g} & {r.identity_residual:.0e}\\\\")
    write('S4_factorization.tex', "\n".join(s4) + "\n\\bottomrule\\end{tabular}\n")

    E = pd.read_csv(T / 'S4e_two_rate_leak.csv'); M = pd.read_csv(T / 'S4_summary.csv')
    M['tau_c'] = M.tau_c.astype(float)
    s5 = ["\\begin{tabular}{rrrrrrr}\\toprule $\\tau_c$ & $w$ & $\\bar\\lambda w$ & exact & two-rate & single-rate & classifier-skill estimate\\\\\\midrule"]
    for _, r in E[E.w > 1].iterrows():
        m = M[(M.tau_c == r.tau_c) & (M.w == r.w)]
        gb = f'{m.ell_meas.iloc[0]:.3f}' if len(m) else '--'
        s5.append(f"{'$\\infty$' if np.isinf(r.tau_c) else int(r.tau_c)} & {int(r.w)} & {r.lam_w:.2f} & {r.ell_exact:.3f} & "
                  f"{r.ell_two_rate:.3f} & {r.ell_local_gaussian:.3f} & {gb}\\\\")
    write('S5_doublewell_leak.tex', "\n".join(s5) + "\n\\bottomrule\\end{tabular}\n")

    S = pd.read_csv(T / 'E4_summary.csv'); Cc = pd.read_csv(T / 'E4c_pca_factorization.csv')
    Sy = pd.read_csv(T / 'E4e_symmetric_sign.csv'); B = pd.read_csv(T / 'E4e_surrogate_bands.csv')
    Sall = pd.read_csv(T / 'E4_irreversibility_surrogates.csv')
    s6 = ["\\begin{tabular}{llrlr}\\toprule state & null & observed $\\dot e_p$ & null median [5\\%, 95\\%] & null max\\\\\\midrule"]
    for _, r in S[S.statistic == 'ep'].iterrows():
        mx = Sall[(Sall.market == r.market) & (Sall.variant == r['null'])].ep.max()
        s6.append(f"{r.market} & {r['null'].replace('_', ' ')} & {r.observed:.4f} & {r['median']:.4f} [{r.q05:.4f}, {r.q95:.4f}] & {mx:.4f}\\\\")
    ob = Cc[Cc.state == 'observed'].ep.iloc[0]; rs = Cc[Cc.state == 'random_signs'].ep
    s6.append(f"SPX & symmetric-sign surrogates & {ob:.4f} & {band(Sy.ep_PCA2_own, 4, False)} & {Sy.ep_PCA2_own.max():.4f}\\\\")
    s6.append(f"SPX & random signs on actual $|r|$ & {ob:.4f} & {band(rs, 4, False)} & {rs.max():.4f}\\\\")
    write('S6_surrogates.tex', "\n".join(s6) + "\n\\bottomrule\\end{tabular}\n")

    V = pd.read_csv(T / 'E4f_sv_variants.csv'); K = pd.read_csv(T / 'E4f_calibration.csv').iloc[0]
    s7 = [f"% calibration: kappa={K.kappa:.3f}, phi_zeta={K.phi_zeta:.3f}, sd_zeta={K.sd_zeta:.3f}, phi_slow={K.phi_slow}, fast share={K.fast_share}",
          "\\begin{tabular}{llllll}\\toprule variant & $\\rho$ & $\\dot e_p$ median [5\\%, 95\\%] & $\\omega_{\\rm win}$ & $\\omega_{\\rm lev}$ & $L_+$\\\\\\midrule"]
    for (v, rho), g in V.groupby(['variant', 'rho'], sort=False):
        s7.append(f"{v} & {rho:.2f} & {band(g.ep, 4, False)} & {f(g.om_win.median())} & {f(g.om_lev.median())} & {g.lev_fwd.median():.0f}\\\\")
    write('S7_sv_variants.tex', "\n".join(s7) + "\n\\bottomrule\\end{tabular}\n")

    O3 = pd.read_csv(T / 'E4e_third_order.csv').iloc[0]
    write('S8_third_order.tex', "\\begin{tabular}{lrrll}\\toprule statistic & observed & reversed & IAAFT null [2.5\\%, 97.5\\%] & $p$\\\\\\midrule\n"
          f"$\\sum_{{k=1}}^{{20}}\\langle z_t^2z_{{t+k}}-z_tz_{{t+k}}^2\\rangle$, $z=|r|$ & {O3.observed:.3f} & {O3.reversed:.3f} & "
          f"{O3.null_median:.3f} [{O3.null_q025:.3f}, {O3.null_q975:.3f}] & {O3.p_two_sided:.3f}\\\\\n\\bottomrule\\end{{tabular}}\n")


def supp_more():
    # S3: Gaussian observer, closed forms against simulation (T2_cov_check.csv, S2_gaussian_irrev.csv)
    Cc = pd.read_csv(T / 'T2_cov_check.csv'); Cc = Cc[(Cc.phi == 0.98) & (Cc.w1 == 5) & (Cc.w2 == 20) & Cc.lag.isin([0, 1, 5])]
    Cc = Cc.drop_duplicates(['lag', 'i', 'j'])
    a = ["\\begin{tabular}{llrrrr}\n\\toprule\nLag & $ij$ & closed form & simulation & s.e. & rel.\\ err.\\ (\\%)\\\\\n\\midrule"]
    a += [f"{int(r.lag)} & {int(r.i)}{int(r.j)} & {r.closed_form:.6f} & {r.sim_mean:.6f} & {r.sim_se:.6f} & {100 * abs(r.rel_err):.3f}\\\\" for r in Cc.itertuples()]
    write('S3_cov_check.tex', "\n".join(a) + "\n\\bottomrule\n\\end{tabular}\n")
    G = pd.read_csv(T / 'S2_gaussian_irrev.csv'); G = G[G.kind != 'centered'].dropna(subset=['ratio_sim'])
    if 'replicate' in G: G = G[G.replicate.isna()]
    G = G[G.kind.str.contains('trailing|current')].copy(); G['o'] = G.kind.str.contains('current').astype(int)
    G = G.sort_values(['phi', 'o', 'w'])
    b = ["\\begin{tabular}{lllrrrrr}\n\\toprule\n$\\phi$ & windows & $(w_1,w_2)$ & $R$ closed/sim & $\\dot e_p$ closed/sim & $d_1$ closed/sim & KLD$_3$ & KLD$_5$\\\\\n\\midrule"]
    b += [f"{r.phi:g} & {r.kind} & ({int(r.w)},{int(r.w2)}) & {r.ratio_closed:.4f} / {r.ratio_sim:.4f} & {r.ep_closed:.3f} / {r.ep_sim:.3f} & "
          f"{r.d1_closed:.3f} / {r.d1_sim:.3f} & {r.kld_q3_D2:.3f} & {r.kld_q5_D2:.3f}\\\\" for r in G.itertuples()]
    write('S3_irrev_check.tex', "\n".join(b) + "\n\\bottomrule\n\\end{tabular}\n")
    # S9: double-well two-window observer currents (S5_summary.csv)
    W = pd.read_csv(T / 'S5_summary.csv'); W['tau_c'] = W.tau_c.astype(float)
    c = ["\\begin{tabular}{llrrrr}\n\\toprule\n$\\tau_c$ & $w$ & $\\dot e_p$ observed & Prop.~2 & latent $(x,c)$ & $R$ observed\\\\\n\\midrule"]
    c += [f"{'$\\infty$' if np.isinf(r.tau_c) else int(r.tau_c)} & {int(r.w)} & {r.obs_ep:.3f} & {r.pred_ep:.3f} & {r.lat_ep:.4f} & {r.obs_ratio:.3f}\\\\" for r in W.itertuples()]
    write('S9_doublewell_current.tex', "\n".join(c) + "\n\\bottomrule\n\\end{tabular}\n")
    # S10: block-bootstrap standard errors of Table II, blocks 20 / 60 days (E4g_bootstrap_se.csv)
    E = pd.read_csv(T / 'E4g_bootstrap_se.csv'); names = {WIN: 'window', LEV: 'leverage, return', DD: 'leverage, drawdown', 'PCA2_common_basis': 'PCA(2) state'}
    d = ["\\begin{tabular}{llrlllll}\\toprule sample & channel & pairs & SE($\\omega$) & SE($\\omega^{\\rm rev}$) & SE($\\omega^{\\rm cen}$) & SE($O$) & SE($P$)\\\\\\midrule"]
    for sample, tag in [('SPX 2000-2026', 'S\\&P 500'), ('SPX 2000-2020', 'S\\&P, to 2020'), ('BTC', 'Bitcoin'), ('ETH', 'Ether')]:
        for key, nm in names.items():
            if sample != 'SPX 2000-2026' and key == DD: continue
            g = E[(E['sample'] == sample) & (E.channel == key)].set_index('block')
            cell = lambda q: f"{1e4 * g.loc[20, q]:.0f} / {1e4 * g.loc[60, q]:.0f}"
            d.append(f"{tag} & {nm} & {int(g.loc[20, 'n_pairs'])} & " + " & ".join(cell(q) for q in ['se_fwd', 'se_rev', 'se_cen', 'se_O', 'se_P']) + "\\\\")
    write('S10_bootstrap.tex', "\n".join(d) + "\n\\bottomrule\\end{tabular}\n")
    # S11: what the market windows cost in information (E9_market_information.csv)
    I = pd.read_csv(T / 'E9_market_information.csv')
    e = ["\\begin{tabular}{lllr}\\toprule observer & method & $\\lambda w$ & $\\ell$\\\\\\midrule"]
    for r in I.itertuples():
        e.append(f"{r.observer} & {'Lemma 1 (noise-free windows of $x$)' if r.method.startswith('exact') else 'simulation, regression'} & "
                 f"{'' if np.isnan(r.lam_w) else f'{r.lam_w:.2f}'} & {r.ell:.3f}\\\\")
    write('S11_market_information.tex', "\n".join(e) + "\n\\bottomrule\\end{tabular}\n")
    # S12: data files behind every figure and table of the main text
    F = [('Fig.~\\Mref{fig:decomp}', 'data/spx\\_2000\\_2026\\_r1.csv'),
         ('Fig.~\\Mref{fig:gauss}', 'S1\\_gaussian\\_leak.csv, T2c\\_observer\\_curves.csv, S2\\_gaussian\\_irrev.csv'),
         ('Fig.~\\Mref{fig:dw}', 'S4d\\_acf.csv, S4e\\_two\\_rate\\_leak.csv, S5\\_summary.csv'),
         ('Fig.~\\Mref{fig:market}', 'E4e\\_pairs.csv, E4c\\_pca\\_factorization.csv, E4e\\_surrogate\\_bands.csv, E4e\\_symmetric\\_sign.csv, E4f\\_sv\\_variants.csv'),
         ('Table~\\Mref{tab:gauss}', 'T2d\\_table1.csv (script T2d\\_table1\\_horizon.py)'),
         ('Table~\\Mref{tab:markets}', 'E4e\\_pairs.csv, E4c\\_pca\\_factorization.csv, E4e\\_surrogate\\_bands.csv, E4g\\_bootstrap\\_se.csv'),
         ('Table~\\Mref{tab:sv}', 'E4f\\_sv\\_variants.csv, E4c\\_pca\\_factorization.csv, E4e\\_pairs.csv'),
         ('numbers in the text', 'numbers.tex, written by P1\\_numbers.py with sources in P1\\_numbers\\_sources.csv')]
    write('S12_data_files.tex', "\\begin{tabular}{lp{12.5cm}}\\toprule item & files in \\texttt{reproducibility\\_pre/tables\\_pre/} (unless stated)\\\\\\midrule\n"
          + "\n".join(f"{a} & {b}\\\\" for a, b in F) + "\n\\bottomrule\\end{tabular}\n")


if __name__ == '__main__':
    tab1(); tab2(); tab3(); supp(); supp_more()
    print('tables written to', OUT)
