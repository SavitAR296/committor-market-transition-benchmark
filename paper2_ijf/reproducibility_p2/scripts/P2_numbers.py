"""P2_numbers -- every number quoted in the manuscript text, as a LaTeX macro (../manuscript_p2/numbers.tex), with its
source table and selection (../manuscript_p2/numbers_sources.csv).  The manuscript types no result number by hand;
a number that is not produced here cannot appear in the text.  No randomness (reads tables_p2/ only).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import P2, TAB

MS = P2.parent / 'manuscript_p2'; MS.mkdir(exist_ok=True)
rows = []


def fx(v, nd):
    return f'\\ensuremath{{{v:.{nd}f}}}'


def pv(v):
    if v < 1e-3:
        m, e = f'{v:.0e}'.split('e'); return f'\\ensuremath{{{m}\\times10^{{{int(e)}}}}}'
    return fx(v, 4 if v < 0.01 else (3 if v < 0.1 else 2))


def add(name, text, raw, source, selection):
    rows.append(dict(macro=name, latex=text, value=raw, source=source, selection=selection))


def date(s):
    d = pd.Timestamp(s); return f'{d.day} {d.strftime("%B %Y")}'


def main():
    F4 = pd.read_csv(TAB / 'P2_F4_summary.csv')
    f4 = lambda w, h, s, c='pc2_gbt': F4[(F4.window == w) & (F4.horizon == h) & (F4['sample'] == s) & (F4.comparator == c)].iloc[0]
    a = f4('exp', 5, '2008-2026')
    add('nDaysFive', f'{int(a.n):,}', a.n, 'P2_F4_summary.csv', 'exp, h=5, 2008-2026, n')
    add('nEventsFive', f'{int(a.events):,}', a.events, 'P2_F4_summary.csv', 'exp, h=5, 2008-2026, events')
    add('EpisFive', str(int(a.episodes)), a.episodes, 'P2_F4_summary.csv', 'exp, h=5, 2008-2026, episodes')
    b = f4('exp', 20, '2008-2026'); add('EpisTwenty', str(int(b.episodes)), b.episodes, 'P2_F4_summary.csv', 'exp, h=20, 2008-2026, episodes')
    X1T = pd.read_csv(TAB / 'P2_X1_tests.csv'); era = X1T[(X1T.market == 'SPX1950') & ~X1T['sample'].str.contains('without')]
    e5, e20 = int(era[era.horizon == 5].episodes.iloc[0]), int(era[era.horizon == 20].episodes.iloc[0])
    add('EpisEraFive', str(e5), e5, 'P2_X1_tests.csv', 'SPX1950 1957-1999, h=5, episodes')
    add('EpisEraTwenty', str(e20), e20, 'P2_X1_tests.csv', 'SPX1950 1957-1999, h=20, episodes')
    spx = [int(f4('exp', h, s).episodes) for h in (5, 20) for s in ('2008-2026', '2009-2026')] + [e5, e20]
    allx = X1T[~X1T['sample'].str.contains('without')]
    alle = spx + [int(allx[(allx.market == m) & (allx.horizon == h)].episodes.iloc[0]) for m in ('BTC', 'ETH') for h in (5, 20)]
    add('EpisMin', str(min(alle)), min(alle), 'P2_F4_summary.csv; P2_X1_tests.csv', 'min event episodes over all samples (S&P, Bitcoin, Ether)')
    add('EpisMax', str(max(alle)), max(alle), 'P2_F4_summary.csv; P2_X1_tests.csv', 'max event episodes over all samples')
    for m, w in [('SPX1950', 'Era'), ('BTC', 'BTC'), ('ETH', 'ETH')]:
        v = int(allx[allx.market == m]['sample'].iloc[0][:4]); add(f'FirstOOS{w}', str(v), v, 'P2_X1_tests.csv', f'{m}: first out-of-sample year')
    main_s = F4['sample'].isin(['2008-2026', '2009-2026'])
    v = F4[(F4.window == 'exp') & main_s].p_EWC_holm.min(); add('HolmMinExp', pv(v), v, 'P2_F4_summary.csv', 'exp, main samples, min Holm-adjusted EWC p')
    M = pd.read_csv(TAB / 'P2_F4_mcs.csv'); ref = M[M.method == 'ou_seqcal']
    add('MCSfam', str(int((ref.mcs_p >= 0.10).sum())), int((ref.mcs_p >= 0.10).sum()), 'P2_F4_mcs.csv', 'families with ou_seqcal in the 90% MCS')
    v = F4[(F4.window == 'exp') & (F4.horizon == 20) & main_s & F4.comparator.isin(['msm_seqcal', 'amc_seqcal'])].p_GW_NW.max()
    add('GWmax', pv(v), v, 'P2_F4_summary.csv', 'exp, h=20, main samples, max GW (NW) p vs MSM/AMC')
    v = f4('roll5', 20, '2009-2026', 'raw_logistic').p_EWC_holm; add('HolmRoll', pv(v), v, 'P2_F4_summary.csv', 'roll5, h=20, 2009-2026, raw logistic, Holm p')
    SP = pd.read_csv(TAB / 'P2_F6_spearman_exp.csv')
    for h, w in [(5, 'Five'), (20, 'Twenty')]:
        v = SP[SP.horizon == h].spearman_S_s.median(); add(f'Spearman{w}', fx(v, 3), v, 'P2_F6_spearman_exp.csv', f'h={h}, median over years')
    OW = pd.read_csv(TAB / 'P2_F6_own_target_exp.csv'); OW = OW[OW.scope == '2008-2026']
    for h, w in [(5, 'Five'), (20, 'Twenty')]:
        for t, k in [('own', 'own'), ('oper', 'operational')]:
            v = OW[(OW.horizon == h) & OW.target.str.startswith(k)].MCB.iloc[0]
            add(f'MCB{t}{w}', fx(v, 4), v, 'P2_F6_own_target_exp.csv', f'h={h}, 2008-2026, target {k}, MCB')
    D2 = pd.read_csv(TAB / 'P2_F2_decomposition_exp.csv'); D2 = D2[(D2.set == 'common7') & (D2['sample'] == '2008-2026')]
    g = lambda h, m, c: float(D2[(D2.horizon == h) & (D2.method == m)][c].iloc[0])
    add('MCBrawFive', fx(g(5, 'ou_raw', 'MCB'), 3), g(5, 'ou_raw', 'MCB'), 'P2_F2_decomposition_exp.csv', 'common7, 2008-2026, h=5, ou_raw, MCB')
    add('DSCrawTwenty', fx(g(20, 'ou_raw', 'DSC'), 4), g(20, 'ou_raw', 'DSC'), 'P2_F2_decomposition_exp.csv', 'common7, 2008-2026, h=20, ou_raw, DSC')
    add('DSCseqTwenty', fx(g(20, 'ou_seqcal', 'DSC'), 4), g(20, 'ou_seqcal', 'DSC'), 'P2_F2_decomposition_exp.csv', 'common7, 2008-2026, h=20, ou_seqcal, DSC')
    B2 = pd.read_csv(TAB / 'P2_F2_bootstrap_exp.csv'); bb = B2[(B2.horizon == 20) & (B2['sample'] == '2008-2026') & (B2.method == 'ou_seqcal')].iloc[0]
    add('dDSCseqLo', fx(bb.dDSC_rawS_minus_m_q05, 4), bb.dDSC_rawS_minus_m_q05, 'P2_F2_bootstrap_exp.csv', 'h=20, 2008-2026, raw S - seq-cal, q05')
    add('dDSCseqHi', fx(bb.dDSC_rawS_minus_m_q95, 4), bb.dDSC_rawS_minus_m_q95, 'P2_F2_bootstrap_exp.csv', 'h=20, 2008-2026, raw S - seq-cal, q95')
    H3 = pd.read_csv(TAB / 'P2_F3_homogeneity_exp.csv'); H3r = pd.read_csv(TAB / 'P2_F3_homogeneity_roll5.csv')
    v = H3[(H3.horizon == 5) & ~H3.valid.astype(bool)].p_value.iloc[0]; add('ChiFive', pv(v), v, 'P2_F3_homogeneity_exp.csv', 'h=5, day-level chi-square (invalid)')
    sel = lambda H, h, t, c: H[(H.horizon == h) & (H.test == t) & (H.clusters == c)].iloc[0]
    v = sel(H3, 5, 'cluster permutation', 'episodes, gap h').p_value; add('PermFive', fx(v, 2), v, 'P2_F3_homogeneity_exp.csv', 'h=5, episode permutation (gap h)')
    v = min(H3[H3.valid.astype(bool)].p_value.min(), H3r[H3r.valid.astype(bool)].p_value.min())
    add('PermMin', fx(v, 2), v, 'P2_F3_homogeneity_exp.csv; P2_F3_homogeneity_roll5.csv', 'min p over all valid cluster tests')
    v = sel(H3, 5, 'quasi-binomial F', 'episodes, gap h').dispersion; add('DispFive', fx(v, 1), v, 'P2_F3_homogeneity_exp.csv', 'h=5, quasi-binomial dispersion (gap h)')
    C3 = pd.read_csv(TAB / 'P2_F3_calibrators_exp.csv'); c3 = C3[(C3.horizon == 20) & (C3['sample'] == '2008-2026')].set_index('calibrator').BSS
    add('IsoTwenty', fx(c3['isotonic on S'], 3), c3['isotonic on S'], 'P2_F3_calibrators_exp.csv', 'h=20, 2008-2026, isotonic, BSS')
    add('SeqTwenty', fx(c3['logistic on S (frozen seq-cal)'], 3), c3['logistic on S (frozen seq-cal)'], 'P2_F3_calibrators_exp.csv', 'h=20, 2008-2026, frozen seq-cal, BSS')
    CF = pd.read_csv(TAB / 'P2_F5_case_facts.csv', dtype=str); cf = lambda c, f: CF[(CF.case == c) & (CF.fact == f)].value.iloc[0]
    for c, w in [('2011', 'Eleven'), ('2020', 'TwentyTwenty'), ('2025', 'TwentyFive')]:
        v = int(cf(c, 'lead of that alarm (trading days)')); add(f'Lead{w}', str(v), v, 'P2_F5_case_facts.csv', f'{c}, lead of first S5>0.99')
    v = int(cf('2011', 'saturated days S5>0.99 outside the set')); add('SatEleven', str(v), v, 'P2_F5_case_facts.csv', '2011 window, saturated days')
    v = int(cf('2011', 'saturated days followed by an entry within 5 days')); add('SatElevenHit', str(v), v, 'P2_F5_case_facts.csv', '2011 window, followed by entry')
    s = cf('2011', 'first S5>0.99 in the 20 days before'); add('AlarmElevenDate', date(s), s, 'P2_F5_case_facts.csv', '2011, first S5>0.99 before entry')
    s = cf('2011', 'first entry'); add('EntryElevenDate', date(s), s, 'P2_F5_case_facts.csv', '2011, first entry')
    XM = pd.read_csv(TAB / 'P2_X1_metrics.csv')
    v = XM[(XM.market == 'SPX1950') & ~XM['sample'].str.contains('without') & (XM.horizon == 5) & (XM.method == 'ou_raw')].BSS.iloc[0]
    add('BSSrawEraFive', fx(v, 1), v, 'P2_X1_metrics.csv', 'SPX1950 1957-1999, h=5, ou_raw, BSS')
    XH = pd.read_csv(TAB / 'P2_X1_homogeneity.csv'); xh = XH[(XH.market == 'SPX1950') & (XH.horizon == 5)].iloc[0]
    add('PermEraFive', fx(xh.p_cluster_permutation, 2), xh.p_cluster_permutation, 'P2_X1_homogeneity.csv', 'SPX1950, h=5, episode permutation')
    add('QBEraFive', pv(xh.p_quasibinomial), xh.p_quasibinomial, 'P2_X1_homogeneity.csv', 'SPX1950, h=5, quasi-binomial F')
    from p2_common import FULL
    Fq = pd.read_csv(TAB / 'features_SPX_r1.csv', parse_dates=['date']).dropna(subset=FULL).reset_index(drop=True)
    und = Fq.rv20.rolling(1000, min_periods=500).quantile(0.85).shift(1).isna()
    add('LabelRows', str(int(und.sum())), int(und.sum()), 'features_SPX_r1.csv', 'complete rows with undefined lagged RV quantile')
    add('LabelFromDate', date(Fq.date[und].min()), str(Fq.date[und].min().date()), 'features_SPX_r1.csv', 'first such row')
    add('LabelToDate', date(Fq.date[und].max()), str(Fq.date[und].max().date()), 'features_SPX_r1.csv', 'last such row')

    # ---- additions of the revision (outline 7: A4-A8, B2-B10)
    o5 = OW[OW.horizon == 5]; own5 = o5[o5.target.str.startswith('own')].iloc[0]; op5 = o5[o5.target.str.startswith('operational')].iloc[0]
    add('NoutB', f'{int(own5.n):,}', own5.n, 'P2_F6_own_target_exp.csv', 'h=5, days outside the learned region')
    add('MeanSoutB', fx(own5.mean_S, 3), own5.mean_S, 'P2_F6_own_target_exp.csv', 'h=5, mean S outside B')
    add('RateOperOutB', fx(op5.event_rate, 3), op5.event_rate, 'P2_F6_own_target_exp.csv', 'h=5, operational event rate outside B')
    add('DSCownFive', fx(own5.DSC, 4), own5.DSC, 'P2_F6_own_target_exp.csv', 'h=5, DSC for own target')
    add('DSCoperFive', fx(op5.DSC, 4), op5.DSC, 'P2_F6_own_target_exp.csv', 'h=5, DSC for operational event')
    B2b = pd.read_csv(TAB / 'P2_F2_bootstrap_exp.csv')
    xh20 = XH[(XH.market == 'SPX1950') & (XH.horizon == 20)].iloc[0]
    add('QBEraTwenty', pv(xh20.p_quasibinomial), xh20.p_quasibinomial, 'P2_X1_homogeneity.csv', 'SPX1950, h=20, quasi-binomial F')
    add('PermEraTwenty', fx(xh20.p_cluster_permutation, 2), xh20.p_cluster_permutation, 'P2_X1_homogeneity.csv', 'SPX1950, h=20, permutation')
    for m, w in [('pc2_logistic', 'Log'), ('pc2_gbt', 'Gbt')]:
        r = B2b[(B2.horizon == 20) & (B2['sample'] == '2008-2026') & (B2.method == m)].iloc[0]
        add(f'dDSC{w}Lo', fx(r.dDSC_rawS_minus_m_q05, 4), r.dDSC_rawS_minus_m_q05, 'P2_F2_bootstrap_exp.csv', f'h=20, raw S - {m}, q05')
        add(f'dDSC{w}Hi', fx(r.dDSC_rawS_minus_m_q95, 4), r.dDSC_rawS_minus_m_q95, 'P2_F2_bootstrap_exp.csv', f'h=20, raw S - {m}, q95')
    DB = pd.read_csv(TAB / 'P2_F4_dbss.csv'); g = DB[(DB.window == 'exp') & (DB.comparator == 'pc2_gbt') & (DB['sample'] == '2008-2026')]
    z20 = g[(g.horizon == 20) & (g.lo90 <= 0)].block; z5 = g[(g.horizon == 5) & (g.lo90 <= 0)].block
    add('GbtZeroBlockTwenty', str(int(z20.min())) if len(z20) else 'none', z20.min() if len(z20) else np.nan, 'P2_F4_dbss.csv', 'h=20, smallest block whose GBT interval includes 0')
    add('GbtZeroBlockFive', str(int(z5.min())) if len(z5) else 'none', z5.min() if len(z5) else np.nan, 'P2_F4_dbss.csv', 'h=5, smallest block whose GBT interval includes 0')
    BS = pd.read_csv(TAB / 'P2_B2_stress_index.csv')
    for h, w in [(5, 'Five'), (20, 'Twenty')]:
        q = BS[(BS.horizon == h) & (BS['sample'] == '2008-2026')]
        for m, k in [('s_seqcal', 's'), ('ou_seqcal', 'ou')]:
            v = q[q.method == m].BSS.iloc[0]; add(f'BSS{k}{w}', fx(v, 3), v, 'P2_B2_stress_index.csv', f'h={h}, 2008-2026, {m} BSS')
        d = q[q.method.str.startswith('dBSS')].iloc[0]
        add(f'dBSSs{w}', fx(d.BSS, 3), d.BSS, 'P2_B2_stress_index.csv', f'h={h}, dBSS s seq-cal - OU seq-cal')
        add(f'dBSSs{w}Lo', fx(d.lo90, 3), d.lo90, 'P2_B2_stress_index.csv', f'h={h}, q05'); add(f'dBSSs{w}Hi', fx(d.hi90, 3), d.hi90, 'P2_B2_stress_index.csv', f'h={h}, q95')
        add(f'pEWCs{w}', pv(d.p_EWC), d.p_EWC, 'P2_B2_stress_index.csv', f'h={h}, EWC p')
    G5 = pd.read_csv(TAB / 'P2_B5_gw_split.csv'); ge = G5[G5.window == 'exp']
    add('GWslopeMin', pv(ge.p_slope.min()), ge.p_slope.min(), 'P2_B5_gw_split.csv', 'exp, h=20, min p of the lagged term (MSM, AMC)')
    add('GWslopeMax', pv(ge.p_slope.max()), ge.p_slope.max(), 'P2_B5_gw_split.csv', 'exp, h=20, max p of the lagged term')
    add('GWconstMax', pv(ge.p_const.max()), ge.p_const.max(), 'P2_B5_gw_split.csv', 'exp, h=20, max p of the constant')
    FL = pd.read_csv(TAB / 'P2_B5_fluctuation.csv'); add('GRcrit', fx(FL.k95.iloc[0], 2), FL.k95.iloc[0], 'P2_B5_fluctuation.csv', 'two-sided 5% critical value, mu = 0.3')
    add('GRreject', str(int(FL.reject.sum())), int(FL.reject.sum()), 'P2_B5_fluctuation.csv', 'number of rejections of 8')
    add('GRmaxExp', fx(FL[FL.window == 'exp'].max_abs.max(), 2), FL[FL.window == 'exp'].max_abs.max(), 'P2_B5_fluctuation.csv', 'exp, max |F_t|')
    S1 = pd.read_csv(TAB / 'P2_S1_size_simulation.csv')
    for h, w in [(5, 'Five'), (20, 'Twenty')]:
        for t, k in [('day-level chi-square', 'Chi'), ('episode permutation', 'Perm'), ('quasi-binomial F', 'QB')]:
            for dl, kd in [(0.0, 'Size'), (1.0, 'Pow')]:
                v = S1[(S1.horizon == h) & (S1.test == t) & (S1.delta == dl)].rejection_rate.iloc[0]
                add(f'{kd}{k}{w}', f'{100 * v:.1f}\\%', v, 'P2_S1_size_simulation.csv', f'h={h}, delta={dl}, {t}, rejection rate (percent)')
    add('SimYearsFive', str(int(S1[S1.horizon == 5].target_years.iloc[0])), 0, 'P2_S1_size_simulation.csv', 'h=5, model years')
    add('SimEpisFive', str(int(S1[S1.horizon == 5].target_episodes.iloc[0])), 0, 'P2_S1_size_simulation.csv', 'h=5, saturation episodes')
    DP = pd.read_csv(TAB / 'P2_F3_drift_penalty_exp.csv'); DP = DP[DP['sample'] == '2008-2026']
    for h, w in [(5, 'Five'), (20, 'Twenty')]:
        q = DP[DP.horizon == h].set_index('method').penalty; oth = q.drop(['ou_raw', 'clim'])
        add(f'PenRaw{w}', fx(q['ou_raw'], 4), q['ou_raw'], 'P2_F3_drift_penalty_exp.csv', f'h={h}, raw S'); add(f'PenClim{w}', fx(q['clim'], 4), q['clim'], 'P2_F3_drift_penalty_exp.csv', f'h={h}, climatology')
        add(f'PenOthLo{w}', fx(oth.min(), 4), oth.min(), 'P2_F3_drift_penalty_exp.csv', f'h={h}, other forecasts, min'); add(f'PenOthHi{w}', fx(oth.max(), 4), oth.max(), 'P2_F3_drift_penalty_exp.csv', f'h={h}, other forecasts, max')
    AU = pd.read_csv(TAB / 'P2_F2_annual_auc_exp.csv'); AU = AU[~AU.constant.astype(bool) & AU.AUC.notna()]
    for h, w in [(5, 'Five'), (20, 'Twenty')]:
        for m, k in [('ou_raw', 'Raw'), ('pc2_gbt', 'Gbt'), ('pc2_logistic', 'Log')]:
            v = AU[(AU.horizon == h) & (AU.method == m)].AUC.median(); add(f'AUCyr{k}{w}', fx(v, 2), v, 'P2_F2_annual_auc_exp.csv', f'h={h}, {m}, median annual AUC')
    D2b = pd.read_csv(TAB / 'P2_F2_decomposition_exp.csv'); D2b = D2b[(D2b.set == 'common7') & D2b['sample'].isin(['2008-2026', '2009-2026'])]
    pairs = [('pc2_logistic', ['pc2_logistic_C0.1', 'pc2_logistic_C10']), ('raw_logistic', ['raw_logistic_C0.1', 'raw_logistic_C1', 'raw_logistic_C10']),
             ('pc2_gbt', ['pc2_gbt_cv']), ('raw_gbt', ['raw_gbt_cv'])]
    ch = [abs(float(D2b[(D2b.horizon == h) & (D2b['sample'] == sm) & (D2b.method == a)].BSS.iloc[0]) - float(D2b[(D2b.horizon == h) & (D2b['sample'] == sm) & (D2b.method == b)].BSS.iloc[0]))
          for h in (5, 20) for sm in ('2008-2026', '2009-2026') for base, alts in pairs for a in [base] for b in alts]
    add('SetMaxChange', fx(max(ch), 3), max(ch), 'P2_F2_decomposition_exp.csv', 'max |BSS change| over classifier settings')
    CF2 = CF
    for c, w in [('2011', 'Eleven'), ('2020', 'TwentyTwenty'), ('2025', 'TwentyFive')]:
        clim = float(cf(c, 'training climatology (h=5)')); sq = float(cf(c, 'max seq-cal (h=5) in the 20 days before')); gb = float(cf(c, 'max PC2 GBT (h=5) in the 20 days before'))
        add(f'Clim{w}', fx(clim, 3), clim, 'P2_F5_case_facts.csv', f'{c}, training climatology'); add(f'SeqMax{w}', fx(sq, 2), sq, 'P2_F5_case_facts.csv', f'{c}, max seq-cal before entry')
        add(f'SeqMult{w}', fx(sq / clim, 0), sq / clim, 'P2_F5_case_facts.csv', f'{c}, seq-cal max / climatology'); add(f'GbtMax{w}', fx(gb, 2), gb, 'P2_F5_case_facts.csv', f'{c}, max GBT before entry')
        add(f'GbtMult{w}', fx(gb / clim, 0), gb / clim, 'P2_F5_case_facts.csv', f'{c}, GBT max / climatology')
        if c != '2011':
            add(f'Sat{w}', str(int(cf(c, 'saturated days S5>0.99 outside the set'))), 0, 'P2_F5_case_facts.csv', f'{c}, saturated days')
            add(f'Sat{w}Hit', str(int(cf(c, 'saturated days followed by an entry within 5 days'))), 0, 'P2_F5_case_facts.csv', f'{c}, followed by entry')
            add(f'Alarm{w}Date', date(cf(c, 'first S5>0.99 in the 20 days before')), 0, 'P2_F5_case_facts.csv', f'{c}, first alarm')
            add(f'Entry{w}Date', date(cf(c, 'first entry')), 0, 'P2_F5_case_facts.csv', f'{c}, first entry')
        add(f'Entries{w}', str(int(cf(c, 'entries'))), 0, 'P2_F5_case_facts.csv', f'{c}, number of entries')
    R = pd.DataFrame(rows); R.to_csv(MS / 'numbers_sources.csv', index=False)
    (MS / 'numbers.tex').write_text('% generated by reproducibility_p2/scripts/P2_numbers.py -- do not edit\n' +
                                    ''.join(f'\\newcommand{{\\{r.macro}}}{{{r.latex}\\xspace}}\n' for r in R.itertuples()))
    print(R[['macro', 'latex', 'source']].to_string(index=False))


if __name__ == '__main__':
    main()
