"""P2_X1 -- the frozen protocol on data it has never seen: S&P 500 1950-1999 (independent Yahoo-derived archive, an
era untouched by the design of the protocol), Bitcoin and Ether (Coin Metrics).

Features: frozen data_daily.build_features (trailing windows; no VIX, so only the common-7 state exists).  Labels:
frozen crisis_labels (RV20 above its lagged 1000-day 85% quantile and 20-day drawdown below -8% for equities, -15% for
crypto).  Sample: complete common-7 rows; the first rows before the label rule is defined (the lagged RV quantile needs
500 rows) are dropped.  First out-of-sample year, fixed before any result was seen: the first calendar year with at
least 1,000 training rows.  Methods (common-7, annual expanding windows): clim, pc2/raw logistic, pc2/raw GBT, ou_raw,
msm_raw, amc_raw and the three *_seqcal (GBT-CV and the logistic C sensitivity of the headline are omitted).
Evaluation per market, horizons 5 and 20, all OOS years and without the first OOS year (the cold-start year, the role
2008 plays in the headline): BSS and AUC (raw scores and classifiers only); CORP MCB/DSC with paired block-bootstrap
intervals of DSC(raw S) - DSC(m); dBSS of each comparator against ou_seqcal with block-bootstrap intervals, EWC and GW
tests, Holm; homogeneity of the saturated-day event rate across years by cluster permutation and quasi-binomial F.
Usage: python scripts/P2_X1_extension.py            (forecast units, resumable; P2_BUDGET seconds per call)
       python scripts/P2_X1_extension.py --evaluate
Outputs: features_X1_<market>.csv, P2_X1_forecasts_<market>.csv, P2_X1_metrics.csv, P2_X1_decomposition.csv,
P2_X1_tests.csv, P2_X1_homogeneity.csv.  Seeds: scikit-learn/deeptime 0 (frozen); bootstrap/permutation 20261025.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src')); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_common import (TAB, CACHE, SPX_ARCHIVE, COMMON, MAIN_METHODS, auc_reported, auc, bss, corp, mbb_index, boot_block, dm_tests,
                       gw_conditional, holm, clusters_from_positions, cluster_permutation_homogeneity, quasibinomial_F,
                       write_csv)
from data_daily import build_features, crisis_labels, load_market                 # frozen
from B_csf_additions import fit_ou_scores, safe_prob_logistic, safe_gbt           # frozen
from p2_e3_frozen import target_direction, msm_scores, amc_scores                 # verbatim copies
from P2_F1_forecasts import seqcal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MARKETS = ['SPX1950', 'BTC', 'ETH']
MIN_TRAIN = 1000
SEED = 20261025
REF = 'ou_seqcal'; CMP = ['msm_seqcal', 'amc_seqcal', 'pc2_logistic', 'pc2_gbt', 'raw_logistic', 'raw_gbt']
T0 = time.time(); BUDGET = float(os.environ.get('P2_BUDGET', '1e9'))


def frame(market):
    if market == 'SPX1950':
        d = pd.read_csv(SPX_ARCHIVE); d['date'] = pd.to_datetime(d['Date']); d = d[d.date < '2000-01-01']
        df = pd.DataFrame(dict(date=d.date.values, close=d.Close.astype(float).values, high=d.High.astype(float).values,
                               low=d.Low.astype(float).values, volume=d.Volume.astype(float).values))
        df['activity'] = df.volume; df['ann'] = 252.0
        return df
    return load_market(market)                                     # frozen loader (ann = 365)


def market_data(market):
    f = TAB / f'features_X1_{market}.csv'
    if not f.exists():
        with np.errstate(divide='ignore', invalid='ignore'):
            build_features(frame(market)).to_csv(f, index=False)
    F = pd.read_csv(f, parse_dates=['date']).dropna(subset=COMMON).reset_index(drop=True)
    labels = {h: crisis_labels(F, h, rule='rv_dd')[0] for h in [5, 20]}
    _, inB, _ = crisis_labels(F, 20, rule='rv_dd'); F['inB'] = inB
    keep = F.rv20.rolling(1000, min_periods=500).quantile(0.85).shift(1).notna().values
    F = F[keep].reset_index(drop=True); labels = {h: v[keep] for h, v in labels.items()}
    years = sorted(F.date.dt.year.unique())
    first = next(Y for Y in years if (F.date < pd.Timestamp(f'{Y}-01-01')).sum() >= MIN_TRAIN)
    return F, labels, [Y for Y in years if Y >= first]


def year_unit(F, labels, Y):
    trmask = F.date < pd.Timestamp(f'{Y}-01-01'); temask = F.date.dt.year == Y
    tr = F[trmask]; te = F[temask]; tridx = np.where(trmask)[0]; teidx = np.where(temask)[0]
    scaler = StandardScaler().fit(tr[COMMON]); Xtr = scaler.transform(tr[COMMON]); Xte = scaler.transform(te[COMMON])
    pca = PCA(n_components=2, random_state=0).fit(Xtr); Ztr = pca.transform(Xtr); Zte = pca.transform(Xte)
    zsc = StandardScaler().fit(Ztr); Ztr = zsc.transform(Ztr); Zte = zsc.transform(Zte)
    ou, _, _ = fit_ou_scores(Ztr, Zte, tr['log_rv20'].values)
    w, thr = target_direction(Ztr, tr['log_rv20'].values)
    msm = msm_scores(Ztr, Zte, w, thr, (5, 20)); amc = amc_scores(Ztr, Zte, w, thr, (5, 20)); rows = []
    for h in [5, 20]:
        ytr = labels[h][tridx].copy(); yte = labels[h][teidx].copy(); oktr = np.isfinite(ytr); oktr[-h:] = False
        clim = float(np.nanmean(ytr[oktr])) if np.any(oktr) else 0.01; yi = ytr[oktr].astype(int)
        M = {'clim': np.full(len(te), clim), 'pc2_logistic': safe_prob_logistic(Ztr[oktr], yi, Zte, C=1.0),
             'pc2_gbt': safe_gbt(Ztr[oktr], yi, Zte), 'raw_logistic': safe_prob_logistic(Xtr[oktr], yi, Xte, C=0.2),
             'raw_gbt': safe_gbt(Xtr[oktr], yi, Xte), 'ou_raw': ou[h][1], 'msm_raw': msm[h][1], 'amc_raw': amc[h][1]}
        ok = np.isfinite(yte) & (~te.inB.values.astype(bool))
        for m, p in M.items():
            rows.append(pd.DataFrame(dict(date=te.date.values, year=Y, set='common7', horizon=h, method=m, p=p, y=yte,
                                          headline=ok, inB=te.inB.values, clim=clim)))
    return pd.concat(rows, ignore_index=True)


def wide(F, h):
    g = F[F.horizon == h]
    P = g.pivot_table(index='date', columns='method', values='p', aggfunc='first')
    meta = g.groupby('date').agg(y=('y', 'first'), cref=('clim', 'first'), year=('year', 'first'))
    return meta.join(P).sort_index()


def evaluate():
    rng = np.random.default_rng(SEED); met, dec, tst, hom = [], [], [], []
    for market in MARKETS:
        F, labels, years = market_data(market)
        posmap = pd.Series(np.arange(len(F)), index=F.date)
        Fc = pd.read_csv(TAB / f'P2_X1_forecasts_{market}.csv', parse_dates=['date'])
        Fc = Fc[Fc.headline.astype(bool) & Fc.y.notna() & Fc.p.notna()]
        for h in [5, 20]:
            W0 = wide(Fc, h)
            for sname, W in [(f'{years[0]}-{years[-1]}', W0), (f'{years[1]}-{years[-1]} (without first OOS year)', W0[W0.year > years[0]])]:
                y = W.y.values; base = (y - W.cref.values) ** 2; n = len(y)
                ev = posmap.loc[W.index[y == 1]].values; eps = int(clusters_from_positions(ev, h).max() + 1) if len(ev) else 0
                key = dict(market=market, horizon=h, sample=sname, n=n, events=int(y.sum()), episodes=eps)
                for m in [c for c in W.columns if c not in ('y', 'cref', 'year')]:
                    met.append(dict(**key, method=m, BSS=bss(y, W[m], W.cref), AUC=auc(y, W[m]) if auc_reported(m) else np.nan))
                mm = [m for m in MAIN_METHODS if m in W]; st = {m: [] for m in mm}
                for ix in mbb_index(n, boot_block(h), 500, rng):
                    if y[ix].sum() == 0:
                        continue
                    for m in mm:
                        st[m].append(corp(y[ix], W[m].values[ix])['DSC'])
                for m in mm:
                    c = corp(y, W[m].values); dd = np.array(st['ou_raw']) - np.array(st[m])
                    dec.append(dict(**key, method=m, MCB=c['MCB'], DSC=c['DSC'], UNC=c['UNC'],
                                    dDSC_rawS_minus_m_q05=np.quantile(dd, .05), dDSC_rawS_minus_m_q95=np.quantile(dd, .95)))
                ix = mbb_index(n, boot_block(h), 1000, rng); fam = []
                for cm in CMP:
                    d = (y - W[REF].values) ** 2 - (y - W[cm].values) ** 2; v = d[ix].mean(1) / base[ix].mean(1)
                    t = dm_tests(d, h) + gw_conditional(d, h)
                    fam.append(dict(**key, comparator=cm, dBSS=d.mean() / base.mean(), lo90=np.quantile(v, .05),
                                    hi90=np.quantile(v, .95), p_EWC=next(r['p_value'] for r in t if r['variance'].startswith('EWC') and r['test'] == 'unconditional'),
                                    p_GW_NW=next(r['p_value'] for r in t if r['variance'].startswith('NW') and r['test'].startswith('cond'))))
                for f, a in zip(fam, holm([f['p_EWC'] for f in fam])):
                    f['p_EWC_holm'] = a
                tst += fam
            W = W0; S = W.ou_raw.values; hi = S > 0.99
            sat = pd.DataFrame(dict(y=W.y.values[hi], year=W.year.values[hi]), index=W.index[hi])
            keep = sat.groupby('year').size().pipe(lambda s: s[s >= 5].index); sat = sat[sat.year.isin(keep)]
            if sat.year.nunique() >= 2:
                pos = posmap.loc[sat.index].values; cid = clusters_from_positions(pos, h, sat.year.values)
                g = pd.DataFrame(dict(c=cid, y=sat.y.values, year=sat.year.values)).groupby('c').agg(e=('y', 'sum'), n=('y', 'size'), year=('year', 'first'))
                gi = g.year.map({Y: k for k, Y in enumerate(sorted(g.year.unique()))}).values
                stat, pp = cluster_permutation_homogeneity(g.e.values, g.n.values, gi, 9999, rng)
                Fq, pF, phi, dfr = quasibinomial_F(g.e.values, g.n.values, gi)
                hom.append(dict(market=market, horizon=h, years=len(np.unique(gi)), days=len(sat), episodes=len(g),
                                p_cluster_permutation=pp, p_quasibinomial=pF, dispersion=phi))
    write_csv(pd.DataFrame(met), 'P2_X1_metrics.csv'); write_csv(pd.DataFrame(dec), 'P2_X1_decomposition.csv')
    write_csv(pd.DataFrame(tst), 'P2_X1_tests.csv'); write_csv(pd.DataFrame(hom), 'P2_X1_homogeneity.csv')
    M = pd.DataFrame(met); print(M[M.method.isin(['ou_raw', 'ou_seqcal', 'pc2_logistic', 'pc2_gbt'])].pivot_table(index=['market', 'sample', 'method'], columns='horizon', values='BSS').round(3).to_string())
    D = pd.DataFrame(dec); print(D[D.method.isin(['ou_raw', 'ou_seqcal', 'pc2_logistic', 'pc2_gbt'])].pivot_table(index=['market', 'sample', 'method'], columns='horizon', values=['DSC', 'dDSC_rawS_minus_m_q05']).round(4).to_string())
    T = pd.DataFrame(tst); print(T.pivot_table(index=['market', 'sample', 'comparator'], columns='horizon', values=['dBSS', 'p_EWC_holm']).round(3).to_string())
    print(T.groupby(['market', 'horizon', 'sample'])[['n', 'events', 'episodes']].first().to_string())
    print(pd.DataFrame(hom).round(4).to_string(index=False)); print('X1 evaluation complete')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--evaluate', action='store_true'); a = ap.parse_args()
    if a.evaluate:
        return evaluate()
    for market in MARKETS:
        F, labels, years = market_data(market)
        print(market, 'rows', len(F), 'from', F.date.min().date(), 'OOS years', years[0], '..', years[-1], flush=True)
        for Y in years:
            f = CACHE / f'X1_{market}_{Y}.pkl'
            if f.exists():
                continue
            if time.time() - T0 > BUDGET:
                print('budget reached; rerun to continue'); return
            pd.to_pickle(year_unit(F, labels, Y), f)
        Fc = pd.concat([pd.read_pickle(CACHE / f'X1_{market}_{Y}.pkl') for Y in years], ignore_index=True)
        Fc = pd.concat([Fc, seqcal(Fc)], ignore_index=True).sort_values(['horizon', 'method', 'date'])
        Fc.to_csv(TAB / f'P2_X1_forecasts_{market}.csv', index=False, float_format='%.10g')
        print(market, 'forecasts complete', flush=True)
    print('X1 units complete')


if __name__ == '__main__':
    main()
