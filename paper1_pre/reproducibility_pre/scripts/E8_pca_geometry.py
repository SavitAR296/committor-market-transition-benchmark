"""E8 -- geometry of the annual common-7 PCA state (loadings by year; 2008-2020 checked against the frozen file).
Outputs: E8_pca_table.csv (wide, Supplemental Table S6), E8_pca_facts.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; LEG = ROOT.parent / 'reproducibility'
sys.path.insert(0, str(ROOT / 'src'))
from pre_common import write_csv


def main():
    L = pd.read_csv(ROOT / 'tables_pre' / 'E3_pca_loadings_2026.csv'); L = L[L.set == 'common7']
    Z = pd.read_csv(LEG / 'tables_csf' / 'B_pca_loadings.csv'); Z = Z[Z.set == 'common7']
    mg = Z.merge(L, on=['year', 'set', 'pc', 'feature'], suffixes=('_f', '_n'))
    maxdiff = float((mg.loading_f - mg.loading_n).abs().max())
    W = L.pivot_table(index='year', columns=['pc', 'feature'], values='loading')
    W.columns = [f'PC{p}_{f}' for p, f in W.columns]
    ex = L.groupby(['year', 'pc']).explained.first().unstack(); W['PC1_explained'] = ex[1]; W['PC2_explained'] = ex[2]
    W = W.reset_index(); write_csv(W, 'E8_pca_table.csv')
    def rng(col, ys):
        v = W[W.year.isin(ys)][col]; return f'{v.min():+.3f}..{v.max():+.3f}'
    yrs = list(W.year); post = [y for y in yrs if y >= 2009]
    facts = [dict(fact='max |loading diff| vs frozen 2008-2020', value=maxdiff)]
    for c in ['dd20', 'ret20', 'log_rv20', 'log_rv5', 'log_act_rel', 'ews_var', 'ews_ar1']:
        facts.append(dict(fact=f'PC1 {c} 2009-2026', value=rng(f'PC1_{c}', post)))
    facts.append(dict(fact='PC1 explained 2009-2026', value=rng('PC1_explained', post)))
    facts.append(dict(fact='PC1 explained 2008', value=float(W[W.year == 2008].PC1_explained.iloc[0])))
    for per, ys in [('2009-2017', range(2009, 2018)), ('2018-2026', range(2018, 2027))]:
        for c in ['ews_var', 'ews_ar1', 'log_act_rel']:
            facts.append(dict(fact=f'PC2 {c} {per}', value=rng(f'PC2_{c}', list(ys))))
    write_csv(pd.DataFrame(facts), 'E8_pca_facts.csv'); print(pd.DataFrame(facts).to_string(index=False))
    print('E8 complete'); (ROOT / 'tables_pre' / '_cache' / 'DONE_E8_pca_geometry').touch()


if __name__ == '__main__':
    main()
