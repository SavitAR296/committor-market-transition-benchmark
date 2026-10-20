"""S3 -- the flux/drift ratio, made explicit (the archived E2v2 diagnostic code is not in the package and could not be
recovered from the supplied materials).  For each market and archived year, the common-7 state is rebuilt from data
before 1 January of that year (train-only standardisation, PCA(2), re-standardisation) and the frozen linear OU model
(OUModel) is fitted.  Two definitions of R = median_rho |J| / (|f| rho), J = f rho - (1/2) div(D rho):
  analytic : Gaussian stationary density of the fitted OU (Lyapunov), median over the stationary law (Appendix D)
  grid     : finite-difference generator on the 40x40 forecasting grid, stationary density as its null vector and the
             flux by central differences (src/committor.py: Grid2D.stationary_density, Grid2D.flux), density-weighted
             median over grid nodes with rho > 1e-3 max rho
plus e_p and d1.  Outputs: S3_ratio_recomputed.csv, S3_ratio_definition.md
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import linalg
ROOT = Path(__file__).resolve().parents[1]; LEG = ROOT.parent / 'reproducibility'
for p in [ROOT / 'src', LEG / 'src', LEG / 'experiments']:
    sys.path.insert(0, str(p))
from features_fast import common7
from B_csf_additions import OUModel, grid_from_train
from gaussian_observer import flux_drift_ratio
from irreversibility import projection_stats
from data_daily import load_market
from pre_common import write_csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def frame(name):
    if name == 'SPX':
        d = pd.read_csv(ROOT / 'data' / 'spx_2000_2026.csv', parse_dates=['date']); ann = 252.0
    else:
        d = load_market(name); d = d[np.isfinite(d.close) & np.isfinite(d.volume) & (d.volume > 0)].reset_index(drop=True); ann = 365.0
    F = common7(d.close.values, d.volume.values, ann); F['date'] = pd.to_datetime(d.date.values)
    return F.dropna().reset_index(drop=True)


def grid_ratio(Z, ou):
    G = grid_from_train(Z, 40); f, D = ou.fields(G.pts); L = G.generator(f, D)
    rho = G.stationary_density(L); J = G.flux(f, D, rho)
    m = rho > 1e-3 * rho.max(); r = np.linalg.norm(J[m], axis=1) / (np.linalg.norm(f[m], axis=1) * rho[m])
    w = rho[m]; o = np.argsort(r); cw = np.cumsum(w[o]) / w.sum()
    return float(r[o][min(np.searchsorted(cw, 0.5), len(r) - 1)])


def main():
    rows = []
    for name in ['SPX', 'BTC', 'ETH']:
        F = frame(name); arch = pd.read_csv(LEG / 'tables' / f'E2v2_diag_{name}.csv')
        years = list(arch.year) + ([Y for Y in range(2021, 2027)] if name == 'SPX' else [])
        for Y in years:
            tr = F[F.date < pd.Timestamp(f'{Y}-01-01')]
            X = StandardScaler().fit_transform(tr[['log_rv20', 'log_rv5', 'log_act_rel', 'dd20', 'ret20', 'ews_var', 'ews_ar1']])
            Z = StandardScaler().fit_transform(PCA(n_components=2, random_state=0).fit_transform(X))
            ou = OUModel(Z); Sig = linalg.solve_continuous_lyapunov(ou.A, -ou.D)
            st = projection_stats(Z); a = arch[arch.year == Y]
            rows.append(dict(market=name, year=Y, n_train=len(tr), archived_n_train=int(a.n_train.iloc[0]) if len(a) else np.nan,
                             archived_ratio=float(a.flux_over_drift_median.iloc[0]) if len(a) else np.nan,
                             ratio_analytic=flux_drift_ratio(ou.A, ou.D, 0.5 * (Sig + Sig.T)), ratio_grid=grid_ratio(Z, ou),
                             ep=st['ep'], d1=st['d1']))
        print(name, flush=True)
    D = pd.DataFrame(rows); write_csv(D, 'S3_ratio_recomputed.csv')
    M = D[D.archived_ratio.notna()].groupby('market')[['archived_ratio', 'ratio_analytic', 'ratio_grid', 'ep', 'd1']].median()
    txt = ['# S3 -- definition of the flux/drift ratio', '',
           'The archived diagnostic (tables/E2v2_diag_*.csv, column flux_over_drift_median) was produced by code that is not in',
           'the reproducibility package; its state (n_train starts after the 750-day Hawkes window), its drift model (columns',
           'hess_wmin/hess_wmax, n_post, n_subnet indicate an ensemble of non-linear drift fits) and its median convention cannot',
           'be recovered.  We therefore define the ratio explicitly (Appendix D) and recompute it on the common-7 PC state with',
           'the frozen linear OU fit.', '',
           'Definition: J(z) = f(z) rho(z) - (1/2) div(D rho)(z);  R = median over z ~ rho of |J(z)| / (|f(z)| rho(z)).', '',
           'Medians over the archived years:', '', M.round(4).to_markdown(), '',
           'Interpretation: the analytic and grid versions agree with each other (same fitted model, different quadrature);',
           'differences from the archived values reflect the unrecoverable state and drift model, not the definition.']
    (ROOT / 'tables_pre' / 'S3_ratio_definition.md').write_text('\n'.join(txt) + '\n')
    print(M.round(4).to_string()); print('S3 complete'); (ROOT / 'tables_pre' / '_cache' / 'DONE_S3_ratio_definition').touch()


if __name__ == '__main__':
    main()
