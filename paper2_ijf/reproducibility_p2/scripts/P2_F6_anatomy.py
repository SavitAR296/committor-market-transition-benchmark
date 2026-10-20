"""P2_F6 -- anatomy of the score: what S is made of, and which event it forecasts.

(a) Within-year Spearman correlation of S with the stress coordinate s = z.w (direction of the learned target B).
(b) Pooled AUC of S and of s for the operational event, paired moving-block-bootstrap interval of the difference
    (blocks of 2h days, B = 1000), and annual AUCs of both.
(c) Own target versus operational target: on headline days outside the learned set (s < thr), the CORP decomposition of
    S against the model's own event hitB_h (the state enters B within h days) and against the operational event.
    Small MCB against hitB with large MCB against the operational event = target mismatch; large MCB against both =
    the fitted dynamics misjudge their own hitting probabilities.
(d) Saturation: share of headline days with S > 0.99, and share of the model's training days with S >= 0.999.
Inputs: P2_F1_anatomy_<tag>.csv, P2_F1_forecasts_<tag>.csv.  Outputs: P2_F6_spearman_<tag>.csv, P2_F6_auc_<tag>.csv,
P2_F6_own_target_<tag>.csv, P2_F6_saturation_<tag>.csv.  Seed 20261024.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_common import TAB, auc, corp, mbb_index, boot_block, write_csv

SEED, B = 20261024, 1000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='exp'); a = ap.parse_args()
    A = pd.read_csv(TAB / f'P2_F1_anatomy_{a.tag}.csv', parse_dates=['date'])
    F = pd.read_csv(TAB / f'P2_F1_forecasts_{a.tag}.csv', parse_dates=['date'])
    hl = F[(F.set == 'common7') & (F.method == 'ou_raw')][['date', 'horizon', 'headline']]
    A = A.merge(hl, on=['date', 'horizon']); A = A[A.headline.astype(bool) & A.y.notna()]
    rng = np.random.default_rng(SEED); sp, au, own, sat = [], [], [], []
    for h in [5, 20]:
        H = A[A.horizon == h].sort_values('date')
        for Y, g in H.groupby('year'):
            rho = spearmanr(g.S, g.s).statistic if g.S.nunique() > 1 else np.nan
            sp.append(dict(horizon=h, year=Y, n=len(g), spearman_S_s=rho, share_S_ge_09995=float((g.S >= 0.9995).mean())))
            au.append(dict(horizon=h, scope=str(Y), n=len(g), events=int(g.y.sum()), AUC_S=auc(g.y, g.S), AUC_s=auc(g.y, g.s)))
            sat.append(dict(horizon=h, year=Y, share_headline_S_gt_099=float((g.S > 0.99).mean()),
                            share_train_S_ge_0999=float(g.share_train_S_ge_0999.iloc[0]), n_train=int(g.n_train.iloc[0])))
        for scope, Hs in [('2008-2026', H), ('2009-2026', H[H.year >= 2009])]:
            y = Hs.y.values.astype(int); S = Hs.S.values; s = Hs.s.values
            dd = []
            for ix in mbb_index(len(y), boot_block(h), B, rng):
                if 0 < y[ix].sum() < len(ix):
                    dd.append(auc(y[ix], S[ix]) - auc(y[ix], s[ix]))
            au.append(dict(horizon=h, scope=scope, n=len(y), events=int(y.sum()), AUC_S=auc(y, S), AUC_s=auc(y, s),
                           dAUC_q05=np.quantile(dd, .05), dAUC_q95=np.quantile(dd, .95)))
            O = Hs[(Hs.s < Hs.thr) & Hs.hitB.notna()]
            for target in ['hitB', 'y']:
                c = corp(O[target].values, O.S.values)
                own.append(dict(horizon=h, scope=scope, target='own (enter learned B)' if target == 'hitB' else 'operational',
                                n=c['n'], event_rate=float(O[target].mean()), mean_S=float(O.S.mean()), BS=c['BS'],
                                MCB=c['MCB'], DSC=c['DSC'], UNC=c['UNC'], AUC=auc(O[target], O.S)))
    write_csv(pd.DataFrame(sp), f'P2_F6_spearman_{a.tag}.csv'); write_csv(pd.DataFrame(au), f'P2_F6_auc_{a.tag}.csv')
    write_csv(pd.DataFrame(own), f'P2_F6_own_target_{a.tag}.csv'); write_csv(pd.DataFrame(sat), f'P2_F6_saturation_{a.tag}.csv')
    SP = pd.DataFrame(sp); print(SP.groupby('horizon').spearman_S_s.describe()[['min', '50%', 'max']].round(3).to_string())
    AU = pd.DataFrame(au); print(AU[AU.scope.str.contains('-')].round(4).to_string(index=False))
    print(pd.DataFrame(own).round(4).to_string(index=False))
    ST = pd.DataFrame(sat); print(ST[ST.horizon == 5][['year', 'share_headline_S_gt_099', 'share_train_S_ge_0999']].round(3).T.to_string(header=False))
    print('F6', a.tag, 'complete')


if __name__ == '__main__':
    main()
