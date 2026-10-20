"""Vectorised common-7 feature builder (identical formulas to the frozen data_daily.build_features for the seven
headline features; verified against features_SPX_2026.csv in E4) plus a centred-window variant for E4(c).
Centred variant: odd symmetric windows 5, 21, 61, 251 replace 5, 20, 60, 250; ret20 -> log c_{t+10} - log c_{t-10};
dd20 -> c_t / max_{|k|<=10} c_{t+k} - 1.  Under time reversal of a reversible input these features map onto themselves."""
import numpy as np
import pandas as pd

COMMON = ['log_rv20', 'log_rv5', 'log_act_rel', 'dd20', 'ret20', 'ews_var', 'ews_ar1']


def rolling_autocorr1(x: pd.Series, W: int = 60) -> pd.Series:
    """pandas x.rolling(W).apply(lambda s: s.autocorr(1)) in closed form via rolling sums (n = W-1 pairs)."""
    n = W - 1
    S = x.rolling(n).sum(); Q = (x * x).rolling(n).sum(); P = (x * x.shift(1)).rolling(n).sum()
    Sa, Sb, Saa, Sbb, Sab = S.shift(1), S, Q.shift(1), Q, P
    num = Sab - Sa * Sb / n
    den = np.sqrt((Saa - Sa * Sa / n) * (Sbb - Sb * Sb / n))
    return num / den


def common7(close, volume, ann=252.0, kind='trailing'):
    c = pd.Series(np.asarray(close, float)); v = pd.Series(np.asarray(volume, float))
    lc = np.log(c); r = lc.diff(); r2 = r * r
    if kind == 'trailing':
        roll = lambda s, w: s.rolling(w)
        rv20 = np.sqrt(roll(r2, 20).sum() * ann / 20); rv5 = np.sqrt(roll(r2, 5).sum() * ann / 5)
        log_act_rel = np.log(roll(v, 20).mean()) - np.log(roll(v, 250).mean())
        dd20 = c / roll(c, 20).max() - 1.0
        ret20 = lc.diff(20)
        lrv = np.log(rv20)
        ews_var = lrv.diff().rolling(60).var(); ews_ar1 = rolling_autocorr1(lrv, 60)
    elif kind == 'centered':
        roll = lambda s, w: s.rolling(w, center=True)
        rv20 = np.sqrt(roll(r2, 21).sum() * ann / 21); rv5 = np.sqrt(roll(r2, 5).sum() * ann / 5)
        log_act_rel = np.log(roll(v, 21).mean()) - np.log(roll(v, 251).mean())
        dd20 = c / roll(c, 21).max() - 1.0
        ret20 = lc.shift(-10) - lc.shift(10)
        lrv = np.log(rv20)
        ews_var = lrv.diff().rolling(61, center=True).var(); ews_ar1 = rolling_autocorr1(lrv, 61).shift(-30)
    else:
        raise ValueError(kind)
    return pd.DataFrame(dict(log_rv20=np.log(rv20), log_rv5=np.log(rv5), log_act_rel=log_act_rel, dd20=dd20,
                             ret20=ret20, ews_var=ews_var, ews_ar1=ews_ar1))
