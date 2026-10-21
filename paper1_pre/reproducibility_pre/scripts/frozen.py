"""Frozen-forcing runs: centered statistic, var(T), lag-0.5 AC at two relaxation rates."""
import numpy as np, sdeint, pathlib

DATA = pathlib.Path(r"C:\Users\xby04\Doubao\chats\2026-10-20\new-chat-1\k0_gen\data")
n1,n2 = 3.0,0.0
ETA2 = 1.5e-3*700.0   # freeze at pre-tipping forcing
N = 8000; t = np.linspace(0,400,N)   # equilibrate
sigma=0.05; B=np.diag([sigma,sigma])
def g(x,tt): return B

def run(r, seed0):
    def drift(z,ti):
        T,S=z
        return np.array([n1-T*(1+np.sqrt((T-S)**2)), ETA2-S*(r+np.sqrt((T-S)**2))])
    from scipy.optimize import fsolve
    def eq(zz):
        T,S=zz; return [n1-T*(1+np.sqrt((T-S)**2)), ETA2-S*(r+np.sqrt((T-S)**2))]
    x0 = fsolve(eq,[1.3,0.0])
    ens=np.zeros((N,2,100))
    for i in range(100):
        np.random.seed(seed0+i)
        ens[:,:,i]=sdeint.itoint(drift,g,x0,t)
    return ens

rows={}
for r in [1.46,0.26]:
    ens = run(r, 20261020+10000)
    tail = slice(N//2, None)   # stationary half
    T = ens[tail,0,:]
    varT = T.var()
    # lag-0.5 AC: lag=0.5/dt points
    lagp = int(round(0.5/(t[1]-t[0])))
    ac = np.mean([np.corrcoef(T[:-lagp,i], T[lagp:,i])[0,1] for i in range(100)])
    # centered statistic (within windows, n=2000)
    n=2000; lags=np.arange(1,100,1)
    Tv=T; Sv=ens[tail,1,:]
    dcc=[]
    for w in range(0, T.shape[0]-n-100, 400):
        acc=0
        for l in lags:
            ccf=(Tv[w:w+n]*Sv[w+l:w+l+n]).mean(axis=0)
            ccfr=(Sv[w:w+n]*Tv[w+l:w+l+n]).mean(axis=0)
            acc += (ccfr-ccf)
        dcc.append(np.abs(acc).mean()/len(lags))
    stat=np.mean(dcc)
    rows[r]=dict(centered_stat=stat, varT=varT, ac_lag05=ac)
    print(f"r={r}: stat={stat:.3e} varT={varT:.4e} AC(0.5)={ac:.3f}", flush=True)

import json, pathlib
OUT=pathlib.Path(r"C:\Users\xby04\Doubao\chats\2026-10-20\new-chat-1\k0_gen\out")
ratio_stat = rows[1.46]['centered_stat']/rows[0.26]['centered_stat']
ratio_var  = rows[1.46]['varT']/rows[0.26]['varT']
import pandas as pd
pd.DataFrame([
    dict(relaxation_rate=1.46, **rows[1.46]),
    dict(relaxation_rate=0.26, **rows[0.26]),
    dict(relaxation_rate="ratio_1.46_over_0.26",
         centered_stat=ratio_stat, varT=ratio_var,
         ac_lag05=rows[1.46]['ac_lag05']/rows[0.26]['ac_lag05']),
]).to_csv(OUT/"modeS_consistency.csv", index=False)
print("VtwoGrowth=",round(ratio_stat,2)," VarGrowth=",round(ratio_var,2), flush=True)
