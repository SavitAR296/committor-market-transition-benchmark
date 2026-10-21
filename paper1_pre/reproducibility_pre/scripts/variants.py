"""Ramp-rate variants and origin-shift analysis for H3 tables."""
import numpy as np, time, sdeint, pathlib
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

DATA = pathlib.Path(r"C:\Users\xby04\Doubao\chats\2026-10-20\new-chat-1\k0_gen\data")
N = 20000; t = np.linspace(0,1000,N)
n1,n2,n3 = 3.0,0.0,0.3
m,n,tf = 300,2000,100; lag=np.arange(1,100,1)

def make_drift(RAMP):
    def f(z,ti):
        T,S=z
        return np.array([n1-T*(1+np.sqrt((T-S)**2)), (n2+RAMP*ti)-S*(n3+np.sqrt((T-S)**2))])
    return f

def signed_dccf_on_path(Tv, Sv):
    """Tv,Sv: (N,) single path -> statistic at window indices."""
    idx = np.arange(n+tf, N, m+tf)
    out = np.zeros(idx.shape[0])
    for k,ti in enumerate(idx):
        base = ti-idx[0]
        acc=0.0
        for l in lag:
            ccf = np.mean(Tv[base:base+n]*Sv[base+l:base+l+n])
            ccfr = np.mean(Sv[base:base+n]*Tv[base+l:base+l+n])
            acc += (ccfr-ccf)
        out[k]=acc/len(lag)
    return out, idx

# ---- origin shift on the mean path of main ensemble ----
z = np.load(DATA/"Stommel_z_0.05_regen.npy")
Tm = z[:,0,:].mean(axis=1); Sm = z[:,1,:].mean(axis=1)
D0, idx = signed_dccf_on_path(Tm, Sm)
t_rep = t[idx]
print("origin-shift table (relative change vs no shift), at t=405/605/785:", flush=True)
orig_rows=[]
for c in [0.5,1.0,2.0]:
    for which, Tv, Sv in [("T", Tm+c, Sm), ("S", Tm, Sm+c)]:
        Dc,_ = signed_dccf_on_path(Tv, Sv)
        rel = (Dc-D0)/np.abs(D0)*100
        orig_rows.append((c,which,rel[15],rel[25],rel[34]))
        print(f"c={c}, shift {which}: rel change at t405={rel[15]:+.2f}%  t605={rel[25]:+.2f}%  t785={rel[34]:+.2f}%", flush=True)
np.save(DATA/"origin_shift.npy", np.array(orig_rows), allow_pickle=True)

# ---- ramp variants: regenerate ensembles at 0.5x and 2x ramp ----
sigma=0.05; B=np.diag([sigma,sigma])
def g(x,tt): return B
def f0(zz):
    T,S=zz; return [n1-T*(1+np.sqrt((T-S)**2)), n2-S*(n3+np.sqrt((T-S)**2))]
x0 = fsolve(f0,[1.30277564,0.0])

for RAMP_mult, tag in [(0.5,"half"),(2.0,"double")]:
    RAMP = 1.5e-3*RAMP_mult
    drift = make_drift(RAMP)
    np.random.seed(20261020)
    ens = np.zeros((N,2,500))
    ts=time.time()
    for i in range(500):
        np.random.seed(20261020+i)
        ens[:,:,i] = sdeint.itoint(drift, g, x0, t)
    print(f"ramp x{RAMP_mult} ensemble done {time.time()-ts:.0f}s", flush=True)
    np.save(DATA/f"Stommel_ramp{tag}.npy", ens)
    Tm2 = ens[:,0,:].mean(axis=1); Sm2 = ens[:,1,:].mean(axis=1)
    Dmp,_ = signed_dccf_on_path(Tm2, Sm2)
    np.save(DATA/f"D_ramp{tag}.npy", Dmp)
    print(f"  t=405: {Dmp[15]:+.2e}  t=605: {Dmp[25]:+.2e}  t=785: {Dmp[34]:+.2e}", flush=True)
