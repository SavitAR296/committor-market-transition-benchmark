"""Signed Kooloth dccf statistic + mean-path/fluctuation decomposition.
Mirrors Stommel_model.ipynb cell 9 (m=300, n=2000, tf=100 lags) but KEEPS THE SIGN."""
import numpy as np, time, pathlib
from scipy.integrate import solve_ivp

DATA = pathlib.Path(r"C:\Users\xby04\Doubao\chats\2026-10-20\new-chat-1\k0_gen\data")
z = np.load(DATA / "Stommel_z_0.05_regen.npy")   # (20000, 2, 500)
N, _, Ne = z.shape
t = np.linspace(0, 1000, N)
dt = t[1]-t[0]
print("loaded", z.shape, flush=True)

n1,n2,n3,RAMP = 3.0,0.0,0.3,1.5e-3
def stommel_det(ti, zz):
    T,S = zz
    return [n1-T*(1+np.sqrt((T-S)**2)), (n2+RAMP*ti)-S*(n3+np.sqrt((T-S)**2))]

# deterministic trajectory
from scipy.optimize import fsolve
def stommel0(zz):
    T,S=zz; return [n1-T*(1+np.sqrt((T-S)**2)), n2-S*(n3+np.sqrt((T-S)**2))]
x0 = fsolve(stommel0, [1.30277564, 0.0])
sol = solve_ivp(stommel_det,[0,1000],x0,dense_output=True,rtol=1e-10,atol=1e-10)
zt = sol.sol(t)  # (2,N)

m, n, tf = 300, 2000, 100
lag = np.arange(1, 100, 1)   # 99 lags

def signed_dccf(Tarr, Sarr):
    """Tarr,Sarr: (N, Ne) -> signed statistic D(k) at window indices (no abs, no ensemble avg)."""
    idx = np.arange(n+tf, Tarr.shape[0], m+tf)
    out = np.zeros((idx.shape[0], Tarr.shape[1]))
    for k, ti in enumerate(idx):
        base = ti - idx[0]
        win_T = Tarr[base:base+n, :]      # (n, Ne)
        acc = np.zeros(Tarr.shape[1])
        for l in lag:
            ccf = (win_T * Sarr[base+l:base+l+n, :]).mean(axis=0)
            ccfr = (Sarr[base:base+n, :] * Tarr[base+l:base+l+n, :]).mean(axis=0)
            acc += (ccfr - ccf)
        out[k] = acc / len(lag)
    return out, idx

t0=time.time()
# per-member signed statistic
D_member, idx = signed_dccf(z[:,0,:], z[:,1,:])     # (nwin, Ne)
print("per-member done", D_member.shape, f"{time.time()-t0:.1f}s", flush=True)

# mean-path statistic: on ensemble-mean trajectory
Tm = z[:,0,:].mean(axis=1, keepdims=True)
Sm = z[:,1,:].mean(axis=1, keepdims=True)
D_mp, _ = signed_dccf(Tm, Sm)                        # (nwin,1)
D_mp = D_mp[:,0]

# noise-free (deterministic) trajectory statistic
D_det, _ = signed_dccf(zt[0,:,None], zt[1,:,None])
D_det = D_det[:,0]

# ensemble raw mean (signed)
D_raw = D_member.mean(axis=1)
# fluctuation = raw - meanpath (cross terms vanish in ensemble mean)
D_fl = D_raw - D_mp

# report times: the notebook reports around t=100..800; pick 4 reporting times
t_rep = t[idx]
np.savez(DATA/"decomp.npz", t=t_rep, D_raw=D_raw, D_mp=D_mp, D_fl=D_fl,
         D_det=D_det, D_member=D_member)
print("reporting times (t):", t_rep, flush=True)
for i, tt in enumerate(t_rep):
    print(f"t={tt:7.1f}  raw={D_raw[i]:+.2e}  meanpath={D_mp[i]:+.2e}  fluct={D_fl[i]:+.2e}  det={D_det[i]:+.2e}", flush=True)
