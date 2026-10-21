"""Generate 500-realization Stommel ensemble (Kooloth setup), fixed seeds."""
import numpy as np, time, sdeint, pathlib
from scipy.optimize import fsolve

OUT = pathlib.Path(r"C:\Users\xby04\Doubao\chats\2026-10-20\new-chat-1\k0_gen\data")
OUT.mkdir(parents=True, exist_ok=True)

N = 20000
t = np.linspace(0, 1000, N)
t0 = time.time()
n1, n2, n3 = 3.0, 0.0, 0.3
RAMP = 1.5e-3

def stommel_sde(z, ti):
    T, S = z
    return np.array([n1 - T*(1+np.sqrt((T-S)**2)),
                     (n2 + RAMP*ti) - S*(n3+np.sqrt((T-S)**2))])
def stommel0(z):
    T, S = z
    return np.array([n1 - T*(1+np.sqrt((T-S)**2)),
                     n2 - S*(n3+np.sqrt((T-S)**2))])

sigma = 0.05
B = np.diag([sigma, sigma])
def g(x, tt): return B

x0 = fsolve(stommel0, np.array([1.30277564, 0.0]))

N_ens = 500
z = np.zeros((N, 2, N_ens))
for i in range(N_ens):
    np.random.seed(20261020 + i)          # sdeint uses global RNG
    z[:, :, i] = sdeint.itoint(stommel_sde, g, x0, t)
    if i % 50 == 0:
        print(f"ensemble {i}/500, elapsed {time.time()-t0:.0f}s", flush=True)

np.save(OUT / "Stommel_z_0.05_regen.npy", z)
print("saved", z.shape, flush=True)
