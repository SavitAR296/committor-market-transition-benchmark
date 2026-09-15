"""
Figures 3 and 4 of the manuscript: analytic benchmarks.
  Fig. 3  Proposition 2 (tilted double well): potential and fold, closed-form committor (monotone in c) with the
          finite-difference solver overlaid, barrier 3/2-scaling, Kramers rate vs exact mean first-passage time.
  Fig. 4  Proposition 3 (CIR limit of nearly-unstable Hawkes): absolute-threshold committor q_n(l) monotone in n,
          Lemma 1 tilt function, clock & amplifier (relaxation time, mean intensity) and the exact bin-count
          autocorrelation of an exponential Hawkes process (rho_{k+1}/rho_k = e^{-beta(1-n)Delta}).
Also writes tables/T_analytic_checks_v2.csv.  No simulation anywhere: closed forms + deterministic quadrature/FD.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from plotstyle import plt, COL, CYCLE, W2
from committor import (U_dw, dU_dw, C_STAR, dw_equilibria, dw_committor, dw_barrier, dw_kramers_rate, dw_mfpt_exact,
                       fd_committor_1d, cir_committor_absolute, tilt_monotonicity_check)
from hawkes import exp_hawkes_bin_autocorr
ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
rows = []

# ------------------------------------------------------------------ Figure 3: Proposition 2
cs = [0.0, 0.15, 0.30, 0.36]
sigma = 0.5
a, b = -1.0, 1.0
z = np.linspace(-1.6, 1.6, 400)
fig, ax = plt.subplots(1, 4, figsize=(W2, 2.0))
for c, col in zip(cs, CYCLE):
    ax[0].plot(z, U_dw(z, c) - U_dw(z, c).min(), color=col, label=f"$c={c}$")
    r = dw_equilibria(c)
    ax[0].plot(r, U_dw(r, c) - U_dw(z, c).min(), "o", ms=2.5, color=col)
ax[0].set_xlabel("$z$"); ax[0].set_ylabel("$U(z;c)-\\min U$"); ax[0].set_title("(a) tilted double well", loc="left")
ax[0].set_ylim(-0.02, 0.9)

zq = np.linspace(a, b, 300)
for c, col in zip(cs, CYCLE):
    q = dw_committor(zq, c, sigma, a, b)
    ax[1].plot(zq, q, color=col, label=f"$c={c}$")
    zfd = np.linspace(-1.5, 1.5, 601)
    qfd = fd_committor_1d(lambda x: -dU_dw(x, c), lambda x: sigma**2 * np.ones_like(x), zfd, a, b)
    sel = (zfd > a) & (zfd < b)
    ax[1].plot(zfd[sel][::40], qfd[sel][::40], "o", ms=2.2, mfc="none", color=col)
    err = np.max(np.abs(np.interp(zfd[sel], zq, q) - qfd[sel]))
    rows.append(dict(quantity="Prop2 committor: closed form vs FD (max abs err)", setting=f"c={c}, sigma={sigma}", value=err))
    rows.append(dict(quantity="Prop2(v) committor at z=-0.3", setting=f"c={c}", value=float(dw_committor(-0.3, c, sigma, a, b)[0])))
ax[1].axvline(-0.3, color=COL["grey"], lw=0.6, ls=":")
ax[1].set_xlabel("$z$"); ax[1].set_ylabel("$q(z;c)$"); ax[1].set_title("(b) committor, $\\sigma=0.5$", loc="left")
ax[1].text(0.03, 0.9, "lines: closed form\ncircles: FD solver", transform=ax[1].transAxes, fontsize=6.5)
ax[1].legend(loc="lower right", handlelength=1.2)

cgrid = np.linspace(0.0, C_STAR - 1e-4, 200)
ex = np.array([dw_barrier(c)[0] for c in cgrid]); ap = np.array([dw_barrier(c)[1] for c in cgrid])
ax[2].plot(C_STAR - cgrid, ex, color=COL["blue"], label="exact $\\Delta U$")
ax[2].plot(C_STAR - cgrid, ap, "--", color=COL["red"], label="$4\\cdot 3^{-5/4}(c^*-c)^{3/2}$")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("$c^*-c$"); ax[2].set_ylabel("barrier $\\Delta U(c)$")
ax[2].set_title("(c) 3/2 barrier scaling", loc="left"); ax[2].legend(loc="lower right")
for c in [0.0, 0.15, 0.3, 0.36]:
    e_, a_ = dw_barrier(c); rows.append(dict(quantity="Prop2 barrier exact vs 3/2-scaling", setting=f"c={c}", value=e_, second=a_))

sig2 = 0.35
cg = np.linspace(0.05, 0.37, 14)
kr = np.array([dw_kramers_rate(c, sig2) for c in cg])
mf = []
for c in cg:
    r = dw_equilibria(c)
    mf.append(1.0 / dw_mfpt_exact(c, sig2, r[0], r[2] if len(r) == 3 else 1.0))
mf = np.array(mf)
ax[3].plot(cg, kr, "-", color=COL["red"], label="Kramers")
ax[3].plot(cg, mf, "o", ms=3, color=COL["blue"], label="$1/\\mathbb{E}\\,\\tau$ (exact)")
ax[3].set_yscale("log"); ax[3].axvline(C_STAR, color=COL["grey"], lw=0.7, ls="--"); ax[3].text(C_STAR - 0.005, kr.min() * 1.3, "$c^*$", ha="right", fontsize=7)
ax[3].set_xlabel("$c$"); ax[3].set_ylabel("escape rate  ($\\sigma=0.35$)"); ax[3].set_title("(d) escape rate", loc="left"); ax[3].legend(loc="lower right")
for c, k1, k2 in zip(cg, kr, mf):
    rows.append(dict(quantity="Kramers rate vs 1/MFPT", setting=f"c={c:.3f}, sigma={sig2}", value=k1, second=k2))
fig.savefig(os.path.join(FIG, "fig3_prop2_double_well.pdf")); plt.close(fig)

# ------------------------------------------------------------------ Figure 4: Proposition 3, Lemma 1, clock & amplifier
fig, ax = plt.subplots(1, 4, figsize=(W2, 2.0))
aA, bB = 0.5, 2.0
ns = [0.5, 0.8, 0.9, 0.95, 0.99]
ls_ = np.linspace(aA, bB, 200)
for n, col in zip(ns, CYCLE):
    ax[0].plot(ls_, cir_committor_absolute(ls_, n, aA, bB), color=col, label=f"$n={n}$")
ax[0].set_xlabel("current intensity $\\ell$ (absolute units)"); ax[0].set_ylabel("$q_n(\\ell)$")
ax[0].set_title("(a) absolute thresholds", loc="left"); ax[0].legend(loc="lower right", handlelength=1.2)
ngrid = np.linspace(0.3, 0.999, 60)
for l0, col in zip([0.9, 1.2, 1.5], [COL["blue"], COL["red"], COL["orange"]]):
    qn = np.array([cir_committor_absolute(l0, n, aA, bB)[0] for n in ngrid])
    ax[1].plot(ngrid, qn, color=col, label=f"$\\ell={l0}$")
    nu = 2.0
    lim0 = (l0 ** (1 - nu) - aA ** (1 - nu)) / (bB ** (1 - nu) - aA ** (1 - nu))
    ax[1].plot([1.0], [lim0], "o", ms=3, color=col, mfc="none")
    for n in [0.5, 0.8, 0.9, 0.95, 0.99]:
        rows.append(dict(quantity="Prop3(iv) q_n(l) absolute threshold", setting=f"l={l0}, n={n}", value=float(cir_committor_absolute(l0, n, aA, bB)[0])))
ax[1].set_xlabel("branching ratio $n$"); ax[1].set_ylabel("$q_n(\\ell)$"); ax[1].set_title("(b) monotone in $n$", loc="left"); ax[1].legend(loc="lower right")
ax[1].text(0.03, 0.92, "circles: $\\varepsilon\\to0$ limit", transform=ax[1].transAxes, fontsize=6.5)
eps = np.linspace(-4, 4, 161)
for nu, col in zip([0.5, 1.0, 2.0, 3.0], CYCLE):
    F, mono = tilt_monotonicity_check(lambda t: t ** (-nu), aA, bB, 1.2, eps)
    ax[2].plot(eps, F, color=col, label=f"$w=t^{{-{nu}}}$")
    rows.append(dict(quantity="Lemma 1 F(eps) nonincreasing", setting=f"nu={nu}", value=mono))
ax[2].set_xlabel("tilt $\\varepsilon$"); ax[2].set_ylabel("$F(\\varepsilon)$"); ax[2].set_title("(c) Lemma 1", loc="left"); ax[2].legend(loc="upper right")
# (d) exact bin-count autocorrelation of an exponential Hawkes process: clock and amplifier
beta, Delta = 1.0, 1.0
ks = np.arange(1, 9)
for n, col in zip([0.5, 0.8, 0.9, 0.95, 0.99], CYCLE):
    Lam, gam, rh = exp_hawkes_bin_autocorr(1.0, n * beta, beta, Delta, kmax=8)
    ax[3].plot(ks, rh, "o-", ms=2.5, color=col, label=f"$n={n}$")
    rows.append(dict(quantity="exp-Hawkes bin autocorr rho_2/rho_1 vs exp(-beta(1-n)Delta)", setting=f"n={n}", value=rh[1] / rh[0], second=np.exp(-beta * (1 - n) * Delta)))
ax[3].set_yscale("log"); ax[3].set_xlabel("lag $k$ ($\\Delta=1/\\beta$)"); ax[3].set_ylabel("$\\rho_k$ of bin counts")
ax[3].set_title("(d) CSD in activity", loc="left"); ax[3].legend(loc="lower left", ncol=2, handlelength=1.0, columnspacing=0.6)
fig.savefig(os.path.join(FIG, "fig4_prop3_cir.pdf")); plt.close(fig)

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "tables", "T_analytic_checks_v2.csv"), index=False)
print(pd.DataFrame(rows).to_string())
