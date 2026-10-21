"""fig6_stommel.pdf: three panels (a) signed decomposition, (b) ramp rates + origin shift, (c) frozen forcing."""
import pandas as pd, numpy as np, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = pathlib.Path(r"C:\Users\xby04\Desktop\paper1_PRE\paper1_pre\kooloth_test")
figdir = pathlib.Path(r"C:\Users\xby04\Desktop\paper1_PRE\paper1_pre\manuscript\figures")
figdir.mkdir(parents=True, exist_ok=True)

ts = pd.read_csv(K/"H3_timeseries_scen10.csv")
eps = pd.read_csv(K/"H3_eps_scaling.csv")
org = pd.read_csv(K/"H3_origin_shift.csv")
modes = pd.read_csv(K/"modeS_consistency.csv")

fig = plt.figure(figsize=(11, 3.4))

# (a) signed decomposition
ax = fig.add_subplot(1, 3, 1)
ax.fill_between(ts.t, ts.D_raw_minus2se, ts.D_raw_plus2se, color="gray", alpha=0.25, label=r"$\pm2$SE")
ax.plot(ts.t, ts.D_raw, "k-", lw=1.5, label="raw ensemble")
ax.plot(ts.t, ts.D_meanpath, "r--", lw=1.2, label="mean-path")
ax.plot(ts.t, ts.D_fluct*10, "g:", lw=1.2, label="fluctuation $\\times$10")
ax.plot(ts.t, ts.D_noisefree, "b-.", lw=1.0, label="noise-free")
ax.axvline(820, color="gray", ls=":")
ax.set_xlabel("reporting time t"); ax.set_ylabel("signed dccf")
ax.set_title("(a) decomposition", fontsize=10); ax.legend(fontsize=7)

# (b left) |D| vs eta2 three ramp rates
ax = fig.add_subplot(1, 3, 2)
ax.plot(eps.eta2, eps.D_orig_ramp15.abs(), "k-o", ms=3, label=r"ramp $1.5\times10^{-3}$")
ax.plot(eps.eta2, eps.D_half_ramp75.abs(), "b-s", ms=3, label=r"$0.75\times10^{-3}$")
ax.plot(eps.eta2, eps.D_double_ramp30.abs(), "r-^", ms=3, label=r"$3.0\times10^{-3}$")
ax.set_xlabel(r"freshwater forcing $\eta_2$"); ax.set_ylabel("|dccf|")
ax.set_title("(b) ramp rate", fontsize=10); ax.legend(fontsize=7)

# (b right origin shift) -> use sub-position manually
ax2 = fig.add_axes([0.56, 0.15, 0.13, 0.72])
Tc = org[org.channel=="T"]; Sc = org[org.channel=="S"]
ax2.plot(Tc.c, Tc.rel_at_t605, "r-o", ms=4, label="T shifted")
ax2.plot(Sc.c, Sc.rel_at_t605, "b-s", ms=4, label="S shifted")
ax2.axhline(0, color="gray", lw=0.5)
ax2.set_xlabel("shift c"); ax2.set_ylabel("% change", fontsize=8)
ax2.set_title("origin shift", fontsize=9); ax2.legend(fontsize=7)

# (c) frozen: normalize each at r=1 (use two points: slow/fast)
ax = fig.add_subplot(1, 3, 3)
rows = modes[modes.relaxation_rate!="ratio_1.46_over_0.26"].astype({"relaxation_rate":float})
r = rows.relaxation_rate.values
ax.plot(r, rows.centered_stat/rows.centered_stat.iloc[0], "k-o", label="centered stat")
ax.plot(r, rows.varT/rows.varT.iloc[0], "b-s", label="var(T)")
ax.plot(r, (1-rows.ac_lag05)/(1-rows.ac_lag05.iloc[0]), "r-^", label=r"1$-$AC(0.5)")
ax.set_xlabel("slow relaxation rate r"); ax.set_ylabel("normalized at r=1.46")
ax.set_title("(c) frozen forcing", fontsize=10); ax.legend(fontsize=7)

fig.subplots_adjust(wspace=0.55, bottom=0.18, top=0.88, left=0.07, right=0.98)
out = figdir/"fig6_stommel.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)
