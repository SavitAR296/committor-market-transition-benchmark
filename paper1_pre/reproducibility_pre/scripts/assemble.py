"""Assemble the 6 kooloth_test CSVs from computed npz/npy results."""
import numpy as np, json, pathlib, pandas as pd

DATA = pathlib.Path(r"C:\Users\xby04\Doubao\chats\2026-10-20\new-chat-1\k0_gen\data")
OUT  = pathlib.Path(r"C:\Users\xby04\Doubao\chats\2026-10-20\new-chat-1\k0_gen\out")
OUT.mkdir(exist_ok=True)

d = np.load(DATA/"decomp.npz")
t_rep=d["t"]; D_raw=d["D_raw"]; D_mp=d["D_mp"]; D_fl=d["D_fl"]; D_det=d["D_det"]; D_mem=d["D_member"]
D_half=np.load(DATA/"D_ramphalf.npy"); D_dbl=np.load(DATA/"D_rampdouble.npy")
orig=np.load(DATA/"origin_shift.npy", allow_pickle=True)

# 4 reporting times (pre-tipping, matching original figure range)
pick=[15,25,34]   # t=405,605,785 (+ early ~205 -> idx 5)
pick=[5,15,25,34]

mp_share = D_mp[pick]/D_raw[pick]*100        # mean-path share
fl_share = D_fl[pick]/D_raw[pick]*100        # fluctuation share (opposite sign)
det_diff = np.abs(D_det[pick]-D_raw[pick])/np.abs(D_raw[pick])*100

# matched-forcing eps ratio: eta2 = RAMP*t. double ramp at t/2.
t_half = t_rep/2.0
D_dbl_at = np.interp(t_half, t_rep, D_dbl)
eps_ratio = np.abs(D_dbl_at[pick])/np.abs(D_raw[pick])

# origin shift T by c=1: rows where col1=='T' and c==1
T1 = [r for r in orig if str(r[1])=="T" and float(r[0])==1.0][0]
origin_min, origin_max = min(float(x) for x in T1[2:]), max(float(x) for x in T1[2:])

# LinApproxDev proxy: |meanpath - noise-free| / |meanpath|
lin_dev = np.abs(D_mp[pick]-D_det[pick])/np.abs(D_mp[pick])*100

print("=== key macro values ===")
print("MP share %:", np.round(mp_share,1))
print("FL share %:", np.round(fl_share,1))
print("DetDiffMax %:", round(det_diff.max(),2))
print("EpsRatio:", np.round(eps_ratio,2))
print("Origin T c=1 %:", [float(x) for x in T1[2:]])
print("LinApproxDev %:", np.round(lin_dev,2))

# --- T6_H3.csv ---
summary = pd.DataFrame([
    dict(metric="MPshareMin", value=float(mp_share.min()), unit="%", note="mean-path share of signed signal at 4 reporting times"),
    dict(metric="MPshareMax", value=float(mp_share.max()), unit="%", note=""),
    dict(metric="FLshareMin", value=float(fl_share.min()), unit="%", note="fluctuation share, opposite sign"),
    dict(metric="FLshareMax", value=float(fl_share.max()), unit="%", note=""),
    dict(metric="DetDiffMax", value=float(det_diff.max()), unit="%", note="max |noise-free - ensemble|/|ensemble|"),
    dict(metric="EpsRatioMin", value=float(eps_ratio.min()), unit="x", note="doubled ramp, matched forcing"),
    dict(metric="EpsRatioMax", value=float(eps_ratio.max()), unit="x", note=""),
    dict(metric="OriginMin", value=float(origin_min), unit="%", note="T origin shifted by 1 unit"),
    dict(metric="OriginMax", value=float(origin_max), unit="%", note=""),
    dict(metric="LinApproxDevMax", value=float(lin_dev.max()), unit="%", note="|meanpath - noise-free|/|meanpath|"),
])
summary.to_csv(OUT/"T6_H3.csv", index=False)

# --- H3_timeseries_scen10.csv (Fig 6a data) ---
se = D_mem.std(axis=1)/np.sqrt(D_mem.shape[1])*2   # ±2 SE
pd.DataFrame(dict(
    t=t_rep, eta2=1.5e-3*t_rep,
    D_raw=D_raw, D_meanpath=D_mp, D_fluct=D_fl, D_noisefree=D_det,
    D_raw_plus2se=D_raw+se, D_raw_minus2se=D_raw-se,
)).to_csv(OUT/"H3_timeseries_scen10.csv", index=False)

# --- H3_eps_scaling.csv (Fig 6b left: D vs eta2 at 3 ramp rates) ---
pd.DataFrame(dict(
    eta2=t_rep*1.5e-3,
    D_orig_ramp15=D_mp, D_half_ramp75=D_half, D_double_ramp30=D_dbl,
)).to_csv(OUT/"H3_eps_scaling.csv", index=False)

# --- H3_origin_shift.csv (Fig 6b right) ---
pd.DataFrame([dict(c=r[0], channel=r[1], rel_at_t405=r[2], rel_at_t605=r[3], rel_at_t785=r[4]) for r in orig]
).to_csv(OUT/"H3_origin_shift.csv", index=False)

# --- H3_meta.json ---
meta = dict(
    source="Kooloth et al. notebook 17904887, Stommel_model.ipynb generation code re-run",
    seeds={f"realization_{i}": 20261020+i for i in range(500)},
    ensemble=dict(N_time=20000, t_max=1000, dt=0.05, n_realizations=500, sigma=0.05,
                 params=dict(n1=3.0, n2=0.0, n3=0.3, ramp=1.5e-3)),
    statistic=dict(window_points=2000, window_tu=100, step_points=400, lags_points="1..99",
                   lags_tu="0.05..4.95", signed=True),
    reporting_times_t=[float(t_rep[i]) for i in pick],
    note="frozen-forcing row (modeS) computed separately; see modeS_consistency.csv",
)
(OUT/"H3_meta.json").write_text(json.dumps(meta, indent=2))
print("\nwrote:", sorted(p.name for p in OUT.iterdir()))
