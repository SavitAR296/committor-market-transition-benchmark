"""
Daily (macro) layer: real data only.

Markets
  BTC, ETH : Coin Metrics community data (PriceUSD, volume_reported_spot_usd_1d, TxCnt)
  SPX      : S&P 500 daily OHLCV 2000-01-03 .. 2020-04-17 (Yahoo Finance via vega-datasets),
             joined with CBOE VIX (datahub) as the implied-volatility coordinate.

Features are strictly causal (trailing windows only).  The daily endogeneity n_t is the
Hawkes branching ratio of "large-move" days (|r_t| above the trailing 90% quantile) estimated by
maximum likelihood + Laplace posterior on a trailing window of `hawkes_window` days.
"""
import os
import numpy as np, pandas as pd
from hawkes import fit_exp_mle, fit_exp_laplace

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_market(name):
    if name in ("BTC", "ETH"):
        df = pd.read_csv(os.path.join(ROOT, "data", f"{name.lower()}.csv"), low_memory=False, parse_dates=["time"])
        df = df.rename(columns={"time": "date"})
        df = df[["date", "PriceUSD", "volume_reported_spot_usd_1d", "TxCnt"]].dropna(subset=["PriceUSD"])
        start = "2013-01-01" if name == "BTC" else "2016-03-01"
        df = df[df.date >= start].reset_index(drop=True)
        df["close"] = df["PriceUSD"].astype(float)
        df["volume"] = df["volume_reported_spot_usd_1d"].astype(float)
        df["activity"] = df["volume"]
        df["high"] = df["low"] = np.nan
        df["ann"] = 365.0
        return df[["date", "close", "high", "low", "volume", "activity", "ann"]]
    if name == "SPX":
        df = pd.read_csv(os.path.join(ROOT, "data", "sp500-2000.csv"), parse_dates=["date"])
        v = pd.read_csv(os.path.join(ROOT, "data", "vix_daily.csv"), parse_dates=["DATE"]).rename(columns={"DATE": "date", "CLOSE": "vix"})
        df = df.merge(v[["date", "vix"]], on="date", how="left")
        df["vix"] = df["vix"].ffill()
        df["activity"] = df["volume"].astype(float)
        df["ann"] = 252.0
        return df[["date", "close", "high", "low", "volume", "activity", "vix", "ann"]]
    raise ValueError(name)


def rolling_hawkes_n(r, window=750, stride=5, q=0.85, qwin=1000, min_events=25):
    """Daily branching ratio of large-move days.  r: array of log returns.  Returns arrays (n_mean, n_sd)
    aligned with r (NaN before enough history); strictly causal."""
    T = len(r)
    absr = np.abs(r)
    thr = pd.Series(absr).rolling(qwin, min_periods=250).quantile(q).shift(1).values   # trailing threshold
    ev = np.where(absr > thr)[0].astype(float)
    n_mean = np.full(T, np.nan); n_sd = np.full(T, np.nan); n_lo = np.full(T, np.nan); n_hi = np.full(T, np.nan)
    beta = np.full(T, np.nan)
    for t in range(window, T, stride):
        e = ev[(ev >= t - window) & (ev < t)] - (t - window)
        if len(e) < min_events:
            continue
        hist = ev[(ev >= t - window - 250) & (ev < t - window)] - (t - window)
        fl = fit_exp_laplace(e, hist, float(window), beta0=0.1)
        n_mean[t] = fl["n_mean"]; n_sd[t] = fl["n_sd"]; n_lo[t] = fl["n_lo90"]; n_hi[t] = fl["n_hi90"]; beta[t] = fl["beta"]
    out = pd.DataFrame(dict(n_mean=n_mean, n_sd=n_sd, n_lo=n_lo, n_hi=n_hi, beta=beta)).ffill()
    return out, ev.astype(int)


def build_features(df, hawkes_window=750):
    df = df.copy()
    df["r"] = np.log(df["close"]).diff()
    ann = df["ann"].iloc[0]
    df["rv20"] = np.sqrt(df["r"].pow(2).rolling(20).sum() * ann / 20)           # annualised 20-day realised vol
    df["rv5"] = np.sqrt(df["r"].pow(2).rolling(5).sum() * ann / 5)
    df["log_rv20"] = np.log(df["rv20"])
    df["log_rv5"] = np.log(df["rv5"])
    if df["high"].notna().any():
        pk = np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))
        df["log_rv20_pk"] = 0.5 * np.log(pk.rolling(20).mean() * ann)
    df["log_act"] = np.log(df["activity"].rolling(20).mean())
    df["log_act_rel"] = df["log_act"] - np.log(df["activity"].rolling(250).mean())     # relative activity
    roll_max = df["close"].rolling(20).max()
    df["dd20"] = df["close"] / roll_max - 1.0                                       # 20-day drawdown (<=0)
    df["ret20"] = np.log(df["close"]).diff(20)
    # classical CSD indicators of the volatility state (used as features and as EWS baselines)
    x = df["log_rv20"]
    df["ews_var"] = x.diff().rolling(60).var()
    df["ews_ar1"] = x.rolling(60).apply(lambda s: pd.Series(s).autocorr(1), raw=False)
    if "vix" in df:
        df["log_vix"] = np.log(df["vix"])
        df["vrp"] = (df["vix"] / 100) ** 2 - df["rv20"] ** 2
    hk, ev = rolling_hawkes_n(df["r"].fillna(0).values, window=hawkes_window)
    df["n_t"] = hk["n_mean"].values; df["n_sd"] = hk["n_sd"].values; df["one_minus_n"] = 1 - df["n_t"]
    df["n_lo"] = hk["n_lo"].values; df["n_hi"] = hk["n_hi"].values; df["hawkes_beta"] = hk["beta"].values
    df["is_event_day"] = 0; df.loc[ev, "is_event_day"] = 1
    return df


def crisis_labels(df, horizon, rule="rv_dd", dd_thr=None, q=0.85, qwin=1000):
    """Rule-based crisis set B^rule (pre-registered): rv20 above the trailing (qwin-day) q-quantile AND
    20-day drawdown worse than dd_thr (-15% crypto, -8% equity).  Event E_t = first entry into B^rule within (t, t+h].
    Alternatives: rule='rv' (only the RV quantile), rule='dd' (only the drawdown)."""
    if dd_thr is None:
        dd_thr = -0.15 if df["ann"].iloc[0] > 300 else -0.08
    q_rv = df["rv20"].rolling(qwin, min_periods=500).quantile(q).shift(1)
    high_rv = df["rv20"] > q_rv
    dd = df["dd20"] < dd_thr
    if rule == "rv_dd":
        inB = (high_rv & dd)
    elif rule == "rv":
        inB = high_rv
    elif rule == "dd":
        inB = dd
    else:
        raise ValueError(rule)
    inB = inB.fillna(False).values
    T = len(df)
    # entry indicator: in B today, not in B yesterday
    entry = inB & ~np.concatenate([[False], inB[:-1]])
    E = np.zeros(T, dtype=float)
    idx_entries = np.where(entry)[0]
    for t in range(T):
        nxt = idx_entries[(idx_entries > t) & (idx_entries <= t + horizon)]
        E[t] = 1.0 if len(nxt) else 0.0
    E[T - horizon:] = np.nan            # label not yet observable
    return E, inB, entry


# ----------------------------------------------------------------------------
# embeddings (fitted on training data only)
# ----------------------------------------------------------------------------
class Standardizer:
    def fit(self, X):
        self.m = np.nanmean(X, 0); self.s = np.nanstd(X, 0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.m) / self.s


class PCA2:
    def fit(self, X, d=2):
        C = np.cov(X.T)
        w, V = np.linalg.eigh(C)
        self.V = V[:, ::-1][:, :d]; self.explained = (w[::-1] / w.sum())[:d]
        return self

    def transform(self, X):
        return X @ self.V


class TICA2:
    """Time-lagged independent component analysis (slowest linear coordinates), symmetrised estimator."""
    def fit(self, X, lag=5, d=2, reg=1e-6):
        X0 = X[:-lag]; X1 = X[lag:]
        m = 0.5 * (X0.mean(0) + X1.mean(0))
        X0 = X0 - m; X1 = X1 - m
        C0 = 0.5 * (X0.T @ X0 + X1.T @ X1) / len(X0) + reg * np.eye(X.shape[1])
        Ct = 0.5 * (X0.T @ X1 + X1.T @ X0) / len(X0)
        # generalised eigenproblem Ct v = lambda C0 v
        L = np.linalg.cholesky(C0); Li = np.linalg.inv(L)
        M = Li @ Ct @ Li.T
        w, V = np.linalg.eigh(M)
        order = np.argsort(w)[::-1]
        self.w = w[order][:d]; self.V = (Li.T @ V[:, order])[:, :d]; self.m = m
        self.timescales = -lag / np.log(np.clip(self.w, 1e-6, 1 - 1e-6))
        return self

    def transform(self, X):
        return (X - self.m) @ self.V
