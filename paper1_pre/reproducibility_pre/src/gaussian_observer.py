"""Gaussian observer model (Proposition 2).

Latent process: stationary AR(1) x_t = phi x_{t-1} + sqrt(1-phi^2) eps_t  (unit variance, r(k)=phi^|k|),
the exact discretisation of the OU process dx = -lam x dt + sqrt(2 lam) dW with phi = exp(-lam).
Observer: m window means y_i(t) = sum_a k_i(a) x_{t-a}.  'trailing' windows use k_i(a) = 1/w_i for a=0..w_i-1;
'centered' windows are symmetric (2 x w moving average for even w: end weights 1/(2w)).

Conventions:  C(s) = Cov(y(t+s), y(t))  (so C(1)_{ij} = Cov(y_i(t+1), y_j(t)), C(-s) = C(s)^T).
Markov projection (population least squares of y_{t+1} on y_t, as in the frozen OU fit):
    Phi = C(1) C(0)^{-1},  A = Phi - I (drift per step),  D = C(0) - Phi C(0) Phi^T (residual covariance).
The fitted OU used by the forecasting pipeline is dZ = A Z dt + G dW, GG^T = D; its stationary covariance
Sigma solves A Sigma + Sigma A^T + D = 0, and with Omega = (A Sigma - Sigma A^T)/2 the stationary current is
J(z) = Omega Sigma^{-1} z rho(z), entropy production e_p = 2 tr(Sigma^{-1} Omega^T D^{-1} Omega).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.special import ndtr

# ------------------------------------------------------------------------------------------- kernels
def window_weights(w: int, kind: str = 'trailing'):
    """Return (lags a, weights k(a)) with y(t) = sum_a k(a) x_{t-a}.  Negative lags = future values."""
    if kind == 'trailing':
        return np.arange(w), np.full(w, 1.0 / w)
    if kind == 'centered':
        if w == 1:
            return np.array([0]), np.array([1.0])
        if w % 2 == 1:
            half = (w - 1) // 2
            return np.arange(-half, half + 1), np.full(w, 1.0 / w)
        half = w // 2
        k = np.full(w + 1, 1.0 / w); k[0] = k[-1] = 0.5 / w
        return np.arange(-half, half + 1), k
    raise ValueError(kind)


def cov_discrete(phi: float, ws, s: int = 0, kinds=None):
    """Exact C(s) = Cov(y(t+s), y(t)) for an AR(1) latent (finite double sums, no approximation)."""
    m = len(ws); kinds = kinds or ['trailing'] * m
    W = [window_weights(w, k) for w, k in zip(ws, kinds)]
    C = np.empty((m, m))
    for i, (ai, ki) in enumerate(W):
        for j, (aj, kj) in enumerate(W):
            lag = s - ai[:, None] + aj[None, :]          # (t+s-a_i) - (t-a_j)
            C[i, j] = np.sum(ki[:, None] * kj[None, :] * phi ** np.abs(lag))
    return C


def cross_xy_discrete(phi: float, ws, kinds=None):
    """c_i = Cov(x_t, y_i(t))."""
    kinds = kinds or ['trailing'] * len(ws)
    out = []
    for w, k in zip(ws, kinds):
        a, kk = window_weights(w, k)
        out.append(np.sum(kk * phi ** np.abs(a)))
    return np.array(out)


def _F2(u, lam):
    u = np.abs(u)
    return (np.exp(-lam * u) + lam * u) / lam ** 2


def cov_continuous(lam: float, ws, s: float = 0.0):
    """Closed form for trailing windows of the OU process (r(u)=e^{-lam|u|}):
       C_ij(s) = [F2(s+w_j) - F2(s) - F2(s-w_i+w_j) + F2(s-w_i)] / (w_i w_j),  F2(u) = (e^{-lam|u|}+lam|u|)/lam^2,
       the second antiderivative of e^{-lam|u|}."""
    ws = np.asarray(ws, float); m = len(ws)
    C = np.empty((m, m))
    for i in range(m):
        for j in range(m):
            wi, wj = ws[i], ws[j]
            C[i, j] = (_F2(s + wj, lam) - _F2(s, lam) - _F2(s - wi + wj, lam) + _F2(s - wi, lam)) / (wi * wj)
    return C


def cross_xy_continuous(lam: float, ws):
    ws = np.asarray(ws, float)
    return (1 - np.exp(-lam * ws)) / (lam * ws)


def posterior_variance(C0, c):
    """s^2 = Var(x_t | y_t) = 1 - c^T C(0)^{-1} c  (independent of the observed value y)."""
    return float(1.0 - c @ np.linalg.solve(C0, c))


def s2_discrete(phi, ws, kinds=None):
    return posterior_variance(cov_discrete(phi, ws, 0, kinds), cross_xy_discrete(phi, ws, kinds))


def s2_continuous(lam, ws):
    return posterior_variance(cov_continuous(lam, ws, 0.0), cross_xy_continuous(lam, ws))


# ------------------------------------------------------------------------------- oracle p_op(x)
class AR1Hitting:
    """u_h(x) = P(max_{1<=k<=h} x_{t+k} >= b | x_t = x) for the unit-variance AR(1), by backward recursion on a
    cell partition of (-L, b) with exact Gaussian cell probabilities (piecewise-constant continuation value)."""

    def __init__(self, phi, b, hmax, dx=0.004, L=7.0):
        self.phi, self.b = phi, b
        self.sig = np.sqrt(1 - phi ** 2)
        edges = np.arange(-L, b + 1e-12, dx)
        if edges[-1] < b:
            edges = np.append(edges, b)
        self.edges = edges; self.centers = 0.5 * (edges[1:] + edges[:-1])
        P = self._cellprob(self.centers)            # (n, ncell)
        absorb = 1 - ndtr((b - phi * self.centers) / self.sig)
        u = np.zeros(len(self.centers)); self.u = [u]
        for _ in range(hmax):
            u = absorb + P @ u
            self.u.append(u)

    def _cellprob(self, x):
        z = (self.edges[None, :] - self.phi * np.asarray(x)[:, None]) / self.sig
        cdf = ndtr(z)
        return np.diff(cdf, axis=1)

    def __call__(self, x, h):
        """Evaluate u_h at arbitrary x (one exact step from x, then the stored continuation values)."""
        x = np.asarray(x, float)
        out = np.empty(x.shape); flat = x.ravel(); res = np.empty(flat.shape)
        for i0 in range(0, len(flat), 4096):
            xx = flat[i0:i0 + 4096]
            absorb = 1 - ndtr((self.b - self.phi * xx) / self.sig)
            res[i0:i0 + 4096] = absorb + self._cellprob(xx) @ self.u[h - 1]
        out[...] = res.reshape(x.shape)
        return np.clip(out, 0, 1)


# ------------------------------------------------------------------------------------ leak L(s^2)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(160)
GH_W = GH_W / GH_W.sum()


def leak_quadrature(u, s2_values):
    """Leak(s^2) = E_y[Var(p_op(X) | y)] with X | y ~ N(m, s^2), m ~ N(0, 1-s^2); u is a vectorised p_op."""
    out = []
    Ep2 = np.sum(GH_W * u(GH_X) ** 2)
    for s2 in np.atleast_1d(s2_values):
        s = np.sqrt(s2); sm = np.sqrt(max(1 - s2, 0.0))
        X = sm * GH_X[:, None] + s * GH_X[None, :]
        r = (u(X) * GH_W[None, :]).sum(1)                # r*(m) = E[u(X)|m]
        out.append(Ep2 - np.sum(GH_W * r ** 2))
    return np.array(out)


def hermite_coefficients(u, nmax=200, n_grid=40001, L=12.0):
    """a_n = E[u(X) h_n(X)], h_n = He_n/sqrt(n!) orthonormal under N(0,1), n = 0..nmax
    (dense trapezoid rule on [-L, L]; the orthonormal three-term recurrence is numerically stable)."""
    x = np.linspace(-L, L, n_grid)
    wq = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    ux = u(x) * wq
    h_prev = np.ones_like(x); h = x.copy()
    a = [np.trapezoid(ux * h_prev, x), np.trapezoid(ux * h, x)]
    for n in range(1, nmax):
        h_next = (x * h - np.sqrt(n) * h_prev) / np.sqrt(n + 1)
        h_prev, h = h, h_next
        a.append(np.trapezoid(ux * h, x))
    return np.array(a)


def leak_hermite(a, s2_values):
    """Exact series (Mehler): Leak(s^2) = sum_{n>=1} a_n^2 [1 - (1-s^2)^n]."""
    n = np.arange(len(a))
    return np.array([np.sum(a[1:] ** 2 * (1 - (1 - s2) ** n[1:])) for s2 in np.atleast_1d(s2_values)])


# ------------------------------------------------------------------------ Markov projection
def markov_projection(C0, C1):
    """Population VAR(1)/OU fit from (C(0), C(1)). Returns dict with Phi, A, D, Sigma (Lyapunov), Omega,
    flux/drift ratio (stationary median), entropy production e_p (per step) and the Markov KLD d1 (per step)."""
    m = C0.shape[0]
    Phi = C1 @ np.linalg.inv(C0)
    A = Phi - np.eye(m)
    D = C0 - Phi @ C0 @ Phi.T
    D = 0.5 * (D + D.T)
    out = dict(Phi=Phi, A=A, D=D, omega_C1=0.5 * (C1 - C1.T))
    try:
        Sig = linalg.solve_continuous_lyapunov(A, -D)
        Sig = 0.5 * (Sig + Sig.T)
        stable = bool(np.all(np.linalg.eigvals(A).real < 0)) and bool(np.all(np.linalg.eigvalsh(Sig) > 0))
    except Exception:
        Sig, stable = np.full_like(C0, np.nan), False
    out['Sigma'] = Sig; out['stable'] = stable
    for tag, S in [('lyap', Sig), ('c0', C0)]:
        Om = 0.5 * (A @ S - S @ A.T)
        out[f'Omega_{tag}'] = Om
        out[f'ratio_{tag}'] = flux_drift_ratio(A, D, S) if np.all(np.isfinite(S)) else np.nan
        out[f'ep_{tag}'] = float(2 * np.trace(np.linalg.solve(S, Om.T) @ np.linalg.solve(D, Om))) \
            if np.all(np.isfinite(S)) else np.nan
    out['d1'] = markov_kld(C0, C1)
    return out


def flux_drift_ratio(A, D, Sigma, n_angle=200001):
    """median over the stationary Gaussian of |v(z)| / |f(z)|, f = A z, v = J/rho = (A + D Sigma^{-1}/2) z.
    Both are linear in z, so the ratio depends only on the direction of the whitened state; the median over
    N(0,Sigma) equals the median over a uniform angle of the whitened vector (exact, deterministic)."""
    m = A.shape[0]
    V = A + 0.5 * D @ np.linalg.inv(Sigma)
    Lc = np.linalg.cholesky(Sigma)
    if m == 2:
        th = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
        E = np.vstack([np.cos(th), np.sin(th)])
    else:
        rng = np.random.default_rng(0); E = rng.normal(size=(m, 400000))
    Z = Lc @ E
    num = np.linalg.norm(V @ Z, axis=0); den = np.linalg.norm(A @ Z, axis=0)
    return float(np.median(num / den))


def markov_kld(C0, C1):
    """Per-step KLD between forward and time-reversed path measures of the Gaussian Markov (VAR(1)) projection:
       d1 = KL( N(0,G) || N(0,G_rev) ) with G = [[C0, C1^T],[C1, C0]] for (y_t, y_{t+1}), G_rev its block transpose.
       log-determinants cancel because G_rev = P G P^T for the block swap P."""
    m = C0.shape[0]
    G = np.block([[C0, C1.T], [C1, C0]])
    Gr = np.block([[C0, C1], [C1.T, C0]])
    return float(0.5 * (np.trace(np.linalg.solve(Gr, G)) - 2 * m))


# ----------------------------------------------------------------------- full-path spectral KLD
def spectral_kld_rate(phi, ws, eta2, kinds=None, n_omega=20001):
    """Full-memory KLD rate between forward and backward measures of y + independent white observation noise
    (variance eta2 per component):  (1/4pi) int [tr(S(w)^{-T} S(w)) - m] dw,  S = K S_x K^dagger + eta2 I.
    For eta2 -> 0 the spectral matrix of m>=2 windows of a scalar latent has rank one and the rate diverges."""
    kinds = kinds or ['trailing'] * len(ws)
    om = np.linspace(-np.pi, np.pi, n_omega)
    Sx = (1 - phi ** 2) / np.abs(1 - phi * np.exp(-1j * om)) ** 2      # spectral density of x (normalised 2pi)
    K = []
    for w, k in zip(ws, kinds):
        a, kk = window_weights(w, k)
        K.append(np.sum(kk[None, :] * np.exp(-1j * om[:, None] * a[None, :]), axis=1))
    K = np.array(K).T                                                   # (n_omega, m)
    m = len(ws)
    vals = np.empty(len(om))
    for i in range(len(om)):
        S = Sx[i] * np.outer(K[i], K[i].conj()) + eta2 * np.eye(m)
        vals[i] = np.real(np.trace(np.linalg.solve(S.T, S))) - m
    return float(np.trapezoid(vals, om) / (4 * np.pi))


# ----------------------------------------------------------------------------------- simulation
def simulate_ar1(phi, T, rng, burn=5000):
    from scipy.signal import lfilter
    e = rng.normal(size=T + burn) * np.sqrt(1 - phi ** 2)
    x = lfilter([1.0], [1.0, -phi], e)
    return x[burn:]


def apply_windows(x, ws, kinds=None):
    """Observed vector y_t (rows with incomplete windows are NaN)."""
    kinds = kinds or ['trailing'] * len(ws)
    T = len(x); Y = np.full((T, len(ws)), np.nan)
    for j, (w, k) in enumerate(zip(ws, kinds)):
        a, kk = window_weights(w, k)
        lo, hi = a.max(), a.min()                     # need t-a in [0,T)
        t = np.arange(max(lo, 0), T + min(hi, 0))
        Y[t, j] = sum(kk[i] * x[t - a[i]] for i in range(len(a)))
    return Y


def empirical_projection(Y):
    """Sample Markov projection (least squares of y_{t+1} on [y_t, 1], i.e. the frozen OUModel)."""
    Y = Y[np.all(np.isfinite(Y), 1)]
    Z0 = Y[:-1]; Z1 = Y[1:]
    mu = Y.mean(0)
    C0 = np.cov(Y.T)
    C1 = ((Z1 - mu).T @ (Z0 - mu)) / (len(Z0) - 1)
    return markov_projection(C0, C1), C0, C1
