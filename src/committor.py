"""
Committor machinery.

Analytic pieces (Section 3 of the paper):
  * one-dimensional diffusions: committor = ratio of scale functions
  * Proposition 2: tilted quartic double well  U(z;c) = z^4/4 - z^2/2 - c z
  * Proposition 3: CIR limit of nearly-unstable Hawkes, absolute-threshold committor q_n(l)
Numerical pieces:
  * finite-difference discretisation of the generator L = f.grad + 1/2 tr(D grad^2) on a 2-D grid
  * committor (L q = 0, q|A = 0, q|B = 1) by sparse direct solve
  * finite-horizon hitting probability v_h by implicit Euler on the backward Kolmogorov equation
  * stationary density / probability flux J = f p - 1/2 div(D p)
All solvers are deterministic; nothing is simulated.
"""
import numpy as np
from scipy import integrate, sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import brentq

# ----------------------------------------------------------------------------
# 1-D closed forms
# ----------------------------------------------------------------------------
def committor_1d(scale_density, a, b, z):
    """q(z) = int_a^z s'(y) dy / int_a^b s'(y) dy for a 1-D diffusion with scale density s'."""
    z = np.atleast_1d(np.asarray(z, float))
    den = integrate.quad(scale_density, a, b, limit=200)[0]
    out = np.array([integrate.quad(scale_density, a, zi, limit=200)[0] / den for zi in z])
    return np.clip(out, 0, 1)


def U_dw(z, c):
    return z**4 / 4 - z**2 / 2 - c * z


def dU_dw(z, c):
    return z**3 - z - c


C_STAR = 2 / (3 * np.sqrt(3))  # fold bifurcation of the tilted double well


def dw_equilibria(c):
    """Real roots of z^3 - z - c = 0 (sorted)."""
    r = np.roots([1, 0, -1, -c])
    r = np.sort(r[np.abs(r.imag) < 1e-9].real)
    return r


def dw_committor(z, c, sigma, a, b):
    """Proposition 2(i): closed-form committor of dz = -U'(z)dt + sigma dW, A=(-inf,a], B=[b,inf)."""
    U0 = U_dw(a, c)
    s = lambda y: np.exp(2 * (U_dw(y, c) - U0) / sigma**2)
    return committor_1d(s, a, b, z)


def dw_barrier(c):
    """Exact barrier U(z_s)-U(z_a) between the left well and the saddle, and the 3/2-scaling approximation."""
    r = dw_equilibria(c)
    if len(r) < 3:
        return 0.0, 0.0
    za, zs = r[0], r[1]
    exact = U_dw(zs, c) - U_dw(za, c)
    approx = 4 * 3 ** (-5 / 4) * (C_STAR - c) ** 1.5
    return exact, approx


def dw_kramers_rate(c, sigma):
    """Overdamped Kramers rate for escape from the left well over the saddle (D = sigma^2/2)."""
    r = dw_equilibria(c)
    if len(r) < 3:
        return np.nan
    za, zs = r[0], r[1]
    Ua = 3 * za**2 - 1
    Us = 3 * zs**2 - 1
    dU = U_dw(zs, c) - U_dw(za, c)
    return np.sqrt(Ua * abs(Us)) / (2 * np.pi) * np.exp(-2 * dU / sigma**2)


def dw_mfpt_exact(c, sigma, z0, b):
    """Exact mean first-passage time from z0 to b for the 1-D diffusion with reflecting -infinity:
       T(z0) = (2/sigma^2) int_{z0}^{b} e^{2U(y)/s2} int_{-inf}^{y} e^{-2U(x)/s2} dx dy."""
    s2 = sigma**2
    inner = lambda y: integrate.quad(lambda x: np.exp(-2 * (U_dw(x, c) - U_dw(y, c)) / s2), -6, y, limit=200)[0]
    return (2 / s2) * integrate.quad(inner, z0, b, limit=200)[0]


# ---- Proposition 3: CIR limit --------------------------------------------------
def cir_scale_density(x, nu, gamma):
    return x ** (-nu) * np.exp(gamma * x)


def cir_committor(x, nu, gamma, a, b):
    s = lambda y: cir_scale_density(y, nu, gamma)
    return committor_1d(s, a, b, x)


def cir_committor_absolute(l, n, a, b, nu=2.0, gamma=2.0):
    """Proposition 3(iv): committor for absolute intensity thresholds (A = {Lambda<=a}, B = {Lambda>=b})
    as a function of the branching ratio n through the tilt eps = 1-n:
        q_n(l) = int_a^l t^{-nu} e^{gamma eps t} dt / int_a^b t^{-nu} e^{gamma eps t} dt."""
    eps = 1.0 - n
    s = lambda t: t ** (-nu) * np.exp(gamma * eps * t - gamma * eps * a)
    return committor_1d(s, a, b, l)


def tilt_monotonicity_check(w, a, b, l, eps_grid):
    """Lemma 1 numerical check: F(eps) = int_a^l w e^{eps t} / int_a^b w e^{eps t} is non-increasing."""
    F = []
    for e in eps_grid:
        num = integrate.quad(lambda t: w(t) * np.exp(e * (t - a)), a, l, limit=200)[0]
        den = integrate.quad(lambda t: w(t) * np.exp(e * (t - a)), a, b, limit=200)[0]
        F.append(num / den)
    F = np.array(F)
    return F, bool(np.all(np.diff(F) <= 1e-10))


# ----------------------------------------------------------------------------
# finite differences: 1-D (verification) and 2-D (working solver)
# ----------------------------------------------------------------------------
def fd_committor_1d(f, D, z, a, b):
    """Solve 1/2 D q'' + f q' = 0 on a uniform grid z with q=0 for z<=a, q=1 for z>=b (central differences)."""
    h = z[1] - z[0]
    N = len(z)
    fz = f(z); Dz = D(z)
    main = -Dz / h**2
    lower = 0.5 * Dz / h**2 - fz / (2 * h)
    upper = 0.5 * Dz / h**2 + fz / (2 * h)
    A = sparse.diags([lower[1:], main, upper[:-1]], [-1, 0, 1], format="lil")
    rhs = np.zeros(N)
    for i in range(N):
        if z[i] <= a or z[i] >= b:
            A[i, :] = 0; A[i, i] = 1.0; rhs[i] = 1.0 if z[i] >= b else 0.0
    return spsolve(A.tocsr(), rhs)


class Grid2D:
    """Uniform grid on [x0,x1]x[y0,y1] with nx x ny points; builds the sparse generator matrix for
    L = f.grad + 1/2 sum_ij D_ij d_i d_j  with first-order upwinding for the drift (monotone scheme)
    and central differences for the second derivatives (mixed term by central differences)."""

    def __init__(self, x0, x1, y0, y1, nx=80, ny=80):
        self.x = np.linspace(x0, x1, nx); self.y = np.linspace(y0, y1, ny)
        self.nx, self.ny = nx, ny
        self.hx = self.x[1] - self.x[0]; self.hy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.pts = np.stack([self.X.ravel(), self.Y.ravel()], axis=1)

    def idx(self, i, j):
        return i * self.ny + j

    def generator(self, f_vals, D_vals):
        """f_vals: (N,2); D_vals: (N,2,2) evaluated at self.pts. Returns sparse (N,N) matrix L and a boolean
        mask of boundary points (where one-sided extrapolation is applied by zero Neumann conditions)."""
        nx, ny, hx, hy = self.nx, self.ny, self.hx, self.hy
        N = nx * ny
        rows, cols, vals = [], [], []
        fx = f_vals[:, 0].reshape(nx, ny); fy = f_vals[:, 1].reshape(nx, ny)
        Dxx = D_vals[:, 0, 0].reshape(nx, ny); Dyy = D_vals[:, 1, 1].reshape(nx, ny); Dxy = D_vals[:, 0, 1].reshape(nx, ny)

        def add(i, j, ii, jj, v):
            ii = min(max(ii, 0), nx - 1); jj = min(max(jj, 0), ny - 1)   # zero-flux (reflecting) at the box edges
            rows.append(self.idx(i, j)); cols.append(self.idx(ii, jj)); vals.append(v)

        for i in range(nx):
            for j in range(ny):
                # diffusion xx, yy
                add(i, j, i + 1, j, 0.5 * Dxx[i, j] / hx**2); add(i, j, i - 1, j, 0.5 * Dxx[i, j] / hx**2)
                add(i, j, i, j + 1, 0.5 * Dyy[i, j] / hy**2); add(i, j, i, j - 1, 0.5 * Dyy[i, j] / hy**2)
                add(i, j, i, j, -Dxx[i, j] / hx**2 - Dyy[i, j] / hy**2)
                # mixed term  D_xy d_x d_y  (factor 2 * 1/2 = 1)
                c = Dxy[i, j] / (4 * hx * hy)
                add(i, j, i + 1, j + 1, c); add(i, j, i - 1, j - 1, c); add(i, j, i + 1, j - 1, -c); add(i, j, i - 1, j + 1, -c)
                # drift, upwind
                if fx[i, j] >= 0:
                    add(i, j, i + 1, j, fx[i, j] / hx); add(i, j, i, j, -fx[i, j] / hx)
                else:
                    add(i, j, i - 1, j, -fx[i, j] / hx); add(i, j, i, j, fx[i, j] / hx)
                if fy[i, j] >= 0:
                    add(i, j, i, j + 1, fy[i, j] / hy); add(i, j, i, j, -fy[i, j] / hy)
                else:
                    add(i, j, i, j - 1, -fy[i, j] / hy); add(i, j, i, j, fy[i, j] / hy)
        L = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        return L

    def committor(self, L, maskA, maskB):
        """Solve L q = 0 with q=0 on A, q=1 on B (masks are boolean arrays of length N)."""
        N = L.shape[0]
        L = L.tolil()
        rhs = np.zeros(N)
        fixed = np.where(maskA | maskB)[0]
        for k in fixed:
            L.rows[k] = [k]; L.data[k] = [1.0]
        rhs[maskB] = 1.0
        q = spsolve(L.tocsr(), rhs)
        return np.clip(q, 0, 1)

    def hitting_prob(self, L, maskB, h, dt=0.25):
        """v_h(z) = P_z(tau_B <= h): backward equation d_t v + L v = 0, v(h,.)=0 off B, v=1 on B; implicit Euler."""
        N = L.shape[0]
        I = sparse.identity(N, format="csr")
        M = (I - dt * L).tolil()
        for k in np.where(maskB)[0]:
            M.rows[k] = [k]; M.data[k] = [1.0]
        M = M.tocsc()
        lu = sparse.linalg.splu(M)
        v = maskB.astype(float)
        nsteps = int(round(h / dt))
        for _ in range(nsteps):
            rhs = v.copy(); rhs[maskB] = 1.0
            v = lu.solve(rhs)
            v[maskB] = 1.0
        return np.clip(v, 0, 1)

    def hitting_prob_path(self, L, maskB, h_max, dt=0.5):
        """v(t, z) = P_z(tau_B <= t) for t = 0, dt, 2dt, ..., h_max (implicit Euler); returns array (nsteps+1, N).
        Used for the 'clock' specification: with a slow multiplicative time-scale parameter c_t the process is a
        time change of the reference process, so P(tau_B <= h | z, c_t) = v(c_t h, z)."""
        N = L.shape[0]
        I = sparse.identity(N, format="csr")
        M = (I - dt * L).tolil()
        for k in np.where(maskB)[0]:
            M.rows[k] = [k]; M.data[k] = [1.0]
        lu = sparse.linalg.splu(M.tocsc())
        v = maskB.astype(float); path = [v.copy()]
        for _ in range(int(round(h_max / dt))):
            rhs = v.copy(); rhs[maskB] = 1.0
            v = lu.solve(rhs); v[maskB] = 1.0
            path.append(np.clip(v, 0, 1))
        return np.array(path)

    def flux(self, f_vals, D_vals, p_vals):
        """Probability flux J = f p - 1/2 sum_j d_j (D_ij p) by central differences (returns (N,2))."""
        nx, ny = self.nx, self.ny
        p = p_vals.reshape(nx, ny)
        J = np.zeros((nx, ny, 2))
        for i in range(2):
            Dp_x = (D_vals[:, i, 0] * p_vals).reshape(nx, ny)
            Dp_y = (D_vals[:, i, 1] * p_vals).reshape(nx, ny)
            dDp = np.gradient(Dp_x, self.hx, axis=0) + np.gradient(Dp_y, self.hy, axis=1)
            J[:, :, i] = f_vals[:, i].reshape(nx, ny) * p - 0.5 * dDp
        return J.reshape(-1, 2)

    def stationary_density(self, L):
        """Stationary density of the discretised generator: left null vector of L (L^T p = 0), normalised."""
        Lt = L.T.tocsc()
        # solve (L^T + e e^T /N) p = e/N  -> enforces normalisation, robust for singular systems
        N = L.shape[0]
        A = Lt + sparse.csc_matrix(np.ones((N, 1)) @ np.ones((1, N))) / N
        p = spsolve(A, np.ones(N) / N)
        p = np.maximum(p, 0)
        return p / (p.sum() * self.hx * self.hy)
