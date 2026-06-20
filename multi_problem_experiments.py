"""
Multi-problem numerical experiments for the inexact inverse iteration paper.

Three NLEP problem classes at varied scales:

  1. loaded_string  (1D rational Hermitian NLEP, NLEVP)
       n in {100, 500, 2000}; tridiagonal + rank-1.
  2. hadeler         (transcendental Hermitian NLEP, NLEVP variant)
       n = 200; dense.
  3. plate2d         (2D Hermitian rational NLEP, sparse pentadiagonal)
       n = 50^2 = 2500; sparse 5-point Laplacian + rank-1.

For each problem instance:
  - Compute reference eigenvalue (Newton-refined to machine precision).
  - Construct synthetic perturbed input (delta_lambda = 1e-6, eta_v = 1e-4).
  - Run Algorithm 4.2 with two inner solvers:
       (i)  direct: LU factorization of T(lam_hat).
       (ii) sketched: sketch-and-precondition LSQR (s = 2n).
  - Time the inner solves and report final residual.

Outputs:
  results_multi.csv : timing and residual table
  figure_scaling.png: scaling plot

Run:  python multi_problem_experiments.py
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator, lsqr as scipy_lsqr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import csv
from typing import Callable, Tuple, Dict, Optional

# ------------------------------------------------------------------ #
#  NLEP base interface                                                #
# ------------------------------------------------------------------ #

class NLEP:
    """Abstract base. Subclasses implement T(lam), Tprime(lam),
    matvec_T (matrix-free product), and the dense-or-sparse flag."""
    name: str
    n: int
    sparse: bool

    def T(self, lam: float):
        raise NotImplementedError

    def Tprime(self, lam: float):
        raise NotImplementedError

    def matvec_T(self, lam: float, x: np.ndarray) -> np.ndarray:
        # Default: form T(lam) and apply
        return self.T(lam) @ x

    def newton_refine(self, lam0: float, max_iter: int = 30,
                      tol: float = 1e-14) -> Tuple[float, np.ndarray]:
        """
        Newton on smallest-magnitude eigenvalue of T(lam).
        Uses dense or sparse eigh as appropriate.
        """
        lam = float(lam0)
        for _ in range(max_iter):
            Tl = self.T(lam)
            if self.sparse:
                # Use scipy.sparse.linalg.eigsh shift-invert near 0
                try:
                    evals, evecs = spla.eigsh(Tl, k=2, sigma=0.0,
                                              which="LM", maxiter=500)
                except Exception:
                    # Fall back to dense (only for small n)
                    Td = Tl.toarray() if hasattr(Tl, "toarray") else Tl
                    evals, evecs = la.eigh(Td)
            else:
                evals, evecs = la.eigh(Tl)
            j = int(np.argmin(np.abs(evals)))
            eps_j = float(evals[j])
            v_j = evecs[:, j]
            Tp = self.Tprime(lam)
            denom = float(np.real(v_j.conj() @ (Tp @ v_j)))
            if abs(denom) < 1e-30:
                break
            d = -eps_j / denom
            lam += d
            if abs(d) < tol * max(1.0, abs(lam)):
                break
        # Recompute eigenvector at refined lam
        Tl = self.T(lam)
        if self.sparse:
            try:
                evals, evecs = spla.eigsh(Tl, k=2, sigma=0.0,
                                          which="LM", maxiter=500)
            except Exception:
                Td = Tl.toarray() if hasattr(Tl, "toarray") else Tl
                evals, evecs = la.eigh(Td)
        else:
            evals, evecs = la.eigh(Tl)
        j = int(np.argmin(np.abs(evals)))
        v = evecs[:, j]
        return lam, v / la.norm(v)


# ------------------------------------------------------------------ #
#  Problem 1: loaded_string 1D                                       #
# ------------------------------------------------------------------ #

class LoadedString1D(NLEP):
    """T(lam) = K - lam M + (lam/(lam-sigma)) e_n e_n^T, K tridiagonal."""

    def __init__(self, n: int = 100, sigma: float = 50.0,
                 kappa: float = 1.0, sparse: bool = True):
        self.name = f"loaded_string_1D(n={n})"
        self.n = n
        self.sigma = sigma
        self.kappa = kappa
        self.sparse = sparse
        h = 1.0 / (n + 1)
        # Build K and M as sparse
        diag_main = (2.0 / h) * np.ones(n)
        diag_off = (-1.0 / h) * np.ones(n - 1)
        self.K_sp = sp.diags([diag_off, diag_main, diag_off],
                             [-1, 0, 1], format="csc")
        self.M_sp = h * sp.eye(n, format="csc")
        # Last-entry rank-1 part as sparse
        rows = np.array([n - 1]); cols = np.array([n - 1])
        self.C_sp = sp.csc_matrix((np.array([kappa]), (rows, cols)),
                                  shape=(n, n))

    def T(self, lam):
        if abs(lam - self.sigma) < 1e-15:
            raise ValueError("lam == sigma")
        T_sp = self.K_sp - lam * self.M_sp + (lam / (lam - self.sigma)) * self.C_sp
        return T_sp if self.sparse else T_sp.toarray()

    def Tprime(self, lam):
        Tp_sp = -self.M_sp + (-self.sigma / (lam - self.sigma) ** 2) * self.C_sp
        return Tp_sp if self.sparse else Tp_sp.toarray()


# ------------------------------------------------------------------ #
#  Problem 2: hadeler (transcendental Hermitian NLEP)                #
# ------------------------------------------------------------------ #

class Hadeler(NLEP):
    """
    Hadeler-style problem: T(lam) = (e^lam - 1) B + lam^2 A_2 - A_0,
    with B, A_0, A_2 SPD.  We construct random SPD instances of matching
    structure (the original NLEVP hadeler is n=8; we scale up).
    """
    def __init__(self, n: int = 200, seed: int = 7):
        self.name = f"hadeler(n={n})"
        self.n = n
        self.sparse = False
        rng = np.random.default_rng(seed)
        # Generate three Hermitian PSD matrices with controlled spectra
        def spd(scale, rng):
            X = rng.standard_normal((n, n))
            return scale * (X @ X.T) / n + 0.1 * np.eye(n)
        self.B  = spd(1.0, rng)
        self.A2 = spd(0.5, rng)
        self.A0 = spd(2.0, rng)

    def T(self, lam):
        return (np.exp(lam) - 1.0) * self.B + (lam ** 2) * self.A2 - self.A0

    def Tprime(self, lam):
        return np.exp(lam) * self.B + 2.0 * lam * self.A2


# ------------------------------------------------------------------ #
#  Problem 3: plate2d (2D rational Hermitian NLEP, sparse)           #
# ------------------------------------------------------------------ #

class Plate2D(NLEP):
    """
    T(lam) = L_2D - lam I + (lam/(lam-sigma)) e_corner e_corner^T,
    where L_2D is the 5-point Laplacian on an N x N grid (n = N^2),
    Dirichlet boundary conditions, and the load is at one corner.
    """
    def __init__(self, N: int = 50, sigma: float = 100.0,
                 kappa: float = 1.0):
        self.N = N
        self.n = N * N
        self.name = f"plate2d(N={N},n={self.n})"
        self.sparse = True
        self.sigma = sigma
        self.kappa = kappa
        # 5-point Laplacian on N x N grid, scaled like (1/h^2)
        h = 1.0 / (N + 1)
        e = np.ones(N)
        T1 = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1])
        I = sp.eye(N)
        self.L = (sp.kron(I, T1) + sp.kron(T1, I)) / (h * h)
        self.L = self.L.tocsc()
        # Load at the corner index 0 (could be any boundary point)
        rows = np.array([0]); cols = np.array([0])
        self.C = sp.csc_matrix((np.array([kappa]), (rows, cols)),
                               shape=(self.n, self.n))
        self.M = sp.eye(self.n, format="csc")  # mass = identity

    def T(self, lam):
        if abs(lam - self.sigma) < 1e-15:
            raise ValueError("lam == sigma")
        return self.L - lam * self.M + (lam / (lam - self.sigma)) * self.C

    def Tprime(self, lam):
        return -self.M + (-self.sigma / (lam - self.sigma) ** 2) * self.C


# ------------------------------------------------------------------ #
#  Inner solvers                                                     #
# ------------------------------------------------------------------ #

def direct_solve(A, b):
    """Direct LU/Cholesky solve. Handles both dense and sparse."""
    if sp.issparse(A):
        return spla.spsolve(A, b)
    else:
        return la.solve(A, b)

def sketch_precond_lsqr(A, b, s: int, tol: float = 1e-12,
                         max_iter: int = 500,
                         rng: Optional[np.random.Generator] = None
                         ) -> Tuple[np.ndarray, int]:
    """
    Sketch-and-precondition LSQR. Works for both dense and sparse A,
    via the matrix-free LinearOperator path.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = A.shape[0]
    if s < n:
        raise ValueError(f"s={s} < n={n}")

    # Form the sketched matrix M = S A.  For sparse A, S A is dense (s x n).
    S = rng.standard_normal((s, n)) / np.sqrt(s)
    if sp.issparse(A):
        M = (A.T @ S.T).T   # (s, n) dense
    else:
        M = S @ A
    Q, R = la.qr(M, mode="economic")

    # Right-preconditioned LSQR via LinearOperator
    def matvec(y):
        z = la.solve_triangular(R, y, lower=False)
        return A @ z

    def rmatvec(z):
        Az = A.conj().T @ z
        return la.solve_triangular(R.conj().T, Az, lower=True)

    B = LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec, dtype=float)
    out = scipy_lsqr(B, b, atol=tol, btol=tol, iter_lim=max_iter)
    y = out[0]
    n_iter = out[2]
    x = la.solve_triangular(R, y, lower=False)
    return x, n_iter


# ------------------------------------------------------------------ #
#  Inexact inverse iteration                                         #
# ------------------------------------------------------------------ #

def angle(x, y):
    if sp.issparse(x): x = np.asarray(x).ravel()
    c = abs(np.vdot(y, x)) / (la.norm(x) * la.norm(y))
    return np.arccos(min(1.0, c))

def matvec_A(A, x):
    if sp.issparse(A): return A @ x
    return A @ x

def normA(A):
    if sp.issparse(A):
        return spla.norm(A) if False else float(spla.svds(
            A, k=1, which="LM", return_singular_vectors=False)[0])
    return la.norm(A, 2)

def inexact_inverse_iteration(A, v_init, v_true, tau, K_outer,
                               solver: str = "sketch",
                               sketch_dim: Optional[int] = None,
                               rng: Optional[np.random.Generator] = None,
                               measure_alpha: bool = False,
                               u1_eigvec: Optional[np.ndarray] = None
                               ) -> Tuple[np.ndarray, Dict]:
    """
    Run K_outer inexact inverse iterations on A.
    solver in {'sketch', 'direct'}.

    If measure_alpha is True, compute the structural-noise alignment
        alpha_k = |<u_1, A u_tilde - v_k>| / ||A u_tilde - v_k||
    at each inner solve (with u_1 the unit smallest-magnitude eigenvector
    of A).  alpha_k = 0 means the inner residual is exactly orthogonal to
    u_1 (best case for Proposition 4.6); alpha_k = 1 means aligned with
    u_1 (worst case).  For random residuals, E[alpha] ~ 1/sqrt(n).

    u1_eigvec, if supplied, is reused (saving one eigh per call).
    """
    n = A.shape[0]
    norm_A = normA(A)

    def residual(v):
        return la.norm(matvec_A(A, v)) / norm_A

    # Precompute u_1 of A if needed and not supplied
    if measure_alpha and u1_eigvec is None:
        if sp.issparse(A):
            try:
                evals, evecs = spla.eigsh(A, k=2, sigma=0.0, which="LM",
                                          maxiter=500)
            except Exception:
                Ad = A.toarray()
                evals, evecs = la.eigh(Ad)
        else:
            evals, evecs = la.eigh(A)
        j = int(np.argmin(np.abs(evals)))
        u1_eigvec = np.asarray(evecs[:, j]).ravel()
        u1_eigvec = u1_eigvec / la.norm(u1_eigvec)

    v = v_init / la.norm(v_init)
    history = {
        "residuals": [residual(v)],
        "angles": [angle(v, v_true)],
        "inner_iters": [],
        "inner_times": [],
        "alpha": [],            # alignment factor per inner solve
        "inner_res_norm": [],   # ||A u_tilde - v_k|| per inner solve
    }

    # For direct solver, factor once
    if solver == "direct":
        if sp.issparse(A):
            lu = spla.splu(A.tocsc())
            solve = lambda b: lu.solve(b)
        else:
            lu_factor = la.lu_factor(A)
            solve = lambda b: la.lu_solve(lu_factor, b)

    if rng is None:
        rng = np.random.default_rng()

    for k in range(K_outer):
        t0 = time.perf_counter()
        if solver == "direct":
            u_tilde = solve(v)
            n_iter = 0
        elif solver == "sketch":
            s_dim = sketch_dim if sketch_dim is not None else 2 * n
            u_tilde, n_iter = sketch_precond_lsqr(
                A, v, s_dim, tol=tau, rng=rng)
        else:
            raise ValueError(f"unknown solver: {solver}")
        dt = time.perf_counter() - t0

        # Measure alpha (structural-noise alignment)
        if measure_alpha:
            r_inner = matvec_A(A, u_tilde) - v
            r_norm = la.norm(r_inner)
            if r_norm > 0:
                alpha_k = float(abs(np.vdot(u1_eigvec, r_inner)) / r_norm)
            else:
                alpha_k = 0.0
            history["alpha"].append(alpha_k)
            history["inner_res_norm"].append(float(r_norm))

        v = u_tilde / la.norm(u_tilde)
        history["residuals"].append(residual(v))
        history["angles"].append(angle(v, v_true))
        history["inner_iters"].append(n_iter)
        history["inner_times"].append(dt)

    return v, history


# ------------------------------------------------------------------ #
#  Synthetic input pair                                              #
# ------------------------------------------------------------------ #

def make_perturbed_input(nlep: NLEP, lam_true: float, v_true: np.ndarray,
                          delta_lambda: float, eta_v: float,
                          rng: np.random.Generator):
    sign = rng.choice([-1.0, 1.0])
    lam_hat = lam_true + sign * delta_lambda
    z = rng.standard_normal(nlep.n)
    z = z - (v_true @ z) * v_true
    z = z / la.norm(z)
    theta = np.arcsin(min(eta_v, 0.99))
    v_hat = np.cos(theta) * v_true + np.sin(theta) * z
    return lam_hat, v_hat / la.norm(v_hat)


# ------------------------------------------------------------------ #
#  Driver                                                            #
# ------------------------------------------------------------------ #

def run_problem(nlep: NLEP, lam0_guess: float, *, K_outer: int = 5,
                tau: float = 1e-12, delta_lambda: float = 1e-6,
                eta_v: float = 1e-4, n_trials: int = 3,
                seed: int = 2026):
    rng = np.random.default_rng(seed)

    # Reference eigenvalue
    print(f"\n=== {nlep.name} ===")
    print(f"  Refining reference eigenvalue from initial guess {lam0_guess}...")
    t0 = time.perf_counter()
    lam_true, v_true = nlep.newton_refine(lam0_guess, tol=1e-14)
    t_ref = time.perf_counter() - t0
    Tl = nlep.T(lam_true)
    res_true = la.norm(matvec_A(Tl, v_true)) / normA(Tl)
    print(f"  lam_true = {lam_true:.15f}  (refined in {t_ref:.2f}s, "
          f"residual {res_true:.2e})")

    # Run direct and sketch solvers, multiple trials
    t_direct_list = []
    t_sketch_list = []
    res_direct_list = []
    res_sketch_list = []
    alpha_loose_list = []   # alpha for sketch+LSQR with tau=1e-4 (LSQR stops early)
    alpha_tight_list = []   # alpha for sketch+LSQR with tau=1e-12 (over-converges)
    alpha_direct_list = []  # alpha for direct solver (residual ~ machine eps)
    natural_floor = None
    nl1 = None

    # Pre-compute u_1 of A once per trial; we will share between calls
    for trial in range(n_trials):
        lam_hat, v_hat = make_perturbed_input(nlep, lam_true, v_true,
                                              delta_lambda, eta_v, rng)
        A = nlep.T(lam_hat)

        # Compute u_1 of A once for alpha measurement
        if sp.issparse(A):
            try:
                evals_A, evecs_A = spla.eigsh(A, k=2, sigma=0.0, which="LM",
                                              maxiter=500)
            except Exception:
                evals_A, evecs_A = la.eigh(A.toarray())
        else:
            evals_A, evecs_A = la.eigh(A)
        j_min = int(np.argmin(np.abs(evals_A)))
        u1_A = np.asarray(evecs_A[:, j_min]).ravel()
        u1_A = u1_A / la.norm(u1_A)

        # Compute natural floor once (using same eigendecomp)
        if natural_floor is None:
            ne = np.sort(np.abs(evals_A))
            nl1 = float(ne[0])
            natural_floor = nl1 / normA(A)

        # Direct solver run (with alpha measurement)
        _, hist_d = inexact_inverse_iteration(
            A, v_hat, v_true, tau=tau, K_outer=K_outer,
            solver="direct", rng=rng,
            measure_alpha=True, u1_eigvec=u1_A)
        t_direct_list.append(np.mean(hist_d["inner_times"]))
        res_direct_list.append(hist_d["residuals"][-1])
        alpha_direct_list.extend(hist_d["alpha"])

        # Sketched solver run at default tau (over-converges)
        _, hist_s = inexact_inverse_iteration(
            A, v_hat, v_true, tau=tau, K_outer=K_outer,
            solver="sketch", sketch_dim=2 * nlep.n, rng=rng,
            measure_alpha=True, u1_eigvec=u1_A)
        t_sketch_list.append(np.mean(hist_s["inner_times"]))
        res_sketch_list.append(hist_s["residuals"][-1])
        alpha_tight_list.extend(hist_s["alpha"])

        # Sketched solver run at tau=1e-4 (LSQR stops early; alpha
        # reflects Krylov-subspace residual structure rather than
        # floating-point noise)
        _, hist_s_loose = inexact_inverse_iteration(
            A, v_hat, v_true, tau=1e-4, K_outer=K_outer,
            solver="sketch", sketch_dim=2 * nlep.n, rng=rng,
            measure_alpha=True, u1_eigvec=u1_A)
        alpha_loose_list.extend(hist_s_loose["alpha"])

    t_direct = float(np.median(t_direct_list))
    t_sketch = float(np.median(t_sketch_list))
    res_direct = float(np.median(res_direct_list))
    res_sketch = float(np.median(res_sketch_list))

    # Aggregate alpha statistics
    alpha_baseline_random = 1.0 / np.sqrt(nlep.n)  # E[alpha] for uniform sphere
    median_alpha_loose = float(np.median(alpha_loose_list)) if alpha_loose_list else float("nan")
    median_alpha_tight = float(np.median(alpha_tight_list)) if alpha_tight_list else float("nan")
    median_alpha_direct = float(np.median(alpha_direct_list)) if alpha_direct_list else float("nan")
    max_alpha_loose = float(np.max(alpha_loose_list)) if alpha_loose_list else float("nan")
    max_alpha_tight = float(np.max(alpha_tight_list)) if alpha_tight_list else float("nan")

    print(f"  natural floor       = {natural_floor:.3e}")
    print(f"  median residual (direct ) = {res_direct:.3e}")
    print(f"  median residual (sketch ) = {res_sketch:.3e}")
    print(f"  median per-step time direct  = {t_direct*1000:.2f} ms")
    print(f"  median per-step time sketch  = {t_sketch*1000:.2f} ms")
    speedup = t_direct / t_sketch if t_sketch > 0 else float("nan")
    print(f"  direct/sketch ratio  = {speedup:.2f}")
    print(f"  -- alpha (structural-noise alignment) --")
    print(f"  random baseline 1/sqrt(n) = {alpha_baseline_random:.3e}")
    print(f"  median alpha, sketch tau=1e-4  (LSQR-stopped):  {median_alpha_loose:.3e}  "
          f"(max {max_alpha_loose:.3e})")
    print(f"  median alpha, sketch tau=1e-12 (over-converged): {median_alpha_tight:.3e}  "
          f"(max {max_alpha_tight:.3e})")
    print(f"  median alpha, direct LU solver: {median_alpha_direct:.3e}")

    return {
        "name": nlep.name,
        "n": nlep.n,
        "sparse": nlep.sparse,
        "lam_true": lam_true,
        "natural_floor": natural_floor,
        "res_direct": res_direct,
        "res_sketch": res_sketch,
        "t_direct_ms": t_direct * 1000,
        "t_sketch_ms": t_sketch * 1000,
        "ratio": speedup,
        "alpha_random_baseline": alpha_baseline_random,
        "alpha_loose_median": median_alpha_loose,
        "alpha_loose_max": max_alpha_loose,
        "alpha_tight_median": median_alpha_tight,
        "alpha_tight_max": max_alpha_tight,
        "alpha_direct_median": median_alpha_direct,
    }


def main():
    results = []

    # ----------------- Problem 1: loaded_string 1D, three sizes ----- #
    for n in (100, 500, 2000):
        nlep = LoadedString1D(n=n, sigma=50.0, kappa=1.0, sparse=True)
        # The fundamental mode is approximately pi^2 ~ 9.87 (independent of n)
        results.append(run_problem(nlep, lam0_guess=12.0,
                                    n_trials=3 if n < 1000 else 2))

    # ----------------- Problem 2: hadeler ---------------------------- #
    nlep = Hadeler(n=200, seed=7)
    # Initial guess for hadeler: there are eigenvalues near 0 typically.
    # Probe a coarse grid for the smallest eigenvalue magnitude of T.
    print(f"\n=== probing hadeler for initial guesses ===")
    grid = np.linspace(-2.0, 2.0, 21)
    smallest = []
    for lg in grid:
        evals = la.eigvalsh(nlep.T(lg))
        smallest.append((lg, float(np.min(np.abs(evals)))))
    # Find local minima in |smallest eigenvalue|
    smallest_arr = np.array([s[1] for s in smallest])
    # Look for sign changes / local minima
    for i in range(1, len(smallest) - 1):
        if smallest_arr[i] < smallest_arr[i-1] and smallest_arr[i] < smallest_arr[i+1]:
            print(f"   candidate eigenvalue near lam={grid[i]:.3f}, "
                  f"min |eig| = {smallest_arr[i]:.3e}")
    # Take the candidate near 0 if any, else first found
    candidate = grid[int(np.argmin(smallest_arr))]
    print(f"   chosen initial guess: lam0 = {candidate:.3f}")
    results.append(run_problem(nlep, lam0_guess=float(candidate),
                                n_trials=3))

    # ----------------- Problem 3: 2D loaded plate -------------------- #
    nlep = Plate2D(N=35, sigma=100.0, kappa=1.0)
    # Fundamental mode of 2D Laplacian on (0,1)^2 with Dirichlet BCs
    # is at lam ~ 2*pi^2 ~ 19.74.
    results.append(run_problem(nlep, lam0_guess=22.0, n_trials=2))

    # --------------- Save results table ---------------- #
    with open("results_multi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "n", "sparse", "lam_true", "natural_floor",
                    "res_direct", "res_sketch", "t_direct_ms",
                    "t_sketch_ms", "direct_over_sketch"])
        for r in results:
            w.writerow([r["name"], r["n"], r["sparse"],
                        f"{r['lam_true']:.10f}",
                        f"{r['natural_floor']:.3e}",
                        f"{r['res_direct']:.3e}",
                        f"{r['res_sketch']:.3e}",
                        f"{r['t_direct_ms']:.2f}",
                        f"{r['t_sketch_ms']:.2f}",
                        f"{r['ratio']:.2f}"])

    # Save alpha-measurement table separately
    with open("results_alpha.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "n", "alpha_random_baseline_1_over_sqrtn",
                    "alpha_loose_median (LSQR-stopped at tau=1e-4)",
                    "alpha_loose_max",
                    "alpha_tight_median (LSQR over-converged, tau=1e-12)",
                    "alpha_tight_max",
                    "alpha_direct_median (LU back-sub residual)"])
        for r in results:
            w.writerow([r["name"], r["n"],
                        f"{r['alpha_random_baseline']:.3e}",
                        f"{r['alpha_loose_median']:.3e}",
                        f"{r['alpha_loose_max']:.3e}",
                        f"{r['alpha_tight_median']:.3e}",
                        f"{r['alpha_tight_max']:.3e}",
                        f"{r['alpha_direct_median']:.3e}"])

    # --------------- Print summary table ---------------- #
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    hdr = ("problem", "n", "natural_floor", "res(dir)", "res(sk)",
            "t_dir [ms]", "t_sk [ms]", "dir/sk")
    print(f"{hdr[0]:<28s} {hdr[1]:>6s} {hdr[2]:>15s} {hdr[3]:>10s} "
          f"{hdr[4]:>10s} {hdr[5]:>11s} {hdr[6]:>11s} {hdr[7]:>8s}")
    for r in results:
        print(f"{r['name']:<28s} {r['n']:>6d} "
              f"{r['natural_floor']:>15.3e} "
              f"{r['res_direct']:>10.3e} {r['res_sketch']:>10.3e} "
              f"{r['t_direct_ms']:>11.2f} {r['t_sketch_ms']:>11.2f} "
              f"{r['ratio']:>8.2f}")

    print("\n" + "=" * 100)
    print("ALPHA (structural-noise alignment) -- tests Assumption 4.5 of the paper")
    print("=" * 100)
    print(f"{'problem':<28s} {'n':>6s} {'1/sqrt(n)':>12s} "
          f"{'alpha_loose':>14s} {'alpha_tight':>14s} {'alpha_direct':>14s}")
    for r in results:
        print(f"{r['name']:<28s} {r['n']:>6d} "
              f"{r['alpha_random_baseline']:>12.3e} "
              f"{r['alpha_loose_median']:>14.3e} "
              f"{r['alpha_tight_median']:>14.3e} "
              f"{r['alpha_direct_median']:>14.3e}")
    print("    (loose   = LSQR stopped at tau=1e-4: residual reflects Krylov dynamics)")
    print("    (tight   = LSQR over-converges at tau=1e-12: residual ~ machine eps)")
    print("    (direct  = LU back-substitution residual ~ machine eps)")

    # --------------- Make scaling figure --------------- #
    # Show direct vs sketch timing across problems
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    names_plot = [r["name"].split("(")[0] + f"\nn={r['n']}" for r in results]
    x = np.arange(len(results))
    ax.bar(x - 0.2, [r["t_direct_ms"] for r in results], 0.4,
           label="direct LU/Cholesky", color="C0")
    ax.bar(x + 0.2, [r["t_sketch_ms"] for r in results], 0.4,
           label="sketch-and-precondition", color="C3")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names_plot, fontsize=8)
    ax.set_ylabel("median per-step time (ms, log scale)")
    ax.set_title("Inner-solver wall time per outer iteration")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    plt.savefig("figure_scaling.png", dpi=150)
    plt.close(fig)

    # --------------- Make alpha figure --------------- #
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    n_arr = np.array([r["n"] for r in results])
    a_loose = np.array([r["alpha_loose_median"] for r in results])
    a_tight = np.array([r["alpha_tight_median"] for r in results])
    a_baseline = 1.0 / np.sqrt(n_arr)

    sort_idx = np.argsort(n_arr)
    ax.loglog(n_arr[sort_idx], a_loose[sort_idx], "o-", label=r"observed $\alpha$ (LSQR, $\tau=10^{-4}$)", color="C3")
    ax.loglog(n_arr[sort_idx], a_tight[sort_idx], "s-", label=r"observed $\alpha$ (LSQR, $\tau=10^{-12}$)", color="C2")
    ax.loglog(n_arr[sort_idx], a_baseline[sort_idx], "k--",
              label=r"random baseline $1/\sqrt{n}$")
    # add labels for problem names
    for r in results:
        ax.annotate(r["name"].split("(")[0],
                    (r["n"], r["alpha_loose_median"]),
                    fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"problem size $n$")
    ax.set_ylabel(r"alignment factor $\alpha = |\langle u_1, r^{\mathrm{in}}\rangle|/\|r^{\mathrm{in}}\|$")
    ax.set_title(r"Structural-noise alignment $\alpha$ vs.\ random baseline $1/\sqrt{n}$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("figure_alpha.png", dpi=150)
    plt.close(fig)

    print("\nWrote figure_scaling.png, figure_alpha.png, "
          "results_multi.csv, results_alpha.csv")


if __name__ == "__main__":
    main()
