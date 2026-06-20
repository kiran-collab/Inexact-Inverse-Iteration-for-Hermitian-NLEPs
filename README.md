# Inexact Inverse Iteration for Hermitian Non-Linear Eigenvalue Problems

Official repository for the paper **"Inexact Inverse Iteration with Sketched Inner
Solvers for Eigenvector Refinement in Hermitian Nonlinear Eigenvalue Problems"**
(submitted to SIAM). This repo contains the reference implementation and the
numerical experiments reported in the paper.

---

## The Problem

A **nonlinear eigenvalue problem (NLEP)** asks for a scalar `λ` (the eigenvalue)
and a nonzero vector `v` (the eigenvector) such that

```
T(λ) v = 0,
```

where `T(λ)` is an `n × n` matrix whose entries depend **nonlinearly** on `λ`.
Unlike the standard linear problem `A v = λ v`, here `λ` can enter through
rational or transcendental functions, so the usual linear-algebra eigensolvers
do not apply directly. We focus on the **Hermitian** case, where `T(λ) = T(λ)*`
for real `λ`, which guarantees real eigenvalues and gives the structure our
method exploits.

The practical difficulty: given a *good but imperfect* approximate eigenpair
`(λ̂, v̂)` — for example from a coarse global solver — how do we **refine the
eigenvector** to high accuracy cheaply, without re-solving the whole problem and
without forming/factorizing `T(λ̂)` exactly at large scale?

---

## The Solution

The method combines two ideas:

1. **Inexact inverse iteration (Algorithm 4.2).**
   Inverse iteration refines the eigenvector by repeatedly solving
   `T(λ̂) w = v_k` and renormalizing. We never solve this system exactly —
   each inner solve is run only to a **relative residual tolerance `τ`**. The
   paper shows that for a Hermitian NLEP the outer error contracts at a rate
   governed by the spectral gap ratio `γ = |λ₁| / |λ₂|`, so a modest inner
   tolerance already yields a high-accuracy eigenvector.

2. **Sketch-and-precondition LSQR inner solver (Algorithm 6.1).**
   Instead of an exact LU factorization, each inner least-squares solve is
   accelerated by **randomized sketching**: a random sketch of dimension
   `s ≈ 2n` builds a cheap preconditioner, and LSQR then converges in very few
   iterations. This replaces an `O(n³)` factorization with much cheaper
   sketched solves and is what makes the method scale to large/sparse problems.

The result is an eigenvector-refinement scheme whose cost per outer step is a
few sketched LSQR solves, with a rigorous accuracy bound

```
||T(λ̂) v_∞|| / ||T(λ̂)||  ≤  2τ / ((1 − γ) |λ₁| cos θ₀).
```

---

## The Input (experimental setup)

The experiments use **no external dataset** — all matrices are generated
programmatically from standard NLEVP benchmark problems, and the approximate
input eigenpair is **synthesized from a computed reference** so the refinement
error can be measured exactly.

**Problem classes:**

| Problem         | Type                              | Size(s)            | Structure                       |
|-----------------|-----------------------------------|--------------------|---------------------------------|
| `loaded_string` | 1D rational Hermitian NLEP        | n ∈ {100,500,2000} | tridiagonal `K` + rank-1 `C`    |
| `hadeler`       | transcendental Hermitian NLEP     | n = 200            | dense                           |
| `plate2d`       | 2D rational Hermitian NLEP        | n = 50² = 2500     | sparse 5-point Laplacian + rank-1 |

For `loaded_string`, `T(λ) = K − λM + (λ/(λ−σ))·C` with `σ = 1.0`, `κ = 1.0`.

**How each input eigenpair is produced:**

1. Compute a **reference eigenvalue** `λ_true` to machine precision
   (`loaded_string` via quadratic linearization; the others via Newton
   refinement) and its eigenvector `v_true`.
2. **Perturb it** to form the algorithm's input:
   - perturbed eigenvalue `λ̂ = λ_true ± δλ`, with `δλ = 1e-6`
   - perturbed eigenvector `v̂` at angle `arcsin(η_v)` from `v_true`, with
     `η_v = 1e-4` (so the input residual is ≈ `η_v`).
3. Run Algorithm 4.2 from `(λ̂, v̂)` and measure the refined residual.

The runs are **reproducible**: the random number generator is seeded
(`np.random.default_rng(seed=2026)`). Default outer iterations `K_outer = 5`.

---

## Repository Layout

```
numerical_experiments.py        main experiments + 3 sensitivity studies (Section 8)
multi_problem_experiments.py    multi-problem scaling study (direct vs sketched solver)
index.txt                       full file-by-file index of the project
results/                        output CSV tables
figures/                        output plots (Figures 1–2, alpha & scaling studies)
```

See [index.txt](index.txt) for a complete description of every file.

## Running

Requires Python 3 with `numpy`, `scipy`, and `matplotlib`.

```bash
python numerical_experiments.py        # main + sensitivity experiments
python multi_problem_experiments.py    # multi-problem scaling study
```
