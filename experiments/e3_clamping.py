"""E3: state interventions do(u|_S := v) and the capacity of the clamped set.

Case C (Section 3 of the paper). A steady diffusion problem with a constant source and zero
Dirichlet data on the outer boundary is solved twice: once as it stands, and
once with the state clamped to zero on a set S. The effect is measured at a
fixed observation window.

Prediction: in 1D a point has positive capacity, so the effect converges to a
non-zero limit. In 2D a point has zero capacity, so the continuum effect is
zero, yet the discrete effect of clamping one cell decays only like 1/|log h|.
A line in 2D has positive capacity and behaves like the 1D point.
"""
import numpy as np, csv
import scipy.sparse as sp
import scipy.sparse.linalg as spla

D = 1.0
SRC = 1.0
OBS_1D, OBS_2D = 0.25, (0.25, 0.5)     # observation point
CLAMP = 0.5                            # clamped location (centre)

def solve_1d(N, clamp=False):
    h = 1.0 / (N + 1)
    x = (np.arange(1, N + 1)) * h                       # interior nodes, u=0 at both ends
    A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format="lil") * (D / h ** 2)
    b = np.full(N, SRC)
    if clamp:
        i = int(np.argmin(np.abs(x - CLAMP)))
        A[i, :] = 0; A[i, i] = 1.0; b[i] = 0.0          # do(u = 0) at one node
    u = spla.spsolve(A.tocsr(), b)
    j = int(np.argmin(np.abs(x - OBS_1D)))
    return float(u[j])

def solve_2d(N, clamp=None):
    h = 1.0 / (N + 1)
    x = np.arange(1, N + 1) * h
    I = sp.identity(N, format="csr")
    T1 = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format="csr")
    A = (sp.kron(I, T1) + sp.kron(T1, I)) * (D / h ** 2)
    A = A.tolil()
    b = np.full(N * N, SRC)
    ic = int(np.argmin(np.abs(x - CLAMP)))
    if clamp == "point":
        idx = [ic * N + ic]
    elif clamp == "line":
        idx = [ic * N + j for j in range(N)]
    else:
        idx = []
    for k in idx:
        A[k, :] = 0; A[k, k] = 1.0; b[k] = 0.0
    u = spla.spsolve(A.tocsr(), b)
    i = int(np.argmin(np.abs(x - OBS_2D[0]))); j = int(np.argmin(np.abs(x - OBS_2D[1])))
    return float(u.reshape(N, N)[i, j])

if __name__ == "__main__":
    rows = []
    print("1D: clamping one node")
    for N in (63, 127, 255, 511, 1023):
        base, cl = solve_1d(N), solve_1d(N, clamp=True)
        tau = cl - base
        rows.append(dict(case="1D point", N=N, baseline=base, clamped=cl, effect=tau))
        print(f"  N={N:5d}  effect={tau: .6f}  (relative {tau/base*100: .2f}%)")

    print("\n2D: clamping one cell")
    for N in (63, 127, 255, 511):
        base, cl = solve_2d(N), solve_2d(N, clamp="point")
        tau = cl - base
        rows.append(dict(case="2D point", N=N, baseline=base, clamped=cl, effect=tau))
        print(f"  N={N:5d}  effect={tau: .3e}  effect*log(1/h)={tau*np.log(N): .4f}")

    print("\n2D: clamping a line")
    for N in (63, 127, 255, 511):
        base, cl = solve_2d(N), solve_2d(N, clamp="line")
        tau = cl - base
        rows.append(dict(case="2D line", N=N, baseline=base, clamped=cl, effect=tau))
        print(f"  N={N:5d}  effect={tau: .6f}  (relative {tau/base*100: .2f}%)")

    # fit the predicted law tau = A / (log(1/h) + C) to the 2D point clamp
    pts = [(r["N"], r["effect"]) for r in rows if r["case"] == "2D point"]
    Ns = np.array([p[0] for p in pts], dtype=float)
    taus = np.array([abs(p[1]) for p in pts])
    slope, intercept = np.polyfit(np.log(Ns), 1.0 / taus, 1)
    A, C = 1.0 / slope, intercept / slope
    print(f"\nfit of the 2D point clamp to tau = A / (log(1/h) + C):  A={A:.4f}, C={C:.3f}")
    for n, t in zip(Ns, taus):
        print(f"  N={int(n):4d}  measured={t:.4e}  fitted={A/(np.log(n)+C):.4e}")
    print("  extrapolation: the effect is still "
          f"{A/(np.log(1e6)+C):.2e} at N = 10^6, i.e. it never becomes negligible in practice.")

    print("\nA point clamp in 2D has no continuum effect, but the discrete effect")
    print("vanishes only logarithmically: the product with log(1/h) is roughly constant.")
    with open("e3_clamping.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written: e3_clamping.csv")
