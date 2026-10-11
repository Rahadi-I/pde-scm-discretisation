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
    A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format="lil", dtype=float) * (D / h ** 2)
    b = np.full(N, SRC)
    if clamp:
        i = int(np.argmin(np.abs(x - CLAMP)))
        A[i, :] = 0; A[i, i] = 1.0; b[i] = 0.0          # do(u = 0) at one node
    u = spla.spsolve(A.tocsr(), b)
    j = int(np.argmin(np.abs(x - OBS_1D)))
    return float(u[j])

def predicted_amplitude(M=400):
    """A = 2 pi u0(x*) G(x_obs, x*) of Proposition 3, from sine series on the unit square.

    u0 solves -D Lap u = SRC with zero Dirichlet data; G is the Green's function of
    -Lap (not -D Lap) with zero Dirichlet data. The clamp correction is
    -u0(x*) p_h with p_h ~ 2 pi G / (log(1/h) + C)."""
    m = np.arange(1, M + 1); MM, NN = np.meshgrid(m, m, indexing="ij")
    lam = np.pi ** 2 * (MM ** 2 + NN ** 2)
    phi = lambda x, y: 2 * np.sin(MM * np.pi * x) * np.sin(NN * np.pi * y)
    proj = 2 * ((1 - np.cos(MM * np.pi)) / (MM * np.pi)) * ((1 - np.cos(NN * np.pi)) / (NN * np.pi))
    u0 = float(np.sum(SRC / D * proj / lam * phi(CLAMP, CLAMP)))
    G = float(np.sum(phi(*OBS_2D) * phi(CLAMP, CLAMP) / lam))
    return 2 * np.pi * u0 * G, u0, G

def solve_2d(N, clamp=None):
    h = 1.0 / (N + 1)
    x = np.arange(1, N + 1) * h
    I = sp.identity(N, format="csr")
    T1 = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format="csr", dtype=float)
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

    # fit the predicted law tau = A / (log(1/h) + C), with h = 1/(N+1), to the 2D point clamp
    pts = [(r["N"], r["effect"]) for r in rows if r["case"] == "2D point"]
    Ns = np.array([p[0] for p in pts], dtype=float)
    taus = np.array([abs(p[1]) for p in pts])
    L = np.log(Ns + 1)                                   # log(1/h)
    slope, intercept = np.polyfit(L, 1.0 / taus, 1)
    A, C = 1.0 / slope, intercept / slope
    A_pred, u0, G = predicted_amplitude()
    print(f"\nfit of the 2D point clamp to tau = A / (log(1/h) + C):  A={A:.4f}, C={C:.3f}")
    print(f"predicted A = 2 pi u0(x*) G(x_obs,x*) = 2 pi * {u0:.6f} * {G:.6f} = {A_pred:.4f}"
          f"   (fit differs by {abs(A-A_pred)/A_pred*100:.2f}%)")
    for n, l, t in zip(Ns, L, taus):
        print(f"  N={int(n):4d}  measured={t:.4e}  fitted={A/(l+C):.4e}  ({abs(A/(l+C)-t)/t*100:.3f}%)")
    ext = A / (np.log(1e6 + 1) + C)
    print(f"  extrapolation: the effect is still {ext:.2e} at N = 10^6, "
          f"{ext/taus[0]:.2f} of its value at N = {int(Ns[0])}.")
    with open("e3_capacity_fit.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["A_fit", "C_fit", "A_predicted", "u0_at_clamp", "G_obs_clamp", "effect_at_N_1e6"])
        w.writerow([A, C, A_pred, u0, G, ext])

    print("\nA point clamp in 2D has no continuum effect, but the discrete effect")
    print("vanishes only logarithmically: the product with log(1/h) is roughly constant.")
    with open("e3_clamping.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written: e3_clamping.csv, e3_capacity_fit.csv")
