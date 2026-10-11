"""E4: nonlinear bistable dynamics and propagation failure.

Case D (Section 3 of the paper). The system is the bistable (Nagumo) equation

    u_t = D u_xx + u (1 - u) (u - a),     0 < a < 1/2,

whose two stable states are u = 0 and u = 1. The intervention sets the state to
one on a small interval, do(u|_S := 1) at t = 0, which in the continuum starts a
front that invades the zero state. On a lattice the front can be pinned: below a
critical coupling D/h^2 it does not move at all (Keener 1987). The effect of the
same intervention is then qualitatively wrong, not merely biased.

Diffusion is treated implicitly (tridiagonal solve, finite differences, so that
the lattice structure that causes pinning is preserved) and the reaction term
explicitly.
"""
import numpy as np, csv
import scipy.sparse as sp
import scipy.sparse.linalg as spla

D, A_PAR, T = 2.0e-5, 0.25, 600.0
SEED_WIDTH = 0.20            # the intervened interval [0, SEED_WIDTH]

def run(N, dt=0.05):
    h = 1.0 / N
    x = (np.arange(N) + 0.5) * h
    lap = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N), format="lil")
    lap[0, 0] = -1.0; lap[-1, -1] = -1.0                 # no-flux boundaries
    L = (D / h ** 2) * lap.tocsr()
    M = sp.identity(N, format="csr") - dt * L
    solve = spla.factorized(M.tocsc())
    u = np.where(x <= SEED_WIDTH, 1.0, 0.0)              # do(u := 1 on S)
    seeded = float(np.sum(u > 0.5) * h)                  # invaded length at t = 0
    nt = int(round(T / dt))
    for _ in range(nt):
        u = solve(u + dt * u * (1 - u) * (u - A_PAR))
    front = float(np.sum(u > 0.5) * h)                   # length of the invaded region
    return float(u.mean()), front, seeded, h

if __name__ == "__main__":
    rows = []
    print(f"bistable front, D={D}, a={A_PAR}, seeded on [0, {SEED_WIDTH}], T={T}")
    print(f"{'N':>6} {'h':>9} {'coupling':>10} {'seeded':>8} {'invaded':>9} {'growth':>9}  status")
    for N in (8, 12, 16, 24, 32, 48, 64, 128, 256, 512):
        eff, front, seeded, h = run(N)
        growth = front - seeded
        status = "pinned" if growth <= 0.02 else "propagating"
        rows.append(dict(N=N, h=h, coupling=D / h ** 2, effect=eff,
                         seeded=seeded, invaded=front, growth=growth, status=status))
        print(f"{N:6d} {h:9.5f} {D/h**2:10.4f} {seeded:8.4f} {front:9.4f} {growth:9.4f}  {status}")
    with open("e4_bistable.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    pinned = [r for r in rows if r["status"] == "pinned"]
    moving = [r for r in rows if r["status"] == "propagating"]
    if pinned and moving:
        Np, cp = max(r["N"] for r in pinned), max(r["coupling"] for r in pinned)
        Nm, cm = min(r["N"] for r in moving), min(r["coupling"] for r in moving)
        print(f"\nfront pinned for N <= {Np} (coupling <= {cp:.4f}); "
              f"propagating for N >= {Nm} (coupling >= {cm:.4f})")
        print(f"critical mesh width between h = {1/Nm:.4f} and h = {1/Np:.4f}")
        print("Below the threshold the intervention has no lasting effect in the discrete")
        print("model, while in the continuum the same intervention starts a travelling front:")
        print("the sign of the conclusion, not only its size, depends on the resolution.")
    print("written: e4_bistable.csv")
