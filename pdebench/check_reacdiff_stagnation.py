"""Test the float32-stagnation explanation for frozen PDEBench 1D reaction-diffusion data.

Hypothesis: once the stored field is spatially uniform, the diffusion term is exactly
zero and each time step adds only rho*u*(1-u)*dt. In float32 that increment rounds to
zero once it falls below half a unit in the last place (ulp) of u, so the trajectory
stops changing. With a diffusion-limited step dt = K/nu this gives a freeze value u*
that satisfies rho*u*(1-u)*K/nu ~= ulp(u*)/2 and does not depend on the initial state.

For every frozen sample the script reports whether the frozen field is uniform and the
implied constant nu*dt = nu * (ulp(u*)/2) / (rho*u*(1-u*)), both under the file's
labels and with nu and rho swapped. On the 20 Test files the implied nu*dt is
2.0e-6 to 2.4e-6 for every frozen case.

    python check_reacdiff_stagnation.py --root . --samples 100
    python check_reacdiff_stagnation.py --root . --files ReacDiff_Nu1.0_Rho5.0.hdf5
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
import numpy as np
import h5py

PAT = re.compile(r"Nu([0-9.]+?)_Rho([0-9]+\.[0-9]+)")


def implied_nu_dt(nu, rho, u):
    half_ulp = float(np.spacing(np.float32(u))) / 2.0
    return nu * half_ulp / (rho * u * (1.0 - u))


def scan(path: Path, n: int, tol: float):
    nu, rho = (float(x) for x in PAT.search(path.name).groups())
    rows = []
    with h5py.File(path, "r") as f:
        d = f["tensor"]
        samples = [np.asarray(d, dtype=np.float64)] if d.ndim == 2 else \
                  (np.asarray(d[i], dtype=np.float64) for i in range(min(n, d.shape[0])))
        for i, u in enumerate(samples):
            step = np.abs(np.diff(u, axis=0)).max(axis=1)
            idx = np.flatnonzero(step < tol)
            if not (idx.size and np.all(step[idx[0]:] < tol)):
                continue
            k = int(idx[0]) + 1
            v = u[k]; m = float(v.mean())
            rows.append(dict(file=path.name, sample=i, nu=nu, rho=rho, freeze_index=k,
                             mean_at_freeze=m, range_at_freeze=float(v.max() - v.min()),
                             nu_dt_as_labelled=implied_nu_dt(nu, rho, m),
                             nu_dt_if_swapped=implied_nu_dt(rho, nu, m)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--files", nargs="*", default=None,
                    help="default: every ReacDiff_*.hdf5 under --root")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--csv", default="reacdiff_stagnation.csv")
    a = ap.parse_args()
    root = Path(a.root)
    paths = [p for name in a.files for p in root.rglob(name)] if a.files \
        else sorted(root.rglob("ReacDiff_*.hdf5"))
    allrows = []
    for p in paths:
        rows = scan(p, a.samples, a.tol)
        allrows += rows
        if not rows:
            print(f"{p.name:40s} no frozen trajectories"); continue
        r = np.array([x["range_at_freeze"] for x in rows])
        lab = np.array([x["nu_dt_as_labelled"] for x in rows])
        sw = np.array([x["nu_dt_if_swapped"] for x in rows])
        print(f"{p.name:40s} frozen {len(rows):4d}  uniform at freeze {np.mean(r < 1e-6):5.0%}  "
              f"nu*dt labelled {np.median(lab):.2e}  swapped {np.median(sw):.2e}")
    if allrows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
        print(f"\nwritten: {a.csv}")


if __name__ == "__main__":
    main()
