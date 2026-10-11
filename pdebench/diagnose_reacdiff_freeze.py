"""Check whether PDEBench 1D reaction-diffusion trajectories freeze before t_end.

For each sample, find the first snapshot after which the field stops changing,
and record the time index, the time, and the spatial mean at that point.

    python diagnose_reacdiff_freeze.py --root . --samples 200

How to read the result:
  same time index across samples, different means  -> a time-axis or save issue
  same mean across samples, different times        -> a value-triggered stop
  no freeze in most samples                         -> isolated to a few samples
"""
import argparse
from pathlib import Path
import numpy as np
import h5py

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--file", default="ReacDiff_Nu1.0_Rho1.0.hdf5")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-7)
    a = ap.parse_args()
    path = next(Path(a.root).rglob(a.file))
    with h5py.File(path, "r") as f:
        t = np.asarray(f["t-coordinate"]); d = f["tensor"]
        n = min(a.samples, d.shape[0]); rows = []
        for i in range(n):
            u = np.asarray(d[i], dtype=float)                 # (n_t, n_x)
            step = np.abs(np.diff(u, axis=0)).max(axis=1)     # largest change per step
            frozen = np.flatnonzero(step < a.tol)
            k = int(frozen[0]) + 1 if frozen.size and np.all(step[frozen[0]:] < a.tol) else None
            rows.append((i, k, None if k is None else float(t[k]),
                         None if k is None else float(u[k].mean()), float(u[0].mean())))
    hit = [r for r in rows if r[1] is not None]
    print(f"file: {path.name}   samples checked: {n}   frozen before the end: {len(hit)}")
    if hit:
        idx = np.array([r[1] for r in hit]); mean = np.array([r[3] for r in hit])
        print(f"  freeze index: min {idx.min()}, median {int(np.median(idx))}, max {idx.max()}  (of {u.shape[0]-1})")
        print(f"  freeze time : min {t[idx.min()]:.3f}, max {t[idx.max()]:.3f}")
        print(f"  mean at freeze: min {mean.min():.5f}, median {np.median(mean):.5f}, max {mean.max():.5f}")
        print(f"  spread of mean at freeze: {mean.std():.2e}   spread of freeze index: {idx.std():.2f}")
        print("  first ten:")
        for r in hit[:10]:
            print(f"    sample {r[0]:4d}: freezes at index {r[1]:3d} (t={r[2]:.3f}), "
                  f"mean there {r[3]:.5f}, initial mean {r[4]:.5f}")
    print(f"  time axis: {len(t)} entries for {u.shape[0]} snapshots, t[0]={t[0]:.3f}, "
          f"t[{u.shape[0]-1}]={t[u.shape[0]-1]:.3f}, t[-1]={t[-1]:.3f}")

if __name__ == "__main__":
    main()
