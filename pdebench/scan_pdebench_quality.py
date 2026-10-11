"""Scan PDEBench files for frozen trajectories and other data defects.

Works for both layouts (flat 'tensor' files and per-sample groups) and any
dimension. For every time-dependent file it reports, over the samples checked:

  - samples whose stored field stops changing before the final snapshot,
    with the freeze index and the spatial mean at the freeze;
  - consecutive identical snapshots anywhere in a trajectory;
  - NaN or Inf values.

Steady problems (no time axis, e.g. Darcy) are reported and skipped.

    python scan_pdebench_quality.py --root .                     # all known files
    python scan_pdebench_quality.py --root . --samples 50        # faster
    python scan_pdebench_quality.py --root . --files 2D_rdb_NA_NA.h5
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import h5py

DEFAULT = [
    # 1D
    "1D_Advection_Sols_beta0.4.hdf5",
    "1D_Burgers_Sols_Nu0.001.hdf5",
    "1D_diff-sorp_NA_NA.h5",
    "ReacDiff_Nu0.5_Rho1.0.hdf5", "ReacDiff_Nu1.0_Rho1.0.hdf5",
    "ReacDiff_Nu1.0_Rho5.0.hdf5", "ReacDiff_Nu5.0_Rho10.0.hdf5",
    # 2D
    "2D_diff-react_NA_NA.h5",
    "2D_rdb_NA_NA.h5",
    "2D_DarcyFlow_beta1.0_Train.hdf5",
]

def iter_samples(f, n):
    """Yield (index, array with time on axis 0) for up to n samples."""
    if "tensor" in f:
        d = f["tensor"]
        if d.ndim >= 2 and d.shape[1] == 1 and "nu" in f:      # Darcy: (N, 1, nx, ny)
            return
        for i in range(min(n, d.shape[0])):
            yield i, np.asarray(d[i], dtype=np.float64)
    else:
        keys = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))
        for i, k in enumerate(keys[:n]):
            yield i, np.asarray(f[k]["data"], dtype=np.float64)

def is_steady(f):
    return "tensor" in f and "nu" in f

def scan_file(path: Path, n: int, tol: float):
    with h5py.File(path, "r") as f:
        if is_steady(f):
            return dict(file=path.name, kind="steady", checked=0)
        frozen, dup_samples, bad = [], 0, 0
        n_t = None; checked = 0
        for i, u in iter_samples(f, n):
            checked += 1
            n_t = u.shape[0]
            if not np.all(np.isfinite(u)):
                bad += 1
            flat = u.reshape(n_t, -1)
            step = np.abs(np.diff(flat, axis=0)).max(axis=1)
            if np.any(step == 0.0):
                dup_samples += 1
            idx = np.flatnonzero(step < tol)
            if idx.size and np.all(step[idx[0]:] < tol):
                k = int(idx[0]) + 1
                if k < n_t - 1:                                 # freezes before the last snapshot
                    frozen.append((i, k, float(flat[k].mean()), float(flat[0].mean())))
        out = dict(file=path.name, kind="transient", checked=checked, n_snapshots=n_t,
                   frozen=len(frozen), with_repeated_frames=dup_samples, non_finite=bad)
        if frozen:
            ks = np.array([r[1] for r in frozen]); ms = np.array([r[2] for r in frozen])
            out.update(freeze_index_min=int(ks.min()), freeze_index_median=int(np.median(ks)),
                       freeze_index_max=int(ks.max()), mean_at_freeze_median=float(np.median(ms)),
                       mean_at_freeze_spread=float(ms.std()), freeze_index_spread=float(ks.std()),
                       examples=frozen[:5])
        return out

def verdict(r):
    if r["kind"] == "steady":
        return "steady problem, not applicable"
    if r["checked"] == 0:
        return "no samples read"
    if r["frozen"] == 0:
        return "no frozen trajectories"
    frac = r["frozen"] / r["checked"]
    if r["mean_at_freeze_spread"] < 1e-3 and r["freeze_index_spread"] > 1:
        pattern = "value-triggered (same mean, different times)"
    elif r["freeze_index_spread"] < 1:
        pattern = "time-triggered (same index)"
    else:
        pattern = "mixed"
    return f"{r['frozen']}/{r['checked']} frozen ({100*frac:.0f}%), {pattern}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--csv", default="pdebench_quality_scan.csv")
    a = ap.parse_args()
    root = Path(a.root); names = a.files or DEFAULT
    rows = []
    for name in names:
        hits = list(root.rglob(name))
        if not hits:
            print(f"{name:45s} not found"); continue
        try:
            r = scan_file(hits[0], a.samples, a.tol)
        except Exception as exc:                               # noqa: BLE001
            print(f"{name:45s} could not be read: {exc}"); continue
        rows.append(r)
        print(f"{name:45s} {verdict(r)}")
        if r.get("frozen"):
            print(f"{'':45s}   freeze index {r['freeze_index_min']}-{r['freeze_index_max']} "
                  f"of {r['n_snapshots']-1}, mean at freeze {r['mean_at_freeze_median']:.5f} "
                  f"(spread {r['mean_at_freeze_spread']:.1e})")
            for (i, k, m, m0) in r["examples"][:3]:
                print(f"{'':45s}   sample {i}: index {k}, mean {m:.5f}, initial mean {m0:.5f}")
        if r.get("with_repeated_frames"):
            print(f"{'':45s}   samples containing exactly repeated frames: {r['with_repeated_frames']}")
        if r.get("non_finite"):
            print(f"{'':45s}   samples with NaN/Inf: {r['non_finite']}")
    if rows:
        keys = sorted({k for r in rows for k in r if k != "examples"})
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"\nwritten: {a.csv}")

if __name__ == "__main__":
    main()
