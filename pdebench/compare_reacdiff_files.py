"""Check whether two PDEBench 1D reaction-diffusion files contain the same data.

The stagnation scan found that ReacDiff_Nu0.5_Rho{r} and ReacDiff_Nu2.0_Rho{r} freeze at
identical values with identical freeze-index distributions for every rho. This script
compares the stored fields sample by sample.

    python compare_reacdiff_files.py --root . --samples 50
"""
import argparse
from pathlib import Path
import numpy as np
import h5py

PAIRS = [("ReacDiff_Nu0.5_Rho{}.hdf5", "ReacDiff_Nu2.0_Rho{}.hdf5"),
         ("ReacDiff_Nu0.5_Rho{}.hdf5", "ReacDiff_Nu1.0_Rho{}.hdf5")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--samples", type=int, default=50)
    a = ap.parse_args()
    root = Path(a.root)
    for rho in ("1.0", "2.0", "5.0", "10.0"):
        for fa, fb in PAIRS:
            pa = next(root.rglob(fa.format(rho))); pb = next(root.rglob(fb.format(rho)))
            with h5py.File(pa, "r") as A, h5py.File(pb, "r") as B:
                ua, ub = A["tensor"], B["tensor"]
                n = min(a.samples, ua.shape[0], ub.shape[0])
                init_same = all(np.array_equal(ua[i, 0], ub[i, 0]) for i in range(n))
                diff = [float(np.abs(np.asarray(ua[i], float) - np.asarray(ub[i], float)).max())
                        for i in range(n)]
                attrs = ({k: A.attrs[k] for k in A.attrs}, {k: B.attrs[k] for k in B.attrs})
            print(f"{pa.name:28s} vs {pb.name:28s} same initial fields: {init_same}  "
                  f"identical trajectories: {sum(d == 0 for d in diff)}/{n}  "
                  f"max |diff| median {np.median(diff):.2e}  attrs {attrs[0]} | {attrs[1]}")


if __name__ == "__main__":
    main()
