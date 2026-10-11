"""Print the internal structure of the PDEBench files this project uses.

Usage (from the data folder):
    python inspect_pdebench.py --root .
    python inspect_pdebench.py --root . --files 2D_rdb_NA_NA.h5

For each file it prints the HDF5 tree (groups, dataset names, shapes, dtypes),
the attributes, and a small numeric summary of the main field, so the loader
can be written against the actual layout rather than against assumptions.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

# One representative file per equation used in the paper.
DEFAULT = [
    "1D_Advection_Sols_beta0.4.hdf5",
    "1D_Burgers_Sols_Nu0.001.hdf5",
    "ReacDiff_Nu1.0_Rho1.0.hdf5",
    "1D_diff-sorp_NA_NA.h5",
    "2D_diff-react_NA_NA.h5",
    "2D_rdb_NA_NA.h5",
    "2D_DarcyFlow_beta1.0_Train.hdf5",
]

def find(root: Path, name: str) -> Path | None:
    hits = list(root.rglob(name))
    return hits[0] if hits else None

def describe(path: Path, max_items: int = 25, peek: bool = True) -> None:
    import h5py, numpy as np
    print("=" * 78)
    print(path)
    print(f"  size: {path.stat().st_size / 1e9:.2f} GB")
    with h5py.File(path, "r") as f:
        if f.attrs:
            print("  file attributes:")
            for k, v in f.attrs.items():
                print(f"    {k} = {v}")
        shown = 0
        def visit(name, obj):
            nonlocal shown
            if shown >= max_items:
                return None
            if isinstance(obj, h5py.Dataset):
                print(f"  dataset  {name:<40s} shape={obj.shape}  dtype={obj.dtype}")
                for k, v in obj.attrs.items():
                    print(f"           attr {k} = {v}")
                shown += 1
            else:
                print(f"  group    {name}")
                for k, v in obj.attrs.items():
                    print(f"           attr {k} = {v}")
            return None
        f.visititems(visit)
        if shown >= max_items:
            print(f"  ... (listing stopped after {max_items} datasets; "
                  f"groups of samples are usually repetitive)")
        if peek:
            # summarise the largest dataset: this is normally the solution field
            big, big_size = None, -1
            def pick(name, obj):
                nonlocal big, big_size
                if isinstance(obj, h5py.Dataset) and obj.size > big_size:
                    big, big_size = name, obj.size
                return None
            f.visititems(pick)
            if big is not None:
                d = f[big]
                sl = tuple(slice(0, 1) if i < d.ndim - 1 else slice(None) for i in range(d.ndim))
                sample = np.asarray(d[sl])
                print(f"  main field: {big}  shape={d.shape}")
                print(f"    first slice: min={float(np.nanmin(sample)):.4g}  "
                      f"max={float(np.nanmax(sample)):.4g}  mean={float(np.nanmean(sample)):.4g}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--max-items", type=int, default=25)
    args = ap.parse_args()
    root = Path(args.root)
    names = args.files if args.files else DEFAULT
    for name in names:
        p = find(root, name)
        if p is None:
            print("=" * 78); print(f"{name}: not found under {root}")
            continue
        try:
            describe(p, max_items=args.max_items)
        except Exception as exc:                      # noqa: BLE001
            print(f"  could not read: {exc}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
