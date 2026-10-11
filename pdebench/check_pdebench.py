"""Check downloaded PDEBench files against the official MD5 list.

Usage:
    python check_pdebench.py --root /path/to/your/pdebench/data \
                             --csv  /path/to/PDEBench/pdebench/data_download/pdebench_data_urls.csv

Reports, for every file listed in the CSV: missing, size mismatch (partial
download), MD5 mismatch (corrupted), unreadable HDF5, or OK. Writes resume
commands for everything that failed, so nothing complete is downloaded twice.
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path

def md5sum(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder where the data were downloaded")
    ap.add_argument("--csv", required=True, help="pdebench_data_urls.csv from the repository")
    ap.add_argument("--pde", default=None, help="optional filter, e.g. 2D_ReacDiff")
    ap.add_argument("--open-hdf5", action="store_true", help="also try opening each file with h5py")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install pandas"); return 1

    df = pd.read_csv(args.csv)
    if args.pde:
        df = df[df["PDE"].str.lower() == args.pde.lower()]
    root = Path(args.root)

    ok, bad = [], []
    for _, row in df.iterrows():
        target = root / str(row["Path"]) / str(row["Filename"])
        if not target.exists():
            # the same file is sometimes saved directly under the root
            alt = root / str(row["Filename"])
            target = alt if alt.exists() else target
        if not target.exists():
            print(f"MISSING   {row['Filename']}")
            bad.append((row, target, "missing")); continue

        size = target.stat().st_size
        digest = md5sum(target)
        if digest != str(row["MD5"]).strip():
            print(f"CORRUPT   {row['Filename']}  ({size/1e6:.1f} MB, md5 {digest[:8]}... "
                  f"expected {str(row['MD5'])[:8]}...)")
            bad.append((row, target, "md5")); continue

        if args.open_hdf5:
            try:
                import h5py
                with h5py.File(target, "r") as f:
                    _ = list(f.keys())
            except Exception as exc:                      # noqa: BLE001
                print(f"UNREADABLE {row['Filename']}: {exc}")
                bad.append((row, target, "hdf5")); continue

        print(f"OK        {row['Filename']}  ({size/1e6:.1f} MB)")
        ok.append(row)

    print(f"\n{len(ok)} complete, {len(bad)} to fix")
    if bad:
        sh = root / "refetch.sh"
        with open(sh, "w", newline="\n") as fh:
            fh.write("#!/bin/sh\n# Resume or re-download the files that failed the check.\n")
            for row, target, why in bad:
                if why in ("md5", "hdf5"):
                    # a corrupted file cannot be resumed reliably: remove it first
                    fh.write(f'rm -f "{target}"\n')
                fh.write(f'mkdir -p "{target.parent}"\n')
                fh.write(f'curl -L -C - -o "{target}" "{row["URL"]}"\n')
        bat = root / "refetch.bat"
        with open(bat, "w", newline="\r\n") as fh:
            fh.write("@echo off\nREM Resume or re-download the files that failed the check.\n")
            for row, target, why in bad:
                win = str(target).replace("/", "\\")
                if why in ("md5", "hdf5"):
                    fh.write(f'del /f /q "{win}"\n')
                fh.write(f'if not exist "{str(target.parent)}" mkdir "{str(target.parent)}"\n')
                fh.write(f'curl -L -C - -o "{win}" "{row["URL"]}"\n')
            fh.write("echo done\n")
        print(f"written: {bat} and {sh}")
        print("Run refetch.bat (repeat it if the connection drops; curl -C - resumes),")
        print("then run this checker again to confirm the MD5 sums.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
