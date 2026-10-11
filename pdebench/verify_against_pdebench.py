"""Cross-check our solvers against PDEBench reference solutions.

advection : u_t + beta u_x = 0 on a periodic domain. The exact solution is a
            pure shift, so the check is exact up to interpolation error and
            validates the time axis, the domain length, and the sign of beta.
reacdiff  : u_t = Nu u_xx + rho u (1 - u) (Fisher-KPP). Checked with an explicit
            step on the PDEBench grid; the tolerance is looser because the
            reference uses a different integrator.

Usage:
    python verify_against_pdebench.py --advection "1D\\Advection\\Train\\1D_Advection_Sols_beta0.4.hdf5"
    python verify_against_pdebench.py --reacdiff  "1D\\ReactionDiffusion\\Train\\ReacDiff_Nu1.0_Rho1.0.hdf5"
"""
from __future__ import annotations
import argparse
import numpy as np
from pdebench_io import open_pde, load_sample


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def check_advection(path, n_samples=3, checkpoints=(10, 50, 100, 200)):
    info = open_pde(path)
    beta = float(info.params.get("beta", np.nan))
    print(repr(info))
    for s_idx in range(n_samples):
        s = load_sample(info, s_idx)
        u, x, t = s["u"], s["x"], s["t"]
        L = float(x[-1] - x[0]) + float(x[1] - x[0])
        k = 2 * np.pi * np.fft.fftfreq(len(x), d=float(x[1] - x[0]))
        u0h = np.fft.fft(u[0])
        errs = []
        for n in checkpoints:
            if n >= u.shape[0]:
                continue
            shift = beta * float(t[n] - t[0])
            pred = np.real(np.fft.ifft(u0h * np.exp(-1j * k * shift)))   # exact transport
            errs.append((n, rel_l2(pred, u[n])))
        line = "  ".join(f"t[{n}]={e:.2e}" for n, e in errs)
        print(f"  sample {s_idx}: relative L2 error   {line}")
    print(f"  domain length L={L:.4f}, beta={beta}")
    print("  interpretation: errors near machine precision confirm the grid,")
    print("  the time axis, and the direction of transport.")


def check_reacdiff(path, n_samples=2, horizon=20, safety=0.1):
    """Fisher-KPP check with an IMEX spectral scheme.

    Explicit Euler is unusable here: with Nu of order one and h about 1e-3 the
    diffusive stability limit is dt < h^2 / (2 Nu) ~ 5e-7. Diffusion is
    therefore integrated exactly in Fourier space (integrating factor) and only
    the reaction term is advanced explicitly, which needs dt < 1 / rho.
    """
    info = open_pde(path)
    nu = float(info.params.get("Nu", np.nan)); rho = float(info.params.get("rho", np.nan))
    print(repr(info))
    for s_idx in range(n_samples):
        s = load_sample(info, s_idx)
        u, x, t = s["u"], s["x"], s["t"]
        h = float(x[1] - x[0]); dt_ref = float(t[1] - t[0])
        sub = max(1, int(np.ceil(dt_ref / (safety / max(rho, 1e-12)))))
        dt = dt_ref / sub
        k = 2 * np.pi * np.fft.fftfreq(len(x), d=h)
        decay = np.exp(-nu * k ** 2 * dt)                 # exact diffusion over dt
        v = u[0].astype(float).copy()
        for _ in range(horizon * sub):
            v = v + dt * rho * v * (1 - v)                # reaction, explicit
            v = np.real(np.fft.ifft(np.fft.fft(v) * decay))
        print(f"  sample {s_idx}: relative L2 error at t[{horizon}] = {rel_l2(v, u[horizon]):.3e}"
              f"   (substeps per snapshot: {sub})")
    print(f"  Nu={nu}, rho={rho}; a few per cent is expected from the different integrator.")
    print("  A large error would mean the reaction term is not rho*u*(1-u).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--advection"); ap.add_argument("--reacdiff")
    a = ap.parse_args()
    if a.advection: check_advection(a.advection)
    if a.reacdiff: check_reacdiff(a.reacdiff)
    if not (a.advection or a.reacdiff): ap.print_help()
