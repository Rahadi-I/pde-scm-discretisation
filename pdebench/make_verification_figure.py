"""Figure: solver verification against PDEBench reference solutions.

Run from the folder that contains pdebench_io.py, with the PDEBench root:

    python make_verification_figure.py --root <PDEBench data folder> \
        --out ../figures/fig_pdebench_verification.pdf

Top row: 1D advection (beta = 0.4). The reference field at a later time is
compared with the exact translation of the initial field; the right panel
shows the relative L2 error over the whole trajectory for several samples.
Bottom row: 1D reaction-diffusion (Nu = rho = 1) reproduced with an
implicit-explicit spectral scheme (exact diffusion, explicit Fisher-KPP term).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pdebench_io import open_pde, load_sample

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.5, "savefig.bbox": "tight"})
C = {"k": "#333333", "b": "#378ADD", "o": "#D85A30", "g": "#1D9E75", "gr": "#888780"}

def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

def advection_run(info, idx, t_show):
    s = load_sample(info, idx); u, x, t = s["u"], s["x"], s["t"]
    beta = float(info.params["beta"]); h = float(x[1] - x[0])
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=h); u0h = np.fft.fft(u[0])
    pred = lambda n: np.real(np.fft.ifft(u0h * np.exp(-1j * k * beta * float(t[n] - t[0]))))
    errs = [rel_l2(pred(n), u[n]) for n in range(u.shape[0])]
    return x, t[: u.shape[0]], u, pred(t_show), errs

def reacdiff_run(info, idx, safety=0.1):
    """Reproduce one released trajectory; return reference and solver trajectories."""
    s = load_sample(info, idx); u, x, t = s["u"], s["x"], s["t"]
    nu, rho = float(info.params["Nu"]), float(info.params["rho"])
    h = float(x[1] - x[0]); dt_ref = float(t[1] - t[0])
    sub = max(1, int(np.ceil(dt_ref / (safety / rho)))); dt = dt_ref / sub
    k = 2 * np.pi * np.fft.fftfreq(len(x), d=h); decay = np.exp(-nu * k ** 2 * dt)
    v = u[0].astype(float).copy(); traj = [v.copy()]
    for n in range(1, u.shape[0]):
        for _ in range(sub):
            v = v + dt * rho * v * (1 - v)
            v = np.real(np.fft.ifft(np.fft.fft(v) * decay))
        traj.append(v.copy())
    traj = np.array(traj)
    errs = [rel_l2(traj[n], u[n]) for n in range(u.shape[0])]
    # time scale on which diffusion removes the dominant initial mode
    amp = np.abs(np.fft.fft(u[0] - u[0].mean()))
    kdom = abs(k[int(np.argmax(amp))])
    return x, t[: u.shape[0]], u, traj, errs, nu * kdom ** 2

def freeze_index(u, tol=1e-7):
    """First snapshot after which the stored field no longer changes, or None."""
    step = np.abs(np.diff(u, axis=0)).max(axis=1)
    frozen = np.flatnonzero(step < tol)
    if frozen.size and np.all(step[frozen[0]:] < tol):
        return int(frozen[0]) + 1
    return None

def find(root: Path, name: str):
    hits = list(root.rglob(name)); return hits[0] if hits else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--advection", default="1D_Advection_Sols_beta0.4.hdf5")
    ap.add_argument("--reacdiff", default="ReacDiff_Nu1.0_Rho1.0.hdf5")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default="fig_pdebench_verification.pdf")
    a = ap.parse_args()
    root = Path(a.root)
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- advection
    p = find(root, a.advection); info = open_pde(p)
    n_show = 100
    for i in range(a.samples):
        x, t, u, pred, errs = advection_run(info, i, n_show)
        ax[0, 1].semilogy(t, np.maximum(errs, 1e-12), color=C["b"], alpha=0.35 + 0.3 * (i == 0), lw=1,
                          label="samples" if i == 0 else None)
        if i == 0:
            ax[0, 0].plot(x, u[0], color=C["gr"], lw=1, label="initial field")
            ax[0, 0].plot(x, u[n_show], color=C["k"], lw=1.6, label=f"PDEBench, $t={t[n_show]:.2f}$")
            ax[0, 0].plot(x, pred, "--", color=C["o"], lw=1.2, label="exact translation")
    ax[0, 0].set_title(f"1D advection, $\\beta={info.params['beta']:g}$ (sample 0)")
    ax[0, 0].set_xlabel("$x$"); ax[0, 0].set_ylabel("$u$")
    ax[0, 0].legend(frameon=True, framealpha=0.9, edgecolor="none", loc="upper center",
                    bbox_to_anchor=(0.5, -0.30), ncol=3)
    ax[0, 1].set_title("Relative $L^2$ error vs. time"); ax[0, 1].set_xlabel("$t$"); ax[0, 1].set_ylabel("error")

    # ---- reaction-diffusion: what the test actually checks
    p = find(root, a.reacdiff); info = open_pde(p)
    rates, errs_live, n_frozen = [], [], 0
    for i in range(a.samples):
        x, t, u, traj, errs, rate = reacdiff_run(info, i)
        rates.append(rate)
        kf = freeze_index(u)
        end = kf if kf is not None else len(t)
        n_frozen += kf is not None
        lab_ref = "PDEBench (evolving)" if i == 0 else None
        lab_sol = "IMEX spectral solver" if i == 0 else None
        ax[1, 0].plot(t[:end], u[:end].mean(axis=1), color=C["k"], lw=1.6, alpha=0.9, label=lab_ref)
        if kf is not None:
            ax[1, 0].plot(t[end - 1:], u[end - 1:].mean(axis=1), ":", color=C["k"], lw=1.2,
                          label="PDEBench (frozen)" if n_frozen == 1 else None)
            ax[1, 0].plot(t[kf], u[kf].mean(), "x", color=C["k"], ms=6)
        ax[1, 0].plot(t, traj.mean(axis=1), "--", color=C["o"], lw=1.2, label=lab_sol)
        ax[1, 1].semilogy(t[1:end], np.maximum(errs[1:end], 1e-12), color=C["g"], alpha=0.35 + 0.3 * (i == 0), lw=1)
        errs_live.extend(errs[1:end])
    pos = [r for r in rates if r > 0]
    t_damp = 1.0 / min(pos) if pos else float('nan')
    ax[1, 0].set_title(rf"1D reaction–diffusion, $\nu={info.params['Nu']:g}$, $\rho={info.params['rho']:g}$: spatial mean")
    ax[1, 0].set_xlabel("$t$"); ax[1, 0].set_ylabel("spatial mean of $u$"); ax[1, 0].legend(frameon=False, loc="upper left")
    if np.isfinite(t_damp):
        mant, expo = f"{t_damp:.1e}".split("e")
        damp_txt = rf"spatial structure decays on $1/\nu k^2 \leq {mant}\times10^{{{int(expo)}}}$;" + "\n"
    else:
        damp_txt = ""
    ax[1, 0].text(0.36, 0.05, damp_txt + r"$\times$: stored field stops evolving",
                  transform=ax[1, 0].transAxes, fontsize=7, color=C["gr"])
    ax[1, 1].set_title("Relative $L^2$ error (evolving part, $t>0$)"); ax[1, 1].set_xlabel("$t$"); ax[1, 1].set_ylabel("error")
    print(f"reaction-diffusion: damping time <= {t_damp:.2e}; samples frozen before t_end: {n_frozen}/{a.samples}; "
          f"error over the evolving part: median {np.median(errs_live):.2e}, max {np.max(errs_live):.2e}")
    fig.tight_layout(); fig.savefig(a.out); print("written:", a.out)

if __name__ == "__main__":
    main()
