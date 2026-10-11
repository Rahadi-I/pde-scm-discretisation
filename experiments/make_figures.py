"""Figures for the Paper 1 manuscript, generated from the experiment outputs."""
import csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
# Run from this folder after the five experiment scripts; figures go to ../figures/.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures") + os.sep
os.makedirs(OUT, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # CSVs are read from this folder
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"k": "#333333", "b": "#378ADD", "g": "#1D9E75", "o": "#D85A30", "p": "#7F77DD", "gr": "#888780"}

def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))

# ---------------------------------------------------------------- Fig: scalar theory
dtf = np.linspace(0.01, 20, 400)             # Delta t / tau_f
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for lam_tau in (0.25, 1.0, 4.0):             # lambda * tau_f
    p = lam_tau * dtf; q = dtf
    ahat = np.cosh((p - q) / 2) / np.cosh((p + q) / 2)
    lamhat = -np.log(ahat) / dtf             # in units of 1/tau_f
    ax[0].plot(dtf, lamhat / lam_tau, label=rf"$\lambda\tau_f={lam_tau}$")
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\Delta t/\tau_f$"); ax[0].set_ylabel(r"$\hat\lambda/\lambda$")
ax[0].set_title("Learned decay rate under refinement")
ax[0].axhline(1, color=C["gr"], lw=0.8, ls="--"); ax[0].legend(frameon=False)
ax[0].text(0.013, 0.08, r"$\hat\lambda\approx\lambda\Delta t/2\tau_f$", fontsize=8)
# effect ratio for psi = integral over [0,T]
T_over = np.linspace(0.1, 10, 200)           # lambda*T
ax[1].plot(T_over, T_over / (1 - np.exp(-T_over)), color=C["o"])
ax[1].set_xlabel(r"$\lambda T$"); ax[1].set_ylabel("estimated / true effect")
ax[1].set_title(r"Bias of $\int_0^T x\,dt$ in the limit $\Delta t\to0$")
fig.tight_layout(); fig.savefig(OUT + "fig_scalar_theory.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: E1 + E2 convergence
e1 = read("e1_convergence.csv"); e2 = read("e2_singular.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for scheme, col, lab in (("ftcs", C["b"], "FTCS, upwind"), ("upwind", C["g"], "upwind, exact in time"), ("centred", C["o"], "centred, exact in time")):
    rows = [r for r in e1 if r["scheme"] == scheme]
    h = np.array([float(r["h"]) for r in rows]); err = np.array([float(r["abs_error"]) for r in rows])
    ax[0].loglog(h, err, "o-", color=col, label=lab, ms=4)
hh = np.array([1e-3, 3e-2])
ax[0].loglog(hh, 0.25 * hh, ":", color=C["gr"]); ax[0].text(2e-3, 1.1e-3, "slope 1", fontsize=7, color=C["gr"])
ax[0].loglog(hh, 3 * hh ** 2, ":", color=C["gr"]); ax[0].text(2e-3, 2.5e-5, "slope 2", fontsize=7, color=C["gr"])
ax[0].set_xlabel("$h$"); ax[0].set_ylabel(r"$|\tau_h-\tau|$"); ax[0].set_title("E1: smooth forcing, derived stencils")
ax[0].legend(frameon=False, loc="lower right")
# E2: regional functional, source on a cell edge vs at a cell centre (from e2_singular.csv)
import importlib.util, sys
spec = importlib.util.spec_from_file_location("e2", "e2_singular.py"); e2m = importlib.util.module_from_spec(spec); spec.loader.exec_module(e2m)
r_edge = [r for r in e2 if r["dim"] == "1" and r["psi"] == "regional"]
r_ctr = [r for r in e2 if r["dim"] == "1" and r["psi"] == "regional (source at cell centre)"]
hs = 1.0 / np.array([int(r["N"]) for r in r_edge])
err_edge = np.array([float(r["error"]) for r in r_edge])
err_ctr = np.array([float(r["error"]) for r in r_ctr])
ax[1].loglog(hs, err_edge, "s-", color=C["o"], ms=4, label="point source on a cell edge")
ax[1].loglog(hs, err_ctr, "o-", color=C["b"], ms=4, label="point source at a cell centre")
ax[1].loglog(hh, 0.4 * hh, ":", color=C["gr"]); ax[1].loglog(hh, 4 * hh ** 2, ":", color=C["gr"])
ax[1].set_xlabel("$h$"); ax[1].set_ylabel(r"$|\tau_h-\tau|$ (regional $\psi$)"); ax[1].set_title("E2: a point source, same solver")
ax[1].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "fig_e1_e2_rates.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: E2 pointwise + E3 capacity
e3 = read("e3_clamping.csv")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
rows2 = [r for r in e2 if r["dim"] == "2" and r["psi"] == "pointwise"]
N2 = np.array([int(r["N"]) for r in rows2]); v2 = np.array([float(r["value"]) for r in rows2])
ax[0].plot(np.log(N2), v2, "o", color=C["o"], label="measured, 2D")
slope = 1 / (2 * np.pi * e2m.D); c0 = v2[0] - slope * np.log(N2[0])
xx = np.linspace(np.log(N2[0]) - 0.3, np.log(N2[-1]) + 0.3, 10)
ax[0].plot(xx, slope * xx + c0, "-", color=C["k"], lw=1, label=r"slope $1/(2\pi D)$")
r1 = [r for r in e2 if r["dim"] == "1" and r["psi"] == "pointwise"]
ax[0].plot(np.log([int(r["N"]) for r in r1]), [float(r["value"]) for r in r1], "s", color=C["b"], label="measured, 1D (finite limit)")
ax[0].set_xlabel(r"$\log(1/h)$"); ax[0].set_ylabel(r"$\psi$ at the source"); ax[0].set_title("E2: pointwise functional at a point source")
ax[0].legend(frameon=False)
for case, col, mk in (("1D point", C["b"], "s"), ("2D line", C["g"], "^"), ("2D point", C["o"], "o")):
    rows = [r for r in e3 if r["case"] == case]
    N = np.array([int(r["N"]) for r in rows]); eff = np.array([abs(float(r["effect"])) for r in rows])
    ax[1].plot(np.log(N + 1), eff, mk, color=col, label=case, ms=5)      # h = 1/(N+1)
    if case == "2D point":
        fit = read("e3_capacity_fit.csv")[0]
        A, Cc = float(fit["A_predicted"]), float(fit["C_fit"])           # A predicted, C fitted
        xx = np.linspace(np.log(N[0] + 1) - 0.3, np.log(N[-1] + 1) + 1.5, 50)
        ax[1].plot(xx, A / (xx + Cc), "-", color=C["o"], lw=1, label=r"$A/(\log(1/h)+C)$, $A$ predicted")
ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\log(1/h)$"); ax[1].set_ylabel(r"$|\tau_h|$"); ax[1].set_title("E3: state clamping and capacity")
ax[1].legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.42))
fig.tight_layout(); fig.savefig(OUT + "fig_e2_pointwise_e3_capacity.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: E4 bistable
e4 = read("e4_bistable.csv")
import scipy.sparse as sp, scipy.sparse.linalg as spla
def snapshot(N, D=2e-5, a=0.25, T=600.0, dt=0.05, seed=0.2):
    h = 1.0 / N; x = (np.arange(N) + 0.5) * h
    lap = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N), format="lil", dtype=float); lap[0, 0] = -1; lap[-1, -1] = -1
    M = sp.identity(N, format="csr") - dt * (D / h ** 2) * lap.tocsr(); solve = spla.factorized(M.tocsc())
    u = np.where(x <= seed, 1.0, 0.0); snaps = {0: u.copy()}
    for n in range(1, int(T / dt) + 1):
        u = solve(u + dt * u * (1 - u) * (u - a))
        if n * dt in (200.0, 400.0, 600.0): snaps[n * dt] = u.copy()
    return x, snaps
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
h4 = np.array([float(r["h"]) for r in e4]); gr = np.array([float(r["growth"]) for r in e4])
st = [r["status"] for r in e4]
for s, col in (("pinned", C["o"]), ("propagating", C["g"])):
    m = np.array([x == s for x in st])
    ax[0].semilogx(h4[m], gr[m], "o", color=col, label=s, ms=5)
ax[0].axvspan(0.0312, 0.0417, color=C["gr"], alpha=0.25, lw=0); ax[0].invert_xaxis()
ax[0].set_xlabel("$h$ (finer to the right)"); ax[0].set_ylabel("front advance at $T$"); ax[0].set_title("E4: effect of do($u|_S$:=1)")
ax[0].legend(frameon=False, loc="upper left")
for k, (N, lab) in enumerate(((24, "$N=24$: pinned"), (128, "$N=128$: propagating"))):
    x, snaps = snapshot(N)
    for t, col in ((0, C["gr"]), (200.0, C["b"]), (400.0, C["p"]), (600.0, C["o"])):
        ax[k + 1].step(x, snaps[t], where="mid", color=col, lw=1, label=f"$t={int(t)}$")
    ax[k + 1].set_title(lab); ax[k + 1].set_xlabel("$x$"); ax[k + 1].set_ylim(-0.05, 1.1)
ax[1].set_ylabel("$u$"); ax[2].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "fig_e4_bistable.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: E5 learned stencil
e5 = read("e5_learned_stencil.csv")
dts = np.array([float(r["dt"]) for r in e5])
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
ax[0].plot(dts, [float(r["D_obs"]) for r in e5], "o", color=C["o"], label="learned (observational)")
ax[0].plot(dts, [float(r["D_predicted_bias"]) for r in e5], "-", color=C["k"], lw=1, label=r"theory $D\Delta t/2\tau_f$")
ax[0].plot(dts, [float(r["D_constrained"]) for r in e5], "x", color=C["p"], label="learned, constrained stencil")
ax[0].plot(dts, [float(r["D_event"]) for r in e5], "s", color=C["g"], label="fitted to interventions")
ax[0].plot(dts, [float(r["D_oracle"]) for r in e5], "^", color=C["b"], label="forcing observed")
ax[0].axhline(0.02, color=C["gr"], ls="--", lw=0.8); ax[0].text(0.11, 0.027, "true $D$", fontsize=7, color=C["gr"])
ax[0].set_yscale("log"); ax[0].set_ylim(1.2e-4, 0.08); ax[0].set_xlabel(r"$\Delta t$"); ax[0].set_ylabel(r"$\hat D$"); ax[0].set_title("E5: implied diffusivity")
ax[0].legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.52))
w = 0.3; xs = np.arange(len(dts))
ax[1].bar(xs - w / 2, [float(r["rmse_obs"]) for r in e5], w, color=C["o"], label="learned stencil")
ax[1].bar(xs + w / 2, [float(r["rmse_true"]) for r in e5], w, color=C["b"], label="true model")
ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{d:g}" for d in dts]); ax[1].set_xlabel(r"$\Delta t$")
ax[1].set_ylabel("one-step RMSE"); ax[1].set_title("Predictive error"); ax[1].legend(frameon=False, fontsize=7)
ax[2].bar(xs - w / 2, [100 * float(r["rel_effect_error_obs"]) for r in e5], w, color=C["o"], label="learned stencil")
ax[2].bar(xs + w / 2, [100 * float(r["rel_effect_error_event"]) for r in e5], w, color=C["g"], label="fitted to interventions")
ax[2].set_xticks(xs); ax[2].set_xticklabels([f"{d:g}" for d in dts]); ax[2].set_xlabel(r"$\Delta t$")
ax[2].set_ylim(0, 140); ax[2].set_ylabel("error in the effect (%)"); ax[2].set_title("Interventional error"); ax[2].legend(frameon=False, fontsize=7, loc="upper center", ncol=1)
fig.tight_layout(); fig.savefig(OUT + "fig_e5_learned.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig: impulse responses true vs learned
spec = importlib.util.spec_from_file_location("e5m", "e5_learned_stencil.py"); e5m = importlib.util.module_from_spec(spec); spec.loader.exec_module(e5m)
fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.5))
dt = 0.1; horizon = 2.0
src = np.fft.fft(np.exp(-((e5m.X - 0.3) ** 2) / (2 * 0.05 ** 2)))
wv = np.exp(-((e5m.X - 0.7) ** 2) / (2 * 0.05 ** 2)); wv /= wv.sum()
r01 = [r for r in e5 if abs(float(r["dt"]) - dt) < 1e-12][0]
for D, col, lab in ((float(r01["D_true"]), C["k"], "true $D$"), (float(r01["D_event"]), C["g"], "fitted to interventions"),
                    (float(r01["D_obs"]), C["o"], "learned stencil")):
    a, b = e5m._step_factors(dt, D * e5m.K ** 2); u = np.zeros(e5m.N, complex); ts, ys = [], []
    for n in range(int(horizon / dt)):
        u = a * u + b * src; ts.append((n + 1) * dt); ys.append(float(wv @ np.real(np.fft.ifft(u))))
    ax.plot(ts, ys, color=col, label=lab)
ax.set_xlabel("time after the release"); ax.set_ylabel("response at the downstream window"); ax.set_title("Response to do($f$ := $f$ + release)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "fig_e5_response.pdf"); plt.close(fig)
print("figures written to", OUT)
