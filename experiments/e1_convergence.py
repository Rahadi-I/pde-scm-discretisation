"""E1: convergence of an estimated intervention effect under grid refinement.

Baseline case (Section 3 of the paper): linear advection-diffusion on a periodic domain,
a smooth forcing intervention, and a derived (consistent) stencil.
Because the problem is linear and the added source is steady, the effect
solves the same equation with data (f' - f), which is computed here mode by
mode: exactly for the continuum, and with the scheme's amplification factor
for the discrete model. Expected outcome: convergence at the scheme's order.
"""
import numpy as np, csv

D, c, T = 0.01, 0.5, 0.4
W_X0, W_S = 0.70, 0.05                   # smooth window defining psi
SRC_X0, SRC_S, SRC_A = 0.25, 0.05, 2.0   # do(f := f + A * Gaussian)

def weight(x):
    """Smooth, mass-normalised window; a sharp indicator adds an O(h)
    quadrature error that masks the convergence order of the scheme."""
    w = np.exp(-((x - W_X0) ** 2) / (2 * W_S ** 2))
    return w / w.sum()

def source(x):
    return SRC_A * np.exp(-((x - SRC_X0) ** 2) / (2 * SRC_S ** 2))

def continuum_effect(Nq=32768):
    """psi(u^iota) - psi(u) for the continuous system, to machine accuracy."""
    x = (np.arange(Nq) + 0.5) / Nq
    k = 2 * np.pi * np.fft.fftfreq(Nq, d=1.0 / Nq)
    lam = D * k ** 2 + 1j * c * k
    fh = np.fft.fft(source(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        uh = np.where(np.abs(lam) > 1e-14, fh * (1 - np.exp(-lam * T)) / np.where(lam == 0, 1, lam), fh * T)
    u = np.real(np.fft.ifft(uh))
    return float(weight(x) @ u)

def discrete_effect(N, scheme, cfl=0.2):
    h = 1.0 / N
    x = (np.arange(N) + 0.5) * h
    # For the implicit schemes the time step is held fixed and small, so that
    # the table isolates the spatial order; for FTCS it follows the stability limit.
    dt = cfl * min(h * h / D, h / abs(c)) if scheme.startswith("ftcs") else T / 4000
    n = max(1, int(round(T / dt))); dt = T / n
    k = 2 * np.pi * np.fft.fftfreq(N, d=h)
    # symbols of the semi-discrete operator: centred diffusion, upwind advection
    if scheme.endswith("centred"):           # centred advection: second order
        adv = 1j * np.sin(k * h) / h
    else:                                     # upwind advection: first order
        adv = (1 - np.exp(-1j * k * h)) / h
    sym = D * (2 * np.cos(k * h) - 2) / h ** 2 - c * adv
    g = 1 + dt * sym if scheme.startswith("ftcs") else (1 + 0.5 * dt * sym) / (1 - 0.5 * dt * sym)
    fh = np.fft.fft(source(x))
    denom = 1 - g
    uh = np.where(np.abs(denom) > 1e-14, dt * fh * (1 - g ** n) / np.where(denom == 0, 1, denom), dt * n * fh)
    u = np.real(np.fft.ifft(uh))
    return float(weight(x) @ u)

if __name__ == "__main__":
    tau_true = continuum_effect()
    print(f"continuum effect tau = {tau_true:.8f}\n")
    rows = []
    for scheme in ("ftcs", "cn", "cn_centred"):
        prev = None
        for N in (32, 64, 128, 256, 512, 1024):
            tau = discrete_effect(N, scheme)
            err = abs(tau - tau_true)
            order = None if prev is None else np.log2(prev / err)
            rows.append(dict(scheme=scheme, N=N, h=1.0 / N, tau=tau, abs_error=err,
                             rel_error=err / abs(tau_true),
                             observed_order="" if order is None else round(float(order), 2)))
            print(f"{scheme:5s} N={N:5d}  tau={tau: .6f}  |tau-tau_true|={err:.3e}"
                  + ("" if order is None else f"   observed order={order:.2f}"))
            prev = err
        print()
    with open("e1_convergence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written: e1_convergence.csv")
