"""E2: singular interventions (point sources) and the regularity of psi.

Case B (Section 3 of the paper). A point source is added to a diffusion problem on a
periodic domain, do(f := f + delta_{x0}), and the resulting effect is measured
with two functionals: a smooth window away from the source, and the pointwise
value at the source itself. Everything is evaluated mode by mode, so the
continuum effect is exact up to the truncation of the series.

Prediction: the regional functional converges at the scheme's order in both
dimensions, whereas the pointwise functional at the source has a finite
continuum limit in 1D but diverges logarithmically in 2D, where the discrete
value grows like log(1/h).
"""
import numpy as np, csv

D, T = 0.05, 0.2
X0 = 0.5                    # source location (both coordinates in 2D)
W_X0, W_S = 0.8, 0.05       # smooth window for the regional functional

# ---------------------------------------------------------------- 1D
def continuum_1d(n_modes=400000, x0=X0):
    """Effect of a unit point source, as a mode series on the unit interval.

    For a Gaussian window of width W_S the spatial integral of each mode is
    available in closed form, exp(-k^2 W_S^2 / 2), so no quadrature is needed.
    """
    n = np.arange(1, n_modes)
    k = 2 * np.pi * n
    amp = (1 - np.exp(-D * k ** 2 * T)) / (D * k ** 2)
    reg = T + 2 * np.sum(amp * np.exp(-0.5 * (k * W_S) ** 2) * np.cos(k * (W_X0 - x0)))
    point = T + 2 * np.sum(amp)
    return reg, point

def dZ_1d(x0=X0, n_modes=400000):
    """Derivative in x0 of the regional effect Z(x0); Proposition 1 predicts
    tau_h - tau = Z'(x0) (x_c - x0) + O(h^2) for a source at x0 placed on the cell centred at x_c."""
    k = 2 * np.pi * np.arange(1, n_modes)
    amp = (1 - np.exp(-D * k ** 2 * T)) / (D * k ** 2)
    return 2 * np.sum(amp * np.exp(-0.5 * (k * W_S) ** 2) * k * np.sin(k * (W_X0 - x0)))

def discrete_1d(N, scheme="ftcs", cfl=0.2, x0=X0):
    h = 1.0 / N
    x = (np.arange(N) + 0.5) * h
    dt = cfl * h * h / D if scheme == "ftcs" else T / 2000
    n = max(1, int(round(T / dt))); dt = T / n
    k = 2 * np.pi * np.fft.fftfreq(N, d=h)
    sym = D * (2 * np.cos(k * h) - 2) / h ** 2
    g = 1 + dt * sym if scheme == "ftcs" else (1 + 0.5 * dt * sym) / (1 - 0.5 * dt * sym)
    i0 = int(np.argmin(np.abs(x - x0)))
    src = np.zeros(N); src[i0] = 1.0 / h                  # unit mass on the cell containing x0
    fh = np.fft.fft(src)
    den = 1 - g
    uh = np.where(np.abs(den) > 1e-14, dt * fh * (1 - g ** n) / np.where(den == 0, 1, den), dt * n * fh)
    u = np.real(np.fft.ifft(uh))
    w = np.exp(-((x - W_X0) ** 2) / (2 * W_S ** 2)); w /= w.sum()
    return float(w @ u), float(u[i0]), float(x[i0])

# ---------------------------------------------------------------- 2D
def continuum_2d(n_modes=1200):
    """Same series in two dimensions; the pointwise sum diverges with n_modes,
    which is the logarithmic divergence of the point-source solution."""
    n = np.arange(-n_modes, n_modes + 1)
    kx = 2 * np.pi * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    amp = np.where(K2 > 0, (1 - np.exp(-D * K2 * T)) / np.where(K2 > 0, D * K2, 1.0), T)
    win = np.exp(-0.5 * K2 * W_S ** 2)
    reg = float(np.sum(amp * win * np.cos(KX * (W_X0 - X0))))
    point = float(np.sum(amp))
    return reg, point

def discrete_2d(N, cfl=0.2):
    h = 1.0 / N
    x = (np.arange(N) + 0.5) * h
    dt = cfl * h * h / (4 * D)
    n = max(1, int(round(T / dt))); dt = T / n
    k = 2 * np.pi * np.fft.fftfreq(N, d=h)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    sym = D * ((2 * np.cos(KX * h) - 2) + (2 * np.cos(KY * h) - 2)) / h ** 2
    g = 1 + dt * sym
    i0 = int(np.argmin(np.abs(x - X0)))
    src = np.zeros((N, N)); src[i0, i0] = 1.0 / h ** 2
    fh = np.fft.fft2(src)
    den = 1 - g
    uh = np.where(np.abs(den) > 1e-14, dt * fh * (1 - g ** n) / np.where(den == 0, 1, den), dt * n * fh)
    u = np.real(np.fft.ifft2(uh))
    Wx = np.exp(-((x - W_X0) ** 2) / (2 * W_S ** 2))
    Wy = np.exp(-((x - X0) ** 2) / (2 * W_S ** 2))
    W = np.outer(Wx, Wy); W /= W.sum()
    return float(np.sum(W * u)), float(u[i0, i0])

# ---------------------------------------------------- control: smooth source
def continuum_smooth_1d(sig=0.05, n_modes=400000):
    k = 2 * np.pi * np.arange(1, n_modes)
    amp = (1 - np.exp(-D * k ** 2 * T)) / (D * k ** 2)
    return T + 2 * np.sum(amp * np.exp(-0.5 * (k * sig) ** 2)
                          * np.exp(-0.5 * (k * W_S) ** 2) * np.cos(k * (W_X0 - X0)))

def discrete_smooth_1d(N, sig=0.05, cfl=0.2):
    h = 1.0 / N
    x = (np.arange(N) + 0.5) * h
    dt = cfl * h * h / D; n = max(1, int(round(T / dt))); dt = T / n
    k = 2 * np.pi * np.fft.fftfreq(N, d=h)
    g = 1 + dt * D * (2 * np.cos(k * h) - 2) / h ** 2
    s = np.exp(-((x - X0) ** 2) / (2 * sig ** 2)); s /= (s.sum() * h)
    fh = np.fft.fft(s); den = 1 - g
    uh = np.where(np.abs(den) > 1e-14, dt * fh * (1 - g ** n) / np.where(den == 0, 1, den), dt * n * fh)
    u = np.real(np.fft.ifft(uh))
    w = np.exp(-((x - W_X0) ** 2) / (2 * W_S ** 2)); w /= w.sum()
    return float(w @ u)


if __name__ == "__main__":
    rows = []
    reg_true, pt_true = continuum_1d()
    print(f"1D continuum: regional psi = {reg_true:.8f}   pointwise psi at the source = {pt_true:.6f}\n")
    prev = None
    dZ = dZ_1d()
    for N in (64, 128, 256, 512, 1024):
        reg, pt, xc = discrete_1d(N)                      # x0 = 0.5 lies on a cell edge
        err = reg - reg_true
        order = "" if prev is None else round(float(np.log2(abs(prev) / abs(err))), 2)
        rows.append(dict(dim=1, N=N, psi="regional", value=reg, error=abs(err), order=order,
                         signed_error=err, predicted_first_order=dZ * (xc - X0)))
        rows.append(dict(dim=1, N=N, psi="pointwise", value=pt, error=abs(pt - pt_true), order="",
                         signed_error="", predicted_first_order=""))
        print(f"1D N={N:5d}  regional={reg:.8f} (err {err:+.3e}, predicted Z'(x0)(xc-x0) {dZ*(xc-X0):+.3e}"
              f"{'' if order=='' else f', order {order}'})   pointwise={pt:.5f}  (continuum {pt_true:.5f})")
        prev = err
    print("\nsame solver and functional, source moved to a cell centre")
    prev = None
    for N in (64, 128, 256, 512, 1024):
        xcen = X0 + 0.5 / N                               # centre of the cell to the right of x0
        ref, _ = continuum_1d(x0=xcen)
        reg, _, _ = discrete_1d(N, x0=xcen)
        err = reg - ref
        order = "" if prev is None else round(float(np.log2(abs(prev) / abs(err))), 2)
        rows.append(dict(dim=1, N=N, psi="regional (source at cell centre)", value=reg, error=abs(err), order=order,
                         signed_error=err, predicted_first_order=0.0))
        print(f"  N={N:5d}  err={err:+.3e}" + ("" if order == "" else f"   order={order}"))
        prev = err
    print()
    for N in (64, 128, 256, 512):
        reg, pt = discrete_2d(N)
        rows.append(dict(dim=2, N=N, psi="regional", value=reg, error="", order="", signed_error="", predicted_first_order=""))
        rows.append(dict(dim=2, N=N, psi="pointwise", value=pt, error="", order="", signed_error="", predicted_first_order=""))
        print(f"2D N={N:5d}  regional={reg:.8f}   pointwise at source={pt:.5f}"
              f"   pointwise/log(1/h)={pt/np.log(N):.5f}")
    print("\ncontrol: the same solver and functional with a smooth source")
    ref = continuum_smooth_1d(); prev = None
    for N in (64, 128, 256, 512):
        v = discrete_smooth_1d(N); err = abs(v - ref)
        order = "" if prev is None else round(float(np.log2(prev / err)), 2)
        rows.append(dict(dim=1, N=N, psi="regional (smooth source)", value=v, error=err, order=order, signed_error="", predicted_first_order=""))
        print(f"  N={N:5d}  err={err:.2e}" + ("" if order == "" else f"   order={order}"))
        prev = err

    v2 = [r["value"] for r in rows if r["dim"] == 2 and r["psi"] == "pointwise"]
    print("2D pointwise increments per doubling:", ", ".join(f"{b-a:.4f}" for a, b in zip(v2, v2[1:])),
          f"  predicted log(2)/(2 pi D) = {np.log(2)/(2*np.pi*D):.4f}")
    print("\nIn 2D the pointwise value has no continuum limit; the ratio to log(1/h)")
    print("settling to a constant is the signature of the logarithmic divergence.")
    with open("e2_singular.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written: e2_singular.csv")
