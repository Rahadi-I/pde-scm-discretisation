"""Independent checks of the formulas in the paper (Lemma 1, Corollary 1, Propositions 1-3) against the experiment outputs. Run after the experiment scripts."""
import numpy as np, sys, csv, importlib.util
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ok = lambda c: "OK " if c else "MISMATCH"

print("== Lemma 1: plim of OLS = (a+rho)/(1+a rho); Cov/Var closed forms")
for a, r in [(0.9, 0.5), (0.3, 0.8), (-0.4, 0.6), (0.95, 0.99)]:
    s2 = 1.0; sf2 = s2/(1-r*r)
    J = np.arange(0, 4000)
    cov_series = sf2*np.sum(a**J * r**(J+1)); cov_closed = sf2*r/(1-a*r)
    j, k = np.meshgrid(J[:1500], J[:1500], indexing='ij')
    var_series = sf2*np.sum(a**(j+k) * r**np.abs(j-k)); var_closed = sf2*(1+a*r)/((1-a*a)*(1-a*r))
    ahat = a + cov_closed/var_closed
    print(f"  a={a:5.2f} rho={r:4.2f}  Cov {cov_series:.6f}/{cov_closed:.6f}  Var {var_series:.4f}/{var_closed:.4f}"
          f"  a+Cov/Var={ahat:.6f}  (a+rho)/(1+a rho)={(a+r)/(1+a*r):.6f}  {ok(abs(ahat-(a+r)/(1+a*r))<1e-12 and abs(var_series-var_closed)/var_closed<1e-6)}")

print("\n== Corollary: cosh form, small- and large-step limits")
for lam, beta in [(0.5, 2.0), (2.0, 0.5), (1.0, 1.0)]:
    for dt in [1e-3, 1e-2, 0.1, 1.0, 30.0]:
        a, r = np.exp(-lam*dt), np.exp(-beta*dt)
        ahat = (a+r)/(1+a*r)
        cosh = np.cosh((lam-beta)*dt/2)/np.cosh((lam+beta)*dt/2)
        lamhat = -np.log(ahat)/dt
        print(f"  lam={lam} beta={beta} dt={dt:6.3f}  ahat-cosh={ahat-cosh:+.1e}  lamhat={lamhat:.6g}"
              f"  lam*dt*beta/2={lam*dt*beta/2:.6g}  min={min(lam,beta)}")
print("  effect ratio lam T/(1-exp(-lam T)) as dt->0: for lam=1,T=2:", 2/(1-np.exp(-2)))

print("\n== Continuous OU forcing (not held constant): exact lag-1 autocorrelation vs cosh form")
for lam, beta in [(0.79, 0.5), (3.16, 0.5), (12.6, 0.5)]:
    for dt in [0.05, 0.1, 0.2]:
        cont = (lam*np.exp(-beta*dt) - beta*np.exp(-lam*dt))/(lam-beta)
        a, r = np.exp(-lam*dt), np.exp(-beta*dt); zoh = (a+r)/(1+a*r)
        lc, lz = -np.log(cont)/dt, -np.log(zoh)/dt
        print(f"  lam={lam:5.2f} dt={dt}: lamhat continuous-OU={lc:.5f}  ZOH/cosh={lz:.5f}  leading lam*beta*dt/2={lam*beta*dt/2:.5f}")

print("\n== Prop. pointwise (2D): increment log2/(2 pi D), D=0.05")
D = 0.05; pred = np.log(2)/(2*np.pi*D)
rows = list(csv.DictReader(open(os.path.join(HERE, 'e2_singular.csv'))))
v = [float(r['value']) for r in rows if r['dim']=='2' and r['psi']=='pointwise']
inc = np.diff(v)
print(f"  predicted {pred:.4f}; measured {', '.join(f'{x:.4f}' for x in inc)}  {ok(all(abs(x-pred)<0.01 for x in inc))}")

print("\n== Prop. point source: edge vs centre orders, and error ~ Z'(x0)(x_c-x0)")
spec = importlib.util.spec_from_file_location("e2", os.path.join(HERE, "e2_singular.py")); e2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e2)
def disc(N, x0):
    h = 1.0/N; x = (np.arange(N)+0.5)*h
    dt = 0.2*h*h/e2.D; nt = max(1, int(round(e2.T/dt))); dt = e2.T/nt
    k = 2*np.pi*np.fft.fftfreq(N, d=h); g = 1 + dt*e2.D*(2*np.cos(k*h)-2)/h**2
    i0 = int(np.argmin(np.abs(x-x0))); s = np.zeros(N); s[i0] = 1.0/h
    fh = np.fft.fft(s); den = 1-g
    uh = np.where(np.abs(den) > 1e-14, dt*fh*(1-g**nt)/np.where(den == 0, 1, den), dt*nt*fh)
    u = np.real(np.fft.ifft(uh)); w = np.exp(-((x-e2.W_X0)**2)/(2*e2.W_S**2)); w /= w.sum()
    return float(w@u), x[i0]
kk = 2*np.pi*np.arange(1, 400000); amp = (1-np.exp(-e2.D*kk**2*e2.T))/(e2.D*kk**2); win = np.exp(-0.5*(kk*e2.W_S)**2)
Z = lambda y: e2.T + 2*np.sum(amp*win*np.cos(kk*(e2.W_X0-y)))
dZ = lambda y: 2*np.sum(amp*win*kk*np.sin(kk*(e2.W_X0-y)))
Ns = [64, 128, 256, 512, 1024]; E, Cc = [], []
for N in Ns:
    te, xc = disc(N, 0.5); E.append(te - Z(0.5))
    tc, _ = disc(N, 0.5+0.5/N); Cc.append(tc - Z(0.5+0.5/N))
    print(f"  N={N:5d}  edge err={E[-1]:+.3e}  predicted Z'(x0)(xc-x0)={dZ(0.5)*(xc-0.5):+.3e}   centre err={Cc[-1]:+.3e}")
oe = np.log2(np.abs(E[:-1])/np.abs(E[1:])); oc = np.log2(np.abs(Cc[:-1])/np.abs(Cc[1:]))
print(f"  orders edge {np.round(oe,2)}  centre {np.round(oc,2)}  ratio edge/centre {np.round(np.abs(E)/np.abs(Cc),1)}")

print("\n== Prop. capacity: A = 2 pi u0(x*) G_{-Delta}(x_obs,x*) on the unit square, D=1")
M = 400
m = np.arange(1, M+1); MM, NN = np.meshgrid(m, m, indexing='ij')
def phi(m, n, x, y): return 2*np.sin(m*np.pi*x)*np.sin(n*np.pi*y)
lamm = np.pi**2*(MM**2+NN**2)
# u0 solves -Delta u = 1: coefficient of phi_mn is <1,phi_mn>/lam
proj = 2*((1-np.cos(MM*np.pi))/(MM*np.pi))*((1-np.cos(NN*np.pi))/(NN*np.pi))  # <1, phi_mn>
u0 = np.sum(proj/lamm*phi(MM, NN, 0.5, 0.5))
G = np.sum(phi(MM, NN, 0.25, 0.5)*phi(MM, NN, 0.5, 0.5)/lamm)
A_pred = 2*np.pi*u0*G
print(f"  u0(x*)={u0:.6f} (known 0.0736713)  G(x_obs,x*)={G:.6f}  A_pred={A_pred:.4f}")
rows = [r for r in csv.DictReader(open(os.path.join(HERE, 'e3_clamping.csv'))) if r['case']=='2D point']
Nn = np.array([float(r['N']) for r in rows]); t = np.array([abs(float(r['effect'])) for r in rows])
for lab, L in (("log N (wrong variable, for comparison)", np.log(Nn)), ("log(1/h)=log(N+1)", np.log(Nn+1))):
    sl, ic = np.polyfit(L, 1/t, 1); A, C = 1/sl, ic/sl
    fit = A/(L+C); print(f"  fit with {lab}: A={A:.4f} C={C:.3f}  max rel misfit={np.max(np.abs(fit-t)/t)*100:.3f}%"
                         f"  A vs pred {abs(A-A_pred)/A_pred*100:.2f}%  N=1e6 -> {A/(np.log(1e6 if 'N)' in lab else 1e6+1)+C):.2e}")
