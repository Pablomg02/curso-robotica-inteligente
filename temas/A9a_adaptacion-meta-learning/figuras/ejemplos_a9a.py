"""Ejemplos numéricos y datos de las figuras nuevas de A9a (numpy, CPU, segundos).

Uso:  python3 ejemplos_a9a.py > salida_ejemplos.txt

Cada ejemplo lleva un número que se cita en las hojas («ejemplos_a9a.py, ejemplo N»). Son ejemplos construidos por
nosotros para seguir las derivaciones con números pequeños. Las cifras del x500 con carga y viento (bucle cerrado,
GRU, base aprendida) salen de figuras_a9a.py, que usa la GPU: no se relanza aquí. Los parámetros del Kalman del x500
(QM, QF, RN, P0, M0) se copian de ese script para que los ejemplos 9, 12 y 13 hablen del mismo filtro.
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)
G = 9.81


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def cab(n, titulo):
    print("=" * 72)
    print(f"EJEMPLO {n}. {titulo}")


def rk4(f, x, t, dt):
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def cuadrada(t, periodo):
    return 1.0 if (t % periodo) < periodo / 2 else -1.0


# ---------------------------------------------------------------------------
cab(1, "REGLA DEL MIT: GANANCIA DE PREALIMENTACIÓN CON UNA PLANTA DE SEGUNDO ORDEN")
# planta y = b G[u], modelo y_m = b_m G[r], G(s) = 1/(s² + s + 1); controlador u = θ r; θ* = b_m / b
b, bm = 2.0, 1.0
print(f"  planta y = b·G[u] con b = {b} (desconocida); modelo y_m = b_m·G[r] con b_m = {bm}; θ* = b_m/b = {bm/b}")
print("  MIT: θ' = -γ e ∂e/∂θ, con ∂e/∂θ = b G[r] = (b/b_m) y_m  ->  θ' = -γ' y_m e (γ' absorbe b/b_m)")


def mit(gam, Tf=80.0, dt=0.002, periodo=20.0, orden=2):
    # estado: [x1, x2] de la planta (y = b x1), [xm1, xm2] del modelo, θ
    def f(t, z):
        r = cuadrada(t, periodo)
        th = z[4]
        u = th * r
        if orden == 2:
            dx = [z[1], -z[0] - z[1] + u]
            dxm = [z[3], -z[2] - z[3] + r]
        else:
            dx = [-z[0] + u, 0.0]
            dxm = [-z[2] + r, 0.0]
        y, ym = b * z[0], bm * z[2]
        e = y - ym
        return np.array(dx + dxm + [-gam * ym * e])
    z = np.array([0, 0, 0, 0, 1.0])            # θ(0) = 1 (el doble de lo ideal)
    ts, ths, es = [], [], []
    n = int(Tf / dt)
    for i in range(n + 1):
        t = i * dt
        if i % 50 == 0:
            ts.append(t); ths.append(z[4]); es.append(b * z[0] - bm * z[2])
        if abs(z[4]) > 50:
            break
        z = rk4(f, z, t, dt)
    return np.array(ts), np.array(ths), np.array(es)


for gam in (0.2, 1.0):
    mu = gam * b * bm * 1.0 ** 2
    raices = np.roots([1, 1, 1, mu])
    print(f"  γ' = {gam}: μ = γ' b b_m r0² = {mu:.2f}; raíces de s³ + s² + s + μ = {np.round(raices, 3)} "
          f"-> {'estable' if np.all(raices.real < 0) else 'INESTABLE'} (Routh: estable si μ < 1)")
    for orden in (2, 1):
        ts, ths, es = mit(gam, orden=orden)
        def cerca(t):
            return ths[np.argmin(abs(ts - t))]
        print(f"    G de orden {orden}: θ(20 s) = {cerca(20):.3f}, θ(40 s) = {cerca(40):.3f}, θ(60 s) = {cerca(60):.3f}, "
              f"máx |e| en 60–80 s = {np.max(abs(es[ts >= 60])) if np.any(ts >= 60) else float('nan'):.3f}; "
              f"último t = {ts[-1]:.1f} s")
    print(f"    primer orden, G = 1/(s+1): s² + s + μ, raíces {np.round(np.roots([1, 1, mu]), 3)} (estable para todo μ > 0)")
X1, Y1 = lin(0, 80, 60, 520), lin(-1.0, 2.0, 230, 20)
for gam in (0.2, 1.0):
    ts, ths, _ = mit(gam)
    print(f"path MIT γ'={gam}:", path(ts[::4], np.clip(ths[::4], -1.0, 2.0), X1, Y1))
print(f"  ejes: θ = 0,5 en y = {Y1(0.5):.1f}; θ = 0 en y = {Y1(0):.1f}; θ = 1 en {Y1(1):.1f}; θ = 2 en {Y1(2):.1f}; θ = -1 en {Y1(-1):.1f}")

# ---------------------------------------------------------------------------
cab(2, "FUNCIÓN DE LYAPUNOV DE UN SISTEMA ESTABLE DE PRIMER ORDEN")
x0 = 1.0
print("  x' = -2x, V = x²: V' = 2 x x' = -4 x² = -4 V  ->  V(t) = V(0) e^{-4t}")
for t in (0, 0.25, 0.5, 1.0):
    print(f"    t = {t}: x = {x0*math.exp(-2*t):.4f}, V = {x0**2*math.exp(-4*t):.4f}")

# ---------------------------------------------------------------------------
cab(3, "MRAC ESCALAR CON LYAPUNOV: x' = a x + b u, u = k x + r, MODELO x_m' = a_m x_m + b r")
a, bb, am, q = 1.0, 1.0, -2.0, 4.0
kstar = (am - a) / bb
p = -q / (2 * am)
gam = 2.0
print(f"  a = {a} (desconocido, planta inestable), b = {bb}, a_m = {am}: k* = (a_m - a)/b = {kstar}")
print(f"  Lyapunov escalar: 2 a_m p = -q con q = {q}  ->  p = {p}; ley k' = -γ p b e x con γ = {gam}")
# instante inicial del caso B (r = 0, x(0) = 1, x_m(0) = 0, k(0) = 0)
e0, x0, k0 = 1.0, 1.0, 0.0
kt0 = k0 - kstar
de = am * e0 + bb * kt0 * x0
dk = -gam * p * bb * e0 * x0
V0 = p * e0 ** 2 + kt0 ** 2 / gam
dV = 2 * p * e0 * de + 2 * kt0 * dk / gam
print(f"  caso B en t = 0: e = {e0}, k̃ = {kt0}; e' = a_m e + b k̃ x = {de}; k' = {dk}; V = p e² + k̃²/γ = {V0}; "
      f"V' = 2 p e e' + 2 k̃ k'/γ = {dV} = -q e² = {-q*e0**2}")


K0 = -2.0                                   # valor nominal de k (modelo nominal con a = 0: k0 = a_m - 0)


def mrac(caso, gam=gam, Tf=40.0, dt=0.001, sigma=0.0, d_amp=0.0, periodo=8.0, z0=None):
    def f(t, z):
        x, xm, k = z
        r = cuadrada(t, periodo) if caso == "A" else 0.0
        d = d_amp * math.sin(2.0 * t)
        u = k * x + r
        e = x - xm
        return np.array([a * x + bb * (u + d), am * xm + bb * r, -gam * (p * bb * e * x + sigma * (k - K0))])
    z = (np.array([0.0, 0.0, 0.0]) if caso == "A" else np.array([1.0, 0.0, 0.0])) if z0 is None else np.array(z0)
    ts, ks, es, Vs = [], [], [], []
    n = int(Tf / dt)
    for i in range(n + 1):
        t = i * dt
        if i % 20 == 0:
            ts.append(t); ks.append(z[2]); es.append(z[0] - z[1]); Vs.append(p * (z[0] - z[1]) ** 2 + (z[2] - kstar) ** 2 / gam)
        z = rk4(f, z, t, dt)
    return np.array(ts), np.array(ks), np.array(es), np.array(Vs)


tA, kA, eA, VA = mrac("A")
tB, kB, eB, VB = mrac("B")
kinf = -1 - math.sqrt(1 + gam * 1.0 ** 2 / p)
for nom, t_, k_, e_, V_ in (("A (r cuadrada ±1, periodo 8 s)", tA, kA, eA, VA), ("B (r = 0, x(0) = 1)", tB, kB, eB, VB)):
    idx = [np.argmin(abs(t_ - tt)) for tt in (0, 2, 5, 10, 20, 40)]
    print(f"  caso {nom}: k(t) en 0, 2, 5, 10, 20, 40 s = {np.round(k_[idx], 3)}; |e| máx en 30–40 s = "
          f"{np.max(abs(e_[t_ >= 30])):.4f}; V(0) = {V_[0]:.3f}, V(40) = {V_[-1]:.3f}; ¿V crece alguna vez? "
          f"{bool(np.any(np.diff(V_) > 1e-9))}")
print(f"  caso B, valor final analítico: (1 + k)² = (1 + k0)² + γ x0²/p  ->  k_∞ = -1 - √(1 + γ) = {kinf:.4f}  (k* = {kstar})")
X3, Y3 = lin(0, 40, 60, 520), lin(-4.0, 0.5, 230, 20)
print("path MRAC k caso A:", path(tA[::10], kA[::10], X3, Y3))
print("path MRAC k caso B:", path(tB[::10], kB[::10], X3, Y3))
print(f"  ejes: k = -3 en y = {Y3(-3):.1f}; k = 0 en y = {Y3(0):.1f}; k = -1 en {Y3(-1):.1f}; k = -2 en {Y3(-2):.1f}; "
      f"k = -4 en {Y3(-4):.1f}; k_∞ en {Y3(kinf):.1f}")

# ---------------------------------------------------------------------------
cab(4, "EXCITACIÓN PERSISTENTE CON DOS PARÁMETROS: y = m s_z - f_z (FLOTAR FRENTE A OSCILACIÓN VERTICAL)")
m_v, fz_v = 2.6, 0.0
A_osc, w_osc = (2 * math.pi / 5.0) ** 2 * 1.0, 2 * math.pi / 5.0     # 1 m de oscilación vertical con periodo 5 s
print(f"  verdad: m = {m_v} kg, f_z = {fz_v} N; regresor φ = (s_z, -1); flotar: s_z = g; oscilación: s_z = g + A sen(ωt), "
      f"A = ω²·1 m = {A_osc:.3f} m/s², ω = {w_osc:.4f} rad/s")
Tp = 5.0
for nom, Am in (("flotar", 0.0), ("oscilación", A_osc)):
    tt = np.linspace(0, Tp, 50001)
    sz = G + Am * np.sin(w_osc * tt)
    Phi = np.stack([sz, -np.ones_like(sz)], 1)
    Gm = np.trapezoid(Phi[:, :, None] * Phi[:, None, :], tt, axis=0)
    ev, evec = np.linalg.eigh(Gm)
    print(f"  {nom}: ∫_0^5 φφᵀ dt = {np.round(Gm, 2).tolist()}; valores propios {np.round(ev, 4)}; "
          f"dirección del menor {np.round(evec[:, 0], 4)}")
print(f"  flotar: la dirección nula es (1, g)/√(1+g²) = {np.round(np.array([1, G]) / math.sqrt(1 + G*G), 4)}: "
      f"subir m en δ y f_z en g·δ no cambia y")
print(f"  oscilación, a mano: ∫ φφᵀ = T [[g² + A²/2, -g], [-g, 1]] con T = 5: det = T² A²/2 = {Tp**2 * A_osc**2 / 2:.3f}")
# filtro de Kalman de dos parámetros con los valores del x500 (figuras_a9a.py: QM, QF, RN, P0)
QM, QF, RN = 1e-4, 0.05, 0.4 ** 2
DTC = 0.02


def kalman2(Am, Qfz=QF, P0fz=2.0 ** 2, n=245, seed=0, Qm=QM):
    rng = np.random.default_rng(seed)
    th = np.array([2.0, 0.0])
    P = np.diag([0.3 ** 2, P0fz])
    Q = np.diag([Qm, Qfz])
    tray = [th.copy()]
    for i in range(n):
        t = (i + 1) * DTC
        sz = G + Am * math.sin(w_osc * t)
        phi = np.array([sz, -1.0])
        y = m_v * sz - fz_v + rng.normal(0, 0.4)
        P = P + Q
        S = phi @ P @ phi + RN
        K = P @ phi / S
        th = th + K * (y - phi @ th)
        P = P - np.outer(K, phi @ P)
        tray.append(th.copy())
    return th, P, np.array(tray)


print("  (a) parámetros constantes (Q = 0: mínimos cuadrados recursivos con el prior P0 = diag(0,3², 2²)), 10 s = 500 pasos:")
for nom, Am, Qfz, P0fz in (("flotar", 0.0, 0.0, 4.0), ("oscilación", A_osc, 0.0, 4.0)):
    for n in (50, 245, 500):
        res = [kalman2(Am, Qfz, P0fz, n=n, seed=s, Qm=0.0) for s in range(64)]
        ms = np.array([r[0][0] for r in res])
        sd = math.sqrt(res[0][1][0, 0])
        print(f"    {nom:10s}, {n*DTC:4.1f} s: mediana de m̂ = {np.median(ms):.3f} kg, mediana |m̂ - m|/M0 = {np.median(abs(ms - m_v))/2*100:5.1f} %; "
              f"desviación típica a posteriori de m = {sd:.3f} kg")
print("  (b) con el paseo aleatorio del x500 (Q = diag(QM, QF) por paso) y 245 pasos (4,9 s):")
for nom, Am, Qfz, P0fz in (("flotar", 0.0, QF, 4.0), ("oscilación", A_osc, QF, 4.0), ("flotar, f_z ≈ 0", 0.0, QF / 100, 0.3 ** 2)):
    res = [kalman2(Am, Qfz, P0fz, seed=s) for s in range(64)]
    ms = np.array([r[0][0] for r in res])
    sd = math.sqrt(res[0][1][0, 0])
    print(f"    {nom:16s}: mediana |m̂ - m|/M0 = {np.median(abs(ms - m_v))/2*100:.1f} %; "
          f"desviación típica a posteriori de m = {sd:.3f} kg (a priori 0,3)")
X4, Y4 = lin(1.8, 2.8, 70, 330), lin(-7.0, 1.0, 250, 20)
for nom, Am in (("flotar", 0.0), ("oscilación", A_osc)):
    _, _, tr = kalman2(Am, 0.0, 4.0, n=500, seed=3, Qm=0.0)
    print(f"path PE {nom}:", path(tr[:, 0], np.clip(tr[:, 1], -7, 1), X4, Y4))
    print(f"  final {nom} (semilla 3, 10 s): m̂ = {tr[-1,0]:.3f}, f̂_z = {tr[-1,1]:.3f}; en 1 s: {np.round(tr[50], 3)}; en 4,9 s: {np.round(tr[245], 3)}")
yz = m_v * G
print(f"  recta de flotar m g - f_z = {yz:.3f}: de (m, f_z) = (1.8, {1.8*G - yz:.2f}) a (2.8, {2.8*G - yz:.2f}); "
      f"en píxeles {X4(1.8):.1f},{Y4(1.8*G - yz):.1f} -> {X4(2.8):.1f},{Y4(2.8*G - yz):.1f}")
print(f"  verdad ({m_v}, 0) en {X4(m_v):.1f},{Y4(0):.1f}; inicio (2, 0) en {X4(2):.1f},{Y4(0):.1f}; ejes: f_z = -6 en y = {Y4(-6):.1f}, "
      f"-4 en {Y4(-4):.1f}, -2 en {Y4(-2):.1f}; m = 2.2 en x = {X4(2.2):.1f}, 2.4 en {X4(2.4):.1f}")

# ---------------------------------------------------------------------------
cab(5, "DERIVA SIN EXCITACIÓN: MRAC ESCALAR CON r = 0 Y UNA PERTURBACIÓN ACOTADA, CON Y SIN MODIFICACIÓN σ")
print("  r = 0, x_m = 0, e = x: la ley k' = -γ p b e x = -γ x² (p = b = 1) es siempre ≤ 0: k solo puede bajar")
print(f"  arranque en k(0) = k0 = {K0} (valor nominal) y x(0) = 0; perturbación d = 0,2 sen(2t) en la entrada; γ = {gam}")
print(f"  modificación σ: k' = -γ (p b e x + σ (k - k0)), que tira de k hacia el nominal k0 = {K0}")
res5 = {}
for sigma in (0.0, 0.1):
    t5, k5, e5, _ = mrac("B", Tf=500.0, dt=0.002, sigma=sigma, d_amp=0.2, z0=[0.0, 0.0, K0])
    res5[sigma] = (t5, k5)
    idx = [np.argmin(abs(t5 - tt)) for tt in (10, 50, 100, 250, 500)]
    print(f"  σ = {sigma}: k en 10, 50, 100, 250, 500 s = {np.round(k5[idx], 3)}; "
          f"|x| máx en 400–500 s = {np.max(abs(e5[t5 >= 400])):.4f}")
tA_s, kA_s, eA_s, _ = mrac("A", Tf=80.0, sigma=0.1)
print(f"  con r cuadrada y σ = 0,1: k(80 s) = {kA_s[-1]:.3f}, |e| máx en 60–80 s = {np.max(abs(eA_s[tA_s >= 60])):.4f} "
      f"(sin σ, k → k* = {kstar} y e → 0): el precio es un sesgo")
X5, Y5 = lin(0, 500, 60, 520), lin(-5.0, -1.0, 230, 20)
for sigma in (0.0, 0.1):
    t5, k5 = res5[sigma]
    print(f"path deriva σ={sigma}:", path(t5[::250], k5[::250], X5, Y5))
print(f"  ejes: k = -1 en y = {Y5(-1):.1f}, -2 en {Y5(-2):.1f}, -3 en {Y5(-3):.1f}, -4 en {Y5(-4):.1f}, -5 en {Y5(-5):.1f}; t = 100 en x = {X5(100):.1f}, "
      f"250 en {X5(250):.1f}")

# ---------------------------------------------------------------------------
cab(6, "MODIFICACIÓN σ: COTA DE LYAPUNOV")
sigma, kst = 0.1, kstar
dk0 = kst - K0
print(f"  V' = -q e² - 2σ k̃ (k - k0) = -q e² - 2σ k̃² - 2σ k̃ (k* - k0)  ≤  -q e² - σ k̃² + σ (k* - k0)²  (porque -2ab ≤ a² + b²)")
print(f"  con q = {q}, σ = {sigma}, k* - k0 = {dk0}: V' < 0 fuera de la elipse q e² + σ k̃² = σ (k* - k0)² = {sigma*dk0**2:.2f}: "
      f"semiejes |e| = {math.sqrt(sigma*dk0**2/q):.3f}, |k̃| = {abs(dk0):.1f}")

# ---------------------------------------------------------------------------
cab(7, "L1 ESCALAR: PREDICTOR, ADAPTACIÓN RÁPIDA Y FILTRO, CON UN RETARDO DE 20 ms EN EL ACTUADOR")
am7 = -2.0


def l1(Gam, w_f, retardo=0.02, Tf=4.0, dt=1e-4):
    nd = int(round(retardo / dt))
    buf = [0.0] * (nd + 1)
    x = xh = sh = uf = 0.0
    out = []
    for i in range(int(Tf / dt)):
        t = i * dt
        sig = (2.0 if t >= 1.0 else 0.0) + 0.5 * math.sin(1.0 * t)
        if w_f is None:
            u = -sh
        else:
            uf += dt * w_f * (sh - uf)
            u = -uf
        buf.append(u)
        u_apl = buf.pop(0)
        xt = xh - x
        x += dt * (am7 * x + (u_apl + sig))
        xh += dt * (am7 * xh + (u + sh))
        sh += dt * (-Gam * xt)
        sh = max(-20.0, min(20.0, sh))                 # proyección: σ̂ en [-20, 20]
        if i % 20 == 0:
            out.append((t, x, u, sh, sig))
    return np.array(out)


casos7 = (("sin filtro, Γ = 100", 100.0, None), ("sin filtro, Γ = 10⁴", 1e4, None), ("filtro ω = 10 rad/s, Γ = 10⁴", 1e4, 10.0),
          ("filtro ω = 10 rad/s, Γ = 10⁶", 1e6, 10.0))
res7 = {}
for nom, Gm_, wf in casos7:
    o = l1(Gm_, wf)
    res7[nom] = o
    t, x, u = o[:, 0], o[:, 1], o[:, 2]
    du = np.abs(np.diff(u)) / (t[1] - t[0])
    ok = np.all(np.isfinite(x)) and np.max(abs(x)) < 10
    print(f"  {nom:30s}: |σ̂| máx = {np.max(abs(o[:, 3])):.2f}; |x| máx tras el escalón (1–4 s) = {np.max(abs(x[t >= 1])):.4f}; |x| máx en 3–4 s = "
          f"{np.max(abs(x[t >= 3])):.4f}; |du/dt| máx en 2–4 s = {np.max(du[t[1:] >= 2]):.1f}; ¿acotado? {ok}")
for nom, Gm_, wf in casos7[1:3]:
    o = l1(Gm_, wf, retardo=0.0)
    t, x = o[:, 0], o[:, 1]
    print(f"  sin retardo, {nom:30s}: |x| máx en 1–4 s = {np.max(abs(x[t >= 1])):.4f}")
print(f"  error de predicción con Γ grande: x̃' = a_m x̃ + σ̃, σ̂' = -Γ x̃  ->  s² - a_m s + Γ = 0: ω_n = √Γ; "
      f"con Γ = 10⁴, ω_n = 100 rad/s ({100/(2*math.pi):.1f} Hz) y amortiguamiento |a_m|/(2√Γ) = {abs(am7)/(2*100):.3f}")
X7, Y7 = lin(0.8, 2.0, 60, 520), lin(-4.0, 1.0, 230, 20)
for nom in ("sin filtro, Γ = 10⁴", "filtro ω = 10 rad/s, Γ = 10⁴"):
    o = res7[nom]
    sel = (o[:, 0] >= 0.8) & (o[:, 0] <= 2.0)
    print(f"path L1 u {nom}:", path(o[sel, 0], np.clip(o[sel, 2], -4, 1), X7, Y7))
print(f"  ejes: u = 0 en y = {Y7(0):.1f}, -2 en {Y7(-2):.1f}, -4 en {Y7(-4):.1f}, 1 en {Y7(1):.1f}; t = 1 en x = {X7(1):.1f}, "
      f"1,5 en {X7(1.5):.1f}, 2 en {X7(2):.1f}")

# ---------------------------------------------------------------------------
cab(8, "EQUIVALENCIA CIERTA, CAUTELA Y SONDEO: y = b u + w, OBJETIVO y = 1")
bh, P8, sw2 = 0.5, 1.0, 0.25
print(f"  b ~ N(b̂ = {bh}, P = {P8}), ruido w ~ N(0, {sw2}). Coste de un paso: E[(y-1)²] = (b̂u - 1)² + P u² + σ_w²")
J1 = lambda u: (bh * u - 1) ** 2 + P8 * u ** 2 + sw2
u_ce = 1 / bh
u_ca = bh / (bh ** 2 + P8)
print(f"  equivalencia cierta: u = 1/b̂ = {u_ce}, coste esperado {J1(u_ce):.3f}")
print(f"  cauteloso: dJ/du = 2b̂(b̂u - 1) + 2Pu = 0 -> u = b̂/(b̂² + P) = {u_ca}, coste {J1(u_ca):.3f}")
print(f"  no hacer nada: u = 0, coste {J1(0):.3f}")
for u in (0.0, u_ca, 1.0, u_ce):
    Pp = P8 * sw2 / (P8 * u ** 2 + sw2)
    print(f"    tras aplicar u = {u:.2f}, varianza de b: P' = P σ_w²/(P u² + σ_w²) = {Pp:.4f}")
# dos pasos: el segundo, cauteloso con el posterior; el primero se elige mirando también el segundo
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(80)
gh_w = gh_w / gh_w.sum()


def J2(u1):
    Pp = P8 * sw2 / (P8 * u1 ** 2 + sw2)
    sd_bh = math.sqrt(max(P8 - Pp, 0.0))            # dispersión de la media a posteriori antes de observar
    bps = bh + sd_bh * gh_x
    c2 = np.sum(gh_w * (Pp / (bps ** 2 + Pp))) + sw2
    return J1(u1) + c2


us = np.linspace(0, 3, 3001)
Js = np.array([J2(u) for u in us])
u_du = us[np.argmin(Js)]
print(f"  dos pasos (el segundo cauteloso con el posterior): J(u1) = coste1(u1) + E[coste2]")
for u in (0.0, u_ca, u_du, u_ce):
    print(f"    u1 = {u:.3f}: coste del paso 1 = {J1(u):.3f}, total = {J2(u):.3f}")
print(f"  u1 óptimo (dual) = {u_du:.3f} > cauteloso {u_ca}: el primer mando sondea")
X8, Y8 = lin(0, 3, 60, 520), lin(1.0, 3.0, 230, 20)
print("path dual J total:", path(us[::30], np.clip(Js[::30], 1, 3), X8, Y8))
print("path dual coste paso 1 + coste2 sin aprender:", path(us[::30], np.clip(np.array([J1(u) + (P8 / (bh**2 + P8) + sw2) for u in us[::30]]), 1, 3), X8, Y8))
print(f"  marcas: cauteloso ({X8(u_ca):.1f},{Y8(J2(u_ca)):.1f}); dual ({X8(u_du):.1f},{Y8(J2(u_du)):.1f}); "
      f"cierta ({X8(u_ce):.1f},{Y8(J2(u_ce)):.1f}); ejes J = 1 en {Y8(1):.1f}, 2 en {Y8(2):.1f}, 3 en {Y8(3):.1f}; "
      f"u = 1 en x = {X8(1):.1f}, 2 en {X8(2):.1f}")
print(f"  coste del segundo paso si no se aprendiera nada (P' = P): {P8/(bh**2+P8) + sw2:.3f}")

# ---------------------------------------------------------------------------
cab(9, "NEURAL-FLY-CONSTANT (φ = I) COMO KALMAN: EL KALMAN f DEL x500 EN RÉGIMEN PERMANENTE")
Pm = QF
for _ in range(10000):
    Pp_ = Pm * RN / (Pm + RN)        # tras corregir
    Pm = Pp_ + QF                     # predicción del paso siguiente
Kss = Pm / (Pm + RN)
print(f"  por componente: Q = {QF} N²/paso, R = {RN:.2f} N², Δt = {DTC} s (50 Hz)")
print(f"  Riccati discreta: P⁻ = {Pm:.4f} N², P⁺ = {Pm*RN/(Pm+RN):.4f} N², ganancia K = P⁻/(P⁻ + R) = {Kss:.4f}")
print(f"  constante de tiempo: -Δt/ln(1-K) = {-DTC/math.log(1-Kss)*1000:.1f} ms; tras un escalón de 3 N, f̂ en 1, 2, 5, 10 pasos = "
      f"{[round(3*(1-(1-Kss)**n), 3) for n in (1, 2, 5, 10)]}")
Qc, Rc = QF / DTC, RN * DTC
print(f"  límite continuo (λ = 0): Q_c = Q/Δt = {Qc:.2f} N²/s, R_c = R Δt = {Rc:.4f} N² s; P∞ = √(Q_c R_c) = {math.sqrt(Qc*Rc):.4f}; "
      f"ganancia P∞/R_c = √(Q_c/R_c) = {math.sqrt(Qc/Rc):.2f} 1/s -> τ = {1000/math.sqrt(Qc/Rc):.1f} ms")
for lam in (0.0, 1.0, 5.0):
    Pinf = Rc * (-lam + math.sqrt(lam ** 2 + Qc / Rc))
    print(f"    con λ = {lam}: 0 = -2λP + Q_c - P²/R_c -> P∞ = R_c(-λ + √(λ² + Q_c/R_c)) = {Pinf:.4f}; ganancia P∞/R_c = {Pinf/Rc:.2f} 1/s")

# ---------------------------------------------------------------------------
cab(10, "MAML CON DOS TAREAS CUADRÁTICAS EN UNA DIMENSIÓN")
h = np.array([1.0, 4.0]); c = np.array([-1.0, 2.0]); al = 0.2
print(f"  L_i(θ) = ½ h_i (θ - c_i)², h = {h}, c = {c}; soporte y consulta con la misma pérdida; α = {al}")


def F(th):
    thp = th - al * h * (th - c)
    return np.sum(0.5 * h * (thp - c) ** 2)


th = 0.0
thp = th - al * h * (th - c)
g_exact = np.sum((1 - al * h) * h * (thp - c))
g_fo = np.sum(h * (thp - c))
eps = 1e-6
g_fd = (F(th + eps) - F(th - eps)) / (2 * eps)
print(f"  en θ = 0: θ'_i = θ - α h_i(θ - c_i) = {thp}; 1 - α h_i = {1 - al*h}")
print(f"  gradiente MAML Σ(1 - αh_i) h_i (θ'_i - c_i) = {g_exact:.4f}; diferencias finitas {g_fd:.4f}; primer orden Σ h_i(θ'_i - c_i) = {g_fo:.4f}")
w_maml = h * (1 - al * h) ** 2
w_fo = h * (1 - al * h)
print(f"  óptimos: MAML θ = Σ w c/Σ w con w = h(1-αh)² = {w_maml} -> {np.sum(w_maml*c)/np.sum(w_maml):.4f}; "
      f"primer orden w = h(1-αh) = {w_fo} -> {np.sum(w_fo*c)/np.sum(w_fo):.4f}; "
      f"preentrenar en todas (sin paso interno) w = h -> {np.sum(h*c)/np.sum(h):.4f}")
for nom, t0 in (("MAML", np.sum(w_maml*c)/np.sum(w_maml)), ("primer orden", np.sum(w_fo*c)/np.sum(w_fo)), ("preentrenado", np.sum(h*c)/np.sum(h))):
    print(f"    {nom:13s}: θ = {t0:.3f}, pérdida tras un paso interno en cada tarea = {F(t0):.4f}")

# ---------------------------------------------------------------------------
cab(11, "BELIEF SOBRE LA TAREA: ¿DRON LIGERO (2,0 kg) O PESADO (2,6 kg)?")
T11, sa = 22.0, 1.5
mus = {"ligero": T11 / 2.0 - G, "pesado": T11 / 2.6 - G}
print(f"  empuje T = {T11} N conocido; a_z = T/m - g + ruido N(0, {sa}²): ligero predice {mus['ligero']:.3f}, pesado {mus['pesado']:.3f} m/s²")
bel = {"ligero": 0.5, "pesado": 0.5}
for n, aobs in enumerate((-1.0, -0.4, -1.6), 1):
    lik = {k: math.exp(-(aobs - mu) ** 2 / (2 * sa ** 2)) for k, mu in mus.items()}
    num = {k: lik[k] * bel[k] for k in bel}
    Z = sum(num.values())
    print(f"  medida {n}: a_z = {aobs}; verosimilitudes exp(-(a-μ)²/2σ²) = ligero {lik['ligero']:.4f}, pesado {lik['pesado']:.4f}; "
          f"belief ligero {bel['ligero']:.3f} -> {num['ligero']/Z:.3f}, pesado {bel['pesado']:.3f} -> {num['pesado']/Z:.3f}")
    bel = {k: num[k] / Z for k in num}

# ---------------------------------------------------------------------------
cab(12, "MODELO DE MEDIDA DEL x500: y = T R e3 = m s - f")
m12, f12 = 2.6, np.array([3.0, -1.0, 0.0])
s_h = np.array([0, 0, G])
y12 = m12 * s_h - f12
print(f"  flotando (p'' = 0): s = p'' + g e3 = {s_h}; con m = {m12} kg y f = {f12} N: y = m s - f = {np.round(y12, 3)} N")
alt = (2.0, 2.0 * G - y12[2])
print(f"  la misma y_z = {y12[2]:.3f} con (m, f_z) = (2.6, 0) y con (2.0, {alt[1]:.3f}): flotando no se distinguen")
sz8 = G + A_osc
print(f"  con aceleración vertical {A_osc:.3f} m/s² (cresta del ocho): s_z = {sz8:.3f}; y_z = {m12*sz8:.3f} (2,6 kg, f_z = 0) frente a "
      f"{2.0*sz8 - alt[1]:.3f} (2,0 kg, f_z = {alt[1]:.2f}): difieren {m12*sz8 - (2.0*sz8 - alt[1]):.3f} N (ruido supuesto 0,4 N)")
print(f"  resistencia supuesta: k_l v + k_q v² a 10 m/s = 0,15·10 + 0,03·100 = {0.15*10 + 0.03*100} N")

# ---------------------------------------------------------------------------
cab(13, "LA PRIMERA CORRECCIÓN DEL KALMAN (m, f_z) AL FLOTAR: CÓMO SE REPARTE LA SORPRESA")
for nom, P0 in (("Kalman (m, f)", np.diag([0.3 ** 2, 2.0 ** 2])), ("f_z ≈ 0", np.diag([0.3 ** 2, 0.3 ** 2]))):
    Q = np.diag([QM, QF if nom == "Kalman (m, f)" else QF / 100])
    th = np.array([2.0, 0.0])
    phi = np.array([G, -1.0])
    y = m_v * G
    P = P0 + Q
    S = phi @ P @ phi + RN
    K = P @ phi / S
    inn = y - phi @ th
    d = K * inn
    print(f"  {nom}: P⁻ = diag{np.round(np.diag(P), 4)}; S = φᵀP⁻φ + R = {S:.3f}; K = P⁻φ/S = {np.round(K, 4)}; "
          f"sorpresa y - φᵀθ̂ = {inn:.3f} N; Δm = {d[0]:.4f} kg, Δf_z = {d[1]:.4f} N; Δm·g - Δf_z = {d[0]*G - d[1]:.3f} N")
print(f"  reparto relativo de la sorpresa: masa P_mm g² : fuerza P_ff = {0.09*G*G:.2f} : 4,05 (Kalman (m, f)) frente a "
      f"{0.09*G*G:.2f} : {0.09 + QF/100:.4f} (f_z ≈ 0)")
X13, Y13 = lin(1.8, 2.8, 70, 330), lin(-7.0, 1.0, 250, 20)
for s_ in (G, G + A_osc, G - A_osc):
    yz = m_v * s_
    f_a, f_b = 1.8 * s_ - yz, 2.8 * s_ - yz
    print(f"  recta s_z = {s_:.3f}: f_z = m s_z - {yz:.3f}; píxeles {X13(1.8):.1f},{Y13(f_a):.1f} -> {X13(2.8):.1f},{Y13(f_b):.1f}")
print(f"  banda f_z ∈ [-0,3, 0,3]: y de {Y13(0.3):.1f} a {Y13(-0.3):.1f}")
