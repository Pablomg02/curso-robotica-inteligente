"""Ejemplos numéricos y datos de las figuras nuevas de A10 (imitation learning y políticas generativas).

Cada número de un ejemplo hecho a mano en la presentación sale de aquí. Las cifras del experimento del x500
(MPPI, BC, flow matching, DAgger, tiempos) siguen saliendo de figuras_a10.py (JAX en GPU), que no hace falta
relanzar: su salida está en salida_gpu.txt y salida_gpu_y015.txt.

Ejecutar:  python3 ejemplos_a10.py > salida_ejemplos.txt     (solo numpy, CPU, unos segundos)
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)


def titulo(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ===========================================================================
titulo("1. BC como regresión: una política lineal de un parámetro, a mano")
o = np.array([-0.2, -0.1, 0.1, 0.2])        # desviación lateral y (m)
a = np.array([0.5, 0.3, -0.2, -0.4])        # orden lateral del experto (normalizada)
th = (o * a).sum() / (o ** 2).sum()
print("  pares (o, a):", list(zip(o.tolist(), a.tolist())))
print(f"  sum o·a = {(o * a).sum():.3f}, sum o² = {(o ** 2).sum():.3f}  ->  theta* = {th:.3f}")
for t in (-2.0, th, -3.0):
    L = np.mean((t * o - a) ** 2)
    print(f"  theta = {t:6.3f}: predicciones {np.round(t * o, 3)}, pérdida media {L:.4f}")
print("  derivada de la pérdida: dL/dtheta = (2/N) sum o (theta o - a); en theta*:",
      f"{2 / 4 * (o * (th * o - a)).sum():.2e}")
# tamaños del x500 (figuras_a10.py)
mac_bc = 18 * 256 + 256 * 256 + 256 * 256 + 256 * 4
n_in_f = 18 + 8 * 4 + 10
mac_f = n_in_f * 256 + 256 * 256 + 256 * 256 + 256 * 32
print(f"  x500: observación 18, acción 4, 300 demos x 300 pasos = {300 * 300} pares")
print(f"  multiplicaciones BC por paso: 18·256 + 256·256 + 256·256 + 256·4 = {mac_bc}")
print(f"  flujo: entrada 18 + 32 + 10 = {n_in_f}; por pasada {n_in_f}·256 + 2·256² + 256·32 = {mac_f}; x10 = {10 * mac_f}")
print(f"  MPPI: 40 pasos x 256 secuencias x 3 iteraciones = {40 * 256 * 3} pasos de la física")

# ===========================================================================
titulo("2. Desplazamiento de covariables: coste extra acumulado (como el ejemplo 6 de A6b)")
T, eps = 100, 0.01
t = np.arange(1, T + 1)
pf = 1 - (1 - eps) ** t
for tt in (1, 10, 50, 100):
    print(f"  t = {tt:3d}: P(ya falló) = {pf[tt - 1]:.3f}, cota t·eps = {tt * eps:.2f}, acumulado {pf[:tt].sum():.2f}")
print(f"  total {pf.sum():.1f}; cota con t·eps {np.sum(t * eps):.1f}; T²eps = {T * T * eps:.0f}")

# ===========================================================================
titulo("3. Media condicionada con el poste: a = +1 (izquierda) o -1 (derecha)")
for p in (0.5, 0.48):
    m = p * 1 + (1 - p) * (-1)
    var = 1 - m ** 2
    print(f"  P(izq) = {p}: E[a|o] = {m:+.2f}; pérdida mínima (varianza) = {var:.4f}")
    for ah in (-1, -0.5, 0, 0.5, 1):
        J = p * (ah - 1) ** 2 + (1 - p) * (ah + 1) ** 2
        print(f"     J({ah:+.1f}) = {J:.4f}")
# gaussiana ajustada por máxima verosimilitud a los dos modos
print("  gaussiana de máxima verosimilitud con 50/50 en ±1: mu = 0, sigma = 1;",
      f"densidad en a = 0: {1 / math.sqrt(2 * math.pi):.3f}, en a = ±1: {math.exp(-0.5) / math.sqrt(2 * math.pi):.3f}")
X, Y = lin(-1.5, 1.5, 50, 410), lin(0, 6.25, 220, 20)
ah = np.linspace(-1.5, 1.5, 61)
J = 0.5 * (ah - 1) ** 2 + 0.5 * (ah + 1) ** 2
print("  path J(â) (p = 0,5):", path(ah, J, X, Y))
print(f"  ejes: â = -1, 0, 1 en x = {X(-1):.1f}, {X(0):.1f}, {X(1):.1f}; J = 0, 1, 2, 4 en y = {Y(0):.1f}, {Y(1):.1f}, {Y(2):.1f}, {Y(4):.1f}")

# ===========================================================================
titulo("4. Modelo de juguete: por qué la media no se queda en el centro")
h = 0.05      # anchura (m) de la franja en la que el lado es incierto
print("  P(izq | y) = Phi(y/h), h = 0,05 m;  media E[a|y] = 2 Phi(y/h) - 1")
for y in (0.0, 0.01, 0.02, 0.05, 0.10):
    print(f"    y = {y:+.2f} m: P(izq) = {Phi(y / h):.3f}, E[a|y] = {2 * Phi(y / h) - 1:+.3f}")
c = 0.01      # metros que se desplaza por paso con a = 1
y = 0.001
tray = [y]
for k in range(12):
    y = y + c * (2 * Phi(y / h) - 1)
    tray.append(y)
print("  y_{k+1} = y_k + 0,01·E[a|y_k], desde y_0 = 1 mm:", np.round(np.array(tray) * 100, 2), "cm")
X, Y = lin(-0.1, 0.1, 50, 410), lin(-1, 1, 200, 30)
ys = np.linspace(-0.1, 0.1, 81)
print("  path E[a|y]:", path(ys, [2 * Phi(v / h) - 1 for v in ys], X, Y))
print(f"  ejes: y = -0,1, -0,05, 0, 0,05, 0,1 en x = {X(-0.1):.0f}, {X(-0.05):.0f}, {X(0):.0f}, {X(0.05):.0f}, {X(0.1):.0f}; a = -1, 0, 1 en y = {Y(-1):.0f}, {Y(0):.0f}, {Y(1):.0f}")

# ===========================================================================
titulo("5. BC implícito: energía con dos mínimos y pérdida InfoNCE")
def energia(av):
    return -np.log(0.5 * np.exp(-(av - 1) ** 2 / (2 * 0.3 ** 2)) + 0.5 * np.exp(-(av + 1) ** 2 / (2 * 0.3 ** 2)))
ag = np.linspace(-2, 2, 401)
Eg = energia(ag)
print(f"  E(o, a) = -log(½ e^(-(a-1)²/0,18) + ½ e^(-(a+1)²/0,18)); mínimos en a = ±{abs(ag[np.argmin(Eg)]):.2f}, E = {Eg.min():.3f}; E(0) = {energia(0.0):.3f}")
Ep, En = 0.2, np.array([2.0, 1.5, 3.0])
pt = math.exp(-Ep) / (math.exp(-Ep) + np.exp(-En).sum())
print(f"  InfoNCE: E(experto) = {Ep}, E(contraejemplos) = {En.tolist()}; e^-E: {math.exp(-Ep):.3f}, {np.round(np.exp(-En), 3).tolist()}")
print(f"    p~ = {pt:.3f}, pérdida -log p~ = {-math.log(pt):.3f}")
pt2 = math.exp(-Ep) / (math.exp(-Ep) + np.exp(-(En + 1)).sum())
print(f"    con las energías de los contraejemplos 1 más altas: p~ = {pt2:.3f}, pérdida {-math.log(pt2):.3f}")
X, Y = lin(-2, 2, 50, 410), lin(0, 6, 210, 30)
print("  path energía:", path(ag[::5], np.minimum(Eg[::5], 6), X, Y))
print(f"  ejes: a = -2, -1, 0, 1, 2 en x = {X(-2):.0f}, {X(-1):.0f}, {X(0):.0f}, {X(1):.0f}, {X(2):.0f}; E = 0, {energia(0.0):.2f} en y = {Y(0):.1f}, {Y(energia(0.0)):.1f}; mínimo en y = {Y(Eg.min()):.1f}")

# ===========================================================================
titulo("6. Behavior Transformers: k-means con k = 2 y desplazamiento")
acc = np.array([0.9, 1.1, 1.0, -0.95, -1.05, -1.0])
cen = np.array([0.5, -0.5])
for it in range(5):
    lab = np.argmin(np.abs(acc[:, None] - cen[None]), 1)
    cen = np.array([acc[lab == j].mean() for j in range(2)])
print("  acciones:", acc.tolist(), "-> centros", np.round(cen, 3).tolist())
for av in (0.9, -1.05):
    j = int(np.argmin(np.abs(av - cen)))
    print(f"  a = {av:+.2f}: grupo {j} (centro {cen[j]:+.2f}), desplazamiento {av - cen[j]:+.2f}")

# ===========================================================================
titulo("7. GAIL: discriminador óptimo y coste que se deriva de él")
print("  max_D  rho_pi log D + rho_E log(1-D)  ->  D* = rho_pi / (rho_pi + rho_E)")
for rp, re_ in ((0.3, 0.1), (0.2, 0.2), (0.1, 0.3), (0.02, 0.3)):
    Ds = rp / (rp + re_)
    Dg = np.linspace(0.001, 0.999, 9999)
    Dnum = Dg[np.argmax(rp * np.log(Dg) + re_ * np.log(1 - Dg))]
    print(f"    rho_pi = {rp:.2f}, rho_E = {re_:.2f}: D* = {Ds:.3f} (búsqueda numérica {Dnum:.3f}); coste log D* = {math.log(Ds):+.3f}; recompensa -log D* = {-math.log(Ds):.3f}")

# ===========================================================================
titulo("8. ACT: trozos y ensamblado temporal")
print(f"  x500: 300 pasos; con trozos que se ejecutan de 4 en 4, {300 // 4} consultas. ACT con k = 100 a 50 Hz: {100 / 50:.0f} s por trozo")
pred = np.array([0.40, 0.44, 0.50, 0.52])   # predicciones para el mismo instante t, de la más antigua (i = 0) a la más nueva
for m in (0.0, 0.5, 2.0):
    w = np.exp(-m * np.arange(4))
    print(f"  m = {m}: pesos {np.round(w, 3).tolist()}, suma {w.sum():.3f}; acción ensamblada {np.sum(w * pred) / w.sum():.4f}")
print("  (m = 0 es la media simple; m grande se queda con la más antigua)")

# ===========================================================================
titulo("9. Difusión: programa coseno (iDDPM, el de Diffusion Policy) y proceso hacia delante")
Nd = 100
sc = 0.008
f = lambda n: math.cos((n / Nd + sc) / (1 + sc) * math.pi / 2) ** 2
abar = np.array([f(n) / f(0) for n in range(Nd + 1)])
for n in (0, 10, 25, 50, 75, 90, 100):
    print(f"  n = {n:3d}: alfa_barra = {abar[n]:.4f}, sqrt = {math.sqrt(abar[n]):.3f}, ruido sqrt(1-alfa_barra) = {math.sqrt(1 - abar[n]):.3f}")
x0, ab, e = 0.8, 0.5, -1.2
xn = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * e
eh = -1.0
x0h = (xn - math.sqrt(1 - ab) * eh) / math.sqrt(ab)
print(f"  a mano: x0 = {x0}, alfa_barra = {ab}, eps = {e}: x^n = {math.sqrt(ab):.3f}·{x0} + {math.sqrt(1 - ab):.3f}·({e}) = {xn:.3f}")
print(f"    la red predice eps^ = {eh}: x0^ = (x^n - {math.sqrt(1 - ab):.3f}·eps^)/{math.sqrt(ab):.3f} = {x0h:.3f}; error de ruido² = {(e - eh) ** 2:.3f}")
# quitarruidos óptimo para datos en ±1: E[x0 | x^n] = tanh(sqrt(ab) x / (1 - ab))
for ab_, xv in ((0.1, 0.2), (0.5, 0.2), (0.9, 0.2), (0.5, 1.0)):
    print(f"  quitarruidos ideal, alfa_barra = {ab_}, x^n = {xv}: E[x0|x^n] = tanh({math.sqrt(ab_) * xv / (1 - ab_):.3f}) = {math.tanh(math.sqrt(ab_) * xv / (1 - ab_)):.3f}")
# densidades para la figura: datos ±1 con desviación 0,1
xs = np.linspace(-3, 3, 121)
sd0 = 0.1
for n in (0, 30, 60, 100):
    m_, v_ = math.sqrt(abar[n]), abar[n] * sd0 ** 2 + 1 - abar[n]
    d = 0.5 * (np.exp(-(xs - m_) ** 2 / (2 * v_)) + np.exp(-(xs + m_) ** 2 / (2 * v_))) / math.sqrt(2 * math.pi * v_)
    Xp, Yp = lin(-3, 3, 0, 200), lin(0, d.max(), 100, 12)
    print(f"  densidad n = {n} (alfa_barra {abar[n]:.3f}, máx {d.max():.2f}):", path(xs, d, Xp, Yp))
# hacia atrás: DDIM determinista con 10 pasos y el quitarruidos ideal
ns = np.linspace(Nd, 0, 11).round().astype(int)
z0 = np.array([-1.9, -1.2, -0.7, -0.25, 0.1, 0.45, 0.9, 1.6])
X, Y = lin(0, 10, 60, 520), lin(-2.5, 2.5, 230, 20)
print("  DDIM, 10 pasos, n:", ns.tolist())
fin = []
for z in z0:
    x = z
    tr = [x]
    for j in range(10):
        a1, a0 = max(abar[ns[j]], 1e-8), abar[ns[j + 1]]
        x0h = math.tanh(math.sqrt(a1) * x / (1 - a1)) if a1 < 1 else x
        eh = (x - math.sqrt(a1) * x0h) / math.sqrt(1 - a1)
        x = math.sqrt(a0) * x0h + math.sqrt(1 - a0) * eh
        tr.append(x)
    fin.append(x)
    print(f"    desde {z:+.2f}: path", path(range(11), tr, X, Y))
print("  finales:", np.round(fin, 3).tolist())

# ===========================================================================
titulo("10. Horizontes: cuánto dura cada trozo")
print(f"  Diffusion Policy con T_a = 8: 8 acciones; a 50 Hz serían {8 / 50 * 1000:.0f} ms entre inferencias")
print(f"  x500: 4 de 8 a 50 Hz = {4 / 50 * 1000:.0f} ms ejecutados, {8 / 50 * 1000:.0f} ms predichos")
print(f"  pi0: 25 a 50 Hz = {25 / 50:.1f} s; 16 a 20 Hz = {16 / 20:.1f} s; trozo de 50 a 50 Hz = {50 / 50:.0f} s")

# ===========================================================================
titulo("11. Flow matching en 1-D: datos en ±1, ruido N(0, 1)")
x0, x1 = -0.5, 1.0
print(f"  camino lineal x0 = {x0}, x1 = {x1}: velocidad x1 - x0 = {x1 - x0}")
for tt in (0, 0.25, 0.5, 0.75, 1):
    print(f"    t = {tt:.2f}: x_t = {(1 - tt) * x0 + tt * x1:+.3f}")


def post_mas(xv, tt):
    """P(x1 = +1 | x_t = x): x_t | x1 ~ N(t x1, (1-t)²)."""
    return 1 / (1 + math.exp(-2 * tt * xv / (1 - tt) ** 2))


def u_marg(xv, tt):
    return (math.tanh(tt * xv / (1 - tt) ** 2) - xv) / (1 - tt)


tt, xv = 0.5, 0.3
g1 = math.exp(-(xv - tt) ** 2 / (2 * (1 - tt) ** 2))
g2 = math.exp(-(xv + tt) ** 2 / (2 * (1 - tt) ** 2))
pp = g1 / (g1 + g2)
v1, v2 = (1 - xv) / (1 - tt), (-1 - xv) / (1 - tt)
um = pp * v1 + (1 - pp) * v2
print(f"  en t = {tt}, x = {xv}: verosimilitudes {g1:.3f} (x1 = +1) y {g2:.3f} (x1 = -1); P(+1|x) = {pp:.3f}")
print(f"    velocidades condicionales (x1 - x)/(1 - t): {v1:+.2f} y {v2:+.2f}")
print(f"    campo marginal u = {pp:.3f}·{v1:.2f} + {1 - pp:.3f}·({v2:.2f}) = {um:.3f};  fórmula cerrada {u_marg(xv, tt):.3f}")
print(f"    E[x1|x] = 2P - 1 = {2 * pp - 1:.3f} = tanh(t x/(1-t)²) = tanh({tt * xv / (1 - tt) ** 2:.2f}) = {math.tanh(tt * xv / (1 - tt) ** 2):.3f}")
# teorema 2 en un punto: la mejor constante para la pérdida condicional es el campo marginal
for v in (0.0, um, 1.4):
    Lc = pp * (v - v1) ** 2 + (1 - pp) * (v - v2) ** 2
    Lm = (v - um) ** 2
    print(f"    v = {v:.3f}: pérdida condicional {Lc:.3f}, pérdida marginal {Lm:.3f}, diferencia {Lc - Lm:.3f}")
# teorema 2 por Monte Carlo: L_CFM - L_FM no depende de v
M = 400000
t_ = rng.uniform(0, 0.98, M)
x0_ = rng.standard_normal(M)
x1_ = np.where(rng.uniform(size=M) < 0.5, 1.0, -1.0)
xt_ = (1 - t_) * x0_ + t_ * x1_
uc = x1_ - x0_
um_ = (np.tanh(t_ * xt_ / (1 - t_) ** 2) - xt_) / (1 - t_)
for nombre, vf in (("v = 0", np.zeros(M)), ("v = 0,5 x", 0.5 * xt_), ("v = u (el marginal)", um_)):
    Lc, Lm = np.mean((vf - uc) ** 2), np.mean((vf - um_) ** 2)
    print(f"    Monte Carlo ({M} muestras, t en [0, 0,98]) {nombre:22s}: L_CFM = {Lc:.3f}, L_FM = {Lm:.3f}, diferencia = {Lc - Lm:.3f}")

# Euler con el campo marginal exacto
def euler(z, N):
    x = z
    tr = [x]
    for i in range(N):
        tt = i / N
        x = x + u_marg(x, tt) / N
        tr.append(x)
    return x, tr


print("  Euler, 10 pasos, x <- x + 0,1·u(x, t), t = 0; 0,1; ...; 0,9:")
for z in (0.3, -0.8, 0.05):
    x = z
    filas = []
    for i in range(10):
        tt = i / 10
        u = u_marg(x, tt)
        filas.append(f"t={tt:.1f}: x={x:+.3f}, u={u:+.3f}")
        x = x + 0.1 * u
    print(f"    desde x0 = {z:+.2f}:")
    for fl in filas:
        print("       ", fl)
    print(f"        final x = {x:+.4f}")
zs = rng.standard_normal(20000)
for N in (1, 2, 3, 5, 10, 30):
    fx = np.array([euler(z, N)[0] for z in zs])
    print(f"  N = {N:2d}: |x - (±1)| < 0,1 en el {np.mean(np.abs(np.abs(fx) - 1) < 0.1) * 100:5.1f} %; media |x| = {np.mean(np.abs(fx)):.3f}; fracción > 0: {np.mean(fx > 0):.3f}")
print("  N = 1 da x0 + (E[x1] - x0) = 0 para todo x0; N = 2 da tanh(x0): ", [round(euler(z, 2)[0], 4) for z in (0.3, 1.0)],
      [round(math.tanh(z), 4) for z in (0.3, 1.0)])
X, Y = lin(0, 1, 60, 520), lin(-2.5, 2.5, 230, 20)
z12 = np.array([-2.0, -1.4, -1.0, -0.7, -0.4, -0.15, 0.1, 0.35, 0.6, 0.9, 1.3, 1.9])
for z in z12:
    _, tr = euler(z, 10)
    print(f"  path euler desde {z:+.2f}:", path(np.linspace(0, 1, 11), tr, X, Y), f" final {tr[-1]:+.3f}")
# caminos condicionales (rectos) para el esquema
pares = [(-1.6, 1), (-0.9, -1), (-0.3, 1), (0.2, -1), (0.7, 1), (1.4, -1)]
for a_, b_ in pares:
    print(f"  recta condicional {a_:+.1f} -> {b_:+d}: M{X(0):.1f},{Y(a_):.1f} L{X(1):.1f},{Y(b_):.1f}")
print(f"  ejes: x = -2, -1, 0, 1, 2 en y = {Y(-2):.0f}, {Y(-1):.0f}, {Y(0):.0f}, {Y(1):.0f}, {Y(2):.0f}; t = 0, 0,5, 1 en x = {X(0):.0f}, {X(0.5):.0f}, {X(1):.0f}")
# pérdida mínima de CFM (irreducible) en el juguete
print(f"  pérdida CFM mínima del juguete (v = u): {np.mean((um_ - uc) ** 2):.3f}; varianza de la velocidad condicional sin condicionar: {np.var(uc):.3f}")

# ===========================================================================
titulo("12. Rectified flow (reflow) en el mismo juguete")
zs2 = rng.standard_normal(4000)
z1 = np.array([euler(z, 30)[0] for z in zs2])   # acoplamiento (x0, phi_1(x0)) con el flujo de 30 pasos
orden = np.argsort(zs2)
print(f"  ¿phi_1 es monótona? {bool(np.all(np.diff(z1[orden]) >= -1e-9))}  (en 1-D las trayectorias del flujo no se cruzan)")
# con un acoplamiento monótono, las rectas x_t = (1-t) x0 + t phi_1(x0) no se cruzan: la velocidad en cada punto es phi_1(x0) - x0
# y un solo paso de Euler desde t = 0 da x0 + (phi_1(x0) - x0) = phi_1(x0)
for z in (0.3, -0.8, 0.05):
    fz = euler(z, 30)[0]
    print(f"    x0 = {z:+.2f}: phi_1(x0) = {fz:+.4f}; flujo original con 1 paso: {euler(z, 1)[0]:+.4f}; flujo rectificado con 1 paso: {z + (fz - z):+.4f}")
lineas = np.stack([(1 - tt) * zs2 + tt * z1 for tt in (0.25, 0.5, 0.75)])
print("  ¿las rectas del acoplamiento siguen ordenadas en t = 0,25; 0,5; 0,75?",
      [bool(np.all(np.diff(l[orden]) >= -1e-9)) for l in lineas])

# ===========================================================================
titulo("13. Destilación: cuentas de latencia")
print(f"  Tagliabue et al.: 55,3 / 0,25 = {55.3 / 0.25:.0f} veces; a 100 Hz el presupuesto es {1000 / 100:.0f} ms; TX2 1,66 ms = {1.66 / 10 * 100:.1f} % del presupuesto")
print(f"  x500: 10,0 / 0,021 = {10.0 / 0.021:.0f} veces; flow matching 0,081 ms por trozo / 4 pasos = {0.081 / 4:.3f} ms por paso; a 50 Hz, presupuesto {1000 / 50:.0f} ms")
# aumentación por muestreo dentro del tubo (juguete 1-D)
xr, ur, K, w = 0.0, 0.2, -1.5, 0.1
for xp in (-w, w):
    print(f"  tubo: x_nominal = {xr}, u_nominal = {ur}, K = {K}, x+ = {xp:+.1f}: u+ = {ur} + ({K})·({xp:+.1f}) = {ur + K * (xp - xr):.2f}")
print("  variante dispersa: 2 n_x muestras (centros de las caras); con n_x = 2, 4 muestras por estado")
