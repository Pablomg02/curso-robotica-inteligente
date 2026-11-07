"""Ejemplos numéricos y datos de las figuras nuevas de A4a (filtrado bayesiano).

Cada número de un ejemplo «a mano» de la presentación sale de aquí; las figuras
de resultados (pasillo, NEES, comparación no lineal, guiñada) siguen saliendo de
figuras_a4a.py, que no hace falta relanzar.
Ejecutar:  python3 ejemplos_a4a.py > salida_ejemplos.txt   (numpy, CPU, segundos)
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def titulo(n, s):
    print("\n" + "=" * 72 + f"\nEjemplo {n}. {s}\n" + "=" * 72)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def gauss(x, m, P):
    return np.exp(-0.5 * (x - m) ** 2 / P) / math.sqrt(2 * math.pi * P)


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])


def Exp(phi):
    a = np.linalg.norm(phi)
    if a < 1e-12:
        return np.eye(3) + hat(phi)
    K = hat(phi / a)
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def chi2_cdf(x, k):
    a, z = k / 2, x / 2
    s, term, n = 0.0, 1.0 / a, 0
    while term > 1e-14 * s or n < 5:
        s += term; n += 1; term *= z / (a + n)
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * s


def chi2_ppf(p, k):
    lo, hi = 0.0, 4.0 * k + 100
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if chi2_cdf(mid, k) < p else (lo, mid)
    return (lo + hi) / 2


rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
titulo(1, "Filtro de Bayes a mano: pasillo de 5 celdas, puertas en las celdas 3 y 5")
bel0 = np.array([0.25, 0.50, 0.25, 0.0, 0.0])
p_av, p_qu = 0.8, 0.2                         # mando «avanza una celda»: avanza con 0,8, se queda con 0,2
T = np.zeros((5, 5))                          # T[i, j] = p(x_k = i | x_{k-1} = j, u)
for j in range(5):
    if j < 4:
        T[j, j] += p_qu; T[j + 1, j] += p_av
    else:
        T[j, j] += 1.0                        # la celda 5 es el final del pasillo
pred = T @ bel0
puerta = np.array([0, 0, 1, 0, 1])
lik = np.where(puerta == 1, 0.9, 0.1)         # p(y = «puerta» | x): 0,9 junto a una puerta, 0,1 junto a pared
prod = lik * pred
eta = 1 / prod.sum()
bel1 = eta * prod
print("  bel(x_{k-1})          =", bel0)
for i in range(5):
    terms = [f"{T[i,j]:.1f}·{bel0[j]:.2f}" for j in range(5) if T[i, j] > 0 and bel0[j] > 0]
    print(f"  celda {i+1}: bel_pred = {' + '.join(terms) if terms else '0'} = {pred[i]:.3f}")
print("  bel_pred(x_k)         =", pred, " suma", pred.sum())
print("  p(y=puerta | x_k)     =", lik)
print("  producto              =", prod, " suma", round(prod.sum(), 4), " -> eta =", round(eta, 4))
print("  bel(x_k)              =", np.round(bel1, 3))
# si no hubiera visto puerta
lik_no = 1 - lik; prod_no = lik_no * pred
print("  (si NO viera puerta: producto", prod_no, "suma", round(prod_no.sum(), 3), "-> bel", np.round(prod_no / prod_no.sum(), 3), ")")
# coordenadas de las barras de la figura de la rejilla
print("  Barras de la figura (4 paneles; altura = valor · 150 px, base en y = 200):")
for nombre, v in (("anterior", bel0), ("predicho", pred), ("verosimilitud", lik), ("nuevo", bel1)):
    print(f"    {nombre:13s}: alturas {[round(150 * float(a), 1) for a in v]}")

# ---------------------------------------------------------------------------
titulo(2, "Coste de un filtro de Bayes en rejilla")
n1, w1 = 401, 161                             # el pasillo de 20 m (figuras_a4a.py): 401 celdas, núcleo de 161 celdas
print(f"  pasillo: n = {n1}; predicción densa n^2 = {n1**2:,}; convolución n·w = {n1*w1:,}; actualización n = {n1}")
celdas = 100 ** 12
print(f"  dron con 12 estados y 100 celdas por eje: n = 100^12 = {celdas:.0e} celdas;"
      f" en float64, {celdas*8:.0e} bytes; predicción densa n^2 = {celdas**2:.0e} operaciones")

# ---------------------------------------------------------------------------
titulo(3, "Un ciclo de Kalman en 1D (dron en estacionario, barómetro)")
m0, P0, F, Gu, Q = 10.0, 2.0, 1.0, 0.0, 2.0
y, R, H = 12.0, 1.0, 1.0
mp, Pp = F * m0 + Gu, F * P0 * F + Q
nu, S = y - H * mp, H * Pp * H + R
K = Pp * H / S
m1, P1 = mp + K * nu, (1 - K * H) * Pp
print(f"  predicción: m^- = {mp}, P^- = {Pp};  innovación nu = {nu}, S = {S};  K = {K};  m = {m1}, P = {P1}")
X = lin(4, 16, 50, 530); Y = lin(0, 0.5, 230, 20)
xx = np.linspace(4, 16, 241)
for nombre, (mm, PP) in (("bel_{k-1}", (m0, P0)), ("predicho", (mp, Pp)), ("verosimilitud", (y, R)), ("nuevo", (m1, P1))):
    print(f"  path {nombre}: max {gauss(mm, mm, PP):.3f}:", path(xx, gauss(xx, mm, PP), X, Y))
print(f"  marcas: x=10 -> {X(10):.1f}; x=12 -> {X(12):.1f}; x=11,6 -> {X(11.6):.1f}; base y = {Y(0):.1f}")

# ---------------------------------------------------------------------------
titulo(4, "Forma de Joseph frente a (I - KH)P^- con una ganancia no óptima")
Pp, R = 4.0, 1.0
for Kx in (0.8, 0.5, 1.0):
    simple = (1 - Kx) * Pp
    joseph = (1 - Kx) ** 2 * Pp + Kx ** 2 * R
    e_pre = rng.normal(0, math.sqrt(Pp), 400000); v = rng.normal(0, math.sqrt(R), 400000)
    e_post = (1 - Kx) * e_pre - Kx * v                      # e = x - m tras actualizar con ganancia Kx
    print(f"  K = {Kx}: (1-K)P^- = {simple:.2f};  Joseph = {joseph:.2f};  varianza real (Monte Carlo) = {e_post.var():.3f}")

# ---------------------------------------------------------------------------
titulo(5, "Riccati del filtro y del LQR en escalar (F = A = 1, H = B = 1, Q = R = 1)")
P = 4.0
seq = [P]
for _ in range(6):
    P = P - P ** 2 / (P + 1) + 1
    seq.append(P)
print("  P^-_k del filtro desde 4:", [round(p, 4) for p in seq])
print(f"  punto fijo: P^2 - P - 1 = 0 -> P = (1 + sqrt 5)/2 = {(1 + math.sqrt(5)) / 2:.4f};  K = P/(P+1) = {((1+math.sqrt(5))/2)/((1+math.sqrt(5))/2+1):.4f}")
Pi = 0.0
seqL = [Pi]
for _ in range(6):
    Pi = 1 + Pi - Pi ** 2 / (1 + Pi)
    seqL.append(Pi)
print("  Pi_k del LQR hacia atrás desde Pi_N = 0:", [round(p, 4) for p in seqL])

# ---------------------------------------------------------------------------
titulo(6, "Filtro complementario: ganancias, polinomio del error y respuesta en frecuencia")
for q, r in ((0.35 ** 2, 0.5 ** 2), (0.35 ** 2, 0.05 ** 2)):
    p12 = math.sqrt(q * r); p11 = math.sqrt(2 * r) * (q * r) ** 0.25; p22 = p11 * p12 / r
    k1, k2 = p11 / r, p12 / r; w0 = (q / r) ** 0.25
    A = np.array([[0, 1.0], [0, 0]]); L = np.array([[0.0], [1.0]]); C = np.array([[1.0, 0]])
    Pm = np.array([[p11, p12], [p12, p22]])
    res = A @ Pm + Pm @ A.T + q * L @ L.T - Pm @ C.T @ C @ Pm / r
    print(f"  q = {q:.4f}, r = {r:.4f}: p11 = {p11:.4f}, p12 = {p12:.4f}, p22 = {p22:.4f}; residuo Riccati {np.abs(res).max():.1e}")
    print(f"    K = ({k1:.3f}, {k2:.3f}); w0 = {w0:.3f} rad/s = {w0/2/math.pi:.3f} Hz; raíces de s^2 + k1 s + k2: {np.roots([1, k1, k2])}")
wn = np.logspace(math.log10(0.05), math.log10(20), 160)          # omega / omega_0
s = 1j * wn
Lp = (math.sqrt(2) * s + 1) / (s ** 2 + math.sqrt(2) * s + 1)   # posición medida -> estimación (paso bajo)
Hp = s ** 2 / (s ** 2 + math.sqrt(2) * s + 1)                   # aceleración integrada dos veces -> estimación (paso alto)
print("  suma de las dos funciones de transferencia (debe ser 1):", np.abs(Lp + Hp - 1).max())
cross = math.sqrt(1 + math.sqrt(2))
print(f"  |L| = |H| en omega/omega0 = sqrt(1 + sqrt 2) = {cross:.3f};  en omega0: |L| = {abs(Lp[np.argmin(abs(wn-1))]):.3f}, |H| = {1/math.sqrt(2):.3f}")
Xb = lin(math.log10(0.05), math.log10(20), 60, 500); Yb = lin(-55, 6, 250, 30)
db = lambda z: np.maximum(20 * np.log10(np.abs(z)), -55)
print("  path bode L:", path(np.log10(wn), db(Lp), Xb, Yb))
print("  path bode H:", path(np.log10(wn), db(Hp), Xb, Yb))
for v in (0.1, 1, 10):
    print(f"  marca omega/omega0 = {v}: x = {Xb(math.log10(v)):.1f}")
print(f"  marcas dB: 0 -> {Yb(0):.1f}; -20 -> {Yb(-20):.1f}; -40 -> {Yb(-40):.1f}; cruce x = {Xb(math.log10(cross)):.1f}")

# ---------------------------------------------------------------------------
titulo(7, "Q discreta desde un ruido blanco continuo (velocidad constante)")
q, dt = 0.5, 0.1
Qd = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
print("  Q_d =", Qd.tolist(), f" -> sigma de posición por paso {math.sqrt(Qd[0,0])*100:.2f} cm, de velocidad {math.sqrt(Qd[1,1]):.3f} m/s")
# comprobación: integrar con 1000 subpasos un ruido blanco de densidad q
nsub, Ntr = 1000, 20000
h = dt / nsub
acc = rng.normal(0, math.sqrt(q / h), (Ntr, nsub))
vel = np.cumsum(acc, axis=1) * h
pos = np.sum(vel, axis=1) * h
print("  Monte Carlo (20 000 trayectorias, 1000 subpasos):", np.cov(np.vstack([pos, vel[:, -1]])).round(6).tolist())
sg, dtg = 0.005, 0.005
print(f"  giróscopo de figuras_a4a.py: densidad {sg} rad/s/sqrt(Hz) a {1/dtg:.0f} Hz -> sigma por muestra {sg/math.sqrt(dtg):.4f} rad/s;"
      f" varianza del ángulo por paso sg^2·dt = {sg**2*dtg:.2e} rad^2")

# ---------------------------------------------------------------------------
titulo(8, "Forma de información en escalar: el barómetro de M1 y un segundo sensor")
mp, Pp = 10.0, 4.0
L0, e0 = 1 / Pp, mp / Pp
L1, e1 = L0 + 1 / 1.0, e0 + 12.0 / 1.0
print(f"  Lambda^- = {L0}, eta^- = {e0};  tras y = 12 (R = 1): Lambda = {L1}, eta = {e1} -> P = {1/L1:.3f}, m = {e1/L1:.3f}")
L2, e2 = L1 + 1.0, e1 + 11.0
print(f"  más un segundo sensor y2 = 11 (R2 = 1): Lambda = {L2}, eta = {e2} -> P = {1/L2:.4f}, m = {e2/L2:.4f}")
L2b, e2b = L0 + 1.0 + 1.0, e0 + 11.0 + 12.0
print(f"  en el otro orden: Lambda = {L2b}, eta = {e2b} -> m = {e2b/L2b:.4f}")

# ---------------------------------------------------------------------------
titulo(9, "NEES a mano y por qué su media vale n")
e = np.array([0.3, -0.2]); Pd = np.diag([0.04, 0.01])
print(f"  e = {e}, P = diag(0,04; 0,01): NEES = 0,09/0,04 + 0,04/0,01 = {e @ np.linalg.solve(Pd, e):.2f}")
Pc = np.array([[0.04, 0.012], [0.012, 0.01]])
Lc = np.linalg.cholesky(Pc)
E = rng.standard_normal((200000, 2)) @ Lc.T
nees = np.einsum("ij,ij->i", E, np.linalg.solve(Pc, E.T).T)
print(f"  200 000 errores con P = [[0,04 0,012];[0,012 0,01]]: NEES medio {nees.mean():.3f}, varianza {nees.var():.3f} (chi2_2: media 2, varianza 4)")
lo, hi = chi2_ppf(0.025, 400) / 200, chi2_ppf(0.975, 400) / 200
print(f"  media de 200 ensayos con n = 2: chi2_400 / 200 -> intervalo del 95 % {lo:.3f}–{hi:.3f}")
print(f"  un ensayo suelto (chi2_2): intervalo del 95 % {chi2_ppf(0.025, 2):.3f}–{chi2_ppf(0.975, 2):.3f}")

# ---------------------------------------------------------------------------
titulo(10, "EKF y UKF frente a la verdad: y = x^2 con x ~ N(2, 1)")
m, P = 2.0, 1.0
true_mean, true_var = m ** 2 + P, 4 * m ** 2 * P + 2 * P ** 2
xs = rng.normal(m, math.sqrt(P), 2_000_000); ys = xs ** 2
print(f"  verdad (analítica): media {true_mean}, varianza {true_var};  Monte Carlo: {ys.mean():.3f}, {ys.var():.3f}")
print(f"  EKF: media g(m) = {m**2}, varianza g'(m)^2 P = {(2*m)**2*P}")
n, alpha, beta, kappa = 1, 1.0, 0.0, 2.0
lam = alpha ** 2 * (n + kappa) - n
sp = np.array([m, m + math.sqrt((n + lam) * P), m - math.sqrt((n + lam) * P)])
Wm = np.array([lam / (n + lam), 1 / (2 * (n + lam)), 1 / (2 * (n + lam))]); Wc = Wm.copy(); Wc[0] += 1 - alpha ** 2 + beta
gy = sp ** 2
um = Wm @ gy; uv = Wc @ (gy - um) ** 2
print(f"  UKF: lambda = {lam}; puntos {sp.round(4)}; pesos {Wm.round(4)}; g(puntos) = {gy.round(4)}")
print(f"       media {um:.4f}, varianza {uv:.4f};  los puntos reproducen m = {Wm@sp:.4f} y P = {Wc@(sp-m)**2:.4f}")
# figura: curva, tangente, puntos sigma y densidades de salida
Xf = lin(-0.5, 4.5, 60, 360); Yf = lin(0, 22, 280, 20)
xc = np.linspace(-0.5, 4.5, 101)
print("  path curva:", path(xc, xc ** 2, Xf, Yf))
xt = np.array([1.0, 4.4])
print("  path tangente:", path(xt, 4 + 4 * (xt - 2), Xf, Yf))
print("  puntos sigma (px):", [(round(float(Xf(a)), 1), round(float(Yf(a * a)), 1)) for a in sp])
xin = np.linspace(-0.5, 4.5, 101)
print("  path entrada N(2,1) (base y = 350, 45 px de alto):", path(xin, gauss(xin, 2, 1) / gauss(2, 2, 1), Xf, lambda d: 350 - 45 * d))
yy = np.linspace(0.2, 22, 220)
true_pdf = (gauss(np.sqrt(yy), 2, 1) + gauss(-np.sqrt(yy), 2, 1)) / (2 * np.sqrt(yy))
Xd = lambda d: 380 + d * 900                     # densidad -> desplazamiento horizontal desde x = 380
print(f"  densidad verdadera máx. {true_pdf.max():.4f} (en y = {yy[np.argmax(true_pdf)]:.2f})")
print("  path densidad verdadera:", "M" + " L".join(f"{Xd(d):.1f},{Yf(v):.1f}" for v, d in zip(yy, true_pdf)))
yg = np.linspace(0, 22, 200)
print("  path densidad EKF N(4,16):", "M" + " L".join(f"{Xd(d):.1f},{Yf(v):.1f}" for v, d in zip(yg, gauss(yg, 4, 16))))
print("  path densidad UKF N(5,18):", "M" + " L".join(f"{Xd(d):.1f},{Yf(v):.1f}" for v, d in zip(yg, gauss(yg, um, uv))))
print(f"  marcas y: 0 -> {Yf(0):.1f}; 4 -> {Yf(4):.1f}; 5 -> {Yf(5):.1f}; 10 -> {Yf(10):.1f}; 20 -> {Yf(20):.1f};"
      f" x: 0 -> {Xf(0):.1f}; 2 -> {Xf(2):.1f}; 4 -> {Xf(4):.1f}")

# ---------------------------------------------------------------------------
titulo(11, "EKF en el modelo bimodal: el jacobiano de la medida")
for mm in (0.1, 1.0, 10.0):
    print(f"  h(x) = x^2/20 -> H = x/10; en m^- = {mm}: H = {mm/10}")

# ---------------------------------------------------------------------------
titulo(12, "Filtro de partículas a mano: 4 partículas, pesos, tamaño efectivo y remuestreo sistemático")
xp = np.array([-1.0, 0.0, 1.0, 2.0]); yv, Rv = 1.0, 0.5
lw = np.exp(-(yv - xp) ** 2 / (2 * Rv))
w = lw / lw.sum()
print(f"  verosimilitudes sin normalizar exp(-(y-x)^2/(2R)) = {lw.round(4)}, suma {lw.sum():.4f}")
print(f"  pesos = {w.round(4)};  media ponderada = {w @ xp:.3f};  tamaño efectivo 1/sum w^2 = {1/np.sum(w**2):.2f}")
cs = np.cumsum(w); r0 = 0.5
u = (r0 + np.arange(4)) / 4
idx = np.searchsorted(cs, u)
print(f"  acumulada {cs.round(4)};  r = {r0} -> u = {u};  índices elegidos {idx + 1};  nuevas partículas {xp[idx]}")
for ww in ([0.25] * 4, [1, 0, 0, 0], [0.7, 0.1, 0.1, 0.1]):
    ww = np.array(ww, float)
    print(f"  pesos {ww}: tamaño efectivo {1/np.sum(ww**2):.2f}")

# ---------------------------------------------------------------------------
titulo(13, "Figura del ciclo del filtro de partículas: 20 partículas, medida y = x^2/20 + v")
rngp = np.random.default_rng(18)
xpart = np.sort(rngp.normal(0, 8, 20))
yv = 5.0
wl = np.exp(-0.5 * (yv - xpart ** 2 / 20) ** 2 / 1.0); wp = wl / wl.sum()
cs = np.cumsum(wp); idx = np.minimum(np.searchsorted(cs, (rngp.random() + np.arange(20)) / 20), 19)
nuevas = xpart[idx]
print("  partículas predichas:", xpart.round(2))
print("  pesos:", wp.round(3), f" tamaño efectivo {1/np.sum(wp**2):.2f}")
vals, cnt = np.unique(nuevas.round(2), return_counts=True)
print("  tras remuestrear (valor: copias):", dict(zip(vals.tolist(), cnt.tolist())))
Xp = lin(-25, 25, 40, 520)
print("  x px de las partículas:", [round(float(Xp(a)), 1) for a in xpart])
print("  altura de los pesos (px, 90 por peso máximo):", [round(float(90 * a / wp.max()), 1) for a in wp])
print(f"  x = ±10 -> {Xp(-10):.1f}, {Xp(10):.1f}; x = 0 -> {Xp(0):.1f}")
xl = np.linspace(-25, 25, 251)
print("  path verosimilitud PF (base y = 170, 90 px de alto):", path(xl, np.exp(-0.5 * (yv - xl ** 2 / 20) ** 2), Xp, lambda d: 170 - 90 * d))
print("  marcas x:", {v: round(float(Xp(v)), 1) for v in (-20, -10, 0, 10, 20)})
neg = int(np.sum(nuevas < 0)); print(f"  tras remuestrear: {neg} partículas con x < 0 y {20 - neg} con x > 0")

# ---------------------------------------------------------------------------
titulo(14, "ESKF: jacobiano de una medida vectorial y lo que ve")
g = np.array([0, 0, 9.81]); Rh = np.eye(3)
Hth = hat(Rh.T @ g)
print("  dron nivelado, v = g = (0, 0, 9,81): H = [R^T g]_x =\n", Hth)
for d in (np.array([0.01, 0, 0]), np.array([0, 0.01, 0]), np.array([0, 0, 0.01])):
    exact = (Rh @ Exp(d)).T @ g - Rh.T @ g
    print(f"  dtheta = {d}: H dtheta = {Hth @ d};  exacto R^T g - R^_T g = {exact.round(6)}")
Rh2 = Exp(np.radians([20.0, -10.0, 35.0]))
d = np.array([0.02, -0.01, 0.015])
exact = (Rh2 @ Exp(d)).T @ g - Rh2.T @ g
print(f"  comprobación con R^ cualquiera y dtheta = {d}: lineal {hat(Rh2.T @ g) @ d}, exacto {exact}")

# ---------------------------------------------------------------------------
titulo(15, "ESKF: la fila de la velocidad con números (error de inclinación y de guiñada)")
for a, dth, txt in ((np.array([0, 0, 9.81]), np.array([0.01, 0, 0]), "estacionario, error de alabeo 0,01 rad"),
                    (np.array([0, 0, 9.81]), np.array([0, 0, 0.01]), "estacionario, error de guiñada 0,01 rad"),
                    (np.array([1.0, 0, 9.81]), np.array([0, 0, 0.01]), "acelerando 1 m/s² hacia delante, error de guiñada 0,01 rad")):
    dv = -np.eye(3) @ hat(a) @ dth
    print(f"  {txt}: delta v' = -R^[a]x dtheta = {dv.round(4)} m/s^2;"
          f" en 10 s: error de velocidad {np.linalg.norm(dv)*10:.3f} m/s, de posición {0.5*np.linalg.norm(dv)*100:.3f} m")
print(f"  ángulo de 0,01 rad = {math.degrees(0.01):.2f}°")
# comprobación numérica de la fila de la actitud: d(dtheta)/dt = -[w]x dtheta - dbg
w_true = np.array([0.3, -0.2, 0.5]); dbg = np.array([0.001, -0.002, 0.0015])
Rt = np.eye(3)
dth0 = np.array([0.01, -0.005, -0.008])
Rn = Rt @ Exp(-dth0)                                              # nominal: R^ = R Exp(-dtheta)
h_ = 1e-4; Tt = 2.0
dlin = dth0.copy()
for _ in range(int(Tt / h_)):
    Rt = Rt @ Exp(w_true * h_)
    w_nom = w_true + dbg                                           # el nominal usa omega_m - b^_g = omega + dbg
    Rn = Rn @ Exp(w_nom * h_)
    dlin = dlin + h_ * (-hat(w_nom) @ dlin - dbg)
def Log(Rm):
    c = max(-1.0, min(1.0, (np.trace(Rm) - 1) / 2)); ang = math.acos(c)
    ww = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    return ww / 2 if ang < 1e-9 else ang / (2 * math.sin(ang)) * ww
print(f"  fila de la actitud, 2 s con error de sesgo {dbg}: dtheta exacto = Log(R^T R) = {Log(Rn.T @ Rt).round(5)}; modelo lineal = {dlin.round(5)}")

# ---------------------------------------------------------------------------
titulo(16, "Prueba de innovación a mano: GNSS en 3D")
nu = np.array([1.0, -0.5, 2.0]); S = np.diag([0.25, 0.25, 1.0])
d2 = nu @ np.linalg.solve(S, nu)
print(f"  nu = {nu}, S = diag(0,25; 0,25; 1): d^2 = 1/0,25 + 0,25/0,25 + 4/1 = {d2:.2f};"
      f" umbral 95 % {chi2_ppf(0.95, 3):.2f}, 99 % {chi2_ppf(0.99, 3):.2f}")
print(f"  la misma innovación con S cuatro veces menor (filtro sobreconfiado): d^2 = {4*d2:.1f} -> rechazada incluso al 99,9 % ({chi2_ppf(0.999, 3):.2f})")

# ---------------------------------------------------------------------------
titulo(17, "Suavizador RTS a mano en escalar")
F, Q = 1.0, 1.0
Pk, mk = 1.0, 0.0
Ppred, mpred = F * Pk * F + Q, F * mk
ms_next, Ps_next = 1.0, 0.8
G = Pk * F / Ppred
ms = mk + G * (ms_next - mpred); Ps = Pk + G * (Ps_next - Ppred) * G
print(f"  P_k = {Pk}, m_k = {mk}; P^-_(k+1) = {Ppred}, m^-_(k+1) = {mpred}; G = {G};"
      f" con m^s_(k+1) = {ms_next}, P^s_(k+1) = {Ps_next}: m^s_k = {ms}, P^s_k = {Ps}")

# ---------------------------------------------------------------------------
titulo(18, "Medidas retrasadas: error por fundir como si fuera de ahora")
for v, dl in ((5, 0.1), (5, 0.2), (15, 0.1)):
    print(f"  {v} m/s y {dl*1000:.0f} ms de retardo: {v*dl:.2f} m")

# ---------------------------------------------------------------------------
titulo(19, "Ciclo del ESKF: dientes de sierra de la varianza (IMU a 200 Hz, medidas a 50 Hz)")
P, Qs, Rs = 0.5, 0.05, 0.2
traza = []
for i in range(13):
    if i > 0:
        P = P + Qs
    traza.append(round(P, 3))
    if i % 4 == 0:
        P = P * Rs / (P + Rs)
        traza.append(round(P, 3))
print("  varianza (antes y después de cada actualización en los pasos 0, 4, 8, 12):", traza)
