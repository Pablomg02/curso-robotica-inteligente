"""Ejemplos numéricos y datos de las figuras nuevas de A3b (numpy, CPU, segundos).

Uso:  python3 ejemplos_a3b.py > salida_ejemplos.txt

Cada ejemplo lleva un número que se cita en las hojas («ejemplos_a3b.py, ejemplo N»).
Las cifras del escenario 2D con obstáculos (éxito, tiempos, tamaño efectivo frente a λ)
salen de figuras_a3b.py, que no se relanza aquí.
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def cab(n, titulo):
    print("=" * 72)
    print(f"EJEMPLO {n}. {titulo}")


# ---------------------------------------------------------------------------
# Escenario 2D del tema (copia exacta del coste de figuras_a3b.py)
DT, H2, AMAX = 0.05, 30, 4.0
GOAL = np.array([8.0, 0.0])
OBS = [(np.array([3.0, 0.2]), 1.0), (np.array([5.5, -0.6]), 0.8)]


def rollout_cost(x0, U):
    K = U.shape[0]
    p = np.repeat(x0[None, :2], K, 0); v = np.repeat(x0[None, 2:], K, 0)
    S = np.zeros(K); parts = np.zeros((K, 4))   # etapa, mando, choque, terminal
    for k in range(H2):
        a = np.clip(U[:, k], -AMAX, AMAX)
        v = v + DT * a; p = p + DT * v
        e = DT * np.sum((p - GOAL) ** 2, 1); m = DT * 0.01 * np.sum(a**2, 1)
        c = sum(1e3 * (np.linalg.norm(p - cc, axis=1) < r + 0.15) for cc, r in OBS)
        S += e + m + c; parts[:, 0] += e; parts[:, 1] += m; parts[:, 2] += c
    t = 10 * np.sum((p - GOAL) ** 2, 1) + 1.0 * np.sum(v**2, 1)
    S += t; parts[:, 3] += t
    return S, parts


cab(1, "COSTE DE UNA SECUENCIA EN EL ESCENARIO 2D")
U0 = np.zeros((1, H2, 2))
S, pr = rollout_cost(np.zeros(4), U0)
print(f"  U = 0 (quedarse quieto): etapa {pr[0,0]:.1f} + mando {pr[0,1]:.1f} + choque {pr[0,2]:.0f}"
      f" + terminal {pr[0,3]:.1f} = S = {S[0]:.1f}")
U1 = np.zeros((1, H2, 2)); U1[0, :, 0] = 4.0   # acelerar a tope hacia la meta
S, pr = rollout_cost(np.zeros(4), U1)
print(f"  a = (4, 0) todo el horizonte: etapa {pr[0,0]:.1f} + mando {pr[0,1]:.2f} + choque {pr[0,2]:.0f}"
      f" + terminal {pr[0,3]:.1f} = S = {S[0]:.1f}")
p_final = 0.5 * 4.0 * (H2 * DT) ** 2
print(f"    (posición final aproximada x = {DT*DT*4*sum(H2-t for t in range(H2)):.2f} m; a/2 t^2 = {p_final:.2f} m)")

# ---------------------------------------------------------------------------
# Problema de juguete para hacer a mano: x_{t+1} = x_t + u_t, x_0 = 0, meta 2, H = 2
# S(U) = (x_2 - 2)^2 + 0,5 (u_0^2 + u_1^2); secuencia nominal U = (0, 0)
def S_toy(U):
    U = np.atleast_2d(U)
    x2 = U[:, 0] + U[:, 1]
    return (x2 - 2) ** 2 + 0.5 * (U[:, 0] ** 2 + U[:, 1] ** 2)


E = np.array([[1, 1], [2, 0], [1, 0], [0, 0], [-1, 1]], float)
Sk = S_toy(E)
cab(2, "EJEMPLO A MANO: CINCO SECUENCIAS")
for k, (e, s) in enumerate(zip(E, Sk), 1):
    x1, x2 = e[0], e[0] + e[1]
    print(f"  k = {k}: E = ({e[0]:+.0f}, {e[1]:+.0f}), x = (0, {x1:.0f}, {x2:.0f}), "
          f"(x2-2)^2 = {(x2-2)**2:.1f}, 0,5|u|^2 = {0.5*(e@e):.1f}, S = {s:.2f}")
# óptimo exacto: gradiente = 0  ->  [[3,2],[2,3]] U = (4,4)
Hs = np.array([[3.0, 2.0], [2.0, 3.0]])
Uopt = np.linalg.solve(Hs, [4.0, 4.0])
print(f"  óptimo exacto U* = {Uopt}, S(U*) = {S_toy(Uopt)[0]:.3f}")
print("  hessiana de S: [[3,2],[2,3]], autovalores", np.linalg.eigvalsh(Hs))

cab(3, "DISPARO ALEATORIO EN EL EJEMPLO")
kb = int(np.argmin(Sk))
print(f"  mejor muestra k = {kb+1}, U <- U + E^{kb+1} = {E[kb]}, S = {Sk[kb]:.2f}")

cab(4, "CEM EN EL EJEMPLO (n = 5, n_e = 2 élites)")
orden = np.argsort(Sk)
el = E[orden[:2]]
mu, sd = el.mean(0), el.std(0)
print(f"  élites: k = {orden[0]+1} y {orden[1]+1}: {el.tolist()}")
print(f"  media nueva mu = {mu}, desviación nueva sigma = {sd} (componente 0 colapsada)")
print(f"  S(mu) = {S_toy(mu)[0]:.3f}")
print(f"  con el mínimo de varianza del script (+0,05): sigma = {sd + 0.05}")
# segunda iteración ilustrativa: el componente 0 ya no se puede mover
rng = np.random.default_rng(3)
U2 = mu + (sd + 0.05) * rng.normal(0, 1, (5, 2))
S2 = S_toy(U2)
print("  2.ª iteración (5 muestras, semilla 3):")
for u, s in zip(U2, S2):
    print(f"    U = ({u[0]:.3f}, {u[1]:.3f}), S = {s:.3f}")
el2 = U2[np.argsort(S2)[:2]]
print(f"    media de las élites = {el2.mean(0)}, S = {S_toy(el2.mean(0))[0]:.3f}")

cab(5, "MPPI EN EL EJEMPLO (lambda = 1)")
lam = 1.0
rho = Sk.min()
a = -(Sk - rho) / lam
ex = np.exp(a)
w = ex / ex.sum()
for k in range(5):
    print(f"  k = {k+1}: S = {Sk[k]:.2f}, S - rho = {Sk[k]-rho:.2f}, exp = {ex[k]:.4f}, w = {w[k]:.4f}, "
          f"w E = ({w[k]*E[k,0]:+.4f}, {w[k]*E[k,1]:+.4f})")
print(f"  suma de exponenciales = {ex.sum():.4f}")
dU = w @ E
print(f"  U nuevo = U + sum w E = {dU}, x_2 = {dU.sum():.3f}, S = {S_toy(dU)[0]:.3f}")
ess = 1 / np.sum(w**2)
print(f"  sum w^2 = {np.sum(w**2):.4f}, ESS = 1/sum w^2 = {ess:.2f} de 5")
print(f"  sin restar rho: exp(-S/lambda) = {np.exp(-Sk/lam)}, pesos = {np.exp(-Sk/lam)/np.exp(-Sk/lam).sum()}")

cab(6, "TEMPERATURA EN EL EJEMPLO")
for lam in (0.1, 0.3, 1.0, 3.0, 10.0, 1e3):
    ex = np.exp(-(Sk - rho) / lam); w = ex / ex.sum()
    dU = w @ E
    print(f"  lambda = {lam:7.1f}: w = {np.round(w,3)}, ESS = {1/np.sum(w**2):.2f}, "
          f"U = ({dU[0]:.3f}, {dU[1]:.3f}), S(U) = {S_toy(dU)[0]:.3f}")
print(f"  media sin pesar de E = {E.mean(0)}, S = {S_toy(E.mean(0))[0]:.3f}")
# misma escala relativa: multiplicar S por 100 y lambda por 100 da los mismos pesos
ex1 = np.exp(-(Sk - rho) / 1.0); ex100 = np.exp(-(100 * Sk - 100 * rho) / 100.0)
print(f"  S x100 y lambda x100: pesos iguales -> {np.allclose(ex1/ex1.sum(), ex100/ex100.sum())}")

# Figura del plano (u0, u1): curvas de nivel, muestras, óptimo y actualizaciones
cab(7, "FIGURA: PLANO (u0, u1) DEL EJEMPLO")
Xp = lin(-1.4, 2.8, 50, 350)
Yp = lin(-1.2, 2.8, 270, 20)
evals, evecs = np.linalg.eigh(Hs)
for nivel in (1.0, 2.0, 4.0):
    c = nivel - S_toy(Uopt)[0]
    th = np.linspace(0, 2 * np.pi, 73)
    pts = np.array([Uopt + math.sqrt(2 * c / evals[0]) * math.cos(t) * evecs[:, 0]
                    + math.sqrt(2 * c / evals[1]) * math.sin(t) * evecs[:, 1] for t in th])
    print(f"path nivel S={nivel}:", path(pts[:, 0], pts[:, 1], Xp, Yp) + " Z")
for k, e in enumerate(E, 1):
    print(f"  muestra {k}: ({Xp(e[0]):.1f}, {Yp(e[1]):.1f})")
print(f"  óptimo: ({Xp(Uopt[0]):.1f}, {Yp(Uopt[1]):.1f})")
print(f"  nominal (0,0): ({Xp(0):.1f}, {Yp(0):.1f})")
w1 = np.exp(-(Sk - rho)); w1 /= w1.sum(); m1 = w1 @ E
print(f"  MPPI lambda=1: ({Xp(m1[0]):.1f}, {Yp(m1[1]):.1f})")
print(f"  CEM media élites: ({Xp(mu[0]):.1f}, {Yp(mu[1]):.1f})")
print(f"  media sin pesar: ({Xp(E.mean(0)[0]):.1f}, {Yp(E.mean(0)[1]):.1f})")
for v in (-1, 0, 1, 2):
    print(f"  marca u0={v}: x={Xp(v):.1f}; u1={v}: y={Yp(v):.1f}")

# Figura de las cinco trayectorias x_t
cab(8, "FIGURA: TRAYECTORIAS x_t DE LAS CINCO MUESTRAS")
Xt = lin(0, 2, 60, 300)
Yt = lin(-1.2, 2.4, 220, 20)
for k, e in enumerate(E, 1):
    xs = [0, e[0], e[0] + e[1]]
    print(f"path tray {k}:", path([0, 1, 2], xs, Xt, Yt))
for v in (-1, 0, 1, 2):
    print(f"  marca x={v}: y={Yt(v):.1f}")

# ---------------------------------------------------------------------------
cab(9, "DISTRIBUCIÓN ÓPTIMA COMO PRIOR POR VEROSIMILITUD (1 paso)")
# x_1 = u, S(u) = (u - 2)^2 + 0,5 u^2; prior p = N(0, 1) (dinámica sin controlar, Sigma = 1)
# q*(u) ∝ exp(-u^2/2) exp(-S(u)/lambda): gaussiana de precisión 1 + 3/lambda y media 4/(lambda+3)
uu = np.linspace(-2.5, 3.5, 121)
Su = (uu - 2) ** 2 + 0.5 * uu**2
print(f"  mínimo de S: u = {4/3:.4f}, S = {(4/3-2)**2+0.5*(4/3)**2:.4f}")
for lam in (0.3, 1.0, 3.0, 10.0):
    m = 4 / (lam + 3); s = 1 / math.sqrt(1 + 3 / lam)
    # comprobación numérica
    qq = np.exp(-uu**2 / 2 - Su / lam); qq /= np.trapezoid(qq, uu)
    mn = np.trapezoid(uu * qq, uu)
    print(f"  lambda = {lam:5.1f}: q* = N({m:.3f}, {s:.3f}^2)   (media numérica {mn:.3f})")
Xq = lin(-2.5, 3.5, 40, 500)
Yq = lin(0, 1.0, 230, 30)
prior = np.exp(-uu**2 / 2)
lik = np.exp(-(Su - Su.min()) / 1.0)
post = np.exp(-uu**2 / 2 - Su / 1.0); post /= post.max()
print("path prior:", path(uu, prior, Xq, Yq))
print("path verosimilitud lambda=1:", path(uu, lik, Xq, Yq))
print("path posterior lambda=1:", path(uu, post, Xq, Yq))
post10 = np.exp(-uu**2 / 2 - Su / 10.0); post10 /= post10.max()
print("path posterior lambda=10:", path(uu, post10, Xq, Yq))
for v in (-2, -1, 0, 1, 2, 3):
    print(f"  marca u={v}: x={Xq(v):.1f}")
print(f"  u=4/3: x={Xq(4/3):.1f}; u=1: x={Xq(1):.1f}; u=0,308: x={Xq(4/13):.1f}")

# ---------------------------------------------------------------------------
cab(10, "TAMAÑO EFECTIVO DE MUESTRA: VARIANZA DE UNA MEDIA PONDERADA")
rng = np.random.default_rng(0)
lam = 1.0
w = np.exp(-(Sk - rho) / lam); w /= w.sum()
# Monte Carlo: varianza de sum w_k z_k con z_k iid N(0,1) y pesos fijos
z = rng.normal(0, 1, (200000, 5))
var_mc = np.var(z @ w)
print(f"  pesos lambda=1: sum w^2 = {np.sum(w**2):.4f}; varianza Monte Carlo de sum w z = {var_mc:.4f}")
print(f"  ESS = {1/np.sum(w**2):.2f}: misma varianza que la media de {1/np.sum(w**2):.2f} muestras iguales")
for nombre, ww in (("iguales", np.full(5, 0.2)), ("una sola", np.array([1, 0, 0, 0, 0.0]))):
    print(f"  pesos {nombre}: ESS = {1/np.sum(ww**2):.2f}")

# ---------------------------------------------------------------------------
cab(11, "RUIDO BLANCO FRENTE A RUIDO CORRELADO (H = 30, Delta t = 0,05 s, sigma = 2 m/s^2)")


def ruido_color(beta, n, H, rng):
    f = np.fft.rfftfreq(H)
    f[0] = f[1]                                   # evitar dividir por cero en f = 0
    esc = f ** (-beta / 2)
    X = (rng.normal(size=(n, len(f))) + 1j * rng.normal(size=(n, len(f)))) / math.sqrt(2)
    X[:, 0] = rng.normal(size=n)                  # f = 0 y Nyquist son reales: la varianza entera
    if H % 2 == 0:                                # en la parte real (si no, el ruido «blanco»
        X[:, -1] = rng.normal(size=n)             # tendría menos potencia en f = 0)
    X = X * esc
    x = np.fft.irfft(X, n=H, axis=1)
    return x / x.std()                            # desviación 1 por paso (en el conjunto)


SIG = 2.0
n = 20000
pesos = DT * DT * np.array([H2 - t for t in range(H2)])   # p_H = Dt^2 sum (H - t) a_t
res = {}
for beta in (0.0, 1.0, 2.0):
    rng = np.random.default_rng(11)
    A = SIG * ruido_color(beta, n, H2, rng)
    pH = A @ pesos
    vH = DT * A.sum(1)
    # correlación entre pasos consecutivos
    rho1 = np.mean(A[:, 1:] * A[:, :-1]) / np.mean(A**2)
    res[beta] = A
    print(f"  beta = {beta:.0f}: desv. por paso {A.std():.2f} m/s^2, correlación con el paso siguiente {rho1:.2f}, "
          f"desv. de la posición final {pH.std():.2f} m, de la velocidad final {vH.std():.2f} m/s")
print(f"  ruido blanco, fórmula: sigma Dt^2 sqrt(sum (H-t)^2) = {SIG*DT*DT*math.sqrt(sum((H2-t)**2 for t in range(H2))):.3f} m")
Xn = lin(0, H2 - 1, 50, 500)
Yn = lin(-6, 6, 220, 20)
for beta in (0.0, 2.0):
    print(f"path ruido beta={beta:.0f}:", path(range(H2), np.clip(res[beta][3], -6, 6), Xn, Yn))
print(f"  marca 0: y={Yn(0):.1f}; +4: y={Yn(4):.1f}; -4: y={Yn(-4):.1f}")

# ---------------------------------------------------------------------------
cab(12, "MODELO APRENDIDO: CONJUNTO DE 5 REDES DE JUGUETE Y EL PLANIFICADOR QUE LAS EXPLOTA")
# Coste real de un mando escalar u: S(u) = (u - 1)^2 + 1. Datos con ruido solo en u en [-1, 2,5].
# Cada «red»: una capa oculta de 30 tanh con pesos aleatorios y la última capa por mínimos cuadrados
# (ridge), entrenada con un remuestreo (bootstrap) de los datos. Su varianza aleatoria: el residuo medio.


def S_real(u):
    return (u - 1) ** 2 + 1


def entrena(u, y, rng, nh=30):
    W = rng.normal(0, 1.5, nh); b = rng.normal(0, 1.5, nh)
    Phi = np.tanh(np.outer(u, W) + b)
    Phi = np.hstack([Phi, np.ones((len(u), 1))])
    c = np.linalg.solve(Phi.T @ Phi + 1e-3 * np.eye(nh + 1), Phi.T @ y)
    var = np.mean((Phi @ c - y) ** 2)
    return (lambda q: np.hstack([np.tanh(np.outer(np.atleast_1d(q), W) + b),
                                 np.ones((np.size(q), 1))]) @ c), var


ug = np.linspace(-3, 4.5, 301)
resumen = []
sueltas = []
for semilla in range(10):
    rng = np.random.default_rng(semilla)
    ud = rng.uniform(-1, 2.5, 40)
    yd = S_real(ud) + rng.normal(0, 0.2, 40)
    modelos = []
    for b_ in range(5):
        idx = rng.integers(0, 40, 40)
        modelos.append(entrena(ud[idx], yd[idx], rng))
    M = np.array([m(ug) for m, _ in modelos])
    media, desv = M.mean(0), M.std(0)
    u_media = ug[np.argmin(media)]
    u_pen = ug[np.argmin(media + 2 * desv)]
    resumen.append((semilla, u_media, u_pen, ud, yd, modelos, M))
    for m_ in M:
        sueltas.append(not (-1 <= ug[np.argmin(m_)] <= 2.5))
    print(f"  semilla {semilla}: argmin de la media del conjunto u = {u_media:+.2f} (S real {S_real(u_media):.2f}); "
          f"con +2 desv.: u = {u_pen:+.2f} (S real {S_real(u_pen):.2f})")
fuera = sum(1 for r in resumen if not (-1 <= r[1] <= 2.5))
print(f"  en {fuera} de 10 semillas el mínimo de la media cae fuera de los datos [-1; 2,5]")
print(f"  planificando con una sola red: el mínimo cae fuera de los datos en {sum(sueltas)} de {len(sueltas)} redes")
pen_fuera = sum(1 for r in resumen if not (-1 <= r[2] <= 2.5))
print(f"  con media + 2 desv.: fuera de los datos en {pen_fuera} de 10 semillas")
# semilla para la figura: la primera en la que el mínimo cae fuera
sel = next(r for r in resumen if not (-1 <= r[1] <= 2.5))
semilla, u_media, u_pen, ud, yd, modelos, M = sel
media, desv = M.mean(0), M.std(0)
print(f"  FIGURA con semilla {semilla}: el planificador elige u = {u_media:.2f}, coste predicho "
      f"{media[np.argmin(media)]:.2f}, coste real {S_real(u_media):.2f}; óptimo real u = 1, S = 1")
print(f"    con penalización media + 2 desv.: u = {u_pen:.2f}, coste real {S_real(u_pen):.2f}")
Xm = lin(-3, 4.5, 50, 500)
Ym = lin(-2, 12, 240, 20)
cl = lambda a: np.clip(a, -2, 12)
print("path real:", path(ug, cl(S_real(ug)), Xm, Ym))
print("path media:", path(ug, cl(media), Xm, Ym))
print("path banda sup:", path(ug, cl(media + 2 * desv), Xm, Ym))
print("path banda inf:", path(ug[::-1], cl((media - 2 * desv)[::-1]), Xm, Ym))
print("  datos:", " ".join(f"({Xm(a):.1f},{Ym(b):.1f})" for a, b in zip(ud, yd)))
print(f"  elegido: x = {Xm(u_media):.1f}, y_pred = {Ym(cl(media[np.argmin(media)])):.1f}, y_real = {Ym(cl(S_real(u_media))):.1f}")
print(f"  penalizado: x = {Xm(u_pen):.1f}, y_real = {Ym(S_real(u_pen)):.1f}")
print(f"  límites de los datos: x = {Xm(-1):.1f} y {Xm(2.5):.1f}")
for v in (-2, 0, 2, 4):
    print(f"  marca u={v}: x={Xm(v):.1f}")
for v in (0, 5, 10):
    print(f"  marca S={v}: y={Ym(v):.1f}")

cab(13, "INCERTIDUMBRE ALEATORIA Y EPISTÉMICA (ley de la varianza total)")
for uq in (0.5, u_media):
    mus = np.array([m(np.array([uq]))[0] for m, _ in modelos])
    vars_ = np.array([v for _, v in modelos])
    ale = vars_.mean(); epi = mus.var(); tot = ale + epi
    print(f"  u = {uq:+.2f}: medias de las 5 redes {np.round(mus,2)}, varianzas {np.round(vars_,3)}")
    print(f"     aleatoria (media de las varianzas) = {ale:.3f}, epistémica (varianza de las medias) = {epi:.3f}, "
          f"total = {tot:.3f}")
    # comprobación por Monte Carlo: mezcla de las 5 gaussianas
    rng = np.random.default_rng(5)
    bsel = rng.integers(0, 5, 400000)
    y = mus[bsel] + np.sqrt(vars_[bsel]) * rng.normal(size=400000)
    print(f"     varianza de la mezcla por Monte Carlo = {y.var():.3f}")

# ---------------------------------------------------------------------------
cab(14, "MPC DIFERENCIABLE: TEOREMA DE LA FUNCIÓN IMPLÍCITA EN UN MPC DE UN PASO")
# x_1 = x_0 + u, J(u; r) = x_1^2 + r u^2. Parámetro aprendible: el peso r.
x0, r = 1.0, 1.0
u_st = -x0 / (1 + r)
dK_du = 2 + 2 * r
dK_dr = 2 * u_st
du_dr = -dK_dr / dK_du
h = 1e-6
fd = (-x0 / (1 + r + h) - (-x0 / (1 + r - h))) / (2 * h)
print(f"  x0 = {x0}, r = {r}: u* = {u_st}, dK/du = {dK_du}, dK/dr = {dK_dr}, du*/dr = {du_dr}")
print(f"  fórmula cerrada x0/(1+r)^2 = {x0/(1+r)**2}; diferencias finitas = {fd:.6f}")
# un paso de aprendizaje: queremos u* = -0,6 (más agresivo): pérdida L = (u* + 0,6)^2
L = (u_st + 0.6) ** 2; dL_dr = 2 * (u_st + 0.6) * du_dr
print(f"  pérdida (u* + 0,6)^2 = {L:.4f}, dL/dr = {dL_dr:.4f}; con paso 1: r = {r - dL_dr:.3f}, "
      f"u* nuevo = {-x0/(1 + r - dL_dr):.4f}")

# ---------------------------------------------------------------------------
cab(15, "CEM EN 1D: LA GAUSSIANA SE ESTRECHA HACIA EL MÍNIMO (figura del algoritmo)")
# S(u) = (u - 2)^2 + 0,5 u^2 (mínimo en u = 4/3), n = 20 muestras, n_e = 4 élites, 3 iteraciones
rng = np.random.default_rng(2)
mu_, sd_ = 0.0, 1.5
Xc = lin(-3, 4, 40, 500)
Yc = lin(0, 1.0, 200, 30)
uu = np.linspace(-3, 4, 141)
print("path coste:", path(uu, np.clip(((uu - 2) ** 2 + 0.5 * uu**2) / 14, 0, 1.0), Xc, Yc))
for it in range(4):
    g = np.exp(-0.5 * ((uu - mu_) / sd_) ** 2)
    print(f"  iteración {it}: mu = {mu_:.3f}, sigma = {sd_:.3f}")
    print(f"path gauss {it}:", path(uu, g, Xc, Yc))
    if it == 3:
        break
    us = mu_ + sd_ * rng.normal(size=20)
    Ss = (us - 2) ** 2 + 0.5 * us**2
    el = us[np.argsort(Ss)[:4]]
    print(f"    élites {np.round(np.sort(el),3)}")
    mu_, sd_ = el.mean(), el.std()
print(f"  mínimo u = 4/3: x = {Xc(4/3):.1f}")
for v in (-2, 0, 2, 4):
    print(f"  marca u={v}: x={Xc(v):.1f}")
