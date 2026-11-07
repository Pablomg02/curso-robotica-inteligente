"""Ejemplos numéricos de las hojas de B3b (render y real2sim).

Uso:  python3 ejemplos_b3b.py > salida_ejemplos.txt      (numpy, CPU, unos segundos)

Cada ejemplo lleva un número que se cita en la hoja correspondiente («ejemplo 6 del script»).
Los números de los ejemplos (triángulos, tramos, gaussianas, parches de imagen) son números de
ejemplo, elegidos para que la cuenta se siga a mano; el rayo de cristal y pared es el de
figuras_b3b.py, sección 1, y la geometría de la captura, la de su sección 3.
También salen de aquí los datos de las figuras nuevas (líneas «path ...» y «fig ...»).
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def titulo(n, texto):
    print("=" * 78)
    print(f"EJEMPLO {n}. {texto}")


def svgpath(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


# --------------------------------------------------------------------------------------------- 1
titulo(1, "Rasterizar un triángulo: proyección, test de cobertura y profundidad (z-buffer)")
f, cx, cy = 100.0, 50.0, 50.0                    # imagen de 100 x 100 px
A, B, C = np.array([-1.0, -1.0, 4.0]), np.array([1.0, -1.0, 4.0]), np.array([0.0, 0.5, 2.0])


def proy(P):
    return np.array([f * P[0] / P[2] + cx, f * P[1] / P[2] + cy])


a2, b2, c2 = proy(A), proy(B), proy(C)
print(f"  cámara: f = {f:.0f} px, centro ({cx:.0f}, {cy:.0f}); x a la derecha, y hacia abajo, z hacia delante")
for n, P, p in (("A", A, a2), ("B", B, b2), ("C", C, c2)):
    print(f"  {n} = {P} m  ->  u = f x/z + cx = {p[0]:.1f}, v = f y/z + cy = {p[1]:.1f} px")


def edge(p, q, r):          # producto vectorial 2D (q - p) x (r - p): signo = lado
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


P = np.array([50.0, 50.0])
area = edge(a2, b2, c2)
lA, lB, lC = edge(b2, c2, P) / area, edge(c2, a2, P) / area, edge(a2, b2, P) / area
print(f"  píxel P = (50, 50): funciones de arista E_BC, E_CA, E_AB = {edge(b2,c2,P):.0f}, {edge(c2,a2,P):.0f}, "
      f"{edge(a2,b2,P):.0f} (todas del signo del área {area:.0f}: P está dentro)")
print(f"  coordenadas baricéntricas lambda = E/área: ({lA:.3f}, {lB:.3f}, {lC:.3f}), suman {lA+lB+lC:.3f}")
inv_z = lA / A[2] + lB / B[2] + lC / C[2]
print(f"  profundidad correcta: 1/z = sum lambda_k / z_k = {inv_z:.4f} -> z = {1/inv_z:.3f} m")
print(f"  interpolando z directamente (error): {lA*A[2] + lB*B[2] + lC*C[2]:.3f} m")
# comprobación 3D: intersección del rayo del píxel (0,0,1) con el plano del triángulo
nrm = np.cross(B - A, C - A)
t = (nrm @ A) / (nrm @ np.array([0, 0, 1.0]))
print(f"  comprobación: el rayo del píxel corta el plano del triángulo en z = {t:.3f} m")
print(f"  z-buffer: una pared a 5 m que también cubre P -> se queda min(2.667, 5) = 2.667 m: gana el triángulo")
print(f"  fig: A' {a2}, B' {b2}, C' {c2}; escala 2 en la figura (100 px -> 200 unidades)")

# --------------------------------------------------------------------------------------------- 2
titulo(2, "Trazar un rayo: intersección con el suelo, rayo de sombra y ruido de Monte Carlo")
o = np.array([0.0, 0.0, 1.5])                       # cámara a 1,5 m (mundo: z arriba)
ang = math.radians(20)
d = np.array([0.0, math.cos(ang), -math.sin(ang)])   # 20° por debajo del horizonte
t_suelo = -o[2] / d[2]
x = o + t_suelo * d
print(f"  rayo o = {o}, d = ({d[0]:.3f}, {d[1]:.4f}, {d[2]:.4f}); suelo z = 0: t = -o_z/d_z = {t_suelo:.3f} m")
print(f"  punto de impacto x = o + t d = ({x[0]:.2f}, {x[1]:.3f}, {x[2]:.2f})")
luz = np.array([0.0, x[1], 10.0])
esf_c, esf_r = np.array([0.0, x[1], 2.0]), 0.5
l = (luz - x) / np.linalg.norm(luz - x)
oc = x - esf_c
bq, cq = 2 * l @ oc, oc @ oc - esf_r ** 2
disc = bq ** 2 - 4 * cq
ts = ((-bq - math.sqrt(disc)) / 2, (-bq + math.sqrt(disc)) / 2)
print(f"  rayo de sombra hacia la luz en {luz}: dirección {l}; esfera de radio {esf_r} m en {esf_c}")
print(f"  |x + s l - c|^2 = r^2 -> s^2 + {bq:.2f} s + {cq:.2f} = 0; discriminante {disc:.2f} > 0: "
      f"corta en s = {ts[0]:.2f} y {ts[1]:.2f} m, antes de la luz ({np.linalg.norm(luz-x):.1f} m): punto en sombra")
# Monte Carlo: fracción de cielo visible p = 0,5 (medio hemisferio tapado por una pared)
rng = np.random.default_rng(1)
p_vis = 0.5
print(f"  Monte Carlo: cielo visible p = {p_vis}; cada muestra vale 1 (ve cielo) o 0 (pared)")
for N in (1, 4, 16, 64, 256):
    est = rng.binomial(N, p_vis, size=20000) / N
    print(f"    N = {N:4d} muestras por píxel: desviación típica {est.std():.4f} (teórica sqrt(p(1-p)/N) = "
          f"{math.sqrt(p_vis*(1-p_vis)/N):.4f})")

# --------------------------------------------------------------------------------------------- 3
titulo(3, "Aleatorización de dominio en un juguete lineal: lo que no se aleatoriza no se aprende a ignorar")
# o1 = s + xi + zeta + e1 (píxel del objeto: señal + luz global + reflejo), o2 = xi + e2 (píxel de fondo)
rng = np.random.default_rng(3)
n, ruido = 20000, 0.05


def datos(xi_fn, zeta=0.0, n=n):
    s = rng.uniform(0, 1, n)
    xi = xi_fn(n)
    o1 = s + xi + zeta + rng.normal(0, ruido, n)
    o2 = xi + rng.normal(0, ruido, n)
    return np.c_[o1, o2, np.ones(n)], s


def ajusta(X, s):
    return np.linalg.lstsq(X, s, rcond=None)[0]


XA, sA = datos(lambda n: np.zeros(n))
XB, sB = datos(lambda n: rng.uniform(-0.5, 0.5, n))
wA, wB = ajusta(XA, sA), ajusta(XB, sB)
print(f"  modelo s_hat = a o1 + b o2 + c, mínimos cuadrados con {n} imágenes de juguete")
print(f"  A, sin aleatorizar (xi = 0):           a = {wA[0]:.3f}, b = {wA[1]:+.3f}, c = {wA[2]:+.3f}")
print(f"  B, aleatorizando xi ~ U(-0,5; 0,5):    a = {wB[0]:.3f}, b = {wB[1]:+.3f}, c = {wB[2]:+.3f}")


def L(w, xi, zeta=0.0):
    X, s = datos(lambda n: np.full(n, xi), zeta)
    return np.mean((X @ w - s) ** 2)


print("  error cuadrático medio L(theta; xi) en mundos de prueba:")
for xi, zeta, nota in ((0.0, 0.0, "como el simulador"), (0.3, 0.0, "luz distinta, dentro del rango"),
                       (0.3, 0.2, "y un reflejo que nunca se aleatorizó")):
    print(f"    xi* = {xi:.1f}, zeta* = {zeta:.1f} ({nota}): A {L(wA, xi, zeta):.4f}   B {L(wB, xi, zeta):.4f}")
xs = np.linspace(-0.5, 0.5, 101)
LB = np.array([L(wB, x) for x in xs])
LA = np.array([L(wA, x) for x in xs])
print(f"  riesgo esperado bajo p (malla de 101 valores de xi): A {LA.mean():.4f}, B {LB.mean():.4f}")
for k in (2, 5, 10):
    print(f"    Markov para A: P(L > {k} L_bar) real = {(LA > k*LA.mean()).mean():.2f}  <=  1/k = {1/k:.2f}")
print(f"  A: L crece como xi^2 ({LA[0]:.4f} en xi = -0,5); B: casi constante ({LB.min():.4f}-{LB.max():.4f})")

# --------------------------------------------------------------------------------------------- 4
titulo(4, "Transmitancia: niebla de densidad constante")
sig, Ltr = 0.5, 2.0
print(f"  sigma = {sig} 1/m durante {Ltr} m: T = exp(-sigma L) = exp(-{sig*Ltr:.1f}) = {math.exp(-sig*Ltr):.4f}")
for dt in (0.5, 0.1, 0.01):
    k = int(round(Ltr / dt))
    print(f"    pasos de {dt} m: producto de exp(-sigma dt) = {math.exp(-sig*dt)**k:.4f};  "
          f"producto de (1 - sigma dt) = {(1-sig*dt)**k:.4f}")
print(f"  niebla homogénea sin fin: w(t) = sigma exp(-sigma t), distancia media de parada 1/sigma = {1/sig:.1f} m")
print(f"  probabilidad de chocar en 1 cm: 1 - exp(-sigma 0,01) = {1-math.exp(-sig*0.01):.6f} ~ sigma dt = {sig*0.01:.6f}")

# --------------------------------------------------------------------------------------------- 5
titulo(5, "Cuadratura con tres tramos (aire, niebla densa, objeto)")
sig3 = np.array([0.0, 2.0, 10.0])
dl3 = np.array([0.5, 0.5, 0.5])
c3 = np.array([0.2, 0.8, 0.5])
al = 1 - np.exp(-sig3 * dl3)
T3 = np.concatenate([[1.0], np.cumprod(1 - al)[:-1]])
w3 = T3 * al
print(f"  sigma = {sig3}, delta = {dl3}, c = {c3}")
print(f"  sigma delta = {sig3*dl3};  alpha = 1 - exp(-sigma delta) = {al}")
print(f"  T = prod_(j<i) (1 - alpha_j) = {T3};  pesos w = T alpha = {w3}; suma {w3.sum():.4f}")
print(f"  C_hat = sum w c = {w3 @ c3:.4f};  llega al fondo T_final = {np.prod(1-al):.4f}")
print(f"  si se usara sigma delta en vez de alpha: pesos {T3*sig3*dl3}, suma {np.sum(T3*sig3*dl3):.3f} > 1 (más luz de la que entra)")
# comprobación: integral exacta en tramos constantes
tt = np.linspace(0, 1.5, 150001)
sg = np.repeat(sig3, 50000)[:150000]
cg = np.repeat(c3, 50000)[:150000]
dtt = np.diff(tt)
Tc = np.exp(-np.concatenate([[0], np.cumsum(sg * dtt)[:-1]]))
print(f"  integral con 150 000 subtramos: {np.sum(Tc*sg*cg*dtt):.4f} (la cuadratura es exacta si sigma y c son constantes en cada tramo)")

# --------------------------------------------------------------------------------------------- 6
titulo(6, "El rayo de cristal y pared (figuras_b3b.py, sección 1), a mano")


def sigma_r(t):
    vidrio = 1.2 * np.exp(-0.5 * ((t - 2.0) / 0.08) ** 2)
    pared = 60.0 / (1 + np.exp(-(t - 4.0) / 0.02))
    return vidrio + pared


def color_r(t):
    return np.where(t < 3.0, 0.9, 0.3)


def render(n, t0=0.5, t1=6.0):
    tt = np.linspace(t0, t1, n + 1)
    tm = 0.5 * (tt[1:] + tt[:-1])
    dl = np.diff(tt)
    a = 1 - np.exp(-sigma_r(tm) * dl)
    T = np.concatenate([[1.0], np.cumprod(1 - a)[:-1]])
    w = T * a
    return (w * color_r(tm)).sum(), (w * tm).sum() / w.sum(), tm, a, w


int_v = 1.2 * 0.08 * math.sqrt(2 * math.pi)
Tv = math.exp(-int_v)
print(f"  paso 1: integral de sigma en el cristal = 1,2 x 0,08 x sqrt(2 pi) = {int_v:.4f}")
print(f"  paso 2: T tras el cristal = exp(-{int_v:.4f}) = {Tv:.4f}; peso del cristal = 1 - T = {1-Tv:.4f}")
print(f"  paso 3: la pared es opaca: se queda con todo lo que llega, peso {Tv:.4f}")
print(f"  paso 4: color = {1-Tv:.3f} x 0,9 + {Tv:.3f} x 0,3 = {(1-Tv)*0.9 + Tv*0.3:.4f}")
print(f"  paso 5: profundidad esperada = {1-Tv:.3f} x 2 + {Tv:.3f} x 4 = {(1-Tv)*2 + Tv*4:.4f} m")
Cref, Dref, _, _, _ = render(200000)
print(f"  referencia numérica (200 000 tramos): color {Cref:.4f}, profundidad {Dref:.3f} m")
for nn in (16, 32, 64, 128, 256):
    c, dd, tm, a, w = render(nn)
    print(f"    {nn:4d} muestras: color {c:.4f} (error {abs(c-Cref):.4f}), profundidad {dd:.3f} m")
rng6 = np.random.default_rng(0)
errs = []
for _ in range(200):
    tt6 = np.sort(np.concatenate([[0.5, 6.0], rng6.uniform(0.5, 6.0, 63)]))
    tm6 = 0.5 * (tt6[1:] + tt6[:-1]); dl6 = np.diff(tt6)
    a6 = 1 - np.exp(-sigma_r(tm6) * dl6); T6 = np.concatenate([[1.0], np.cumprod(1 - a6)[:-1]])
    errs.append(abs((T6 * a6 * color_r(tm6)).sum() - Cref))
print(f"    64 muestras con posiciones al azar (200 repeticiones): error medio {np.mean(errs):.4f}")
c, dd, tm, a, w = render(16)
k = np.argmin(abs(tm - 2.0))
print(f"  con 16 muestras: delta = {5.5/16:.4f} m; la muestra más cercana al cristal está en t = {tm[k]:.4f} m, "
      f"sigma = {sigma_r(tm[k]):.3f}, alpha = {a[k]:.4f} (frente a {1-Tv:.4f}): el cristal pesa de más")
# mediana: donde la opacidad acumulada 1 - T cruza 0,5
tg = np.linspace(0.5, 6.0, 550001)
sg = sigma_r(tg)
Tg = np.exp(-np.concatenate([[0], np.cumsum(0.5 * (sg[1:] + sg[:-1]) * np.diff(tg))]))
med = tg[np.argmax(1 - Tg >= 0.5)]
print(f"  mediana de la profundidad (1 - T = 0,5): {med:.3f} m  (la pared)")
print(f"  opacidad acumulada justo después del cristal (t = 2,5 m): {1 - Tg[np.searchsorted(tg, 2.5)]:.4f}")
X = lin(0.5, 6.0, 60, 520)
Y = lin(0, 1.0, 230, 30)
sel = np.linspace(0, len(tg) - 1, 551).astype(int)
print("  path opacidad acumulada:", svgpath(tg[sel], 1 - Tg[sel], X, Y))
print(f"  fig: x(2) = {X(2):.1f}, x(3,572) = {X(3.572):.1f}, x(4) = {X(4):.1f}, x(med) = {X(med):.1f}, "
      f"y(0,214) = {Y(1-Tv):.1f}, y(0,5) = {Y(0.5):.1f}, y(1) = {Y(1):.1f}")

# --------------------------------------------------------------------------------------------- 7
titulo(7, "Codificación posicional gamma(p): puntos cercanos que quedan lejos")
p = 0.3
Lp = 3
g = []
for l_ in range(Lp):
    g += [math.sin(2 ** l_ * math.pi * p), math.cos(2 ** l_ * math.pi * p)]
print(f"  gamma({p}) con L = {Lp}: " + ", ".join(f"{v:.3f}" for v in g))
q = 0.32


def gam(p, L):
    return np.concatenate([[np.sin(2 ** l_ * np.pi * p), np.cos(2 ** l_ * np.pi * p)] for l_ in range(L)])


for L_ in (3, 10):
    print(f"  distancia entre p = 0,30 y 0,32: en p, {abs(q-p):.2f}; en gamma con L = {L_}: "
          f"{np.linalg.norm(gam(p, L_) - gam(q, L_)):.3f}")
# por qué ayuda: una red con pesos acotados es Lipschitz; separar puntos cercanos exige pendiente
print("  un borde: la salida debe pasar de 0 a 1 entre p = 0,30 y p = 0,32 (por ejemplo, de aire a pared)")
print(f"    con p crudo: distancia 0,02 -> pendiente (constante de Lipschitz) mínima {1/0.02:.0f}")
for L_ in (3, 6, 10):
    dg = np.linalg.norm(gam(p, L_) - gam(q, L_))
    print(f"    con gamma, L = {L_:2d}: distancia {dg:.3f} -> pendiente mínima {1/dg:.2f}")
print(f"  dos puntos lejanos, p = 0,3 y p = -0,7: distancia en gamma (L = 10) {np.linalg.norm(gam(0.3,10)-gam(-0.7,10)):.3f}; "
      f"la máxima posible es 2 sqrt(L) = {2*math.sqrt(10):.3f}")
print(f"  periodo: gamma(p + 2) = gamma(p) (la frecuencia más baja es pi): por eso x se normaliza a [-1, 1]; "
      f"|gamma(0,3) - gamma(2,3)| = {np.linalg.norm(gam(0.3,10)-gam(2.3,10)):.1e}")
pp = np.linspace(-1, 1, 2201)
X7 = lin(-1, 1, 50, 490)
for row, l_ in enumerate((0, 2, 4, 6)):
    Y7 = lin(-1, 1, 40 + 62 * row + 26, 40 + 62 * row - 26)
    print(f"  path sin l={l_}:", svgpath(pp, np.sin(2 ** l_ * np.pi * pp), X7, Y7))
    print(f"    fig l={l_}: centro y = {40+62*row}; sin en 0,30 -> y = {Y7(math.sin(2**l_*math.pi*0.3)):.1f}, "
          f"en 0,32 -> y = {Y7(math.sin(2**l_*math.pi*0.32)):.1f}")
print(f"  fig: x(-1) = {X7(-1):.0f}, x(0) = {X7(0):.0f}, x(0,30) = {X7(0.30):.1f}, x(0,32) = {X7(0.32):.1f}, x(1) = {X7(1):.0f}")

# --------------------------------------------------------------------------------------------- 8
titulo(8, "Covarianza de una gaussiana como R S S^T R^T (ejemplo en 2D)")
th = math.radians(30)
R2 = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
S2 = np.diag([2.0, 0.5])
Sig2 = R2 @ S2 @ S2.T @ R2.T
print(f"  theta = 30°, s = (2; 0,5):  R = {R2.round(3).tolist()}, S S^T = diag(4; 0,25)")
print(f"  Sigma = R S S^T R^T = {Sig2.round(4).tolist()}")
print(f"  a mano: Sigma_11 = 4 cos^2 + 0,25 sin^2 = {4*math.cos(th)**2 + 0.25*math.sin(th)**2:.4f}; "
      f"Sigma_12 = (4 - 0,25) sin cos = {3.75*math.sin(th)*math.cos(th):.4f}; "
      f"Sigma_22 = 4 sin^2 + 0,25 cos^2 = {4*math.sin(th)**2 + 0.25*math.cos(th)**2:.4f}")
ev, evec = np.linalg.eigh(Sig2)
print(f"  autovalores {ev} (= s^2), autovectores (columnas) {evec.round(3).tolist()} (= columnas de R, salvo signo)")
print(f"  det Sigma = {np.linalg.det(Sig2):.4f} = (2 x 0,5)^2")
mal = Sig2.copy()
mal[1, 1] -= 1.0
print(f"  paso de gradiente sobre la matriz directamente (Sigma_22 - 1): {mal.round(4).tolist()}, "
      f"autovalores {np.linalg.eigvalsh(mal).round(4)}: ya no es una covarianza")
v = np.array([1.0, -2.0])
print(f"  v^T Sigma v con v = (1, -2): {v @ Sig2 @ v:.4f} = |S^T R^T v|^2 = {np.linalg.norm(S2.T @ R2.T @ v)**2:.4f} >= 0")
print(f"  activaciones: escala = exp(s_crudo): s_crudo = -3 -> {math.exp(-3):.4f} m; opacidad = sigmoide(a): "
      f"a = 0 -> 0,5; a = 4 -> {1/(1+math.exp(-4)):.4f}; a = -6 -> {1/(1+math.exp(6)):.4f}")
qv = np.array([2.0, 0.0, 0.0, 1.0])
print(f"  cuaternión sin normalizar {qv} -> se normaliza a {qv/np.linalg.norm(qv)} antes de construir R")
P59 = 3 + 3 + 4 + 1 + 48
print(f"  parámetros: 3 + 3 + 4 + 1 + 16 x 3 = {P59}; un millón: {1e6*P59*4/1e6:.0f} MB en float32")
# figura: elipse a 1 sigma, escala 40 unidades por metro, centro (150, 120); y hacia abajo en SVG
sc, c0 = 40.0, np.array([150.0, 120.0])
for k_ in range(2):
    e = R2[:, k_] * S2[k_, k_]
    print(f"  fig eje {k_+1}: de {c0} a ({c0[0]+sc*e[0]:.1f}, {c0[1]-sc*e[1]:.1f})")
print(f"  fig elipse rx = {sc*2:.0f}, ry = {sc*0.5:.0f}, rotate(-30)")

# --------------------------------------------------------------------------------------------- 9
titulo(9, "Color que depende de la dirección: armónicos esféricos de grado 0 y 1")
C0 = 0.5 / math.sqrt(math.pi)
C1 = math.sqrt(3 / (4 * math.pi))
print(f"  Y_00 = 1/(2 sqrt(pi)) = {C0:.4f};  Y_1 = sqrt(3/(4 pi)) (y, z, x) = {C1:.4f} (y, z, x)")
k00, k11 = 1.6, 0.4
for nm, dv in (("+x", (1, 0, 0)), ("-x", (-1, 0, 0)), ("+y", (0, 1, 0))):
    c_ = k00 * C0 + k11 * C1 * dv[0]
    print(f"  k_00 = {k00}, k_11 = {k11} (término en x), vista desde d = {nm}: c = {k00*C0:.3f} + {k11*C1*dv[0]:+.3f} = {c_:.3f}")
print(f"  funciones por grado l: 2l + 1 -> 1 + 3 + 5 + 7 = {1+3+5+7} por canal; x 3 canales = {16*3}")

# --------------------------------------------------------------------------------------------- 10
titulo(10, "Proyección de una gaussiana: Sigma' = J Sigma J^T en 2D (plano x-z, imagen 1D u = f x / z)")
f10 = 300.0
rng = np.random.default_rng(10)
for mu, nota in ((np.array([1.0, 4.0]), "lejos"), (np.array([0.5, 1.5]), "cerca")):
    Sg = np.diag([0.2 ** 2, 0.5 ** 2])
    x_, z_ = mu
    J = np.array([f10 / z_, -f10 * x_ / z_ ** 2])
    var = J @ Sg @ J
    pts = rng.multivariate_normal(mu, Sg, 400000)
    pts = pts[pts[:, 1] > 0.2]
    u = f10 * pts[:, 0] / pts[:, 1]
    print(f"  {nota}: mu = {mu} m, Sigma = diag(0,04; 0,25)")
    print(f"    u(mu) = f x/z = {f10*x_/z_:.1f} px;  J = (f/z, -f x/z^2) = ({J[0]:.2f}, {J[1]:.2f}) px/m")
    print(f"    Sigma' = J Sigma J^T = {J[0]**2:.2f} x 0,04 + {J[1]**2:.2f} x 0,25 = {J[0]**2*0.04:.1f} + "
          f"{J[1]**2*0.25:.1f} = {var:.1f} px^2  ->  desviación {math.sqrt(var):.2f} px")
    p16, p84 = np.percentile(u, [15.87, 84.13])
    print(f"    Monte Carlo (400 000 puntos proyectados): media {u.mean():.1f} px, mitad del intervalo 16-84 % "
          f"{(p84-p16)/2:.2f} px, desviación {u.std():.2f} px")
# rotación de cámara: W Sigma W^T es exacto
phi = math.radians(10)
W2 = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
Sg = np.diag([0.04, 0.25])
print(f"  cámara girada 10°: W Sigma W^T = {(W2 @ Sg @ W2.T).round(4).tolist()} (exacto: el giro es lineal)")
# figura: vista cenital, x a la derecha [-0,5; 2], z hacia arriba [0; 5]; escala 60 por metro
Xf = lin(-0.5, 2.0, 30, 180)
Zf = lin(0, 5, 290, 20)
print(f"  fig: origen ({Xf(0):.0f}, {Zf(0):.0f}); mu ({Xf(1):.0f}, {Zf(4):.0f}); rx = {0.2*60:.0f}, ry = {0.5*60:.0f}; "
      f"plano imagen z = 1 en y = {Zf(1):.0f}; u/f = 0,25 -> x = {Xf(0.25):.0f}; "
      f"+-1 sigma: u/f = {(75+17.69)/300:.4f}, {(75-17.69)/300:.4f} -> x = {Xf((75+17.69)/300):.1f}, {Xf((75-17.69)/300):.1f}")
for uf in ((75 + 17.69) / 300, (75 - 17.69) / 300):
    print(f"    rayo u/f = {uf:.4f}: hasta z = 4,8 -> ({Xf(uf*4.8):.1f}, {Zf(4.8):.1f})")

# --------------------------------------------------------------------------------------------- 11
titulo(11, "Un píxel: alpha de cada gaussiana y composición de delante hacia atrás")
Sp = np.diag([16.0, 4.0])
Dl = np.array([4.0, 2.0])
o1 = 0.8
m2 = Dl @ np.linalg.inv(Sp) @ Dl
a1 = o1 * math.exp(-0.5 * m2)
print(f"  gaussiana 1: Sigma' = diag(16, 4) px^2 (desviaciones 4 y 2 px), o = {o1}; píxel a Delta = (4, 2) px")
print(f"    Delta^T Sigma'^-1 Delta = 16/16 + 4/4 = {m2:.1f};  alpha = o exp(-1/2 x {m2:.0f}) = {a1:.4f}")
alph = np.array([a1, 0.5, 0.9])
col = np.array([1.0, 0.2, 0.6])
T_ = np.concatenate([[1.0], np.cumprod(1 - alph)[:-1]])
w_ = T_ * alph
print(f"  tres gaussianas ordenadas por profundidad: alpha = {alph.round(3)}, color = {col}")
print(f"    T = {T_.round(4)};  pesos T alpha = {w_.round(4)};  C = {w_ @ col:.4f};  T tras las tres = {np.prod(1-alph):.4f}")
print(f"  reglas del rasterizador de 3DGS: se salta alpha < 1/255 = {1/255:.4f}; se limita alpha a 0,99; "
      f"se para antes de que la opacidad acumulada pase de 0,9999")
# ¿cuántas gaussianas opacas hacen falta para saturar?
for a_ in (0.3, 0.5, 0.9):
    n_ = math.ceil(math.log(1e-4) / math.log(1 - a_))
    print(f"    con alpha = {a_} todas iguales: T < 1e-4 tras {n_} gaussianas")

# --------------------------------------------------------------------------------------------- 12
titulo(12, "Tiles de 16 x 16 y claves de ordenación")
print(f"  1080p: {1920//16} x {math.ceil(1080/16)} = {1920//16 * math.ceil(1080/16)} tiles")
k99 = math.sqrt(-2 * math.log(0.01))
print(f"  intervalo del 99 % de una gaussiana 2D: radio de Mahalanobis sqrt(-2 ln 0,01) = {k99:.3f}")
sd = 17.69
r99 = k99 * sd
cxg, cyg = 100.0, 40.0
x0t, x1t = int((cxg - r99) // 16), int((cxg + r99) // 16)
y0t, y1t = int(max(0, cyg - r99) // 16), int((cyg + r99) // 16)
print(f"  gaussiana del ejemplo 10 (desviación {sd} px, isótropa por simplificar) centrada en ({cxg:.0f}, {cyg:.0f}): "
      f"recuadro +-{r99:.1f} px -> tiles x {x0t}-{x1t}, y {y0t}-{y1t}: {(x1t-x0t+1)*(y1t-y0t+1)} instancias")
gs = [("A", 2.0, [5, 6]), ("B", 3.5, [6]), ("C", 1.2, [6, 7])]
keys = []
for nm, z, tiles in gs:
    for tl in tiles:
        keys.append((tl << 32 | int(z * 1000), nm, tl, z))
keys.sort()
print("  claves (tile << 32) | profundidad en mm, ordenadas: " +
      "; ".join(f"tile {tl}, {nm} a {z} m" for _, nm, tl, z in keys))

# --------------------------------------------------------------------------------------------- 13
titulo(13, "Pérdida L1 frente a SSIM en un parche de 8 píxeles")
ref = np.array([0.2, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8])
brillo = ref + 0.1
borroso = np.convolve(np.r_[ref[0], ref, ref[-1]], np.ones(3) / 3, mode="valid")
C1s, C2s = 0.01 ** 2, 0.03 ** 2


def ssim(x, y):
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - mx) * (y - my)).mean()
    lum = (2 * mx * my + C1s) / (mx ** 2 + my ** 2 + C1s)
    cs = (2 * cxy + C2s) / (vx + vy + C2s)
    return lum * cs, lum, cs


print(f"  referencia {ref}")
for nm, x_ in (("más brillo (+0,1)", brillo), ("desenfocado (media de 3)", borroso)):
    s_, lum, cs = ssim(x_, ref)
    print(f"  {nm}: {x_.round(3)}")
    print(f"    L1 = {np.abs(x_-ref).mean():.4f};  SSIM = luminancia {lum:.4f} x contraste-estructura {cs:.4f} = {s_:.4f}; "
          f"1 - SSIM = {1-s_:.4f}")
    print(f"    pérdida con lambda = 0,2 y D-SSIM = 1 - SSIM: 0,8 x {np.abs(x_-ref).mean():.4f} + 0,2 x {1-s_:.4f} = "
          f"{0.8*np.abs(x_-ref).mean() + 0.2*(1-s_):.4f}")

# --------------------------------------------------------------------------------------------- 14
titulo(14, "Densificación: clonar y dividir")
s_big = np.array([0.40, 0.40, 0.10])
print(f"  dividir: escala {s_big} m -> dos gaussianas de {s_big/1.6} m (1/1,6 = {1/1.6:.3f}), "
      f"centros muestreados de la gaussiana original")
print(f"  clonar: copia del mismo tamaño desplazada en la dirección del gradiente de posición en pantalla")
print(f"  umbral del gradiente medio de posición en pantalla: tau_pos = 0,0002")
print(f"  calendario hasta 30 000 iteraciones: densificar cada 100; bandas de SH en 1000, 2000 y 3000; "
      f"opacidad a casi cero cada 3000 ({len(range(3000, 30001, 3000))} veces si durase todo el entrenamiento)")
Xt = lin(0, 30000, 40, 1150)
print(f"  fig: x(0) {Xt(0):.0f}, x(1000) {Xt(1000):.1f}, x(2000) {Xt(2000):.1f}, x(3000) {Xt(3000):.1f}, "
      f"x(7000) {Xt(7000):.1f}, x(30000) {Xt(30000):.0f}; reinicios: " + ", ".join(f"{Xt(k):.0f}" for k in range(3000, 30001, 3000)))

# --------------------------------------------------------------------------------------------- 15
titulo(15, "Bundle adjustment: una similitud no cambia el coste")
rng = np.random.default_rng(15)


def rotz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def rotx(a):
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


Kc = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]])
Xs = rng.uniform([-2, -2, 6], [2, 2, 10], size=(20, 3))
cams = [(rotz(0.1 * i) @ rotx(0.05 * i), rng.normal(0, 0.5, 3)) for i in range(5)]


def proyecta(Rc, tc, X):
    pc = (Rc @ X.T).T + tc
    ph = (Kc @ pc.T).T
    return ph[:, :2] / ph[:, 2:]


obs = [proyecta(Rc, tc, Xs) for Rc, tc in cams]
s_, Q = 0.8, rotz(0.35) @ rotx(-0.2)
pv = np.array([1.0, 2.0, 3.0])
Xs2 = s_ * (Q @ Xs.T).T + pv
dif = 0
for (Rc, tc), ob in zip(cams, obs):
    R2_ = Rc @ Q.T
    t2_ = s_ * tc - R2_ @ pv
    dif = max(dif, np.abs(proyecta(R2_, t2_, Xs2) - ob).max())
print(f"  5 cámaras, 20 puntos; similitud s = {s_}, Q = giro de 20° y -11,5°, p = {pv}")
print(f"  diferencia máxima entre las proyecciones antes y después: {dif:.2e} px (cero salvo redondeo)")
print(f"  grados de libertad libres: escala 1 + rotación 3 + traslación 3 = 7")
print(f"  escena a escala 0,8: 1 m de vuelo en la física son 1/0,8 = {1/0.8:.2f} unidades de la escena")

# --------------------------------------------------------------------------------------------- 16
titulo(16, "Fijar la escala con el VIO: similitud de mínimos cuadrados (Umeyama)")
rng = np.random.default_rng(16)
c_col = rng.uniform(-3, 3, size=(30, 3))
s_true, Q_true, t_true = 1.25, rotz(0.6) @ rotx(0.3), np.array([2.0, -1.0, 0.5])
p_vio = s_true * (Q_true @ c_col.T).T + t_true + rng.normal(0, 0.02, (30, 3))
mc, mv = c_col.mean(0), p_vio.mean(0)
A_, B_ = c_col - mc, p_vio - mv
Uu, Dd, Vt = np.linalg.svd(B_.T @ A_ / 30)
Ss = np.eye(3)
if np.linalg.det(Uu) * np.linalg.det(Vt) < 0:
    Ss[2, 2] = -1
Qh = Uu @ Ss @ Vt
sh = np.trace(np.diag(Dd) @ Ss) / (A_ ** 2).sum(1).mean()
print(f"  30 centros de cámara de COLMAP y sus posiciones del VIO (ruido 2 cm); verdad s = {s_true}")
print(f"  Umeyama: s = {sh:.4f}; error de rotación {math.degrees(math.acos(np.clip((np.trace(Qh.T@Q_true)-1)/2,-1,1))):.3f}°")
print(f"  solo escala, con la dispersión: s = sqrt(media |p - p_media|^2 / media |c - c_media|^2) = "
      f"{math.sqrt((B_**2).sum(1).mean() / (A_**2).sum(1).mean()):.4f}")

# --------------------------------------------------------------------------------------------- 17
titulo(17, "El ángulo a la vista de captura más cercana, a mano (figuras_b3b.py, sección 3)")
r, hc, ho = 5.0, 1.5, 1.0
e_cap = math.degrees(math.atan((hc - ho) / r))
print(f"  objeto a {ho} m de altura, captura a {hc} m y {r} m de distancia: la cámara mira {e_cap:.2f}° hacia abajo")
for h in (1.5, 3, 5, 8, 12):
    e = math.degrees(math.atan((h - ho) / r))
    print(f"    dron a {h:4.1f} m: mira atan(({h} - 1)/5) = {e:.2f}° hacia abajo; ángulo a la captura {e - e_cap:.2f}°")
e2 = math.degrees(math.atan((hc - ho) / 2))
print(f"  a 2 m de distancia y 1,5 m de altura: {e2:.2f} - {e_cap:.2f} = {e2-e_cap:.2f}°")
Xa = lin(0, 5, 60, 260)          # 40 unidades por metro en los dos ejes
Ya = lin(0, 5, 270, 70)
print(f"  fig (40 u/m): objeto ({Xa(0):.0f}, {Ya(1):.0f}); captura ({Xa(5):.0f}, {Ya(1.5):.0f}); dron 5 m ({Xa(5):.0f}, {Ya(5):.0f}); "
      f"suelo y = {Ya(0):.0f}")
for ang_ in (e_cap, math.degrees(math.atan(4 / 5))):
    print(f"    arco de radio 90 en el objeto, a {ang_:.2f}°: ({Xa(0) + 90*math.cos(math.radians(ang_)):.1f}, "
          f"{Ya(1) - 90*math.sin(math.radians(ang_)):.1f})")
