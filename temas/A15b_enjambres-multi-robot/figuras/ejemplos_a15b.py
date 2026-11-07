"""Ejemplos numéricos y datos de las figuras nuevas de A15b (enjambres y multi-robot real).

Cada número de un ejemplo «a mano» de la presentación sale de aquí. Las cifras de Vicsek con 100 y 400
partículas siguen saliendo de vicsek_a15b.py, y las del experimento con drones, de figuras_a15b.py (GPU);
ninguno de los dos hace falta relanzarlo: donde se usan aquí, se copian de salida_vicsek.txt y salida_gpu.txt.
Los valores marcados como SUPUESTO son números pequeños elegidos aquí para que se sigan a mano.
Ejecutar:  python3 ejemplos_a15b.py > salida_ejemplos.txt   (numpy, CPU, segundos)
"""
import itertools
import math
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def titulo(n, s):
    print("\n" + "=" * 72 + f"\nEjemplo {n}. {s}\n" + "=" * 72)


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# ---------------------------------------------------------------------------
titulo(1, "Boids: las tres reglas para un pájaro con cuatro vecinos (SUPUESTO)")
xi, vi = np.array([0.0, 0.0]), np.array([1.0, 0.0])
X = np.array([[0.5, 0.5], [2.0, 1.0], [1.0, -2.0], [4.0, 0.0]])
V = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 0.0], [0.0, 1.0]])
R, RS = 3.0, 1.0
d = np.linalg.norm(X - xi, axis=1)
print("  distancias a los vecinos:", np.round(d, 3), f"(radio de vecindad r = {R}, radio de separación r_s = {RS})")
vec = d < R
cerca = d < RS
print("  en la vecindad:", vec, "| demasiado cerca:", cerca)
sep = sum((xi - X[j]) / d[j] ** 2 for j in range(len(X)) if cerca[j])
ali = V[vec].mean(0) - vi
coh = X[vec].mean(0) - xi
print(f"  separación  a_sep = sum (x_i - x_j)/|x_i - x_j|^2 = {sep}")
print(f"  alineación  a_ali = media(v_j) - v_i = {V[vec].mean(0)} - {vi} = {ali}")
print(f"  cohesión    a_coh = media(x_j) - x_i = {coh}")
w = (1.5, 1.0, 1.0)
a = w[0] * sep + w[1] * ali + w[2] * coh
print(f"  suma con pesos {w}: a = {a}; nueva velocidad con dt = 0,1: {vi + 0.1 * a}")
# reparto por prioridad de Reynolds: se acumulan peticiones en orden hasta agotar a_max; la última se recorta
AMAX = 1.6
acum, usado = np.zeros(2), 0.0
for nombre, req in (("separación", sep), ("alineación", ali), ("cohesión", coh)):
    m = np.linalg.norm(req)
    if usado + m <= AMAX:
        acum, usado = acum + req, usado + m
        print(f"  prioridad: {nombre} |a| = {m:.3f} entra entera (gastado {usado:.3f} de {AMAX})")
    else:
        f = max(AMAX - usado, 0.0) / m
        acum, usado = acum + f * req, AMAX
        print(f"  prioridad: {nombre} |a| = {m:.3f} se recorta a {f:.3f} de su tamaño (gastado {usado:.3f})")
print(f"  aceleración por prioridad: {acum}")
# figura: espacio físico a 40 px por unidad, con el pájaro i en (130, 145); velocidades de los vecinos a 24 px por m/s
FX, FY = (lambda v: 130 + 40 * v), (lambda v: 145 - 40 * v)
print("  figura: escala 40 px por unidad; origen en", (FX(0), FY(0)), "; radios r y r_s:", 40 * R, 40 * RS, "px")
for j in range(len(X)):
    print(f"   vecino {j+1}: ({FX(X[j,0]):.1f}, {FY(X[j,1]):.1f}); flecha de velocidad x0.6 hasta "
          f"({FX(X[j,0]+0.6*V[j,0]):.1f}, {FY(X[j,1]+0.6*V[j,1]):.1f})")
for nombre, v in (("sep", sep), ("ali", ali), ("coh", coh)):
    print(f"   flecha {nombre} desde el pájaro hasta ({FX(v[0]):.1f}, {FY(v[1]):.1f})")

# ---------------------------------------------------------------------------
titulo(2, "Vicsek: un paso para una partícula con dos vecinas, y el parámetro de orden (SUPUESTO)")
th = np.deg2rad([350.0, 10.0, 60.0])
print("  ángulos (grados):", np.rad2deg(th), "| media aritmética (incorrecta):", np.rad2deg(th).mean())
s, c = np.sin(th).sum(), np.cos(th).sum()
med = math.atan2(s, c)
print(f"  suma de senos {s:.4f}, suma de cosenos {c:.4f}, dirección media atan2 = {math.degrees(med):.2f} grados")
for eta in (0.5, 2.0):
    print(f"  con eta = {eta} rad, el ruido es uniforme en [{-eta/2:.2f}, {eta/2:.2f}] rad = "
          f"[{-math.degrees(eta/2):.1f}, {math.degrees(eta/2):.1f}] grados")
dth = 0.1
print(f"  si el sorteo da {dth} rad (+{math.degrees(dth):.2f} grados): nuevo ángulo {math.degrees(med + dth):.2f} grados")
v0 = 0.03
print(f"  avance con v0 = {v0}: dx = {v0*math.cos(med+dth):.4f}, dy = {v0*math.sin(med+dth):.4f}")
for nombre, ang in (("los tres de arriba", [350, 10, 60]), ("alineados", [30, 30, 30]), ("a 120 grados", [0, 120, 240]),
                    ("dos y dos opuestos", [0, 0, 180, 180])):
    z = np.exp(1j * np.deg2rad(ang))
    print(f"  v_a ({nombre}, {ang}) = |media de e^(i theta)| = {abs(z.mean()):.3f}")
# tamaño finito: con direcciones al azar, E|media| = sqrt(pi)/(2 sqrt N) (paseo aleatorio en 2-D)
rng = np.random.default_rng(0)
sim = {100: 0.125, 400: 0.062}      # salida_vicsek.txt, eta = 5,0
for n in (100, 400):
    mc = np.abs(np.exp(1j * rng.uniform(-np.pi, np.pi, (20000, n))).mean(1)).mean()
    print(f"  N = {n}: direcciones al azar, sqrt(pi)/(2 sqrt N) = {math.sqrt(math.pi)/(2*math.sqrt(n)):.4f}, "
          f"Monte Carlo {mc:.4f}; simulación con eta = 5 (salida_vicsek.txt): {sim[n]}")
# figura de la regla: partícula i en (150, 150), vecinas en (200, 110) y (120, 90), una fuera en (290, 60); radio 100 px
def flecha(x, y, grados, largo):
    a = math.radians(grados)
    return f"({x:.1f}, {y:.1f}) -> ({x + largo*math.cos(a):.1f}, {y - largo*math.sin(a):.1f})"
for nom, (x_, y_, g_) in (("i", (150, 150, 350)), ("j1", (200, 110, 10)), ("j2", (120, 90, 60)), ("fuera", (290, 60, 200))):
    print(f"  figura regla: {nom} rumbo {g_} grados, flecha {flecha(x_, y_, g_, 40)}")
print(f"  figura regla: dirección media {flecha(150, 150, math.degrees(med), 85)}; bordes del ruido (eta = 0,5) "
      f"{flecha(150, 150, math.degrees(med) - 14.32, 85)} y {flecha(150, 150, math.degrees(med) + 14.32, 85)}; "
      f"sorteado {flecha(150, 150, math.degrees(med + dth), 85)}")
print(f"  distancias en la figura: j1 {math.hypot(50, 40):.1f} px, j2 {math.hypot(30, 60):.1f} px, fuera {math.hypot(140, 90):.1f} px (radio 100)")
print(f"  cociente 100/400: al azar {math.sqrt(400/100):.2f}; simulación {sim[100]/sim[400]:.2f}")

# ---------------------------------------------------------------------------
titulo(3, "Consenso: laplaciano de 4 drones a mano, autovalores y convergencia (SUPUESTO)")
grafos = {
    "cadena 1-2-3-4": [(0, 1), (1, 2), (2, 3)],
    "anillo (cadena + 4-1)": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "cadena rota (sin 2-3)": [(0, 1), (2, 3)],
}
x0 = np.array([0.0, 2.0, 6.0, 12.0])
print("  valores iniciales x(0) =", x0, "| media", x0.mean())
res = {}
for nombre, E in grafos.items():
    A = np.zeros((4, 4))
    for i, j in E:
        A[i, j] = A[j, i] = 1
    D = np.diag(A.sum(1))
    L = D - A
    lam, U = np.linalg.eigh(L)
    res[nombre] = (L, lam, U)
    print(f"\n  {nombre}:\n  A =\n{A}\n  D = diag{tuple(np.diag(D))}\n  L = D - A =\n{L}")
    print(f"  L 1 = {L @ np.ones(4)}; autovalores {np.round(lam, 4)}; lambda_2 = {lam[1]:.4f}; lambda_max = {lam[-1]:.4f}")
    print(f"  dx/dt en t = 0: -L x(0) = {-L @ x0}; suma {(-L @ x0).sum():.1f} (la media no cambia)")
    e0 = x0 - x0.mean()
    for t in (0.0, 1.0, 2.0, 4.0):
        xt = U @ np.diag(np.exp(-lam * t)) @ U.T @ x0
        et = np.linalg.norm(xt - x0.mean())
        cota = math.exp(-lam[1] * t) * np.linalg.norm(e0)
        print(f"   t = {t}: x = {np.round(xt, 3)}, |e| = {et:.3f}, cota e^(-lambda2 t)|e(0)| = {cota:.3f}")
    if lam[1] > 1e-9:
        print(f"  tiempo para dividir el error por 10: ln 10 / lambda_2 = {math.log(10)/lam[1]:.2f}")
L, lam, U = res["cadena 1-2-3-4"]
print("\n  cadena: autovalores teóricos 2 - 2 cos(k pi / 4):", np.round([2 - 2 * math.cos(k * math.pi / 4) for k in range(4)], 4))
print("  autovector de lambda_2 (modo lento):", np.round(U[:, 1], 3))
c = U.T @ (x0 - x0.mean())
print("  coordenadas del error inicial en la base de autovectores:", np.round(c, 3))
# paso discreto x(k+1) = x(k) - eps L x(k): estable si eps < 2 / lambda_max
for eps in (0.3, 0.55, 0.6):
    x = x0.copy()
    for _ in range(40):
        x = x - eps * L @ x
    print(f"  discreto, eps = {eps} (límite 2/lambda_max = {2/lam[-1]:.3f}): tras 40 pasos x = {np.round(x, 3)}")
# figura: x_i(t) en la cadena, t en [0, 6] -> [60, 520], x en [0, 12] -> [250, 30]
TX, XY = lin(0, 6, 60, 520), lin(0, 12, 250, 30)
ts = np.linspace(0, 6, 31)
for i in range(4):
    xs = [(U @ np.diag(np.exp(-lam * t)) @ U.T @ x0)[i] for t in ts]
    print(f"  path consenso x{i+1}:", path(ts, xs, TX, XY))
# figura: |e(t)| frente a la cota, en escala log; |e| en [0.01, 20] -> y log
e0n = np.linalg.norm(x0 - x0.mean())
LY = lambda v: 250 - (math.log10(v) - math.log10(0.01)) / (math.log10(20) - math.log10(0.01)) * 220
for nombre in ("cadena 1-2-3-4", "anillo (cadena + 4-1)"):
    Lg, lg, Ug = res[nombre]
    es = [max(np.linalg.norm(Ug @ np.diag(np.exp(-lg * t)) @ Ug.T @ x0 - x0.mean()), 0.01) for t in ts]
    print(f"  path error {nombre}:", path(ts, es, TX, LY))
    print(f"  path cota {nombre}:", path(ts, [max(math.exp(-lg[1]*t)*e0n, 0.01) for t in ts], TX, LY))
print("  marcas eje log: " + ", ".join(f"{v}: y {LY(v):.1f}" for v in (0.01, 0.1, 1, 10)))

# ---------------------------------------------------------------------------
titulo(4, "ORCA: dos drones que se cruzan (SUPUESTO)")
pA, pB = np.array([0.0, 0.0]), np.array([4.0, 0.5])
vA, vB = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
rA = rB = 0.5
Rr, tau = rA + rB, 3.0
p = pB - pA
vrel = vA - vB
print(f"  p_B - p_A = {p}, |p| = {np.linalg.norm(p):.4f}; r_A + r_B = {Rr}; tau = {tau} s")
print(f"  velocidad relativa v_A - v_B = {vrel}")


def en_vo(v):
    """v en VO^tau: existe t en (0, tau] con |t v - p| < R."""
    # mínimo de |t v - p| para t en [0, tau]
    vv = v @ v
    t = np.clip((v @ p) / vv if vv > 0 else 0.0, 0.0, tau)
    return np.linalg.norm(t * v - p) < Rr


print(f"  ¿v_rel dentro del obstáculo de velocidades? {en_vo(vrel)}")
# tiempo hasta el choque con la velocidad actual: |p - t v_rel| = R
a2, b2, c2 = vrel @ vrel, -2 * p @ vrel, p @ p - Rr ** 2
tc = (-b2 - math.sqrt(b2 ** 2 - 4 * a2 * c2)) / (2 * a2)
print(f"  con estas velocidades chocarían en t = {tc:.3f} s (menor que tau = {tau})")
# frontera del VO truncado: dos patas tangentes y el arco del círculo de centro p/tau y radio R/tau
cen, rad = p / tau, Rr / tau
alpha = math.atan2(p[1], p[0])
beta = math.asin(Rr / np.linalg.norm(p))
print(f"  disco de corte: centro p/tau = {cen}, radio R/tau = {rad}; semiapertura del cono asin(R/|p|) = {math.degrees(beta):.3f} grados")
cands = []
# arco: punto del círculo más cercano a v_rel
dvc = vrel - cen
qa = cen + rad * dvc / np.linalg.norm(dvc)
if (qa - cen) @ qa <= 0:          # solo vale el arco que mira al origen (entre los puntos de tangencia)
    cands.append(("arco", qa))
else:
    print(f"   el punto del círculo más cercano {np.round(qa, 4)} no está en el arco frontal: no es frontera")
for sgn, nom in ((+1, "pata izquierda"), (-1, "pata derecha")):
    dirp = np.array([math.cos(alpha + sgn * beta), math.sin(alpha + sgn * beta)])
    # la pata empieza en el punto de tangencia con el disco de corte
    t0 = math.sqrt(np.linalg.norm(cen) ** 2 - rad ** 2)
    s_ = max(vrel @ dirp, t0)
    cands.append((nom, s_ * dirp))
for nom, q in cands:
    print(f"   candidato {nom}: {np.round(q, 4)}, distancia {np.linalg.norm(q - vrel):.4f}")
nom, qmin = min(cands, key=lambda c: np.linalg.norm(c[1] - vrel))
u = qmin - vrel
n = u / np.linalg.norm(u)
print(f"  punto más cercano de la frontera: {nom} {np.round(qmin, 4)}")
print(f"  u = {np.round(u, 4)} (|u| = {np.linalg.norm(u):.4f}); n = {np.round(n, 4)}")
# comprobación por fuerza bruta en una rejilla fina
g = np.linspace(-1, 3, 801)
best = None
for gx in g:
    for gy in g[::1]:
        v = np.array([gx, gy])
        if not en_vo(v):
            dd = np.linalg.norm(v - vrel)
            if best is None or dd < best[0]:
                best = (dd, v)
print(f"  comprobación por rejilla: punto fuera del VO más cercano {np.round(best[1], 3)}, distancia {best[0]:.4f}")
vA_new, vB_new = vA + u / 2, vB - u / 2
print(f"  ORCA_A|B: (v - (v_A + u/2)) . n >= 0, con v_A + u/2 = {np.round(vA + u/2, 4)}")
print(f"  ORCA_B|A: (v - (v_B - u/2)) . (-n) >= 0, con v_B - u/2 = {np.round(vB - u/2, 4)}")
print(f"  con v_pref = velocidad actual, el programa lineal da v_A' = {np.round(vA_new, 4)}, v_B' = {np.round(vB_new, 4)}")
vr2 = vA_new - vB_new
ts_ = np.linspace(0, tau, 20001)
dist = np.linalg.norm(p[None] - ts_[:, None] * vr2[None], axis=1)
print(f"  nueva velocidad relativa {np.round(vr2, 4)}; distancia mínima entre centros en [0, tau]: {dist.min():.4f} "
      f"(R = {Rr}) en t = {ts_[dist.argmin()]:.3f}")
print(f"  cambio de velocidad de cada dron: |u|/2 = {np.linalg.norm(u)/2:.4f} m/s")
# figura del espacio de velocidades: v en [-0.5, 3.0] x [-1.0, 1.5] -> x [40, 390], y [260, 10] (100 px por m/s)
VX, VY = lin(-0.5, 3.0, 40, 390), lin(-1.0, 1.5, 260, 10)
print(f"  figura: origen ({VX(0):.1f}, {VY(0):.1f}); centro disco ({VX(cen[0]):.1f}, {VY(cen[1]):.1f}) radio {rad*100:.1f}")
t0 = math.sqrt(np.linalg.norm(cen) ** 2 - rad ** 2)
for sgn in (+1, -1):
    dirp = np.array([math.cos(alpha + sgn * beta), math.sin(alpha + sgn * beta)])
    a_, b_ = t0 * dirp, 3.2 * dirp
    print(f"   pata {sgn:+d}: de ({VX(a_[0]):.1f}, {VY(a_[1]):.1f}) a ({VX(b_[0]):.1f}, {VY(b_[1]):.1f})")
# figura del semiplano en el espacio de velocidades de A: 300 px por m/s, v_A = (1, 0) en (200, 100)
AX, AY = (lambda v: 200 + (v - 1.0) * 300), (lambda v: 100 - v * 300)
pt = vA + u / 2
dl = np.array([n[1], -n[0]])         # dirección de la recta (perpendicular a n)
print(f"  figura semiplano: v_A ({AX(vA[0]):.1f}, {AY(vA[1]):.1f}); v_A + u/2 ({AX(pt[0]):.1f}, {AY(pt[1]):.1f})")
for s_ in (-0.63, 0.63):
    e_ = pt + s_ * dl
    print(f"   extremo de la recta ({AX(e_[0]):.1f}, {AY(e_[1]):.1f})")
e_ = pt + 0.13 * n
print(f"   flecha n hasta ({AX(e_[0]):.1f}, {AY(e_[1]):.1f}); lado permitido: hacia n")
print(f"   v_rel ({VX(vrel[0]):.1f}, {VY(vrel[1]):.1f}); punto más cercano ({VX(qmin[0]):.1f}, {VY(qmin[1]):.1f})")

# ---------------------------------------------------------------------------
titulo(5, "CBBA simplificado: 2 drones y 3 tareas en una línea (SUPUESTO)")
pos_dron = {"A": 0.0, "B": 10.0}
pos_tarea = {"T1": 4.0, "T2": 7.0, "T3": 9.0}
RECOMP, DESC, LT = 10.0, 0.9, 3       # recompensa, descuento por unidad de distancia, tamaño máximo del lote


def puntuacion_lote(dron, lote):
    """Suma de 10 * 0,9^(distancia recorrida al llegar), visitando las tareas en el orden del lote."""
    x, dist, tot = pos_dron[dron], 0.0, 0.0
    for t in lote:
        dist += abs(pos_tarea[t] - x)
        x = pos_tarea[t]
        tot += RECOMP * DESC ** dist
    return tot


def marginal(dron, lote, t):
    return puntuacion_lote(dron, lote + [t]) - puntuacion_lote(dron, lote)


y = {t: 0.0 for t in pos_tarea}      # puja ganadora conocida
z = {t: None for t in pos_tarea}     # ganador conocido
lotes = {"A": [], "B": []}
pujas = {"A": {}, "B": {}}
for ronda in (1, 2):
    print(f"\n  RONDA {ronda} · fase 1 (construir el lote), con las pujas ganadoras conocidas {dict((k, round(v, 2)) for k, v in y.items())}")
    for dron in ("A", "B"):
        while len(lotes[dron]) < LT:
            libres = [t for t in pos_tarea if t not in lotes[dron]]
            c = {t: marginal(dron, lotes[dron], t) for t in libres}
            print(f"   {dron}, lote {lotes[dron]}: puntuaciones marginales " +
                  ", ".join(f"{t} {c[t]:.2f}" + ("" if c[t] > y[t] or z[t] == dron else f" (no supera {y[t]:.2f})") for t in libres))
            validas = {t: v for t, v in c.items() if v > y[t] and z[t] != dron}
            if not validas:
                break
            t = max(validas, key=validas.get)
            lotes[dron].append(t)
            pujas[dron][t] = c[t]
        print(f"   {dron}: lote {lotes[dron]} con pujas {dict((k, round(v, 2)) for k, v in pujas[dron].items())}")
    print(f"  RONDA {ronda} · fase 2 (consenso): se intercambian pujas y gana la más alta en cada tarea")
    for t in pos_tarea:
        ofertas = {d: pujas[d][t] for d in ("A", "B") if t in pujas[d]}
        if ofertas:
            g_ = max(ofertas, key=ofertas.get)
            if ofertas[g_] > y[t]:
                y[t], z[t] = ofertas[g_], g_
        print(f"   {t}: ofertas {dict((k, round(v, 2)) for k, v in ofertas.items())} -> gana {z[t]} con {y[t]:.2f}")
    for dron in ("A", "B"):
        for k, t in enumerate(lotes[dron]):
            if z[t] != dron:
                soltadas = lotes[dron][k:]
                print(f"   {dron} pierde {t}: suelta {soltadas} (lo que iba detrás se puntuó suponiendo que pasaba por {t})")
                lotes[dron] = lotes[dron][:k]
                for s_ in soltadas:
                    pujas[dron].pop(s_, None)
                break
    print(f"   lotes tras el consenso: {lotes}")
total = puntuacion_lote("A", lotes["A"]) + puntuacion_lote("B", lotes["B"])
print(f"\n  asignación final {lotes}, puntuación total {total:.3f}")
mejor = None
for asign in itertools.product(("A", "B"), repeat=3):
    tA = [t for t, d in zip(pos_tarea, asign) if d == "A"]
    tB = [t for t, d in zip(pos_tarea, asign) if d == "B"]
    sA = max(puntuacion_lote("A", list(o)) for o in itertools.permutations(tA)) if tA else 0.0
    sB = max(puntuacion_lote("B", list(o)) for o in itertools.permutations(tB)) if tB else 0.0
    if mejor is None or sA + sB > mejor[0]:
        mejor = (sA + sB, tA, tB)
print(f"  óptimo por fuerza bruta (8 repartos, mejor orden en cada uno): A {mejor[1]}, B {mejor[2]}, total {mejor[0]:.3f}")
print(f"  todo para A en orden T1, T2, T3: {puntuacion_lote('A', ['T1','T2','T3']):.3f}; todo para B en orden T3, T2, T1: "
      f"{puntuacion_lote('B', ['T3','T2','T1']):.3f}")

# ---------------------------------------------------------------------------
titulo(6, "Other-play: palancas con etiquetas permutadas")


def juego(pagos):
    """pagos: lista de pagos por palanca. Simetrías: permutaciones de las palancas con el mismo pago."""
    k = len(pagos)
    grupos = {}
    for i, v in enumerate(pagos):
        grupos.setdefault(v, []).append(i)
    # todas las permutaciones que solo mezclan palancas del mismo pago (se muestrean al azar si son muchas)
    return k, grupos


for pagos in ([1.0, 1.0, 0.9], [1.0] * 9 + [0.9]):
    k, grupos = juego(pagos)
    print(f"\n  {k} palancas, pagos {pagos}")
    # J(a, b) = pago si eligen la misma palanca, 0 si no
    J = lambda a, b: pagos[a] if a == b else 0.0
    # autojuego: cada política determinista 'elige la palanca a', jugando consigo misma
    print("   autojuego J(a, a): " + ", ".join(f"palanca {a}: {J(a, a)}" for a in range(k)))
    # juego cruzado: dos semillas que han convergido a una palanca de pago 1 al azar
    unos = grupos[1.0]
    xp = np.mean([J(a, b) for a in unos for b in unos])
    print(f"   juego cruzado entre semillas que eligen al azar una de las {len(unos)} palancas de 1: {xp:.4f} = 1/{len(unos)}")
    # other-play: E_phi J(a, phi(a)), phi uniforme en las permutaciones dentro de cada grupo de pago
    rng = np.random.default_rng(1)
    for a in (unos[0], grupos[0.9][0]):
        vals = []
        for _ in range(20000):
            phi = list(range(k))
            for g_ in grupos.values():
                perm = rng.permutation(g_)
                for src, dst in zip(g_, perm):
                    phi[src] = dst
            vals.append(J(a, phi[a]))
        exacto = pagos[a] / len(grupos[pagos[a]])
        print(f"   other-play J_OP(palanca {a}, pago {pagos[a]}) = pago / tamaño del grupo = {exacto:.4f}; Monte Carlo {np.mean(vals):.4f}")
# 3 palancas: tabla de las dos simetrías (identidad y cambiar 0 <-> 1)
print("\n  3 palancas, tabla J(a, phi(a)) para phi = identidad y phi = intercambiar 0 y 1:")
pagos = [1.0, 1.0, 0.9]
for a in range(3):
    ida = pagos[a]
    sw = {0: 1, 1: 0, 2: 2}[a]
    print(f"   palanca {a}: identidad {ida}; intercambio {pagos[a] if sw == a else 0.0}; media {(ida + (pagos[a] if sw == a else 0.0))/2:.2f}")

# ---------------------------------------------------------------------------
titulo(7, "Juego cruzado: medias de la matriz 4 x 4 sin comunicación (copiada de salida_gpu.txt)")
M = np.array([[0.895, 0.891, 0.864, 0.891], [0.891, 0.892, 0.861, 0.891],
              [0.871, 0.877, 0.895, 0.876], [0.891, 0.89, 0.859, 0.893]])
print("  M[i, j] = P(encontrarlo) con el dron 0 de la semilla i y los drones 1 y 2 de la semilla j")
print(M)
diag = np.diag(M)
fuera = M[~np.eye(4, dtype=bool)]
print(f"  diagonal (autojuego): {diag} -> media {diag.mean():.5f}")
print(f"  fuera de la diagonal (12 casillas, juego cruzado): suma {fuera.sum():.3f} -> media {fuera.mean():.5f}; mínimo {fuera.min()}")
print(f"  brecha {diag.mean() - fuera.mean():.4f}")
mask = ~np.eye(4, dtype=bool)
for s in range(4):
    fila = M[s][mask[s]]
    col = M[:, s][mask[:, s]]
    print(f"  semilla {s}: dron 0 con otras (fila) {fila.mean():.4f}; drones 1-2 con otros (columna) {col.mean():.4f}")
sin2 = M[np.ix_([0, 1, 3], [0, 1, 3])]
print(f"  sin la semilla 2: juego cruzado {sin2[~np.eye(3, dtype=bool)].mean():.4f}")
print(f"  parejas posibles: 4 x 4 = 16; autojuego 4; juego cruzado 12")

# ---------------------------------------------------------------------------
titulo(8, "El mentiroso: mensajes sumados frente a mensajes por emisor (SUPUESTO)")
q = 0.3
p0 = {"X": 0.5, "Y": 0.3, "resto": 0.2}
print(f"  mapa previo p0 = {p0}; q = {q} (una mirada que no encuentra multiplica la masa por q)")


def creencia(n):
    m = {c: p0[c] * q ** n.get(c, 0) for c in p0}
    s = sum(m.values())
    return m, {c: m[c] / s for c in m}


mundos = {
    "las dos miradas son reales": {"X": 1, "Y": 1},
    "la mirada a X es falsa (el dron 2 miente)": {"Y": 1},
    "la mirada a Y es falsa": {"X": 1},
}
print("  el dron 0 recibe la suma de lo anunciado por los drones 1 y 2: X 1 mirada, Y 1 mirada (igual en los tres mundos)")
for nombre, n in mundos.items():
    m, b = creencia(n)
    print(f"   {nombre}: masas {dict((k, round(v, 4)) for k, v in m.items())}, P(encontrarlo antes) = {1 - sum(m.values()):.4f}, "
          f"belief {dict((k, round(v, 3)) for k, v in b.items())}")
m_creido, b_creido = creencia({"X": 1, "Y": 1})
m_real, b_real = creencia({"Y": 1})
print(f"  si el dron 0 cree el mapa sumado, ve X con {b_creido['X']:.3f}; la verdad (X falsa) es {b_real['X']:.3f}")
print(f"  mirar X ahora rendiría de verdad m(X)(1 - q) = {m_real['X'] * (1 - q):.4f}; según lo creído, {m_creido['X'] * (1 - q):.4f}")
print("  con un mapa por emisor: dron 1 -> Y; dron 2 -> X, pero la posición anunciada del dron 2 no está en X:"
      " la mirada es incompatible y se puede descartar")
print(f"  notación: mirar dos veces una celda de 0,2 deja m = 0,2 q^2 = {0.2 * q**2:.3f}; una mirada a una celda de 0,2"
      f" la detecta con probabilidad r = 0,2 (1 - q) = {0.2 * (1 - q):.2f}")
