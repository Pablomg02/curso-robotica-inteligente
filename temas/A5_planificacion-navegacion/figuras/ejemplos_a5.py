"""Ejemplos numéricos de las hojas del tema A5 (planificación y navegación), para seguir a mano.

Uso:  python3 ejemplos_a5.py > salida_ejemplos.txt
Requiere numpy; corre en CPU en unos segundos. Cada ejemplo lleva su número, que es el que citan
las hojas («ejemplo 4 del script»). Los trazados SVG se imprimen como 'path etiqueta: M...'
escalados al viewBox de la figura correspondiente.

El mapa, los waypoints y las funciones de minimum snap y Dubins son los de figuras_a5.py
(copiados aquí para no relanzar RRT y RRT*, que son lo lento de aquel script).
"""
import heapq
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)
G = 9.81
MASA = 2.064          # x500 de A1


def cab(n, titulo):
    print("=" * 72)
    print(f"EJEMPLO {n}. {titulo}")


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


# ---------------------------------------------------------------- mapa de figuras_a5.py
OBS = [((5.0, 3.0), 1.8), ((9.0, 7.0), 2.0), ((13.0, 3.5), 2.2), ((16.5, 7.5), 1.5)]
START, GOAL = np.array([1.0, 1.0]), np.array([19.0, 9.0])
OC = np.array([c for c, r in OBS]); ORAD = np.array([r for c, r in OBS])


def libres(P, m=0.3):
    P = np.atleast_2d(P)
    d = np.linalg.norm(P[:, None, :] - OC[None], axis=2) - ORAD[None] - m
    return (d.min(axis=1) > 0) & (P[:, 0] >= 0) & (P[:, 0] <= 20) & (P[:, 1] >= 0) & (P[:, 1] <= 10)


# =====================================================================================
cab(1, "A* A MANO EN UNA REJILLA 4-CONEXA DE 6 x 4 (coste 1 por paso, heurística Manhattan)")
# columnas x = 0..5, filas y = 0..3 (y hacia arriba). Muro en x = 2 y x = 3 para y = 0..2.
NX, NY = 6, 4
MURO = {(2, 0), (2, 1), (2, 2), (3, 1), (3, 2)}
S, GL = (0, 1), (5, 1)


def busca(hfun, eps=1.0, verbose=False):
    """A* con desempate: menor f, después mayor g, después orden de inserción."""
    cnt = 0
    openq = [(eps * hfun(S), 0, cnt, S)]
    g = {S: 0}; came = {S: None}; cerrados = []; orden = []
    while openq:
        f, mg, _, u = heapq.heappop(openq)
        if u in cerrados:
            continue
        cerrados.append(u)
        orden.append((u, g[u], hfun(u), g[u] + eps * hfun(u)))
        if u == GL:
            break
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            v = (u[0] + dx, u[1] + dy)
            if not (0 <= v[0] < NX and 0 <= v[1] < NY) or v in MURO:
                continue
            ng = g[u] + 1
            if ng < g.get(v, 99):
                g[v] = ng; came[v] = u; cnt += 1
                heapq.heappush(openq, (ng + eps * hfun(v), -ng, cnt, v))
    p = [GL]
    while came[p[-1]] is not None:
        p.append(came[p[-1]])
    return orden, p[::-1], g


manh = lambda a: abs(a[0] - GL[0]) + abs(a[1] - GL[1])
orden, camino, gval = busca(manh)
print("  muro:", sorted(MURO), " inicio", S, " meta", GL)
print("  orden de expansión de A* (celda, g, h, f):")
for k, (u, g_, h_, f_) in enumerate(orden, 1):
    print(f"    {k:2d}. {u}  g={g_}  h={h_}  f={f_}")
print(f"  A*: {len(orden)} celdas expandidas; camino {camino}, coste {len(camino)-1}")
ordD, camD, _ = busca(lambda a: 0)
print(f"  Dijkstra (h = 0): {len(ordD)} celdas expandidas, coste {len(camD)-1}")
libres_tot = NX * NY - len(MURO)
print(f"  celdas libres: {libres_tot}")
print("  h de cada celda libre (Manhattan a la meta):")
for y in range(NY - 1, -1, -1):
    print("    " + " ".join(("##" if (x, y) in MURO else f"{manh((x, y)):2d}") for x in range(NX)))
print("  g final (coste desde el inicio) de las celdas tocadas por A*:")
for y in range(NY - 1, -1, -1):
    print("    " + " ".join(("##" if (x, y) in MURO else (f"{gval[(x,y)]:2d}" if (x, y) in gval else " .")) for x in range(NX)))

# =====================================================================================
cab(2, "ADMISIBILIDAD Y CONSISTENCIA: Manhattan en una rejilla 8-conexa sobreestima")
print(f"  vecino en diagonal de la meta: h Manhattan = 2, coste real h* = sqrt(2) = {math.sqrt(2):.3f} -> no admisible")
print(f"  euclídea: h = sqrt(2) = h*; consistente porque |n - meta| <= |n - n'| + |n' - meta| (desigualdad triangular)")
print(f"  octile (8-conexa): h = max(dx,dy) + (sqrt2 - 1) min(dx,dy); para (3, 1): {3 + (math.sqrt(2)-1)*1:.3f} = coste exacto sin obstáculos")

# =====================================================================================
cab(3, "DIJKSTRA, A* Y A* INFLADO EN EL MAPA DE LA PRESENTACIÓN (rejilla 8-conexa de 0,1 m)")
res = 0.1
nx, ny = int(20 / res) + 1, int(10 / res) + 1
gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
occ = ~libres(np.stack([gi.ravel() * res, gj.ravel() * res], axis=1)).reshape(nx, ny)


def astar_mapa(eps):
    s = (int(START[0] / res), int(START[1] / res)); g = (int(GOAL[0] / res), int(GOAL[1] / res))
    h = lambda a: res * math.hypot(a[0] - g[0], a[1] - g[1])
    openq = [(eps * h(s), 0.0, s)]; cost = {s: 0.0}; exp = 0   # mismo recuento que figuras_a5.py
    while openq:
        _, c, u = heapq.heappop(openq)
        if c > cost[u] + 1e-12:
            continue
        exp += 1
        if u == g:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0:
                    continue
                v = (u[0] + di, u[1] + dj)
                if not (0 <= v[0] < nx and 0 <= v[1] < ny) or occ[v]:
                    continue
                nc = c + res * math.hypot(di, dj)
                if nc < cost.get(v, 1e18):
                    cost[v] = nc; heapq.heappush(openq, (nc + eps * h(v), nc, v))
    return cost[g], exp


print(f"  celdas: {nx*ny}, libres: {int((~occ).sum())}")
for nm, e in (("Dijkstra (eps = 0)", 0.0), ("A* (eps = 1)", 1.0), ("A* inflado (eps = 1,5)", 1.5), ("A* inflado (eps = 3)", 3.0)):
    c, ex = astar_mapa(e)
    print(f"  {nm:24s}: longitud {c:.2f} m, {ex} nodos expandidos")

# =====================================================================================
cab(4, "CUÁNTO ALARGA UNA REJILLA 8-CONEXA")
th = np.linspace(0, math.pi / 4, 100001)
r = np.cos(th) + (math.sqrt(2) - 1) * np.sin(th)
print(f"  cociente máximo {r.max():.4f} en {math.degrees(th[r.argmax()]):.2f}°; tan(22,5°) = {math.tan(math.pi/8):.4f} = sqrt2 - 1 = {math.sqrt(2)-1:.4f};"
      f" valor exacto sqrt(4 - 2 sqrt2) = {math.sqrt(4-2*math.sqrt(2)):.4f}")
dx_, dy_ = 10, 4
cg = dx_ + (math.sqrt(2) - 1) * dy_; cr = math.hypot(dx_, dy_)
print(f"  desplazamiento ({dx_}, {dy_}) celdas: rejilla {dy_} diagonales + {dx_-dy_} rectas = {cg:.3f}; recta {cr:.3f}; cociente {cg/cr:.4f}; ángulo {math.degrees(math.atan2(dy_, dx_)):.1f}°")
print(f"  en el mapa: A* 21,65 m frente a 21,07 m recortado -> {(21.65/21.07-1)*100:.1f} %")

# =====================================================================================
cab(5, "UNA EXTENSIÓN DE RRT A MANO (paso eta = 1 m)")
arbol = np.array([[1.0, 1.0], [2.0, 1.5], [1.5, 2.5]])
xr = np.array([4.0, 4.0])
d = np.linalg.norm(arbol - xr, axis=1)
k = int(np.argmin(d))
xn = arbol[k] + (xr - arbol[k]) * min(1.0, 1.0 / d[k])
print(f"  árbol {arbol.tolist()}; muestra x_rand = {xr.tolist()}")
print(f"  distancias a x_rand: {np.round(d, 3).tolist()} -> más cercano x_near = {arbol[k].tolist()} (nodo {k})")
print(f"  x_new = x_near + eta (x_rand - x_near)/|x_rand - x_near| = {np.round(xn, 3).tolist()}; libre: {bool(libres(xn)[0])}")
X5 = lin(0, 5, 30, 330); Y5 = lin(0, 5, 280, 30)
for p in arbol:
    print(f"   px nodo ({X5(p[0]):.1f},{Y5(p[1]):.1f})")
print(f"   px x_rand ({X5(xr[0]):.1f},{Y5(xr[1]):.1f}); x_new ({X5(xn[0]):.1f},{Y5(xn[1]):.1f})")

# =====================================================================================
cab(6, "RRT*: ELEGIR PADRE Y RECABLEAR, A MANO")
# nodos con coste desde la raíz (ya en el árbol); el nuevo x_new cae en (3, 2)
nodos = {"A": (np.array([0.0, 0.0]), 0.0), "B": (np.array([2.0, 0.0]), 2.0),
         "C": (np.array([1.0, 2.0]), 2.236), "D": (np.array([4.0, 3.0]), 6.0)}
# C es hijo de A (coste |AC| = 2,236); B hijo de A; D cuelga de B por un rodeo y cuesta 6,0
xnew = np.array([3.0, 2.0])
rad = 2.0
print(f"  x_new = {xnew.tolist()}, radio de vecinos r = {rad}")
vec = {}
for nm, (p, c) in nodos.items():
    dd = np.linalg.norm(p - xnew)
    if dd <= rad + 1e-9:
        vec[nm] = (p, c, dd)
        print(f"   vecino {nm} en {p.tolist()}: coste {c:.3f} + distancia {dd:.3f} = {c+dd:.3f}")
    else:
        print(f"   {nm} en {p.tolist()} a {dd:.3f} > r: no es vecino")
padre = min(vec, key=lambda n: vec[n][1] + vec[n][2])
cnew = vec[padre][1] + vec[padre][2]
print(f"  mejor padre: {padre}, coste de x_new {cnew:.3f}")
for nm, (p, c, dd) in vec.items():
    if nm == padre:
        continue
    alt = cnew + dd
    print(f"   recableado de {nm}: coste actual {c:.3f}, por x_new {cnew:.3f} + {dd:.3f} = {alt:.3f} -> {'RECABLEA' if alt < c else 'se queda'}")
X6 = lin(-0.5, 5, 30, 360); Y6 = lin(-0.5, 3.6, 250, 25)
for nm, (p, c) in nodos.items():
    print(f"   px {nm} ({X6(p[0]):.1f},{Y6(p[1]):.1f})")
print(f"   px x_new ({X6(xnew[0]):.1f},{Y6(xnew[1]):.1f}); radio en px {rad*(330/5.5):.1f}")

# =====================================================================================
cab(7, "RADIO DE CONEXIÓN DE RRT*: r_n = min(2,5; 6 sqrt(log n / n)) en el mapa (d = 2)")
A_libre = 200 - math.pi * sum(r_ ** 2 for _, r_ in OBS)
print(f"  área libre aproximada (sin margen): 200 - pi(1,8² + 2² + 2,2² + 1,5²) = {A_libre:.1f} m²")
for n in (10, 30, 100, 1000, 3000, 10000):
    rn = min(2.5, 6 * math.sqrt(math.log(n) / n))
    vecinos = n * math.pi * rn ** 2 / A_libre
    print(f"  n = {n:5d}: r = {rn:.3f} m; vecinos esperados n·pi r²/A = {vecinos:.2f}; 0,73 log n = {0.73*math.log(n):.2f}")
print(f"  constante: pi·6²/A = {math.pi*36/A_libre:.3f}")

# =====================================================================================
# minimum snap (copiado de figuras_a5.py)
def coef_deriv(k, t, n=8):
    row = np.zeros(n)
    for i in range(k, n):
        row[i] = math.factorial(i) / math.factorial(i - k) * t ** (i - k)
    return row


def Q_snap(T, n=8, r=4):
    Q = np.zeros((n, n))
    for i in range(r, n):
        for j in range(r, n):
            ci = math.factorial(i) / math.factorial(i - r); cj = math.factorial(j) / math.factorial(j - r)
            Q[i, j] = ci * cj / (i + j - 2 * r + 1)
    return Q / T ** (2 * r - 1)


def fila(k, tau, T):
    return coef_deriv(k, tau) / T ** k


def min_snap(wps, Ts, cont=4):
    M = len(Ts); n = 8; N = M * n
    H = np.zeros((N, N))
    for s, T in enumerate(Ts):
        H[s*n:(s+1)*n, s*n:(s+1)*n] = Q_snap(T)
    A, b = [], []
    for s, T in enumerate(Ts):
        r0 = np.zeros(N); r0[s*n:(s+1)*n] = fila(0, 0.0, T); A.append(r0); b.append(wps[s])
        r1 = np.zeros(N); r1[s*n:(s+1)*n] = fila(0, 1.0, T); A.append(r1); b.append(wps[s+1])
        if s < M - 1:
            for k in range(1, cont + 1):
                r = np.zeros(N); r[s*n:(s+1)*n] = fila(k, 1.0, T); r[(s+1)*n:(s+2)*n] = -fila(k, 0.0, Ts[s+1])
                A.append(r); b.append(0.0)
    for k in (1, 2, 3):
        r = np.zeros(N); r[0:n] = fila(k, 0.0, Ts[0]); A.append(r); b.append(0.0)
        r = np.zeros(N); r[(M-1)*n:M*n] = fila(k, 1.0, Ts[-1]); A.append(r); b.append(0.0)
    A = np.array(A); b = np.array(b)
    KKT = np.block([[2 * H, A.T], [A, np.zeros((len(b), len(b)))]])
    sol = np.linalg.solve(KKT, np.concatenate([np.zeros(N), b]))
    c = sol[:N].reshape(M, n)
    J = c.reshape(-1) @ H @ c.reshape(-1)
    return c, J, A, KKT


def evalua(c, Ts, k, npts=60):
    ts, vs = [], []
    t0 = 0.0
    for s, T in enumerate(Ts):
        for tau in np.linspace(0, 1, npts, endpoint=(s == len(Ts) - 1)):
            ts.append(t0 + tau * T); vs.append(fila(k, tau, T) @ c[s])
        t0 += T
    return np.array(ts), np.array(vs)


# =====================================================================================
cab(8, "PLANITUD SOBRE LA TRAYECTORIA DE LA FIGURA (T = 8 s, 2 s por tramo, z constante, psi = 0)")
wx = np.array([0.0, 4.0, 8.0, 12.0, 16.0]); wy = np.array([0.0, 3.0, 0.0, -3.0, 0.0])
dist = np.hypot(np.diff(wx), np.diff(wy))
for nombre, Ts in (("2 s por tramo", dist / dist.sum() * 8.0), ("reparto 2,87 · 1,13 · 1,13 · 2,87", np.array([2.87, 1.13, 1.13, 2.87]))):
    cx, _, _, _ = min_snap(wx, Ts); cy, _, _, _ = min_snap(wy, Ts)
    tt, ax = evalua(cx, Ts, 2, 400); _, ay = evalua(cy, Ts, 2, 400)
    _, jx = evalua(cx, Ts, 3, 400); _, jy = evalua(cy, Ts, 3, 400)
    acc = np.stack([ax, ay, np.zeros_like(ax)], 1); jer = np.stack([jx, jy, np.zeros_like(jx)], 1)
    tvec = acc + np.array([0, 0, G]); tn = np.linalg.norm(tvec, axis=1)
    zb = tvec / tn[:, None]
    incl = np.degrees(np.arccos(zb[:, 2]))
    f = MASA * tn
    hw = (MASA / f)[:, None] * (jer - np.sum(zb * jer, 1)[:, None] * zb)
    xc = np.array([1.0, 0, 0])
    yb = np.cross(zb, xc); yb /= np.linalg.norm(yb, axis=1)[:, None]; xb = np.cross(yb, zb)
    wxb = -np.sum(hw * yb, 1); wyb = np.sum(hw * xb, 1)
    k = int(np.argmax(incl))
    print(f"  {nombre}: inclinación máx. {incl.max():.1f}° en t = {tt[k]:.2f} s (|a| = {np.linalg.norm(acc[k]):.2f} m/s², empuje {f[k]:.1f} N);"
          f" |omega| máx. {np.max(np.hypot(wxb, wyb)):.2f} rad/s; empuje de 20,25 N a {f.max():.1f} N")
    if nombre.startswith("2 s"):
        k0 = int(np.argmin(abs(tt - 0.5)))
        print(f"   en t = {tt[k0]:.2f} s: a = ({ax[k0]:.3f}, {ay[k0]:.3f}) m/s², j = ({jx[k0]:.3f}, {jy[k0]:.3f}) m/s³; t = a + g e3 = "
              f"({tvec[k0,0]:.3f}, {tvec[k0,1]:.3f}, {tvec[k0,2]:.2f}); |t| = {tn[k0]:.3f}; f = {f[k0]:.2f} N; z_B = {np.round(zb[k0], 4).tolist()};"
              f" inclinación {incl[k0]:.1f}°; h_w = {np.round(hw[k0], 4).tolist()}; omega_x = {wxb[k0]:.4f}, omega_y = {wyb[k0]:.4f} rad/s")
        Xp = lin(0, 8, 60, 500); Yp = lin(0, 35, 200, 20)
        print("path inclinacion:", path(tt[::4], incl[::4], Xp, Yp))
        print(f"   inclinación en px: 10° -> {Yp(10):.1f}, 20° -> {Yp(20):.1f}, 30° -> {Yp(30):.1f}")
        print(f"   inclinación máx. arctan(5,96/9,81) = {math.degrees(math.atan(5.96/9.81)):.1f}°")

# =====================================================================================
cab(9, "LA MATRIZ Q DE UN TRAMO (en t, sin normalizar): Q_jk = j!/(j-4)! k!/(k-4)! T^(j+k-7)/(j+k-7)")
def Q_t(T):
    Q = np.zeros((8, 8))
    for j in range(4, 8):
        for k in range(4, 8):
            Q[j, k] = math.factorial(j) / math.factorial(j - 4) * math.factorial(k) / math.factorial(k - 4) * T ** (j + k - 7) / (j + k - 7)
    return Q
Q1 = Q_t(1.0)
print("  T = 1 s, bloque j, k = 4..7:")
print(Q1[4:, 4:])
print(f"  comprobación: p(t) = t^4 en [0, 1] -> p'''' = 24, coste 24² = {24**2}; Q_44 = {Q1[4,4]:.0f}")
c = np.zeros(8); c[4] = 1; c[5] = 1
print(f"  p(t) = t^4 + t^5: p'''' = 24 + 120 t; integral de (24 + 120 t)² = 576 + 2·24·120/2 + 14400/3 = {576 + 24*120 + 14400/3:.0f}; c^T Q c = {c @ Q1 @ c:.0f}")
print(f"  T = 2 s: Q_44 = 576·2 = {Q_t(2.0)[4,4]:.0f}, Q_77 = 840²·2^7/7 = {Q_t(2.0)[7,7]:.0f}")

# =====================================================================================
cab(10, "MINIMUM SNAP CON DOS TRAMOS: waypoints 0 -> 1 -> 0 en un eje, T = (1, 1) s, reposo en los extremos")
w2 = np.array([0.0, 1.0, 0.0]); T2 = np.array([1.0, 1.0])
c2, J2, A2, K2 = min_snap(w2, T2)
print(f"  incógnitas: 16 coeficientes; restricciones: {A2.shape[0]} (4 de posición, 4 de continuidad v, a, j, s en el centro, 6 de reposo);"
      f" matriz KKT {K2.shape[0]} x {K2.shape[1]}")
tt, pp = evalua(c2, T2, 0, 200)
for kk, nm in ((1, "v"), (2, "a"), (3, "j"), (4, "s"), (5, "d5"), (6, "d6")):
    izq = fila(kk, 1.0, T2[0]) @ c2[0]; der = fila(kk, 0.0, T2[1]) @ c2[1]
    print(f"   {nm} en el waypoint central: izquierda {izq:+.4f}, derecha {der:+.4f}")
print(f"  coste óptimo J = {J2:.2f}")
# Forma de Richter: un tramo de (0,0,0,0) a (1, v, a, j); coeficientes lineales en las derivadas
def tramo(d0, d1, T):
    """Coeficientes (en tau) de un grado 7 con derivadas 0..3 dadas en los dos extremos."""
    A = np.array([fila(k, 0.0, T) for k in range(4)] + [fila(k, 1.0, T) for k in range(4)])
    return np.linalg.solve(A, np.concatenate([d0, d1]))
def J_a(a, v=0.0, j=0.0):
    c1 = tramo(np.zeros(4), np.array([1.0, v, a, j]), 1.0)
    c2_ = tramo(np.array([1.0, v, a, j]), np.zeros(4), 1.0)
    return c1 @ Q_snap(1.0) @ c1 + c2_ @ Q_snap(1.0) @ c2_
J0, Jp, Jm = J_a(0.0), J_a(1.0), J_a(-1.0)
gam = (Jp + Jm - 2 * J0) / 2; bet = (Jp - Jm) / 2
print(f"  con simetría (v = j = 0 en el centro): J(a) = {J0:.2f} + {bet:.2f} a + {gam:.4f} a²")
a_opt = -bet / (2 * gam)
print(f"  dJ/da = 0 -> a* = {a_opt:.4f} m/s²; J(a*) = {J_a(a_opt):.2f}; KKT: a en el centro = {fila(2, 1.0, 1.0) @ c2[0]:.4f}")
print(f"  con a = 0: J = {J0:.2f} ({J0/J_a(a_opt):.2f} veces el óptimo)")
c1o = tramo(np.zeros(4), np.array([1.0, 0, a_opt, 0]), 1.0)
print(f"  coeficientes del tramo 1 (p(t) = sum c_j t^j, T = 1): {np.round(c1o, 3).tolist()}")
p0r = np.poly1d([-20, 70, -84, 35, 0, 0, 0, 0]); phi = np.poly1d([-2, 6.5, -7, 2.5, 0, 0, 0, 0])
jq = lambda P, Q_: np.polyint(P.deriv(4) * Q_.deriv(4))(1)
print("  tramo 1 = (35t^4 - 84t^5 + 70t^6 - 20t^7) + a·phi, phi = 2,5t^4 - 7t^5 + 6,5t^6 - 2t^7; derivadas 0..3 de phi en t = 1: "
      + ", ".join(f"{phi.deriv(k)(1) if k else phi(1):.0f}" for k in range(4)) + "; en t = 0: todas 0")
print(f"  costes por tramo: reposo a reposo {jq(p0r, p0r):.0f}; cruzado {jq(p0r, phi):.0f}; phi {jq(phi, phi):.0f}"
      f" -> J(a) = 2(100800 + 2·10080 a + 1200 a²)")
print(f"  velocidad máx. {np.max(abs(evalua(c2, T2, 1, 400)[1])):.3f} m/s; aceleración en t = 0,5: {fila(2, 0.5, 1.0) @ c2[0]:.3f} m/s²")
X10 = lin(0, 2, 50, 330); Y10 = lin(-0.1, 1.1, 150, 20)
print("path dos tramos:", path(tt[::2], pp[::2], X10, Y10))
ta_, aa_ = evalua(c2, T2, 2, 200)
Ya = lin(-10, 7, 268, 178)
print("path dos tramos acel:", path(ta_[::2], aa_[::2], X10, Ya))
print(f"   aceleración: min {aa_.min():.2f}, max {aa_.max():.2f} m/s²; px de 0 -> {Ya(0):.1f}; de -8,4 -> {Ya(-8.4):.1f}; de 5 -> {Ya(5):.1f}; t = 1 -> {X10(1):.1f}; t = 2 -> {X10(2):.1f}; p = 1 -> {Y10(1):.1f}; p = 0 -> {Y10(0):.1f}")
# sin la simetría: búsqueda en (v, a, j)
from itertools import product
mejor = min(product(np.linspace(-1, 1, 21), np.linspace(-40, 0, 81), np.linspace(-20, 20, 41)), key=lambda x: J_a(x[1], x[0], x[2]))
print(f"  búsqueda en rejilla sobre (v, a, j): mínimo en v = {mejor[0]:.2f}, a = {mejor[1]:.2f}, j = {mejor[2]:.2f} (simetría confirmada)")

# =====================================================================================
cab(11, "FORMA DE RICHTER EN EL EJEMPLO DE DOS TRAMOS: d_P* = -R_PP^-1 R_FP^T d_F")
# vector sin repetir: [p0 v0 a0 j0 | p1 | v1 a1 j1 | p2 v2 a2 j2]; F = todo menos (v1, a1, j1)
def Ainv(T):
    A = np.array([fila(k, 0.0, T) for k in range(4)] + [fila(k, 1.0, T) for k in range(4)])
    return np.linalg.inv(A)
Rb = []
for T in T2:
    Ai = Ainv(T); Rb.append(Ai.T @ Q_snap(T) @ Ai)
# C: d_1 = [p0 v0 a0 j0 p1 v1 a1 j1], d_2 = [p1 v1 a1 j1 p2 v2 a2 j2]
idx_glob = ["p0", "v0", "a0", "j0", "p1", "v1", "a1", "j1", "p2", "v2", "a2", "j2"]
d1 = ["p0", "v0", "a0", "j0", "p1", "v1", "a1", "j1"]; d2 = ["p1", "v1", "a1", "j1", "p2", "v2", "a2", "j2"]
Rg = np.zeros((12, 12))
for Rb_, dd in zip(Rb, (d1, d2)):
    Cm = np.zeros((8, 12))
    for i, nm in enumerate(dd):
        Cm[i, idx_glob.index(nm)] = 1
    Rg += Cm.T @ Rb_ @ Cm
P = [idx_glob.index(x) for x in ("v1", "a1", "j1")]; F = [i for i in range(12) if i not in P]
dF = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0.0])
RPP = Rg[np.ix_(P, P)]; RFP = Rg[np.ix_(F, P)]
dP = -np.linalg.solve(RPP, RFP.T @ dF)
print(f"  d_F (p0 v0 a0 j0 p1 p2 v2 a2 j2) = {dF.tolist()}")
print(f"  R_PP =\n{RPP}")
print(f"  R_FP^T d_F = {RFP.T @ dF}")
print(f"  d_P* = (v1, a1, j1) = {np.round(dP, 4).tolist()}  (KKT dio a = {fila(2, 1.0, 1.0) @ c2[0]:.4f})")
dall = np.zeros(12); dall[F] = dF; dall[P] = dP
print(f"  J = d^T R d = {dall @ Rg @ dall:.2f}")
print(f"  tamaño: 9 fijadas + 3 libres; un sistema 3 x 3 frente al KKT de 30 x 30")

# =====================================================================================
cab(12, "LEY DE ESCALA EN EL TIEMPO")
for a in (0.5, 2.0):
    _, Ja, _, _ = min_snap(w2, T2 * a)
    print(f"  dos tramos con T = ({a}, {a}) s: J = {Ja:.2f}; J(1)/alpha^7 = {J2/a**7:.2f}")
print(f"  figura de la hoja: 8483 / 66,3 = {8483/66.3:.1f}; 2^7 = 128; 1132 · (8/6)^7 = {1132*(8/6)**7:.0f}")
print(f"  v: 6,62 / 3,31 = {6.62/3.31:.2f}; a: 10,60 / 2,65 = {10.60/2.65:.2f}")
print(f"  T para a_max = 5 m/s² desde T = 8 s y a_max = 5,96: 8 sqrt(5,96/5) = {8*math.sqrt(5.96/5):.2f} s")
print(f"  T para inclinación de 30° (a = g tan 30° = {G*math.tan(math.radians(30)):.2f}): 8 sqrt(5,96/5,66) = {8*math.sqrt(5.96/(G*math.tan(math.radians(30)))):.2f} s")

# =====================================================================================
cab(13, "BERNSTEIN: UNA CÚBICA DENTRO DE SU ENVOLVENTE CONVEXA")
q = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]])
def bern(tau, n=3):
    return np.array([math.comb(n, k) * tau**k * (1 - tau)**(n - k) for k in range(n + 1)])
for tau in (0.0, 0.25, 0.5, 1.0):
    b = bern(tau)
    print(f"  tau = {tau}: pesos {np.round(b, 4).tolist()} (suma {b.sum():.0f}) -> punto {np.round(b @ q, 4).tolist()}")
print(f"  derivada: puntos de control 3(q_k+1 - q_k) = {(3*np.diff(q, axis=0)).tolist()}; en tau = 0,5: {np.round(bern(0.5, 2) @ (3*np.diff(q, axis=0)), 4).tolist()}")
print(f"  restricción y <= 2 en los puntos de control -> curva: y máx. = {max(bern(t) @ q[:,1] for t in np.linspace(0,1,1001)):.3f} (<= 2, conservador: el máximo real es 1,5)")
X13 = lin(-0.5, 4.5, 30, 330); Y13 = lin(-0.5, 2.6, 230, 30)
ts13 = np.linspace(0, 1, 41)
pts = np.array([bern(t) @ q for t in ts13])
print("path bernstein:", path(pts[:, 0], pts[:, 1], X13, Y13))
print("   px puntos de control:", [(round(X13(a), 1), round(Y13(b), 1)) for a, b in q])
print(f"   px punto tau = 0,5: ({X13(2.0):.1f},{Y13(1.5):.1f}); y = 2 -> {Y13(2):.1f}; y = 0 -> {Y13(0):.1f}")

# =====================================================================================
cab(14, "MINCO: EL SISTEMA DE BANDA Y LOS GRADIENTES (s = 4, mismo problema de dos tramos con el punto central q libre)")
def minco(q_, Ts):
    """Sistema M c = b para s = 4: condiciones de inicio (4), por punto interior 2s = 8, de final (4)."""
    wps = np.array([0.0, q_, 0.0]); Mn = len(Ts); n = 8; N = Mn * n
    rows, b = [], []
    for k in range(4):
        r = np.zeros(N); r[0:n] = fila(k, 0.0, Ts[0]); rows.append(r); b.append(0.0)
    for s in range(Mn - 1):
        r = np.zeros(N); r[s*n:(s+1)*n] = fila(0, 1.0, Ts[s]); rows.append(r); b.append(wps[s+1])
        for k in range(0, 7):
            r = np.zeros(N); r[s*n:(s+1)*n] = fila(k, 1.0, Ts[s]); r[(s+1)*n:(s+2)*n] = -fila(k, 0.0, Ts[s+1])
            rows.append(r); b.append(0.0)
    for k in range(4):
        r = np.zeros(N); r[(Mn-1)*n:] = fila(k, 1.0, Ts[-1]); rows.append(r); b.append(wps[-1] if k == 0 else 0.0)
    Mm = np.array(rows); b = np.array(b)
    c = np.linalg.solve(Mm, b)
    H = np.zeros((N, N))
    for s, T in enumerate(Ts):
        H[s*n:(s+1)*n, s*n:(s+1)*n] = Q_snap(T)
    return c, c @ H @ c, Mm, b, H
cm, Jm_, Mm, bm, Hm = minco(1.0, T2)
print(f"  M es {Mm.shape[0]} x {Mm.shape[1]} (cuadrada: 4 + 8·(M-1) + 4 = 8M); J = {Jm_:.2f} (igual que el ejemplo 10: {J2:.2f})")
print(f"  diferencia de coeficientes con el KKT: {np.max(abs(cm - c2.reshape(-1))):.1e}")
# gradiente respecto a q por el método adjunto: J = c^T H c, M c = b(q), db/dq = e (fila del waypoint)
G_adj = np.linalg.solve(Mm.T, 2 * Hm @ cm)
dbdq = np.zeros(len(bm)); dbdq[4] = 1.0
dJdq = G_adj @ dbdq
h = 1e-5
fd = (minco(1.0 + h, T2)[1] - minco(1.0 - h, T2)[1]) / (2 * h)
print(f"  dJ/dq adjunto = {dJdq:.3f}; diferencias finitas = {fd:.3f}; como J = kappa q², kappa = {Jm_:.2f} y dJ/dq = 2 kappa q = {2*Jm_:.2f}")
for i in range(2):
    Tp = T2.copy(); Tp[i] += h; Tm = T2.copy(); Tm[i] -= h
    gi_ = (minco(1.0, Tp)[1] - minco(1.0, Tm)[1]) / (2 * h)
    print(f"  dJ/dT_{i+1} (diferencias finitas) = {gi_:.2f}")
print(f"  Euler (J homogénea de grado -7 en los T): T1 dJ/dT1 + T2 dJ/dT2 = -7 J = {-7*Jm_:.2f}")
Ta = np.array([1.2, 0.8]); _, Jta, _, _, _ = minco(1.0, Ta)
s_ = 0
for i in range(2):
    Tp = Ta.copy(); Tp[i] += h; Tm = Ta.copy(); Tm[i] -= h
    s_ += Ta[i] * (minco(1.0, Tp)[1] - minco(1.0, Tm)[1]) / (2 * h)
print(f"  con T = (1,2; 0,8): J = {Jta:.2f}; sum T_i dJ/dT_i = {s_:.2f}; -7 J = {-7*Jta:.2f}")
print(f"  con T_i = e^tau_i: dJ/dtau_i = T_i dJ/dT_i")
# patrón de la matriz M para 3 tramos (para la figura)
_, _, M3, _, _ = minco(1.0, np.array([1.0, 1.0, 1.0])) if False else (None, None, None, None, None)
def patron(Mn):
    n = 8; N = Mn * n; rows = []
    for k in range(4):
        rows.append(set(range(0, n)))
    for s in range(Mn - 1):
        for k in range(8):
            rows.append(set(range(s*n, (s+2)*n)) if k > 0 else set(range(s*n, (s+1)*n)))
    for k in range(4):
        rows.append(set(range((Mn-1)*n, N)))
    return rows
pat = patron(3)
print(f"  patrón de M con 3 tramos: {len(pat)} filas x 24 columnas; cada fila toca como mucho 2 tramos (16 columnas)")
for i, rset in enumerate(pat):
    print("   " + "".join("#" if j in rset else "." for j in range(24)))

# =====================================================================================
cab(15, "POR QUÉ UN POLINOMIO NO ES DE TIEMPO MÍNIMO: 10 m de reposo a reposo con |a| <= 5 m/s²")
D, amax = 10.0, 5.0
Tbb = 2 * math.sqrt(D / amax)
tau = np.linspace(0, 1, 100001)
p2 = 420*tau**2 - 1680*tau**3 + 2100*tau**4 - 840*tau**5   # p'' en tau de 35t^4 - 84t^5 + 70t^6 - 20t^7
kmax = np.max(abs(p2))
Tsn = math.sqrt(D * kmax / amax)
print(f"  bang-bang: acelerar a 5 m/s² hasta la mitad y frenar: T = 2 sqrt(D/a) = {Tbb:.3f} s")
print(f"  polinomio de grado 7 de reposo a reposo (único con v, a, j = 0 en los extremos): p = D(35 tau^4 - 84 tau^5 + 70 tau^6 - 20 tau^7)")
print(f"   max |p''(tau)| = {kmax:.3f} D -> |a|max = {kmax:.3f} D / T²; para 5 m/s²: T = sqrt(D·{kmax:.3f}/5) = {Tsn:.3f} s ({(Tsn/Tbb-1)*100:.0f} % más lento)")
print(f"   tau del máximo: {tau[np.argmax(p2)]:.4f}")
Xt = lin(0, Tsn, 60, 500); Ya15 = lin(-6, 6, 220, 40)
ts_ = tau[::1000] * Tsn
print("path acel snap:", path(ts_, p2[::1000] * D / Tsn**2, Xt, Ya15))
print(f"path acel bangbang: M{Xt(0):.1f},{Ya15(0):.1f} L{Xt(0):.1f},{Ya15(5):.1f} L{Xt(Tbb/2):.1f},{Ya15(5):.1f} L{Xt(Tbb/2):.1f},{Ya15(-5):.1f} L{Xt(Tbb):.1f},{Ya15(-5):.1f} L{Xt(Tbb):.1f},{Ya15(0):.1f}")
print(f"   px: a = 5 -> {Ya15(5):.1f}; 0 -> {Ya15(0):.1f}; -5 -> {Ya15(-5):.1f}; t = Tbb -> {Xt(Tbb):.1f}; t = 1 s -> {Xt(1):.1f}; 2 s -> {Xt(2):.1f}; 3 s -> {Xt(3):.1f}")

# =====================================================================================
cab(16, "REJILLA DE OCUPACIÓN LOG-ODDS: una celda vista cuatro veces")
lg = lambda p: math.log(p / (1 - p)); pr = lambda l: 1 / (1 + math.exp(-l))
l0 = lg(0.5); lh = lg(0.7); lm = lg(0.4)
print(f"  a priori p = 0,5 -> l0 = {l0:.3f}; medida 'ocupada' con p(m|z) = 0,7 -> {lh:.3f}; 'libre' con 0,4 -> {lm:.3f}")
l = l0
for k, (nm, lz) in enumerate((("ocupada", lh), ("ocupada", lh), ("ocupada", lh), ("libre", lm)), 1):
    l = l + lz - l0
    print(f"   medida {k} ({nm}): l = {l:.3f} -> p = {pr(l):.3f}")
print(f"  con límites [-2, 3,5]: tras 10 'ocupada' l = {min(3.5, 10*lh):.2f} (p = {pr(min(3.5, 10*lh)):.3f}) y hacen falta {math.ceil((3.5 - 0) / -lm)} 'libre' para volver a 0,5")
print(f"  sin límite: tras 10 'ocupada' l = {10*lh:.2f}; hacen falta {math.ceil(10*lh / -lm)} 'libre' para volver a 0,5")

# =====================================================================================
cab(17, "CAMPO DE DISTANCIAS (ESDF) DEL MAPA: valor y gradiente en unos puntos")
def esdf_g(p):
    p = np.asarray(p, float)
    d = np.array([np.linalg.norm(p - np.array(c)) - r_ for c, r_ in OBS])
    k = int(np.argmin(d)); c = np.array(OBS[k][0])
    return d[k], (p - c) / np.linalg.norm(p - c), k, d
for p in ((7.0, 5.0), (9.0, 6.0), (11.0, 5.0), (3.0, 7.0)):
    dk, gk, k, dd = esdf_g(p)
    print(f"  p = {p}: distancias a los 4 obstáculos {np.round(dd, 3).tolist()} -> d = {dk:.3f} m (obstáculo {k+1}); grad d = {np.round(gk, 3).tolist()}")
X17 = lin(0, 20, 40, 500); Y17 = lin(0, 10, 250, 20)
for p in ((7.0, 5.0), (9.0, 6.0), (11.0, 5.0), (3.0, 7.0)):
    dk, gk, _, _ = esdf_g(p)
    print(f"   px p ({X17(p[0]):.1f},{Y17(p[1]):.1f}) -> punta flecha ({X17(p[0]+1.2*gk[0]):.1f},{Y17(p[1]+1.2*gk[1]):.1f})")
xs = np.linspace(0, 20, 401)
dline = np.array([esdf_g((x, 5.0))[0] for x in xs])
Xc = lin(0, 20, 40, 500); Yc = lin(-2.5, 4, 150, 20)
print("path corte y5:", path(xs, dline, Xc, Yc))
print(f"   corte y = 5: px d=0 -> {Yc(0):.1f}; d=0,6 -> {Yc(0.6):.1f}; d=2 -> {Yc(2):.1f}; d=-2 -> {Yc(-2):.1f}; min {dline.min():.2f} en x = {xs[dline.argmin()]:.2f}")

# =====================================================================================
cab(18, "UN PASO DE GRADIENTE DEL PLANIFICADOR LOCAL, A MANO (w = 10, d_s = 0,6 m, paso 0,02)")
pm, pi_, pp_ = np.array([0.0, 0.0]), np.array([1.0, 0.2]), np.array([2.0, 0.0])
ai = pp_ - 2 * pi_ + pm
print(f"  suavidad: a_i = p_i+1 - 2 p_i + p_i-1 = {ai.tolist()}; |a_i|² = {ai@ai:.3f}; grad respecto a p_i del término i = -4 a_i = {(-4*ai).tolist()}")
print(f"   paso: p_i - 0,02·(-4 a_i) = {(pi_ - 0.02*(-4*ai)).round(4).tolist()} (baja hacia la recta)")
dd_, w_, ds_ = 0.2, 10.0, 0.6
print(f"  colisión: d = {dd_}, penalización w (d_s - d)² = {w_*(ds_-dd_)**2:.2f}; gradiente -2 w (d_s - d) grad d = -{2*w_*(ds_-dd_):.1f} grad d")
print(f"   paso: el punto se mueve 0,02·{2*w_*(ds_-dd_):.1f} = {0.02*2*w_*(ds_-dd_):.2f} m hacia fuera, a lo largo de grad d")
dt_ = 0.1
tt_ = np.array([0.9, 1.0, 1.1]); xx_ = tt_**3
print(f"  segundas diferencias = Dt² p'': p = t³, Dt = 0,1, t = 1: (p(1,1) - 2 p(1) + p(0,9)) / Dt² = {(xx_[2]-2*xx_[1]+xx_[0])/dt_**2:.3f}; p''(1) = 6")

# =====================================================================================
cab(19, "DISTANCIA DE PARADA PASO A PASO")
a30 = G * math.tan(math.radians(30))
print(f"  deceleración horizontal con inclinación 30° sin perder altura: a = g tan 30° = {a30:.3f} m/s²")
for v, tl, a in ((10.0, 0.1, a30), (10.0, 0.4, 3.0)):
    print(f"  v = {v} m/s, t_l = {tl} s, a = {a:.2f}: v t_l = {v*tl:.2f} m; v²/2a = {v*v/(2*a):.2f} m; total {v*tl + v*v/(2*a):.2f} m; tiempo de frenado v/a = {v/a:.2f} s")
for r_, tl, a in ((10.0, 0.1, a30), (10.0, 0.4, 3.0), (5.0, 0.1, a30), (20.0, 0.1, a30)):
    vmax = a * (-tl + math.sqrt(tl**2 + 2 * r_ / a))
    print(f"  alcance {r_:.0f} m, t_l = {tl}, a = {a:.2f}: v_max = a(-t_l + sqrt(t_l² + 2r/a)) = {vmax:.2f} m/s")
Xs = lin(0, 2.0, 60, 460); Ys = lin(0, 11, 200, 30)
tf = 0.1 + 10 / a30
print(f"path parada: M{Xs(0):.1f},{Ys(10):.1f} L{Xs(0.1):.1f},{Ys(10):.1f} L{Xs(tf):.1f},{Ys(0):.1f}")
print(f"   px: t = 0,1 -> {Xs(0.1):.1f}; t_f = {tf:.3f} s -> {Xs(tf):.1f}; v = 10 -> {Ys(10):.1f}; v = 0 -> {Ys(0):.1f}; t = 1 -> {Xs(1):.1f}; t = 2 -> {Xs(2):.1f}; v = 5 -> {Ys(5):.1f}")

# =====================================================================================
cab(20, "GIRO COORDINADO")
for V, ph in ((25.0, 30.0), (25.0, 45.0), (15.0, 30.0), (20.0, 45.0)):
    t_ = math.tan(math.radians(ph)); R_ = V * V / (G * t_)
    print(f"  V = {V:.0f} m/s, phi = {ph:.0f}°: g tan phi = {G*t_:.3f} m/s²; R = {V*V:.0f}/{G*t_:.3f} = {R_:.1f} m; psi_dot = V/R = {V/R_:.4f} rad/s = {math.degrees(V/R_):.1f}°/s;"
          f" vuelta en {2*math.pi*R_/V:.1f} s; n = 1/cos phi = {1/math.cos(math.radians(ph)):.3f}")

# =====================================================================================
# Dubins (copiado de figuras_a5.py)
def mod2pi(a):
    return a % (2 * math.pi)


def dubins(q0, q1, R):
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    D = math.hypot(dx, dy) / R; th = math.atan2(dy, dx)
    a, b = mod2pi(q0[2] - th), mod2pi(q1[2] - th)
    sa, sb, ca, cb, cab_ = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    sols = {}
    p2 = 2 + D * D - 2 * cab_ + 2 * D * (sa - sb)
    if p2 >= 0:
        t = mod2pi(-a + math.atan2(cb - ca, D + sa - sb)); p = math.sqrt(p2); q = mod2pi(b - math.atan2(cb - ca, D + sa - sb))
        sols["LSL"] = (t, p, q)
    p2 = 2 + D * D - 2 * cab_ + 2 * D * (sb - sa)
    if p2 >= 0:
        t = mod2pi(a - math.atan2(ca - cb, D - sa + sb)); p = math.sqrt(p2); q = mod2pi(-b + math.atan2(ca - cb, D - sa + sb))
        sols["RSR"] = (t, p, q)
    p2 = -2 + D * D + 2 * cab_ + 2 * D * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2); t = mod2pi(-a + math.atan2(-ca - cb, D + sa + sb) - math.atan2(-2, p)); q = mod2pi(-b + math.atan2(-ca - cb, D + sa + sb) - math.atan2(-2, p))
        sols["LSR"] = (t, p, q)
    p2 = -2 + D * D + 2 * cab_ - 2 * D * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2); t = mod2pi(a - math.atan2(ca + cb, D - sa - sb) + math.atan2(2, p)); q = mod2pi(b - math.atan2(ca + cb, D - sa - sb) + math.atan2(2, p))
        sols["RSL"] = (t, p, q)
    c = (6 - D * D + 2 * cab_ + 2 * D * (sa - sb)) / 8
    if abs(c) <= 1:
        p = mod2pi(2 * math.pi - math.acos(c)); t = mod2pi(a - math.atan2(ca - cb, D - sa + sb) + p / 2); q = mod2pi(a - b - t + p)
        sols["RLR"] = (t, p, q)
    c = (6 - D * D + 2 * cab_ + 2 * D * (sb - sa)) / 8
    if abs(c) <= 1:
        p = mod2pi(2 * math.pi - math.acos(c)); t = mod2pi(-a - math.atan2(ca - cb, D + sa - sb) + p / 2); q = mod2pi(b - a - t + p)
        sols["LRL"] = (t, p, q)
    return {k: R * sum(v) for k, v in sols.items()}, sols


def muestrea(q0, tipo, seg, R, ds=None):
    x, y, h = q0; pts = [(x, y)]
    ds = ds or R / 25
    for letra, L in zip(tipo, seg):
        n = max(2, int(L * R / ds))
        for _ in range(n):
            st = L * R / n
            if letra == "S":
                x += st * math.cos(h); y += st * math.sin(h)
            else:
                s = 1 if letra == "L" else -1
                h2 = h + s * st / R
                x += s * R * (math.sin(h2) - math.sin(h)); y += -s * R * (math.cos(h2) - math.cos(h)); h = h2
            pts.append((x, y))
    return np.array(pts)


cab(21, "LOS SEIS TIPOS DE DUBINS, CADA UNO EN UNA CONFIGURACIÓN DONDE ES EL MÁS CORTO (R = 1, inicio (0, 0, 0°))")
CONF = {"LSL": (0.0, 4.0, 180.0), "RSR": (0.0, -4.0, 180.0), "LSR": (3.0, 4.0, 0.0),
        "RSL": (3.0, -4.0, 0.0), "RLR": (0.5, 1.0, 180.0), "LRL": (0.5, -1.0, 180.0)}
for kk, (x1, y1, h1) in CONF.items():
    q1_ = (x1, y1, math.radians(h1))
    LA, SA = dubins((0.0, 0.0, 0.0), q1_, 1.0)
    orden_ = sorted(LA, key=LA.get)
    t, p, q_ = SA[kk]
    seg = f"{math.degrees(t):.0f}°, " + (f"{p:.2f} R" if kk[1] == "S" else f"{math.degrees(p):.0f}°") + f", {math.degrees(q_):.0f}°"
    print(f"  {kk}: meta ({x1}, {y1}, {h1:.0f}°); longitud {LA[kk]:.2f} R; tramos {seg}; orden: " + ", ".join(f"{k} {LA[k]:.2f}" for k in orden_))
    pts = muestrea((0.0, 0.0, 0.0), kk, SA[kk], 1.0)
    s0 = 1 if kk[0] == "L" else -1; s1 = 1 if kk[2] == "L" else -1
    c0_ = (0.0, s0 * 1.0); hh = math.radians(h1); c1_ = (x1 - s1 * math.sin(hh), y1 + s1 * math.cos(hh))
    # encuadre: caja de la trayectoria y de los dos círculos, centrada en un panel de 176 px con escala <= 26 px por R
    xs_ = np.concatenate([pts[:, 0], [c0_[0] - 1, c0_[0] + 1, c1_[0] - 1, c1_[0] + 1]])
    ys_ = np.concatenate([pts[:, 1], [c0_[1] - 1, c0_[1] + 1, c1_[1] - 1, c1_[1] + 1]])
    ESC = min(26.0, 130 / (xs_.max() - xs_.min()), 130 / (ys_.max() - ys_.min()))
    cxm, cym = (xs_.max() + xs_.min()) / 2, (ys_.max() + ys_.min()) / 2
    XA = lambda v: 88 + (v - cxm) * ESC; YA = lambda v: 92 - (v - cym) * ESC
    print(f"path seis {kk}:", path(pts[:, 0], pts[:, 1], XA, YA))
    print(f"   px inicio ({XA(0):.1f},{YA(0):.1f}); meta ({XA(x1):.1f},{YA(y1):.1f}); centro inicial ({XA(c0_[0]):.1f},{YA(c0_[1]):.1f}); centro final ({XA(c1_[0]):.1f},{YA(c1_[1]):.1f}); R {ESC:.1f} px")

cab(22, "DUBINS PASO A PASO EN EL EJEMPLO DE LA PRESENTACIÓN: LSL y RSL, R = 110 m, (0, 0, 0°) -> (400, 150, 180°)")
R = 25.0**2 / (G * math.tan(math.radians(30)))
p0 = np.array([0.0, 0.0]); p1 = np.array([400.0, 150.0]); ps0, ps1 = 0.0, math.pi
izq = lambda p, ps: p + R * np.array([-math.sin(ps), math.cos(ps)])
der = lambda p, ps: p + R * np.array([math.sin(ps), -math.cos(ps)])
c0, c1 = izq(p0, ps0), izq(p1, ps1)
v = c1 - c0; ell = np.linalg.norm(v); th = math.atan2(v[1], v[0])
a1 = mod2pi(th - ps0); a2 = mod2pi(ps1 - th)
print(f"  R = {R:.2f} m")
print(f"  LSL: c0 = {np.round(c0, 2).tolist()}, c1 = {np.round(c1, 2).tolist()}; c1 - c0 = {np.round(v, 2).tolist()}; |c1 - c0| = {ell:.2f} m; theta = {math.degrees(th):.2f}°")
print(f"   arco 1 = (theta - psi0) mod 360 = {math.degrees(a1):.2f}° = {a1:.4f} rad; arco 2 = (psi1 - theta) mod 360 = {math.degrees(a2):.2f}° = {a2:.4f} rad")
print(f"   L = R (a1 + a2) + |c1 - c0| = {R:.2f}·{a1+a2:.4f} + {ell:.2f} = {R*(a1+a2):.1f} + {ell:.1f} = {R*(a1+a2)+ell:.1f} m")
c0r = der(p0, ps0)
v = c1 - c0r; ell = np.linalg.norm(v); thc = math.atan2(v[1], v[0])
S_ = math.sqrt(ell**2 - 4 * R**2); al = math.asin(2 * R / ell)
hs = thc - al
b1 = mod2pi(ps0 - hs); b2 = mod2pi(ps1 - hs)
print(f"  RSL: c0 (derecha) = {np.round(c0r, 2).tolist()}, c1 (izquierda) = {np.round(c1, 2).tolist()}; |c1 - c0| = ell = {ell:.2f} m (> 2R = {2*R:.1f}); rumbo de c1 - c0 = {math.degrees(thc):.2f}°")
print(f"   recta S = sqrt(ell² - 4R²) = {S_:.2f} m; alpha = arcsin(2R/ell) = {math.degrees(al):.2f}° (= arctan(2R/S) = {math.degrees(math.atan(2*R/S_)):.2f}°)")
print(f"   rumbo de la recta = {math.degrees(thc):.2f}° - {math.degrees(al):.2f}° = {math.degrees(hs):.2f}°")
print(f"   arco 1 (a la derecha) = (psi0 - rumbo) mod 360 = {math.degrees(b1):.2f}°; arco 2 (a la izquierda) = (psi1 - rumbo) mod 360 = {math.degrees(b2):.2f}°")
print(f"   L = R (a1 + a2) + S = {R:.2f}·{b1+b2:.4f} + {S_:.2f} = {R*(b1+b2)+S_:.1f} m")
Lref, _ = dubins((0, 0, 0), (400, 150, math.pi), R)
print(f"  comprobación con la forma cerrada de figuras_a5.py: LSL {Lref['LSL']:.1f}, RSL {Lref['RSL']:.1f}")
print(f"  error de la versión anterior de la hoja: arctan(2R/ell) = {math.degrees(math.atan(2*R/ell)):.2f}° frente a arcsin(2R/ell) = {math.degrees(al):.2f}°")
# figura de la geometría RSL: escala de la figura de Dubins de la presentación
esc4 = 0.348   # la imprime figuras_a5.py; aquí se recalcula igual
X4b = lin(-150, 560, 20, 270); Y4b = lin(-240, 300, 250, 10)
t1 = c0r + R * np.array([math.cos(hs + math.pi/2), math.sin(hs + math.pi/2)])
t2 = c1 + R * np.array([math.cos(hs - math.pi/2), math.sin(hs - math.pi/2)])
print(f"   puntos de tangencia RSL: {np.round(t1, 1).tolist()} y {np.round(t2, 1).tolist()}; |t2 - t1| = {np.linalg.norm(t2-t1):.2f} (= S)")

cab(23, "VIENTO: TRAYECTORIA SOBRE EL SUELO DE UN GIRO COMPLETO A 25 m/s, alabeo 30°, viento de 10 m/s hacia +x")
V, W = 25.0, 10.0
R0 = V * V / a30; wr = V / R0; Tv = 2 * math.pi / wr
ts = np.linspace(0, Tv, 241)
# rumbo inicial 90° (perpendicular al viento) y giro a la izquierda
psi = math.pi / 2 + wr * ts
xa = R0 * (np.sin(psi) - 1); ya = -R0 * np.cos(psi)
xg = xa + W * ts; yg = ya
print(f"  R aire = {R0:.1f} m, periodo {Tv:.2f} s, deriva {W*Tv:.0f} m")
print(f"  radio de curvatura sobre el suelo: a favor (rumbo 0°) {(V+W)**2/a30:.0f} m; en contra (rumbo 180°) {(V-W)**2/a30:.0f} m")
k1 = int(np.argmin(abs(np.cos(psi) - 1))); k2 = int(np.argmin(abs(np.cos(psi) + 1)))
print(f"   rumbo 0° en t = {ts[k1]:.1f} s, posición suelo ({xg[k1]:.0f}, {yg[k1]:.0f}); rumbo 180° en t = {ts[k2]:.1f} s, ({xg[k2]:.0f}, {yg[k2]:.0f})")
# también dos vueltas para ver la trocoide
ts2 = np.linspace(0, 2 * Tv, 481); psi2 = math.pi / 2 + wr * ts2
xg2 = R0 * (np.sin(psi2) - 1) + W * ts2; yg2 = -R0 * np.cos(psi2)
X23 = lin(-260, 600, 20, 500); Y23 = lin(-140, 140, 230, 30)
print(f"   encuadre: x [{xg2.min():.0f}, {xg2.max():.0f}], y [{yg2.min():.0f}, {yg2.max():.0f}]")
esc23 = min(480 / (xg2.max() - xg2.min() + 40), 200 / (yg2.max() - yg2.min() + 40))
X23 = lambda v: 20 + (v - xg2.min() + 20) * esc23; Y23 = lambda v: 230 - (v - yg2.min() + 20) * esc23
print(f"   escala {esc23:.3f} px/m")
print("path trocoide suelo:", path(xg2, yg2, X23, Y23))
ca = R0 * (np.sin(psi2) - 1); cb = -R0 * np.cos(psi2)
print("path circulo aire:", path(ca[:241], cb[:241], X23, Y23))
print(f"   px inicio ({X23(0):.1f},{Y23(0):.1f}); tras una vuelta ({X23(W*Tv):.1f},{Y23(0):.1f}); tras dos ({X23(2*W*Tv):.1f},{Y23(0):.1f})")
print(f"   px a favor ({X23(xg[k1]):.1f},{Y23(yg[k1]):.1f}); en contra ({X23(xg[k2]):.1f},{Y23(yg[k2]):.1f})")
