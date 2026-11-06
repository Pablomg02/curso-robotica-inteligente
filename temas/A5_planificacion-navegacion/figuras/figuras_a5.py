"""Datos de las figuras y tablas del tema A5 (planificación y navegación).

Uso:  python3 figuras_a5.py > salida.txt      (numpy; menos de un minuto)
"""
import heapq
import math
import time
import numpy as np


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


# Mapa 2D de 20 m x 10 m con obstáculos circulares
OBS = [((5.0, 3.0), 1.8), ((9.0, 7.0), 2.0), ((13.0, 3.5), 2.2), ((16.5, 7.5), 1.5)]
START, GOAL = np.array([1.0, 1.0]), np.array([19.0, 9.0])
OC = np.array([c for c, r in OBS]); ORAD = np.array([r for c, r in OBS])
def libres(P, m=0.3):
    """Vectorizado: P (n, 2) -> (n,) booleano."""
    P = np.atleast_2d(P)
    d = np.linalg.norm(P[:, None, :] - OC[None], axis=2) - ORAD[None] - m
    return (d.min(axis=1) > 0) & (P[:, 0] >= 0) & (P[:, 0] <= 20) & (P[:, 1] >= 0) & (P[:, 1] <= 10)
def libre(p, m=0.3):
    return bool(libres(p, m)[0])
def segmento_libre(a, b, paso=0.05):
    n = max(2, int(np.linalg.norm(b - a) / paso))
    s = np.linspace(0, 1, n)[:, None]
    return bool(libres(a[None] + (b - a)[None] * s).all())

print("=" * 70)
print("1. A* EN REJILLA (0,1 m, 8-conexa) FRENTE A RRT Y RRT*")
res = 0.1
nx, ny = int(20 / res) + 1, int(10 / res) + 1
gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
occ = ~libres(np.stack([gi.ravel() * res, gj.ravel() * res], axis=1)).reshape(nx, ny)
def astar():
    s = (int(START[0] / res), int(START[1] / res)); g = (int(GOAL[0] / res), int(GOAL[1] / res))
    h = lambda a: res * math.hypot(a[0] - g[0], a[1] - g[1])
    openq = [(h(s), 0.0, s)]; came = {s: None}; cost = {s: 0.0}; exp = 0
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
                    cost[v] = nc; came[v] = u; heapq.heappush(openq, (nc + h(v), nc, v))
    p = [g]
    while came[p[-1]] is not None:
        p.append(came[p[-1]])
    p = np.array(p[::-1]) * res
    return p, cost[g], exp
t0 = time.perf_counter(); pa, ca, ea = astar(); ta = time.perf_counter() - t0
print(f"  A*: longitud {ca:.2f} m, {ea} nodos expandidos de {nx*ny}, {1000*ta:.0f} ms (Python)")
def atajos(P):
    """Recorta el camino uniendo cada punto con el más lejano visible (string pulling voraz)."""
    out = [P[0]]; i = 0
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and not segmento_libre(P[i], P[j]):
            j -= 1
        out.append(P[j]); i = j
    return np.array(out)
pa_c = atajos(pa)
la_c = np.sum(np.linalg.norm(np.diff(pa_c, axis=0), axis=1))
print(f"  A* recortado por visibilidad: {la_c:.2f} m con {len(pa_c)} vértices (la rejilla 8-conexa alarga hasta un 8,2 %: {(ca/la_c-1)*100:.1f} % aquí)")

def rrt(n_iter, estrella, seed, eta=1.0):
    """RRT y RRT* con recableado; los costes se propagan a los descendientes (sin ciclos)."""
    rng = np.random.default_rng(seed)
    V = [START.copy()]; par = [-1]; cst = [0.0]; hijos = [[]]
    mejor = []
    def propaga(i):
        pila = [i]
        while pila:
            u = pila.pop()
            for h in hijos[u]:
                cst[h] = cst[u] + np.linalg.norm(V[h] - V[u]); pila.append(h)
    def coste_meta():
        Va = np.array(V); cerca = np.where(np.linalg.norm(Va - GOAL, axis=1) < 0.5)[0]
        return (min(cerca, key=lambda i: cst[i]) if len(cerca) else None)
    for it in range(n_iter):
        q = GOAL.copy() if rng.random() < 0.05 else rng.random(2) * np.array([20.0, 10.0])
        Va = np.array(V)
        d = np.linalg.norm(Va - q, axis=1); k = int(np.argmin(d))
        nuevo = V[k] + (q - V[k]) * min(1.0, eta / max(d[k], 1e-9))
        if libre(nuevo) and segmento_libre(V[k], nuevo):
            padre, c = k, cst[k] + np.linalg.norm(nuevo - V[k])
            vec = []
            if estrella:
                rad = min(2.5, 6.0 * math.sqrt(math.log(len(V) + 1) / (len(V) + 1)))
                vec = list(np.where(np.linalg.norm(Va - nuevo, axis=1) < rad)[0])
                for i in vec:
                    ci = cst[i] + np.linalg.norm(nuevo - V[i])
                    if ci < c and segmento_libre(V[i], nuevo):
                        padre, c = i, ci
            V.append(nuevo); par.append(padre); cst.append(c); hijos.append([]); n = len(V) - 1
            hijos[padre].append(n)
            for i in vec:
                ci = c + np.linalg.norm(V[i] - nuevo)
                if ci < cst[i] - 1e-9 and segmento_libre(nuevo, V[i]):
                    hijos[par[i]].remove(i); par[i] = n; hijos[n].append(i); cst[i] = ci; propaga(i)
        m = coste_meta()
        mejor.append(cst[m] if m is not None else np.nan)
    m = coste_meta(); camino = []
    if m is not None:
        j = m
        while j != -1:
            camino.append(V[j]); j = par[j]
    return np.array(camino[::-1]), np.array(mejor), np.array(V), par
pr, mr, Vr, _ = rrt(3000, False, 0)
ps, ms, Vs, _ = rrt(3000, True, 0)
for nm, m in (("RRT", mr), ("RRT*", ms)):
    vals = {n: m[n - 1] for n in (500, 1000, 3000)}
    print(f"  {nm:5s}: coste del mejor camino tras 500 / 1000 / 3000 iteraciones: "
          + " / ".join(f"{v:.2f}" for v in vals.values()) + " m")
X1 = lin(0, 20, 40, 500)
Y1 = lin(0, 10, 250, 20)
print("path astar:", path(pa[:, 0], pa[:, 1], X1, Y1))
print("path rrt:", path(pr[:, 0], pr[:, 1], X1, Y1))
print("path rrtstar:", path(ps[:, 0], ps[:, 1], X1, Y1))
print("path astarcorto:", path(pa_c[:, 0], pa_c[:, 1], X1, Y1))
for (c, r) in OBS:
    print(f"  obstáculo: cx {X1(c[0]):.1f} cy {Y1(c[1]):.1f} r {r*23:.1f}")
print(f"  inicio ({X1(START[0]):.1f},{Y1(START[1]):.1f}); meta ({X1(GOAL[0]):.1f},{Y1(GOAL[1]):.1f})")

print("=" * 70)
print("2. MINIMUM SNAP: polinomios de grado 7 por tramo, continuidad impuesta hasta el snap")
def coef_deriv(k, t, n=8):
    """Fila con d^k/dt^k de [1, t, ..., t^7]."""
    row = np.zeros(n)
    for i in range(k, n):
        row[i] = math.factorial(i) / math.factorial(i - k) * t ** (i - k)
    return row
def Q_snap(T, n=8, r=4):
    """Coste de snap de un tramo en tiempo normalizado tau = t/T: c^T Q c con p(t) = sum c_j tau^j."""
    Q = np.zeros((n, n))
    for i in range(r, n):
        for j in range(r, n):
            ci = math.factorial(i) / math.factorial(i - r); cj = math.factorial(j) / math.factorial(j - r)
            Q[i, j] = ci * cj / (i + j - 2 * r + 1)
    return Q / T ** (2 * r - 1)
def fila(k, tau, T):
    """d^k p / dt^k en tau (tiempo normalizado) para un tramo de duración T."""
    return coef_deriv(k, tau) / T ** k
CONT = 4   # se impone continuidad hasta el snap; el óptimo sale continuo hasta la derivada 6
def min_snap(wps, Ts, devolver_saltos=False):
    """wps: (M+1,) posiciones 1D; Ts: duraciones de los M tramos. Reposo en los extremos (v, a, j = 0)."""
    M = len(Ts); n = 8; N = M * n
    H = np.zeros((N, N))
    for s, T in enumerate(Ts):
        H[s*n:(s+1)*n, s*n:(s+1)*n] = Q_snap(T)
    A, b = [], []
    for s, T in enumerate(Ts):
        r0 = np.zeros(N); r0[s*n:(s+1)*n] = fila(0, 0.0, T); A.append(r0); b.append(wps[s])
        r1 = np.zeros(N); r1[s*n:(s+1)*n] = fila(0, 1.0, T); A.append(r1); b.append(wps[s+1])
        if s < M - 1:
            for k in range(1, CONT + 1):
                r = np.zeros(N); r[s*n:(s+1)*n] = fila(k, 1.0, T); r[(s+1)*n:(s+2)*n] = -fila(k, 0.0, Ts[s+1])
                A.append(r); b.append(0.0)
    for k in (1, 2, 3):
        r = np.zeros(N); r[0:n] = fila(k, 0.0, Ts[0]); A.append(r); b.append(0.0)
        r = np.zeros(N); r[(M-1)*n:M*n] = fila(k, 1.0, Ts[-1]); A.append(r); b.append(0.0)
    A = np.array(A); b = np.array(b)
    KKT = np.block([[2 * H, A.T], [A, np.zeros((len(b), len(b)))]])
    sol = np.linalg.solve(KKT, np.concatenate([np.zeros(N), b]))
    c = sol[:N].reshape(M, n)
    assert np.allclose(A @ sol[:N], b, atol=1e-8), "restricciones no satisfechas"
    J = c.reshape(-1) @ H @ c.reshape(-1)
    if devolver_saltos:
        saltos = [max(abs(fila(k, 1.0, Ts[s]) @ c[s] - fila(k, 0.0, Ts[s+1]) @ c[s+1]) for s in range(M - 1)) for k in (5, 6)]
        return c, J, saltos, A.shape[0]
    return c, J
def evalua(c, Ts, k, npts=60):
    ts, vs = [], []
    t0 = 0.0
    for s, T in enumerate(Ts):
        for tau in np.linspace(0, 1, npts, endpoint=(s == len(Ts) - 1)):
            ts.append(t0 + tau * T); vs.append(fila(k, tau, T) @ c[s])
        t0 += T
    return np.array(ts), np.array(vs)
wx = np.array([0.0, 4.0, 8.0, 12.0, 16.0]); wy = np.array([0.0, 3.0, 0.0, -3.0, 0.0])
dist = np.hypot(np.diff(wx), np.diff(wy))
res2 = {}
for Ttot in (6.0, 8.0, 12.0):
    Ts = dist / dist.sum() * Ttot                  # tiempos proporcionales a la distancia
    cx, Jx = min_snap(wx, Ts); cy, Jy = min_snap(wy, Ts)
    _, ax = evalua(cx, Ts, 2); _, ay = evalua(cy, Ts, 2)
    _, vx = evalua(cx, Ts, 1); _, vy = evalua(cy, Ts, 1)
    res2[Ttot] = (cx, cy, Ts, Jx + Jy)
    print(f"  T = {Ttot:4.1f} s: coste de snap {Jx+Jy:10.1f}; velocidad máx. {np.max(np.hypot(vx,vy)):.2f} m/s;"
          f" aceleración máx. {np.max(np.hypot(ax,ay)):.2f} m/s^2")
r = res2[6.0][3] / res2[12.0][3]
_, _, saltos, nres = min_snap(wx, dist / dist.sum() * 8.0, devolver_saltos=True)
print(f"  {nres} restricciones para 32 coeficientes por eje (continuidad impuesta hasta el snap); saltos en las derivadas 5 y 6 del óptimo:"
      f" {saltos[0]:.1e} y {saltos[1]:.1e} (la teoría dice que son continuas)")
print(f"  cociente de coste T=6 / T=12: {r:.1f} (la escala temporal predice 2^7 = 128)")
# Reparto de tiempos con T = 8 s fijo: tramos (a, 4-a, 4-a, a) por simetría; búsqueda en a
def coste_total(Ts_):
    return min_snap(wx, Ts_)[1] + min_snap(wy, Ts_)[1]
aa = np.linspace(0.4, 3.6, 321)
JJ = [coste_total(np.array([x_, 4 - x_, 4 - x_, x_])) for x_ in aa]
a_opt = aa[int(np.argmin(JJ))]
Ts_opt = np.array([a_opt, 4 - a_opt, 4 - a_opt, a_opt])
cxo, Jxo = min_snap(wx, Ts_opt); cyo, Jyo = min_snap(wy, Ts_opt)
def resumen(cx_, cy_, Ts_):
    tt, vx_ = evalua(cx_, Ts_, 1); _, vy_ = evalua(cy_, Ts_, 1); _, ax_ = evalua(cx_, Ts_, 2); _, ay_ = evalua(cy_, Ts_, 2)
    v = np.hypot(vx_, vy_)
    t_wp = np.cumsum(Ts_)[1] ; k = np.argmin(abs(tt - t_wp))
    return tt, v, v.max(), np.hypot(ax_, ay_).max(), v[k]
cx, cy, Ts, J8 = res2[8.0]
tt8, v8, vm8, am8, vc8 = resumen(cx, cy, Ts)
tto, vo, vmo, amo, vco = resumen(cxo, cyo, Ts_opt)
print(f"  T = 8 s, tiempos iguales (2 s por tramo): coste {J8:.1f}, v máx. {vm8:.2f} m/s, a máx. {am8:.2f} m/s^2, velocidad en el waypoint central {vc8:.2f} m/s")
print(f"  T = 8 s, reparto optimizado ({a_opt:.2f}, {4-a_opt:.2f}, {4-a_opt:.2f}, {a_opt:.2f}) s: coste {Jxo+Jyo:.1f} ({J8/(Jxo+Jyo):.1f} veces menor),"
      f" v máx. {vmo:.2f} m/s, a máx. {amo:.2f} m/s^2, velocidad en el waypoint central {vco:.2f} m/s")
X2v = lin(0, 8, 340, 500); Y2v = lin(0, 6, 230, 110)
print("path vel igual:", path(tt8, v8, X2v, Y2v))
print("path vel opt:", path(tto, vo, X2v, Y2v))
_, xo = evalua(cxo, Ts_opt, 0); _, yo = evalua(cyo, Ts_opt, 0)
_, xx = evalua(cx, Ts, 0); _, yy = evalua(cy, Ts, 0)
X2 = lin(-1, 17, 30, 300); Y2 = lin(-4, 4, 230, 50)
print("path snap xy:", path(xx, yy, X2, Y2))
print("path snap xy opt:", path(xo, yo, X2, Y2))
print("  waypoints:", [(round(float(X2(a)), 1), round(float(Y2(b)), 1)) for a, b in zip(wx, wy)])

print("=" * 70)
print("3. PLANIFICADOR LOCAL SOBRE UN CAMPO DE DISTANCIAS (ESDF): suavidad + colisión, descenso por gradiente")
def esdf(p):
    d = np.array([np.hypot(p[..., 0] - c[0], p[..., 1] - c[1]) - r for c, r in OBS])
    k = np.argmin(d, axis=0)
    return d.min(axis=0)
Npts = 40
P0 = np.linspace(START, GOAL, Npts)                  # línea recta: atraviesa obstáculos
def coste(P, w_c=10.0, dseg=0.6):
    acc = P[2:] - 2 * P[1:-1] + P[:-2]
    d = esdf(P[1:-1])
    return np.sum(acc**2) + w_c * np.sum(np.maximum(0, dseg - d) ** 2)
def grad(P, w_c=10.0, dseg=0.6):
    """Gradiente analítico del coste (suavidad + colisión con el obstáculo más cercano)."""
    g = np.zeros_like(P)
    acc = P[2:] - 2 * P[1:-1] + P[:-2]
    g[2:] += 2 * acc; g[1:-1] += -4 * acc; g[:-2] += 2 * acc
    Q = P[1:-1]
    dif = Q[:, None, :] - OC[None]; nrm = np.linalg.norm(dif, axis=2)
    d = nrm - ORAD[None]; k = np.argmin(d, axis=1); dk = d[np.arange(len(Q)), k]
    u = dif[np.arange(len(Q)), k] / nrm[np.arange(len(Q)), k][:, None]
    g[1:-1] += (-2 * w_c * np.maximum(0, dseg - dk))[:, None] * u
    g[0] = 0; g[-1] = 0
    return g
P = P0 + np.random.default_rng(0).normal(0, 0.05, P0.shape) * np.r_[0, np.ones(Npts - 2), 0][:, None]
dmin0 = esdf(P0).min()
t0 = time.perf_counter()
for it in range(4000):
    P = P - 0.02 * grad(P)
tl = time.perf_counter() - t0
print(f"  distancia mínima a obstáculos: recta {dmin0:.2f} m (dentro si < 0) -> optimizada {esdf(P).min():.2f} m;"
      f" longitud {np.sum(np.linalg.norm(np.diff(P,axis=0),axis=1)):.2f} m (recta {np.linalg.norm(GOAL-START):.2f} m); {tl:.1f} s en Python")
print("path local ini:", path(P0[:, 0], P0[:, 1], X1, Y1))
print("path local opt:", path(P[:, 0], P[:, 1], X1, Y1))

print("=" * 70)
print("4. DUBINS: radio mínimo R = V^2 / (g tan(phi_max)) y caminos más cortos")
G = 9.81
for V, ph in ((25.0, 30.0), (25.0, 45.0), (15.0, 30.0)):
    print(f"  V = {V:.0f} m/s, alabeo {ph:.0f}°: R = {V**2/(G*math.tan(math.radians(ph))):.0f} m")
def mod2pi(a):
    return a % (2 * math.pi)
def dubins(q0, q1, R):
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    D = math.hypot(dx, dy) / R; th = math.atan2(dy, dx)
    a, b = mod2pi(q0[2] - th), mod2pi(q1[2] - th)
    sa, sb, ca, cb, cab = math.sin(a), math.sin(b), math.cos(a), math.cos(b), math.cos(a - b)
    sols = {}
    p2 = 2 + D * D - 2 * cab + 2 * D * (sa - sb)                 # LSL
    if p2 >= 0:
        t = mod2pi(-a + math.atan2(cb - ca, D + sa - sb)); p = math.sqrt(p2); q = mod2pi(b - math.atan2(cb - ca, D + sa - sb))
        sols["LSL"] = (t, p, q)
    p2 = 2 + D * D - 2 * cab + 2 * D * (sb - sa)                 # RSR
    if p2 >= 0:
        t = mod2pi(a - math.atan2(ca - cb, D - sa + sb)); p = math.sqrt(p2); q = mod2pi(-b + math.atan2(ca - cb, D - sa + sb))
        sols["RSR"] = (t, p, q)
    p2 = -2 + D * D + 2 * cab + 2 * D * (sa + sb)                # LSR
    if p2 >= 0:
        p = math.sqrt(p2); t = mod2pi(-a + math.atan2(-ca - cb, D + sa + sb) - math.atan2(-2, p)); q = mod2pi(-b + math.atan2(-ca - cb, D + sa + sb) - math.atan2(-2, p))
        sols["LSR"] = (t, p, q)
    p2 = -2 + D * D + 2 * cab - 2 * D * (sa + sb)                # RSL
    if p2 >= 0:
        p = math.sqrt(p2); t = mod2pi(a - math.atan2(ca + cb, D - sa - sb) + math.atan2(2, p)); q = mod2pi(b - math.atan2(ca + cb, D - sa - sb) + math.atan2(2, p))
        sols["RSL"] = (t, p, q)
    c = (6 - D * D + 2 * cab + 2 * D * (sa - sb)) / 8            # RLR
    if abs(c) <= 1:
        p = mod2pi(2 * math.pi - math.acos(c)); t = mod2pi(a - math.atan2(ca - cb, D - sa + sb) + p / 2); q = mod2pi(a - b - t + p)
        sols["RLR"] = (t, p, q)
    c = (6 - D * D + 2 * cab + 2 * D * (sb - sa)) / 8            # LRL
    if abs(c) <= 1:
        p = mod2pi(2 * math.pi - math.acos(c)); t = mod2pi(-a - math.atan2(ca - cb, D + sa - sb) + p / 2); q = mod2pi(b - a - t + p)
        sols["LRL"] = (t, p, q)
    return {k: R * sum(v) for k, v in sols.items()}, sols
def muestrea(q0, tipo, seg, R, ds=2.0):
    x, y, h = q0; pts = [(x, y)]
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
R = 25.0**2 / (G * math.tan(math.radians(30)))
q0, q1 = (0.0, 0.0, 0.0), (400.0, 150.0, math.pi)
L, sols = dubins(q0, q1, R)
mejor = min(L, key=L.get)
print(f"  de (0, 0, 0°) a (400, 150, 180°) con R = {R:.0f} m: " + ", ".join(f"{k} {v:.0f} m" for k, v in sorted(L.items(), key=lambda kv: kv[1]))
      + f"; mejor {mejor}; distancia en línea recta {math.hypot(400,150):.0f} m")
todos = {k: muestrea(q0, k, sols[k], R) for k in L}
allp = np.vstack(list(todos.values()))
(xmin, ymin), (xmax, ymax) = allp.min(axis=0) - 20, allp.max(axis=0) + 20
esc = min(460 / (xmax - xmin), 220 / (ymax - ymin))       # misma escala en x e y
X4 = lambda v: 30 + (v - xmin) * esc; Y4 = lambda v: 240 - (v - ymin) * esc
print(f"  escala {esc:.3f} px/m; inicio ({X4(0):.1f},{Y4(0):.1f}); meta ({X4(400):.1f},{Y4(150):.1f}); R en px {R*esc:.1f}; ancho {(xmax-xmin)*esc:.0f} alto {(ymax-ymin)*esc:.0f}")
for k, pts in todos.items():
    fin = np.hypot(pts[-1, 0] - q1[0], pts[-1, 1] - q1[1])
    print(f"  {k}: error de cierre {fin:.2f} m")
    print(f"path dubins {k}:", path(pts[:, 0], pts[:, 1], X4, Y4))

print("=" * 70)
print("5. DISTANCIA DE PARADA d = v t_lat + v^2 / (2 a): alcance mínimo de percepción")
for nombre, tl, acc in (("t_lat 0,1 s, a = g tan 30°", 0.1, G * math.tan(math.radians(30))),
                        ("valores por defecto de PX4: CP_DELAY 0,4 s, MPC_ACC_HOR 3 m/s^2", 0.4, 3.0)):
    print("  " + nombre + ": " + ", ".join(f"v {v:.0f} m/s -> {v*tl + v*v/(2*acc):.1f} m" for v in (3, 5, 10, 20)))

print("=" * 70)
print("6. GIRO A ALABEO CONSTANTE CON VIENTO: radio de curvatura sobre el suelo R = (V ± W)^2 / (g tan(phi))")
V, W, ph = 25.0, 10.0, 30.0
gt = G * math.tan(math.radians(ph))
print(f"  V = {V:.0f} m/s, W = {W:.0f} m/s, alabeo {ph:.0f}°: sin viento {V*V/gt:.0f} m; a favor del viento {(V+W)**2/gt:.0f} m; contra el viento {(V-W)**2/gt:.0f} m;"
      f" deriva durante una vuelta completa {W*2*math.pi*V/gt:.0f} m (periodo {2*math.pi*V/gt:.1f} s)")
