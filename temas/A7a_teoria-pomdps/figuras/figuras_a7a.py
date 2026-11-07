"""Datos de las figuras y tablas del tema A7a (teoría de POMDPs).

Uso:  uv run --with numpy python figuras_a7a.py > salida.txt        (solo CPU; unos minutos)
      python figuras_a7a.py 1 3     (solo esas secciones)

  1. El tigre (Kaelbling, Littman y Cassandra, 1998) resuelto exactamente con vectores α: función de valor
     lineal a trozos y convexa para varios horizontes, número de vectores con y sin poda, umbrales de la política.
  2. Actualización del belief en el tigre y regla «abre cuando hayas oído dos veces más un lado».
  3. QMDP frente al belief MDP exacto en un pasillo con un cartel (el valor de la información).
  4. POMCP en el tigre frente a la política exacta, según el número de simulaciones.
  5. Búsqueda con un dron en una rejilla con detección imperfecta: belief del objetivo y cuatro políticas.
"""
import sys
import math
import time
import numpy as np

SECCIONES = {int(x) for x in sys.argv[1:]} or {1, 2, 3, 4, 5, 6}


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# ====================================================================== EL TIGRE
# estados: 0 tigre a la izquierda, 1 a la derecha; acciones: 0 escuchar, 1 abrir izquierda, 2 abrir derecha
# observaciones: 0 se oye a la izquierda, 1 a la derecha
NS, NA, NO = 2, 3, 2
P_OIR = 0.85
R = np.array([[-1.0, -100.0, 10.0],      # tigre a la izquierda
              [-1.0, 10.0, -100.0]])    # tigre a la derecha
T = np.zeros((NA, NS, NS))
T[0] = np.eye(2)
T[1] = T[2] = 0.5
O = np.zeros((NA, NS, NO))                # O[a, s', o]
O[0] = [[P_OIR, 1 - P_OIR], [1 - P_OIR, P_OIR]]
O[1] = O[2] = 0.5
GAMMA_T = 0.95


TOL_INT = 1e-9          # anchura mínima del intervalo en el que un vector es máximo (evita facetas numéricas)


def envolvente(alfas):
    """Vectores α que forman la envolvente superior exacta sobre p = b(tigre izquierda) ∈ [0, 1].
    Cada α es la recta V(p) = p·α0 + (1-p)·α1 = α1 + (α0-α1)·p. Algoritmo de la envolvente convexa de rectas:
    O(n log n) en tiempo y O(n) en memoria (no se evalúa en ninguna rejilla)."""
    alfas = np.unique(np.round(alfas, 9), axis=0)       # vectores repetidos o casi idénticos
    m = alfas[:, 0] - alfas[:, 1]
    c = alfas[:, 1]
    orden = np.lexsort((-c, m))               # pendiente creciente; a igual pendiente, mayor ordenada primero
    hull = []
    for i in orden:
        if hull and abs(m[hull[-1]] - m[i]) < 1e-12:
            continue                          # misma pendiente, ordenada menor: dominada
        while len(hull) >= 2:
            j, k = hull[-2], hull[-1]
            x_jk = (c[j] - c[k]) / (m[k] - m[j])
            x_ji = (c[j] - c[i]) / (m[i] - m[j])
            if x_ji <= x_jk:
                hull.pop()
            else:
                break
        hull.append(i)
    # recortar a [0, 1]
    xs = [-np.inf] + [(c[hull[t]] - c[hull[t + 1]]) / (m[hull[t + 1]] - m[hull[t]]) for t in range(len(hull) - 1)] + [np.inf]
    keep = [hull[t] for t in range(len(hull)) if min(xs[t + 1], 1.0) - max(xs[t], 0.0) > TOL_INT]
    return alfas[keep]


def backup(G, gamma):
    """Una iteración exacta de valor en el espacio de vectores α (enumeración + poda)."""
    nuevos = []
    n_enum = 0
    for a in range(NA):
        # g[a,o][k](s) = Σ_s' T(s,a,s') O(o|s',a) α_k(s')
        g = [np.array([T[a] @ (O[a][:, o] * al) for al in G]) for o in range(NO)]
        for i in range(len(G)):
            for j in range(len(G)):
                nuevos.append(R[:, a] + gamma * (g[0][i] + g[1][j]))
                n_enum += 1
    nuevos = np.array(nuevos)
    return envolvente(nuevos), n_enum


def accion_de(alfa, G_prev, gamma):
    """Qué acción genera cada vector (para pintar regiones): se recalcula por coincidencia."""
    for a in range(NA):
        g = [np.array([T[a] @ (O[a][:, o] * al) for al in G_prev]) for o in range(NO)]
        for i in range(len(G_prev)):
            for j in range(len(G_prev)):
                if np.allclose(R[:, a] + gamma * (g[0][i] + g[1][j]), alfa):
                    return a
    return -1


def resolver_tigre(gamma, horizontes):
    G = np.zeros((1, NS))
    hist = {0: G}
    cuentas = []
    for t in range(1, max(horizontes) + 1):
        Gp = G
        G, n_enum = backup(G, gamma)
        cuentas.append((t, n_enum, len(G)))
        hist[t] = G
        hist[-t] = Gp
    return hist, cuentas


if 1 in SECCIONES or 2 in SECCIONES or 4 in SECCIONES:
    print("=" * 70)
    print(f"1. EL TIGRE: ITERACIÓN DE VALOR EXACTA CON VECTORES α (γ = {GAMMA_T})")
    HZ = 400
    hist, cuentas = resolver_tigre(GAMMA_T, [1, 2, 3, HZ])
    for t, ne, n in cuentas[:8]:
        print(f"  horizonte {t}: {ne} vectores enumerados, {n} en la envolvente")
    for t in (20, 50, 100, 200, 400):
        print(f"  horizonte {t}: {cuentas[t-1][2]} vectores en la envolvente")
    G_inf = hist[HZ]
    p = np.linspace(0, 1, 401)
    X1 = lin(0, 1, 60, 520)
    for t in (1, 2, 3, HZ):
        V = np.max(np.outer(p, hist[t][:, 0]) + np.outer(1 - p, hist[t][:, 1]), axis=1)
        print(f"  V_{t}: en p = 0,5 vale {V[200]:.3f}; en p = 0 y 1, {V[0]:.3f}")
    # escala común para las curvas de horizonte 1, 2, 3 (sin descuento acumulado) y el infinito
    VV = {t: np.max(np.outer(p, hist[t][:, 0]) + np.outer(1 - p, hist[t][:, 1]), axis=1) for t in (1, 2, 3, HZ)}
    ymin = min(v.min() for v in VV.values()); ymax = max(v.max() for v in VV.values())
    print(f"  rango de valores para la figura: {ymin:.2f} a {ymax:.2f}")
    Yv = lin(-5, 30, 250, 30)
    if 1 in SECCIONES:
        for t in (1, 2, 3):
            print(f"path V{t}:", path(p, VV[t], X1, Yv))
        print("path Vinf:", path(p, VV[HZ], X1, Yv))
        # cada vector α del horizonte «infinito», como recta fina, solo donde vale más de -5
        for k, al in enumerate(G_inf):
            v = p * al[0] + (1 - p) * al[1]
            msk = v >= -5
            print(f"path alfa {k}:", path(p[msk][[0, -1]], v[msk][[0, -1]], X1, Yv))
        print(f"  marcas y: 0 -> {Yv(0):.1f}, 10 -> {Yv(10):.1f}, 20 -> {Yv(20):.1f}, 30 -> {Yv(30):.1f}")
    # política en el horizonte «infinito»: qué acción maximiza en cada p
    acc = np.array([accion_de(G_inf[np.argmax(G_inf @ np.array([pp, 1 - pp]))], hist[-HZ], GAMMA_T) for pp in p])
    cambios = [(p[i], acc[i - 1], acc[i]) for i in range(1, len(p)) if acc[i] != acc[i - 1]]
    nombres = {0: "escuchar", 1: "abrir izquierda", 2: "abrir derecha"}
    for pp, a0, a1 in cambios:
        print(f"  umbral en p ≈ {pp:.4f}: de «{nombres[a0]}» a «{nombres[a1]}»")
    # umbrales precisos por bisección sobre el valor de escuchar frente a abrir
    def q_vals(pp):
        b = np.array([pp, 1 - pp])
        out = []
        for a in range(NA):
            q = b @ R[:, a]
            for o in range(NO):
                bp = (b @ T[a]) * O[a][:, o]
                po = bp.sum()
                if po > 0:
                    q += GAMMA_T * po * np.max(G_inf @ (bp / po))
            out.append(q)
        return np.array(out)
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        qv = q_vals(mid)
        if qv[0] >= max(qv[1], qv[2]):
            lo = mid
        else:
            hi = mid
    UMBRAL = (lo + hi) / 2
    print(f"  umbral por bisección (Q de escuchar = Q de abrir): p = {UMBRAL:.4f} (y, por simetría, {1-UMBRAL:.4f})")
    print(f"  Q en p = 0,5: escuchar {q_vals(0.5)[0]:.3f}, abrir {q_vals(0.5)[1]:.3f}")

if 2 in SECCIONES:
    print("=" * 70)
    print("2. ACTUALIZACIÓN DEL BELIEF EN EL TIGRE")
    for n in range(0, 5):
        b = P_OIR ** n / (P_OIR ** n + (1 - P_OIR) ** n)
        print(f"  {n} veces más «izquierda» que «derecha»: b(tigre izquierda) = {b:.4f}  -> {'abrir derecha' if b > UMBRAL else 'escuchar'}")
    # una secuencia concreta
    seq = [0, 1, 0, 0]          # izq, der, izq, izq: tras la última, b supera el umbral y la política abre
    b = 0.5
    tray = [b]
    for o in seq:
        lik = np.array([O[0][0, o], O[0][1, o]])
        bb = np.array([b, 1 - b]) * lik
        b = bb[0] / bb.sum()
        tray.append(b)
    print("  secuencia izq, der, izq, izq:", " -> ".join(f"{x:.3f}" for x in tray))
    X2, Y2 = lin(0, 4, 60, 520), lin(0.4, 1, 250, 30)
    print("path tray:", path(range(5), tray, X2, Y2))
    print("puntos tray:", " ".join(f"{X2(k):.1f},{Y2(v):.1f}" for k, v in enumerate(tray)))
    print(f"path umbral: M{X2(0):.1f},{Y2(UMBRAL):.1f} L{X2(4):.1f},{Y2(UMBRAL):.1f}")
    print(f"  marcas y: 0,5 -> {Y2(0.5):.1f}, 0,85 -> {Y2(0.85):.1f}, 1 -> {Y2(1):.1f}; umbral {UMBRAL:.4f} -> {Y2(UMBRAL):.1f}")

# ====================================================================== QMDP Y EL CARTEL
if 3 in SECCIONES:
    print("=" * 70)
    print("3. QMDP FRENTE AL BELIEF MDP EXACTO: PASILLO CON UN CARTEL")
    L = 8                      # casillas 0..7; cartel en 0; bifurcación en 7; salida en 3
    CARTEL, BIF, SALIDA = 0, 7, 3
    G3 = 0.95
    R_OK, R_MAL, C_PASO = 10.0, -20.0, -0.1
    # estado oculto: lado del premio (0 izq, 1 der). Acciones: 0 mover a la izquierda, 1 a la derecha,
    # 2 elegir izquierda (solo en la bifurcación; terminal), 3 elegir derecha.
    def mover(x, a):
        return max(0, x - 1) if a == 0 else min(L - 1, x + 1)
    # MDP con el lado conocido: V*(x, lado)
    V = np.zeros((L, 2))
    for _ in range(2000):
        Vn = np.zeros_like(V)
        for x in range(L):
            for lado in range(2):
                qs = [C_PASO + G3 * V[mover(x, a), lado] for a in (0, 1)]
                if x == BIF:
                    qs += [R_OK if lado == 0 else R_MAL, R_OK if lado == 1 else R_MAL]
                Vn[x, lado] = max(qs)
        V = Vn
    def q_mdp(x, lado):
        qs = [C_PASO + G3 * V[mover(x, a), lado] for a in (0, 1)]
        qs += ([R_OK if lado == 0 else R_MAL, R_OK if lado == 1 else R_MAL] if x == BIF else [-np.inf, -np.inf])
        return np.array(qs)
    # belief MDP exacto: el cartel es perfecto, así que b ∈ {0,5; 1 (izq); 0 (der)} -> índice 0: ½, 1: izq, 2: der
    VB = np.zeros((L, 3))
    def sig_b(x2, bi):
        return (1 if True else 0) if False else bi
    for _ in range(2000):
        VBn = np.zeros_like(VB)
        for x in range(L):
            for bi in range(3):
                qs = []
                for a in (0, 1):
                    x2 = mover(x, a)
                    if bi == 0 and x2 == CARTEL:
                        v2 = 0.5 * VB[x2, 1] + 0.5 * VB[x2, 2]      # al llegar al cartel se ve el lado
                    else:
                        v2 = VB[x2, bi]
                    qs.append(C_PASO + G3 * v2)
                if x == BIF:
                    pizq = {0: 0.5, 1: 1.0, 2: 0.0}[bi]
                    qs += [pizq * R_OK + (1 - pizq) * R_MAL, (1 - pizq) * R_OK + pizq * R_MAL]
                VBn[x, bi] = max(qs)
        VB = VBn
    print(f"  valor exacto desde la salida (b = ½): {VB[SALIDA, 0]:.3f}")
    print(f"  valor si conociera el lado (MDP): {V[SALIDA].mean():.3f}; valor de adivinar en la bifurcación: {0.5*R_OK+0.5*R_MAL:.1f}")
    n_ir = BIF - SALIDA
    v_adivinar = sum(G3 ** k * C_PASO for k in range(n_ir)) + G3 ** n_ir * (0.5 * R_OK + 0.5 * R_MAL)
    print(f"  ir directo a la bifurcación ({n_ir} pasos) y adivinar: {v_adivinar:.3f}")
    # simular la política exacta y la QMDP
    def simular(politica, lado, max_pasos=60):
        x, bi, ret, g, tray = SALIDA, 0, 0.0, 1.0, [SALIDA]
        for t in range(max_pasos):
            a = politica(x, bi)
            if a >= 2:
                r = R_OK if (a - 2) == lado else R_MAL
                return ret + g * r, tray, t + 1, (a - 2) == lado
            x = mover(x, a)
            if x == CARTEL:
                bi = 1 + lado
            ret += g * C_PASO; g *= G3
            tray.append(x)
        return ret, tray, max_pasos, None
    def pol_exacta(x, bi):
        qs = []
        for a in (0, 1):
            x2 = mover(x, a)
            v2 = 0.5 * VB[x2, 1] + 0.5 * VB[x2, 2] if (bi == 0 and x2 == CARTEL) else VB[x2, bi]
            qs.append(C_PASO + G3 * v2)
        if x == BIF:
            pizq = {0: 0.5, 1: 1.0, 2: 0.0}[bi]
            qs += [pizq * R_OK + (1 - pizq) * R_MAL, (1 - pizq) * R_OK + pizq * R_MAL]
        return int(np.argmax(qs))
    def pol_qmdp(x, bi):
        pizq = {0: 0.5, 1: 1.0, 2: 0.0}[bi]
        q = pizq * q_mdp(x, 0) + (1 - pizq) * q_mdp(x, 1)
        return int(np.argmax(q))
    for nombre, pol in (("exacta", pol_exacta), ("QMDP", pol_qmdp)):
        rs = []
        for lado in (0, 1):
            ret, tray, pasos, ok = simular(pol, lado)
            rs.append(ret)
            print(f"  {nombre:6s} con el premio a la {'izquierda' if lado == 0 else 'derecha'}: retorno {ret:.3f} en {pasos} pasos; "
                  f"{'acierta' if ok else ('falla' if ok is False else 'no elige nunca')}; recorrido {tray[:16]}{'…' if len(tray) > 16 else ''}")
        print(f"  {nombre}: retorno medio {np.mean(rs):.3f}")
    print(f"  QMDP en la bifurcación con b = ½: Q = {np.round(0.5*q_mdp(BIF,0)+0.5*q_mdp(BIF,1), 3)} (mover izq., mover der., elegir izq., elegir der.)")

# ====================================================================== POMCP EN EL TIGRE
if 4 in SECCIONES:
    print("=" * 70)
    print("4. POMCP EN EL TIGRE FRENTE A LA POLÍTICA EXACTA")

    def generador(s, a, rng):
        r = R[s, a]
        if a == 0:
            s2 = s
            o = s if rng.random() < P_OIR else 1 - s
        else:
            s2 = int(rng.random() < 0.5)
            o = int(rng.random() < 0.5)
        return s2, o, r

    class POMCP:
        def __init__(self, n_sim, gamma=GAMMA_T, c=110.0, eps=0.01, rng=None, despliegue="aleatorio"):
            self.n_sim, self.g, self.c, self.desp = n_sim, gamma, c, despliegue
            self.prof = int(math.ceil(math.log(eps) / math.log(gamma)))
            self.rng = rng
            self.N, self.Na, self.Qa, self.B = {}, {}, {}, {}

        def rollout(self, s, d):
            ret, g = 0.0, 1.0
            while d < self.prof:
                a = int(self.rng.integers(NA)) if self.desp == "aleatorio" else 0     # «escuchar»: siempre escucha
                s, o, r = generador(s, a, self.rng)
                ret += g * r; g *= self.g; d += 1
            return ret

        def simular(self, s, h, d):
            if d >= self.prof:
                return 0.0
            if h not in self.N:
                self.N[h] = 0
                self.Na[h] = np.zeros(NA); self.Qa[h] = np.zeros(NA)
                return self.rollout(s, d)
            n, na, qa = self.N[h], self.Na[h], self.Qa[h]
            ucb = np.where(na == 0, np.inf, qa + self.c * np.sqrt(np.log(max(n, 1)) / np.maximum(na, 1)))
            a = int(np.argmax(ucb))
            s2, o, r = generador(s, a, self.rng)
            ret = r + self.g * self.simular(s2, h + (a, o), d + 1)
            self.B.setdefault(h + (a, o), []).append(s2) if d == 0 else None
            self.N[h] += 1; na[a] += 1; qa[a] += (ret - qa[a]) / na[a]
            return ret

        def planificar(self, particulas, h):
            for _ in range(self.n_sim):
                s = particulas[int(self.rng.integers(len(particulas)))]
                self.simular(s, h, 0)
            return int(np.argmax(self.Qa[h]))

    def episodio_pomcp(n_sim, rng, pasos=10, despliegue="aleatorio"):
        s = int(rng.random() < 0.5)
        pl = POMCP(n_sim, rng=rng, despliegue=despliegue)
        part = [int(rng.random() < 0.5) for _ in range(1000)]
        h = ()
        ret, g = 0.0, 1.0
        for t in range(pasos):
            pl.N, pl.Na, pl.Qa, pl.B = {}, {}, {}, {}    # árbol nuevo en cada paso (sin reutilizar el subárbol)
            a = pl.planificar(part, ())
            s2, o, r = generador(s, a, rng)
            ret += g * r; g *= GAMMA_T
            nuevas = pl.B.get((a, o), [])
            if len(nuevas) < 50:                         # reinvigoración: el belief verdadero tras abrir es ½
                nuevas = nuevas + ([int(rng.random() < 0.5) for _ in range(200)] if a != 0 else
                                   [x for x in part for _ in range(1) if (rng.random() < (P_OIR if x == o else 1 - P_OIR))][:200])
            part = nuevas if nuevas else [int(rng.random() < 0.5) for _ in range(200)]
            s = s2
        return ret

    def episodio_exacto(rng, pasos=10):
        s = int(rng.random() < 0.5)
        b = 0.5
        ret, g = 0.0, 1.0
        for t in range(pasos):
            a = 0 if UMBRAL_L <= b <= UMBRAL else (2 if b > UMBRAL else 1)
            s2, o, r = generador(s, a, rng)
            ret += g * r; g *= GAMMA_T
            if a == 0:
                lk = np.array([O[0][0, o], O[0][1, o]]); bb = np.array([b, 1 - b]) * lk; b = bb[0] / bb.sum()
            else:
                b = 0.5
            s = s2
        return ret

    UMBRAL_L = 1 - UMBRAL
    rng = np.random.default_rng(0)
    ex = np.array([episodio_exacto(rng) for _ in range(40000)])
    print(f"  política exacta (vectores α), 40 000 episodios de 10 pasos: retorno descontado {ex.mean():.3f} ± {ex.std()/math.sqrt(len(ex)):.3f}")
    res4 = {}
    for desp in ("aleatorio", "escuchar"):
        res4[desp] = []
        for n_sim in (4, 16, 64, 256, 1024):
            rng = np.random.default_rng(n_sim)
            t0 = time.perf_counter()
            n_ep = 300 if n_sim <= 256 else 150
            rr = np.array([episodio_pomcp(n_sim, rng, despliegue=desp) for _ in range(n_ep)])
            dt = time.perf_counter() - t0
            res4[desp].append((n_sim, rr.mean(), rr.std() / math.sqrt(len(rr))))
            print(f"  POMCP, despliegue {desp:9s}, {n_sim:5d} simulaciones por paso: retorno {rr.mean():8.3f} ± {rr.std()/math.sqrt(len(rr)):.3f} ({n_ep} episodios, {dt:.0f} s)", flush=True)
    todos = [r for v in res4.values() for r in v]
    X4 = lin(math.log2(4), math.log2(1024), 60, 520)
    lo4 = min(min(r[1] - 2 * r[2] for r in todos), ex.mean()) - 2
    hi4 = max(max(r[1] + 2 * r[2] for r in todos), ex.mean()) + 2
    print(f"  rango para la figura: {lo4:.1f} a {hi4:.1f}")
    Y4 = lin(lo4, hi4, 250, 30)
    for desp, res in res4.items():
        print(f"path pomcp {desp}:", path([math.log2(r[0]) for r in res], [r[1] for r in res], X4, Y4))
        print(f"path pomcp barras {desp}:", " ".join(f"M{X4(math.log2(r[0])):.1f},{Y4(r[1]-2*r[2]):.1f} L{X4(math.log2(r[0])):.1f},{Y4(r[1]+2*r[2]):.1f}" for r in res))
    print(f"path exacta: M60.0,{Y4(ex.mean()):.1f} L520.0,{Y4(ex.mean()):.1f}")
    paso_m = 50 if hi4 - lo4 > 120 else (20 if hi4 - lo4 > 50 else 10)
    print(f"  marcas del eje y: " + ", ".join(f"{v} -> {Y4(v):.1f}" for v in range(int(math.ceil(lo4 / paso_m)) * paso_m, int(hi4) + 1, paso_m)))

# ====================================================================== BÚSQUEDA CON UN DRON
if 5 in SECCIONES or 6 in SECCIONES:
    print("=" * 70)
    print("5. BÚSQUEDA CON UN DRON EN UNA REJILLA: DETECCIÓN IMPERFECTA Y BELIEF SOBRE EL OBJETIVO")
    NX = 6
    PD = 0.6                       # probabilidad de detectar si el objetivo está en la casilla
    MAXP = 80
    G5 = 0.98
    celdas = [(i, j) for i in range(NX) for j in range(NX)]
    idx = {c: k for k, c in enumerate(celdas)}
    MOV = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # prior no uniforme: el objetivo es más probable lejos de la salida (una «zona caliente»)
    prior = np.array([math.exp(-((i - 4) ** 2 + (j - 3) ** 2) / 6.0) + 0.15 for i, j in celdas])
    prior /= prior.sum()
    print(f"  rejilla {NX}×{NX}, P(detectar | está) = {PD}, sin falsas alarmas, salida en (0, 0); prior con una zona caliente en (4, 3)")

    def mover(c, a):
        i, j = c[0] + MOV[a][0], c[1] + MOV[a][1]
        return (min(max(i, 0), NX - 1), min(max(j, 0), NX - 1))

    def actualizar(b, c):
        """Belief tras mirar la casilla c y NO detectar."""
        b = b.copy(); b[idx[c]] *= (1 - PD); return b / b.sum()

    def dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def hacia(c, objetivo):
        best = None
        for a in range(4):
            c2 = mover(c, a)
            d = dist(c2, objetivo)
            if best is None or d < best[0]:
                best = (d, a)
        return best[1]

    def pol_barrido(c, b, t, estado):
        # recorrido en serpentina, repetido
        ruta = estado.setdefault("ruta", [(i, j if i % 2 == 0 else NX - 1 - j) for i in range(NX) for j in range(NX)])
        k = estado.get("k", 0)
        while ruta[k % len(ruta)] == c:
            k += 1
        estado["k"] = k
        return hacia(c, ruta[k % len(ruta)])

    def pol_max(c, b, t, estado):
        # ir a la casilla de mayor belief (desempate por distancia)
        punt = b - 1e-9 * np.array([dist(c, d) for d in celdas])
        return hacia(c, celdas[int(np.argmax(punt))])

    DIST = np.array([[dist(a_, b_) for b_ in celdas] for a_ in celdas], dtype=float)

    def pol_ganancia(c, b, t, estado):
        # mayor probabilidad de detección por paso de viaje: b(d)·PD / (1 + distancia)
        punt = b * PD / (1 + DIST[idx[c]])
        return hacia(c, celdas[int(np.argmax(punt))])

    class POMCPBusqueda:
        """POMCP con partículas = casilla del objetivo; la posición del dron es conocida."""
        def __init__(self, n_sim, rng, prof=30, c=3.0, despliegue="aleatorio"):
            self.n_sim, self.rng, self.prof, self.c, self.desp = n_sim, rng, prof, c, despliegue

        def rollout(self, c, s, d, b):
            """Despliegue desde la casilla c con el belief b del nodo (actualizado con los fallos del camino)."""
            ret, g = 0.0, 1.0
            while d < self.prof:
                if self.desp == "aleatorio":
                    a = int(self.rng.integers(4))
                else:                                   # heurística de ganancia por distancia sobre el belief de la raíz
                    a = pol_ganancia(c, b, 0, None)
                c = mover(c, a)
                if self.desp != "aleatorio":
                    b = actualizar(b, c)
                if c == s and self.rng.random() < PD:
                    return ret + g * 0.0
                ret += g * (-1.0); g *= G5; d += 1
            return ret

        def simular(self, c, s, h, d, b):
            if d >= self.prof:
                return 0.0
            if h not in self.N:
                self.N[h] = 0; self.Na[h] = np.zeros(4); self.Qa[h] = np.zeros(4)
                return self.rollout(c, s, d, b)
            n, na, qa = self.N[h], self.Na[h], self.Qa[h]
            ucb = np.where(na == 0, np.inf, qa + self.c * np.sqrt(np.log(max(n, 1)) / np.maximum(na, 1)))
            a = int(np.argmax(ucb))
            c2 = mover(c, a)
            det = (c2 == s) and (self.rng.random() < PD)
            if det:
                ret = 0.0
            else:
                b2 = actualizar(b, c2) if self.desp != "aleatorio" else b      # el despliegue aleatorio no usa el belief
                ret = -1.0 + G5 * self.simular(c2, s, h + (a,), d + 1, b2)
            self.N[h] += 1; na[a] += 1; qa[a] += (ret - qa[a]) / na[a]
            return ret

        def accion(self, c, b):
            self.N, self.Na, self.Qa = {}, {}, {}
            for _ in range(self.n_sim):
                s = celdas[int(self.rng.choice(len(celdas), p=b))]   # partícula muestreada del belief exacto
                self.simular(c, s, (), 0, b)
            return int(np.argmax(self.Qa[()]))

    def episodio(pol, rng, objetivo, guardar=False):
        c = (0, 0)
        b = prior.copy()
        estado = {}
        hist = []
        for t in range(MAXP):
            if c == objetivo and rng.random() < PD:
                return t, hist
            b = actualizar(b, c)
            if guardar:
                hist.append((c, b.copy()))
            a = pol(c, b, t, estado)
            c = mover(c, a)
        return MAXP, hist

    rng0 = np.random.default_rng(5)
    objetivos = [celdas[int(rng0.choice(len(celdas), p=prior))] for _ in range(600)]
    resultados = {}
    HEUR = (("barrido en serpentina", pol_barrido, 600), ("ir al máximo del belief", pol_max, 600),
            ("ganancia por distancia", pol_ganancia, 600))
    for nombre, pol, n_ep in (HEUR if 5 in SECCIONES else ()):
        rng = np.random.default_rng(11)
        T_ = np.array([episodio(pol, rng, objetivos[k])[0] for k in range(n_ep)])
        resultados[nombre] = T_
        print(f"  {nombre:24s}: pasos hasta detectar, media {T_.mean():5.1f} ± {T_.std()/math.sqrt(len(T_)):.1f}, mediana {np.median(T_):.0f}; "
              f"sin encontrar en {MAXP} pasos: {np.mean(T_ >= MAXP)*100:.1f} %")
    CONF = ((100, "aleatorio", 200), (500, "aleatorio", 200), (100, "ganancia", 150), (300, "ganancia", 150))
    for n_sim, desp, n_ep in (CONF if 5 in SECCIONES else ()):
        rng = np.random.default_rng(13)
        pl = POMCPBusqueda(n_sim, rng, despliegue=desp)
        pol = lambda c, b, t, e: pl.accion(c, b)
        t0 = time.perf_counter()
        T_ = np.array([episodio(pol, rng, objetivos[k])[0] for k in range(n_ep)])
        resultados[f"POMCP {n_sim} {desp}"] = T_
        print(f"  POMCP, {n_sim:4d} sim., desp. {desp:9s}: pasos hasta detectar, media {T_.mean():5.1f} ± {T_.std()/math.sqrt(len(T_)):.1f}, mediana {np.median(T_):.0f}; "
              f"sin encontrar en {MAXP} pasos: {np.mean(T_ >= MAXP)*100:.1f} % ({n_ep} episodios, {time.perf_counter()-t0:.0f} s)", flush=True)
    # belief para la figura: un episodio con el objetivo en una casilla poco probable y la política de ganancia
    rng = np.random.default_rng(3)
    _, hist = episodio(pol_ganancia, rng, (1, 5), guardar=True)
    for k in (0, 5, 15):
        if k < len(hist):
            c, b = hist[k]
            print(f"belief paso {k}: dron en {c}; " + " ".join(f"{x:.4f}" for x in b))
    print("prior: " + " ".join(f"{x:.4f}" for x in prior))

    if 6 in SECCIONES:
        print("=" * 70)
        print("6. DIAGNÓSTICO: ¿POR QUÉ EL DESPLIEGUE HEURÍSTICO NO AYUDA A POMCP CON POCAS SIMULACIONES?")
        b0 = actualizar(prior, (0, 0))
        rng = np.random.default_rng(21)
        for desp in ("aleatorio", "ganancia"):
            pl = POMCPBusqueda(100, rng, despliegue=desp)
            # dispersión de un despliegue desde la salida, sobre objetivos muestreados del belief
            rets = [pl.rollout((0, 0), celdas[int(rng.choice(len(celdas), p=b0))], 0, b0) for _ in range(2000)]
            print(f"  despliegue {desp:9s}: retorno de un despliegue desde la salida, media {np.mean(rets):6.2f}, desviación {np.std(rets):5.2f}")
            # valor de cada primera acción con muchas simulaciones (referencia) y elección con 100 simulaciones
            ref = POMCPBusqueda(5000, rng, despliegue=desp); ref.accion((0, 0), b0)
            elecciones = [POMCPBusqueda(100, rng, despliegue=desp).accion((0, 0), b0) for _ in range(40)]
            frec = np.bincount(elecciones, minlength=4) / len(elecciones)
            print(f"     Q de cada acción con 5000 simulaciones: {np.round(ref.Qa[()], 2)} (mejor {int(np.argmax(ref.Qa[()]))}); "
                  f"elección con 100 simulaciones en 40 repeticiones: {np.round(frec, 2)}", flush=True)

