"""Ejemplos numéricos y datos de las figuras nuevas de A7a (teoría de POMDPs).

Cada número de un ejemplo hecho a mano en la presentación sale de aquí. Las cifras de los
experimentos (POMCP, búsqueda con el dron) siguen saliendo de figuras_a7a.py, que no hace falta
relanzar.

Ejecutar:  python3 ejemplos_a7a.py > salida_ejemplos.txt     (solo numpy, CPU, segundos)
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def titulo(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# ---------------------------------------------------------------------------
# El tigre (Kaelbling, Littman y Cassandra, 1998, §5.1), como en figuras_a7a.py
# estados: 0 = tigre a la izquierda (TI), 1 = a la derecha (TD)
# acciones: 0 escuchar, 1 abrir izquierda, 2 abrir derecha
# observaciones: 0 se oye a la izquierda (OI), 1 a la derecha (OD)
# ---------------------------------------------------------------------------
NS, NA, NO = 2, 3, 2
P_OIR = 0.85
GAMMA = 0.95
R = np.array([[-1.0, -100.0, 10.0],
              [-1.0, 10.0, -100.0]])          # R[s, a]
T = np.zeros((NA, NS, NS))                    # T[a, s, s']
T[0] = np.eye(2)
T[1] = T[2] = 0.5
O = np.zeros((NA, NS, NO))                    # O[a, s', z]
O[0] = [[P_OIR, 1 - P_OIR], [1 - P_OIR, P_OIR]]
O[1] = O[2] = 0.5
ACC = ["escuchar", "abrir izq.", "abrir der."]
OBS = ["oye izq.", "oye der."]


def actualizar(b, a, z):
    """Devuelve (b_pred, sin_normalizar, P(z|a,b), b')."""
    bpred = b @ T[a]                          # Σ_s b(s) T(s,a,s')
    num = O[a][:, z] * bpred                  # O(s',a,z) · b̄(s')
    pz = num.sum()
    return bpred, num, pz, num / pz


# ===========================================================================
titulo("1. El tigre escrito como matrices")
print("R[s, a] (filas TI, TD; columnas escuchar, abrir izq., abrir der.):\n", R)
for a in range(NA):
    print(f"T(·, {ACC[a]}, ·) (fila s, columna s'):\n", T[a])
for a in (0, 1):
    print(f"O(·, {ACC[a]}, ·) (fila s', columna z = OI, OD):\n", O[a])

# ===========================================================================
titulo("2. Actualización del belief paso a paso (tigre)")
b = np.array([0.5, 0.5])
pasos = [("escuchar, oye izq.", 0, 0), ("escuchar, oye izq.", 0, 0)]
for nombre, a, z in pasos:
    bp, num, pz, b2 = actualizar(b, a, z)
    print(f"  desde b = {b}: {nombre}")
    print(f"     predicción  b̄ = b·T       = {bp}")
    print(f"     corrección  O·b̄           = {num}   (O = {O[a][:, z]})")
    print(f"     P(z|a,b)    = suma         = {pz:.4f}")
    print(f"     b'          = O·b̄ / P(z)   = {b2}")
    b = b2
print("  observación contraria:")
for b0 in (np.array([0.85, 0.15]), np.array([0.85 ** 2, 0.15 ** 2]) / (0.85 ** 2 + 0.15 ** 2)):
    bp, num, pz, b2 = actualizar(b0, 0, 1)
    print(f"     desde b = {b0}: escuchar, oye der. -> O·b̄ = {num}, P(z) = {pz:.4f}, b' = {b2}")
bp, num, pz, b2 = actualizar(np.array([0.9698, 0.0302]), 1, 0)
print(f"  abrir la izquierda desde b = (0,9698; 0,0302): b̄ = {bp}, O·b̄ = {num}, P(z) = {pz:.2f}, b' = {b2}")

titulo("3. Forma cerrada: cociente de probabilidades (odds)")
print("  b/(1-b) = (0,85/0,15)^n · b0/(1-b0), con n = (# oye izq.) - (# oye der.)")
r = P_OIR / (1 - P_OIR)
print(f"  0,85/0,15 = {r:.4f}  (= 17/3)")
for n in range(0, 5):
    odds = r ** n
    print(f"  n = {n}: odds = {odds:9.3f}, b = {odds/(1+odds):.4f}")
print(f"  umbral de la política 0,9603 -> odds {0.9603/(1-0.9603):.1f}: hace falta n >= 2")

# ===========================================================================
titulo("4. Árbol de dos niveles: el denominador es la probabilidad de cada rama")
b = np.array([0.5, 0.5])
tot = 0.0
filas = []
for z1 in range(NO):
    _, _, p1, b1 = actualizar(b, 0, z1)
    for z2 in range(NO):
        _, _, p2, b2 = actualizar(b1, 0, z2)
        filas.append((z1, z2, p1, b1[0], p2, b2[0], p1 * p2))
        tot += p1 * p2
for z1, z2, p1, b1, p2, b2, pp in filas:
    print(f"  {OBS[z1]} (P = {p1:.3f}, b = {b1:.3f}) -> {OBS[z2]} (P = {p2:.3f}, b = {b2:.4f}); camino: {pp:.4f}")
print(f"  suma de los cuatro caminos: {tot:.4f}")
print("  martingala: Σ_z P(z) b^z")
for b0 in (np.array([0.5, 0.5]), np.array([0.85, 0.15])):
    m = sum(actualizar(b0, 0, z)[2] * actualizar(b0, 0, z)[3] for z in range(NO))
    ps = [actualizar(b0, 0, z)[2] for z in range(NO)]
    bs = [actualizar(b0, 0, z)[3][0] for z in range(NO)]
    print(f"     desde b = {b0[0]:.2f}: {ps[0]:.3f}·{bs[0]:.4f} + {ps[1]:.3f}·{bs[1]:.4f} = {m[0]:.4f}")
# probabilidad de un camino con el tigre a la izquierda de verdad
print(f"  si el tigre está a la izquierda: P(OI) = {P_OIR}, P(OI, OI) = {P_OIR**2:.4f}; "
      f"promediando sobre el belief 0,5: {0.5*P_OIR**2+0.5*(1-P_OIR)**2:.4f}")

# ===========================================================================
titulo("5. Recompensa del belief MDP: ρ(b,a) = Σ_s b(s) R(s,a)")
for p in (0.5, 0.85, 0.97):
    bb = np.array([p, 1 - p])
    print(f"  b = ({p}; {1-p:.2f}): " + ", ".join(f"{ACC[a]} {bb @ R[:, a]:.2f}" for a in range(NA)))

# ===========================================================================
titulo("6. Cota del estado de información aproximado (ejemplo inventado)")
eps, delta, rho, H = 0.1, 0.05, 20.0, 10
alfa = 0.0
cotas = []
for t in range(H, 0, -1):
    alfa = eps + rho * delta + alfa
    cotas.append((t, alfa))
print(f"  ε = {eps}, δ = {delta}, ρ(V̂) = {rho} (constante), horizonte {H}")
for t, a_ in cotas[::-1][:3] + cotas[::-1][-2:]:
    print(f"   α_{t} = {a_:.2f}")
print(f"  pérdida máxima de la política voraz desde t = 1: 2·α_1 = {2*cotas[-1][1]:.1f}")
print(f"  si δ baja a 0,01: α_1 = {H*(eps+rho*0.01):.2f}")

# ===========================================================================
titulo("7. Vectores α de horizonte 1: un plan = una recta en p")
G1 = np.array([R[:, a] for a in range(NA)])      # α(TI), α(TD)
for a in range(NA):
    print(f"  plan «{ACC[a]}»: α = {G1[a]}, V(p) = {G1[a,1]:.0f} + ({G1[a,0]:.0f} - ({G1[a,1]:.0f}))·p")
# cruces escuchar / abrir derecha
p_x = (-1 - (-100)) / (10 - (-100))
print(f"  escuchar = abrir derecha cuando -1 = -100 + 110 p -> p = {p_x:.3f}; por simetría, 0,100")
for p in (0.0, 0.5, 0.9, 1.0):
    bb = np.array([p, 1 - p])
    print(f"   p = {p}: " + ", ".join(f"{ACC[a]} {G1[a] @ bb:.1f}" for a in range(NA)) + f"; V1 = {max(G1 @ bb):.1f}")
X7, Y7 = lin(0, 1, 60, 520), lin(-100, 10, 250, 30)
for a in range(NA):
    print(f"path alfa1 {ACC[a]}:", path([0, 1], [G1[a, 1], G1[a, 0]], X7, Y7))
pp = np.linspace(0, 1, 201)
V1 = np.max(np.outer(pp, G1[:, 0]) + np.outer(1 - pp, G1[:, 1]), axis=1)
print("path V1:", path(pp[[0, 20, 180, 200]], V1[[0, 20, 180, 200]], X7, Y7))
print(f"  marcas y: 10 -> {Y7(10):.1f}, 0 -> {Y7(0):.1f}, -50 -> {Y7(-50):.1f}, -100 -> {Y7(-100):.1f}; p=0,1 -> x {X7(0.1):.1f}, p=0,9 -> x {X7(0.9):.1f}")


# ===========================================================================
def envolvente_rejilla(alfas, n=20001):
    """Vectores que son máximos en algún p de una rejilla fina (solo para conjuntos pequeños)."""
    p = np.linspace(0, 1, n)
    V = np.outer(p, alfas[:, 0]) + np.outer(1 - p, alfas[:, 1])
    ganadores = np.unique(np.argmax(V, axis=1))
    return ganadores


titulo("8. Un backup exacto a mano: de Γ1 (3 vectores) a Γ2 (27 candidatos, 5 quedan)")
# g^{a,z}_α(s) = Σ_s' T(s,a,s') O(s',a,z) α(s')
nombres1 = ACC
for a in (0,):
    for z in range(NO):
        for k in range(NA):
            g = T[a] @ (O[a][:, z] * G1[k])
            print(f"  g^(escuchar, {OBS[z]})_(α={ACC[k]}) = {g}")
cands, etiquetas = [], []
for a in range(NA):
    for i in range(NA):
        for j in range(NA):
            g0 = T[a] @ (O[a][:, 0] * G1[i])
            g1 = T[a] @ (O[a][:, 1] * G1[j])
            cands.append(R[:, a] + GAMMA * (g0 + g1))
            etiquetas.append(f"{ACC[a]}; si OI -> {ACC[i]}, si OD -> {ACC[j]}")
cands = np.array(cands)
print(f"  candidatos: {len(cands)} = |A|·|Γ1|^|Z| = 3·3²")
ej = etiquetas.index("escuchar; si OI -> abrir der., si OD -> abrir izq.")
print(f"  ejemplo «escuchar; si OI abre der.; si OD abre izq.»: α = {cands[ej]}")
g0 = T[0] @ (O[0][:, 0] * G1[2]); g1 = T[0] @ (O[0][:, 1] * G1[1])
print(f"     = R(·,escuchar) + γ (g0 + g1) = {R[:,0]} + 0,95·({g0} + {g1}) = {R[:,0]} + 0,95·{g0+g1}")
win = envolvente_rejilla(cands)
print(f"  en la envolvente: {len(win)}")
pg = np.linspace(0, 1, 20001)
Vg = np.outer(pg, cands[:, 0]) + np.outer(1 - pg, cands[:, 1])
am = np.argmax(Vg, axis=1)
for k in win:
    zona = pg[am == k]
    print(f"   {etiquetas[k]:55s} α = {cands[k]}, máximo en p ∈ [{zona.min():.3f}, {zona.max():.3f}]")
print(f"  V2(0,5) = {Vg[10000].max():.3f}")
# cruces de la envolvente de Γ2
for k1, k2 in zip(win[:-1], win[1:]):
    pass

# ===========================================================================
titulo("9. Valor de la información en el tigre (horizonte infinito, γ = 0,95)")
# Beliefs alcanzables desde 0,5: b_n = 0,85^n / (0,85^n + 0,15^n), n entero; abrir vuelve a n = 0.
N = 30
ns = np.arange(-N, N + 1)
bn = 1.0 / (1.0 + ((1 - P_OIR) / P_OIR) ** ns)
V = np.zeros(len(ns))
for _ in range(3000):
    Vn = np.empty_like(V)
    for k, n in enumerate(ns):
        p = bn[k]
        pOI = p * P_OIR + (1 - p) * (1 - P_OIR)
        up = V[min(k + 1, len(ns) - 1)]; dn = V[max(k - 1, 0)]
        q_esc = -1 + GAMMA * (pOI * up + (1 - pOI) * dn)
        q_ai = p * -100 + (1 - p) * 10 + GAMMA * V[N]
        q_ad = p * 10 + (1 - p) * -100 + GAMMA * V[N]
        Vn[k] = max(q_esc, q_ai, q_ad)
    V = Vn
k0 = N
print(f"  V(0,5) = {V[k0]:.3f}; V(0,85) = V(0,15) = {V[k0+1]:.3f}; V(0,970) = {V[k0+2]:.3f}")
pOI = 0.5
EV = 0.5 * V[k0 + 1] + 0.5 * V[k0 - 1]
print(f"  esperanza tras escuchar: 0,5·V(0,85) + 0,5·V(0,15) = {EV:.3f} >= V(0,5) = {V[k0]:.3f}")
print(f"  Q(0,5, escuchar) = -1 + 0,95·{EV:.3f} = {-1 + GAMMA*EV:.3f}")
print(f"  Q(0,5, abrir)    = -45 + 0,95·{V[k0]:.3f} = {-45 + GAMMA*V[k0]:.3f}")
print(f"  ganancia de Jensen en p = 0,5: {EV - V[k0]:.3f}")

# envolvente V∞ para la figura de Jensen: iteración exacta con poda (como figuras_a7a.py)


def envolvente(alfas):
    alfas = np.unique(np.round(alfas, 9), axis=0)
    m = alfas[:, 0] - alfas[:, 1]
    c = alfas[:, 1]
    orden = np.lexsort((-c, m))
    hull = []
    for i in orden:
        if hull and abs(m[hull[-1]] - m[i]) < 1e-12:
            continue
        while len(hull) >= 2:
            j, k = hull[-2], hull[-1]
            if (c[j] - c[i]) / (m[i] - m[j]) <= (c[j] - c[k]) / (m[k] - m[j]):
                hull.pop()
            else:
                break
        hull.append(i)
    xs = [-np.inf] + [(c[hull[t]] - c[hull[t + 1]]) / (m[hull[t + 1]] - m[hull[t]]) for t in range(len(hull) - 1)] + [np.inf]
    keep = [hull[t] for t in range(len(hull)) if min(xs[t + 1], 1.0) - max(xs[t], 0.0) > 1e-9]
    return alfas[keep]


def backup_exacto(G):
    nuevos = []
    for a in range(NA):
        g = [np.array([T[a] @ (O[a][:, z] * al) for al in G]) for z in range(NO)]
        s = (g[0][:, None, :] + g[1][None, :, :]).reshape(-1, NS)
        nuevos.append(R[:, a] + GAMMA * s)
    return envolvente(np.vstack(nuevos))


G = np.zeros((1, NS))
for t in range(400):
    G = backup_exacto(G)
print(f"  vectores α tras 400 iteraciones: {len(G)}")
Vinf = lambda p: np.max(G @ np.array([p, 1 - p]))
print(f"  con los vectores: V∞(0,5) = {Vinf(0.5):.3f}, V∞(0,85) = {Vinf(0.85):.3f}, V∞(0,15) = {Vinf(0.15):.3f}")
# figura de Jensen: V∞ en [0, 1] y la cuerda entre 0,15 y 0,85
X9, Y9 = lin(0, 1, 60, 520), lin(15, 30, 250, 30)
pp = np.linspace(0, 1, 201)
print("path Vinf:", path(pp, [Vinf(x) for x in pp], X9, Y9))
print(f"  puntos: (0,15; {Vinf(0.15):.2f}) -> ({X9(0.15):.1f}, {Y9(Vinf(0.15)):.1f}); (0,85) -> ({X9(0.85):.1f}, {Y9(Vinf(0.85)):.1f}); "
      f"(0,5; V) -> ({X9(0.5):.1f}, {Y9(Vinf(0.5)):.1f}); media de la cuerda -> ({X9(0.5):.1f}, {Y9(EV):.1f})")
print(f"  marcas y: 15 -> {Y9(15):.1f}, 20 -> {Y9(20):.1f}, 25 -> {Y9(25):.1f}, 30 -> {Y9(30):.1f}")

# ===========================================================================
titulo("10. Las dos maldiciones con números (tigre: |A| = 3, |Z| = 2)")
for t in (1, 2, 3, 5, 10, 20):
    print(f"  t = {t:2d}: historias (|A||Z|)^t = {6**t:.3g}")
g_ = 1
for t in range(1, 6):
    g_ = 3 * g_ ** 2
    print(f"  sin podar, |Γ_{t}| = 3·|Γ_{t-1}|² = {g_:.3g}")
print("  con poda (figuras_a7a.py, sección 1): 3, 5, 9, 7, 13 vectores en los horizontes 1 a 5")
print("  dimensión del símplex: tigre 1; rejilla 6×6 de la parte 5: 35; Tag (870 estados): 869")

# ===========================================================================
titulo("11. QMDP en el pasillo con cartel, a mano (γ = 0,95, -0,1 por paso)")
g = GAMMA
c = -0.1
s = lambda n: c * (1 - g ** n) / (1 - g)          # coste de n pasos
print(f"  coste de n pasos: -0,1·(1-0,95^n)/0,05; n = 4: {s(4):.4f}; n = 10: {s(10):.4f}; n = 60: {s(60):.4f}")
print(f"  0,95^4 = {g**4:.4f}; 0,95^10 = {g**10:.4f}; 0,95^60 = {g**60:.4f}")
print(f"  conociendo el lado: 4 pasos y elegir bien: {s(4):.3f} + {g**4:.4f}·10 = {s(4) + g**4*10:.3f}")
print(f"  ir directo y adivinar: {s(4):.3f} + {g**4:.4f}·(-5) = {s(4) + g**4*(-5):.3f}")
print(f"  exacta: 3 pasos al cartel + 7 a la bifurcación y elegir bien: {s(10):.3f} + {g**10:.4f}·10 = {s(10) + g**10*10:.3f}")
print(f"  QMDP en la bifurcación con b = ½: Q(mover der.) = -0,1 + 0,95·10 = {c + g*10:.2f}; Q(elegir) = ½·10 + ½·(-20) = -5")
print(f"  QMDP se queda 60 pasos: {s(60):.3f}  (sin final, el límite es -0,1/0,05 = -2)")

# ===========================================================================
titulo("12. Backup en un punto (PBVI) en b = 0,5 desde Γ1")
bb = np.array([0.5, 0.5])
tot_ops = 0
mejores = {}
for a in range(NA):
    alfa_a = R[:, a].copy()
    for z in range(NO):
        vals = []
        for k in range(NA):
            gk = T[a] @ (O[a][:, z] * G1[k])
            vals.append((gk @ bb, k, gk))
            tot_ops += 1
        v, kbest, gbest = max(vals, key=lambda x: x[0])
        if a == 0:
            print(f"  a = escuchar, z = {OBS[z]}: g·b = " + ", ".join(f"{ACC[k]} {vv:.2f}" for vv, k, _ in vals) + f" -> elige {ACC[kbest]}")
        alfa_a = alfa_a + GAMMA * gbest
    mejores[a] = alfa_a
    print(f"  α para «{ACC[a]}»: {alfa_a}, valor en b = 0,5: {alfa_a @ bb:.3f}")
print(f"  productos g·b calculados: {tot_ops} (= |A|·|Z|·|Γ1|), frente a 27 candidatos completos")
print(f"  ganador en b = 0,5: escuchar, {mejores[0] @ bb:.3f} (exacto V2(0,5) = -1,950)")
print(f"  cota de PBVI en el tigre: (Rmax - Rmin)/(1-γ)² = 110/0,0025 = {110/(1-GAMMA)**2:.0f}; con ε_B = 0,01: {110*0.01/(1-GAMMA)**2:.0f}")
print(f"  rango real de V∞: de {Vinf(0.5):.2f} a {Vinf(0.0):.2f}")

# ===========================================================================
titulo("13. POMCP: la regla UCB con números (c = 110, como en figuras_a7a.py)")
Nh = 20
nodos = [("escuchar", 12, 5.0), ("abrir izq.", 4, -30.0), ("abrir der.", 4, -35.0)]
for nombre, na, q in nodos:
    bono = 110 * math.sqrt(math.log(Nh) / na)
    print(f"  {nombre:10s}: N(ha) = {na:2d}, Q = {q:6.1f}, bono = 110·sqrt(ln {Nh}/{na}) = {bono:5.1f}, UCB = {q + bono:6.1f}")
for cc in (1.0, 10.0):
    print(f"  con c = {cc}: " + ", ".join(f"{n} {q + cc*math.sqrt(math.log(Nh)/na):.1f}" for n, na, q in nodos))

titulo("14. POMCP: belief de partículas actualizado por rechazo (tigre)")
rng = np.random.default_rng(7)
part = (rng.random(1000) < 0.5).astype(int)        # 0 = TI, 1 = TD
print(f"  1000 partículas: TI {np.sum(part==0)}, TD {np.sum(part==1)}")
zs = np.where(rng.random(1000) < P_OIR, part, 1 - part)   # observación simulada de cada partícula al escuchar
kept = part[zs == 0]                                      # la real fue «oye izq.»
print(f"  la observación real es «oye izq.»: se quedan {len(kept)} partículas, de ellas TI {np.sum(kept==0)} "
      f"-> fracción {np.mean(kept==0):.3f} (Bayes exacto: 0,850)")
zs2 = np.where(rng.random(len(kept)) < P_OIR, kept, 1 - kept)
kept2 = kept[zs2 == 0]
print(f"  otra «oye izq.»: se quedan {len(kept2)}, fracción TI {np.mean(kept2==0):.3f} (exacto 0,970)")

titulo("15. POMCP en el tigre con 50 simulaciones: el árbol tras planificar desde b = ½")


class POMCPMini:
    def __init__(self, n_sim, rng, c=110.0, prof=20):
        self.n_sim, self.rng, self.c, self.prof = n_sim, rng, c, prof
        self.N, self.Na, self.Qa = {}, {}, {}

    def gen(self, s, a):
        if a == 0:
            return s, (s if self.rng.random() < P_OIR else 1 - s), R[s, a]
        return int(self.rng.random() < 0.5), int(self.rng.random() < 0.5), R[s, a]

    def rollout(self, s, d):
        ret, g_ = 0.0, 1.0
        while d < self.prof:
            s, z, r = self.gen(s, 0)
            ret += g_ * r; g_ *= GAMMA; d += 1
        return ret

    def sim(self, s, h, d):
        if d >= self.prof:
            return 0.0
        if h not in self.N:
            self.N[h] = 0; self.Na[h] = np.zeros(NA); self.Qa[h] = np.zeros(NA)
            return self.rollout(s, d)
        na, qa = self.Na[h], self.Qa[h]
        ucb = np.where(na == 0, np.inf, qa + self.c * np.sqrt(np.log(max(self.N[h], 1)) / np.maximum(na, 1)))
        a = int(np.argmax(ucb))
        s2, z, r = self.gen(s, a)
        ret = r + GAMMA * self.sim(s2, h + (a, z), d + 1)
        self.N[h] += 1; na[a] += 1; qa[a] += (ret - qa[a]) / na[a]
        return ret


pm = POMCPMini(50, np.random.default_rng(3))
for _ in range(50):
    pm.sim(int(pm.rng.random() < 0.5), (), 0)
print(f"  raíz: N = {pm.N[()]}, N(a) = {pm.Na[()]}, Q(a) = {np.round(pm.Qa[()], 1)}")
for z in range(NO):
    h = (0, z)
    if h in pm.N:
        print(f"  nodo (escuchar, {OBS[z]}): N = {pm.N[h]}, N(a) = {pm.Na[h]}, Q(a) = {np.round(pm.Qa[h], 1)}")
print(f"  nodos de historia en el árbol: {len(pm.N)}")

# ===========================================================================
titulo("16. DESPOT: tamaño del árbol completo frente al de K escenarios")
for A_, Z_, D_, K_ in ((3, 2, 10, 500), (4, 10, 5, 500)):
    full = (A_ * Z_) ** D_
    desp = A_ ** D_ * K_
    print(f"  |A| = {A_}, |Z| = {Z_}, D = {D_}, K = {K_}: completo (|A||Z|)^D = {full:.3g}; DESPOT |A|^D·K = {desp:.3g}; cociente {full/desp:.3g}")

titulo("17. Observaciones continuas: hijos de observación con y sin ensanchamiento progresivo")
for Nv in (10, 100, 1000):
    hijos = math.ceil(Nv ** 0.5)
    print(f"  {Nv:5d} visitas a un nodo de acción: sin ensanchar, {Nv} hijos de 1 visita; "
          f"con hijos <= N^0,5: {hijos} hijos, unas {Nv/hijos:.0f} visitas cada uno")

# ===========================================================================
titulo("18. ρ-POMDP: entropía negativa desplazada en el tigre")
for p in (0.5, 0.85, 0.97, 1.0):
    bb = np.array([p, 1 - p])
    h = -sum(x * math.log2(x) for x in bb if x > 0)
    print(f"  b = ({p}; {1-p:.2f}): entropía {h:.3f} bits, ρ_ent = log2 2 - H = {1 - h:.3f}")

# ===========================================================================
titulo("19. Búsqueda: mirar y no detectar (P_D = 0,6), con tres casillas")
PD = 0.6
b = np.array([0.5, 0.3, 0.2])
for paso in range(3):
    num = b * np.array([1 - PD, 1, 1])
    pz = num.sum()
    print(f"  mira la casilla 1 desde b = {b}: sin normalizar {num}, P(no detectar) = 1 - 0,6·{b[0]:.4f} = {pz:.4f}, b' = {num/pz}")
    b = num / pz
print(f"  factor de la casilla mirada: 0,4/(1 - 0,6·b), entre 0,4 (b -> 0) y 1 (b -> 1)")

# ===========================================================================
titulo("20. Densidad ε_B de un conjunto de puntos en el tigre")
Bp = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
pg = np.linspace(0, 1, 100001)
dmin = np.min(np.abs(pg[:, None] - Bp[None, :]), axis=1)
print(f"  B = {Bp} (en p): mayor distancia en p = {dmin.max():.4f}, en p = {pg[np.argmax(dmin)]:.3f}")
print(f"  en norma 1 sobre (p, 1-p) la distancia es 2|p - p'|: ε_B = {2*dmin.max():.3f}")
print(f"  cota: 110·{2*dmin.max():.2f}/0,05² = {110*2*dmin.max()/(1-GAMMA)**2:.0f}")

titulo("21. Partículas por rechazo: recuentos para la figura")
print(f"  inicio: {len(part)} (TI {np.sum(part==0)}, TD {np.sum(part==1)})")
print(f"  tras la 1.ª «oye izq.»: {len(kept)} (TI {np.sum(kept==0)}, TD {np.sum(kept==1)})")
print(f"  tras la 2.ª «oye izq.»: {len(kept2)} (TI {np.sum(kept2==0)}, TD {np.sum(kept2==1)})")

titulo("22. Curva de ρ_ent(p) para la figura")
X22, Y22 = lin(0, 1, 60, 520), lin(0, 1, 230, 30)
pp = np.linspace(0.0005, 0.9995, 201)
rh = [1 + x * math.log2(x) + (1 - x) * math.log2(1 - x) for x in pp]
print("path rho_ent:", path(np.r_[0, pp, 1], np.r_[1, rh, 1], X22, Y22))
for p in (0.5, 0.85, 0.97):
    print(f"  p = {p}: ({X22(p):.1f}, {Y22(1 + p*math.log2(p) + (1-p)*math.log2(1-p)):.1f})")
print(f"  pendiente dρ/dp = log2(p/(1-p)): en p = 0,99 vale {math.log2(0.99/0.01):.2f}; en 0,9999, {math.log2(0.9999/0.0001):.2f}")
