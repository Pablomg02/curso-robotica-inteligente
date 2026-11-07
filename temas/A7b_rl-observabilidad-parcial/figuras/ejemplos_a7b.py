"""Ejemplos numéricos y datos de las figuras nuevas de A7b (deep RL bajo observabilidad parcial).

Cada número de un ejemplo hecho a mano en la presentación sale de aquí. Las cifras de los experimentos
(PPO con cuatro entradas, sondeo de la GRU entrenada) siguen saliendo de figuras_a7b.py (GPU), que no
hace falta relanzar: su salida está en salida_gpu.txt.

Ejecutar:  python3 ejemplos_a7b.py > salida_ejemplos.txt     (solo numpy, CPU, unos segundos)
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def titulo(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# ===========================================================================
titulo("1. Aliasing: dos estados con la misma observación (gamma = 0,9)")
# A: izquierda -> +1 y pasa a B; derecha -> 0 y se queda en A.  B: al revés.
G = 0.9


def valor_sin_memoria(p):
    """Política que elige «izquierda» con probabilidad p en cualquier estado (no los distingue).
    V_A = p (1 + G V_B) + (1-p) G V_A ;  V_B = (1-p)(1 + G V_A) + p G V_B."""
    M = np.array([[1 - (1 - p) * G, -p * G],
                  [-(1 - p) * G, 1 - p * G]])
    rhs = np.array([p, 1 - p])
    VA, VB = np.linalg.solve(M, rhs)
    return VA, VB, 0.5 * (VA + VB)


for p, nombre in [(1.0, "determinista «siempre izquierda»"), (0.5, "estocástica, ½ cada acción"),
                  (0.0, "determinista «siempre derecha»"), (0.3, "izquierda con 0,3")]:
    VA, VB, V = valor_sin_memoria(p)
    print(f"  {nombre:34s}: V_A = {VA:.3f}, V_B = {VB:.3f}, media = {V:.3f}")
print("  serie geométrica: 1/(1-G) =", 1 / (1 - G), "; ½/(1-G) =", 0.5 / (1 - G))
ps = np.linspace(0, 1, 101)
Vs = np.array([valor_sin_memoria(p)[2] for p in ps])
print(f"  máximo de V(p) en p = {ps[Vs.argmax()]:.2f}, V = {Vs.max():.3f}")
for p in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9):
    print(f"    V({p:.1f}) = {valor_sin_memoria(p)[2]:.3f}")
# con memoria: primero «izquierda»; desde A da +1 y llega a B; desde B da 0 y sigue en B.
# desde B sabe dónde está y alterna derecha, izquierda, ...: +1 por paso = 1/(1-G) = 10 desde el paso 1
cola = G * 1 / (1 - G)
print(f"  con memoria: desde A 1 + {cola:.1f} = {1 + cola:.1f}; desde B 0 + {cola:.1f} = {cola:.1f}; media {0.5 * (1 + cola) + 0.5 * cola:.2f}")
print("  estado visible: +1 en cada paso desde el primero = 1/(1-G) =", 1 / (1 - G))
print("  recompensas de los cinco primeros pasos y su valor descontado:")
for nombre, rs in [("determinista desde A", [1, 0, 0, 0, 0]), ("determinista desde B", [0, 0, 0, 0, 0]),
                   ("con memoria desde A", [1, 1, 1, 1, 1]), ("con memoria desde B", [0, 1, 1, 1, 1])]:
    print(f"    {nombre:22s}: {rs}  suma descontada de esos 5 = {sum(r * G ** t for t, r in enumerate(rs)):.3f}")
# figura V(p): eje x de 60 a 380 (p de 0 a 1), eje y de 215 a 40 (V de 0 a 10)
X, Y = lin(0, 1, 60, 380), lin(0, 10, 215, 40)
print("  path V(p):", path(ps[::5], Vs[::5], X, Y))
print(f"  punto p=0,5: ({X(0.5):.1f}, {Y(5):.1f}); p=1: ({X(1):.1f}, {Y(0.5):.1f}); p=0: ({X(0):.1f}, {Y(0.5):.1f})")
print(f"  rectas: con memoria 9,5 en y = {Y(9.5):.2f}; estado visible 10 en y = {Y(10):.2f}; marcas V = 0, 5, 10 en y = {Y(0):.1f}, {Y(5):.1f}, {Y(10):.1f}")

# ===========================================================================
titulo("2. Qué recibe la política en cada variante (dimensiones y parámetros, como figuras_a7b.py)")
NX = 6
NC = NX * NX
K = 8
H = 128
OBS = {"posicion": NC, "pila": NC * (K + 1), "gru": NC, "belief": 2 * NC}


def n_mlp(nin, nout):
    return (nin * H + H) + (H * H + H) + (H * nout + nout)


def n_gru(nin, nh):
    return 3 * ((nin + nh) * nh + nh)


for tipo, d in OBS.items():
    if tipo == "gru":
        na = n_gru(d, H) + n_mlp(H, 4)
        nc = n_gru(d, H) + n_mlp(H, 1)
    else:
        na, nc = n_mlp(d, 4), n_mlp(d, 1)
    print(f"  {tipo:8s}: entrada {d:4d} números; parámetros actor {na:6d}, crítico {nc:6d}")
print(f"  GRU sola (36 -> 128): 3 · ((36 + 128)·128 + 128) = {n_gru(NC, H)}")
print(f"  matriz de cada puerta: {NC + H} × {H}")

# ===========================================================================
titulo("3. Una GRU de una unidad, a mano (convención de figuras_a7b.py: h = (1-z) h + z h~)")
# entrada x = 1 si el dron está en la casilla que vigila esta unidad, 0 si no
wz, uz, bz = 4.0, 0.0, -3.0      # puerta de actualización: se abre al ver la casilla
wr, ur, br = 0.0, 0.0, 0.0       # puerta de reinicio: r = 0,5 siempre
wh, uh, bh = 3.0, 1.0, 0.0       # candidato


def gru1(h, x):
    z = sig(wz * x + uz * h + bz)
    r = sig(wr * x + ur * h + br)
    hc = math.tanh(wh * x + uh * r * h + bh)
    hn = (1 - z) * h + z * hc
    # derivada local dh_t / dh_{t-1} (con uz = ur = 0 las puertas no dependen de h)
    dh = (1 - z) + z * (1 - hc ** 2) * uh * r
    return hn, z, r, hc, dh


def elman1(h, x, w=3.0, u=0.5):
    hn = math.tanh(w * x + u * h)
    return hn, (1 - hn ** 2) * u


xs = [1] + [0] * 19
h, he = 0.0, 0.0
prod_g, prod_e = 1.0, 1.0
gru_tray, elm_tray = [], []
print("   t  x |   z      r      h~     h_t   dh_t/dh_t-1 | Elman h_t  dh/dh")
for t, x in enumerate(xs, 1):
    h, z, r, hc, dh = gru1(h, x)
    he, dhe = elman1(he, x)
    gru_tray.append(h); elm_tray.append(he)
    if t > 1:
        prod_g *= dh; prod_e *= dhe
    if t <= 4 or t in (10, 20):
        print(f"  {t:2d}  {x} | {z:.3f}  {r:.3f}  {hc:.3f}  {h:.3f}   {dh:.3f}      | {he:.4f}   {dhe:.3f}")
print(f"  tras 20 pasos: GRU h = {gru_tray[-1]:.3f}; Elman h = {elm_tray[-1]:.2e}")
print(f"  producto de derivadas locales de t = 2 a 20 (cuánto gradiente llega al paso 1): GRU {prod_g:.3f}; Elman {prod_e:.2e}")
print(f"  (1 - z) con x = 0: {1 - sig(bz):.4f};  (1 - z)^19 = {(1 - sig(bz)) ** 19:.3f}")
# figura: trayectorias de h en 20 pasos; eje x de 50 a 520 (t de 1 a 20), eje y de 190 a 30 (h de 0 a 1)
X, Y = lin(1, 20, 50, 520), lin(0, 1, 190, 30)
print("  path GRU:", path(range(1, 21), gru_tray, X, Y))
print("  path Elman:", path(range(1, 21), elm_tray, X, Y))

# ===========================================================================
titulo("4. PPO recurrente: las cuentas del lote (CFG de figuras_a7b.py)")
N, L, E, M, NUPD = 1024, 32, 4, 4, 500
print(f"  transiciones por actualización: N·L = {N}·{L} = {N * L}")
print(f"  minilote: {N // M} entornos × {L} pasos = {N // M * L} transiciones")
print(f"  pasos de gradiente por actualización: E·M = {E}·{M} = {E * M}")
print(f"  pasos de entorno en total: {N}·{L}·{NUPD} = {N * L * NUPD} ≈ {N * L * NUPD / 1e6:.1f} M")
for lon in (34.5, 35.6, 36.4):
    print(f"  con episodios de {lon} pasos: finales por tramo de {L} ≈ {L / lon:.2f}")

titulo("5. Reiniciar o no el estado oculto: la GRU de una unidad cruza un final de episodio")
# episodio 1: pasos 1-4, pasa por la casilla en el paso 2 y termina en el 4. Episodio 2: pasos 5-8, nunca la ve.
xs = [0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0]       # d_t = 1: el episodio termina en el paso t
hr, hn_ = 0.0, 0.0
print("   t  ep  x  d_{t-1} | h con reinicio | h sin reinicio")
for t in range(8):
    dprev = d[t - 1] if t > 0 else 0
    hr = 0.0 if dprev else hr
    hr = gru1(hr, xs[t])[0]
    hn_ = gru1(hn_, xs[t])[0]
    print(f"  {t + 1:2d}  {1 if t < 4 else 2}   {xs[t]}     {dprev}    |     {hr:.3f}      |     {hn_:.3f}")

titulo("6. Barajar pasos rompe el cociente de PPO antes de actualizar nada")
# política de dos acciones con logits (w·h, 0): pi(a=0 | h) = sig(w h)
w = 2.0
h_ok = 0.0
for x in [1, 0]:
    h_ok = gru1(h_ok, x)[0]
h_mal = gru1(0.0, 0)[0]            # el paso 2 recalculado desde h = 0, como si fuera el primero
p_ok, p_mal = sig(w * h_ok), sig(w * h_mal)
print(f"  paso 2 del episodio (vio la casilla en el paso 1): h correcto = {h_ok:.3f}, pi(a=0) = {p_ok:.3f}")
print(f"  recalculado desde cero (pasos barajados): h = {h_mal:.3f}, pi(a=0) = {p_mal:.3f}")
print(f"  cociente pi_nuevo / pi_viejo con los mismos parámetros = {p_mal / p_ok:.3f}  (debería ser 1; clip en [0,8; 1,2])")

# ===========================================================================
titulo("7. Memorias de POPGym: parámetros de cada recurrencia (entrada 36, estado 128)")
nin, nh = NC, H
print(f"  Elman: (36+128)·128 + 128 = {(nin + nh) * nh + nh}")
print(f"  GRU:   3 matrices      = {3 * ((nin + nh) * nh + nh)}")
print(f"  LSTM:  4 matrices      = {4 * ((nin + nh) * nh + nh)}  (cociente LSTM/GRU = {4 / 3:.3f})")
for Kv in (8, 32, 80, 1500):
    print(f"  atención con ventana de {Kv:5d} pasos: {Kv} productos q·k por paso, {Kv * Kv} por ventana completa")

# ===========================================================================
titulo("8. El belief de la búsqueda es función de cuántas veces se ha mirado cada casilla")
PD = 0.6
b0 = np.array([0.5, 0.3, 0.2])
b = b0.copy()
for c in [0, 1, 0]:
    b = b * (1 - PD * (np.arange(3) == c)); b /= b.sum()
    print(f"  mirar la casilla {c + 1} y no detectar: b = {b}")
n = np.array([2, 1, 0])
num = b0 * (1 - PD) ** n
print(f"  fórmula directa con n = {n}: numeradores {num}, suma {num.sum():.3f}, b = {num / num.sum()}")
print(f"  log b = log b0 + n·log(0,4) - log Z: log 0,4 = {math.log(0.4):.4f}; log b = {np.log(num / num.sum())}")

# ===========================================================================
titulo("9. Sondeo lineal a mano: mínimos cuadrados con regularización (lambda = 0,01)")
hs = np.array([-1.0, 0.0, 1.0, 2.0])         # una unidad del estado oculto en 4 pasos
B = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]])   # belief exacto de dos casillas
Xa = np.stack([hs, np.ones(4)], 1)
lam = 0.01
XtX = Xa.T @ Xa
XtB = Xa.T @ B
W = np.linalg.solve(XtX + lam * np.eye(2), XtB)
print("  X = [h, 1]:\n", Xa)
print("  XᵀX =\n", XtX, "\n  XᵀB =\n", XtB)
print("  W = (XᵀX + λI)⁻¹ XᵀB =\n", W)
Bh = Xa @ W
print("  belief leído X W =\n", Bh)
print("  suma de cada fila:", Bh.sum(1))
P = np.clip(Bh, 1e-6, None); P /= P.sum(1, keepdims=True)
tv = 0.5 * np.abs(P - B).sum(1)
print("  variación total por paso:", tv, " media", tv.mean())
ss_res = ((Bh - B) ** 2).sum(); ss_tot = ((B - B.mean(0)) ** 2).sum()
print(f"  R² = 1 - {ss_res:.4f} / {ss_tot:.4f} = {1 - ss_res / ss_tot:.4f}")
W0 = np.linalg.solve(XtX, XtB)
print("  sin regularizar (λ = 0): W =\n", W0)

titulo("10. Variación total: qué mide")
p = np.array([0.2, 0.3, 0.5]); q = np.array([0.25, 0.25, 0.5])
print(f"  p = {p}, q = {q}: ½ Σ|p - q| = {0.5 * np.abs(p - q).sum():.3f}")
import itertools
mejor = max(((abs(p[list(A)].sum() - q[list(A)].sum()), A) for r in range(4) for A in itertools.combinations(range(3), r)))
print(f"  mayor diferencia de probabilidad sobre un conjunto de casillas: {mejor[0]:.3f} en el conjunto {[i + 1 for i in mejor[1]]}")
u = np.ones(3) / 3
print(f"  uniforme frente a p: {0.5 * np.abs(p - u).sum():.3f};  p frente a una casilla segura (0,0,1): {0.5 * np.abs(p - np.array([0, 0, 1.0])).sum():.3f}")

# ===========================================================================
titulo("11. La trampa del sondeo, en pequeño: paseo aleatorio frente a un recorrido fijo (serpentina)")
CELDAS = [(i, j) for i in range(NX) for j in range(NX)]
MOV = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])
PRIOR = np.array([math.exp(-((i - 4) ** 2 + (j - 3) ** 2) / 6.0) + 0.15 for i, j in CELDAS])
PRIOR /= PRIOR.sum()
rng = np.random.default_rng(0)


SERP = []                                     # barrido en serpentina de A7a: filas alternas
for i in range(NX):
    SERP += [(i, j) for j in (range(NX) if i % 2 == 0 else range(NX - 1, -1, -1))]


def episodios(n_ep, modo, maxp=80):
    """Bucle de evaluación de figuras_a7b.py (mira; si no detecta, actualiza el belief; se mueve) con dos
    políticas de juguete: «paseo» (acción al azar) y «serpentina» (siempre el mismo recorrido)."""
    regs = []
    for k in range(n_ep):
        obj = rng.choice(NC, p=PRIOR)
        c = (0, 0); b = PRIOR.copy(); cuenta = np.zeros(NC); pila = [0] * K
        for t in range(maxp):
            ic = c[0] * NX + c[1]
            if ic == obj and rng.random() < PD:
                break
            b = b.copy(); b[ic] *= 1 - PD; b /= b.sum()
            cuenta = cuenta.copy(); cuenta[ic] += 1
            oh = np.zeros(NC); oh[ic] = 1
            regs.append((k, t, oh, b, cuenta, np.concatenate([np.eye(NC)[pila].reshape(-1), oh])))
            if modo == "paseo":
                a = rng.integers(4)
                c = (min(max(c[0] + MOV[a][0], 0), NX - 1), min(max(c[1] + MOV[a][1], 0), NX - 1))
            else:                               # siguiente casilla de la serpentina (al acabar, vuelve hacia el inicio)
                d = SERP[(SERP.index(c) + 1) % NC]
                c = (c[0] + int(np.sign(d[0] - c[0])), c[1]) if d[0] != c[0] else (c[0], c[1] + int(np.sign(d[1] - c[1])))
            pila = pila[1:] + [c[0] * NX + c[1]]
    return regs


def ortog(shape, g, rng):
    a = rng.standard_normal(shape)
    q, r = np.linalg.qr(a if shape[0] >= shape[1] else a.T)
    q = q * np.sign(np.diag(r))
    q = q if shape[0] >= shape[1] else q.T
    return g * q[:shape[0], :shape[1]]


def gru_aleatoria(regs, seed):
    """GRU de 128 unidades sin entrenar, iniciada como en figuras_a7b.py (ortogonal, ganancia 1, sesgos 0)."""
    r_ = np.random.default_rng(seed)
    Wz, Wr, Wh = (ortog((NC + H, H), 1.0, r_) for _ in range(3))
    out, zs, h, ep = [], [], np.zeros(H), None
    for rg in regs:
        if rg[0] != ep:
            h, ep = np.zeros(H), rg[0]
        x = rg[2]
        xh = np.concatenate([x, h])
        z = sig(xh @ Wz); r = sig(xh @ Wr)
        hc = np.tanh(np.concatenate([x, r * h]) @ Wh)
        h = (1 - z) * h + z * hc
        out.append(h); zs.append(z.mean())
    return np.array(out), float(np.mean(zs))


def sondeo(F, Bs, eps, n_tr, nombre):
    tr = eps < n_tr; te = ~tr
    Xa = np.concatenate([F, np.ones((len(F), 1))], 1)
    W = np.linalg.solve(Xa[tr].T @ Xa[tr] + 1e-2 * np.eye(Xa.shape[1]), Xa[tr].T @ Bs[tr])
    P = np.clip(Xa[te] @ W, 1e-6, None); P /= P.sum(1, keepdims=True)
    tv = 0.5 * np.abs(P - Bs[te]).sum(1).mean()
    r2 = 1 - ((Xa[te] @ W - Bs[te]) ** 2).sum() / ((Bs[te] - Bs[tr].mean(0)) ** 2).sum()
    print(f"  {nombre:46s}: {F.shape[1]:4d} rasgos; variación total {tv:.3f}; R² {r2:.3f}")
    return tv, r2


n_ep = 400
n_tr = int(0.7 * n_ep)
for modo in ("paseo", "serpentina"):
    regs = episodios(n_ep, modo)
    Bs = np.array([r[3] for r in regs]); eps = np.array([r[0] for r in regs])
    print(f"  [{modo}] {n_ep} episodios, {len(regs)} estados; ajuste con los episodios 0-{n_tr - 1}, "
          f"prueba con el resto ({(eps >= n_tr).sum()} estados)")
    tv_prior = 0.5 * np.abs(PRIOR[None] - Bs[eps >= n_tr]).sum(1).mean()
    print(f"  el prior fijo: variación total {tv_prior:.3f}")
    sondeo(np.array([r[2] for r in regs]), Bs, eps, n_tr, "solo la casilla actual")
    sondeo(np.array([r[5] for r in regs]), Bs, eps, n_tr, "las 8 últimas casillas y la actual")
    sondeo(np.array([r[4] for r in regs]), Bs, eps, n_tr, "cuántas veces se ha mirado cada casilla")
    for sd in (1, 2):
        Hr, zm = gru_aleatoria(regs, sd)
        sondeo(Hr, Bs, eps, n_tr, f"GRU sin entrenar de 128 unidades (semilla {sd})")
    print(f"  puerta z media de la GRU sin entrenar: {zm:.3f}")
lb = np.log(Bs)
print("  comprobación: log b - log b0 - n·log 0,4 es igual en todas las casillas (= -log Z):",
      np.allclose((lb - np.log(PRIOR)[None] - np.array([r[4] for r in regs]) * math.log(1 - PD)).std(1), 0, atol=1e-9))

# ===========================================================================
titulo("12. Percepción activa: qué casilla conviene mirar (regla de ganancia de A7a)")
for nombre, bc, dist in [("casilla cercana", 0.30, 1), ("casilla lejana", 0.50, 3)]:
    g = bc * PD / (1 + dist)
    print(f"  {nombre}: b = {bc}, distancia {dist}: P(detectar al llegar) = {bc * PD:.2f}; por paso de viaje {g:.3f}")
bc = 0.30
print(f"  si mira la cercana y no detecta: su belief pasa de {bc} a {bc * (1 - PD) / (1 - PD * bc):.3f}")
