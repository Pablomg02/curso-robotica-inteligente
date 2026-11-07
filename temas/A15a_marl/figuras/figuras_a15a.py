"""Datos de las figuras y tablas del tema A15a (MARL).

Uso:  uv run --with "jax[cuda13]" --with optax python -u figuras_a15a.py > salida_gpu.txt     (GPU; unos minutos)
      Con A15A_RAPIDO=1, una prueba corta. En CPU funciona igual, más despacio.

1. Cuántas políticas conjuntas deterministas tiene un Dec-POMDP pequeño (por qué es NEXP-completo en la práctica).
2. Un juego matricial cooperativo con un óptimo aislado: VDN, una mezcla monótona (como QMIX) y una Q conjunta,
   entrenadas con datos uniformes; qué acción conjunta eligen al actuar de forma descentralizada.
3. Búsqueda cooperativa con tres drones en una rejilla de 12 × 12 celdas: el objetivo está en una celda según un mapa
   previo (dos zonas probables, al azar en cada episodio); cada dron mira la celda en la que está y lo detecta con
   probabilidad 0,7. Recompensa de equipo: la probabilidad de detectarlo en ese paso según el belief del equipo
   (la recompensa del belief-MDP de A7a), así que la suma de un episodio es la probabilidad de encontrarlo.
   Métodos: aleatorio, barrido por franjas, heurística voraz centralizada sobre el belief compartido, IPPO y MAPPO sin
   comunicación (cada dron solo sabe dónde ha mirado él) y MAPPO con comunicación (belief compartido y posiciones de
   los demás). Las observaciones son mapas centrados en cada dron; una variante repite MAPPO con comunicación con los
   mismos mapas sin centrar. La política con comunicación se prueba también con mensajes retrasados y sin mensajes.
"""
import math
import os
import time
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")   # resultados repetibles en la misma GPU
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax

print("dispositivo:", jax.devices()[0], "| jax", jax.__version__)
RAPIDO = bool(os.environ.get("A15A_RAPIDO"))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# ---------------------------------------------------------------------- 1. tamaño del espacio de políticas
print("=" * 70)
print("1. POLÍTICAS CONJUNTAS DETERMINISTAS DE UN DEC-POMDP (2 AGENTES, 2 ACCIONES, 2 OBSERVACIONES)")
for Tn in range(1, 7):
    nodos = (2 ** Tn - 1)                       # nodos de un árbol de política de horizonte T con |O| = 2
    log10_ag = nodos * math.log10(2)            # |A|^nodos políticas por agente
    print(f"  horizonte {Tn}: {nodos} nodos por árbol; políticas por agente 10^{log10_ag:.1f}; conjuntas 10^{2 * log10_ag:.1f}")

# ---------------------------------------------------------------------- 2. juego matricial
print("=" * 70)
print("2. JUEGO MATRICIAL COOPERATIVO (2 AGENTES, 3 ACCIONES), DATOS UNIFORMES SOBRE LAS 9 ACCIONES CONJUNTAS")
PAGO = jnp.array([[8.0, -12.0, -12.0], [-12.0, 0.0, 0.0], [-12.0, 0.0, 0.0]])
print("  matriz de pagos:", np.asarray(PAGO).tolist())


def q_tot(tipo, p):
    if tipo == "VDN":
        return p["q1"][:, None] + p["q2"][None, :]
    if tipo == "conjunta":
        return p["tab"]
    q1 = jnp.broadcast_to(p["q1"][:, None], (3, 3))
    q2 = jnp.broadcast_to(p["q2"][None, :], (3, 3))
    h = jax.nn.elu(jnp.abs(p["w1"])[0][None, None] * q1[..., None] + jnp.abs(p["w1"])[1][None, None] * q2[..., None] + p["b1"])
    return h @ jnp.abs(p["w2"]) + p["b2"]


for tipo in ("VDN", "mezcla monótona (QMIX)", "conjunta"):
    resul = []
    for semilla in range(5):
        k = jax.random.split(jax.random.PRNGKey(semilla), 6)
        p = {"q1": 0.1 * jax.random.normal(k[0], (3,)), "q2": 0.1 * jax.random.normal(k[1], (3,)),
             "w1": jax.random.normal(k[2], (2, 16)), "b1": 0.1 * jax.random.normal(k[3], (16,)),
             "w2": 0.3 * jax.random.normal(k[4], (16,)), "b2": jnp.zeros(()), "tab": 0.1 * jax.random.normal(k[5], (3, 3))}
        opt = optax.adam(1e-2)
        st = opt.init(p)
        perd = lambda p: jnp.mean((q_tot(tipo, p) - PAGO) ** 2)
        @jax.jit
        def bucle(p, st):
            def f(c, _):
                p, st = c
                l, g = jax.value_and_grad(perd)(p)
                u, st = opt.update(g, st, p)
                return (optax.apply_updates(p, u), st), l
            return lax.scan(f, (p, st), None, length=20000)
        (p, st), L = bucle(p, st)
        Q = np.asarray(q_tot(tipo, p))
        if tipo == "conjunta":
            a1, a2 = np.unravel_index(np.argmax(Q), Q.shape)
        else:                                    # cada agente elige con su propia Q (descentralizado)
            a1, a2 = int(np.argmax(np.asarray(p["q1"]))), int(np.argmax(np.asarray(p["q2"])))
        resul.append((a1, a2, float(PAGO[a1, a2]), float(L[-1]), Q))
    pagos = [r[2] for r in resul]
    print(f"  {tipo:24s}: acción conjunta elegida en 5 semillas {[(int(r[0]), int(r[1])) for r in resul]} | pago medio {np.mean(pagos):.1f} | "
          f"error cuadrático final {np.mean([r[3] for r in resul]):.2f}")
    print(f"    Q_tot aprendida (semilla 0): {np.round(resul[0][4], 1).tolist()}")

# ---------------------------------------------------------------------- 3. búsqueda cooperativa
H = W = 12
C = H * W
N = 3
T = 80
PD = 0.7
QN = 1.0 - PD
MOV = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], dtype=jnp.int32)    # quieto, +y, -y, +x, -x
CX = jnp.arange(C) % W
CY = jnp.arange(C) // W
GAMMA, LAM, ESC_R = 0.99, 0.95, 10.0
R_EGO = W - 1                     # vista egocéntrica de 23 x 23 celdas centrada en el dron: ve toda la rejilla
L_EGO = 2 * R_EGO + 1
D_OBS = 3 * L_EGO * L_EGO + N + 3
D_ABS = 3 * C + N + 3              # variante con mapas absolutos (posición propia, belief y demás drones, sin centrar)
DENTRO = jnp.ones(C)


def mapa_previo(key):
    """Dos zonas probables (gaussianas de 1,5 celdas) en sitios al azar, más un 20 % repartido por igual."""
    cen = jax.random.uniform(key, (2, 2), minval=2.0, maxval=10.0)
    d2 = (CX[None] - cen[:, :1]) ** 2 + (CY[None] - cen[:, 1:]) ** 2
    g = jnp.exp(-d2 / (2 * 1.5 ** 2)).sum(0)
    return 0.8 * g / g.sum() + 0.2 / C


def estado_inicial(key):
    """Todos los drones salen de la esquina (0, 0)."""
    return (jnp.zeros((N, 2), jnp.int32), jnp.zeros((N, C), jnp.int32), jnp.zeros(C, jnp.int32), mapa_previo(key), jnp.zeros((), jnp.int32))


def paso(st, a):
    pos, co, ca, pr, t = st
    m = pr * QN ** ca                                   # P(objetivo en c y aún no detectado)
    pos = jnp.clip(pos + MOV[a], 0, W - 1)
    oh = jax.nn.one_hot(pos[:, 1] * W + pos[:, 0], C, dtype=jnp.int32)
    k = oh.sum(0)
    r = jnp.sum(m * (1.0 - QN ** k))                    # probabilidad de detectarlo en este paso
    return (pos, co + oh, ca + k, pr, t + 1), r


def ego(mapa, p):
    """Recorta el mapa (C,) en una ventana de L_EGO x L_EGO centrada en la celda p (ceros fuera de la rejilla)."""
    m = jnp.pad(mapa.reshape(H, W), R_EGO)
    return lax.dynamic_slice(m, (p[1], p[0]), (L_EGO, L_EGO)).reshape(-1)


def observacion(st, comm, otros_cnt=None, otros_pos=None, centrada=True):
    """(N, D_OBS), todo en coordenadas centradas en cada dron: belief (propio o compartido), posiciones de los demás y
    máscara de la rejilla; más identidad, tiempo y posición absoluta."""
    pos, co, ca, pr, t = st
    oh = jax.nn.one_hot(pos[:, 1] * W + pos[:, 0], C)
    if otros_cnt is None:
        otros_cnt = (ca[None] - co) if comm else jnp.zeros_like(co)
        otros_pos = (oh.sum(0)[None] - oh) if comm else jnp.zeros_like(oh)
    m = pr[None] * QN ** (co + otros_cnt)
    b = m / m.sum(-1, keepdims=True) * C
    if not centrada:
        return jnp.concatenate([oh, b, otros_pos, jnp.eye(N), jnp.full((N, 1), t / T), pos / (W - 1.0)], -1)
    eg = jax.vmap(lambda bb, oo, p: jnp.concatenate([ego(bb, p), ego(oo, p), ego(DENTRO, p)]))(b, otros_pos, pos)
    return jnp.concatenate([eg, jnp.eye(N), jnp.full((N, 1), t / T), pos / (W - 1.0)], -1)


def mlp_init(key, sizes, ult):
    ks = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (a, b)) * math.sqrt(2.0 / a) * (1.0 if i < len(sizes) - 2 else ult), jnp.zeros(b))
            for i, (k, a, b) in enumerate(zip(ks, sizes[:-1], sizes[1:]))]


def mlp(ps, x):
    for Wm, b in ps[:-1]:
        x = jnp.tanh(x @ Wm + b)
    Wm, b = ps[-1]
    return x @ Wm + b


E_TR = 256 if RAPIDO else 512
N_UPD = int(os.environ.get("A15A_UPD", 40 if RAPIDO else 600))
EPOCAS, N_MB, CLIP, ENT, LR = 4, 4, 0.2, 0.01, 5e-4
opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(LR))


def hacer_actualizacion(comm, central, centrada=True):
    def rollout(params, key):
        pa, pc = params
        k0, k1 = jax.random.split(key)
        st = jax.vmap(estado_inicial)(jax.random.split(k0, E_TR))
        def f(st, k):
            o = jax.vmap(lambda s: observacion(s, comm, centrada=centrada))(st)
            g = jax.vmap(lambda s: observacion(s, True, centrada=centrada))(st) if central else o
            lg = mlp(pa, o)
            a = jax.random.categorical(k, lg)
            lp = jnp.take_along_axis(jax.nn.log_softmax(lg), a[..., None], -1)[..., 0]
            v = mlp(pc, g)[..., 0]
            st, r = jax.vmap(paso)(st, a)
            return st, (o, g, a, lp, v, r)
        _, tr = lax.scan(f, st, jax.random.split(k1, T))
        return tr

    def perdida(params, o, g, a, lp0, adv, ret):
        pa, pc = params
        lps = jax.nn.log_softmax(mlp(pa, o))
        lp = jnp.take_along_axis(lps, a[..., None], -1)[..., 0]
        rt = jnp.exp(lp - lp0)
        an = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.mean(jnp.minimum(rt * an, jnp.clip(rt, 1 - CLIP, 1 + CLIP) * an))
        vl = jnp.mean((mlp(pc, g)[..., 0] - ret) ** 2)
        ent = -jnp.mean(jnp.sum(jnp.exp(lps) * lps, -1))
        return pg + 0.5 * vl - ENT * ent

    @jax.jit
    def actualizar(params, st_opt, key):
        k1, k2 = jax.random.split(key)
        o, g, a, lp, v, r = rollout(params, k1)
        r = r * ESC_R
        vn = jnp.concatenate([v[1:], jnp.zeros_like(v[:1])])
        d = r[..., None] + GAMMA * vn - v
        def fg(acc, x):
            acc = x + GAMMA * LAM * acc
            return acc, acc
        _, adv = lax.scan(fg, jnp.zeros_like(d[0]), d[::-1])
        adv = adv[::-1]
        ret = adv + v
        n = T * E_TR * N
        datos = [x.reshape((n,) + x.shape[3:]) for x in (o, g, a, lp, adv, ret)]
        def epoca(c, k):
            params, st_opt = c
            idx = jax.random.permutation(k, n).reshape(N_MB, -1)
            def mb(c, i):
                params, st_opt = c
                l, gr = jax.value_and_grad(perdida)(params, *[x[i] for x in datos])
                u, st_opt = opt.update(gr, st_opt, params)
                return (optax.apply_updates(params, u), st_opt), l
            return lax.scan(mb, (params, st_opt), idx)
        (params, st_opt), L = lax.scan(epoca, (params, st_opt), jax.random.split(k2, EPOCAS))
        return params, st_opt, L.mean(), r.sum(0).mean() / ESC_R
    return actualizar


_EVAL = {}


def evaluar(accion, claves, retraso=0, solo_propio=False, pa=None):
    """Despliega 'accion(pa, st, k, otros_cnt, otros_pos) -> a' en un episodio por clave. Devuelve P(detectar antes de
    T), pasos esperados hasta detectarlo (con tope T) y solapamiento (fracción de celdas vistas por más de un dron).
    Los parámetros de la red se pasan como argumento: la función compilada se reutiliza y no los lleva dentro."""
    clave = (accion, retraso, solo_propio)
    if clave not in _EVAL:
        _EVAL[clave] = jax.jit(jax.vmap(lambda pa, key: _episodio(accion, retraso, solo_propio, pa, key), (None, 0)))
    pdet, tesp, sol = _EVAL[clave](pa, claves)
    return float(pdet.mean()), float(tesp.mean()), float(sol.mean())


def _episodio(accion, retraso, solo_propio, pa, key):
    k0, k1 = jax.random.split(key)
    st = estado_inicial(k0)
    D = max(retraso, 1)
    hist = (jnp.zeros((D, N, C), jnp.int32), jnp.zeros((D, N, C)))
    def f(c, inp):
        st, hist = c
        t, k = inp
        pos, co, ca, pr, _ = st
        oh = jax.nn.one_hot(pos[:, 1] * W + pos[:, 0], C)
        oc, op = ca[None] - co, oh.sum(0)[None] - oh
        if retraso > 0:                                  # mensajes con 'retraso' pasos de antigüedad
            j = t % D
            oc_r, op_r = hist[0][j], hist[1][j]
            hist = (hist[0].at[j].set(oc), hist[1].at[j].set(op))
            oc, op = oc_r, op_r
        if solo_propio:
            oc, op = jnp.zeros_like(oc), jnp.zeros_like(op)
        masa = jnp.sum(pr * QN ** ca)
        a = accion(pa, st, k, oc, op)
        st, r = paso(st, a)
        return (st, hist), (r, masa)
    (stT, _), (r, masa) = lax.scan(f, (st, hist), (jnp.arange(T), jax.random.split(k1, T)))
    vistos = (stT[1] > 0).sum(0)
    return r.sum(), masa.sum(), jnp.sum(vistos >= 2) / jnp.maximum(jnp.sum(vistos >= 1), 1)


_POLS = {}


def accion_politica(comm, voraz=False, centrada=True):
    if (comm, voraz, centrada) not in _POLS:
        def acc(pa, st, k, oc, op):
            lg = mlp(pa, observacion(st, comm, oc, op, centrada))
            return jnp.argmax(lg, -1) if voraz else jax.random.categorical(k, lg)
        _POLS[(comm, voraz, centrada)] = acc
    return _POLS[(comm, voraz, centrada)]


def accion_aleatoria(pa, st, k, oc, op):
    return jax.random.randint(k, (N,), 0, 5)


def accion_voraz(pa, st, k, oc, op):
    """Centralizada con el belief compartido: cada dron, por turnos, elige la celda con más masa descontada por la
    distancia (0,9 por celda), rebaja esa celda para los siguientes y da un paso hacia ella."""
    pos, co, ca, pr, t = st
    b = pr * QN ** ca
    b = b / b.sum()
    def f(bb, j):
        d = jnp.abs(CX - pos[j, 0]) + jnp.abs(CY - pos[j, 1])
        c = jnp.argmax(bb * 0.9 ** d)
        dx, dy = jnp.sign(CX[c] - pos[j, 0]), jnp.sign(CY[c] - pos[j, 1])
        a = jnp.where(dx > 0, 3, jnp.where(dx < 0, 4, jnp.where(dy > 0, 1, jnp.where(dy < 0, 2, 0))))
        return bb.at[c].multiply(QN), a
    return lax.scan(f, b, jnp.arange(N))[1]


def plan_franjas():
    """Barrido en zigzag: el dron i recorre las columnas 4i a 4i+3, ida y vuelta, y repite."""
    acts = np.zeros((T, N), np.int32)
    for i in range(N):
        cols = list(range(4 * i, 4 * i + 4))
        ruta = []
        for rep in range(4):
            cs = cols if rep % 2 == 0 else cols[::-1]
            for j, cx in enumerate(cs):
                ys = range(H) if (j + rep) % 2 == 0 else range(H - 1, -1, -1)
                ruta += [(cx, y) for y in ys]
        x, y, w = 0, 0, 0
        for t in range(T):
            while w < len(ruta) and ruta[w] == (x, y):
                w += 1
            tx, ty = ruta[min(w, len(ruta) - 1)]
            a = 3 if tx > x else 4 if tx < x else 1 if ty > y else 2 if ty < y else 0
            x, y = x + [0, 0, 0, 1, -1][a], y + [0, 1, -1, 0, 0][a]
            acts[t, i] = a
    return jnp.asarray(acts)


ACT_FR = plan_franjas()


def accion_franjas(pa, st, k, oc, op):
    return ACT_FR[st[4]]


print("=" * 70)
print(f"3. BÚSQUEDA COOPERATIVA: {N} DRONES, REJILLA {H}x{W}, DETECCIÓN {PD}, {T} PASOS")
N_EV = 400 if RAPIDO else 2000
claves_ev = jax.random.split(jax.random.PRNGKey(9999), N_EV)
claves_curva = jax.random.split(jax.random.PRNGKey(8888), 256)
RES = {}
for nombre, acc in (("aleatorio", accion_aleatoria), ("barrido por franjas", accion_franjas),
                    ("voraz centralizada (belief compartido)", accion_voraz)):
    RES[nombre] = evaluar(acc, claves_ev)
    print(f"  {nombre:40s}: P(encontrarlo en {T} pasos) {RES[nombre][0]:.3f} | pasos esperados {RES[nombre][1]:.1f} | solapamiento {RES[nombre][2]:.2f}")

METODOS = (("IPPO sin comunicación", False, False, True), ("MAPPO sin comunicación", False, True, True),
           ("MAPPO con comunicación", True, True, True), ("MAPPO con comunicación, mapas absolutos", True, True, False))
SEMILLAS = (0,) if RAPIDO else (0, 1, 2)
CADA = 10 if RAPIDO else 20
CURVAS, POL = {}, {}
for nombre, comm, central, centrada in METODOS:
    actualizar = hacer_actualizacion(comm, central, centrada)
    D = D_OBS if centrada else D_ABS
    res_s, cur_s = [], []
    for s in SEMILLAS:
        t0 = time.perf_counter()
        k = jax.random.split(jax.random.PRNGKey(100 + s), 3)
        params = (mlp_init(k[0], [D, 256, 256, 5], 0.01), mlp_init(k[1], [D, 256, 256, 1], 1.0))
        st_opt = opt.init(params)
        cur = []
        clave = k[2]
        for u in range(N_UPD):
            clave, ku = jax.random.split(clave)
            params, st_opt, L, R = actualizar(params, st_opt, ku)
            if (u + 1) % CADA == 0 or u == 0:
                cur.append((u + 1, evaluar(accion_politica(comm, centrada=centrada), claves_curva, pa=params[0])[0]))
        ev = evaluar(accion_politica(comm, centrada=centrada), claves_ev, pa=params[0])
        ev_det = evaluar(accion_politica(comm, voraz=True, centrada=centrada), claves_ev, pa=params[0])
        extra = ()
        if comm and centrada:
            extra = (evaluar(accion_politica(comm, centrada=centrada), claves_ev, retraso=10, pa=params[0]),
                     evaluar(accion_politica(comm, centrada=centrada), claves_ev, solo_propio=True, pa=params[0]))
        res_s.append((ev, extra, ev_det))
        cur_s.append(cur)
        POL[(nombre, s)] = params[0]
        print(f"    {nombre}, semilla {s}: P {ev[0]:.3f} | pasos {ev[1]:.1f} | solapamiento {ev[2]:.2f} | eligiendo la acción más probable P {ev_det[0]:.3f}"
              + (f" | con mensajes de hace 10 pasos P {extra[0][0]:.3f} | sin mensajes P {extra[1][0]:.3f}" if extra else "")
              + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    ev = np.mean([r[0] for r in res_s], 0)
    RES[nombre] = tuple(ev)
    CURVAS[nombre] = (np.array([c[0] for c in cur_s[0]]), np.mean([[c[1] for c in cs] for cs in cur_s], 0))
    linea = f"  {nombre:40s}: P(encontrarlo en {T} pasos) {ev[0]:.3f} | pasos esperados {ev[1]:.1f} | solapamiento {ev[2]:.2f}"
    linea += f" | rango entre semillas de P {min(r[0][0] for r in res_s):.3f}-{max(r[0][0] for r in res_s):.3f}"
    linea += f" | eligiendo la acción más probable P {np.mean([r[2][0] for r in res_s]):.3f}"
    if comm and centrada:
        e1 = np.mean([r[1][0] for r in res_s], 0)
        e2 = np.mean([r[1][1] for r in res_s], 0)
        RES[nombre + ", mensajes de hace 10 pasos"] = tuple(e1)
        RES[nombre + ", sin mensajes"] = tuple(e2)
        linea += f"\n  {nombre + ', mensajes de hace 10 pasos':40s}: P {e1[0]:.3f} | pasos esperados {e1[1]:.1f} | solapamiento {e1[2]:.2f}"
        linea += f"\n  {nombre + ', sin mensajes':40s}: P {e2[0]:.3f} | pasos esperados {e2[1]:.1f} | solapamiento {e2[2]:.2f}"
    print(linea, flush=True)

# ---------------------------------------------------------------------- figuras
print("=" * 70)
X = lin(0, N_UPD, 60, 520)
Y = lin(0.0, 1.0, 250, 30)
for nombre in CURVAS:
    xs, ys = CURVAS[nombre]
    print(f"path curva {nombre}:", path(xs, ys, X, Y))
for nombre in ("aleatorio", "barrido por franjas", "voraz centralizada (belief compartido)"):
    print(f"nivel {nombre}: y = {Y(RES[nombre][0]):.1f}")

# un episodio de ejemplo: mapa previo y recorridos (sin comunicación frente a con comunicación)
k0 = jax.random.PRNGKey(31)
st0 = estado_inicial(k0)
pr = np.asarray(st0[3])
print("mapa previo:", " ".join(f"{v / pr.max():.2f}" for v in pr))
for nombre, comm in (("MAPPO sin comunicación", False), ("MAPPO con comunicación", True), ("voraz centralizada (belief compartido)", None)):
    acc = accion_voraz if comm is None else accion_politica(comm)
    pa = None if comm is None else POL[(nombre, 0)]
    st = st0
    P = [np.asarray(st[0])]
    k = jax.random.PRNGKey(5)
    for t in range(T):
        k, ka = jax.random.split(k)
        pos, co, ca, prr, _ = st
        oh = jax.nn.one_hot(pos[:, 1] * W + pos[:, 0], C)
        a = acc(pa, st, ka, ca[None] - co, oh.sum(0)[None] - oh)
        st, _ = paso(st, a)
        P.append(np.asarray(st[0]))
    P = np.array(P)
    XC, YC = lin(-0.5, W - 0.5, 0, 216), lin(-0.5, H - 0.5, 216, 0)
    for i in range(N):
        print(f"path recorrido {nombre} {i}:", path(P[:, i, 0] + 0.08 * (i - 1), P[:, i, 1] + 0.08 * (i - 1), XC, YC))
    print(f"  {nombre}: P(encontrarlo) en este episodio {1 - float(jnp.sum(st[3] * QN ** st[2])):.3f}")
