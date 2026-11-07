"""Datos de las figuras y tablas del tema A15b (enjambres y multi-robot real).

Uso:  uv run --with "jax[cuda13]" --with optax python -u figuras_a15b.py > salida_gpu.txt     (GPU; unos 40 minutos)
      Con A15B_RAPIDO=1, una prueba corta. En CPU funciona igual, mucho más despacio.

La búsqueda cooperativa de A15a (tres drones, rejilla de 12 x 12, mapa previo con dos zonas probables, detección con
probabilidad 0,7, recompensa del belief-MDP), ahora con compañeros que no son los de entrenamiento y con cosas que fallan:
  1. Equipos entrenados por autojuego con MAPPO, sin y con comunicación, 4 semillas cada uno. Juego cruzado: el dron 0
     de un equipo con los drones 1 y 2 de otro (coordinación sin acuerdo previo).
  2. Un dron que deja de funcionar a mitad de episodio; un compañero que miente (dice haber mirado la celda más
     probable sin mirarla); mensajes con 10 pasos de retraso.
  3. Dos remedios: entrenamiento con condiciones variadas (fallos, mentirosos y retrasos al azar) y entrenamiento del
     dron 0 con una población de compañeros congelados (como Fictitious Co-Play), probado con compañeros nuevos.
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
RAPIDO = bool(os.environ.get("A15B_RAPIDO"))

H = W = 12
C = H * W
N = 3
T = 80
PD = 0.7
QN = 1.0 - PD
DR = 10                                     # retraso de los mensajes, en pasos
MOV = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], dtype=jnp.int32)
CX = jnp.arange(C) % W
CY = jnp.arange(C) // W
GAMMA, LAM, ESC_R = 0.99, 0.95, 10.0
R_EGO = W - 1
L_EGO = 2 * R_EGO + 1
D_OBS = 3 * L_EGO * L_EGO + N + 3
DENTRO = jnp.ones(C)


def mapa_previo(key):
    cen = jax.random.uniform(key, (2, 2), minval=2.0, maxval=10.0)
    d2 = (CX[None] - cen[:, :1]) ** 2 + (CY[None] - cen[:, 1:]) ** 2
    g = jnp.exp(-d2 / (2 * 1.5 ** 2)).sum(0)
    return 0.8 * g / g.sum() + 0.2 / C


# condiciones de un episodio: (instante del fallo, dron que falla, dron mentiroso o -1, mensajes retrasados 0/1)
SIN_FALLO = (T + 1, 0, -1, 0)


def cond_aleatoria(key):
    """Entrenamiento variado: 1/4 normal, 1/4 un dron falla entre los pasos 10 y 50, 1/4 un mentiroso, 1/4 retraso."""
    k1, k2, k3 = jax.random.split(key, 3)
    tipo = jax.random.randint(k1, (), 0, 4)
    quien = jax.random.randint(k2, (), 0, N)
    tf = jax.random.randint(k3, (), 10, 51)
    return (jnp.where(tipo == 1, tf, T + 1), quien, jnp.where(tipo == 2, quien, -1), (tipo == 3).astype(jnp.int32))


def estado_inicial(key, cond):
    z = jnp.zeros((N, C), jnp.int32)
    return dict(pos=jnp.zeros((N, 2), jnp.int32), co=z, falso=z, pr=mapa_previo(key), t=jnp.zeros((), jnp.int32),
                buf_c=jnp.zeros((DR, N, C), jnp.int32), buf_p=jnp.zeros((DR, N, C)),
                cond=tuple(jnp.asarray(x, jnp.int32) for x in cond))


def activos(st):
    tf, kf, _, _ = st["cond"]
    return ~((jnp.arange(N) == kf) & (st["t"] >= tf))


def paso(st, a):
    """Mueve y mira. Un dron caído no se mueve ni mira. El mentiroso anuncia además una mirada a la celda más probable
    según lo que el equipo cree, sin mirarla. Recompensa: probabilidad de detección con las miradas reales."""
    act = activos(st)
    ca = st["co"].sum(0)
    m = st["pr"] * QN ** ca
    a = jnp.where(act, a, 0)
    pos = jnp.clip(st["pos"] + MOV[a], 0, W - 1)
    oh = jax.nn.one_hot(pos[:, 1] * W + pos[:, 0], C, dtype=jnp.int32) * act[:, None]
    k = oh.sum(0)
    r = jnp.sum(m * (1.0 - QN ** k))
    _, _, km, _ = st["cond"]
    anunciado = st["pr"] * QN ** (ca + st["falso"].sum(0))
    falso = st["falso"] + (jnp.arange(N)[:, None] == km) * jax.nn.one_hot(jnp.argmax(anunciado), C, dtype=jnp.int32)
    # mensajes: lo que cada dron anuncia (miradas reales + falsas) y su posición, guardados para el retraso
    j = st["t"] % DR
    pos_oh = jax.nn.one_hot(st["pos"][:, 1] * W + st["pos"][:, 0], C)
    buf_c = st["buf_c"].at[j].set(st["co"] + st["falso"])
    buf_p = st["buf_p"].at[j].set(pos_oh)
    st2 = dict(st, pos=pos, co=st["co"] + oh, falso=falso, t=st["t"] + 1, buf_c=buf_c, buf_p=buf_p)
    return st2, r


def mensajes(st):
    """(N, C) miradas anunciadas por los demás y (N, C) sus posiciones, según lleguen al instante o con retraso."""
    anun = st["co"] + st["falso"]
    pos_oh = jax.nn.one_hot(st["pos"][:, 1] * W + st["pos"][:, 0], C)
    j = st["t"] % DR                                      # la entrada más antigua del búfer (de hace DR pasos)
    viejo = st["t"] >= DR
    anun_r = jnp.where(viejo, st["buf_c"][j], 0)
    pos_r = jnp.where(viejo, st["buf_p"][j], 0.0)
    ret = st["cond"][3] == 1
    anun = jnp.where(ret, anun_r, anun)
    pos_oh = jnp.where(ret, pos_r, pos_oh)
    return anun.sum(0)[None] - anun, pos_oh.sum(0)[None] - pos_oh


def ego(mapa, p):
    m = jnp.pad(mapa.reshape(H, W), R_EGO)
    return lax.dynamic_slice(m, (p[1], p[0]), (L_EGO, L_EGO)).reshape(-1)


def observacion(st, comm):
    oc, op = mensajes(st)
    if not comm:
        oc, op = jnp.zeros_like(oc), jnp.zeros_like(op)
    m = st["pr"][None] * QN ** (st["co"] + oc)
    b = m / m.sum(-1, keepdims=True) * C
    eg = jax.vmap(lambda bb, oo, p: jnp.concatenate([ego(bb, p), ego(oo, p), ego(DENTRO, p)]))(b, op, st["pos"])
    return jnp.concatenate([eg, jnp.eye(N), jnp.full((N, 1), st["t"] / T), st["pos"] / (W - 1.0)], -1)


def global_critico(st):
    """Crítico centralizado: la verdad (miradas reales de todos y posiciones), centrada en cada dron."""
    ca = st["co"].sum(0)
    m = st["pr"] * QN ** ca
    b = jnp.broadcast_to(m / m.sum() * C, (N, C))
    oh = jax.nn.one_hot(st["pos"][:, 1] * W + st["pos"][:, 0], C)
    op = oh.sum(0)[None] - oh
    eg = jax.vmap(lambda bb, oo, p: jnp.concatenate([ego(bb, p), ego(oo, p), ego(DENTRO, p)]))(b, op, st["pos"])
    return jnp.concatenate([eg, jnp.eye(N), jnp.full((N, 1), st["t"] / T), st["pos"] / (W - 1.0)], -1)


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
N_UPD = int(os.environ.get("A15B_UPD", 40 if RAPIDO else 600))
EPOCAS, N_MB, CLIP, ENT, LR = 4, 4, 0.2, 0.01, 5e-4
opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(LR))


def hacer_actualizacion(comm, variado, poblacion):
    """variado: condiciones al azar en cada episodio. poblacion: los drones 1 y 2 son compañeros congelados, elegidos al
    azar por episodio de una población (pila de parámetros), y solo aprende el dron 0."""
    def rollout(params, pob, key):
        pa, pc = params
        k0, k1, k2, k3 = jax.random.split(key, 4)
        conds = jax.vmap(cond_aleatoria)(jax.random.split(k2, E_TR)) if variado else \
            tuple(jnp.full((E_TR,), x, jnp.int32) for x in SIN_FALLO)
        st = jax.vmap(estado_inicial)(jax.random.split(k0, E_TR), conds)
        if poblacion:
            n_pob = jax.tree_util.tree_leaves(pob)[0].shape[0]
            quien = jax.random.randint(k3, (E_TR,), 0, n_pob)
        def f(st, k):
            ka, kb = jax.random.split(k)
            o = jax.vmap(lambda s: observacion(s, comm))(st)
            g = jax.vmap(global_critico)(st)
            lg = mlp(pa, o)
            if poblacion:
                lgp = jax.vmap(lambda p: mlp(p, o))(pob)[quien, jnp.arange(E_TR)]     # todos los compañeros, y se elige
                lg_ej = jnp.where((jnp.arange(N) == 0)[None, :, None], lg, lgp)
            else:
                lg_ej = lg
            a = jax.random.categorical(ka, lg_ej)
            lp = jnp.take_along_axis(jax.nn.log_softmax(lg), a[..., None], -1)[..., 0]
            v = mlp(pc, g)[..., 0]
            st, r = jax.vmap(paso)(st, a)
            return st, (o, g, a, lp, v, r)
        _, tr = lax.scan(f, st, jax.random.split(k1, T))
        return tr

    def perdida(params, o, g, a, lp0, adv, ret, w):
        pa, pc = params
        lps = jax.nn.log_softmax(mlp(pa, o))
        lp = jnp.take_along_axis(lps, a[..., None], -1)[..., 0]
        rt = jnp.exp(lp - lp0)
        an = (adv - jnp.sum(w * adv) / w.sum()) / (jnp.sqrt(jnp.sum(w * (adv - jnp.sum(w * adv) / w.sum()) ** 2) / w.sum()) + 1e-8)
        pg = -jnp.sum(w * jnp.minimum(rt * an, jnp.clip(rt, 1 - CLIP, 1 + CLIP) * an)) / w.sum()
        vl = jnp.sum(w * (mlp(pc, g)[..., 0] - ret) ** 2) / w.sum()
        ent = -jnp.sum(w * jnp.sum(jnp.exp(lps) * lps, -1)) / w.sum()
        return pg + 0.5 * vl - ENT * ent

    @jax.jit
    def actualizar(params, st_opt, pob, key):
        k1, k2 = jax.random.split(key)
        o, g, a, lp, v, r = rollout(params, pob, k1)
        r = r * ESC_R
        vn = jnp.concatenate([v[1:], jnp.zeros_like(v[:1])])
        d = r[..., None] + GAMMA * vn - v
        def fg(acc, x):
            acc = x + GAMMA * LAM * acc
            return acc, acc
        _, adv = lax.scan(fg, jnp.zeros_like(d[0]), d[::-1])
        adv = adv[::-1]
        ret = adv + v
        w = jnp.broadcast_to((jnp.arange(N) == 0) if poblacion else jnp.ones(N, bool), v.shape).astype(jnp.float32)
        n = T * E_TR * N
        datos = [x.reshape((n,) + x.shape[3:]) for x in (o, g, a, lp, adv, ret, w)]
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
        return params, st_opt, L.mean()
    return actualizar


def entrenar(comm, variado=False, pob=None, semilla=0, guardar_en=()):
    actualizar = hacer_actualizacion(comm, variado, pob is not None)
    k = jax.random.split(jax.random.PRNGKey(1000 + semilla), 3)
    params = (mlp_init(k[0], [D_OBS, 256, 256, 5], 0.01), mlp_init(k[1], [D_OBS, 256, 256, 1], 1.0))
    st_opt = opt.init(params)
    clave = k[2]
    puntos = {}
    for u in range(N_UPD):
        clave, ku = jax.random.split(clave)
        params, st_opt, L = actualizar(params, st_opt, pob, ku)
        if (u + 1) in guardar_en:
            puntos[u + 1] = params[0]
    return params[0], puntos


# ---------------------------------------------------------------------- evaluación
def politica_logits(pa, st, comm):
    return mlp(pa, observacion(st, comm))


def voraz_descentralizada(st, comm):
    """Cada dron, con su propio belief (el suyo o el del equipo según haya mensajes), va hacia la celda con más masa
    descontada por la distancia; no sabe qué harán los demás."""
    oc, _ = mensajes(st)
    oc = oc if comm else jnp.zeros_like(oc)
    m = st["pr"][None] * QN ** (st["co"] + oc)
    def uno(mi, p):
        d = jnp.abs(CX - p[0]) + jnp.abs(CY - p[1])
        c = jnp.argmax(mi * 0.9 ** d)
        dx, dy = jnp.sign(CX[c] - p[0]), jnp.sign(CY[c] - p[1])
        return jnp.where(dx > 0, 3, jnp.where(dx < 0, 4, jnp.where(dy > 0, 1, jnp.where(dy < 0, 2, 0))))
    return jax.vmap(uno)(m, st["pos"])


_EVAL = {}


def evaluar(tipo0, tipo12, p0, p12, claves, cond):
    """Dron 0 con 'tipo0' y parámetros p0; drones 1 y 2 con 'tipo12' y p12. tipo: ('red', comm) o ('voraz', comm).
    Devuelve P(encontrarlo en T pasos) y pasos esperados."""
    clave = (tipo0, tipo12, tuple(cond))
    if clave not in _EVAL:
        def acciones(tipo, p, st, k):
            if tipo[0] == "red":
                return jax.random.categorical(k, politica_logits(p, st, tipo[1]))
            return voraz_descentralizada(st, tipo[1])
        def episodio(p0, p12, key):
            k0, k1 = jax.random.split(key)
            st = estado_inicial(k0, cond)
            def f(st, k):
                ka, kb = jax.random.split(k)
                a = jnp.where(jnp.arange(N) == 0, acciones(tipo0, p0, st, ka), acciones(tipo12, p12, st, kb))
                masa = jnp.sum(st["pr"] * QN ** st["co"].sum(0))
                st, r = paso(st, a)
                return st, (r, masa)
            _, (r, masa) = lax.scan(f, st, jax.random.split(k1, T))
            return r.sum(), masa.sum()
        _EVAL[clave] = jax.jit(jax.vmap(episodio, (None, None, 0)))
    p, t = _EVAL[clave](p0, p12, claves)
    return float(p.mean()), float(t.mean())


N_EV = 400 if RAPIDO else 2000
claves_ev = jax.random.split(jax.random.PRNGKey(9999), N_EV)
CONDS = {"normal": SIN_FALLO, "un dron cae en el paso 30": (30, 2, -1, 0), "un compañero miente": (T + 1, 0, 2, 0),
         "mensajes con 10 pasos de retraso": (T + 1, 0, -1, 1)}
SEM = (0, 1) if RAPIDO else (0, 1, 2, 3)

print("=" * 70)
print("0. REFERENCIA: VORAZ DESCENTRALIZADA (cada dron con su belief; sin coordinarse)")
for comm in (False, True):
    for nc, cond in CONDS.items():
        P, t = evaluar(("voraz", comm), ("voraz", comm), None, None, claves_ev, cond)
        print(f"  voraz, {'con' if comm else 'sin'} comunicación, {nc}: P {P:.3f} | pasos {t:.1f}")

POL = {}
print("=" * 70)
print(f"1. AUTOJUEGO: MAPPO SIN Y CON COMUNICACIÓN, {len(SEM)} SEMILLAS, {N_UPD} ACTUALIZACIONES")
CHECK = (N_UPD // 4, N_UPD // 2, N_UPD)
CHECKPOINTS = []
for comm in (False, True):
    for s in SEM:
        t0 = time.perf_counter()
        pa, pts = entrenar(comm, semilla=s + (0 if not comm else 50), guardar_en=CHECK)
        POL[("sp", comm, s)] = pa
        if not comm and s < len(SEM) - 1:
            CHECKPOINTS += [pts[c] for c in CHECK]
        res = {nc: evaluar(("red", comm), ("red", comm), pa, pa, claves_ev, cond) for nc, cond in CONDS.items()}
        print(f"    {'con' if comm else 'sin'} comunicación, semilla {s}: " +
              " | ".join(f"{nc} P {v[0]:.3f}" for nc, v in res.items()) + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
        POL[("sp_res", comm, s)] = res

print("=" * 70)
print("2. JUEGO CRUZADO: DRON 0 DE LA SEMILLA i CON LOS DRONES 1 Y 2 DE LA SEMILLA j (condición normal)")
for comm in (False, True):
    M = np.zeros((len(SEM), len(SEM)))
    for i in SEM:
        for j in SEM:
            M[i, j] = evaluar(("red", comm), ("red", comm), POL[("sp", comm, i)], POL[("sp", comm, j)], claves_ev, SIN_FALLO)[0]
    diag = np.mean(np.diag(M))
    fuera = (M.sum() - np.trace(M)) / (M.size - len(SEM))
    print(f"  {'con' if comm else 'sin'} comunicación: autojuego (diagonal) {diag:.3f} | juego cruzado (fuera de la diagonal) {fuera:.3f} "
          f"| peor pareja {M.min():.3f}")
    print(f"    matriz: {np.round(M, 3).tolist()}")
# mezclar un dron entrenado con comunicación con compañeros entrenados sin ella, y con compañeros voraces
for comm0 in (False, True):
    for tipo12, nombre in ((("voraz", comm0), "voraces"),):
        v = [evaluar(("red", comm0), tipo12, POL[("sp", comm0, s)], None, claves_ev, SIN_FALLO)[0] for s in SEM]
        print(f"  dron 0 de autojuego {'con' if comm0 else 'sin'} comunicación con dos compañeros {nombre}: P {np.mean(v):.3f}")

print("=" * 70)
print("3. ENTRENAMIENTO VARIADO (CON COMUNICACIÓN; FALLOS, MENTIROSOS Y RETRASOS AL AZAR)")
for s in SEM:
    t0 = time.perf_counter()
    pa, _ = entrenar(True, variado=True, semilla=200 + s)
    POL[("var", True, s)] = pa
    res = {nc: evaluar(("red", True), ("red", True), pa, pa, claves_ev, cond) for nc, cond in CONDS.items()}
    POL[("var_res", True, s)] = res
    print(f"    variado, semilla {s}: " + " | ".join(f"{nc} P {v[0]:.3f}" for nc, v in res.items()) + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
M = np.zeros((len(SEM), len(SEM)))
for i in SEM:
    for j in SEM:
        M[i, j] = evaluar(("red", True), ("red", True), POL[("var", True, i)], POL[("var", True, j)], claves_ev, SIN_FALLO)[0]
print(f"  variado: autojuego (diagonal) {np.mean(np.diag(M)):.3f} | juego cruzado {(M.sum() - np.trace(M)) / (M.size - len(SEM)):.3f}")

print("=" * 70)
print("RESUMEN POR CONDICIÓN (media de semillas, autojuego)")
for nombre, clave in (("MAPPO sin comunicación", ("sp_res", False)), ("MAPPO con comunicación", ("sp_res", True)),
                      ("MAPPO con comunicación, entrenamiento variado", ("var_res", True))):
    fila = {nc: np.mean([POL[clave + (s,)][nc][0] for s in SEM]) for nc in CONDS}
    fila_t = {nc: np.mean([POL[clave + (s,)][nc][1] for s in SEM]) for nc in CONDS}
    print(f"  {nombre:46s}: " + " | ".join(f"{nc} P {fila[nc]:.3f} (pasos {fila_t[nc]:.1f})" for nc in CONDS))

print("=" * 70)
print(f"4. POBLACIÓN DE COMPAÑEROS (COMO FICTITIOUS CO-PLAY): SOLO APRENDE EL DRON 0, SIN COMUNICACIÓN")
pob = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *CHECKPOINTS)
print(f"  población: {len(CHECKPOINTS)} políticas (semillas 0 a {len(SEM) - 2} del autojuego sin comunicación, "
      f"en las actualizaciones {CHECK})")
nuevo = POL[("sp", False, SEM[-1])]                     # compañeros que no estaban en la población
res_fcp, res_sp = [], []
for s in SEM[:2]:
    t0 = time.perf_counter()
    pa, _ = entrenar(False, pob=pob, semilla=300 + s)
    fcp = (evaluar(("red", False), ("red", False), pa, nuevo, claves_ev, SIN_FALLO)[0],
           evaluar(("red", False), ("voraz", False), pa, None, claves_ev, SIN_FALLO)[0])
    res_fcp.append(fcp)
    print(f"    población, semilla {s}: con compañeros nuevos (autojuego, semilla {SEM[-1]}) P {fcp[0]:.3f} | con compañeros voraces P {fcp[1]:.3f}"
          f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
for s in SEM[:-1]:
    res_sp.append((evaluar(("red", False), ("red", False), POL[("sp", False, s)], nuevo, claves_ev, SIN_FALLO)[0],
                   evaluar(("red", False), ("voraz", False), POL[("sp", False, s)], None, claves_ev, SIN_FALLO)[0]))
print(f"  dron 0 de autojuego (semillas 0 a {len(SEM) - 2}): con compañeros nuevos P {np.mean([r[0] for r in res_sp]):.3f} | "
      f"con compañeros voraces P {np.mean([r[1] for r in res_sp]):.3f}")
print(f"  dron 0 entrenado con la población: con compañeros nuevos P {np.mean([r[0] for r in res_fcp]):.3f} | "
      f"con compañeros voraces P {np.mean([r[1] for r in res_fcp]):.3f}")
print(f"  referencia: equipo nuevo completo (autojuego, semilla {SEM[-1]}) P "
      f"{evaluar(('red', False), ('red', False), nuevo, nuevo, claves_ev, SIN_FALLO)[0]:.3f}")
