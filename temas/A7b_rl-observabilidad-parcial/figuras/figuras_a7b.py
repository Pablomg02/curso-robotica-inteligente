"""Datos de las figuras y tablas del tema A7b (deep RL bajo observabilidad parcial).

Uso:  uv run --with "jax[cuda13]" --with optax python -u figuras_a7b.py > salida.txt      (GPU; unos minutos)
      JAX_PLATFORMS=cpu uv run --with jax --with optax python -u figuras_a7b.py         (CPU; mucho más lento)

La búsqueda con un dron de A7a (rejilla 6 × 6, objetivo quieto con un prior con zona caliente, P_D = 0,6, sin
falsas alarmas, como mucho 80 pasos), ahora con PPO y cuatro entradas para la política:
  posicion  solo la casilla actual (sin memoria)
  pila      las 8 últimas casillas (memoria corta y fija)
  gru       la casilla actual en una GRU (actor y crítico con GRU separadas)
  belief    el belief exacto (filtro de Bayes) y la casilla actual, en una MLP
Evaluación con el mismo protocolo, los mismos objetivos y la misma semilla de detección que A7a (sección 5), para
comparar con sus heurísticas. Después, un sondeo lineal: ¿se puede leer el belief exacto en el estado de la GRU?
"""
import math
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax

print("dispositivo:", jax.devices()[0], "| jax", jax.__version__)
NX, PD, MAXP = 6, 0.6, 80
NC = NX * NX
CELDAS = [(i, j) for i in range(NX) for j in range(NX)]
MOV = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])
PRIOR = np.array([math.exp(-((i - 4) ** 2 + (j - 3) ** 2) / 6.0) + 0.15 for i, j in CELDAS])
PRIOR /= PRIOR.sum()
PRIOR_J = jnp.array(PRIOR)
MOV_J = jnp.array(MOV)
K_PILA = 8
GAMMA = 0.99
H = 128


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# ---------------------------------------------------------------------- entorno vectorizado
# estado: pos (2), objetivo (1), t (1), belief (NC), pila de casillas (K_PILA)
def reset_uno(key):
    obj = jax.random.choice(key, NC, p=PRIOR_J)
    return dict(pos=jnp.zeros(2, jnp.int32), obj=obj, t=jnp.zeros((), jnp.int32), b=PRIOR_J,
                pila=jnp.zeros(K_PILA, jnp.int32))


def celda(pos):
    return pos[0] * NX + pos[1]


def step_uno(S, a, key):
    k1, k2 = jax.random.split(key)
    pos2 = jnp.clip(S["pos"] + MOV_J[a], 0, NX - 1)
    c2 = celda(pos2)
    det = (c2 == S["obj"]) & (jax.random.uniform(k1) < PD)
    b2 = S["b"] * (1.0 - PD * jax.nn.one_hot(c2, NC))
    b2 = b2 / b2.sum()
    t2 = S["t"] + 1
    hecho = det | (t2 >= MAXP)
    S2 = dict(pos=pos2, obj=S["obj"], t=t2, b=b2, pila=jnp.concatenate([S["pila"][1:], c2[None]]))
    R0 = reset_uno(k2)
    S3 = jax.tree.map(lambda x, y: jnp.where(hecho, x, y), R0, S2)
    return S3, -1.0, hecho, det


def obs_uno(S, tipo):
    oh = jax.nn.one_hot(celda(S["pos"]), NC)
    if tipo in ("posicion", "gru"):
        return oh
    if tipo == "pila":
        return jnp.concatenate([jax.nn.one_hot(S["pila"], NC).reshape(-1), oh])
    return jnp.concatenate([S["b"] * NC, oh])          # belief (escalado) + casilla


OBS_DIM = {"posicion": NC, "gru": NC, "pila": NC * (K_PILA + 1), "belief": 2 * NC}


# ---------------------------------------------------------------------- redes
def denso(key, a, b, g=math.sqrt(2)):
    return (jax.nn.initializers.orthogonal(g)(key, (a, b)), jnp.zeros(b))


def gru_init(key, nin, nh):
    k = jax.random.split(key, 3)
    return {n: denso(kk, nin + nh, nh, 1.0) for n, kk in zip(("z", "r", "h"), k)}


def gru(p, h, x):
    xh = jnp.concatenate([x, h], -1)
    z = jax.nn.sigmoid(xh @ p["z"][0] + p["z"][1])
    r = jax.nn.sigmoid(xh @ p["r"][0] + p["r"][1])
    hh = jnp.tanh(jnp.concatenate([x, r * h], -1) @ p["h"][0] + p["h"][1])
    return (1 - z) * h + z * hh


def red_init(key, tipo, nout, g_fin):
    k = jax.random.split(key, 4)
    nin = OBS_DIM[tipo]
    p = {}
    if tipo == "gru":
        p["gru"] = gru_init(k[0], nin, H)
        nin = H
    p["l1"] = denso(k[1], nin, H)
    p["l2"] = denso(k[2], H, H)
    p["out"] = denso(k[3], H, nout, g_fin)
    return p


def red(p, h, x, tipo):
    """Devuelve (salida, h nuevo). Para las redes sin memoria, h no se usa."""
    if tipo == "gru":
        h = gru(p["gru"], h, x)
        x = h
    x = jnp.tanh(x @ p["l1"][0] + p["l1"][1])
    x = jnp.tanh(x @ p["l2"][0] + p["l2"][1])
    return x @ p["out"][0] + p["out"][1], h


# ---------------------------------------------------------------------- PPO (recurrente si hace falta)
CFG = dict(n_env=1024, n_pasos=32, epocas=4, n_mini=4, lam=0.95, clip=0.2, lr=5e-4, c_v=0.5, c_ent=0.01,
           max_grad=0.5, n_upd=500)
import os
if os.environ.get("A7B_RAPIDO"):              # prueba de humo: pocas actualizaciones y episodios
    CFG["n_upd"] = 20


def entrenar(tipo, seed, cfg=CFG):
    key = jax.random.PRNGKey(seed)
    key, ka, kc, ke = jax.random.split(key, 4)
    params = {"pi": red_init(ka, tipo, 4, 0.01), "v": red_init(kc, tipo, 1, 1.0)}
    opt = optax.chain(optax.clip_by_global_norm(cfg["max_grad"]), optax.adam(cfg["lr"], eps=1e-5))
    ost = opt.init(params)
    N = cfg["n_env"]
    S = jax.vmap(reset_uno)(jax.random.split(ke, N))
    # escalonar el inicio: avanzar cada entorno un número aleatorio de pasos con acciones al azar
    hz = jnp.zeros((N, H))
    obs_f = jax.vmap(lambda s: obs_uno(s, tipo))
    step_v = jax.vmap(step_uno)

    def rollout(c, _):
        params, S, ha, hc, key = c
        key, ks, kst = jax.random.split(key, 3)
        o = obs_f(S)
        logits, ha2 = red(params["pi"], ha, o, tipo)
        v, hc2 = red(params["v"], hc, o, tipo)
        a = jax.random.categorical(ks, logits)
        lp = jax.nn.log_softmax(logits)[jnp.arange(N), a]
        S2, r, hecho, det = step_v(S, a, jax.random.split(kst, N))
        ha2 = jnp.where(hecho[:, None], 0.0, ha2)
        hc2 = jnp.where(hecho[:, None], 0.0, hc2)
        return (params, S2, ha2, hc2, key), (o, a, lp, v[:, 0], r * jnp.ones(N), hecho, det)

    def actualizar(st, _):
        params, ost, S, ha, hc, key = st
        ha0, hc0 = ha, hc
        (params, S, ha, hc, key), (O, A, LP, V, R, D, DET) = lax.scan(rollout, (params, S, ha, hc, key), None, length=cfg["n_pasos"])
        v_ult, _ = red(params["v"], hc, obs_f(S), tipo)
        def gae(c, x):
            adv, vs = c
            r, v, d = x
            delta = r + GAMMA * vs * (1 - d) - v
            adv = delta + GAMMA * cfg["lam"] * (1 - d) * adv
            return (adv, v), adv
        _, ADV = lax.scan(gae, (jnp.zeros(N), v_ult[:, 0]), (R, V, D.astype(jnp.float32)), reverse=True)
        RET = ADV + V
        D_prev = jnp.concatenate([jnp.zeros((1, N), bool), D[:-1]], 0)   # ¿terminó el episodio antes de este paso?

        def perdida(params, idx):
            o, a, lp0, adv, ret, dp = O[:, idx], A[:, idx], LP[:, idx], ADV[:, idx], RET[:, idx], D_prev[:, idx]
            def paso(c, x):
                ha, hc = c
                oo, dd = x
                ha = jnp.where(dd[:, None], 0.0, ha); hc = jnp.where(dd[:, None], 0.0, hc)
                lg, ha = red(params["pi"], ha, oo, tipo)
                vv, hc = red(params["v"], hc, oo, tipo)
                return (ha, hc), (lg, vv[:, 0])
            _, (LG, VV) = lax.scan(paso, (ha0[idx], hc0[idx]), (o, dp))
            logp = jax.nn.log_softmax(LG)
            lpn = jnp.take_along_axis(logp, a[..., None], -1)[..., 0]
            ratio = jnp.exp(lpn - lp0)
            advn = (adv - adv.mean()) / (adv.std() + 1e-8)
            lpi = -jnp.mean(jnp.minimum(ratio * advn, jnp.clip(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * advn))
            lv = 0.5 * jnp.mean((VV - ret) ** 2)
            ent = -jnp.mean(jnp.sum(jnp.exp(logp) * logp, -1))
            return lpi + cfg["c_v"] * lv - cfg["c_ent"] * ent

        def epoca(c, k):
            params, ost = c
            perm = jax.random.permutation(k, N).reshape(cfg["n_mini"], -1)     # minilotes de entornos completos
            def mb(c2, idx):
                params, ost = c2
                g = jax.grad(perdida)(params, idx)
                u, ost = opt.update(g, ost, params)
                return (optax.apply_updates(params, u), ost), None
            (params, ost), _ = lax.scan(mb, (params, ost), perm)
            return (params, ost), None
        key, ke2 = jax.random.split(key)
        (params, ost), _ = lax.scan(epoca, (params, ost), jax.random.split(ke2, cfg["epocas"]))
        # métrica: longitud media de los episodios que terminaron en este lote
        fin = D.sum()
        return (params, ost, S, ha, hc, key), (fin, DET.sum())

    act = jax.jit(actualizar)
    st = (params, ost, S, hz, hz, key)
    st, _ = act(st, None)
    t0 = time.perf_counter()
    hist = []
    pasos_por_upd = N * cfg["n_pasos"]
    for i in range(cfg["n_upd"]):
        st, (fin, ndet) = act(st, None)
        if i % 10 == 0 or i == cfg["n_upd"] - 1:
            hist.append((i, pasos_por_upd / max(float(fin), 1.0)))     # pasos por episodio terminado ≈ longitud media
    jax.block_until_ready(st)
    return st[0], hist, time.perf_counter() - t0


# ---------------------------------------------------------------------- evaluación con el protocolo de A7a
def objetivos_a7a():
    rng0 = np.random.default_rng(5)
    return [CELDAS[int(rng0.choice(NC, p=PRIOR))] for _ in range(600)]


def evaluar(params, tipo, n_ep=600, muestrear=True, guardar=False):
    """Bucle de A7a: en cada paso, si el dron está en la casilla del objetivo, lo detecta con P_D; si no,
    actualiza el belief, la política elige y el dron se mueve. Devuelve los pasos hasta detectar."""
    @jax.jit
    def pol(params, h, o, key):
        lg, h2 = red(params["pi"], h, o, tipo)
        a = jnp.where(muestrear, jax.random.categorical(key, lg), jnp.argmax(lg))
        return a, h2
    objs = objetivos_a7a()
    rng = np.random.default_rng(11)
    key = jax.random.PRNGKey(99)
    Ts, registro = [], []
    for k in range(n_ep):
        c = (0, 0); b = PRIOR.copy(); pila = [0] * K_PILA; h = jnp.zeros(H)
        T = MAXP
        for t in range(MAXP):
            if c == objs[k] and rng.random() < PD:
                T = t; break
            ic = c[0] * NX + c[1]
            b = b.copy(); b[ic] *= (1 - PD); b /= b.sum()
            oh = np.zeros(NC); oh[ic] = 1
            if tipo in ("posicion", "gru"):
                o = oh
            elif tipo == "pila":
                o = np.concatenate([np.eye(NC)[pila].reshape(-1), oh])
            else:
                o = np.concatenate([b * NC, oh])
            key, ks = jax.random.split(key)
            a, h = pol(params, h, jnp.array(o, jnp.float32), ks)
            if guardar:
                registro.append((np.asarray(h), b.copy(), k, t, o.copy()))
            a = int(a)
            c = (min(max(c[0] + MOV[a][0], 0), NX - 1), min(max(c[1] + MOV[a][1], 0), NX - 1))
            pila = pila[1:] + [c[0] * NX + c[1]]
        Ts.append(T)
    Ts = np.array(Ts)
    return Ts, registro


print("=" * 70)
print("1. PPO EN LA BÚSQUEDA CON CUATRO ENTRADAS (3 SEMILLAS), EVALUADO CON EL PROTOCOLO DE A7a")
print("   referencias de A7a (mismos objetivos): barrido 37,9; máximo del belief 34,0; ganancia por distancia 32,4 pasos de media")
res = {}
modelos = {}
X1, Y1 = lin(0, CFG["n_upd"], 60, 520), lin(30, 56, 250, 30)
for tipo in ("posicion", "pila", "gru", "belief"):
    res[tipo] = []
    for sd in range(3):
        params, hist, tt = entrenar(tipo, sd)
        Ts, _ = evaluar(params, tipo, n_ep=60 if os.environ.get("A7B_RAPIDO") else 600)
        res[tipo].append((Ts.mean(), Ts.std() / math.sqrt(len(Ts)), np.median(Ts), np.mean(Ts >= MAXP), tt))
        modelos[(tipo, sd)] = params
        print(f"  {tipo:8s} semilla {sd}: pasos hasta detectar {Ts.mean():5.1f} ± {Ts.std()/math.sqrt(len(Ts)):.1f}, mediana {np.median(Ts):.0f}, "
              f"sin encontrar {np.mean(Ts >= MAXP)*100:4.1f} %; entrenamiento {tt:.0f} s; longitud en entrenamiento al final {hist[-1][1]:.1f}", flush=True)
        if sd == 0:
            # sin el primer registro: en la primera actualización casi ningún episodio ha terminado y el cociente no mide nada
            print(f"path curva {tipo}:", path([x[0] for x in hist[1:]], np.clip([x[1] for x in hist[1:]], 30, 56), X1, Y1))
    m = np.array([r[0] for r in res[tipo]])
    print(f"  => {tipo}: media entre semillas {m.mean():.1f} (de {m.min():.1f} a {m.max():.1f})")
pasos_tot = CFG["n_env"] * CFG["n_pasos"] * CFG["n_upd"]
print(f"  pasos de entorno por entrenamiento: {pasos_tot/1e6:.1f} M")

print("=" * 70)
print("2. ¿HAY UN BELIEF EN LA GRU? SONDEO LINEAL DEL BELIEF EXACTO DESDE EL ESTADO OCULTO")
def ocultos(params, regs):
    """Estados ocultos de una GRU recorriendo las mismas observaciones (episodio a episodio)."""
    f = jax.jit(lambda h, o: gru(params["pi"]["gru"], h, o))
    out, h, ep_prev = [], jnp.zeros(H), None
    for r in regs:
        if r[2] != ep_prev:
            h, ep_prev = jnp.zeros(H), r[2]
        h = f(h, jnp.array(r[4], jnp.float32))
        out.append(np.asarray(h))
    return np.array(out)


def sondeo(Hs, Bs, eps, nombre, n_tr):
    tr = eps < n_tr; te = ~tr
    Xa = np.concatenate([Hs, np.ones((len(Hs), 1))], 1)
    W = np.linalg.solve(Xa[tr].T @ Xa[tr] + 1e-2 * np.eye(Xa.shape[1]), Xa[tr].T @ Bs[tr])
    P = np.clip(Xa[te] @ W, 1e-6, None); P /= P.sum(1, keepdims=True)
    tv = 0.5 * np.abs(P - Bs[te]).sum(1).mean()
    tv_prior = 0.5 * np.abs(PRIOR[None] - Bs[te]).sum(1).mean()
    r2 = 1 - ((Xa[te] @ W - Bs[te]) ** 2).sum() / ((Bs[te] - Bs[tr].mean(0)) ** 2).sum()
    print(f"  {nombre}: {te.sum()} estados de prueba; variación total media al belief exacto {tv:.3f} "
          f"(con el prior fijo {tv_prior:.3f}); R² {r2:.3f}")
    return P, te


n_ep_s = 40 if os.environ.get("A7B_RAPIDO") else 400
_, reg = evaluar(modelos[("gru", 0)], "gru", n_ep=n_ep_s, guardar=True)
Bs = np.array([r[1] for r in reg]); eps = np.array([r[2] for r in reg]); ts = np.array([r[3] for r in reg])
print(f"  trayectorias de la GRU entrenada (semilla 0): {n_ep_s} episodios, {len(reg)} estados; ajuste con el 70 % de los episodios")
H_ent = np.array([r[0] for r in reg])
P_ent, te = sondeo(H_ent, Bs, eps, "estado de la GRU entrenada", int(0.7 * n_ep_s))
kr = jax.random.PRNGKey(123)
H_rnd = ocultos({"pi": red_init(kr, "gru", 4, 0.01)}, reg)
sondeo(H_rnd, Bs, eps, "estado de una GRU sin entrenar, mismas observaciones", int(0.7 * n_ep_s))
# solo la casilla actual (one-hot) como entrada del sondeo: ¿cuánto se explica sin memoria?
O_act = np.array([r[4] for r in reg])
sondeo(O_act, Bs, eps, "solo la casilla actual (sin memoria)", int(0.7 * n_ep_s))
Ete, Tte, Bte, Pte = eps[te], ts[te], Bs[te], P_ent
# un ejemplo para la figura: el primer episodio de prueba en su paso 12 (o el último si es más corto)
e0 = Ete[0]
sel = np.where(Ete == e0)[0]
j = sel[min(12, len(sel) - 1)]
print(f"ejemplo: episodio {e0}, paso {Tte[j]}")
print("belief exacto: " + " ".join(f"{x:.4f}" for x in Bte[j]))
print("belief leído: " + " ".join(f"{x:.4f}" for x in Pte[j]))
