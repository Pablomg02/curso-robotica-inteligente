"""Datos de las figuras y tablas del tema A8b (world models fundacionales y planificación latente).

Uso:  uv run --with "jax[cuda13]" --with optax python -u figuras_a8b.py > salida.txt      (GPU; unos minutos)

Planificación con MPPI (A3b) en el x500 de A8a: ir a un punto aleatorio y mantenerse, con observación parcial
(sin velocidad) y masa desconocida (±15 %). Cuatro modelos para planificar:
  física nominal     el simulador con la masa nominal y el estado verdadero (referencia con modelo físico)
  RSSM               el world model de A8a (reconstruye la observación); coste sobre la posición decodificada
  latente sin reconstrucción   codificador de una ventana de observaciones, dinámica latente con consistencia
                     (al estilo de TD-MPC) y una cabeza que predice solo la posición
  MLP de un paso     sobre la última observación (sin memoria)
Y como referencia, el controlador de posición que generó los datos (usa el estado verdadero).
Las funciones de datos, física y RSSM son las de figuras_a8a.py (copiadas para que el script sea autónomo).
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
G = 9.81
M0 = 2.0
J = jnp.array([0.022, 0.022, 0.040])
DT, SUB = 0.002, 10            # física a 500 Hz, control a 50 Hz
TAU_M = 0.03
T_EP = 150                     # 3 s
N_TR, N_TE = 3000, 256
T_FIL, T_PRED = 25, 50         # 0,5 s filtrando, 1 s prediciendo


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def rotmat(q):
    w, x, y, z = q
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qmul(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return jnp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


# ---------------------------------------------------------------------- simulador y controlador de datos
def paso_fisica(x, a, m):
    """x = (p, v, q, om, T, tau); a = CTBR normalizado: empuje colectivo y velocidades angulares."""
    p, v, q, om, Tr, tau = x
    T_cmd = M0 * G * (1.0 + 0.8 * a[0])
    om_d = jnp.array([4.0, 4.0, 2.0]) * a[1:4]
    tau_cmd = J * 20.0 * (om_d - om) + jnp.cross(om, J * om)
    def sub(i, x):
        p, v, q, om, Tr, tau = x
        Tr = Tr + DT / TAU_M * (T_cmd - Tr)
        tau = tau + DT / TAU_M * (tau_cmd - tau)
        acc = rotmat(q) @ jnp.array([0.0, 0.0, Tr / m]) - jnp.array([0.0, 0.0, G]) - 0.15 / m * v
        v = v + DT * acc
        p = p + DT * v
        om = om + DT * (tau - jnp.cross(om, J * om)) / J
        q = q + 0.5 * DT * qmul(q, jnp.concatenate([jnp.zeros(1), om]))
        return (p, v, q / jnp.linalg.norm(q), om, Tr, tau)
    return lax.fori_loop(0, SUB, sub, (p, v, q, om, Tr, tau))


def controlador(x, p_ref, ruido):
    """Controlador de posición sencillo (usa el estado verdadero, solo para generar datos) + ruido de exploración."""
    p, v, q, om, Tr, tau = x
    acc_d = -2.0 * (p - p_ref) - 2.5 * v + jnp.array([0.0, 0.0, G])
    R = rotmat(q)
    T = jnp.dot(acc_d, R[:, 2]) * M0
    zb_d = acc_d / jnp.linalg.norm(acc_d)
    err = jnp.cross(R[:, 2], zb_d)                 # error de inclinación en el mundo
    om_d = R.T @ (6.0 * err)
    a = jnp.concatenate([jnp.array([T / (M0 * G) - 1.0]) / 0.8, om_d / jnp.array([4.0, 4.0, 2.0])])
    return jnp.clip(a + ruido, -1.0, 1.0)


def obs_de(x, key):
    p, v, q, om, Tr, tau = x
    k1, k2, k3 = jax.random.split(key, 3)
    ang = 0.0175 * jax.random.normal(k2, (3,))
    dq = jnp.concatenate([jnp.ones(1), 0.5 * ang]); dq = dq / jnp.linalg.norm(dq)
    return jnp.concatenate([p + 0.02 * jax.random.normal(k1, (3,)), rotmat(qmul(q, dq)).reshape(-1),
                            om + 0.05 * jax.random.normal(k3, (3,))])


def episodio(key):
    k = jax.random.split(key, 6)
    m = M0 * jax.random.uniform(k[0], (), minval=0.85, maxval=1.15)
    p0 = jax.random.uniform(k[1], (3,), minval=-0.5, maxval=0.5)
    p_ref = jax.random.uniform(k[2], (3,), minval=-1.0, maxval=1.0)
    x0 = (p0, jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), M0 * G, jnp.zeros(3))
    def f(c, kk):
        x, ou = c
        k1, k2, k3 = jax.random.split(kk, 3)
        ou = 0.9 * ou + 0.15 * jax.random.normal(k1, (4,))          # ruido de exploración suave (Ornstein-Uhlenbeck)
        a = controlador(x, p_ref, ou)
        o = obs_de(x, k2)
        x2 = paso_fisica(x, a, m)
        return (x2, ou), (o, a, x[0], x[1], controlador(x, p_ref, jnp.zeros(4)))
    _, (O, A, P, V, AL) = lax.scan(f, (x0, jnp.zeros(4)), jax.random.split(k[3], T_EP))
    return O, A, P, V, jnp.full(T_EP, m), p_ref, AL          # p_ref y la acción sin ruido, para la política previa


import os
SOLO_TIEMPOS = bool(os.environ.get("A8B_SOLO_TIEMPOS"))       # mide la latencia por paso con pesos aleatorios (p. ej. en CPU)
RAPIDO = bool(os.environ.get("A8B_RAPIDO"))

print("=" * 70)
print("1. DATOS")
t0 = time.perf_counter()
if SOLO_TIEMPOS:
    N_TR, N_TE = 64, 16
gen = jax.jit(jax.vmap(episodio))
O_tr, A_tr, P_tr, V_tr, M_tr, Ref_tr, AL_tr = gen(jax.random.split(jax.random.PRNGKey(0), N_TR))
O_te, A_te, P_te, V_te, M_te, Ref_te, AL_te = gen(jax.random.split(jax.random.PRNGKey(1), N_TE))
ok = np.isfinite(np.asarray(O_tr)).all() and np.isfinite(np.asarray(O_te)).all()
print(f"  {N_TR} + {N_TE} episodios de {T_EP} pasos (3 s a 50 Hz) en {time.perf_counter()-t0:.1f} s; finitos: {ok}")
print(f"  |posición| media {float(jnp.abs(P_tr).mean()):.2f} m; |velocidad| media {float(jnp.linalg.norm(V_tr, axis=-1).mean()):.2f} m/s")
mu_o, sd_o = O_tr.reshape(-1, 15).mean(0), O_tr.reshape(-1, 15).std(0) + 1e-3
norm = lambda O: (O - mu_o) / sd_o

# ---------------------------------------------------------------------- modelos
H, Z, E = 128, 16, 128


def denso(key, a, b, g=math.sqrt(2)):
    return (jax.nn.initializers.orthogonal(g)(key, (a, b)), jnp.zeros(b))


def mlp_init(key, sizes):
    ks = jax.random.split(key, len(sizes) - 1)
    return [denso(k, a, b, 1.0 if i == len(sizes) - 2 else math.sqrt(2)) for i, (k, a, b) in enumerate(zip(ks, sizes[:-1], sizes[1:]))]


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jax.nn.elu(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def gru_init(key, nin, nh):
    k = jax.random.split(key, 3)
    return {n: denso(kk, nin + nh, nh, 1.0) for n, kk in zip(("z", "r", "h"), k)}


def gru(p, h, x):
    xh = jnp.concatenate([x, h], -1)
    z = jax.nn.sigmoid(xh @ p["z"][0] + p["z"][1])
    r = jax.nn.sigmoid(xh @ p["r"][0] + p["r"][1])
    hh = jnp.tanh(jnp.concatenate([x, r * h], -1) @ p["h"][0] + p["h"][1])
    return (1 - z) * h + z * hh


def rssm_init(key):
    k = jax.random.split(key, 5)
    return {"gru": gru_init(k[0], Z + 4, H), "enc": mlp_init(k[1], [15, E, E]),
            "prior": mlp_init(k[2], [H, E, 2 * Z]), "post": mlp_init(k[3], [H + E, E, 2 * Z]),
            "dec": mlp_init(k[4], [H + Z, E, E, 15])}


def gauss(stats):
    mu, s = jnp.split(stats, 2, -1)
    return mu, jax.nn.softplus(s) + 0.1


def kl(mq, sq, mp, sp):
    return jnp.sum(jnp.log(sp / sq) + (sq ** 2 + (mq - mp) ** 2) / (2 * sp ** 2) - 0.5, -1)


BETA_DYN, BETA_REP, FREE = 1.0, 0.1, 1.0       # pesos y bits libres de DreamerV3 (versión de Nature y arXiv v2)


def rssm_perdida(p, O, A, key, estocastico=True, beta_dyn=BETA_DYN):
    """O: (B, T, 15) normalizadas; A: (B, T, 4). Filtrado con el posterior; pérdida de reconstrucción + KL."""
    B = O.shape[0]
    e = mlp(p["enc"], O)
    def f(c, x):
        h, z, k = c
        et, a_prev = x
        k, ks = jax.random.split(k)
        h = gru(p["gru"], h, jnp.concatenate([z, a_prev], -1))
        mp, sp = gauss(mlp(p["prior"], h))
        mq, sq = gauss(mlp(p["post"], jnp.concatenate([h, et], -1)))
        if estocastico:
            zq = mq + sq * jax.random.normal(ks, mq.shape)
            kd = kl(lax.stop_gradient(mq), lax.stop_gradient(sq), mp, sp)
            kr = kl(mq, sq, lax.stop_gradient(mp), lax.stop_gradient(sp))
            reg = beta_dyn * jnp.maximum(kd, FREE) + BETA_REP * jnp.maximum(kr, FREE)
        else:
            zq = mq
            reg = 0.5 * jnp.sum((lax.stop_gradient(mq) - mp) ** 2, -1) + 0.1 * jnp.sum((mq - lax.stop_gradient(mp)) ** 2, -1)
        rec = mlp(p["dec"], jnp.concatenate([h, zq], -1))
        return (h, zq, k), (reg, rec)
    A_prev = jnp.concatenate([jnp.zeros((B, 1, 4)), A[:, :-1]], 1)
    (_, _, _), (REG, REC) = lax.scan(f, (jnp.zeros((B, H)), jnp.zeros((B, Z)), key),
                                        (jnp.swapaxes(e, 0, 1), jnp.swapaxes(A_prev, 0, 1)))
    REC = jnp.swapaxes(REC, 0, 1)
    l_rec = 0.5 * jnp.sum((REC - O) ** 2, -1).mean()
    return l_rec + REG.mean(), (l_rec, REG.mean())


def rssm_predecir(p, O, A, key, estocastico=True, n_muestras=1):
    """Filtra T_FIL pasos con observaciones y predice T_PRED pasos en bucle abierto con las acciones.
    Devuelve las observaciones predichas (normalizadas) y los estados latentes del filtrado."""
    B = O.shape[0]
    e = mlp(p["enc"], O[:, :T_FIL])
    A_prev = jnp.concatenate([jnp.zeros((B, 1, 4)), A[:, :-1]], 1)
    def fil(c, x):
        h, z, k = c
        et, a_prev = x
        k, ks = jax.random.split(k)
        h = gru(p["gru"], h, jnp.concatenate([z, a_prev], -1))
        mq, sq = gauss(mlp(p["post"], jnp.concatenate([h, et], -1)))
        z = mq + (sq * jax.random.normal(ks, mq.shape) if estocastico else 0.0)
        return (h, z, k), jnp.concatenate([h, mq], -1)
    (h, z, k), LAT = lax.scan(fil, (jnp.zeros((B, H)), jnp.zeros((B, Z)), key),
                              (jnp.swapaxes(e, 0, 1), jnp.swapaxes(A_prev[:, :T_FIL], 0, 1)))
    def imag(c, a_prev):
        h, z, k = c
        k, ks = jax.random.split(k)
        h = gru(p["gru"], h, jnp.concatenate([z, a_prev], -1))
        mp, sp = gauss(mlp(p["prior"], h))
        z = mp + (sp * jax.random.normal(ks, mp.shape) if estocastico else 0.0)
        return (h, z, k), mlp(p["dec"], jnp.concatenate([h, z], -1))
    def una(kk):
        _, PRED = lax.scan(imag, (h, z, kk), jnp.swapaxes(A_prev[:, T_FIL:T_FIL + T_PRED], 0, 1))
        return jnp.swapaxes(PRED, 0, 1)
    PREDS = jax.vmap(una)(jax.random.split(k, n_muestras))
    return PREDS, jnp.swapaxes(LAT, 0, 1)


def mlp1_init(key):
    return mlp_init(key, [15 + 4, 256, 256, 15])


def mlp1_perdida(p, O, A, key):
    d = mlp(p, jnp.concatenate([O[:, :-1], A[:, :-1]], -1))
    l = 0.5 * jnp.sum((O[:, :-1] + d - O[:, 1:]) ** 2, -1).mean()
    return l, (l, 0.0)


def mlp1_predecir(p, O, A):
    o = O[:, T_FIL - 1]
    def f(o, a):
        o2 = o + mlp(p, jnp.concatenate([o, a], -1))
        return o2, o2
    _, PRED = lax.scan(f, o, jnp.swapaxes(A[:, T_FIL - 1:T_FIL - 1 + T_PRED], 0, 1))
    return jnp.swapaxes(PRED, 0, 1)[None]


PASOS = 300 if RAPIDO else 6000


def entrenar(nombre, init, perdida, pasos=PASOS, B=64, L=50, seed=0):
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    p = init(ki)
    opt = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(3e-4))
    st = opt.init(p)
    On = norm(O_tr)
    @jax.jit
    def upd(p, st, key):
        k1, k2, k3 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (B,), 0, N_TR)
        t0 = jax.random.randint(k2, (B,), 0, T_EP - L)
        o = jax.vmap(lambda i, t: lax.dynamic_slice(On[i], (t, 0), (L, 15)))(idx, t0)
        a = jax.vmap(lambda i, t: lax.dynamic_slice(A_tr[i], (t, 0), (L, 4)))(idx, t0)
        (l, aux), g = jax.value_and_grad(perdida, has_aux=True)(p, o, a, k3)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, aux
    t0 = time.perf_counter()
    for i in range(pasos):
        key, kk = jax.random.split(key)
        p, st, aux = upd(p, st, kk)
        if i % 2000 == 0 or i == pasos - 1:
            print(f"    {nombre} paso {i}: reconstrucción {float(aux[0]):.3f}, KL/regularizador {float(aux[1]):.3f}", flush=True)
    jax.block_until_ready(p)
    print(f"  {nombre}: {pasos} pasos de optimización en {time.perf_counter()-t0:.0f} s")
    return p



# ---------------------------------------------------------------------- modelo latente sin reconstrucción (al estilo de TD-MPC2)
K_VEN, ZL, V_SIM, HS = 8, 64, 8, 15     # ventana del codificador, dimensión latente, grupos de SimNorm, horizonte de entrenamiento
RHO, C_CONS = 0.9, 20.0


def simnorm(z):
    """SimNorm de TD-MPC2: el latente se parte en grupos de V_SIM y cada grupo pasa por un softmax."""
    s = z.shape
    return jax.nn.softmax(z.reshape(s[:-1] + (-1, V_SIM)), -1).reshape(s)


def lat_init(key):
    k = jax.random.split(key, 3)
    return {"enc": mlp_init(k[0], [K_VEN * 15 + (K_VEN - 1) * 4, 256, 256, ZL]),
            "din": mlp_init(k[1], [ZL + 4, 256, 256, ZL]),
            "pos": mlp_init(k[2], [ZL, 128, 3])}


def lat_enc(p, Wo, Wa):
    return simnorm(mlp(p["enc"], jnp.concatenate([Wo.reshape(Wo.shape[:-2] + (-1,)), Wa.reshape(Wa.shape[:-2] + (-1,))], -1)))


def lat_din(p, z, a):
    return simnorm(mlp(p["din"], jnp.concatenate([z, a], -1)))


def lat_perdida(p, Wo, Wa, Af, Pf):
    """Wo: (B, HS+1, K, 15) ventanas que acaban en t0..t0+HS; Wa: (B, HS+1, K-1, 4); Af: (B, HS, 4) acciones a_t0..;
    Pf: (B, HS+1, 3) posiciones observadas (normalizadas). Consistencia latente + cabeza de posición (sin reconstruir o)."""
    E = lat_enc(p, Wo, Wa)
    z0 = E[:, 0]
    lp0 = jnp.sum((mlp(p["pos"], z0) - Pf[:, 0]) ** 2, -1)
    def f(z, x):
        a, e2, p2 = x
        z2 = lat_din(p, z, a)
        return z2, (jnp.sum((z2 - lax.stop_gradient(e2)) ** 2, -1), jnp.sum((mlp(p["pos"], z2) - p2) ** 2, -1))
    _, (LC, LP) = lax.scan(f, z0, (jnp.swapaxes(Af, 0, 1), jnp.swapaxes(E[:, 1:], 0, 1), jnp.swapaxes(Pf[:, 1:], 0, 1)))
    w = RHO ** jnp.arange(1, HS + 1)
    lpos = lp0.mean() + (w[:, None] * LP).mean()
    lcons = (w[:, None] * LC).mean()
    return lpos + C_CONS * lcons, (lpos, lcons)


def entrenar_lat(nombre, O_all, A_all, pasos=PASOS, B=128, seed=3, p=None):
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    p = lat_init(ki) if p is None else p
    opt = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(3e-4))
    st = opt.init(p)
    On = norm(O_all)
    Opad = jnp.concatenate([jnp.repeat(On[:, :1], K_VEN - 1, 1), On], 1)
    Apad = jnp.concatenate([jnp.zeros((On.shape[0], K_VEN - 1, 4)), A_all], 1)
    n = On.shape[0]
    def ventanas(i, t):
        so = lax.dynamic_slice(Opad[i], (t, 0), (HS + K_VEN, 15))
        sa = lax.dynamic_slice(Apad[i], (t, 0), (HS + K_VEN - 1, 4))
        Wo = jax.vmap(lambda j: lax.dynamic_slice(so, (j, 0), (K_VEN, 15)))(jnp.arange(HS + 1))
        Wa = jax.vmap(lambda j: lax.dynamic_slice(sa, (j, 0), (K_VEN - 1, 4)))(jnp.arange(HS + 1))
        Af = lax.dynamic_slice(A_all[i], (t, 0), (HS, 4))
        Pf = lax.dynamic_slice(On[i], (t, 0), (HS + 1, 15))[:, :3]
        return Wo, Wa, Af, Pf
    @jax.jit
    def upd(p, st, key):
        k1, k2 = jax.random.split(key)
        idx = jax.random.randint(k1, (B,), 0, n)
        t0 = jax.random.randint(k2, (B,), 0, T_EP - HS)
        Wo, Wa, Af, Pf = jax.vmap(ventanas)(idx, t0)
        (l, aux), g = jax.value_and_grad(lat_perdida, has_aux=True)(p, Wo, Wa, Af, Pf)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, aux
    t0 = time.perf_counter()
    for i in range(pasos):
        key, kk = jax.random.split(key)
        p, st, aux = upd(p, st, kk)
        if i % 2000 == 0 or i == pasos - 1:
            print(f"    {nombre} paso {i}: posición {float(aux[0]):.3f}, consistencia {float(aux[1]):.4f}", flush=True)
    jax.block_until_ready(p)
    print(f"  {nombre}: {pasos} pasos de optimización en {time.perf_counter()-t0:.0f} s")
    return p


def entrenar_con(nombre, p, perdida, O_all, A_all, pasos, B=64, L=50, seed=7):
    """Como entrenar(), pero partiendo de unos parámetros y con otro conjunto de datos (iteración de datos)."""
    key = jax.random.PRNGKey(seed)
    opt = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(3e-4))
    st = opt.init(p)
    On = norm(O_all)
    n = On.shape[0]
    @jax.jit
    def upd(p, st, key):
        k1, k2, k3 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (B,), 0, n)
        t0 = jax.random.randint(k2, (B,), 0, T_EP - L)
        o = jax.vmap(lambda i, t: lax.dynamic_slice(On[i], (t, 0), (L, 15)))(idx, t0)
        a = jax.vmap(lambda i, t: lax.dynamic_slice(A_all[i], (t, 0), (L, 4)))(idx, t0)
        (l, aux), g = jax.value_and_grad(perdida, has_aux=True)(p, o, a, k3)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, aux
    t0 = time.perf_counter()
    for i in range(pasos):
        key, kk = jax.random.split(key)
        p, st, aux = upd(p, st, kk)
        if i % 1000 == 0 or i == pasos - 1:
            print(f"    {nombre} paso {i}: reconstrucción {float(aux[0]):.3f}, KL {float(aux[1]):.3f}", flush=True)
    jax.block_until_ready(p)
    print(f"  {nombre}: {pasos} pasos en {time.perf_counter()-t0:.0f} s")
    return p



# ---------------------------------------------------------------------- creencia, imaginación y política previa de cada modelo
# Cada modelo es un dict con:
#   act(b, o_n, a_prev, x, t) -> b     actualiza la creencia con la observación normalizada (x: estado verdadero, solo física)
#   s0(b) -> s                          estado desde el que se imagina
#   paso(s, a) -> s                     un paso de imaginación
#   pos(s) -> posición en metros        rasgos(s) -> entrada de la política previa
SDP, MUP = sd_o[:3], mu_o[:3]
X_NULO = (jnp.zeros(3), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), jnp.asarray(M0 * G), jnp.zeros(3))


def modelo_fisica():
    return dict(act=lambda b, o, a_prev, x, t: x, s0=lambda b: b, paso=lambda x, a: paso_fisica(x, a, M0),
                pos=lambda x: x[0], rasgos=None, b0=X_NULO)


def modelo_rssm(p):
    def act(b, o, a_prev, x, t):
        h, z = b
        h = gru(p["gru"], h, jnp.concatenate([z, a_prev], -1))
        mq, _ = gauss(mlp(p["post"], jnp.concatenate([h, mlp(p["enc"], o)], -1)))
        return (h, mq)
    def paso(s, a):
        h, z = s
        h = gru(p["gru"], h, jnp.concatenate([z, a], -1))
        return (h, gauss(mlp(p["prior"], h))[0])
    return dict(act=act, s0=lambda b: b, paso=paso,
                pos=lambda s: mlp(p["dec"], jnp.concatenate(s, -1))[:3] * SDP + MUP,
                rasgos=lambda s: jnp.concatenate(s, -1), b0=(jnp.zeros(H), jnp.zeros(Z)))


def modelo_lat(p):
    def act(b, o, a_prev, x, t):
        Wo, Wa = b
        Wo = jnp.where(t == 0, jnp.tile(o, (K_VEN, 1)), jnp.concatenate([Wo[1:], o[None]], 0))
        Wa = jnp.where(t == 0, jnp.zeros_like(Wa), jnp.concatenate([Wa[1:], a_prev[None]], 0))
        return (Wo, Wa)
    return dict(act=act, s0=lambda b: lat_enc(p, *b), paso=lambda z, a: lat_din(p, z, a),
                pos=lambda z: mlp(p["pos"], z) * SDP + MUP, rasgos=lambda z: z,
                b0=(jnp.zeros((K_VEN, 15)), jnp.zeros((K_VEN - 1, 4))))


def modelo_mlp1(p):
    return dict(act=lambda b, o, a_prev, x, t: o, s0=lambda b: b,
                paso=lambda o, a: o + mlp(p, jnp.concatenate([o, a], -1)),
                pos=lambda o: o[:3] * SDP + MUP, rasgos=lambda o: o, b0=jnp.zeros(15))


def imaginar(mod, b, A):
    """Posiciones tras cada acción de A (len(A), 3)."""
    def f(s, a):
        s = mod["paso"](s, a)
        return s, mod["pos"](s)
    return lax.scan(f, mod["s0"](b), A)[1]


def politica(pp, mod, s, p_ref):
    """Política previa: rasgos del estado del modelo y objetivo relativo a la posición que estima el modelo."""
    return jnp.tanh(mlp(pp, jnp.concatenate([mod["rasgos"](s), (p_ref - mod["pos"](s)) / SDP], -1)))


def imaginar_residuo(mod, pp, b, D, p_ref):
    """Acciones = política previa sobre el estado imaginado + residuo D. Devuelve posiciones y acciones aplicadas."""
    def f(s, d):
        a = jnp.clip(politica(pp, mod, s, p_ref) + d, -1.0, 1.0)
        s = mod["paso"](s, a)
        return s, (mod["pos"](s), a)
    return lax.scan(f, mod["s0"](b), D)[1]


# ---------------------------------------------------------------------- MPPI (A3b) con cualquiera de los modelos
HP = int(os.environ.get("A8B_HP", "25"))  # horizonte de 0,5 s
NS, NE = 256, 32                       # 256 secuencias, 32 élites
SIGS = (0.3, 0.15, 0.08)               # tres iteraciones con ruido decreciente
W_A, W_DA = 0.02, 0.1
W_V, LAM_V = 1.0, 1.5                  # campo de velocidades deseado v = -LAM_V (p - p_ref)
T_EV, N_EV = 150, 64


def coste(P, A, p_ref, a_prev):
    """Posición, seguir un campo de velocidades hacia el objetivo (la velocidad sale de diferencias de las posiciones
    predichas, así sirve con cualquier modelo) y acciones pequeñas y suaves."""
    Vp = jnp.diff(P, axis=0) / (SUB * DT)
    dA = jnp.diff(jnp.concatenate([a_prev[None], A], 0), axis=0)
    return (jnp.sum((P - p_ref) ** 2) + W_V * jnp.sum((Vp + LAM_V * (P[1:] - p_ref)) ** 2) / LAM_V ** 2
            + W_A * jnp.sum(A ** 2) + W_DA * jnp.sum(dA ** 2))


RHO_R = 0.9                            # correlación del ruido de muestreo entre pasos (0: independiente)


def ruido_suave(key):
    """Ruido de muestreo correlado en el tiempo (AR(1)), de varianza unidad: parecido al ruido de exploración de los datos."""
    rho = RHO_R
    n = jax.random.normal(key, (HP, NS, 4))
    def f(e, x):
        e = rho * e + math.sqrt(1 - rho ** 2) * x
        return e, e
    E = jnp.concatenate([n[:1], lax.scan(f, n[0], n[1:])[1]], 0)       # (HP, NS, 4)
    return jnp.swapaxes(E, 0, 1)


def planificar(mod, pp, b, U, p_ref, a_prev, key):
    """MPPI con élites. Sin política previa (pp None) se optimizan las acciones; con ella, un residuo sobre la política."""
    for s in SIGS:
        key, k = jax.random.split(key)
        D = U[None] + s * ruido_suave(k)
        if pp is None:
            D = jnp.clip(D, -1.0, 1.0).at[0].set(U).at[1].set(0.0)             # la media anterior y «flotar»
            A = D
            P = jax.vmap(lambda d: imaginar(mod, b, d))(D)
        else:
            D = jnp.clip(D, -0.5, 0.5).at[0].set(U).at[1].set(0.0)             # la media anterior y la política sola
            P, A = jax.vmap(lambda d: imaginar_residuo(mod, pp, b, d, p_ref))(D)
        c = jax.vmap(coste, (0, 0, None, None))(P, A, p_ref, a_prev)
        c = jnp.where(jnp.isfinite(c), c, 1e9)
        nc, ie = lax.top_k(-c, NE)
        ce = -nc
        w = jax.nn.softmax(-2.0 * (ce - ce[0]) / (jnp.std(ce) + 1e-6))
        U = jnp.sum(w[:, None, None] * D[ie], 0)
    return U


def condiciones(key):
    k = jax.random.split(key, 4)
    m = M0 * jax.random.uniform(k[0], (), minval=0.85, maxval=1.15)
    p0 = jax.random.uniform(k[1], (3,), minval=-0.5, maxval=0.5)
    p_ref = jax.random.uniform(k[2], (3,), minval=-1.0, maxval=1.0)
    return m, p0, p_ref, k[3]


def x_inicial(p0):
    return (p0, jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), jnp.asarray(M0 * G), jnp.zeros(3))


def accion(mod, pp, modo, b, U, p_ref, a_prev, key):
    """modo: 'mppi' (sin política previa), 'politica' (la política imitada sola) o 'mppi+politica'."""
    if modo == "politica":
        return politica(pp, mod, mod["s0"](b), p_ref), U
    U = planificar(mod, pp if modo == "mppi+politica" else None, b, U, p_ref, a_prev, key)
    if modo == "mppi":
        return U[0], U
    return jnp.clip(politica(pp, mod, mod["s0"](b), p_ref) + U[0], -1.0, 1.0), U


def episodio_mpc(mod, pp, modo, key, ruido_exp=0.0):
    m, p0, p_ref, kk = condiciones(key)
    def f(c, inp):
        x, b, U, a_prev, ou = c
        t, kt = inp
        k1, k2, k3 = jax.random.split(kt, 3)
        o = obs_de(x, k1)
        b = mod["act"](b, norm(o), a_prev, x, t)
        a, U = accion(mod, pp, modo, b, U, p_ref, a_prev, k2)
        ou = 0.9 * ou + 0.15 * jax.random.normal(k3, (4,))
        a = jnp.clip(a + ruido_exp * ou, -1.0, 1.0)
        x2 = paso_fisica(x, a, m)
        U = jnp.concatenate([U[1:], jnp.zeros((1, 4)) if pp is not None and modo != "mppi" else U[-1:]], 0)
        return (x2, b, U, a, ou), (o, a, x[0], x[1])
    c0 = (x_inicial(p0), mod["b0"], jnp.zeros((HP, 4)), jnp.zeros(4), jnp.zeros(4))
    _, (O, A, P, V) = lax.scan(f, c0, (jnp.arange(T_EV), jax.random.split(kk, T_EV)))
    return O, A, P, V, p_ref, m


def episodio_controlador(key):
    m, p0, p_ref, kk = condiciones(key)
    def f(x, kt):
        o = obs_de(x, kt)
        a = controlador(x, p_ref, jnp.zeros(4))
        return paso_fisica(x, a, m), (o, a, x[0], x[1])
    _, (O, A, P, V) = lax.scan(f, x_inicial(p0), jax.random.split(kk, T_EV))
    return O, A, P, V, p_ref, m


def latencia(mod, pp, modo, repeticiones=50):
    """Tiempo de un paso de control (actualizar la creencia + decidir la acción) para un solo dron."""
    @jax.jit
    def paso(b, U, o, a_prev, x, t, p_ref, key):
        b = mod["act"](b, o, a_prev, x, t)
        return b, accion(mod, pp, modo, b, U, p_ref, a_prev, key)
    args = (mod["b0"], jnp.zeros((HP, 4)), jnp.zeros(15), jnp.zeros(4), x_inicial(jnp.zeros(3)), 1, jnp.ones(3) * 0.5)
    jax.block_until_ready(paso(*args, jax.random.PRNGKey(0)))
    t0 = time.perf_counter()
    for i in range(repeticiones):
        out = paso(*args, jax.random.PRNGKey(i))
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / repeticiones * 1e3


def pol_init(key, nin):
    return mlp_init(key, [nin, 256, 256, 4])


EVALS = HP * NS * len(SIGS)
if SOLO_TIEMPOS:
    print("=" * 70)
    print(f"TIEMPOS POR PASO DE CONTROL CON PESOS ALEATORIOS ({jax.devices()[0].platform}); {EVALS} pasos de modelo por paso de control")
    k0 = jax.random.PRNGKey(0)
    for nombre, mod, nin in (("física nominal", modelo_fisica(), None), ("RSSM", modelo_rssm(rssm_init(k0)), H + Z),
                             ("latente sin reconstrucción", modelo_lat(lat_init(k0)), ZL),
                             ("MLP de un paso", modelo_mlp1(mlp1_init(k0)), 15)):
        print(f"  {nombre:28s}: MPPI {latencia(mod, None, 'mppi', 20):7.1f} ms por paso", end="")
        if nin:
            pp = pol_init(k0, nin + 3)
            print(f"; MPPI + política {latencia(mod, pp, 'mppi+politica', 20):7.1f} ms; política sola {latencia(mod, pp, 'politica', 20):6.2f} ms")
        else:
            print()
    raise SystemExit

# ---------------------------------------------------------------------- 2. entrenamiento
print("=" * 70)
print("2. ENTRENAMIENTO DE LOS MODELOS (MISMOS DATOS QUE A8a: 3000 EPISODIOS DEL CONTROLADOR CON RUIDO)")
p_rssm = entrenar("RSSM", rssm_init, lambda p, o, a, k: rssm_perdida(p, o, a, k, True))
p_lat = entrenar_lat("latente sin reconstrucción", O_tr, A_tr)
p_mlp = entrenar("MLP de un paso", mlp1_init, mlp1_perdida)
print(f"  parámetros: RSSM {sum(x.size for x in jax.tree_util.tree_leaves(p_rssm))}, "
      f"latente {sum(x.size for x in jax.tree_util.tree_leaves(p_lat))}, MLP {sum(x.size for x in jax.tree_util.tree_leaves(p_mlp))}")

FIS = modelo_fisica()
MODELOS = {"RSSM": modelo_rssm(p_rssm), "latente sin reconstrucción": modelo_lat(p_lat), "MLP de un paso": modelo_mlp1(p_mlp)}


def rasgos_datos(mod, O, A, Ref):
    """Entrada de la política (creencia tras ver o_t y objetivo relativo) para todos los pasos de unos episodios."""
    def uno(O1, A1, r):
        A_prev = jnp.concatenate([jnp.zeros((1, 4)), A1[:-1]], 0)
        def f(b, inp):
            o, a_prev, t = inp
            b = mod["act"](b, norm(o), a_prev, None, t)
            s = mod["s0"](b)
            return b, jnp.concatenate([mod["rasgos"](s), (r - mod["pos"](s)) / SDP], -1)
        return lax.scan(f, mod["b0"], (O1, A_prev, jnp.arange(O1.shape[0])))[1]
    return jax.jit(jax.vmap(uno))(O, A, Ref)


def entrenar_politica(nombre, mod, pasos=300 if RAPIDO else 4000, B=1024, seed=11):
    """Imitación (clonado de comportamiento) del controlador de los datos, sin su ruido de exploración, desde la creencia
    del modelo y el objetivo: la política previa (maestro con estado verdadero, alumno con creencia, como en A6b)."""
    R = rasgos_datos(mod, O_tr, A_tr, Ref_tr)
    R = R.reshape(-1, R.shape[-1]); Y = AL_tr.reshape(-1, 4)
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    pp = pol_init(ki, R.shape[-1])
    opt = optax.adam(1e-3)
    st = opt.init(pp)
    @jax.jit
    def upd(pp, st, k, R, Y):                     # R e Y como argumentos: cerrados como constantes, XLA los copia al compilar
        idx = jax.random.randint(k, (B,), 0, R.shape[0])
        l, g = jax.value_and_grad(lambda q: jnp.mean(jnp.sum((jnp.tanh(mlp(q, R[idx])) - Y[idx]) ** 2, -1)))(pp)
        u, st = opt.update(g, st, pp)
        return optax.apply_updates(pp, u), st, l
    for i in range(pasos):
        key, kk = jax.random.split(key)
        pp, st, l = upd(pp, st, kk, R, Y)
    Rt = rasgos_datos(mod, O_te, A_te, Ref_te).reshape(-1, R.shape[-1])
    lt = float(jnp.mean(jnp.sum((jnp.tanh(mlp(pp, Rt)) - AL_te.reshape(-1, 4)) ** 2, -1)))
    print(f"  política imitada desde {nombre}: error cuadrático {float(l):.4f} (entrenamiento), {lt:.4f} (prueba); "
          f"varianza de la acción del maestro {float(jnp.sum(AL_te.reshape(-1, 4).var(0))):.4f}")
    return pp


POL = {n: entrenar_politica(n, m) for n, m in MODELOS.items()}

# ---------------------------------------------------------------------- 3. control en bucle cerrado
print("=" * 70)
print(f"3. CONTROL EN BUCLE CERRADO: IR A UN PUNTO Y MANTENERSE ({N_EV} EPISODIOS DE 3 s; MPPI CON HORIZONTE 0,5 s, "
      f"{NS} SECUENCIAS x {len(SIGS)} ITERACIONES)")
claves_ev = jax.random.split(jax.random.PRNGKey(1234), N_EV)


def resumen(nombre, P, p_ref):
    E = np.asarray(P) - np.asarray(p_ref)[:, None]
    d = np.linalg.norm(E, axis=-1)                                                 # (N, T)
    fin = d[:, 100:].mean(1)
    fxy = np.linalg.norm(E[:, 100:, :2], axis=-1).mean(1)
    fz = np.abs(E[:, 100:, 2]).mean(1)
    fallo = ~np.isfinite(fin) | (fin > 1.5)
    ok = ~fallo
    if ok.sum() == 0:
        print(f"  {nombre:40s}: todos los episodios divergen ({fallo.sum()}/{len(fin)})", flush=True)
        return d, fin, fallo
    print(f"  {nombre:40s}: último segundo: mediana {np.median(fin[ok])*100:5.1f} cm (horizontal {np.median(fxy[ok])*100:5.1f}, "
          f"vertical {np.median(fz[ok])*100:5.1f}), p90 {np.percentile(fin[ok], 90)*100:5.1f} cm; "
          f"divergen (>1,5 m o NaN) {fallo.sum()}/{len(fin)}", flush=True)
    return d, fin, fallo


def actividad(A):
    return float(np.nanmean(np.abs(np.diff(np.asarray(A), axis=1)).sum(-1)))


LOGS, CURVAS = {}, {}
O_c, A_c, P_c, V_c, R_c, M_c = jax.jit(jax.vmap(episodio_controlador))(claves_ev)
CURVAS["controlador de los datos"], _, _ = resumen("controlador de los datos (estado verdadero)", P_c, R_c)
print(f"      actividad media |Δa| por paso {actividad(A_c):.3f}")


def evaluar(etiqueta, mod, pp, modo):
    t0 = time.perf_counter()
    O_m, A_m, P_m, V_m, R_m, M_m = jax.jit(jax.vmap(lambda k: episodio_mpc(mod, pp, modo, k)))(claves_ev)
    jax.block_until_ready(P_m)
    LOGS[etiqueta] = (O_m, A_m, P_m)
    CURVAS[etiqueta], fin, fallo = resumen(etiqueta, P_m, R_m)
    print(f"      ({time.perf_counter()-t0:.0f} s; actividad media |Δa| por paso {actividad(A_m):.3f})")
    return fin, fallo


evaluar("física nominal · MPPI", FIS, None, "mppi")
# ablación con la física (cada evaluar() vuelve a trazar el planificador con los valores globales del momento)
W_V = 0.0
evaluar("física nominal · MPPI sin término de velocidad", FIS, None, "mppi")
RHO_R = 0.0
evaluar("física nominal · MPPI sin término de velocidad, ruido independiente", FIS, None, "mppi")
W_V = 1.0
evaluar("física nominal · MPPI, ruido independiente", FIS, None, "mppi")
RHO_R = 0.9
for nombre, mod in MODELOS.items():
    for modo in ("mppi", "politica", "mppi+politica"):
        evaluar(f"{nombre} · {modo}", mod, POL[nombre], modo)

for etiqueta in ("física nominal · MPPI", "RSSM · politica", "RSSM · mppi+politica"):
    P_m = LOGS[etiqueta][2]
    zerr = (np.asarray(P_m)[:, 100:, 2] - np.asarray(R_c)[:, None, 2]).mean(1)
    mm = np.asarray(M_c) / M0
    print(f"  {etiqueta}: error vertical medio en el último segundo con masa > +7 %: {np.nanmean(zerr[mm > 1.07])*100:+.1f} cm; "
          f"masa < −7 %: {np.nanmean(zerr[mm < 0.93])*100:+.1f} cm")
zc = (np.asarray(P_c)[:, 100:, 2] - np.asarray(R_c)[:, None, 2]).mean(1)
print(f"  controlador de los datos: ídem {zc[mm > 1.07].mean()*100:+.1f} cm y {zc[mm < 0.93].mean()*100:+.1f} cm")

tt = np.arange(T_EV) * 0.02
X5, Y5 = lin(0, 3.0, 60, 520), lin(0, 1.0, 250, 30)
for etiqueta, d in CURVAS.items():
    med = np.nanmedian(np.where(np.isfinite(d), d, np.nan), 0)
    print(f"path curva {etiqueta}:", path(tt, np.clip(med, 0, 1.0), X5, Y5))

print("  latencia por paso de control (un dron, GPU; actualizar la creencia + decidir):")
print(f"    física nominal · MPPI                   : {latencia(FIS, None, 'mppi'):6.1f} ms  ({EVALS} pasos del modelo)")
for nombre, mod in MODELOS.items():
    print(f"    {nombre:28s}: MPPI {latencia(mod, None, 'mppi'):6.1f} ms; MPPI + política {latencia(mod, POL[nombre], 'mppi+politica'):6.1f} ms; "
          f"política sola {latencia(mod, POL[nombre], 'politica'):6.2f} ms")

# ---------------------------------------------------------------------- 4. error del modelo en sus datos y en los del planificador
print("=" * 70)
print("4. ERROR DE PREDICCIÓN EN BUCLE ABIERTO: CON LAS ACCIONES DE LOS DATOS Y CON LAS DEL PROPIO PLANIFICADOR")


def error_abierto(mod, O, A, P, n_pred=50):
    def uno(O1, A1, P1):
        A_prev = jnp.concatenate([jnp.zeros((1, 4)), A1[:-1]], 0)
        def f(b, inp):
            o, a_prev, t = inp
            return mod["act"](b, norm(o), a_prev, None, t), None
        b, _ = lax.scan(f, mod["b0"], (O1[:T_FIL], A_prev[:T_FIL], jnp.arange(T_FIL)))
        Pp = imaginar(mod, b, A1[T_FIL - 1:T_FIL - 1 + n_pred])
        return jnp.linalg.norm(Pp - P1[T_FIL:T_FIL + n_pred], axis=-1)
    return np.asarray(jax.jit(jax.vmap(uno))(O, A, P))


def fila_error(nombre, e):
    return f"{np.nanmedian(e[:, 4])*100:5.1f}, {np.nanmedian(e[:, 24])*100:5.1f}, {np.nanmedian(e[:, 49])*100:6.1f}"


for nombre, mod in MODELOS.items():
    e_d = error_abierto(mod, O_te, A_te, P_te)
    txt = f"  {nombre:28s}: mediana a 0,1 / 0,5 / 1 s (cm): acciones de los datos {fila_error(nombre, e_d)}"
    for modo in ("mppi", "mppi+politica"):
        O_m, A_m, P_m = LOGS[f"{nombre} · {modo}"]
        txt += f" | de su {modo} {fila_error(nombre, error_abierto(mod, O_m, A_m, P_m))}"
    print(txt)

# ---------------------------------------------------------------------- 5. una iteración de datos (el bucle de Dyna)
print("=" * 70)
print("5. ITERACIÓN DE DATOS: 1000 EPISODIOS CON EL MPPI DEL RSSM (+ RUIDO), REENTRENAR Y VOLVER A EVALUAR")
N_IT = 200 if RAPIDO else 1000
claves_it = jax.random.split(jax.random.PRNGKey(99), N_IT)
t0 = time.perf_counter()
partes = []
f_it = jax.jit(jax.vmap(lambda k: episodio_mpc(MODELOS["RSSM"], None, "mppi", k, 0.5)))
for i in range(0, N_IT, 250):
    O_n, A_n, P_n, V_n, R_n, M_n = f_it(claves_it[i:i + 250])
    partes.append((O_n, A_n))
O_new = jnp.concatenate([q[0] for q in partes]); A_new = jnp.concatenate([q[1] for q in partes])
ok = np.isfinite(np.asarray(O_new)).all(axis=(1, 2)) & (np.abs(np.asarray(O_new)[:, :, :3]).max(axis=(1, 2)) < 5.0)
print(f"  {N_IT} episodios nuevos en {time.perf_counter()-t0:.0f} s; se quedan {ok.sum()} (finitos y a menos de 5 m)")
O_mix = jnp.concatenate([O_tr, O_new[ok]]); A_mix = jnp.concatenate([A_tr, A_new[ok]])
p_rssm2 = entrenar_con("RSSM reentrenado", p_rssm, lambda p, o, a, k: rssm_perdida(p, o, a, k, True), O_mix, A_mix,
                       pasos=300 if RAPIDO else 3000)
mod2 = modelo_rssm(p_rssm2)
evaluar("RSSM reentrenado · mppi", mod2, None, "mppi")
e_d = error_abierto(mod2, O_te, A_te, P_te)
O_m, A_m, P_m = LOGS["RSSM reentrenado · mppi"]
e_p = error_abierto(mod2, O_m, A_m, P_m)
print(f"  RSSM reentrenado: mediana a 0,1 / 0,5 / 1 s (cm): acciones de los datos {fila_error('', e_d)} | de su mppi {fila_error('', e_p)}")
d = CURVAS["RSSM reentrenado · mppi"]
med = np.nanmedian(np.where(np.isfinite(d), d, np.nan), 0)
print("path curva RSSM reentrenado · mppi:", path(tt, np.clip(med, 0, 1.0), X5, Y5))
