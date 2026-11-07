"""Datos de las figuras y tablas del tema A8a (model-based RL y world models latentes).

Uso:  uv run --with "jax[cuda13]" --with optax python -u figuras_a8a.py > salida.txt      (GPU; unos minutos)

Un RSSM mínimo (estilo PlaNet/Dreamer) aprende el x500 a partir de observaciones parciales: posición con ruido de
2 cm, actitud con 1° y giróscopo con 0,05 rad/s; sin velocidad; masa distinta en cada episodio (±15 %). Control
a 50 Hz (CTBR de un controlador de posición con ruido de exploración), física a 500 Hz.

  1. Datos: 3000 episodios de entrenamiento y 1000 de prueba, de 3 s.
  2. Tres modelos con la misma capacidad: RSSM (camino determinista + estocástico, KL con bits libres y pesos
     de DreamerV3), su versión determinista (sin estocasticidad) y un modelo sin memoria (MLP de un paso).
  3. Predicción en bucle abierto: filtrar 0,5 s con observaciones y predecir 1 s solo con las acciones.
  4. Sondeo lineal: ¿están la velocidad y la masa (que no se observan) en el estado latente?
"""
import math
import time
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
N_TR, N_TE = 3000, 1000
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
        return (x2, ou), (o, a, x[0], x[1])
    _, (O, A, P, V) = lax.scan(f, (x0, jnp.zeros(4)), jax.random.split(k[3], T_EP))
    return O, A, P, V, jnp.full(T_EP, m)


print("=" * 70)
print("1. DATOS")
t0 = time.perf_counter()
gen = jax.jit(jax.vmap(episodio))
O_tr, A_tr, P_tr, V_tr, M_tr = gen(jax.random.split(jax.random.PRNGKey(0), N_TR))
O_te, A_te, P_te, V_te, M_te = gen(jax.random.split(jax.random.PRNGKey(1), N_TE))
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


import os
PASOS = 300 if os.environ.get("A8A_RAPIDO") else 6000


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


print("=" * 70)
print("2. ENTRENAMIENTO (mismas secuencias de 50 pasos, Adam 3e-4)")
p_rssm = entrenar("RSSM", rssm_init, lambda p, o, a, k: rssm_perdida(p, o, a, k, True))
p_rssm05 = entrenar("RSSM con β_dyn = 0,5 (arXiv v1 de DreamerV3)", rssm_init, lambda p, o, a, k: rssm_perdida(p, o, a, k, True, 0.5))
p_det = entrenar("RSSM determinista", rssm_init, lambda p, o, a, k: rssm_perdida(p, o, a, k, False))
p_mlp = entrenar("MLP de un paso", mlp1_init, mlp1_perdida)

print("=" * 70)
print("3. PREDICCIÓN EN BUCLE ABIERTO: 0,5 s FILTRANDO Y 1 s SOLO CON LAS ACCIONES (1000 EPISODIOS DE PRUEBA)")
Ote_n = norm(O_te)
P_ref = np.asarray(P_te[:, T_FIL:T_FIL + T_PRED])           # posición verdadera (sin ruido)
des = lambda Pn: np.asarray(Pn[..., :3] * sd_o[:3] + mu_o[:3])
pred_r, LAT_r = rssm_predecir(p_rssm, Ote_n, A_te, jax.random.PRNGKey(5), True, n_muestras=20)
pred_r05, _ = rssm_predecir(p_rssm05, Ote_n, A_te, jax.random.PRNGKey(5), True, n_muestras=20)
pred_d, LAT_d = rssm_predecir(p_det, Ote_n, A_te, jax.random.PRNGKey(5), False, n_muestras=1)
pred_m = mlp1_predecir(p_mlp, Ote_n, A_te)
# referencia: «el dron se queda donde estaba» (última posición observada)
pred_c = np.repeat(np.asarray(O_te[:, T_FIL - 1:T_FIL, :3])[None], T_PRED, axis=2)
err = {}
for nombre, Pp in (("RSSM (media de 20 muestras)", des(pred_r).mean(0)), ("RSSM, β_dyn = 0,5", des(pred_r05).mean(0)), ("RSSM determinista", des(pred_d)[0]),
                   ("MLP de un paso", des(pred_m)[0]), ("quieto en la última posición", pred_c[0])):
    e = np.linalg.norm(Pp - P_ref, axis=-1)          # (N, T_PRED)
    err[nombre] = e.mean(0)
    print(f"  {nombre:30s}: error de posición a 0,1 s {e[:, 4].mean()*100:6.1f} cm, a 0,5 s {e[:, 24].mean()*100:6.1f} cm, a 1 s {e[:, 49].mean()*100:6.1f} cm")
# dispersión de las muestras del RSSM frente al error: ¿está calibrada la incertidumbre?
Ps = des(pred_r)                                         # (20, N, T, 3)
disp = np.linalg.norm(Ps - Ps.mean(0, keepdims=True), axis=-1).mean(0)   # (N, T)
e_r = np.linalg.norm(Ps.mean(0) - P_ref, axis=-1)
for t in (4, 24, 49):
    c = np.corrcoef(disp[:, t], e_r[:, t])[0, 1]
    print(f"  RSSM a {(t+1)*0.02:.1f} s: dispersión media de las muestras {disp[:, t].mean()*100:5.1f} cm; correlación dispersión-error entre episodios {c:.2f}")
X3, Y3 = lin(0, 1.0, 60, 520), lin(0, 1.0, 250, 30)
tt = (np.arange(T_PRED) + 1) * 0.02
for nombre, clave in (("rssm", "RSSM (media de 20 muestras)"), ("det", "RSSM determinista"), ("mlp", "MLP de un paso"), ("quieto", "quieto en la última posición")):
    print(f"path err {nombre}:", path(tt, np.clip(err[clave], 0, 1.0), X3, Y3))
# un episodio de ejemplo: x verdadera, observada y muestras del RSSM
j = int(np.argmax(np.abs(np.asarray(V_te[:, T_FIL, 0]))))   # uno que se mueve
tt_all = np.arange(T_FIL + T_PRED) * 0.02
X4 = lin(0, (T_FIL + T_PRED - 1) * 0.02, 60, 520)
xs = np.concatenate([np.asarray(P_te[j, :T_FIL + T_PRED, 0])])
lo, hi = float(min(xs.min(), Ps[:, j, :, 0].min())) - 0.1, float(max(xs.max(), Ps[:, j, :, 0].max())) + 0.1
Y4 = lin(lo, hi, 250, 30)
print(f"  ejemplo: episodio {j}; eje x de {lo:.2f} a {hi:.2f} m; marcas: " + ", ".join(f"{v:.1f} -> {Y4(v):.1f}" for v in np.arange(math.ceil(lo * 2) / 2, hi, 0.5)))
print("path ej verdad:", path(tt_all, xs, X4, Y4))
print("path ej obs:", path(tt_all[:T_FIL], np.asarray(O_te[j, :T_FIL, 0]), X4, Y4))
for s in range(5):
    print(f"path ej muestra {s}:", path(tt_all[T_FIL:], Ps[s, j, :, 0], X4, Y4))
print("path ej det:", path(tt_all[T_FIL:], des(pred_d)[0, j, :, 0], X4, Y4))
print("path ej mlp:", path(tt_all[T_FIL:], np.clip(des(pred_m)[0, j, :, 0], lo, hi), X4, Y4))
print(f"  línea de fin del filtrado en x = {X4((T_FIL - 0.5) * 0.02):.1f}")

print("=" * 70)
print("4. SONDEO LINEAL DEL ESTADO LATENTE (VELOCIDAD Y MASA NO SE OBSERVAN): 600 EPISODIOS PARA AJUSTAR, 400 PARA PROBAR")
def filtrar(p, O, A, key, estocastico=True):
    """Estados latentes (h, media del posterior) en todos los pasos de cada episodio."""
    B = O.shape[0]
    e = mlp(p["enc"], O)
    A_prev = jnp.concatenate([jnp.zeros((B, 1, 4)), A[:, :-1]], 1)
    def fil(c, x):
        h, z, k = c
        et, a_prev = x
        k, ks = jax.random.split(k)
        h = gru(p["gru"], h, jnp.concatenate([z, a_prev], -1))
        mq, sq = gauss(mlp(p["post"], jnp.concatenate([h, et], -1)))
        z = mq + (sq * jax.random.normal(ks, mq.shape) if estocastico else 0.0)
        return (h, z, k), jnp.concatenate([h, mq], -1)
    _, LAT = lax.scan(fil, (jnp.zeros((B, H)), jnp.zeros((B, Z)), key), (jnp.swapaxes(e, 0, 1), jnp.swapaxes(A_prev, 0, 1)))
    return np.asarray(jnp.swapaxes(LAT, 0, 1))
def r2(X, Y, ntr, lam=1.0):
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    W = np.linalg.solve(Xa[:ntr].T @ Xa[:ntr] + lam * np.eye(Xa.shape[1]), Xa[:ntr].T @ Y[:ntr])
    Yp = Xa[ntr:] @ W
    return 1 - ((Yp - Y[ntr:]) ** 2).sum(0) / ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum(0)
ntr = 600
Mt = np.asarray(M_te[:, 0])[:, None]
LATS = {"RSSM": filtrar(p_rssm, Ote_n, A_te, jax.random.PRNGKey(7), True),
        "RSSM, β_dyn = 0,5": filtrar(p_rssm05, Ote_n, A_te, jax.random.PRNGKey(7), True),
        "RSSM determinista": filtrar(p_det, Ote_n, A_te, jax.random.PRNGKey(7), False)}
for t_s, ti in ((0.5, T_FIL - 1), (1.5, 74), (3.0, T_EP - 1)):
    Vt = np.asarray(V_te[:, ti])
    for nombre, L in LATS.items():
        rv = r2(L[:, ti], Vt, ntr); rm = r2(L[:, ti], Mt, ntr)
        print(f"  a {t_s:.1f} s, {nombre:18s}: R² velocidad (x, y, z) {rv[0]:.2f} {rv[1]:.2f} {rv[2]:.2f}; R² masa {rm[0]:.2f}")
    Ob = np.asarray(O_te[:, ti]); rv = r2(Ob, Vt, ntr); rm = r2(Ob, Mt, ntr)
    print(f"  a {t_s:.1f} s, {'última observación':18s}: R² velocidad (x, y, z) {rv[0]:.2f} {rv[1]:.2f} {rv[2]:.2f}; R² masa {rm[0]:.2f}")
Ob2 = np.asarray(jnp.concatenate([O_te[:, T_FIL - 1], O_te[:, T_FIL - 2], A_te[:, T_FIL - 2]], -1))
rv = r2(Ob2, np.asarray(V_te[:, T_FIL - 1]), ntr)
print(f"  a 0.5 s, {'dos observaciones':18s}: R² velocidad (x, y, z) {rv[0]:.2f} {rv[1]:.2f} {rv[2]:.2f}  (diferencia finita con 2 cm de ruido en 20 ms)")
