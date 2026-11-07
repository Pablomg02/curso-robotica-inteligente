"""Datos de las figuras y tablas del tema A6b (RL en robots reales).

Uso:
  uv run --with "jax[cuda13]" --with optax python figuras_a6b.py > salida.txt     (GPU; unos minutos)

Todo con el x500 de A6a y aleatorización por entorno (masa ±20 %, ganancia de empuje ±10 %,
constante del motor de 20 a 60 ms). Observación «realista» del actor: posición con ruido de 2 cm y
40 ms de retardo, actitud con ruido de 1°, giróscopo con ruido de 0,05 rad/s, la acción anterior, sin
velocidad, y la historia de los 4 últimos pasos. Observación «privilegiada»: estado verdadero y
parámetros físicos.

  1. Crítico simétrico frente a asimétrico (actor con la observación realista), 5 semillas.
  2. Maestro privilegiado y alumno con la observación realista: clonación (BC) frente a DAgger.
  3. Espacio de acción: CTBR frente a empujes de rotor (SRT); con y sin término de suavidad; evaluación
     nominal, con ruido de sensores y con 20 ms de retardo de acción no modelado.
  4. Retardo que añade un filtro paso bajo en la acción (cálculo analítico).

Uso por secciones: python figuras_a6b.py 1 3   (sin argumentos, todas).
"""
import math
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax

print("dispositivo:", jax.devices()[0], "| jax", jax.__version__)
M0, G = 2.0, 9.81
J = jnp.array([0.022, 0.022, 0.040])
L_ARM, KAPPA = 0.25, 0.016           # brazo (cruz) y relación par/empuje de un rotor
DT, SUB = 0.002, 5
EP_LEN = 500
H_OBS = 4                            # historia del actor (pasos de 10 ms)
D_POS = 5                            # la más antigua de 5 posiciones: 40 ms de retardo


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


# estado: [p3 v3 q4 om3 | f4 (empujes de rotor reales) | aprev4 | t1 | pref3 | mfac, kT, taum | phist 3*D_POS | ohist OBS1*H_OBS]
OBS1 = 3 + 9 + 3 + 4                 # una observación realista: pos. retrasada, R, giróscopo, acción anterior
I_P, I_V, I_Q, I_OM, I_F, I_A, I_T, I_REF, I_PAR = 0, 3, 6, 10, 13, 17, 21, 22, 25
I_PH = 28
I_OH = I_PH + 3 * D_POS
NS = I_OH + OBS1 * H_OBS


def mando_rotores(s, a, modo):
    """Empujes de rotor deseados (antes del retardo del motor) a partir de la acción."""
    mfac, kT = s[I_PAR], s[I_PAR + 1]
    if modo == "ctbr":
        om = s[I_OM:I_OM + 3]
        T_cmd = M0 * G * (1.0 + 0.8 * a[0])
        om_d = jnp.array([4.0, 4.0, 2.0]) * a[1:4]
        tau = J * 20.0 * (om_d - om) + jnp.cross(om, J * om)
        # asignación en cruz: f0 (+x), f1 (+y), f2 (-x), f3 (-y)
        f = jnp.array([T_cmd / 4 - tau[1] / (2 * L_ARM) + tau[2] / (4 * KAPPA),
                       T_cmd / 4 + tau[0] / (2 * L_ARM) - tau[2] / (4 * KAPPA),
                       T_cmd / 4 + tau[1] / (2 * L_ARM) + tau[2] / (4 * KAPPA),
                       T_cmd / 4 - tau[0] / (2 * L_ARM) - tau[2] / (4 * KAPPA)])
    else:                            # "srt": empuje de cada rotor
        f = M0 * G / 4 * (1.0 + 0.8 * a)
    return kT * jnp.clip(f, 0.0, M0 * G)


def fisica(s, f_cmd):
    p, v, q, om, f = s[I_P:I_P + 3], s[I_V:I_V + 3], s[I_Q:I_Q + 4], s[I_OM:I_OM + 3], s[I_F:I_F + 4]
    m = M0 * s[I_PAR]
    taum = s[I_PAR + 2]
    def sub(i, x):
        p, v, q, om, f = x
        f = f + DT / taum * (f_cmd - f)
        T = jnp.sum(f)
        tau = jnp.array([L_ARM * (f[1] - f[3]), L_ARM * (f[2] - f[0]), KAPPA * (f[0] - f[1] + f[2] - f[3])])
        acc = rotmat(q) @ jnp.array([0.0, 0.0, T / m]) - jnp.array([0.0, 0.0, G]) - 0.15 / m * v
        v = v + DT * acc
        p = p + DT * v
        om = om + DT * (tau - jnp.cross(om, J * om)) / J
        q = q + 0.5 * DT * qmul(q, jnp.concatenate([jnp.zeros(1), om]))
        q = q / jnp.linalg.norm(q)
        return (p, v, q, om, f)
    p, v, q, om, f = lax.fori_loop(0, SUB, sub, (p, v, q, om, f))
    return p, v, q, om, f


def obs_realista(s, key):
    """Una observación con ruido: posición retrasada D_POS pasos, R, giróscopo y acción anterior."""
    k1, k2, k3 = jax.random.split(key, 3)
    p_ret = s[I_PH:I_PH + 3]                                   # la más antigua del histórico
    e = p_ret - s[I_REF:I_REF + 3] + 0.02 * jax.random.normal(k1, (3,))
    ang = 0.0175 * jax.random.normal(k2, (3,))                 # 1° de ruido de actitud
    dq = jnp.concatenate([jnp.ones(1), 0.5 * ang]); dq = dq / jnp.linalg.norm(dq)
    R = rotmat(qmul(s[I_Q:I_Q + 4], dq)).reshape(-1)
    om = s[I_OM:I_OM + 3] + 0.05 * jax.random.normal(k3, (3,))
    return jnp.concatenate([e, R, om, s[I_A:I_A + 4]])


def obs_privilegiada(s):
    return jnp.concatenate([s[I_P:I_P + 3] - s[I_REF:I_REF + 3], s[I_V:I_V + 3], rotmat(s[I_Q:I_Q + 4]).reshape(-1),
                            s[I_OM:I_OM + 3], s[I_A:I_A + 4], s[I_PAR:I_PAR + 3] - jnp.array([1.0, 1.0, 0.04])])


def obs_priv_ruidosa(s, key):
    """La observación privilegiada con el ruido de sensores de la realista (sin retardo): 2 cm, 0,1 m/s, 1°, 0,05 rad/s."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    ang = 0.0175 * jax.random.normal(k3, (3,))
    dq = jnp.concatenate([jnp.ones(1), 0.5 * ang]); dq = dq / jnp.linalg.norm(dq)
    return jnp.concatenate([s[I_P:I_P + 3] - s[I_REF:I_REF + 3] + 0.02 * jax.random.normal(k1, (3,)),
                            s[I_V:I_V + 3] + 0.1 * jax.random.normal(k2, (3,)),
                            rotmat(qmul(s[I_Q:I_Q + 4], dq)).reshape(-1),
                            s[I_OM:I_OM + 3] + 0.05 * jax.random.normal(k4, (3,)), s[I_A:I_A + 4],
                            s[I_PAR:I_PAR + 3] - jnp.array([1.0, 1.0, 0.04])])


def obs_historia(s):
    return s[I_OH:I_OH + OBS1 * H_OBS]


def estado_inicial(key, dr=True):
    k = jax.random.split(key, 8)
    p = jax.random.uniform(k[0], (3,), minval=-1.5, maxval=1.5)
    v = jax.random.uniform(k[1], (3,), minval=-1.0, maxval=1.0)
    eje = jax.random.normal(k[2], (3,)); eje = eje / jnp.linalg.norm(eje)
    ang = jax.random.uniform(k[3], (), minval=0.0, maxval=math.radians(30))
    q = jnp.concatenate([jnp.array([jnp.cos(ang / 2)]), jnp.sin(ang / 2) * eje])
    om = jax.random.uniform(k[4], (3,), minval=-1.0, maxval=1.0)
    if dr:
        mfac = jax.random.uniform(k[5], (), minval=0.8, maxval=1.2)
        kT = jax.random.uniform(k[6], (), minval=0.9, maxval=1.1)
        taum = jax.random.uniform(k[7], (), minval=0.02, maxval=0.06)
    else:
        mfac, kT, taum = 1.0, 1.0, 0.04
    f = jnp.full(4, M0 * mfac * G / 4)
    base = jnp.concatenate([p, v, q, om, f, jnp.zeros(4), jnp.zeros(1), jnp.zeros(3), jnp.array([mfac, kT, taum]),
                            jnp.tile(p, D_POS), jnp.zeros(OBS1 * H_OBS)])
    # rellenar la historia de observaciones con la primera (sin ruido extra)
    o0 = obs_realista(base, k[0])
    return base.at[I_OH:].set(jnp.tile(o0, H_OBS))


def hacer_paso(modo, w_suav):
    def paso(s, a, key):
        a = jnp.clip(a, -1.0, 1.0)
        f_cmd = mando_rotores(s, a, modo)
        p, v, q, om, f = fisica(s, f_cmd)
        t = s[I_T] + 1
        ph = jnp.concatenate([s[I_PH + 3:I_PH + 3 * D_POS], p])          # desplaza el histórico de posiciones
        s2 = jnp.concatenate([p, v, q, om, f, a, jnp.array([t]), s[I_REF:I_REF + 3], s[I_PAR:I_PAR + 3], ph,
                              s[I_OH:I_OH + OBS1 * H_OBS]])
        k1, k2 = jax.random.split(key)
        o_nueva = obs_realista(s2, k1)
        s2 = s2.at[I_OH:].set(jnp.concatenate([s[I_OH + OBS1:I_OH + OBS1 * H_OBS], o_nueva]))
        e = p - s2[I_REF:I_REF + 3]
        incl = jnp.arccos(jnp.clip(rotmat(q)[2, 2], -1, 1))
        coste = jnp.linalg.norm(e) + 0.1 * jnp.linalg.norm(v) + 0.05 * jnp.linalg.norm(om) + w_suav * jnp.sum((a - s[I_A:I_A + 4]) ** 2)
        choque = (jnp.linalg.norm(e) > 4.0) | (incl > math.radians(80)) | ~jnp.all(jnp.isfinite(s2))
        r = 1.0 - jnp.tanh(coste) - 10.0 * choque
        tiempo = t >= EP_LEN
        hecho = choque | tiempo
        s_fin = s2
        s_new = jnp.where(hecho, estado_inicial(k2), s2)
        return s_new, r, choque, tiempo, s_fin
    return paso


def mlp_init(key, tam, g_final):
    ks = jax.random.split(key, len(tam) - 1)
    return [(jax.nn.initializers.orthogonal(g_final if i == len(tam) - 2 else math.sqrt(2))(ks[i], (a, b)), jnp.zeros(b))
            for i, (a, b) in enumerate(zip(tam[:-1], tam[1:]))]


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def logprob(m, ls, a):
    return jnp.sum(-0.5 * ((a - m) / jnp.exp(ls)) ** 2 - ls - 0.5 * math.log(2 * math.pi), axis=-1)


OBS_FN = {"priv": lambda s: obs_privilegiada(s), "hist": lambda s: obs_historia(s)}
OBS_DIM = {"priv": 25, "hist": OBS1 * H_OBS}
BASE = dict(n_env=4096, n_pasos=32, epocas=5, n_mini=4, gamma=0.99, lam=0.95, clip=0.2, lr=3e-4, c_v=0.5,
            max_grad=0.5, n_upd=600, actor="hist", critico="priv", modo="ctbr", w_suav=0.1)


def entrenar(cfg, seed=0):
    paso = hacer_paso(cfg["modo"], cfg["w_suav"])
    oa, oc = OBS_FN[cfg["actor"]], OBS_FN[cfg["critico"]]
    key = jax.random.PRNGKey(seed)
    key, k1, k2, ke, kt = jax.random.split(key, 5)
    red = {"pi": mlp_init(k1, [OBS_DIM[cfg["actor"]], 64, 64, 4], 0.01), "v": mlp_init(k2, [OBS_DIM[cfg["critico"]], 64, 64, 1], 1.0),
           "logstd": jnp.full(4, -0.5)}
    opt = optax.chain(optax.clip_by_global_norm(cfg["max_grad"]), optax.adam(cfg["lr"], eps=1e-5))
    opt_st = opt.init(red)
    S = jax.vmap(estado_inicial)(jax.random.split(ke, cfg["n_env"]))
    S = S.at[:, I_T].set(jax.random.randint(kt, (cfg["n_env"],), 0, EP_LEN).astype(jnp.float32))

    def rollout(c, _):
        red, S, key = c
        key, ka, kr = jax.random.split(key, 3)
        OA, OC = jax.vmap(oa)(S), jax.vmap(oc)(S)
        m = mlp(red["pi"], OA)
        a = m + jnp.exp(red["logstd"]) * jax.random.normal(ka, m.shape)
        lp = logprob(m, red["logstd"], a)
        v = mlp(red["v"], OC)[:, 0]
        S2, r, ch, ti, Sfin = jax.vmap(paso)(S, a, jax.random.split(kr, cfg["n_env"]))
        vfin = mlp(red["v"], jax.vmap(oc)(Sfin))[:, 0]
        r_aj = r + cfg["gamma"] * vfin * (ti & ~ch)
        return (red, S2, key), (OA, OC, a, lp, v, r_aj, ch | ti, r, ch)

    def actualizar(st, _):
        red, opt_st, S, key = st
        (red, S, key), (OA, OC, A, LP, V, R, D, Rc, CH) = lax.scan(rollout, (red, S, key), None, length=cfg["n_pasos"])
        v_ult = mlp(red["v"], jax.vmap(oc)(S))[:, 0]
        def gae(c, x):
            adv, vs = c
            r, v, d = x
            delta = r + cfg["gamma"] * vs * (1 - d) - v
            adv = delta + cfg["gamma"] * cfg["lam"] * (1 - d) * adv
            return (adv, v), adv
        _, ADV = lax.scan(gae, (jnp.zeros_like(v_ult), v_ult), (R, V, D.astype(jnp.float32)), reverse=True)
        RET = ADV + V
        B = cfg["n_env"] * cfg["n_pasos"]
        datos = tuple(x.reshape((B,) + x.shape[2:]) for x in (OA, OC, A, LP, ADV, RET))
        def perdida(red, OA, OC, A, LP, ADV, RET):
            m = mlp(red["pi"], OA)
            lp = logprob(m, red["logstd"], A)
            ratio = jnp.exp(lp - LP)
            adv = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            lpi = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * adv))
            lv = 0.5 * jnp.mean((mlp(red["v"], OC)[:, 0] - RET) ** 2)
            return lpi + cfg["c_v"] * lv, lv
        def epoca(c, _):
            red, opt_st, key = c
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, B)
            mini = tuple(x[perm].reshape((cfg["n_mini"], B // cfg["n_mini"]) + x.shape[1:]) for x in datos)
            def pm(c2, mb):
                red, opt_st = c2
                (l, lv), g = jax.value_and_grad(perdida, has_aux=True)(red, *mb)
                u, opt_st = opt.update(g, opt_st, red)
                return (optax.apply_updates(red, u), opt_st), lv
            (red, opt_st), lv = lax.scan(pm, (red, opt_st), mini)
            return (red, opt_st, key), lv
        (red, opt_st, key), lv = lax.scan(epoca, (red, opt_st, key), None, length=cfg["epocas"])
        return (red, opt_st, S, key), (Rc.mean(), CH.mean(), lv.mean())

    act = jax.jit(actualizar)
    st = (red, opt_st, S, key)
    st, _ = act(st, None)
    t0 = time.perf_counter()
    hist = []
    for i in range(cfg["n_upd"]):
        st, met = act(st, None)
        if i % 20 == 0 or i == cfg["n_upd"] - 1:
            hist.append((i,) + tuple(float(x) for x in met))
    jax.block_until_ready(st)
    return st[0], hist, time.perf_counter() - t0


def evaluar(politica, actor, modo="ctbr", n=400, pasos=1000, seed=123, retardo=0, ruido=False):
    """politica(obs) -> acción determinista. Devuelve choques, error medio en los últimos 5 s, error vertical
    frente a la masa, suavidad media |Δa| y actividad de los rotores (Σ|Δf_cmd| por paso, en pesos del dron)."""
    paso = hacer_paso(modo, 0.1)
    oa = OBS_FN[actor]
    @jax.jit
    def correr(key):
        S = jax.vmap(estado_inicial)(jax.random.split(key, n))
        buf0 = jnp.zeros((retardo + 1, n, 4))          # acciones en vuelo: retardo no modelado de `retardo` pasos
        fc0 = S[:, I_F:I_F + 4]
        def f(c, k):
            S, buf, fprev = c
            S = S.at[:, I_T].set(0.0)
            kk = jax.random.fold_in(key, 10_000 + k)
            O = jax.vmap(obs_priv_ruidosa)(S, jax.random.split(kk, n)) if ruido else jax.vmap(oa)(S)
            a_env = politica(O)
            buf = jnp.concatenate([buf[1:], a_env[None]], axis=0)
            a = buf[0]
            S2, r, ch, ti, _ = jax.vmap(paso)(S, a, jax.random.split(jax.random.fold_in(key, k), n))
            e = S[:, I_P:I_P + 3] - S[:, I_REF:I_REF + 3]
            da = jnp.abs(jnp.clip(a, -1, 1) - S[:, I_A:I_A + 4]).mean(1)
            fc = jax.vmap(lambda s_, a_: mando_rotores(s_, jnp.clip(a_, -1, 1), modo))(S, a)
            act_r = jnp.sum(jnp.abs(fc - fprev), axis=1) / (M0 * G)
            return (S2, buf, fc), (jnp.linalg.norm(e, axis=1), e[:, 2], ch, da, S[:, I_PAR], act_r)
        _, out = lax.scan(f, (S, buf0, fc0), jnp.arange(pasos))
        return out
    E, EZ, CH, DA, MF, AR = (np.asarray(x) for x in correr(jax.random.PRNGKey(seed)))
    vivos = ~CH.any(0)
    ez = EZ[500:, :].mean(0)
    mf = MF[0, :]
    pend = np.polyfit(mf[vivos], ez[vivos], 1)[0] if vivos.sum() > 10 else float("nan")
    return dict(choques=1 - vivos.mean(), err=E[500:, vivos].mean() if vivos.any() else float("nan"),
                pend_z=pend, suav=DA[500:, vivos].mean() if vivos.any() else float("nan"),
                act=AR[500:, vivos].mean() if vivos.any() else float("nan"))


def diagnostico(red, modo, n=64, pasos=1000, seed=7):
    """Oscilación de la acción en estacionario nominal: frecuencia dominante de a(t) en los últimos 5 s
    del entorno que más oscila, y autovalores de ∂π/∂a_prev en su estado final."""
    paso = hacer_paso(modo, 0.1)
    pol = lambda O: mlp(red["pi"], O)
    @jax.jit
    def correr(key):
        S = jax.vmap(estado_inicial)(jax.random.split(key, n))
        def f(S, k):
            S = S.at[:, I_T].set(0.0)
            a = pol(jax.vmap(obs_privilegiada)(S))
            S2, *_ = jax.vmap(paso)(S, a, jax.random.split(jax.random.fold_in(key, k), n))
            return S2, (jnp.clip(a, -1, 1), S)
        _, (A, SS) = lax.scan(f, S, jnp.arange(pasos))
        return A, SS
    A, SS = correr(jax.random.PRNGKey(seed))
    A = np.asarray(A)[pasos // 2:]
    var = A.var(0).sum(1)
    j = int(var.argmax())
    x = A[:, j, :] - A[:, j, :].mean(0)
    P = (np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(1)
    fr = np.fft.rfftfreq(x.shape[0], d=SUB * DT)
    f_dom = fr[1:][P[1:].argmax()]
    s_fin = SS[-1, j]
    Jo = jax.jacfwd(lambda o: pol(o))(obs_privilegiada(s_fin))
    ev = np.linalg.eigvals(np.asarray(Jo[:, 18:22]))
    return dict(desv=float(np.sqrt(var[j])), f=float(f_dom), ev=ev, frac=float((var > 1e-6).mean()))


def pol_red(red):
    return lambda O: mlp(red["pi"], O)


import sys
SECCIONES = {int(x) for x in sys.argv[1:]} or {1, 2, 3, 4}
# ====================================================================== 1. CRÍTICO
if 1 in SECCIONES:
    print("=" * 70)
    print("1. ACTOR CON OBSERVACIÓN REALISTA: CRÍTICO SIMÉTRICO (MISMA OBSERVACIÓN) FRENTE A ASIMÉTRICO (ESTADO Y PARÁMETROS)")
    res1 = {}
    for critico in ("hist", "priv"):
        filas = []
        for sd in range(5):
            red, h, tt = entrenar(dict(BASE, critico=critico), seed=sd)
            ev = evaluar(pol_red(red), "hist")
            filas.append((ev["choques"], ev["err"], h[-1][3], tt))
            Xc, Yc = lin(0, BASE["n_upd"], 60, 520), lin(-0.1, 0.7, 250, 30)
            print(f"path curva {critico} {sd}:", path([x[0] for x in h], np.clip([x[1] for x in h], -0.1, 0.7), Xc, Yc))
            print(f"    recompensa media por paso: mín. {min(x[1] for x in h):.3f}, final {h[-1][1]:.3f}")
            print(f"  crítico {'asimétrico' if critico == 'priv' else 'simétrico '} semilla {sd}: choques {ev['choques']*100:5.1f} %, error {ev['err']*100:6.1f} cm, "
                  f"pérdida del valor al final {h[-1][3]:.4f}, {tt:.0f} s")
        res1[critico] = filas
        ch = np.array([f[0] for f in filas]); er = np.array([f[1] for f in filas])
        print(f"  => {critico}: choques medios {ch.mean()*100:.1f} %; error medio {np.nanmean(er)*100:.1f} cm (de {np.nanmin(er)*100:.1f} a {np.nanmax(er)*100:.1f})")

if 2 in SECCIONES:
    # ====================================================================== 2. MAESTRO Y ALUMNO
    print("=" * 70)
    print("2. MAESTRO PRIVILEGIADO Y ALUMNO CON LA OBSERVACIÓN REALISTA: CLONACIÓN FRENTE A DAgger")
    maestro, hm, tm = entrenar(dict(BASE, actor="priv", critico="priv"), seed=0)
    ev_m = evaluar(pol_red(maestro), "priv")
    print(f"  maestro (ve estado y parámetros): choques {ev_m['choques']*100:.1f} %, error {ev_m['err']*100:.1f} cm, pendiente del error vertical con la masa {ev_m['pend_z']*100:.1f} cm por unidad; {tm:.0f} s")
    paso_e = hacer_paso("ctbr", 0.1)


    def entrenar_alumno(modo, seed=0, iters=40, n_env=4096, n_pasos=32):
        key = jax.random.PRNGKey(100 + seed)
        key, ka, ke = jax.random.split(key, 3)
        alumno = mlp_init(ka, [OBS_DIM["hist"], 64, 64, 4], 0.01)
        opt = optax.adam(1e-3)
        ost = opt.init(alumno)
        S = jax.vmap(estado_inicial)(jax.random.split(ke, n_env))
        tam_buf = 4 * n_env * n_pasos
        buf_o = jnp.zeros((tam_buf, OBS_DIM["hist"])); buf_a = jnp.zeros((tam_buf, 4))

        @jax.jit
        def recoger(alumno, S, key, beta):
            def f(c, _):
                S, key = c
                key, kr, kb = jax.random.split(key, 3)
                Oa = jax.vmap(obs_historia)(S)
                a_m = mlp(maestro["pi"], jax.vmap(obs_privilegiada)(S))
                a_s = mlp(alumno, Oa)
                usa_m = jax.random.uniform(kb, (S.shape[0], 1)) < beta      # DAgger con mezcla β
                a = jnp.where(usa_m, a_m, a_s) + 0.1 * jax.random.normal(kr, a_m.shape)
                S2, r, ch, ti, _ = jax.vmap(paso_e)(S, a, jax.random.split(kr, S.shape[0]))
                return (S2, key), (Oa, a_m)
            (S, key), (O, Am) = lax.scan(f, (S, key), None, length=n_pasos)
            return S, key, O.reshape(-1, O.shape[-1]), Am.reshape(-1, 4)

        @jax.jit
        def ajustar(alumno, ost, bo, ba, key):
            def ep(c, k):
                alumno, ost = c
                perm = jax.random.permutation(k, bo.shape[0])
                mo = bo[perm].reshape(16, -1, bo.shape[1]); ma = ba[perm].reshape(16, -1, 4)
                def mb(c2, x):
                    alumno, ost = c2
                    l, g = jax.value_and_grad(lambda p: jnp.mean((mlp(p, x[0]) - x[1]) ** 2))(alumno)
                    u, ost = opt.update(g, ost, alumno)
                    return (optax.apply_updates(alumno, u), ost), l
                (alumno, ost), l = lax.scan(mb, (alumno, ost), (mo, ma))
                return (alumno, ost), l.mean()
            (alumno, ost), l = lax.scan(ep, (alumno, ost), jax.random.split(key, 4))
            return alumno, ost, l[-1]

        t0 = time.perf_counter()
        lleno = 0
        for it in range(iters):
            beta = 1.0 if (modo == "bc" or it == 0) else 0.0
            S, key, O, Am = recoger(alumno, S, key, beta)
            n = O.shape[0]
            i0 = (it * n) % tam_buf
            buf_o = lax.dynamic_update_slice(buf_o, O, (i0, 0)); buf_a = lax.dynamic_update_slice(buf_a, Am, (i0, 0))
            lleno = min(tam_buf, lleno + n)
            key, kf = jax.random.split(key)
            alumno, ost, l = ajustar(alumno, ost, buf_o[:lleno], buf_a[:lleno], kf)
        jax.block_until_ready(alumno)
        return alumno, float(l), time.perf_counter() - t0


    res2 = {}
    for modo in ("bc", "dagger"):
        filas = []
        for sd in range(3):
            alumno, l, tt = entrenar_alumno(modo, seed=sd)
            ev = evaluar(lambda O, al=alumno: mlp(al, O), "hist")
            filas.append((ev["choques"], ev["err"]))
            print(f"  alumno {modo:6s} semilla {sd}: pérdida de imitación {l:.4f}; choques {ev['choques']*100:5.1f} %, error {ev['err']*100:6.1f} cm; {tt:.0f} s")
        res2[modo] = filas

# ====================================================================== 3. ESPACIO DE ACCIÓN Y SUAVIDAD
if 3 in SECCIONES:
    print("=" * 70)
    print("3. ESPACIO DE ACCIÓN Y SUAVIDAD (ACTOR Y CRÍTICO PRIVILEGIADOS, 5 SEMILLAS)")
    print("   evaluación: nominal; con el ruido de sensores en la observación; con 20 ms de retardo de acción no modelado")
    print("   actividad = Σ|Δf_cmd| de los 4 rotores por paso de 10 ms, en pesos del dron")
    for modo, w in (("ctbr", 0.1), ("srt", 0.1), ("ctbr", 0.0), ("srt", 0.0)):
        filas = []
        for sd in range(5):
            red, h, tt = entrenar(dict(BASE, actor="priv", critico="priv", modo=modo, w_suav=w), seed=sd)
            ev0 = evaluar(pol_red(red), "priv", modo=modo)
            evr = evaluar(pol_red(red), "priv", modo=modo, ruido=True)
            ev2 = evaluar(pol_red(red), "priv", modo=modo, retardo=2)
            filas.append((ev0["choques"], ev0["err"], ev0["act"], evr["choques"], evr["err"], evr["act"], ev2["choques"], ev2["err"]))
            dg = diagnostico(red, modo)
            evs = " ".join(f"{z.real:+.2f}{z.imag:+.2f}j" for z in sorted(dg["ev"], key=lambda z: -abs(z)))
            print(f"     semilla {sd}: nominal actividad {ev0['act']:.4f}, error {ev0['err']*100:.1f} cm; entornos que oscilan {dg['frac']*100:.0f} %; "
                  f"el que más: desviación de a {dg['desv']:.3f}, frecuencia dominante {dg['f']:.1f} Hz; autovalores de ∂π/∂a_prev {evs}")
        f = np.array(filas)
        print(f"  {modo.upper():4s} suavidad {w}: nominal {f[:,0].mean()*100:4.1f} % choques, {np.nanmean(f[:,1])*100:5.1f} cm, actividad {np.nanmean(f[:,2]):.4f} | "
              f"con ruido {f[:,3].mean()*100:4.1f} %, {np.nanmean(f[:,4])*100:5.1f} cm, actividad {np.nanmean(f[:,5]):.4f} (de {np.nanmin(f[:,5]):.4f} a {np.nanmax(f[:,5]):.4f}) | "
              f"con 20 ms {f[:,6].mean()*100:4.1f} %, {np.nanmean(f[:,7])*100:5.1f} cm")
        print("     por semilla (actividad con ruido / error con ruido / choques con 20 ms): " + "; ".join(f"{x[5]:.4f} / {x[4]*100:.1f} cm / {x[6]*100:.0f} %" for x in filas))

# ====================================================================== 4. RETARDO DE UN FILTRO PASO BAJO
if 4 in SECCIONES:
    print("=" * 70)
    print("4. RETARDO DE FASE DE UN FILTRO PASO BAJO DE PRIMER ORDEN EN LA ACCIÓN: atan(ωτ)/ω")
    fr = np.linspace(0.05, 20, 200)
    X6, Y6 = lin(0, 20, 60, 520), lin(0, 25, 250, 30)
    for tau in (0.005, 0.010, 0.020):
        w_ = 2 * np.pi * fr
        dl = np.arctan(w_ * tau) / w_ * 1000
        print(f"  τ = {tau*1000:.0f} ms (corte {1/(2*np.pi*tau):.1f} Hz): retardo a 1 Hz {np.interp(1, fr, dl):.1f} ms, a 5 Hz {np.interp(5, fr, dl):.1f} ms, a 10 Hz {np.interp(10, fr, dl):.1f} ms")
        print(f"path filtro {tau*1000:.0f}:", path(fr, dl, X6, Y6))
    print("  media móvil de N muestras a 100 Hz: retardo de grupo constante (N-1)/2 · 10 ms; N = 5 -> 20 ms")
