"""Datos de las figuras y tablas del tema A6a (deep RL para control).

Uso:
  uv run --with "jax[cuda13]" --with optax python figuras_a6a.py > salida.txt     (GPU; unos minutos)
  JAX_PLATFORMS=cpu uv run --with jax --with optax python figuras_a6a.py > salida.txt   (CPU; bastante más lento)

Secciones:
  1. Objetivo recortado de PPO (analítico).
  2. GAE: sesgo y varianza frente a λ con un crítico imperfecto (paseo aleatorio de 19 estados).
  3. Moldeado de la recompensa: potencial frente a una bonificación ingenua (rejilla, iteración de valor).
  4. PPO en JAX con el x500 (empuje colectivo y velocidades angulares), miles de entornos: curva de
     aprendizaje, error en estacionario y seguimiento de un círculo.
"""
import math
import time
import numpy as np


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def suavizar(v, k=5):
    """Media móvil centrada de k registros (los extremos con la ventana que quepa)."""
    v = np.asarray(v, dtype=float)
    return np.array([v[max(0, i - k // 2): i + k // 2 + 1].mean() for i in range(len(v))])


# ====================================================================== 1. OBJETIVO RECORTADO
print("=" * 70)
print("1. OBJETIVO RECORTADO DE PPO: min(r A, clip(r, 1-ε, 1+ε) A) con ε = 0,2")
eps = 0.2
r = np.linspace(0.0, 2.0, 201)
X1 = lin(0, 2, 60, 520)
Y1 = lin(-2, 2, 250, 30)
for A, nombre in ((1.0, "A>0"), (-1.0, "A<0")):
    L = np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)
    print(f"path clip {nombre}:", path(r, L, X1, Y1))
    print(f"path sinclip {nombre}:", path(r, r * A, X1, Y1))
print(f"  x(0,8) = {X1(0.8):.1f}; x(1) = {X1(1):.1f}; x(1,2) = {X1(1.2):.1f}; y(0) = {Y1(0):.1f}; y(1) = {Y1(1):.1f}; y(-1) = {Y1(-1):.1f}")

# ====================================================================== 2. GAE
print("=" * 70)
print("2. GAE: SESGO Y VARIANZA DEL GRADIENTE DE LA POLÍTICA FRENTE A λ, CON UN CRÍTICO IMPERFECTO")
print("   cadena de 19 estados; la acción «derecha» avanza con prob. 0,6 y «izquierda» con 0,4; +1 al salir por la")
print("   derecha, 0 por la izquierda; π(derecha) = σ(θ), θ = 0; γ = 0,97; se empieza en el centro")
rng = np.random.default_rng(0)
NS, GAM = 19, 0.97
P_DER = {1: 0.6, 0: 0.4}


def valor_exacto(theta):
    pd = 1 / (1 + math.exp(-theta))
    p_up = pd * 0.6 + (1 - pd) * 0.4
    A = np.zeros((NS, NS)); b = np.zeros(NS)
    for i in range(NS):
        A[i, i] = 1.0
        if i + 1 < NS:
            A[i, i + 1] -= GAM * p_up
        else:
            b[i] += GAM * p_up * 1.0      # sale por la derecha: +1
        if i - 1 >= 0:
            A[i, i - 1] -= GAM * (1 - p_up)
    return np.linalg.solve(A, b)


TH = 0.0
Vtrue = valor_exacto(TH)
g_true = (valor_exacto(TH + 1e-5)[NS // 2] - valor_exacto(TH - 1e-5)[NS // 2]) / 2e-5
# crítico imperfecto: error suave (subestima el centro) más ruido fijo por estado
err = -0.15 * np.sin(np.linspace(0, math.pi, NS)) + rng.normal(0, 0.05, NS)
Vhat = Vtrue + err
pd = 1 / (1 + math.exp(-TH))


def episodio():
    s, tr = NS // 2, []
    while True:
        a = 1 if rng.random() < pd else 0
        sube = rng.random() < P_DER[a]
        s2 = s + (1 if sube else -1)
        fin = s2 < 0 or s2 >= NS
        r = 1.0 if s2 >= NS else 0.0
        tr.append((s, a, r, s2, fin))
        if fin:
            return tr
        s = s2


episodios = [episodio() for _ in range(20000)]
print(f"  gradiente verdadero dJ/dθ = {g_true:.4f}; longitud media de episodio {np.mean([len(e) for e in episodios]):.0f} pasos")
res_gae = []
for lam in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
    gs = []
    for tr in episodios:
        T = len(tr); adv = 0.0; g = 0.0
        advs = np.zeros(T)
        for t in reversed(range(T)):
            s, a, r, s2, fin = tr[t]
            v2 = 0.0 if fin else Vhat[s2]
            delta = r + GAM * v2 - Vhat[s]
            adv = delta + (0.0 if fin else GAM * lam * adv)
            advs[t] = adv
        for t in range(T):
            a = tr[t][1]
            glog = (1 - pd) if a == 1 else -pd
            g += GAM ** t * glog * advs[t]
        gs.append(g)
    gs = np.array(gs)
    sesgo = gs.mean() - g_true
    res_gae.append((lam, sesgo, gs.std()))
    print(f"  λ = {lam:4.2f}: media {gs.mean():.4f} (sesgo {sesgo:+.4f}, {abs(sesgo)/abs(g_true)*100:5.1f} %); desviación por episodio {gs.std():.4f}; "
          f"error cuadrático con 64 episodios {math.sqrt(sesgo**2 + gs.var()/64):.4f}")

# ====================================================================== 3. MOLDEADO DE LA RECOMPENSA
print("=" * 70)
print("3. MOLDEADO: REJILLA 9x9, META EN (8,8), γ = 0,95, ITERACIÓN DE VALOR")
N, G3 = 9, 0.95
meta = (8, 8)
acciones = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def resolver(recomp, terminal_en_meta=True):
    Vv = np.zeros((N, N))
    for _ in range(2000):
        Vn = np.zeros((N, N))
        for x in range(N):
            for y in range(N):
                if terminal_en_meta and (x, y) == meta:
                    continue
                mejores = []
                for dx, dy in acciones:
                    x2, y2 = min(N - 1, max(0, x + dx)), min(N - 1, max(0, y + dy))
                    cont = 0.0 if (terminal_en_meta and (x2, y2) == meta) else Vv[x2, y2]
                    mejores.append(recomp((x, y), (x2, y2)) + G3 * cont)
                Vn[x, y] = max(mejores)
        if np.max(np.abs(Vn - Vv)) < 1e-10:
            break
        Vv = Vn
    return Vv


def politica_llega(recomp, inicio=(0, 0), max_pasos=200):
    Vv = resolver(recomp)
    s, pasos = inicio, 0
    visit = []
    while s != meta and pasos < max_pasos:
        x, y = s
        cands = []
        for dx, dy in acciones:
            x2, y2 = min(N - 1, max(0, x + dx)), min(N - 1, max(0, y + dy))
            cont = 0.0 if (x2, y2) == meta else Vv[x2, y2]
            cands.append((recomp(s, (x2, y2)) + G3 * cont, (x2, y2)))
        s = max(cands)[1]
        visit.append(s)
        pasos += 1
    return s == meta, pasos, visit[-4:]


def dist(s):
    return abs(s[0] - meta[0]) + abs(s[1] - meta[1])


escasa = lambda s, s2: 10.0 if s2 == meta else 0.0
phi = lambda s: -dist(s)
potencial = lambda s, s2: escasa(s, s2) + (G3 * (0.0 if s2 == meta else phi(s2)) - phi(s))
ingenua = lambda s, s2: escasa(s, s2) + (1.0 if dist(s2) == 1 else 0.0)        # premio por «estar cerca»
for nombre, rf in (("escasa (10 al llegar)", escasa), ("con potencial Φ = -distancia", potencial), ("con +1 por estar a 1 casilla", ingenua)):
    llega, pasos, ult = politica_llega(rf)
    print(f"  {nombre:32s}: ¿llega a la meta? {llega}; pasos {pasos}; últimas casillas {ult}")

# ====================================================================== 4. PPO CON EL x500
print("=" * 70)
print("4. PPO EN JAX: x500 CON EMPUJE COLECTIVO Y VELOCIDADES ANGULARES (CTBR), 100 Hz")
import jax
import jax.numpy as jnp
from jax import lax
import optax

print("  dispositivo:", jax.devices()[0], "| jax", jax.__version__, "| optax", optax.__version__)
M, G = 2.0, 9.81
J = jnp.array([0.022, 0.022, 0.040])
DT, SUB = 0.002, 5                # física a 500 Hz, control a 100 Hz
TAU = 0.03                        # retardo de primer orden de empuje y par
K_OM = 20.0                       # lazo interno de velocidad angular (P)
EP_LEN = 500                      # 5 s


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


def fisica(s, a):
    """s = [p(3) v(3) q(4) om(3) T(1) tau(3) a_prev(4) t(1) pref(3)]; a en [-1, 1]^4."""
    p, v, q, om, T, tau = s[0:3], s[3:6], s[6:10], s[10:13], s[13], s[14:17]
    T_cmd = M * G * (1.0 + 0.8 * a[0])
    om_d = jnp.array([4.0, 4.0, 2.0]) * a[1:4]
    tau_cmd = J * K_OM * (om_d - om) + jnp.cross(om, J * om)
    def sub(i, x):
        p, v, q, om, T, tau = x
        T = T + DT / TAU * (T_cmd - T)
        tau = tau + DT / TAU * (tau_cmd - tau)
        acc = rotmat(q) @ jnp.array([0.0, 0.0, T / M]) - jnp.array([0.0, 0.0, G]) - 0.15 / M * v
        v = v + DT * acc
        p = p + DT * v
        om = om + DT * (tau - jnp.cross(om, J * om)) / J
        q = q + 0.5 * DT * qmul(q, jnp.concatenate([jnp.zeros(1), om]))
        q = q / jnp.linalg.norm(q)
        return (p, v, q, om, T, tau)
    p, v, q, om, T, tau = lax.fori_loop(0, SUB, sub, (p, v, q, om, T, tau))
    return jnp.concatenate([p, v, q, om, jnp.array([T]), tau])


def estado_inicial(key):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    p = jax.random.uniform(k1, (3,), minval=-1.5, maxval=1.5)
    v = jax.random.uniform(k2, (3,), minval=-1.0, maxval=1.0)
    eje = jax.random.normal(k3, (3,)); eje = eje / jnp.linalg.norm(eje)
    ang = jax.random.uniform(k4, (), minval=0.0, maxval=math.radians(30))
    q = jnp.concatenate([jnp.array([jnp.cos(ang / 2)]), jnp.sin(ang / 2) * eje])
    om = jax.random.uniform(k5, (3,), minval=-1.0, maxval=1.0)
    return jnp.concatenate([p, v, q, om, jnp.array([M * G]), jnp.zeros(3), jnp.zeros(4), jnp.zeros(1), jnp.zeros(3)])


def observar(s):
    p, v, q, om, aprev, pref = s[0:3], s[3:6], s[6:10], s[10:13], s[17:21], s[22:25]
    return jnp.concatenate([p - pref, v, rotmat(q).reshape(-1), om, aprev])     # 22


RECOMPENSA = {"tipo": "acotada"}


def paso(s, a, key, tipo="acotada"):
    a = jnp.clip(a, -1.0, 1.0)
    fis = fisica(s[0:17], a)
    t = s[21] + 1
    s2 = jnp.concatenate([fis, a, jnp.array([t]), s[22:25]])
    e = s2[0:3] - s2[22:25]
    inclin = jnp.arccos(jnp.clip(rotmat(s2[6:10])[2, 2], -1, 1))
    coste = jnp.linalg.norm(e) + 0.1 * jnp.linalg.norm(s2[3:6]) + 0.05 * jnp.linalg.norm(s2[10:13]) \
        + 0.1 * jnp.sum((a - s[17:21]) ** 2)
    choque = (jnp.linalg.norm(e) > 4.0) | (inclin > math.radians(80)) | ~jnp.all(jnp.isfinite(s2))
    if tipo == "acotada":
        r = 1.0 - jnp.tanh(coste) - 10.0 * choque      # en [0, 1] mientras vuela: estrellarse siempre sale caro
    else:
        r = 1.0 - coste - 10.0 * choque                 # «ingenua»: negativa por paso si el coste pasa de 1
    tiempo = t >= EP_LEN
    hecho = choque | tiempo
    s_obs_final = observar(s2)                   # observación antes de reiniciar (para el corte por tiempo)
    s_new = jnp.where(hecho, estado_inicial(key), s2)
    return s_new, r, choque, tiempo, s_obs_final


def mlp_init(key, tamanos, escala_final):
    ps = []
    ks = jax.random.split(key, len(tamanos) - 1)
    for i, (a, b) in enumerate(zip(tamanos[:-1], tamanos[1:])):
        g = escala_final if i == len(tamanos) - 2 else math.sqrt(2)
        W = jax.nn.initializers.orthogonal(g)(ks[i], (a, b))
        ps.append((W, jnp.zeros(b)))
    return ps


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


OBS, ACT = 22, 4


def init_red(key):
    k1, k2 = jax.random.split(key)
    return {"pi": mlp_init(k1, [OBS, 64, 64, ACT], 0.01), "v": mlp_init(k2, [OBS, 64, 64, 1], 1.0),
            "logstd": jnp.full(ACT, -0.5)}


def logprob(media, logstd, a):
    return jnp.sum(-0.5 * ((a - media) / jnp.exp(logstd)) ** 2 - logstd - 0.5 * math.log(2 * math.pi), axis=-1)


CFG = dict(n_env=4096, n_pasos=32, epocas=5, n_mini=4, gamma=0.99, lam=0.95, clip=0.2, lr=3e-4,
           c_v=0.5, c_ent=0.0, max_grad=0.5, n_upd=600)


def entrenar(cfg, seed=0, registrar=True, tipo="acotada"):
    key = jax.random.PRNGKey(seed)
    key, kr, ke = jax.random.split(key, 3)
    red = init_red(kr)
    opt = optax.chain(optax.clip_by_global_norm(cfg["max_grad"]), optax.adam(cfg["lr"], eps=1e-5))
    opt_st = opt.init(red)
    S = jax.vmap(estado_inicial)(jax.random.split(ke, cfg["n_env"]))
    # escalonar el tiempo inicial de los episodios para no sincronizar los reinicios (B6)
    key, kt = jax.random.split(key)
    S = S.at[:, 21].set(jax.random.randint(kt, (cfg["n_env"],), 0, EP_LEN).astype(jnp.float32))

    def rollout(carry, _):
        red, S, key = carry
        key, ka, kr2 = jax.random.split(key, 3)
        O = jax.vmap(observar)(S)
        media = mlp(red["pi"], O)
        a = media + jnp.exp(red["logstd"]) * jax.random.normal(ka, media.shape)
        lp = logprob(media, red["logstd"], a)
        v = mlp(red["v"], O)[:, 0]
        S2, r, choque, tiempo, Ofin = jax.vmap(lambda s, a_, k: paso(s, a_, k, tipo))(S, a, jax.random.split(kr2, cfg["n_env"]))
        vfin = mlp(red["v"], Ofin)[:, 0]
        # corte por tiempo: no es un final real; se arranca el valor del estado final (Rudin et al.)
        r_aj = r + cfg["gamma"] * vfin * (tiempo & ~choque)
        hecho = choque | tiempo
        return (red, S2, key), (O, a, lp, v, r_aj, hecho, r, choque)

    def actualizar(estado, _):
        red, opt_st, S, key = estado
        (red, S, key), (O, A, LP, V, R, D, Rcrudo, CH) = lax.scan(rollout, (red, S, key), None, length=cfg["n_pasos"])
        v_ult = mlp(red["v"], jax.vmap(observar)(S))[:, 0]
        def gae(carry, x):
            ventaja, v_sig = carry
            r, v, d = x
            delta = r + cfg["gamma"] * v_sig * (1 - d) - v
            ventaja = delta + cfg["gamma"] * cfg["lam"] * (1 - d) * ventaja
            return (ventaja, v), ventaja
        _, ADV = lax.scan(gae, (jnp.zeros_like(v_ult), v_ult), (R, V, D.astype(jnp.float32)), reverse=True)
        RET = ADV + V
        B = cfg["n_env"] * cfg["n_pasos"]
        datos = tuple(x.reshape((B,) + x.shape[2:]) for x in (O, A, LP, ADV, RET))

        def perdida(red, O, A, LP, ADV, RET):
            media = mlp(red["pi"], O)
            lp = logprob(media, red["logstd"], A)
            ratio = jnp.exp(lp - LP)
            adv = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            l_pi = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * adv))
            v = mlp(red["v"], O)[:, 0]
            l_v = 0.5 * jnp.mean((v - RET) ** 2)
            ent = jnp.sum(red["logstd"] + 0.5 * math.log(2 * math.pi * math.e))
            kl = jnp.mean((ratio - 1) - (lp - LP))
            return l_pi + cfg["c_v"] * l_v - cfg["c_ent"] * ent, (l_pi, l_v, kl, jnp.mean(jnp.abs(ratio - 1) > cfg["clip"]))

        def epoca(c, _):
            red, opt_st, key = c
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, B)
            mini = tuple(x[perm].reshape((cfg["n_mini"], B // cfg["n_mini"]) + x.shape[1:]) for x in datos)
            def paso_mini(c2, mb):
                red, opt_st = c2
                (l, aux), g = jax.value_and_grad(perdida, has_aux=True)(red, *mb)
                upd, opt_st = opt.update(g, opt_st, red)
                red = optax.apply_updates(red, upd)
                return (red, opt_st), aux
            (red, opt_st), aux = lax.scan(paso_mini, (red, opt_st), mini)
            return (red, opt_st, key), aux
        (red, opt_st, key), aux = lax.scan(epoca, (red, opt_st, key), None, length=cfg["epocas"])
        met = dict(r=Rcrudo.mean(), choques=CH.mean(), kl=aux[2].mean(), fr_clip=aux[3].mean(), l_v=aux[1].mean(),
                   std=jnp.exp(red["logstd"]).mean())
        return (red, opt_st, S, key), met

    actualizar_j = jax.jit(actualizar)
    estado = (red, opt_st, S, key)
    estado, _ = actualizar_j(estado, None)          # compila
    jax.block_until_ready(estado)
    hist = []
    t0 = time.perf_counter()
    for i in range(cfg["n_upd"]):
        estado, met = actualizar_j(estado, None)
        if i % 10 == 0 or i == cfg["n_upd"] - 1:
            met = {k: float(v) for k, v in met.items()}
            dt = time.perf_counter() - t0
            pasos = (i + 1) * cfg["n_env"] * cfg["n_pasos"]
            hist.append((pasos, dt, met["r"], met["choques"], met["kl"], met["fr_clip"], met["std"]))
            if registrar and (i % 50 == 0 or i == cfg["n_upd"] - 1):
                print(f"  act. {i:4d}: {pasos/1e6:6.1f} M pasos, {dt:6.1f} s, r media/paso {met['r']:6.3f}, "
                      f"choques/paso {met['choques']:.4f}, KL {met['kl']:.4f}, fracción recortada {met['fr_clip']:.3f}, σ {met['std']:.3f}")
    jax.block_until_ready(estado)
    t_total = time.perf_counter() - t0
    return estado[0], hist, t_total


print("  --- recompensa acotada 1 - tanh(coste) - 10·choque, semilla 0 (curva de la figura):")
red, hist, t_total = entrenar(CFG)
pasos_tot = CFG["n_env"] * CFG["n_pasos"] * CFG["n_upd"]
print(f"  r media/paso en el último 10 % (4096 x 32): {np.mean([x[2] for x in hist[-6:]]):.3f}")
print(f"  total: {pasos_tot/1e6:.1f} M pasos en {t_total:.1f} s ({pasos_tot/t_total/1e6:.2f} M pasos/s con el aprendizaje incluido)")
red2, hist2, _ = entrenar(CFG, registrar=False)
print(f"  misma semilla otra vez: r final {hist2[-1][2]:.4f} frente a {hist[-1][2]:.4f}; σ final {hist2[-1][6]:.4f} frente a {hist[-1][6]:.4f}"
      " (dentro del mismo proceso, idéntico; entre procesos en GPU puede cambiar)")

# --- evaluación determinista: estacionario desde 200 estados iniciales y seguimiento de un círculo
@jax.jit
def evaluar(red, key, circulo):
    S = jax.vmap(estado_inicial)(jax.random.split(key, 200))
    def f(S, k):
        t = k * 0.01
        pref = jnp.where(circulo, jnp.array([jnp.cos(2 * math.pi * t / 6.0) - 1.0, jnp.sin(2 * math.pi * t / 6.0), 0.0]), jnp.zeros(3))
        S = S.at[:, 22:25].set(pref)
        S = S.at[:, 21].set(0.0)                 # que no corte por tiempo
        O = jax.vmap(observar)(S)
        a = mlp(red["pi"], O)
        S2, r, ch, ti, _ = jax.vmap(paso)(S, a, jax.random.split(jax.random.PRNGKey(0), 200))
        e = jnp.linalg.norm(S[:, 0:3] - pref, axis=1)
        return S2, (e, ch)
    _, (E, CH) = lax.scan(f, S, jnp.arange(1000))
    return E, CH


print("  --- 5 semillas por recompensa: choques por paso al final del entrenamiento y evaluación en estacionario")
for tipo in ("ingenua", "acotada"):
    filas = []
    for sd in range(5):
        rd, h, _ = entrenar(CFG, seed=sd, registrar=False, tipo=tipo)
        Es, CHs = evaluar(rd, jax.random.PRNGKey(123), False)
        Es = np.asarray(Es); CHs = np.asarray(CHs)
        filas.append((h[-1][3], CHs.any(0).mean(), Es[500:].mean()))
        print(f"  {tipo:8s} semilla {sd}: choques/paso al final {h[-1][3]:.4f}; evaluación: choques {CHs.any(0).mean()*100:5.1f} %, error medio últimos 5 s {Es[500:].mean()*100:7.1f} cm")
    n_bien = sum(1 for f in filas if f[1] == 0 and f[2] < 0.2)
    print(f"  {tipo}: {n_bien} de 5 semillas vuelan sin choques con error < 20 cm")
E, CH = evaluar(red, jax.random.PRNGKey(123), False)
E = np.asarray(E); CH = np.asarray(CH)
print(f"  estacionario (200 arranques aleatorios, 10 s): choques {CH.any(0).mean()*100:.1f} %; error medio a 1 s {E[100].mean()*100:.1f} cm, "
      f"a 3 s {E[300].mean()*100:.1f} cm, últimos 5 s {E[500:].mean()*100:.2f} cm (mediana de RMS {np.median(np.sqrt((E[500:]**2).mean(0)))*100:.2f} cm)")
Ec, CHc = evaluar(red, jax.random.PRNGKey(7), True)
Ec = np.asarray(Ec)
print(f"  círculo de 1 m de radio en 6 s (≈ 1,05 m/s), sin haberlo entrenado: error medio en los últimos 5 s {Ec[500:].mean()*100:.1f} cm")
X4 = lin(0, pasos_tot, 60, 520)
rs = [h[2] for h in hist]
Y4 = lin(0.0, 1.0, 250, 30)
print("path curva:", path([h[0] for h in hist], suavizar(rs), X4, Y4))
print(f"  ejes: r de {min(rs):.3f} a {max(rs):.3f}; y(0) = {Y4(0):.1f}; y(0,5) = {Y4(0.5):.1f}; y(1) = {Y4(1):.1f}; y(min) = {Y4(min(rs)):.1f}")
print("  ticks x (M pasos): " + " ".join(f"{m}->{X4(m*1e6):.0f}" for m in (0, 10, 20, 30, 40)))
print("  tiempo de reloj en cada décima parte: " + " ".join(f"{h[0]/1e6:.0f}M@{h[1]:.0f}s" for h in hist[::5]))
EM = E.mean(1)
X5 = lin(0, 3.0, 60, 520); Y5 = lin(0, max(1.5, float(EM.max())), 250, 30)
print("path error:", path(np.arange(301) * 0.01, EM[:301], X5, Y5))
print(f"  y(1 m) = {Y5(1.0):.1f}; y(0,5 m) = {Y5(0.5):.1f}; ymax = {max(1.5, float(EM.max())):.2f}")

# --- sensibilidad: menos entornos con el mismo número de pasos (y el mismo lote por actualización)
print("  --- mismo presupuesto de pasos con otro reparto entornos x pasos por actualización:")
hist_rep = []
for n_env, n_pasos in ((256, 512), (1024, 128), (16384, 8)):
    cfg = dict(CFG, n_env=n_env, n_pasos=n_pasos)
    _, h, tt = entrenar(cfg, registrar=False)
    hist_rep.append((n_env, n_pasos, h))
    print(f"  {n_env:6d} entornos x {n_pasos:4d} pasos: r media/paso en el último 10 % {np.mean([x[2] for x in h[-6:]]):.3f}, choques/paso {h[-1][3]:.5f}, tiempo {tt:.0f} s")
for n_env, n_pasos, h in hist_rep:
    print(f"path rep {n_env}:", path([x[0] for x in h], suavizar([x[2] for x in h]), X4, Y4))
