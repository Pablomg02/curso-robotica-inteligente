"""Datos de las figuras y tablas del tema A10 (imitation learning y políticas generativas).

Uso:  uv run --with "jax[cuda13]" --with optax python -u figuras_a10.py > salida_gpu.txt     (GPU; unos minutos)
      Con A10_RAPIDO=1, una prueba corta.

El x500 de A8a (física a 500 Hz, control a 50 Hz) tiene que ir de x = -2,5 m a x = +2,5 m con un poste vertical de
0,6 m de radio en medio. El experto es un MPPI con la física y el estado verdadero (A3b, A8b) y un coste que castiga
acercarse al poste: todos los vuelos empiezan casi centrados (±2 cm), así que es el ruido del propio MPPI el que decide
si lo rodea por la izquierda o por la derecha, y las demostraciones son bimodales sin que el estado lo explique
(con A10_Y0=0.15, la desviación inicial decide el lado y la acción dado el estado es casi unimodal). Alumnos, todos con el estado verdadero como observación:
  BC                       red que predice la acción con error cuadrático
  BC con trozos            predice 8 acciones y ejecuta 4 (trozos de acciones, como ACT)
  flow matching            política generativa de trozos de 8 acciones (flujo condicional lineal, 10 pasos de Euler)
  DAgger                   BC con iteraciones en las que el experto etiqueta los estados del alumno
  flow matching + DAgger   lo mismo con la política generativa
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
RAPIDO = bool(os.environ.get("A10_RAPIDO"))
G, M0 = 9.81, 2.0
J = jnp.array([0.022, 0.022, 0.040])
DT, SUB = 0.002, 10
TAU_M = 0.03
T_EP = 300                       # 6 s
META = jnp.array([2.5, 0.0, 0.0])
R_POSTE = 0.6


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


def paso_fisica(x, a):
    p, v, q, om, Tr, tau = x
    T_cmd = M0 * G * (1.0 + 0.8 * a[0])
    om_d = jnp.array([4.0, 4.0, 2.0]) * a[1:4]
    tau_cmd = J * 20.0 * (om_d - om) + jnp.cross(om, J * om)
    def sub(i, x):
        p, v, q, om, Tr, tau = x
        Tr = Tr + DT / TAU_M * (T_cmd - Tr)
        tau = tau + DT / TAU_M * (tau_cmd - tau)
        acc = rotmat(q) @ jnp.array([0.0, 0.0, Tr / M0]) - jnp.array([0.0, 0.0, G]) - 0.15 / M0 * v
        v = v + DT * acc
        p = p + DT * v
        om = om + DT * (tau - jnp.cross(om, J * om)) / J
        q = q + 0.5 * DT * qmul(q, jnp.concatenate([jnp.zeros(1), om]))
        return (p, v, q / jnp.linalg.norm(q), om, Tr, tau)
    return lax.fori_loop(0, SUB, sub, (p, v, q, om, Tr, tau))


Y0 = float(os.environ.get("A10_Y0", "0.02"))   # desviación lateral inicial máxima: pequeña, para que el estado no decida el lado


def x_inicial(key):
    y0 = jax.random.uniform(key, (), minval=-Y0, maxval=Y0)
    return (jnp.array([-2.5, y0, 0.0]), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), jnp.asarray(M0 * G), jnp.zeros(3))


def observacion(x):
    p, v, q, om, Tr, tau = x
    return jnp.concatenate([p - META, v, rotmat(q).reshape(-1), om])                 # 18


# ---------------------------------------------------------------------- experto: MPPI con la física (A8b)
HP, NS, NE = 40, 256, 32
SIGS = (0.3, 0.15, 0.08)
V_MAX = 2.0


def ruido_suave(key, rho=0.9):
    n = jax.random.normal(key, (HP, NS, 4))
    def f(e, x):
        e = rho * e + math.sqrt(1 - rho ** 2) * x
        return e, e
    return jnp.swapaxes(jnp.concatenate([n[:1], lax.scan(f, n[0], n[1:])[1]], 0), 0, 1)


def imaginar(x, A):
    def f(x, a):
        x = paso_fisica(x, a)
        return x, (x[0], x[1])
    return lax.scan(f, x, A)[1]


def coste(P, Vv, A, a_prev):
    d = jnp.linalg.norm(P[:, :2], axis=-1)
    dA = jnp.diff(jnp.concatenate([a_prev[None], A], 0), axis=0)
    return (jnp.sum((P - META) ** 2) + 100.0 * jnp.sum(jax.nn.relu(R_POSTE + 0.35 - d) ** 2)
            + 2.0 * jnp.sum(jax.nn.relu(jnp.linalg.norm(Vv, axis=-1) - V_MAX) ** 2)
            + 0.02 * jnp.sum(A ** 2) + 0.1 * jnp.sum(dA ** 2))


def planificar(x, U, a_prev, key):
    for s in SIGS:
        key, k = jax.random.split(key)
        D = jnp.clip(U[None] + s * ruido_suave(k), -1.0, 1.0).at[0].set(U)
        P, Vv = jax.vmap(lambda d: imaginar(x, d))(D)
        c = jax.vmap(coste, (0, 0, 0, None))(P, Vv, D, a_prev)
        c = jnp.where(jnp.isfinite(c), c, 1e9)
        nc, ie = lax.top_k(-c, NE)
        ce = -nc
        w = jax.nn.softmax(-2.0 * (ce - ce[0]) / (jnp.std(ce) + 1e-6))
        U = jnp.sum(w[:, None, None] * D[ie], 0)
    return U


K_TROZO, K_EJEC = 8, 4


def episodio_experto(key):
    """Devuelve observaciones, acciones ejecutadas, trozos planificados (las 8 primeras acciones del plan) y posiciones."""
    k0, k1 = jax.random.split(key)
    def f(c, k):
        x, U, a_prev = c
        U = planificar(x, U, a_prev, k)
        a = U[0]
        o = observacion(x)
        x2 = paso_fisica(x, a)
        return (x2, jnp.concatenate([U[1:], U[-1:]], 0), a), (o, a, U[:K_TROZO], x[0])
    _, (O, A, CH, P) = lax.scan(f, (x_inicial(k0), jnp.zeros((HP, 4)), jnp.zeros(4)), jax.random.split(k1, T_EP))
    return O, A, CH, P


def resumen_tray(P):
    """Éxito (a menos de 0,3 m de la meta al final, sin tocar el poste), choque y lado por el que rodea el poste."""
    P = np.asarray(P)
    d = np.linalg.norm(P[:, :, :2], axis=-1)
    choque = (d < R_POSTE).any(1) | ~np.isfinite(P).all(axis=(1, 2))
    final = np.linalg.norm(P[:, -1] - np.asarray(META), axis=-1)
    exito = (final < 0.3) & ~choque
    i = np.argmin(np.where(np.isfinite(d), np.abs(P[:, :, 0]), 1e9), axis=1)
    lado = np.sign(P[np.arange(len(P)), i, 1])
    return exito, choque, final, lado


print("=" * 70)
print("1. EXPERTO: MPPI CON LA FÍSICA Y EL ESTADO VERDADERO (256 SECUENCIAS DE 0,8 s, 3 ITERACIONES)")
N_DEMO = 60 if RAPIDO else 300
t0 = time.perf_counter()
gen_exp = jax.jit(jax.vmap(episodio_experto))
partes = [gen_exp(jax.random.split(jax.random.PRNGKey(i), 100)) for i in range(max(N_DEMO // 100, 1))] if not RAPIDO else \
         [gen_exp(jax.random.split(jax.random.PRNGKey(0), N_DEMO))]
O_d = jnp.concatenate([q[0] for q in partes]); A_d = jnp.concatenate([q[1] for q in partes])
CH_d = jnp.concatenate([q[2] for q in partes]); P_d = jnp.concatenate([q[3] for q in partes])
ex, ch, fi, la = resumen_tray(P_d)
print(f"  {len(ex)} demostraciones en {time.perf_counter()-t0:.0f} s: éxito {ex.mean()*100:.0f} %, choques {ch.mean()*100:.0f} %, "
      f"por la izquierda (y > 0) {np.mean(la > 0)*100:.1f} %, por la derecha {np.mean(la < 0)*100:.1f} %")
ok = ex                                                       # solo se imitan las demostraciones buenas
O_d, A_d, CH_d = O_d[ok], A_d[ok], CH_d[ok]
mu_o, sd_o = O_d.reshape(-1, 18).mean(0), O_d.reshape(-1, 18).std(0) + 1e-3
# trozos ejecutados: a_t ... a_{t+7} de la propia demostración (rellenando al final con la última acción)
A_pad = jnp.concatenate([A_d, jnp.repeat(A_d[:, -1:], K_TROZO, 1)], 1)
CHE_d = jnp.stack([A_pad[:, j:j + T_EP] for j in range(K_TROZO)], 2)          # (N, T, 8, 4)
print(f"  datos de imitación: {O_d.shape[0]} demostraciones × {T_EP} pasos = {O_d.shape[0] * T_EP} pares")
mu_a, sd_a = A_d.reshape(-1, 4).mean(0), A_d.reshape(-1, 4).std(0) + 1e-3      # normalización de las acciones (flujo)
print(f"  desviación típica de las acciones del experto: {', '.join(f'{v:.3f}' for v in np.asarray(sd_a))}")


# ---------------------------------------------------------------------- redes
def mlp_init(key, sizes):
    ks = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (a, b)) * math.sqrt(2.0 / a) * (1.0 if i < len(sizes) - 2 else 0.1), jnp.zeros(b))
            for i, (k, a, b) in enumerate(zip(ks, sizes[:-1], sizes[1:]))]


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jax.nn.gelu(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def norm_o(o):
    return (o - mu_o) / sd_o


FREQ = jnp.array([1.0, 2.0, 4.0, 8.0, 16.0])


def emb_t(tau):
    return jnp.concatenate([jnp.sin(math.pi * FREQ * tau), jnp.cos(math.pi * FREQ * tau)], -1)


def flujo_v(p, x, tau, o):
    return mlp(p, jnp.concatenate([norm_o(o), x, emb_t(tau)], -1))


N_EULER = 10


def flujo_muestra(p, o, key):
    x = jax.random.normal(key, (K_TROZO * 4,))
    def f(x, i):
        tau = i / N_EULER
        return x + flujo_v(p, x, tau[None], o) / N_EULER, None
    x, _ = lax.scan(f, x, jnp.arange(N_EULER, dtype=jnp.float32))
    return jnp.clip(x.reshape(K_TROZO, 4) * sd_a + mu_a, -1.0, 1.0)


def perdida_bc(p, o, a):
    return jnp.mean(jnp.sum((jnp.tanh(mlp(p, norm_o(o))) - a.reshape(a.shape[0], -1)) ** 2, -1))


def perdida_flujo(p, o, a, key):
    k1, k2 = jax.random.split(key)
    x1 = ((a - mu_a) / sd_a).reshape(a.shape[0], -1)
    x0 = jax.random.normal(k1, x1.shape)
    tau = jax.random.uniform(k2, (x1.shape[0], 1))
    xt = (1 - tau) * x0 + tau * x1
    return jnp.mean(jnp.sum((flujo_v(p, xt, tau, o) - (x1 - x0)) ** 2, -1))


def entrenar(tipo, O, Y, pasos, B=256, seed=0):
    """tipo: 'bc' (Y: (n, 4)), 'bc_trozos' (Y: (n, 8, 4)) o 'flujo' (Y: (n, 8, 4))."""
    n_out = 4 if tipo == "bc" else K_TROZO * 4
    n_in = 18 + (K_TROZO * 4 + 10 if tipo == "flujo" else 0)
    p = mlp_init(jax.random.PRNGKey(seed), [n_in, 256, 256, 256, n_out])
    opt = optax.adam(optax.cosine_decay_schedule(1e-3, pasos))
    st = opt.init(p)
    @jax.jit
    def bucle(p, st, keys, O, Y):
        def f(c, k):
            p, st = c
            k1, k2 = jax.random.split(k)
            i = jax.random.randint(k1, (B,), 0, O.shape[0])
            if tipo == "flujo":
                l, g = jax.value_and_grad(perdida_flujo)(p, O[i], Y[i], k2)
            else:
                l, g = jax.value_and_grad(perdida_bc)(p, O[i], Y[i])
            u, st = opt.update(g, st, p)
            return (optax.apply_updates(p, u), st), l
        return lax.scan(f, (p, st), keys)
    (p, st), L = bucle(p, st, jax.random.split(jax.random.PRNGKey(seed + 100), pasos), O, Y)
    return p, float(L[-200:].mean())


# ---------------------------------------------------------------------- alumnos en bucle cerrado
def politica(tipo, p):
    """Devuelve (paso, estado inicial): paso(estado_alumno, o, k) -> (a, estado_alumno)."""
    if tipo == "bc":
        return (lambda s, o, k: (jnp.tanh(mlp(p, norm_o(o))), s)), jnp.zeros((K_TROZO, 4))
    def paso(s, o, k, t):
        buf = s
        nuevo = jnp.tanh(mlp(p, norm_o(o))).reshape(K_TROZO, 4) if tipo == "bc_trozos" else flujo_muestra(p, o, k)
        buf = jnp.where(t % K_EJEC == 0, nuevo, buf)
        return buf[t % K_EJEC], buf
    return paso, jnp.zeros((K_TROZO, 4))


def episodio_alumno(tipo, p, key, etiquetar=False):
    """Despliega el alumno. Con etiquetar, el experto corre a la sombra (MPPI con su plan caliente desde el estado del
    alumno) y devuelve sus etiquetas: la acción y el trozo de 8 acciones de su plan (DAgger)."""
    paso, s0 = politica(tipo, p)
    k0, k1 = jax.random.split(key)
    def f(c, inp):
        x, s, U, a_prev = c
        t, k = inp
        ka, ke = jax.random.split(k)
        o = observacion(x)
        if tipo == "bc":
            a, s = paso(s, o, ka)
        else:
            a, s = paso(s, o, ka, t)
        if etiquetar:
            U = planificar(x, U, a_prev, ke)
        x2 = paso_fisica(x, a)
        return (x2, s, jnp.concatenate([U[1:], U[-1:]], 0), a), (o, x[0], U[0], U[:K_TROZO], a)
    _, (O, P, AE, CHE, A) = lax.scan(f, (x_inicial(k0), s0, jnp.zeros((HP, 4)), jnp.zeros(4)),
                                     (jnp.arange(T_EP), jax.random.split(k1, T_EP)))
    return O, P, AE, CHE, A


N_EV = 50 if RAPIDO else 200
claves_ev = jax.random.split(jax.random.PRNGKey(4242), N_EV)
PASOS_BC = 2000 if RAPIDO else 20000
RES, TRAY = {}, {}


def evaluar(nombre, tipo, p):
    t0 = time.perf_counter()
    O, P, _, _, A = jax.jit(jax.vmap(lambda k: episodio_alumno(tipo, p, k)))(claves_ev)
    ex, ch, fi, la = resumen_tray(P)
    act = float(np.nanmean(np.abs(np.diff(np.asarray(A), axis=1)).sum(-1)))
    RES[nombre] = (ex.mean(), ch.mean(), np.median(fi), np.mean(la > 0), np.mean(la < 0), act)
    TRAY[nombre] = np.asarray(P)
    print(f"  {nombre:24s}: éxito {ex.mean()*100:5.1f} % | choques {ch.mean()*100:5.1f} % | distancia final mediana {np.median(fi):5.2f} m | "
          f"izquierda {np.mean(la > 0)*100:5.1f} % / derecha {np.mean(la < 0)*100:5.1f} % | actividad |Δa| {act:.3f}  ({time.perf_counter()-t0:.0f} s)", flush=True)


print("=" * 70)
print(f"2. ALUMNOS ENTRENADOS CON LAS DEMOSTRACIONES ({N_EV} EPISODIOS DE PRUEBA, CON OTRAS DESVIACIONES INICIALES)")
On = O_d.reshape(-1, 18)
Oe, Ae, _, Pe = gen_exp(claves_ev)
ex, ch, fi, la = resumen_tray(Pe)
RES["experto (MPPI)"] = (ex.mean(), ch.mean(), np.median(fi), np.mean(la > 0), np.mean(la < 0),
                         float(np.abs(np.diff(np.asarray(Ae), axis=1)).sum(-1).mean()))
TRAY["experto (MPPI)"] = np.asarray(Pe)
print(f"  {'experto (MPPI)':24s}: éxito {ex.mean()*100:5.1f} % | choques {ch.mean()*100:5.1f} % | distancia final mediana {np.median(fi):5.2f} m | "
      f"izquierda {np.mean(la > 0)*100:5.1f} % / derecha {np.mean(la < 0)*100:5.1f} % | actividad |Δa| {RES['experto (MPPI)'][5]:.3f}")
p_bc, l1 = entrenar("bc", On, A_d.reshape(-1, 4), PASOS_BC)
p_bct, l2 = entrenar("bc_trozos", On, CHE_d.reshape(-1, K_TROZO, 4), PASOS_BC)
p_fm, l3 = entrenar("flujo", On, CHE_d.reshape(-1, K_TROZO, 4), PASOS_BC * 2)
print(f"  pérdidas finales de entrenamiento: BC {l1:.4f}, BC con trozos {l2:.4f}, flow matching {l3:.4f}")
evaluar("BC", "bc", p_bc)
evaluar("BC con trozos", "bc_trozos", p_bct)
evaluar("flow matching (trozos)", "flujo", p_fm)

# ---------------------------------------------------------------------- 3. DAgger
print("=" * 70)
N_IT, N_ROLL = (2, 30) if RAPIDO else (4, 100)
print(f"3. DAGGER: {N_IT} ITERACIONES; EN CADA UNA, {N_ROLL} DESPLIEGUES DEL ALUMNO ETIQUETADOS POR EL EXPERTO A LA SOMBRA")
for tipo, nombre in (("bc", "DAgger (BC)"), ("flujo", "flow matching + DAgger")):
    O_ag = On
    Y_ag = A_d.reshape(-1, 4) if tipo == "bc" else CHE_d.reshape(-1, K_TROZO, 4)
    p = p_bc if tipo == "bc" else p_fm
    for it in range(N_IT):
        claves = jax.random.split(jax.random.PRNGKey(777 + 10 * it + (0 if tipo == "bc" else 5000)), N_ROLL)
        O, P, AE, CHE, A = jax.jit(jax.vmap(lambda k: episodio_alumno(tipo, p, k, etiquetar=True)))(claves)
        ok = np.isfinite(np.asarray(O)).all(axis=(1, 2))
        O_ag = jnp.concatenate([O_ag, O[ok].reshape(-1, 18)])
        Y_ag = jnp.concatenate([Y_ag, (AE[ok].reshape(-1, 4) if tipo == "bc" else CHE[ok].reshape(-1, K_TROZO, 4))])
        p, _ = entrenar(tipo, O_ag, Y_ag, PASOS_BC if tipo == "bc" else PASOS_BC * 2, seed=it + 1)
        ex, ch, fi, la = resumen_tray(P)
        print(f"    iteración {it + 1}: los despliegues del alumno (antes de reentrenar) tuvieron éxito {ex.mean()*100:.0f} %, "
              f"choques {ch.mean()*100:.0f} %; datos {O_ag.shape[0]} pares", flush=True)
    evaluar(nombre, tipo, p)
    if tipo == "bc":
        p_dag = p
    else:
        p_fmd = p

# ---------------------------------------------------------------------- 4. tiempos
print("=" * 70)
print("4. TIEMPO DE INFERENCIA POR PASO DE CONTROL (UN DRON, GPU)")


def medir(f, *args, n=50):
    jax.block_until_ready(f(*args))
    t0 = time.perf_counter()
    for i in range(n):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3


x0 = x_inicial(jax.random.PRNGKey(0))
o0 = observacion(x0)
t_exp = medir(jax.jit(planificar), x0, jnp.zeros((HP, 4)), jnp.zeros(4), jax.random.PRNGKey(1))
t_bc = medir(jax.jit(lambda o: jnp.tanh(mlp(p_bc, norm_o(o)))), o0)
t_fm = medir(jax.jit(lambda o, k: flujo_muestra(p_fm, o, k)), o0, jax.random.PRNGKey(2))
print(f"  experto MPPI: {t_exp:.2f} ms ({HP * NS * len(SIGS)} pasos de la física por paso de control)")
print(f"  BC: {t_bc:.3f} ms por paso")
print(f"  flow matching: {t_fm:.3f} ms por trozo de 8 acciones ({N_EULER} pasos de Euler), uno cada {K_EJEC} pasos de control")

# ---------------------------------------------------------------------- 5. trayectorias para la figura
X1, Y1 = lin(-2.8, 2.8, 20, 280), lin(-1.4, 1.4, 150, 10)
for nombre in ("experto (MPPI)", "BC", "flow matching (trozos)", "DAgger (BC)"):
    P = TRAY[nombre]
    for j in range(12):
        Pj = P[j]
        ok = np.isfinite(Pj).all(1)
        Pj = Pj[ok][::3]
        print(f"path tray {nombre} {j}:", path(Pj[:, 0], np.clip(Pj[:, 1], -1.4, 1.4), X1, Y1))
print(f"  poste: centro ({X1(0):.1f}, {Y1(0):.1f}), radio en píxeles {X1(R_POSTE) - X1(0):.1f}; inicio x = {X1(-2.5):.1f}, meta x = {X1(2.5):.1f}")
