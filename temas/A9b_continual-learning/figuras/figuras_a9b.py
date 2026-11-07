"""Datos de las figuras y tablas del tema A9b (continual learning: olvido y pérdida de plasticidad).

Uso:  uv run --with "jax[cuda13]" --with optax python -u figuras_a9b.py > salida_gpu.txt     (GPU; unos minutos)
      Con A9B_RAPIDO=1, una prueba corta. En CPU funciona igual, más despacio.

Un modelo de la fuerza aerodinámica residual de un cuadricóptero que se sigue entrenando durante el despliegue.
Entradas: velocidad v (3), empuje normalizado T (1) y eje de empuje z_b (3). Salida: fuerza residual (3, en N).
Función verdadera (supuesta, no medida): resistencia lineal y cuadrática, arrastre de rotor proporcional al empuje y a
la velocidad perpendicular al eje de empuje, y una pérdida de empuje cuadrática.

1. OLVIDO: la dinámica no cambia; cambia la misión. Cinco regímenes de vuelo en secuencia (flotar, agresivo,
   avance lento, subidas y bajadas, lateral rápido). Métodos: ajuste fino, EWC, repetición con memoria pequeña y
   entrenamiento conjunto (cota). Matriz de errores R[i, j] y métricas de GEM adaptadas a errores.
2. PLASTICIDAD: el mundo sí cambia. 300 tareas seguidas, cada una con un viento y una carga distintos; se mide el
   error al final de cada tarea. Métodos: Adam, Adam + L2, encoger y perturbar, retropropagación continua
   (reinicializa unidades de poca utilidad) y reiniciar en cada tarea. Unidades latentes y norma de los pesos.
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
RAPIDO = bool(os.environ.get("A9B_RAPIDO"))
G, M0 = 9.81, 2.0


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# ---------------------------------------------------------------------- dinámica residual verdadera (supuesta)
def residual(x, vw=jnp.zeros(3), dm=0.0):
    """x = (v, T, z_b). vw: viento; dm: masa añadida (kg). Devuelve la fuerza residual en N."""
    v, T, zb = x[:3], x[3], x[4:7]
    vr = vw - v
    f = 0.15 * vr + 0.03 * jnp.linalg.norm(vr) * vr                       # resistencia
    vperp = vr - jnp.dot(vr, zb) * zb
    f = f + 0.04 * T * vperp                                              # arrastre de rotor
    f = f - 0.06 * T ** 2 * zb                                            # pérdida de empuje
    return f - dm * G * jnp.array([0.0, 0.0, 1.0])


def muestras(key, n, reg):
    """Regímenes de vuelo: 0 flotar, 1 avance lento en x, 2 subidas y bajadas, 3 lateral rápido en y, 4 agresivo.
    Orden de la secuencia de la sección 1: ORDEN (el agresivo, el más amplio, en segundo lugar)."""
    k = jax.random.split(key, 4)
    u = jax.random.normal(k[0], (n, 3))
    centros = jnp.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0], [0, 4.5, 0], [0, 0, 0]], jnp.float32)
    escalas = jnp.array([[0.3, 0.3, 0.2], [0.6, 0.3, 0.2], [0.3, 0.3, 1.5], [0.4, 1.0, 0.2], [3.0, 3.0, 1.5]], jnp.float32)
    v = centros[reg] + escalas[reg] * u
    tilt = jnp.where(reg == 4, 0.6, jnp.where(reg == 3, 0.3, 0.12))
    a = tilt * jax.random.normal(k[1], (n, 2))
    zb = jnp.concatenate([a, jnp.ones((n, 1))], 1)
    zb = zb / jnp.linalg.norm(zb, axis=1, keepdims=True)
    T = jnp.where(reg == 4, 1.0 + 0.4 * jax.random.normal(k[2], (n,)), 1.0 + 0.1 * jax.random.normal(k[2], (n,)))
    return jnp.concatenate([v, T[:, None], zb], 1)


ORDEN = [0, 4, 1, 2, 3]
NOMBRES_REG = ["flotar", "agresivo", "avance lento", "subidas y bajadas", "lateral rápido"]     # en el orden de la secuencia


# ---------------------------------------------------------------------- red
def init_red(key, sizes=(7, 64, 64, 3)):
    ks = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (a, b)) * math.sqrt(2.0 / a), jnp.zeros(b)) for k, a, b in zip(ks, sizes[:-1], sizes[1:])]


XS = jnp.array([2.0, 2.0, 1.0, 0.3, 0.5, 0.5, 0.2])   # escala de las entradas
XM = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.9])


def red(p, x, devolver_ocultas=False):
    h = (x - XM) / XS
    ocultas = []
    for W, b in p[:-1]:
        h = jax.nn.relu(h @ W + b)
        ocultas.append(h)
    W, b = p[-1]
    y = h @ W + b
    return (y, ocultas) if devolver_ocultas else y


def mse(p, X, Y):
    return jnp.mean(jnp.sum((red(p, X) - Y) ** 2, -1))


# ---------------------------------------------------------------------- 1. olvido
print("=" * 70)
print("1. OLVIDO: CINCO REGÍMENES DE VUELO EN SECUENCIA (LA DINÁMICA ES LA MISMA; CAMBIA LA MISIÓN)")
N_REG = 5
PASOS_REG = 300 if RAPIDO else 3000
B = 128
key = jax.random.PRNGKey(0)
X_te = [muestras(jax.random.fold_in(key, 100 + r), 2000, r) for r in ORDEN]
Y_te = [jax.vmap(residual)(X) for X in X_te]
var_te = [float(jnp.mean(jnp.sum((Y - Y.mean(0)) ** 2, -1))) for Y in Y_te]
print("  varianza de la fuerza en cada régimen (N²): " + ", ".join(f"{n} {v:.2f}" for n, v in zip(NOMBRES_REG, var_te)))
Xte_all, Yte_all = jnp.stack(X_te), jnp.stack(Y_te)
opt = optax.adam(1e-3)


def entrenar_secuencia(metodo, seed=0, lam_ewc=None, n_mem=200):
    """Entrena en los cinco regímenes seguidos. Devuelve la matriz R[i, j] (error en j tras entrenar en i) y R0."""
    p = init_red(jax.random.PRNGKey(seed))
    R0 = np.array([float(mse(p, X_te[j], Y_te[j])) for j in range(N_REG)])
    R = np.zeros((N_REG, N_REG))
    anclas = []                                         # (parámetros, Fisher diagonal) de cada régimen terminado
    memoria = []                                        # (X, Y) de cada régimen terminado, n_mem puntos
    for i in range(N_REG):
        st = opt.init(p)
        if metodo == "conjunto":
            regs = [ORDEN[j] for j in range(i + 1)]
        else:
            regs = [ORDEN[i]]
        def perdida(p, X, Y, Xm, Ym, anc):
            l = mse(p, X, Y)
            if metodo == "repetición" and Xm is not None:
                l = 0.5 * l + 0.5 * mse(p, Xm, Ym)
            if metodo == "EWC":
                for (pa, Fd) in anc:
                    l = l + 0.5 * lam_ewc * sum(jnp.sum(Fi * (q - qa) ** 2) for Fi, q, qa in
                                                 zip(jax.tree_util.tree_leaves(Fd), jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(pa)))
            return l
        Xm_all = jnp.concatenate([m[0] for m in memoria]) if memoria else None
        Ym_all = jnp.concatenate([m[1] for m in memoria]) if memoria else None
        def upd(p, st, k, Xm, Ym, anc):
            k1, k2, k3 = jax.random.split(k, 3)
            r = jax.random.choice(k1, jnp.array(regs), (B,))
            X = jax.vmap(lambda kk, rr: muestras(kk, 1, rr)[0])(jax.random.split(k2, B), r)
            Y = jax.vmap(residual)(X)
            if Xm is not None:
                im = jax.random.randint(k3, (B,), 0, Xm.shape[0])
                Xb, Yb = Xm[im], Ym[im]
            else:
                Xb, Yb = None, None
            l, g = jax.value_and_grad(perdida)(p, X, Y, Xb, Yb, anc)
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st, l
        @jax.jit
        def bucle(p, st, keys, Xm, Ym, anc):
            def f(c, k):
                p, st, l = upd(*c, k, Xm, Ym, anc)
                return (p, st), l
            return lax.scan(f, (p, st), keys)
        (p, st), _ = bucle(p, st, jax.random.split(jax.random.PRNGKey(1000 * seed + i), PASOS_REG), Xm_all, Ym_all, tuple(anclas))
        for j in range(N_REG):
            R[i, j] = float(mse(p, X_te[j], Y_te[j]))
        # al terminar el régimen i: Fisher diagonal de una verosimilitud gaussiana de varianza unidad, E[sum_k (d f_k/d theta)^2]
        # (con el jacobiano de la salida; el gradiente del error se anula cuando el ajuste es bueno y la subestima) y memoria
        Xi = muestras(jax.random.PRNGKey(5000 + i), 512, ORDEN[i])
        Yi = jax.vmap(residual)(Xi)
        Js = jax.vmap(lambda x: jax.jacrev(lambda q: red(q, x[None])[0])(p))(Xi)
        Fd = jax.tree_util.tree_map(lambda J: jnp.mean(jnp.sum(J ** 2, 1), 0), Js)
        anclas.append((p, Fd))
        memoria.append((Xi[:n_mem], Yi[:n_mem]))
    return R, R0


def metricas(R, R0):
    """Métricas de GEM con errores en lugar de aciertos. FWT, respecto a predecir la media de cada régimen (la red
    aleatoria inicial es mucho peor que eso y haría la transferencia siempre positiva)."""
    T = R.shape[0]
    acc = R[-1].mean()
    bwt = np.mean([R[-1, j] - R[j, j] for j in range(T - 1)])              # > 0: olvido (el error sube)
    fwt = np.mean([var_te[j] - R[j - 1, j] for j in range(1, T)])          # > 0: lo aprendido antes ayuda
    return acc, bwt, fwt


RES1 = {}
t0 = time.perf_counter()
for metodo, kw in (("ajuste fino", {}), ("EWC", {"lam_ewc": 0.01}), ("EWC fuerte", {"lam_ewc": 1.0}),
                   ("repetición", {}), ("conjunto", {})):
    nombre = metodo
    Rs = [entrenar_secuencia("EWC" if metodo.startswith("EWC") else metodo, seed=s, **kw) for s in range(3)]
    R = np.mean([r[0] for r in Rs], 0); R0 = np.mean([r[1] for r in Rs], 0)
    RES1[nombre] = R
    acc, bwt, fwt = metricas(R, R0)
    print(f"  {nombre:12s}: error medio final {acc:.4f} N² | olvido (BWT) {bwt:+.4f} N² | transferencia (FWT) {fwt:+.4f} N² | "
          f"error en el régimen recién aprendido {np.mean(np.diag(R)):.4f} N²  ({time.perf_counter()-t0:.0f} s)", flush=True)
    print(f"      error final por régimen: " + ", ".join(f"{NOMBRES_REG[j]} {R[-1, j]:.4f}" for j in range(N_REG)))
print("  (media de 3 semillas; errores cuadráticos medios de la fuerza en datos de prueba de cada régimen)")
print("  matriz R del ajuste fino (fila: tras entrenar en; columna: error en):")
for i in range(N_REG):
    print("    " + " ".join(f"{RES1['ajuste fino'][i, j]:7.4f}" for j in range(N_REG)))

# ---------------------------------------------------------------------- 2. plasticidad
print("=" * 70)
N_TAR = 40 if RAPIDO else 300
P_TAR = 200 if RAPIDO else 500
print(f"2. PLASTICIDAD: {N_TAR} TAREAS SEGUIDAS (CADA UNA, UN VIENTO Y UNA CARGA DISTINTOS), {P_TAR} PASOS POR TAREA")


def tarea(key):
    k1, k2, k3 = jax.random.split(key, 3)
    ang = jax.random.uniform(k1, (), minval=0, maxval=2 * math.pi)
    vel = jax.random.uniform(k2, (), minval=0, maxval=8.0)
    vw = jnp.array([vel * jnp.cos(ang), vel * jnp.sin(ang), 0.0])
    dm = jax.random.uniform(k3, (), minval=-0.3, maxval=0.5)
    return vw, dm


def lote(key, vw, dm, n=64):
    X = muestras(key, n, 4)                                     # todas las tareas con las mismas entradas (régimen amplio)
    return X, jax.vmap(lambda x: residual(x, vw, dm))(X)


def maestro(key):
    """Tarea sin estructura compartida: una red aleatoria de entradas a salidas (cada tarea, otra)."""
    k1, k2, k3 = jax.random.split(key, 3)
    return (jax.random.normal(k1, (7, 32)) * 1.5, jax.random.normal(k2, (32,)) * 0.5, jax.random.normal(k3, (32, 3)) / math.sqrt(32))


def lote_maestro(key, mt, n=64):
    X = muestras(key, n, 4)
    W1, b1, W2 = mt
    return X, jnp.tanh(((X - XM) / XS) @ W1 + b1) @ W2


RHO_CBP, MADUREZ, DECAE = 1e-4, 100, 0.99                      # retropropagación continua: tasa de reemplazo, edad mínima


def utilidad_init(p):
    return [jnp.zeros(W.shape[1]) for W, b in p[:-1]], [jnp.zeros(W.shape[1]) for W, b in p[:-1]], jnp.zeros(len(p) - 1)


def cbp_paso(p, ut, edad, acum, X, key):
    """Retropropagación continua (al estilo de Dohare et al.): utilidad = media móvil de |h_i| · sum_k |W_sig[i, k]|;
    en cada capa oculta, al acumular una unidad de reemplazo, se reinicia la unidad madura de menor utilidad:
    pesos de entrada nuevos, pesos de salida a cero."""
    _, hs = red(p, X, True)
    p = list(p)
    ut2, edad2, acum2 = [], [], []
    for l, h in enumerate(hs):
        W_sig = p[l + 1][0]
        u = DECAE * ut[l] + (1 - DECAE) * jnp.mean(jnp.abs(h), 0) * jnp.sum(jnp.abs(W_sig), 1)
        e = edad[l] + 1
        a = acum[l] + RHO_CBP * h.shape[1]
        toca = a >= 1.0
        cand = jnp.where(e > MADUREZ, u, jnp.inf)
        i = jnp.argmin(cand)
        hay = toca & jnp.isfinite(cand[i])
        W, b = p[l]
        Wn = jax.random.normal(jax.random.fold_in(key, l), (W.shape[0],)) * math.sqrt(2.0 / W.shape[0])
        W = jnp.where(hay, W.at[:, i].set(Wn), W)
        b = jnp.where(hay, b.at[i].set(0.0), b)
        W_sig = jnp.where(hay, W_sig.at[i, :].set(0.0), W_sig)
        p[l] = (W, b)
        p[l + 1] = (W_sig, p[l + 1][1])
        u = jnp.where(hay, u.at[i].set(0.0), u)
        e = jnp.where(hay, e.at[i].set(0), e)
        a = jnp.where(toca, a - 1.0, a)
        ut2.append(u); edad2.append(e); acum2.append(a)
    return p, ut2, edad2, jnp.stack(acum2)


def latentes(p, X, tau=0.0):
    """Fracción de unidades τ-latentes (Sokar et al.): actividad media normalizada por la media de su capa <= tau."""
    _, hs = red(p, X, True)
    fr = []
    for h in hs:
        s = jnp.mean(jnp.abs(h), 0)
        fr.append(jnp.mean(s / (jnp.mean(s) + 1e-9) <= tau))
    return float(jnp.mean(jnp.array(fr)))


def norma(p):
    return float(jnp.sqrt(sum(jnp.sum(W ** 2) for W, b in p)))


METODOS2 = ("Adam", "Adam + L2", "encoger y perturbar", "retropropagación continua", "reiniciar en cada tarea")


def correr_flujo(metodo, seed, tipo="viento", n_tar=None, lr=1e-3):
    n_tar = N_TAR if n_tar is None else n_tar
    key = jax.random.PRNGKey(77 + seed + (1000 if tipo == "maestro" else 0))
    p = init_red(jax.random.fold_in(key, 0))
    opt2 = optax.adamw(lr, weight_decay=1e-3) if metodo == "Adam + L2" else optax.adam(lr)
    st = opt2.init(p)
    ut, edad, acum = [jnp.zeros(64), jnp.zeros(64)], [jnp.zeros(64), jnp.zeros(64)], jnp.zeros(2)
    Xp = muestras(jax.random.fold_in(key, 9), 1024, 4)
    @jax.jit
    def una_tarea(p, st, ut, edad, acum, k):
        kt, ks = jax.random.split(k)
        if tipo == "viento":
            vw, dm = tarea(kt)
            gen = lambda kk, n=64: lote(kk, vw, dm, n)
        else:
            mt = maestro(kt)
            gen = lambda kk, n=64: lote_maestro(kk, mt, n)
        def f(c, kk):
            p, st, ut, edad, acum = c
            k1, k2 = jax.random.split(kk)
            X, Y = gen(k1)
            l, g = jax.value_and_grad(mse)(p, X, Y)
            u, st = opt2.update(g, st, p)
            p = optax.apply_updates(p, u)
            if metodo == "retropropagación continua":
                p, ut, edad, acum = cbp_paso(p, ut, edad, acum, X, k2)
            return (p, st, ut, edad, acum), l
        (p, st, ut, edad, acum), L = lax.scan(f, (p, st, ut, edad, acum), jax.random.split(ks, P_TAR))
        Xt, Yt = gen(jax.random.fold_in(k, 3), 1024)
        var = jnp.mean(jnp.sum((Yt - Yt.mean(0)) ** 2, -1))
        return p, st, ut, edad, acum, mse(p, Xt, Yt) / var, jnp.mean(L[:20])
    err, err_ini, lat, nor = [], [], [], []
    for j in range(n_tar):
        kj = jax.random.fold_in(key, 1000 + j)
        if metodo == "reiniciar en cada tarea" and j > 0:
            p = init_red(jax.random.fold_in(kj, 1)); st = opt2.init(p)
        if metodo == "encoger y perturbar" and j > 0:
            p0 = init_red(jax.random.fold_in(kj, 2))
            p = jax.tree_util.tree_map(lambda a, b: 0.6 * a + 0.1 * b, p, p0)
        p, st, ut, edad, acum, e, e0 = una_tarea(p, st, ut, edad, acum, kj)
        err.append(float(e)); err_ini.append(float(e0))
        if j % 10 == 0 or j == n_tar - 1:
            lat.append(latentes(p, Xp)); nor.append(norma(p))
    return np.array(err), np.array(lat), np.array(nor)


RES2 = {}
t0 = time.perf_counter()
NS2 = 2 if RAPIDO else 3
for metodo in METODOS2:
    rs = [correr_flujo(metodo, s) for s in range(NS2)]
    E = np.mean([r[0] for r in rs], 0); Lt = np.mean([r[1] for r in rs], 0); Nr = np.mean([r[2] for r in rs], 0)
    RES2[metodo] = (E, Lt, Nr)
    q = max(N_TAR // 10, 1)
    print(f"  {metodo:26s}: error normalizado al final de cada tarea: primeras {q} {E[:q].mean():.3f} | últimas {q} {E[-q:].mean():.3f} | "
          f"unidades muertas {Lt[0]*100:4.1f} % -> {Lt[-1]*100:4.1f} % | norma de los pesos {Nr[0]:5.1f} -> {Nr[-1]:5.1f}  "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)
print(f"  (media de {NS2} semillas; error normalizado = error cuadrático medio / varianza de la fuerza en esa tarea, en datos nuevos)")

print("=" * 70)
N_TAR3 = 60 if RAPIDO else 600
print(f"3. PLASTICIDAD SIN ESTRUCTURA COMPARTIDA: {N_TAR3} TAREAS, CADA UNA UNA RED MAESTRA ALEATORIA, {P_TAR} PASOS POR TAREA")
RES3 = {}
t0 = time.perf_counter()
for metodo, lr in (("Adam", 1e-3), ("encoger y perturbar", 1e-3), ("retropropagación continua", 1e-3), ("reiniciar en cada tarea", 1e-3),
                   ("Adam", 1e-2), ("retropropagación continua", 1e-2)):
    rs = [correr_flujo(metodo, s, "maestro", N_TAR3, lr) for s in range(NS2)]
    E = np.mean([r[0] for r in rs], 0); Lt = np.mean([r[1] for r in rs], 0); Nr = np.mean([r[2] for r in rs], 0)
    metodo = metodo + (", tasa 1e-2" if lr == 1e-2 else "")
    RES3[metodo] = (E, Lt, Nr)
    q = max(N_TAR3 // 10, 1)
    print(f"  {metodo:26s}: error normalizado al final de cada tarea: primeras {q} {E[:q].mean():.3f} | últimas {q} {E[-q:].mean():.3f} | "
          f"unidades muertas {Lt[0]*100:4.1f} % -> {Lt[-1]*100:4.1f} % | norma de los pesos {Nr[0]:5.1f} -> {Nr[-1]:5.1f}  "
          f"({time.perf_counter()-t0:.0f} s)", flush=True)

def suave(E, w):
    return np.convolve(E, np.ones(w) / w, mode="valid")
X8 = lin(0, N_TAR, 60, 520)
for metodo, (E, Lt, Nr) in RES2.items():
    w = max(N_TAR // 30, 1); s = suave(E, w)
    print(f"path flujo viento {metodo}:", path(np.arange(len(s)) + w / 2, np.clip(s, 0, 0.06), X8, lin(0, 0.06, 250, 30)))
X9 = lin(0, N_TAR3, 60, 520)
for metodo, (E, Lt, Nr) in RES3.items():
    w = max(N_TAR3 // 30, 1); s = suave(E, w)
    print(f"path flujo maestro {metodo}:", path(np.arange(len(s)) + w / 2, np.clip(s, 0, 0.3), X9, lin(0, 0.3, 250, 30)))
    print(f"path latentes maestro {metodo}:", path(np.linspace(0, N_TAR3, len(Lt)), np.clip(Lt, 0, 0.6), X9, lin(0, 0.6, 250, 30)))
