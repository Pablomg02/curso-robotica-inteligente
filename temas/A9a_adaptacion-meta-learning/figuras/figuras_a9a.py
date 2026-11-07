"""Datos de las figuras y tablas del tema A9a (adaptación online y meta-learning).

Uso:  uv run --with "jax[cuda13]" --with optax python -u figuras_a9a.py > salida_gpu.txt     (GPU; unos minutos)
      Con A9A_RAPIDO=1, una prueba corta. En CPU funciona igual, más despacio.

x500 de A8a (física a 500 Hz, control a 50 Hz, retardo de motor de 30 ms) con masa desconocida y viento. Fuerza
aerodinámica supuesta (no medida): f = k_l (v_w - v) + k_q |v_w - v| (v_w - v), con k_l = 0,15 N s/m y
k_q = 0,03 N s²/m² (a 10 m/s de viento relativo, unos 4,5 N). El controlador es el geométrico de A8a con más ganancia;
las variantes solo cambian qué masa y qué fuerza externa cree conocer:
  PD nominal           masa nominal, fuerza externa nula
  PID                  PD + integral de la posición (con saturación)
  Kalman f             estima solo la fuerza externa constante (la masa se queda en la nominal)
  base aprendida       como Kalman f, pero f = c + Phi(v) a con una base Phi meta-aprendida (al estilo de Neural-Fly, sin su
                       término adversario); se adaptan c y a
  Kalman (m, f)        estima masa y fuerza externa con un filtro de Kalman sobre un modelo lineal en los parámetros
  Kalman (m, f) reg.   lo mismo con una pseudo-medida débil m = M0 en cada paso (regularización hacia el valor nominal)
  historial (GRU)      una red recurrente estima masa y fuerza desde el historial (al estilo del módulo de adaptación de RMA)
  oráculo              masa y fuerza verdaderas
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
RAPIDO = bool(os.environ.get("A9A_RAPIDO"))
G = 9.81
M0 = 2.0
J = jnp.array([0.022, 0.022, 0.040])
DT, SUB = 0.002, 10            # física a 500 Hz, control a 50 Hz
DTC = DT * SUB
TAU_M = 0.03
K_L, K_Q = 0.15, 0.03
E3 = jnp.array([0.0, 0.0, 1.0])
T_EP = 400                     # 8 s
SIG_ACC = float(os.environ.get("A9A_SIG_ACC", "0.05"))   # ruido del acelerómetro tras filtrar vibraciones (m/s², supuesto)


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


def fuerza_aero(v, vw):
    vr = vw - v
    return K_L * vr + K_Q * jnp.linalg.norm(vr) * vr


def paso_fisica(x, a, m, vw):
    """x = (p, v, q, om, T, tau); a = CTBR normalizado. Devuelve el estado siguiente y la fuerza específica media
    (lo que mide un acelerómetro ideal, en ejes mundo) y el vector de empuje medio durante el paso de control."""
    p, v, q, om, Tr, tau = x
    T_cmd = M0 * G * (1.0 + 0.8 * a[0])
    om_d = jnp.array([4.0, 4.0, 2.0]) * a[1:4]
    tau_cmd = J * 20.0 * (om_d - om) + jnp.cross(om, J * om)
    def sub(i, c):
        (p, v, q, om, Tr, tau), s_acc, y_acc = c
        Tr = Tr + DT / TAU_M * (T_cmd - Tr)
        tau = tau + DT / TAU_M * (tau_cmd - tau)
        yv = rotmat(q) @ jnp.array([0.0, 0.0, Tr])
        s = (yv + fuerza_aero(v, vw)) / m
        v = v + DT * (s - G * E3)
        p = p + DT * v
        om = om + DT * (tau - jnp.cross(om, J * om)) / J
        q = q + 0.5 * DT * qmul(q, jnp.concatenate([jnp.zeros(1), om]))
        return (p, v, q / jnp.linalg.norm(q), om, Tr, tau), s_acc + s / SUB, y_acc + yv / SUB
    return lax.fori_loop(0, SUB, sub, ((p, v, q, om, Tr, tau), jnp.zeros(3), jnp.zeros(3)))


KP, KD, KI = 4.0, 4.0, 2.0


def control(x, p_ref, v_ref, a_ref, m_hat, f_hat, integ=jnp.zeros(3)):
    """Controlador geométrico de A8a: aceleración deseada -> fuerza con la masa y la fuerza externa que cree -> CTBR."""
    p, v, q, om, Tr, tau = x
    a_d = a_ref - KP * (p - p_ref) - KD * (v - v_ref) - integ
    F = m_hat * (a_d + G * E3) - f_hat
    R = rotmat(q)
    T = jnp.dot(F, R[:, 2])
    zb = F / (jnp.linalg.norm(F) + 1e-6)
    om_d = R.T @ (6.0 * jnp.cross(R[:, 2], zb))
    a = jnp.concatenate([jnp.array([T / (M0 * G) - 1.0]) / 0.8, om_d / jnp.array([4.0, 4.0, 2.0])])
    return jnp.clip(a, -1.0, 1.0)


# ---------------------------------------------------------------------- escenarios
def referencia(tipo, t):
    """tipo 0: flotar en el origen; 1: ocho horizontal (2 m, periodo 10 s) con oscilación vertical de 1 m (periodo 5 s)."""
    w = 2 * math.pi / 10.0
    p8 = jnp.array([2.0 * jnp.sin(w * t), 1.0 * jnp.sin(2 * w * t), 1.0 * jnp.sin(2 * w * t + 1.0)])
    v8 = jnp.array([2.0 * w * jnp.cos(w * t), 2.0 * w * jnp.cos(2 * w * t), 2.0 * w * jnp.cos(2 * w * t + 1.0)])
    a8 = jnp.array([-2.0 * w * w * jnp.sin(w * t), -4.0 * w * w * jnp.sin(2 * w * t), -4.0 * w * w * jnp.sin(2 * w * t + 1.0)])
    p8 = p8 - jnp.array([0.0, 0.0, jnp.sin(1.0)])           # que empiece en el origen
    s = jnp.asarray(tipo, jnp.float32)
    return s * p8, s * v8, s * a8


CARGA, VIENTO = (1.1, 1.4), (4.0, 10.0)      # rangos de entrenamiento y de las pruebas normales


def escenario(key, tipo, carga=CARGA, viento=VIENTO, vert=(0.0, 0.0)):
    """Masa con carga que se suelta a los 5 s; viento que entra a los 2 s (flotar) o desde el principio (ocho), con ráfagas."""
    k = jax.random.split(key, 6)
    m_carga = M0 * jax.random.uniform(k[0], (), minval=carga[0], maxval=carga[1])
    m_sin = M0 * jax.random.uniform(k[1], (), minval=0.9, maxval=1.1)
    rumbo = jax.random.uniform(k[2], (), minval=0, maxval=2 * math.pi)
    vel = jax.random.uniform(k[3], (), minval=viento[0], maxval=viento[1])
    dirw = jnp.array([jnp.cos(rumbo), jnp.sin(rumbo), 0.0])
    t = jnp.arange(T_EP) * DTC
    entra = jnp.where(tipo == 0, (t >= 2.0).astype(jnp.float32), jnp.ones(T_EP))
    def ou(e, n):
        e = 0.95 * e + 0.3 * n
        return e, e
    raf = lax.scan(ou, jnp.zeros(3), jax.random.normal(k[4], (T_EP, 3)) * jnp.array([1.0, 1.0, 0.3]))[1]
    kv = jax.random.fold_in(k[3], 1)
    bajada = jax.random.uniform(kv, (), minval=vert[0], maxval=vert[1])          # corriente descendente (m/s)
    VW = entra[:, None] * (vel * dirw + raf - bajada * E3)
    M = jnp.where(t < 5.0, m_carga, m_sin)
    return M, VW, k[5]


def x_inicial():
    return (jnp.zeros(3), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), jnp.asarray(M0 * G), jnp.zeros(3))


# ---------------------------------------------------------------------- estimadores
def kalman(theta, P, y, Phi, Q, Rn):
    P = P + Q
    S = Phi @ P @ Phi.T + Rn
    K = P @ Phi.T @ jnp.linalg.inv(S)
    theta = theta + K @ (y - Phi @ theta)
    P = (jnp.eye(theta.shape[0]) - K @ Phi) @ P
    return theta, 0.5 * (P + P.T)


QM, QF, RN = 1e-4, 0.05, 0.4 ** 2          # paseo aleatorio de masa (kg²/paso) y fuerza (N²/paso); ruido de medida (N²)

# ---------------------------------------------------------------------- base meta-aprendida (al estilo de Neural-Fly, sin el término adversario)
NB = 6


def base_init(key):
    k = jax.random.split(key, 3)
    def d(kk, a, b, g):
        return (jax.nn.initializers.orthogonal(g)(kk, (a, b)), jnp.zeros(b))
    return [d(k[0], 3, 64, math.sqrt(2)), d(k[1], 64, 64, math.sqrt(2)), d(k[2], 64, 3 * NB, 1.0)]


def base(pb, v):
    h = v / 5.0
    for W, b in pb[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = pb[-1]
    return (h @ W + b).reshape(3, NB)


def base_completa(pb, v):
    """Columnas constantes (fuerza constante: viento medio, error de masa al flotar) más la base aprendida de la velocidad."""
    return jnp.concatenate([jnp.eye(3), base(pb, v)], 1)


NBC = NB + 3


def phi_constante(v):
    return jnp.eye(3)


# ---------------------------------------------------------------------- estimador por historial (al estilo de RMA)
HG = 64


def gru_init(key, nin, nh):
    k = jax.random.split(key, 4)
    d = lambda kk, a, b: (jax.nn.initializers.orthogonal(1.0)(kk, (a, b)), jnp.zeros(b))
    return {"z": d(k[0], nin + nh, nh), "r": d(k[1], nin + nh, nh), "h": d(k[2], nin + nh, nh), "out": d(k[3], nh, 4)}


def gru_paso(p, h, x):
    xh = jnp.concatenate([x, h], -1)
    z = jax.nn.sigmoid(xh @ p["z"][0] + p["z"][1])
    r = jax.nn.sigmoid(xh @ p["r"][0] + p["r"][1])
    hh = jnp.tanh(jnp.concatenate([x, r * h], -1) @ p["h"][0] + p["h"][1])
    return (1 - z) * h + z * hh


def gru_salida(p, h):
    o = h @ p["out"][0] + p["out"][1]
    return M0 * (1.0 + o[0]), M0 * G * o[1:4] * 0.3


def entrada_gru(s_med, y, v):
    return jnp.concatenate([s_med / G, y / (M0 * G), v / 5.0])


# ---------------------------------------------------------------------- un episodio con cualquier controlador
CONTROLADORES = ("PD nominal", "PID", "Kalman f", "base aprendida", "Kalman (m, f)", "Kalman (m, f) reg.", "Kalman (m, f), f_z ≈ 0",
                 "historial (GRU)", "oráculo")
KALMAN_MF = ("Kalman (m, f)", "Kalman (m, f) reg.", "Kalman (m, f), f_z ≈ 0")
P0A, QA = None, None                         # covarianza inicial y de paseo del vector a de la base (se fijan en la sección 1)
R_REG = 1.0                                  # varianza (kg²) de la pseudo-medida m = M0 de la variante regularizada


def episodio(nombre, key, tipo, pb=None, pg=None, ruido_ctrl=0.0, carga=CARGA, viento=VIENTO, vert=(0.0, 0.0)):
    M, VW, kk = escenario(key, tipo, carga, viento, vert)
    nb = NBC if nombre == "base aprendida" else 3
    phi = (lambda v: base_completa(pb, v)) if nombre == "base aprendida" else phi_constante
    th0 = jnp.concatenate([jnp.array([M0]), jnp.zeros(nb)])
    if nombre == "base aprendida":
        P0 = jax.scipy.linalg.block_diag(jnp.array([[0.3 ** 2]]), 2.0 ** 2 * jnp.eye(3), P0A)
        Q = jax.scipy.linalg.block_diag(jnp.array([[QM]]), QF * jnp.eye(3), QA)
    elif nombre == "Kalman (m, f), f_z ≈ 0":            # se le dice que el viento casi no empuja en vertical
        P0 = jnp.diag(jnp.array([0.3 ** 2, 2.0 ** 2, 2.0 ** 2, 0.3 ** 2]))
        Q = jnp.diag(jnp.array([QM, QF, QF, QF / 100]))
    else:
        P0 = jnp.diag(jnp.concatenate([jnp.array([0.3 ** 2]), jnp.full(3, 2.0 ** 2)]))
        Q = jnp.diag(jnp.concatenate([jnp.array([QM]), jnp.full(3, QF)]))
    def f(c, inp):
        x, th, P, integ, h, s_f, y_f = c
        t, m, vw, kt = inp
        k1, k2 = jax.random.split(kt)
        p_ref, v_ref, a_ref = referencia(tipo, t * DTC)
        v = x[1]
        if nombre == "PD nominal":
            mh, fh = M0, jnp.zeros(3)
        elif nombre == "PID":
            mh, fh = M0, jnp.zeros(3)
        elif nombre == "Kalman f":
            mh, fh = M0, th[1:4]
        elif nombre == "base aprendida":
            mh, fh = M0, phi(v) @ th[1:]
        elif nombre in KALMAN_MF:
            mh, fh = jnp.clip(th[0], 0.5 * M0, 2.0 * M0), th[1:4]
        elif nombre == "historial (GRU)":
            mh, fh = gru_salida(pg, h)
            mh = jnp.clip(mh, 0.5 * M0, 2.0 * M0)
        else:
            mh, fh = m, fuerza_aero(v, vw)
        a = control(x, p_ref, v_ref, a_ref, mh, fh, integ if nombre == "PID" else jnp.zeros(3))
        a = jnp.clip(a + ruido_ctrl * jax.random.normal(k2, (4,)), -1, 1)
        x2, s, y = paso_fisica(x, a, m, vw)
        # medidas: fuerza específica con ruido (acelerómetro, ejes mundo) y empuje conocido por el modelo de motor
        s_med = s + SIG_ACC * jax.random.normal(k1, (3,))
        al = 0.4                                              # filtro de primer orden (≈ 30 ms) de medida y regresor
        s_f = (1 - al) * s_f + al * s_med
        y_f = (1 - al) * y_f + al * y
        integ = jnp.clip(integ + KI * DTC * (x2[0] - p_ref), -3.0, 3.0)
        if nombre in ("Kalman f", "base aprendida"):        # masa nominal: y - M0 s = -f = -Phi(v) a
            thf, Pf = kalman(th[1:], P[1:, 1:], y_f - M0 * s_f, -phi(x2[1]), Q[1:, 1:], RN * jnp.eye(3))
            th = th.at[1:].set(thf); P = P.at[1:, 1:].set(Pf)
        elif nombre in KALMAN_MF:
            Phi = jnp.concatenate([s_f[:, None], -jnp.eye(3)], 1)
            th, P = kalman(th, P, y_f, Phi, Q, RN * jnp.eye(3))
            if nombre == "Kalman (m, f) reg.":              # pseudo-medida m = M0: tira de la masa hacia el valor nominal
                H1 = jnp.zeros((1, 4)).at[0, 0].set(1.0)
                th, P = kalman(th, P, jnp.array([M0]), H1, jnp.zeros((4, 4)), jnp.array([[R_REG]]))
        if nombre == "historial (GRU)":
            h = gru_paso(pg, h, entrada_gru(s_med, y, x2[1]))
        return (x2, th, P, integ, h, s_f, y_f), (x[0] - p_ref, mh, x2[1], s_med, y, fuerza_aero(x2[1], vw), m, a)
    c0 = (x_inicial(), th0, P0, jnp.zeros(3), jnp.zeros(HG), G * E3, M0 * G * E3)
    _, (E, MH, V, S, Y, F, MT, A) = lax.scan(f, c0, (jnp.arange(T_EP), M, VW, jax.random.split(kk, T_EP)))
    return dict(E=E, MH=MH, V=V, S=S, Y=Y, F=F, M=MT, A=A)


def rms(E, a, b):
    return np.sqrt(np.mean(np.sum(np.asarray(E)[:, int(a / DTC):int(b / DTC)] ** 2, -1), 1))


# ---------------------------------------------------------------------- 1. base meta-aprendida
print("=" * 70)
print("1. META-APRENDIZAJE DE LA BASE Phi(v) PARA LA FUERZA AERODINÁMICA (CADA EPISODIO, UN VIENTO DISTINTO)")
N_META = 400 if RAPIDO else 2000
t0 = time.perf_counter()
claves_meta = jax.random.split(jax.random.PRNGKey(10), N_META)
tipos_meta = jnp.arange(N_META) % 2
gen = jax.jit(jax.vmap(lambda k, ti: episodio("oráculo", k, ti, ruido_ctrl=0.25)))
D = gen(claves_meta, tipos_meta)
Vm, Fm = D["V"], D["F"]
print(f"  {N_META} episodios de 8 s con el oráculo y ruido de control en {time.perf_counter()-t0:.1f} s; "
      f"|f| media {float(jnp.linalg.norm(Fm, axis=-1).mean()):.2f} N, |v| media {float(jnp.linalg.norm(Vm, axis=-1).mean()):.2f} m/s")


def perdida_meta(pb, V, F, key, lam=0.1, n_sop=40):
    """Regresión de cresta por episodio sobre n_sop puntos (soporte), error en el resto (consulta): la base se
    entrena para que un vector a ajustado con pocos datos explique la fuerza de ese viento (bucle interno cerrado)."""
    def una(Ve, Fe, k):
        idx = jax.random.permutation(k, Ve.shape[0])
        s, qq = idx[:n_sop], idx[n_sop:n_sop + 160]
        Phs = jax.vmap(lambda v: base_completa(pb, v))(Ve[s]).reshape(-1, NBC)      # (3 n_sop, NBC)
        a = jnp.linalg.solve(Phs.T @ Phs + lam * jnp.eye(NBC), Phs.T @ Fe[s].reshape(-1))
        pred = jax.vmap(lambda v: base_completa(pb, v) @ a)(Ve[qq])
        return jnp.mean(jnp.sum((pred - Fe[qq]) ** 2, -1))
    return jnp.mean(jax.vmap(una)(V, F, jax.random.split(key, V.shape[0])))


def entrenar_base(pasos=300 if RAPIDO else 3000, B=32):
    pb = base_init(jax.random.PRNGKey(3))
    opt = optax.adam(1e-3)
    st = opt.init(pb)
    @jax.jit
    def upd(pb, st, k, V, F):
        k1, k2 = jax.random.split(k)
        i = jax.random.randint(k1, (B,), 0, int(N_META * 0.8))
        l, g = jax.value_and_grad(perdida_meta)(pb, V[i], F[i], k2)
        u, st = opt.update(g, st, pb)
        return optax.apply_updates(pb, u), st, l
    key = jax.random.PRNGKey(4)
    for it in range(pasos):
        key, kk = jax.random.split(key)
        pb, st, l = upd(pb, st, kk, Vm, Fm)
    return pb


pb = entrenar_base()
nv = int(N_META * 0.8)
l_val = float(perdida_meta(pb, Vm[nv:], Fm[nv:], jax.random.PRNGKey(5)))
# referencia: base constante (la fuerza como un vector constante por episodio, es decir, el Kalman f)
def perdida_const(V, F, key, n_sop=40):
    def una(Ve, Fe, k):
        idx = jax.random.permutation(k, Ve.shape[0])
        s, qq = idx[:n_sop], idx[n_sop:n_sop + 160]
        return jnp.mean(jnp.sum((Fe[qq] - Fe[s].mean(0)) ** 2, -1))
    return jnp.mean(jax.vmap(una)(V, F, jax.random.split(key, V.shape[0])))
l_const = float(perdida_const(Vm[nv:], Fm[nv:], jax.random.PRNGKey(5)))
var_f = float(jnp.mean(jnp.sum((Fm[nv:] - Fm[nv:].mean(1, keepdims=True)) ** 2, -1)))
def ajuste_a(Ve, Fe, lam=0.1):
    Ph = jax.vmap(lambda v: base_completa(pb, v))(Ve).reshape(-1, NBC)
    return jnp.linalg.solve(Ph.T @ Ph + lam * jnp.eye(NBC), Ph.T @ Fe.reshape(-1))[3:]
A_val = jax.vmap(ajuste_a)(Vm[nv:], Fm[nv:])
P0A = jnp.cov(A_val.T) + 1e-3 * jnp.eye(NB)
QA = (QF / 2.0 ** 2) * P0A                   # la misma velocidad relativa de paseo que la base constante (QF frente a 2²)
print(f"  vector a por episodio: desviación típica media {float(jnp.sqrt(jnp.diag(P0A)).mean()):.2f}")
print(f"  error cuadrático medio de la fuerza en episodios de validación con 40 puntos de cada viento: constante + base aprendida "
      f"{l_val:.3f} N², vector constante {l_const:.3f} N² (variación de f dentro del episodio {var_f:.3f} N²)")

# ---------------------------------------------------------------------- 2. estimador por historial
print("=" * 70)
print("2. ESTIMADOR DE MASA Y FUERZA DESDE EL HISTORIAL (GRU, SUPERVISADO CON LOS VALORES VERDADEROS DEL SIMULADOR)")
X_in = jax.vmap(jax.vmap(entrada_gru))(D["S"], D["Y"], D["V"])            # (N, T, 9)
Y_m, Y_f = D["M"], D["F"]


def perdida_gru(pg, X, Ym, Yf):
    def una(Xe, Yme, Yfe):
        def f(h, x):
            h = gru_paso(pg, h, x)
            return h, gru_salida(pg, h)
        _, (mh, fh) = lax.scan(f, jnp.zeros(HG), Xe)
        return jnp.mean(((mh - Yme) / M0) ** 2) + jnp.mean(jnp.sum(((fh - Yfe) / (M0 * G)) ** 2, -1))
    return jnp.mean(jax.vmap(una)(X, Ym, Yf))


def entrenar_gru(pasos=300 if RAPIDO else 4000, B=32):
    pg = gru_init(jax.random.PRNGKey(6), 9, HG)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    st = opt.init(pg)
    @jax.jit
    def upd(pg, st, k, X, Ym, Yf):
        i = jax.random.randint(k, (B,), 0, nv)
        l, g = jax.value_and_grad(perdida_gru)(pg, X[i], Ym[i], Yf[i])
        u, st = opt.update(g, st, pg)
        return optax.apply_updates(pg, u), st, l
    key = jax.random.PRNGKey(7)
    t0 = time.perf_counter()
    for it in range(pasos):
        key, kk = jax.random.split(key)
        pg, st, l = upd(pg, st, kk, X_in, Y_m, Y_f)
        if it % 1000 == 0 or it == pasos - 1:
            print(f"    paso {it}: pérdida {float(l):.4f}", flush=True)
    print(f"  {pasos} pasos en {time.perf_counter()-t0:.0f} s; pérdida en validación "
          f"{float(perdida_gru(pg, X_in[nv:], Y_m[nv:], Y_f[nv:])):.4f}")
    return pg


pg = entrenar_gru()

# ---------------------------------------------------------------------- 3. bucle cerrado
print("=" * 70)
N_EV = 64
print(f"3. BUCLE CERRADO ({N_EV} EPISODIOS POR ESCENARIO; MISMAS MASAS, VIENTOS Y RUIDO PARA TODOS LOS CONTROLADORES)")
claves_ev = jax.random.split(jax.random.PRNGKey(2024), N_EV)
RES = {}
for tipo, nom_tipo in ((0, "flotar"), (1, "ocho")):
    print(f"  escenario «{nom_tipo}»: " + ("carga de +10–40 % que se suelta a los 5 s; viento de 4–10 m/s que entra a los 2 s"
                                         if tipo == 0 else "ocho de 2 m en 10 s con oscilación vertical de 1 m; viento de 4–10 m/s desde el inicio; carga que se suelta a los 5 s"))
    for nombre in CONTROLADORES:
        R = jax.jit(jax.vmap(lambda k: episodio(nombre, k, tipo, pb=pb, pg=pg)))(claves_ev)
        RES[(tipo, nombre)] = R
        E = R["E"]
        if tipo == 0:
            w = (rms(E, 0, 2), rms(E, 2, 5), rms(E, 5, 8))
            pico = np.linalg.norm(np.asarray(E)[:, int(5 / DTC):int(6 / DTC)], axis=-1).max(1)
            print(f"    {nombre:16s}: RMS (cm) 0–2 s {np.median(w[0])*100:5.1f} | 2–5 s (viento) {np.median(w[1])*100:5.1f} | "
                  f"5–8 s (sin carga) {np.median(w[2])*100:5.1f} | pico tras soltar {np.median(pico)*100:5.1f}", flush=True)
        else:
            w = (rms(E, 1, 5), rms(E, 5.5, 8))
            print(f"    {nombre:16s}: RMS (cm) 1–5 s {np.median(w[0])*100:5.1f} | 5,5–8 s {np.median(w[1])*100:5.1f}", flush=True)

print("  fuera de distribución (flotar): carga de +50–70 % y viento de 12–15 m/s; la GRU y la base se entrenaron con +10–40 % y 4–10 m/s")
for nombre in ("PD nominal", "Kalman f", "base aprendida", "Kalman (m, f) reg.", "historial (GRU)", "oráculo"):
    R = jax.jit(jax.vmap(lambda k: episodio(nombre, k, 0, pb=pb, pg=pg, carga=(1.5, 1.7), viento=(12.0, 15.0))))(claves_ev)
    E = R["E"]
    w = (rms(E, 0, 2), rms(E, 2, 5), rms(E, 5, 8))
    div = int(np.sum(~np.isfinite(w[1]) | (w[1] > 1.5)))
    print(f"    {nombre:16s}: RMS (cm) 0–2 s {np.nanmedian(w[0])*100:5.1f} | 2–5 s (viento) {np.nanmedian(w[1])*100:5.1f} | "
          f"5–8 s (sin carga) {np.nanmedian(w[2])*100:5.1f}; divergen {div}/{N_EV}", flush=True)
    if nombre == "historial (GRU)":
        em = np.abs(np.asarray(R["MH"]) - np.asarray(R["M"])) / M0
        print(f"      masa estimada por la GRU: error mediano a los 4,9 s {np.median(em[:, int(4.9 / DTC)])*100:5.1f} % "
              f"(con carga de +50–70 %)")
print("  fuera de distribución (flotar): corriente descendente de 2–4 m/s junto al viento horizontal (rompe «f_z ≈ 0»)")
for nombre in ("Kalman f", "Kalman (m, f), f_z ≈ 0", "historial (GRU)", "oráculo"):
    R = jax.jit(jax.vmap(lambda k: episodio(nombre, k, 0, pb=pb, pg=pg, vert=(2.0, 4.0))))(claves_ev)
    E = R["E"]
    w = (rms(E, 2, 5), rms(E, 5, 8))
    em = np.abs(np.asarray(R["MH"]) - np.asarray(R["M"])) / M0
    ez = np.abs(np.asarray(E)[:, int(2.5 / DTC):int(5 / DTC), 2]).mean(1)
    print(f"    {nombre:22s}: RMS (cm) 2–5 s {np.median(w[0])*100:5.1f} | 5–8 s {np.median(w[1])*100:5.1f}; error vertical medio "
          f"2,5–5 s {np.median(ez)*100:4.1f} cm; error de masa a los 4,9 s {np.median(em[:, int(4.9 / DTC)])*100:5.1f} %", flush=True)

tt = np.arange(T_EP) * DTC
X6, Y6 = lin(0, 8.0, 60, 520), lin(0, 0.5, 250, 30)
for nombre in CONTROLADORES:
    d = np.median(np.linalg.norm(np.asarray(RES[(0, nombre)]["E"]), axis=-1), 0)
    print(f"path flotar {nombre}:", path(tt, np.clip(d, 0, 0.5), X6, Y6))

# ---------------------------------------------------------------------- 4. identificabilidad
print("=" * 70)
print("4. ¿SE IDENTIFICA LA MASA? ESTIMACIÓN DEL KALMAN (m, f) FLOTANDO Y EN EL OCHO (CON ACELERACIÓN VERTICAL)")
for tipo, nom_tipo in ((0, "flotar"), (1, "ocho")):
    for nombre in KALMAN_MF + ("historial (GRU)",):
        R = RES[(tipo, nombre)]
        em = np.abs(np.asarray(R["MH"]) - np.asarray(R["M"])) / M0
        print(f"  {nom_tipo:7s} {nombre:20s}: mediana de |m̂ − m|/M0 a los 4,9 s {np.median(em[:, int(4.9 / DTC)])*100:5.1f} %, "
              f"a los 8 s {np.median(em[:, -1])*100:5.1f} %; con la masa nominal fija sería "
              f"{np.median(np.abs(np.asarray(R['M'])[:, int(4.9 / DTC)] - M0) / M0)*100:5.1f} % y {np.median(np.abs(np.asarray(R['M'])[:, -1] - M0) / M0)*100:5.1f} %")
X7, Y7 = lin(0, 8.0, 60, 520), lin(0.5, 1.5, 250, 30)
i = 3
print(f"path masa verdadera:", path(tt, np.asarray(RES[(0, "oráculo")]["M"])[i] / M0, X7, Y7))
for tipo, nom_tipo, nombre in ((0, "flotar", "Kalman (m, f)"), (1, "ocho", "Kalman (m, f)"), (0, "flotar", "Kalman (m, f), f_z ≈ 0"),
                               (0, "flotar", "historial (GRU)")):
    R = RES[(tipo, nombre)]
    print(f"path masa {nom_tipo} {nombre}:", path(tt, np.clip(np.asarray(R["MH"])[i] / M0, 0.5, 1.5), X7, Y7))
print(f"  episodio del ejemplo: {i}; masa con carga {float(RES[(0, 'oráculo')]['M'][i, 0] / M0):.2f} M0, sin carga {float(RES[(0, 'oráculo')]['M'][i, -1] / M0):.2f} M0")
