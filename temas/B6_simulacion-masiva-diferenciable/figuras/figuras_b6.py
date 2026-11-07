"""Datos de las figuras y tablas del tema B6 (simulación masiva y diferenciable).

Uso:
  JAX_PLATFORMS=cpu uv run --with jax python figuras_b6.py > salida_cpu.txt
  uv run --with "jax[cuda13]" python figuras_b6.py > salida_gpu.txt      (si tienes GPU NVIDIA)

Secciones:
  1. Rendimiento: pasos de entorno por segundo de un x500 vectorizado frente al número de entornos.
  2. Gradientes de primer orden (a través del simulador) frente a orden cero: varianza y horizonte.
  3. Discontinuidad: un muro que el gradiente de primer orden no ve.
Los tiempos dependen de la máquina; el script imprime el dispositivo usado.
"""
import math
import time
import platform

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", False)
print("dispositivo:", jax.devices()[0], "| jax", jax.__version__, "|", platform.processor() or platform.machine())

# ====================================================================== 1. RENDIMIENTO
print("=" * 70)
print("1. PASOS DE ENTORNO POR SEGUNDO: x500 (13 estados + 4 motores), política MLP 17-64-64-4,")
print("   10 subpasos de física de 1 ms por paso de control a 100 Hz, 100 pasos de control por medida")

M, G = 2.0, 9.81
J = jnp.array([0.022, 0.022, 0.040])
L_BRAZO, KF, KM, TAU_M, W_MAX = 0.25, 8.5e-6, 1.4e-7, 0.033, 1100.0
CD = 0.3
DT_F, SUB = 1e-3, 10


def qmul(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return jnp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotar(q, v):
    qv = jnp.concatenate([jnp.zeros(1), v])
    qc = q * jnp.array([1.0, -1.0, -1.0, -1.0])
    return qmul(qmul(q, qv), qc)[1:]


def fisica(s, w_cmd):
    """Un subpaso de 1 ms (Euler semiimplícito). s = [p(3), v(3), q(4), om(3), w(4)]."""
    p, v, q, om, w = s[0:3], s[3:6], s[6:10], s[10:13], s[13:17]
    w = w + DT_F / TAU_M * (jnp.clip(w_cmd, 0.0, W_MAX) - w)
    f = KF * w ** 2
    T = jnp.sum(f)
    tau = jnp.array([L_BRAZO * (f[1] - f[3]), L_BRAZO * (f[2] - f[0]), KM * (w[0] ** 2 - w[1] ** 2 + w[2] ** 2 - w[3] ** 2)])
    a = rotar(q, jnp.array([0.0, 0.0, T / M])) - jnp.array([0.0, 0.0, G]) - CD / M * v
    v = v + DT_F * a
    p = p + DT_F * v
    domega = (tau - jnp.cross(om, J * om)) / J
    om = om + DT_F * domega
    q = q + 0.5 * DT_F * qmul(q, jnp.concatenate([jnp.zeros(1), om]))
    q = q / jnp.linalg.norm(q)
    return jnp.concatenate([p, v, q, om, w])


def politica(params, s):
    W1, b1, W2, b2, W3, b3 = params
    h = jnp.tanh(W1 @ s + b1)
    h = jnp.tanh(W2 @ h + b2)
    return 760.0 + 300.0 * jnp.tanh(W3 @ h + b3)      # velocidades de motor alrededor de las de estacionario (≈ 760 rad/s)


def paso_entorno(params, s):
    u = politica(params, s)
    s = lax.fori_loop(0, SUB, lambda i, x: fisica(x, u), s)
    # reinicio automático si se aleja o se vuelca: sin ramas, con where
    malo = (jnp.abs(s[2]) > 20.0) | (jnp.abs(s[0]) > 20.0) | ~jnp.all(jnp.isfinite(s))
    s0 = jnp.zeros(17).at[6].set(1.0).at[13:17].set(760.0)
    return jnp.where(malo, s0, s)


k = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(k, 3)
params = (0.1 * jax.random.normal(k1, (64, 17)), jnp.zeros(64), 0.1 * jax.random.normal(k2, (64, 64)), jnp.zeros(64),
          0.01 * jax.random.normal(k3, (4, 64)), jnp.zeros(4))


@jax.jit
def rollout(params, S):
    def cuerpo(S, _):
        return jax.vmap(paso_entorno, in_axes=(None, 0))(params, S), None
    S, _ = lax.scan(cuerpo, S, None, length=100)
    return S


def estado_inicial(n):
    s0 = jnp.zeros(17).at[6].set(1.0).at[13:17].set(760.0)
    return jnp.tile(s0, (n, 1))


es_gpu = jax.devices()[0].platform == "gpu"
lista_n = [1, 16, 256, 4096, 16384, 65536] + ([262144, 1048576] if es_gpu else [])
res = []
for n in lista_n:
    S = estado_inicial(n)
    rollout(params, S).block_until_ready()          # compila
    reps = 3 if n >= 65536 else 5
    t0 = time.perf_counter()
    for _ in range(reps):
        S = rollout(params, S)
    S.block_until_ready()
    dt = (time.perf_counter() - t0) / reps
    sps = n * 100 / dt
    res.append((n, sps))
    print(f"  {n:8d} entornos: {sps:14.0f} pasos de entorno/s ({sps*SUB:.3g} subpasos de física/s); {dt*1e3:8.1f} ms por 100 pasos")

# referencia: un solo entorno en Python puro con numpy, paso a paso
import numpy as onp
p_np = [onp.asarray(x) for x in params]


def paso_np(s):
    W1, b1, W2, b2, W3, b3 = p_np
    h = onp.tanh(W1 @ s + b1); h = onp.tanh(W2 @ h + b2)
    u = 760.0 + 300.0 * onp.tanh(W3 @ h + b3)
    for _ in range(SUB):
        p, v, q, om, w = s[0:3], s[3:6], s[6:10], s[10:13], s[13:17]
        w = w + DT_F / TAU_M * (onp.clip(u, 0, W_MAX) - w)
        f = KF * w ** 2; T = f.sum()
        tau = onp.array([L_BRAZO * (f[1] - f[3]), L_BRAZO * (f[2] - f[0]), KM * (w[0]**2 - w[1]**2 + w[2]**2 - w[3]**2)])
        qw, qx, qy, qz = q
        R = onp.array([[1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                       [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
                       [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]])
        a = R @ onp.array([0, 0, T / M]) - onp.array([0, 0, G]) - CD / M * v
        v = v + DT_F * a; p = p + DT_F * v
        Jn = onp.asarray(J)
        om = om + DT_F * (tau - onp.cross(om, Jn * om)) / Jn
        dq = 0.5 * DT_F * onp.array([-qx*om[0] - qy*om[1] - qz*om[2], qw*om[0] + qy*om[2] - qz*om[1],
                                      qw*om[1] - qx*om[2] + qz*om[0], qw*om[2] + qx*om[1] - qy*om[0]])
        q = q + dq; q = q / onp.linalg.norm(q)
        s = onp.concatenate([p, v, q, om, w])
    return s


s = onp.asarray(estado_inicial(1)[0])
t0 = time.perf_counter()
for _ in range(500):
    s = paso_np(s)
sps_np = 500 / (time.perf_counter() - t0)
print(f"  referencia: 1 entorno en Python con numpy, sin vectorizar: {sps_np:.0f} pasos de entorno/s")
X1 = lambda n: 60 + (math.log10(n) - 0) / 6.3 * 460
Y1 = lambda v: 250 - (math.log10(v) - 2) / 8 * 220
print("path rend:", "M" + " L".join(f"{X1(n):.1f},{Y1(v):.1f}" for n, v in res))
print("  puntos:", " ".join(f"{X1(n):.1f},{Y1(v):.1f}" for n, v in res))
print(f"  referencia numpy y = {Y1(sps_np):.1f}")
print("  ticks x (entornos): " + " ".join(f"{n}->{X1(n):.0f}" for n in (1, 10, 100, 1e3, 1e4, 1e5, 1e6)))
print("  ticks y (pasos/s): " + " ".join(f"1e{e}->{Y1(10**e):.0f}" for e in range(2, 11)))

# ====================================================================== 2. GRADIENTES
print("=" * 70)
print("2. GRADIENTE DE PRIMER ORDEN (RETROPROPAGACIÓN A TRAVÉS DEL SIMULADOR) FRENTE A ORDEN CERO")
print("   dron plano, estacionario con ráfagas, política lineal con 6 ganancias, coste cuadrático, 100 Hz")
DT = 0.01
m2, J2, c2 = 2.0, 0.022, 0.15
SIGW = 2.0      # m/s^2 de ráfaga por paso


def paso_plano(th, x, w):
    kx, kv, kz, kvz, kphi, kom = th
    px, pz, phi, vx, vz, om = x
    phi_d = jnp.clip((kx * px + kv * vx) / G, -0.6, 0.6)     # ẍ ≈ -g·φ, así que φ_d = +(kx x + kv vx)/g
    T = m2 * (G - kz * pz - kvz * vz)
    tau = J2 * (kphi * (phi_d - phi) - kom * om)
    ax = -(T / m2) * jnp.sin(phi) - c2 / m2 * vx + w[0]
    az = (T / m2) * jnp.cos(phi) - G - c2 / m2 * vz + w[1]
    vx = vx + DT * ax; vz = vz + DT * az; om = om + DT * tau / J2
    px = px + DT * vx; pz = pz + DT * vz; phi = phi + DT * om
    xn = jnp.array([px, pz, phi, vx, vz, om])
    return xn, px ** 2 + pz ** 2 + 0.01 * (T / m2 - G) ** 2 + 0.01 * phi_d ** 2


def coste(th, x0, ws):
    def f(x, w):
        return paso_plano(th, x, w)
    _, c = lax.scan(f, x0, ws)
    return jnp.mean(c)


grad_coste = jax.jit(jax.vmap(jax.grad(coste), in_axes=(None, 0, 0)), static_argnums=())
coste_v = jax.jit(jax.vmap(coste, in_axes=(None, 0, 0)))
TH_BUENO = jnp.array([1.2, 1.6, 6.0, 4.5, 60.0, 14.0])
TH_LIMITE = jnp.array([1.2, 0.25, 6.0, 4.5, 60.0, 14.0])     # poco amortiguamiento en x


def x0s(key, n):
    return 0.3 * jax.random.normal(key, (n, 6)) * jnp.array([1, 1, 0.1, 1, 1, 0.1])


def es_grad(th, key, H, n_pert=16, sigma=0.05):
    """Orden cero antitético con números aleatorios comunes: n_pert perturbaciones, 2*n_pert simulaciones."""
    kx, kw, ke = jax.random.split(key, 3)
    x0 = x0s(kx, 1)[0]
    ws = SIGW * jax.random.normal(kw, (H, 2))
    eps = jax.random.normal(ke, (n_pert, 6))
    escala = sigma * jnp.abs(th)
    def una(e):
        return (coste(th + escala * e, x0, ws) - coste(th - escala * e, x0, ws)) / 2 * e / escala
    return jnp.mean(jax.vmap(una)(eps), axis=0)


es_grad_v = jax.jit(jax.vmap(es_grad, in_axes=(None, 0, None)), static_argnums=(2,))
print("  H (pasos) | ganancias | ‖media‖ 1.er orden | std relativa 1.er orden | std relativa orden 0 (16 pert., 32 sim.) | ‖g‖ máx. de una muestra")
filas = []
for nombre, th in (("bien amortiguadas", TH_BUENO), ("poco amortiguadas", TH_LIMITE)):
    for H in (10, 50, 100, 300, 1000, 3000):
        key = jax.random.PRNGKey(H)
        ka, kb, kc = jax.random.split(key, 3)
        n = 512
        g1 = grad_coste(th, x0s(ka, n), SIGW * jax.random.normal(kb, (n, H, 2)))
        g1 = np.asarray(g1)
        m1 = g1.mean(0)
        sd1 = np.sqrt(((g1 - m1) ** 2).sum(1).mean()) / (np.linalg.norm(m1) + 1e-12)
        g0 = np.asarray(es_grad_v(th, jax.random.split(kc, 256), H))
        m0 = g0.mean(0)
        sd0 = np.sqrt(((g0 - m0) ** 2).sum(1).mean()) / (np.linalg.norm(m1) + 1e-12)
        gmax = np.linalg.norm(g1, axis=1).max()
        coseno = float(m0 @ m1 / (np.linalg.norm(m0) * np.linalg.norm(m1) + 1e-12))
        filas.append((nombre, H, np.linalg.norm(m1), sd1, sd0, gmax, coseno))
        print(f"  {H:5d} | {nombre:17s} | {np.linalg.norm(m1):10.3g} | {sd1:8.3f} | {sd0:8.3f} | {gmax:10.3g} | coseno medias {coseno:.3f}")

# coste de cálculo de cada estimador (H = 1000, 512 estimaciones)
H = 1000
kk = jax.random.split(jax.random.PRNGKey(5), 3)
X0 = x0s(kk[0], 512); WS = SIGW * jax.random.normal(kk[1], (512, H, 2)); KS = jax.random.split(kk[2], 512)
grad_coste(TH_BUENO, X0, WS).block_until_ready(); es_grad_v(TH_BUENO, KS, H).block_until_ready(); coste_v(TH_BUENO, X0, WS).block_until_ready()
def cron(f, n=3):
    t0 = time.perf_counter()
    for _ in range(n):
        r = f()
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / n
t_fwd = cron(lambda: coste_v(TH_BUENO, X0, WS)); t_g1 = cron(lambda: grad_coste(TH_BUENO, X0, WS)); t_g0 = cron(lambda: es_grad_v(TH_BUENO, KS, H))
print(f"  tiempo para 512 estimaciones con H = 1000: solo simular {t_fwd*1e3:.1f} ms; primer orden {t_g1*1e3:.1f} ms ({t_g1/t_fwd:.1f}x); orden cero con 32 simulaciones {t_g0*1e3:.1f} ms ({t_g0/t_fwd:.1f}x)")

# radio espectral del lazo cerrado linealizado en estacionario, para cada juego de ganancias
for nombre, th in (("bien amortiguadas", TH_BUENO), ("poco amortiguadas", TH_LIMITE)):
    Jx = jax.jacfwd(lambda x: paso_plano(th, x, jnp.zeros(2))[0])(jnp.zeros(6))
    rho = np.abs(np.linalg.eigvals(np.asarray(Jx))).max()
    print(f"  {nombre}: radio espectral del paso linealizado {rho:.5f}; constante de tiempo más lenta {-DT/math.log(rho):.2f} s")

# ====================================================================== 3. DISCONTINUIDAD
print("=" * 70)
print("3. UN MURO QUE EL GRADIENTE DE PRIMER ORDEN NO VE")
print("   el dron vuela a 5 m/s hacia un muro de 3 m a 10 m; θ es la altura de consigna; ráfagas verticales;")
print("   coste = 2·θ (energía de subir) + 100 si al cruzar el muro está por debajo de 3 m")
H_MURO, X_MURO, V_AV = 3.0, 10.0, 5.0
N_MURO = int(X_MURO / V_AV / DT)       # 200 pasos hasta el muro


def coste_muro(theta, ws, suave=None):
    def f(st, w):
        z, vz = st
        az = 4.0 * (theta - z) - 3.0 * vz + w
        vz = vz + DT * az; z = z + DT * vz
        return (z, vz), None
    (z, vz), _ = lax.scan(f, (theta * 1.0, jnp.array(0.0)), ws)   # ya en crucero a la altura de consigna
    if suave is None:
        choque = jnp.where(z < H_MURO, 1.0, 0.0)
    else:
        choque = jax.nn.sigmoid((H_MURO - z) / suave)
    return 2.0 * theta + 100.0 * choque


SIGZ = 20.0     # m/s^2 por paso: ráfagas fuertes, z con σ ≈ 0,4 m
for suave in (None, 0.01):
    cm = jax.jit(jax.vmap(lambda th, ws: coste_muro(th, ws, suave), in_axes=(None, 0)))
    gm = jax.jit(jax.vmap(jax.grad(lambda th, ws: coste_muro(th, ws, suave)), in_axes=(None, 0)))
    etiqueta = "muro duro (indicador)" if suave is None else f"muro suavizado (sigmoide de {suave} m)"
    print(f"  --- {etiqueta}")
    for theta in (3.4, 3.8, 4.2, 4.6):
        ws = SIGZ * jax.random.normal(jax.random.PRNGKey(7), (200000, N_MURO))
        d = 0.02
        Jp = float(cm(theta + d, ws).mean()); Jm = float(cm(theta - d, ws).mean())
        verdadero = (Jp - Jm) / (2 * d)
        g = np.asarray(gm(theta, ws[:4096]))
        pchoque = float((cm(theta, ws).mean() - 2 * theta) / 100)
        # orden cero: 64 perturbaciones de σ = 0,1 m por estimación, 200 estimaciones
        def est0(key):
            ke, kw = jax.random.split(key)
            e = jax.random.normal(ke, (64,))
            wsl = SIGZ * jax.random.normal(kw, (64, N_MURO))
            jp = jax.vmap(lambda ei, wi: coste_muro(theta + 0.1 * ei, wi, suave))(e, wsl)
            jm = jax.vmap(lambda ei, wi: coste_muro(theta - 0.1 * ei, wi, suave))(e, wsl)
            return jnp.mean((jp - jm) / 2 * e / 0.1)
        g0 = np.asarray(jax.jit(jax.vmap(est0))(jax.random.split(jax.random.PRNGKey(11), 200)))
        print(f"  θ = {theta:.1f} m: P(choque) = {pchoque:.3f}; dE[J]/dθ verdadero {verdadero:7.2f}; "
              f"primer orden (media de 4096) {g.mean():7.2f} ± {g.std()/math.sqrt(4096):.2f} (std por muestra {g.std():.1f}); "
              f"orden cero (64 pert.) {g0.mean():7.2f}, std por estimación {g0.std():.2f}")
# curva E[J](θ) para la figura (muro duro)
cm = jax.jit(jax.vmap(lambda th, ws: coste_muro(th, ws, None), in_axes=(None, 0)))
ws = SIGZ * jax.random.normal(jax.random.PRNGKey(3), (100000, N_MURO))
ths = np.linspace(2.6, 5.0, 49)
EJ = [float(cm(t, ws).mean()) for t in ths]
i_min = int(np.argmin(EJ))
print(f"  mínimo de E[J] en θ ≈ {ths[i_min]:.2f} m (E[J] = {EJ[i_min]:.2f}); el gradiente de primer orden vale 2 en todas partes: empuja a bajar hasta chocar")
X3 = lambda t: 60 + (t - 2.6) / 2.4 * 460
Y3 = lambda v: 250 - v / 110 * 220
print("path EJ:", "M" + " L".join(f"{X3(t):.1f},{Y3(v):.1f}" for t, v in zip(ths, EJ)))
print(f"  x(θ mín) = {X3(ths[i_min]):.1f}, y = {Y3(EJ[i_min]):.1f}; x(3) = {X3(3):.1f}; x(4) = {X3(4):.1f}; x(5) = {X3(5):.1f}")
print("  ticks y: " + " ".join(f"{v}->{Y3(v):.1f}" for v in (0, 25, 50, 75, 100)))
