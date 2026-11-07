"""Datos de las figuras y tablas del tema B4 (tiempo, asincronía y latencia en simulación).

Uso:  python3 figuras_b4.py > salida.txt      (numpy; un par de minutos)

Modelo común (secciones 2–4): un eje horizontal del dron en vuelo estacionario como doble
integrador, p'' = u + w, con u la aceleración mandada (inclinación por g) y w una ráfaga
blanca. Control LQR a 100 Hz. La medida de [p, v] llega con retardo.
"""
import math
import numpy as np


def solve_discrete_are(A, B, Q, R, it=20000):
    """Riccati discreta por iteración de punto fijo (converge para (A, B) estabilizable)."""
    P = Q.copy()
    for _ in range(it):
        Pn = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
        if np.max(np.abs(Pn - P)) < 1e-12 * max(1.0, np.max(np.abs(P))):
            return Pn
        P = Pn
    return P


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


# ---------------------------------------------------------------- 1. planificador multifrecuencia
print("=" * 70)
print("1. PLANIFICADOR MULTIFRECUENCIA: QUÉ SE EJECUTA EN CADA PASO DE 1 ms (primeros 100 ms)")
DT_FIS = 1e-3
tareas = [("física", 1000), ("IMU", 200), ("lazo de rate", 500), ("actitud", 250),
          ("posición", 50), ("cámara", 30), ("GNSS", 10)]
X1 = lin(0, 100, 150, 1150)
for nombre, f in tareas:
    T = 1.0 / f
    # disparo cuando el reloj simulado alcanza el siguiente instante programado (acumulando, sin deriva)
    disparos, siguiente = [], 0.0
    for k in range(101):
        t = k * DT_FIS
        if t + 1e-12 >= siguiente:
            disparos.append(k)
            siguiente += T
    errores = [k * DT_FIS - j * T for j, k in enumerate(disparos)]
    print(f"  {nombre:13s} {f:5d} Hz: {len(disparos):3d} disparos en 100 ms; retraso máx. por redondeo al paso {max(errores)*1e3:.2f} ms;"
          f" primeros (ms): {disparos[:8]}")
    if f <= 250:
        print(f"    x de los disparos: {' '.join(f'{X1(k):.0f}' for k in disparos)}")
# cámara a 30 Hz sobre una base de 1 ms: jitter determinista
T = 1 / 30
inst = [math.ceil(j * T / DT_FIS - 1e-9) * DT_FIS for j in range(30)]
dif = np.diff(inst) * 1e3
print(f"  cámara a 30 Hz sobre pasos de 1 ms: intervalos {sorted(set(np.round(dif, 3)))} ms (media {dif.mean():.3f} ms)")
n_pasos = round(T / DT_FIS)
f_real = 1 / (n_pasos * DT_FIS)
print(f"  si en cambio se dispara cada round(T/dt) = {n_pasos} pasos: {f_real:.3f} Hz; en 60 s, {60*f_real:.0f} imágenes en vez de 1800 ({60*f_real-1800:+.0f})")

# ---------------------------------------------------------------- 2. edad de la observación
print("=" * 70)
print("2. EDAD DE LA ÚLTIMA OBSERVACIÓN QUE VE EL CONTROL")
F_CAM, LAT, F_CTRL = 30.0, 0.040, 100.0      # cámara a 30 Hz, 40 ms de la captura a la llegada (supuesto), control a 100 Hz
t_cap = np.arange(0, 2, 1 / F_CAM)
t_lleg = t_cap + LAT
tc = np.arange(0.2, 1.2, 1 / F_CTRL)
edad = []
for t in tc:
    idx = np.searchsorted(t_lleg, t, side="right") - 1
    edad.append(t - t_cap[idx])
edad = np.array(edad)
print(f"  edad en el instante del control: media {edad.mean()*1e3:.1f} ms, mín {edad.min()*1e3:.1f}, máx {edad.max()*1e3:.1f} ms")
print(f"  teoría: latencia + T_cam/2 = {(LAT + 0.5/F_CAM)*1e3:.1f} ms (más el redondeo al periodo del control)")
tt = np.linspace(0.2, 0.35, 3001)
ed = []
for t in tt:
    idx = np.searchsorted(t_lleg, t, side="right") - 1
    ed.append(t - t_cap[idx])
X2 = lin(0.2, 0.35, 60, 430)
Y2 = lin(0, 0.09, 250, 30)
print("path edad:", path(tt, ed, X2, Y2))
tcs = tc[(tc >= 0.2) & (tc <= 0.35)]
eds = edad[(tc >= 0.2) & (tc <= 0.35)]
print("  puntos del control:", " ".join(f"{X2(a):.1f},{Y2(b):.1f}" for a, b in zip(tcs, eds)))
print(f"  y(40 ms) = {Y2(0.04):.1f}; y(73,3 ms) = {Y2(LAT + 1/F_CAM):.1f}; y(media) = {Y2(edad.mean()):.1f}")
print("  llegadas (x):", " ".join(f"{X2(a):.1f}" for a in t_lleg if 0.2 <= a <= 0.35))

# ---------------------------------------------------------------- 3. retardo, predictor y desajuste
print("=" * 70)
print("3. EJE HORIZONTAL A 100 Hz CON MEDIDA RETRASADA: SIN COMPENSAR, CON PREDICTOR Y CON PREDICTOR DESAJUSTADO")
DT = 0.01
A = np.array([[1, DT], [0, 1]])
B = np.array([[0.5 * DT ** 2], [DT]])
Q = np.diag([1.0, 0.0])
Rr = np.array([[1.0 / 4096]])   # en continuo: ω_n = (q/r)^(1/4) = 8 rad/s, ζ = 0,707
P = solve_discrete_are(A, B, Q, Rr)
K = np.linalg.solve(Rr + B.T @ P @ B, B.T @ P @ A)
polos = np.linalg.eigvals(A - B @ K)
print(f"  K = {K.ravel()}; polos en lazo cerrado |z| = {np.abs(polos)}; ancho de banda ≈ {np.abs(np.log(polos)).max()/DT:.2f} rad/s")
SIG_W = 1.0   # m/s^2 de ráfaga blanca (desviación por paso)
AMAX = 4.9    # saturación: unos 26,6° de inclinación


def simular(d_real, d_modelo, n=40000, seed=0, jitter=0.0, perdida=0.0, usa_marca=False, saturar=True, rafaga=None):
    """Simula el eje con la medida de [p, v] tomada en cada paso y entregada con retardo.

    d_real: retardo nominal en pasos. jitter: desviación típica (en pasos) del retardo de cada medida.
    Las medidas llegan en orden (FIFO): una no adelanta a la anterior, así que el jitter solo puede
    retrasar. perdida: probabilidad de que una medida no llegue nunca.
    d_modelo: retardo que supone el predictor (None = sin predictor). usa_marca: el predictor usa la
    edad real de la última medida (su marca de tiempo) en lugar de d_modelo.
    """
    rng = np.random.default_rng(seed)
    # instante de llegada de cada medida
    llegada = np.full(n, np.inf)
    ultima_ll = -np.inf
    malo = False
    for j in range(n):
        if rafaga is not None:
            # Gilbert-Elliott: dos estados; en el malo se pierde todo, en el bueno nada
            p_gb, p_bg = rafaga
            malo = (rng.random() < p_gb) if not malo else (rng.random() >= p_bg)
            if malo:
                continue
        elif rng.random() < perdida:
            continue
        d = d_real + (jitter * rng.standard_normal() if jitter > 0 else 0.0)
        a = max(j + max(0, round(d)), ultima_ll)
        llegada[j] = a
        ultima_ll = a
    orden = [j for j in np.argsort(llegada, kind="stable") if np.isfinite(llegada[j])]
    xs = np.zeros((n + 1, 2))
    us = np.zeros(n)
    x = np.zeros(2)
    J, ptr, ultima_j, edades = 0.0, 0, -1, []
    for k in range(n):
        xs[k] = x
        while ptr < len(orden) and llegada[orden[ptr]] <= k:
            ultima_j = max(ultima_j, orden[ptr]); ptr += 1
        if ultima_j < 0:
            xe, edad = np.zeros(2), 0
        else:
            edad = k - ultima_j
            dd = edad if (usa_marca or d_modelo is None) else d_modelo
            if d_modelo is None:
                xe = xs[ultima_j].copy()
            else:
                # predicción desde la medida: con la marca, desde su instante real; sin ella, se supone tomada hace d_modelo pasos
                xe = xs[ultima_j].copy()
                for i2 in range(dd):
                    kk = k - dd + i2
                    xe = A @ xe + B.ravel() * (us[kk] if kk >= 0 else 0.0)
        edades.append(edad)
        u = float(-(K @ xe)[0])
        if saturar:
            u = max(-AMAX, min(AMAX, u))
        us[k] = u
        J += x @ Q @ x + Rr[0, 0] * u * u
        x = A @ x + B.ravel() * (u + SIG_W * rng.standard_normal())
        if abs(x[0]) > 1e3:
            return float("inf"), float("inf"), float("nan")
    return J / n, float(np.sqrt(np.mean(np.square(xs[:n, 0])))), float(np.mean(edades)) * DT


def estable_sin_predictor(d):
    """Radio espectral del sistema aumentado con u_k = -K x_{k-d} (lineal, sin saturación)."""
    nx = 2 * (d + 1)
    M = np.zeros((nx, nx))
    M[0:2, 0:2] = A
    M[0:2, 2 * d:2 * d + 2] += -B @ K
    for i in range(1, d + 1):
        M[2 * i:2 * i + 2, 2 * (i - 1):2 * (i - 1) + 2] = np.eye(2)
    return np.abs(np.linalg.eigvals(M)).max()


d_lim = next(d for d in range(1, 300) if estable_sin_predictor(d) >= 1)
kp, kd = K.ravel()
w2 = (kd ** 2 + math.sqrt(kd ** 4 + 4 * kp ** 2)) / 2          # |L(jw)| = 1 con L(s) = (kp + kd s)/s^2
wc = math.sqrt(w2)
pm = math.atan(kd * wc / kp)
print(f"  aproximación continua: cruce ω_c = {wc:.2f} rad/s, margen de fase {math.degrees(pm):.1f}°, retardo máximo PM/ω_c = {pm/wc*1e3:.1f} ms;"
      f" menos medio paso del retenedor ({DT/2*1e3:.0f} ms): {(pm/wc - DT/2)*1e3:.1f} ms")
print(f"  sin predictor, el lazo lineal se vuelve inestable con un retardo de {d_lim} pasos = {d_lim*DT*1e3:.0f} ms")
retardos = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
J0 = simular(0, 0)[0]
print(f"  coste de referencia sin retardo J0 = {J0:.3e}")
print("  retardo real (ms) | J/J0 sin predictor | predictor exacto | predictor que supone 50 ms | que supone 100 ms | RMS pos. (cm): sin pred. / exacto")
filas = {}
for d in retardos:
    js, rs, _ = simular(d, None)
    je, re_, _ = simular(d, d)
    j5 = simular(d, 5)[0]
    j10 = simular(d, 10)[0]
    filas[d] = (js / J0, je / J0, j5 / J0, j10 / J0)
    print(f"  {d*10:5d} | {js/J0:8.2f} | {je/J0:8.2f} | {j5/J0:8.2f} | {j10/J0:8.2f} | {rs*100:7.2f} / {re_*100:5.2f}")
X3 = lin(0, 150, 60, 520)
Y3 = lin(0, math.log10(30), 250, 30)
for i, nombre in enumerate(["sin predictor", "predictor exacto", "supone 50 ms", "supone 100 ms"]):
    pts = [(d * 10, filas[d][i]) for d in retardos if np.isfinite(filas[d][i]) and filas[d][i] < 30]
    print(f"path coste {nombre}:", path([p[0] for p in pts], [math.log10(p[1]) for p in pts], X3, Y3))
    print(f"  último punto {nombre}: x={X3(pts[-1][0]):.1f} y={Y3(math.log10(pts[-1][1])):.1f}")
print(f"  ticks y: 1 -> {Y3(0):.1f}; 2 -> {Y3(math.log10(2)):.1f}; 5 -> {Y3(math.log10(5)):.1f}; 10 -> {Y3(1):.1f}; 30 -> {Y3(math.log10(30)):.1f}")
print(f"  ticks x: 0 {X3(0):.0f}; 50 {X3(50):.0f}; 100 {X3(100):.0f}; 150 {X3(150):.0f}; límite {d_lim*10} ms -> {X3(d_lim*10):.1f}")

# ---------------------------------------------------------------- 4. jitter y pérdidas: la marca de tiempo
print("=" * 70)
print("4. JITTER Y PÉRDIDAS: PREDICTOR CON EL RETARDO NOMINAL FRENTE A PREDICTOR CON LA MARCA DE TIEMPO")
D0 = 8
D0 = 8
print(f"  (retardo medio {D0*10} ms; J/J0)")
for jit in (0, 1, 2, 4):
    jn, _, ed = simular(D0, D0, jitter=jit, seed=1)
    jm = simular(D0, D0, jitter=jit, seed=1, usa_marca=True)[0]
    print(f"  jitter σ = {jit*10:3d} ms: edad media {ed*1e3:.0f} ms; J/J0 nominal {jn/J0:.2f} | con marca de tiempo {jm/J0:.2f}")
for p in (0.0, 0.2, 0.5, 0.8):
    jn, _, ed = simular(D0, D0, perdida=p, seed=2)
    jm = simular(D0, D0, perdida=p, seed=2, usa_marca=True)[0]
    print(f"  pérdida {p*100:3.0f} %: edad media {ed*1e3:.0f} ms; J/J0 nominal {jn/J0:.2f} | con marca de tiempo {jm/J0:.2f}")

for lr in (5, 20):
    p_bg = 1 / lr
    p_gb = 0.2 * p_bg / 0.8
    jr, _, ed = simular(D0, D0, rafaga=(p_gb, p_bg), seed=3)
    jrm = simular(D0, D0, rafaga=(p_gb, p_bg), seed=3, usa_marca=True)[0]
    print(f"  pérdida del 20 % a ráfagas de {lr} pasos ({lr*10} ms) de media: edad media {ed*1e3:.0f} ms; J/J0 nominal {jr/J0:.2f} | con marca {jrm/J0:.2f}")

# error de predicción irreducible: la ráfaga dentro de la ventana de retardo no se ha observado
print("  desviación típica del error del predictor exacto (solo por la ráfaga no observada):")
G = B.copy()
for d in (5, 10, 15):
    S = np.zeros((2, 2))
    for i in range(d):
        Ai = np.linalg.matrix_power(A, i)
        S += Ai @ G @ G.T @ Ai.T * SIG_W ** 2
    print(f"    {d*10} ms: posición {math.sqrt(S[0,0])*100:.2f} cm, velocidad {math.sqrt(S[1,1])*100:.1f} cm/s")

# ---------------------------------------------------------------- 5. coste de un búfer de retardo
print("=" * 70)
print("5. MEMORIA DE LOS BÚFERES DE RETARDO EN UN SIMULADOR VECTORIZADO")
for n_env in (1024, 4096, 16384):
    for ms, f in ((200, 1000), (200, 200)):
        D = int(ms * f / 1000)
        obs = 13 + 4      # estado completo + acción, float32
        print(f"  {n_env:6d} entornos, {ms} ms a {f} Hz ({D} casillas) x {obs} floats: {n_env*D*obs*4/1e6:.1f} MB")
