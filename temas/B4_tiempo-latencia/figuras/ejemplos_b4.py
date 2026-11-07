"""Ejemplos numéricos y datos de las figuras nuevas del tema B4 (desarrollo de la matemática).

Uso:  python3 ejemplos_b4.py > salida_ejemplos.txt      (numpy, CPU, unos segundos)

Cada ejemplo lleva el número con el que se cita en las hojas («ejemplo N del script»).
Los experimentos largos del eje con viento (costes J/J0) están en figuras_b4.py y no se repiten
aquí: este script solo reproduce lo que se sigue a mano y lo que dibujan las figuras nuevas.
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def titulo(n, texto):
    print("=" * 78)
    print(f"EJEMPLO {n}. {texto}")


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def solve_discrete_are(A, B, Q, R, it=20000):
    P = Q.copy()
    for _ in range(it):
        Pn = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
        if np.max(np.abs(Pn - P)) < 1e-12 * max(1.0, np.max(np.abs(P))):
            return Pn
        P = Pn
    return P


# =============================================================================== PARTE 1
titulo(1, "RELOJ DE UN APARATO: t_ap = (1 + eps) t + b, con valores de ejemplo")
eps, b = 20e-6, 3e-3            # 20 ppm de deriva y 3 ms de desfase (valores de ejemplo)
for t in (0, 60, 600):
    print(f"  t = {t:4d} s: t_ap - t = eps*t + b = {eps*t*1e3:6.2f} + {b*1e3:.2f} = {(eps*t+b)*1e3:6.2f} ms")
X = lin(0, 600, 70, 500); Y = lin(0, 16e-3, 170, 30)
print("  figura (error del reloj frente a t, 0–600 s):")
print(f"    solo desfase: M{X(0):.1f},{Y(b):.1f} L{X(600):.1f},{Y(b):.1f}")
print(f"    solo deriva:  M{X(0):.1f},{Y(0):.1f} L{X(600):.1f},{Y(eps*600):.1f}")
print(f"    los dos:      M{X(0):.1f},{Y(b):.1f} L{X(600):.1f},{Y(eps*600+b):.1f}")
print(f"    ticks y: 0 -> {Y(0):.1f}; 5 ms -> {Y(5e-3):.1f}; 10 ms -> {Y(10e-3):.1f}; 15 ms -> {Y(15e-3):.1f}")
print(f"    ticks x: 0 -> {X(0):.0f}; 300 -> {X(300):.0f}; 600 -> {X(600):.0f}")

titulo(2, "PLANIFICADOR SIN DERIVA A ESCALA DE 1 ms: actitud a 250 Hz y cámara a 30 Hz")
DT = 1e-3
T_cam, T_act = 1 / 30, 1 / 250
print("  regla: en el paso k (t_k = k*dt), la tarea se dispara si t_k >= t_prox; al dispararse, t_prox += T")
prox, fil = 0.0, []
for k in range(0, 71):
    if k * DT + 1e-12 >= prox:
        fil.append((k, prox))
        prox += T_cam
print("  cámara: j | instante programado j*T (ms) | paso en que se dispara | retraso respecto al programado (ms)")
for j, (k, p) in enumerate(fil):
    print(f"    {j} | {p*1e3:8.3f} | {k:3d} | {k - p*1e3:5.3f}")
print("  regla equivalente sin acumular flotantes: k_j = ceil(j * T / dt) = ceil(1000 j / 30)")
print("    k_j para j = 0..6:", [math.ceil(1000 * j / 30 - 1e-9) for j in range(7)])
print("    intervalos (ms):", np.diff([math.ceil(1000 * j / 30 - 1e-9) for j in range(7)]).tolist())
print("  actitud a 250 Hz: T/dt = 4 exacto: pasos", [k for k in range(0, 71) if k % 4 == 0][:6], "...")
# coordenadas de la figura de 0 a 70 ms, 13,5 px por ms, origen en x = 200
XF = lin(0, 70, 200, 1145)
print("  figura (x de cada ms):", " ".join(f"{XF(k):.0f}" for k in range(0, 71, 10)))
print("  x de los disparos de actitud:", " ".join(f"{XF(k):.0f}" for k in range(0, 71, 4)))
print("  x de los instantes programados de la cámara:", " ".join(f"{XF(p*1e3):.1f}" for _, p in fil))
print("  x de los disparos de la cámara:", " ".join(f"{XF(k):.1f}" for k, _ in fil))
print("  x de los ticks de 1 ms (figura):", f"{XF(0):.0f} a {XF(70):.0f} cada {XF(1)-XF(0):.0f} px")

titulo(3, "CONTAR PASOS REDONDEADOS FRENTE A ACUMULAR EL INSTANTE: 1818 FRENTE A 1800")
n = round(T_cam / DT)
T_ef = n * DT
print(f"  periodo redondeado: round(T/dt) = round({T_cam/DT:.4f}) = {n} pasos = {T_ef*1e3:.0f} ms")
print(f"  frecuencia efectiva 1/{T_ef*1e3:.0f} ms = {1/T_ef:.4f} Hz; error relativo {(1/T_ef - 30)/30*100:+.3f} %")
print(f"  imágenes en (0, 60 s]: floor(60000/{n}) = {60000 // n}  (60 s x {1/T_ef:.3f} Hz = {60/T_ef:.2f})")
cnt = sum(1 for j in range(1, 5000) if math.ceil(1000 * j / 30 - 1e-9) <= 60000)
print(f"  acumulando el instante: imágenes en (0, 60 s] = {cnt}")
desf = [(k * n - math.ceil(1000 * k / 30 - 1e-9)) for k in (1, 10, 100, 1800)]
print("  adelanto acumulado de la versión redondeada respecto a la exacta tras 1, 10, 100 y 1800 imágenes (ms):", desf)
print(f"    = j * (33,333 - 33) ms: tras 1800 imágenes, 1800/3 = {1800/3:.0f} ms adelantada (18 imágenes de 33 ms = {18*33} ms)")

titulo(4, "EL TIEMPO EN float32 ACUMULANDO t += dt FRENTE A t = k * dt")
t32 = np.float32(0.0)
dt32 = np.float32(DT)
marcas = {}
for k in range(1, 3_600_001):
    t32 = np.float32(t32 + dt32)
    if k in (1000, 10_000, 60_000, 600_000, 3_600_000):
        marcas[k] = float(t32)
for k, v in marcas.items():
    exacto = k * DT
    print(f"  tras {k:7d} pasos (t = {exacto:6.0f} s): float32 acumulado = {v:.6f} s; error = {(v - exacto)*1e3:+9.3f} ms")
print(f"  dt en float32 = {float(dt32):.12f} (no es 0,001 exacto)")
for t in (1, 16, 60, 3600):
    print(f"  separación entre float32 consecutivos cerca de {t:4d} s: {float(np.spacing(np.float32(t)))*1e6:.3f} us")
print("  por qué el error es sistemático: en cada tramo [2^e, 2^(e+1)) la separación es fija y cada suma se redondea igual")
for lo, hi in ((8, 16), (16, 32), (32, 64)):
    sp = float(np.spacing(np.float32(lo)))
    n_sep = float(dt32) / sp
    paso_real = round(n_sep) * sp
    print(f"    [{lo:2d}, {hi:2d}) s: separación {sp*1e6:.3f} us; 1 ms = {n_sep:.3f} separaciones -> se suman {round(n_sep)}:"
          f" cada paso avanza {paso_real*1e3:.5f} ms ({(paso_real-DT)*1e6:+.3f} us por paso)")
sp = float(np.spacing(np.float32(32)))
print(f"    de 32 a 60 s: 28000 pasos x {(round(float(dt32)/sp)*sp-DT)*1e6:+.3f} us = {28000*(round(float(dt32)/sp)*sp-DT)*1e3:+.1f} ms")
print(f"  un contador int32 de pasos de 1 ms dura 2^31 ms = {2**31/1000/86400:.1f} días")
print("  con k entero y t = k*dt en float64 el error es < 1e-12 s en todo el rango")

# =============================================================================== PARTE 2
titulo(5, "TRES TIPOS DE RETARDO SOBRE UNA SENOIDE DE 2 Hz (datos de la figura)")
f0 = 2.0
tt = np.linspace(0, 0.75, 751)
x = np.sin(2 * np.pi * f0 * tt)
tau_tr = 0.06                     # transporte: 60 ms
y_tr = np.sin(2 * np.pi * f0 * (tt - tau_tr))
tau_m = 0.05                      # primer orden de 50 ms
y_din = np.zeros_like(tt)
# se arranca en régimen permanente (amplitud y fase del primer orden), para que no se vea el transitorio
y_din[0] = math.sin(-math.atan(2 * np.pi * f0 * tau_m)) / math.sqrt(1 + (2 * np.pi * f0 * tau_m) ** 2)
for i in range(1, len(tt)):
    h = tt[i] - tt[i - 1]
    y_din[i] = y_din[i - 1] + h / tau_m * (x[i - 1] - y_din[i - 1])
Ts = 0.05                         # muestreo y retención a 20 Hz
y_zoh = np.sin(2 * np.pi * f0 * np.floor(tt / Ts) * Ts)
w = 2 * np.pi * f0
print(f"  transporte 60 ms: misma amplitud, fase -w*tau = {-w*tau_tr*180/math.pi:.1f} grados")
print(f"  primer orden de 50 ms: amplitud 1/sqrt(1+(w tau)^2) = {1/math.sqrt(1+(w*tau_m)**2):.3f}, fase -atan(w tau) = {-math.degrees(math.atan(w*tau_m)):.1f} grados"
      f" (equivale a {math.atan(w*tau_m)/w*1e3:.1f} ms a 2 Hz)")
print(f"  retención a 20 Hz: retardo medio Ts/2 = {Ts/2*1e3:.0f} ms, fase {-w*Ts/2*180/math.pi:.1f} grados")
# tres paneles de 0 a 0,75 s; x en [60, 380]; cada panel de alto 60
XP = lin(0, 0.75, 60, 380)
for idx, (nombre, y) in enumerate((("transporte", y_tr), ("primer orden", y_din), ("retención", y_zoh))):
    y0 = 55 + idx * 85
    YP = lin(-1, 1, y0 + 30, y0 - 30)
    sel = slice(None, None, 5)
    print(f"  panel {idx} ({nombre}), centro y = {y0}:")
    print("    original:", path(tt[sel], x[sel], XP, YP))
    if nombre == "retención":
        # escalera: se dibuja con todos los puntos para que los saltos salgan verticales
        pts = []
        for kk in range(int(0.75 / Ts) + 1):
            t0, t1 = kk * Ts, min((kk + 1) * Ts, 0.75)
            v = math.sin(2 * np.pi * f0 * t0)
            pts += [(t0, v), (t1, v)]
        print("    retrasada:", "M" + " L".join(f"{XP(a):.1f},{YP(b_):.1f}" for a, b_ in pts))
    else:
        print("    retrasada:", path(tt[sel], y[sel], XP, YP))

titulo(6, "BÚFER CIRCULAR: D = 8 CASILLAS, RETARDO d = 3, Y MEMORIA")
D, d = 8, 3
for k in range(8, 13):
    print(f"  k = {k:2d}: se escribe x_{k} en la casilla k mod D = {k % D}; se lee x_{k-d} de (k-d) mod D = {(k-d) % D}")
for nenv, ms, f in ((16384, 200, 1000), (16384, 200, 200)):
    Dc = ms * f // 1000
    print(f"  {nenv} entornos x {Dc} casillas x 17 float32 x 4 B = {nenv*Dc*17*4/1e6:.1f} MB")
# geometría de la figura: 8 casillas en un anillo
cx, cy, R = 150, 130, 85
print("  centros de las casillas en el anillo (casilla 0 arriba, sentido horario):")
for i in range(D):
    a = -math.pi / 2 + 2 * math.pi * i / D
    print(f"    {i}: ({cx + R*math.cos(a):.1f}, {cy + R*math.sin(a):.1f})")

titulo(7, "RETARDO FRACCIONARIO POR INTERPOLACIÓN LINEAL: |H| Y FASE")


def Hfrac(alpha, th):
    return (1 - alpha) + alpha * np.exp(-1j * th)


for alpha in (0.0, 0.25, 0.5):
    vals = [abs(Hfrac(alpha, th)) for th in (0.0, math.pi / 5, math.pi / 2, math.pi)]
    form = [math.sqrt(1 - 2 * alpha * (1 - alpha) * (1 - math.cos(th))) for th in (0.0, math.pi / 5, math.pi / 2, math.pi)]
    print(f"  alpha = {alpha}: |H| en theta = 0, pi/5, pi/2, pi: {np.round(vals, 4).tolist()} (fórmula {np.round(form, 4).tolist()}); |1-2 alpha| = {abs(1-2*alpha)}")
th = 0.1
print(f"  baja frecuencia (theta = {th}): fase del corchete con alpha = 0,25: {np.angle(Hfrac(0.25, th)):.5f} rad; -alpha*theta = {-0.25*th:.5f} rad")
print("  giróscopo a 1 kHz, retardo de 2,5 ms -> m = 2, alpha = 0,5; |H| = |cos(theta/2)|:")
for fz in (100, 250, 400, 500):
    thz = 2 * math.pi * fz / 1000
    print(f"    {fz:3d} Hz (theta = {thz:.3f}): |H| = {abs(Hfrac(0.5, thz)):.3f}")
XH = lin(0, math.pi, 60, 500); YH = lin(0, 1, 230, 30)
ths = np.linspace(0, math.pi, 61)
for alpha in (0.0, 0.25, 0.5):
    print(f"  path |H| alpha={alpha}:", path(ths, [abs(Hfrac(alpha, t_)) for t_ in ths], XH, YH))
print(f"  ticks: x pi/2 -> {XH(math.pi/2):.0f}, pi -> {XH(math.pi):.0f}; y 0,5 -> {YH(0.5):.0f}, 0,707 -> {YH(0.7071):.1f}")

titulo(8, "RETARDO PURO EN FRECUENCIA: FASE -w*tau Y MARGEN CONSUMIDO EN EL EJE (w_c = 12,084 rad/s)")
wc, pm_deg = 12.084, 65.53      # del ejemplo 9 (cruce y margen del eje con el LQR discreto)
for tau in (0.005, 0.010, 0.050, 0.090):
    print(f"  tau = {tau*1e3:4.0f} ms: fase en w_c = -w_c tau = {-wc*tau:.4f} rad = {-math.degrees(wc*tau):6.2f} grados; margen que queda {pm_deg - math.degrees(wc*tau):5.1f} grados")
print(f"  cada 10 ms de retardo cuestan {math.degrees(wc*0.01):.2f} grados a {wc} rad/s")
print(f"  retenedor a 100 Hz: T/2 = 5 ms -> {math.degrees(wc*0.005):.2f} grados")
print(f"  retardo que agota el margen: {math.radians(pm_deg):.4f} rad / {wc} rad/s = {math.radians(pm_deg)/wc*1e3:.2f} ms")
XW = lin(0, 30, 60, 520); YW = lin(-120, 0, 250, 30)
for tau in (0.010, 0.050, 0.0947):
    print(f"  recta tau = {tau*1e3:.1f} ms: M{XW(0):.1f},{YW(0):.1f} L{XW(30):.1f},{YW(max(-120, -math.degrees(30*tau))):.1f}"
          + (f" (corta y = -120 en w = {120/math.degrees(tau):.2f} -> x = {XW(120/math.degrees(tau)):.1f})" if math.degrees(30 * tau) > 120 else ""))
print(f"  w_c -> x = {XW(wc):.1f}; -65,5 grados -> y = {YW(-65.5):.1f}; ticks y: -30 {YW(-30):.1f}, -60 {YW(-60):.1f}, -90 {YW(-90):.1f}, -120 {YW(-120):.1f}")
print(f"  ticks x: 0 {XW(0):.0f}, 10 {XW(10):.0f}, 20 {XW(20):.0f}, 30 {XW(30):.0f}")
for tau in (0.010, 0.050, 0.0947):
    print(f"    punto en w_c para tau = {tau*1e3:.1f} ms: y = {YW(-math.degrees(wc*tau)):.1f}")

titulo(9, "EL EJE DEL EJEMPLO: LQR, LAZO, CRUCE, MARGEN Y RETARDO MÁXIMO")
Ts = 0.01
A = np.array([[1, Ts], [0, 1]])
B = np.array([[0.5 * Ts ** 2], [Ts]])
Q = np.diag([1.0, 0.0])
Rr = np.array([[1.0 / 4096]])
print(f"  A = [[1, {Ts}], [0, 1]]; B = [{B[0,0]:.5f}, {B[1,0]:.2f}]; Q = diag(1, 0); R = 1/4096 = {1/4096:.3e}")
kpc = math.sqrt(1 / Rr[0, 0]); kdc = math.sqrt(2 * kpc)
print(f"  en continuo, LQR de p'' = u con q = 1, r = 1/4096: k_p = sqrt(q/r) = {kpc:.2f} = w_n^2 (w_n = {math.sqrt(kpc):.2f} rad/s); k_d = sqrt(2 k_p) = {kdc:.3f} = 2 zeta w_n (zeta = {kdc/(2*math.sqrt(kpc)):.3f})")
P = solve_discrete_are(A, B, Q, Rr)
K = np.linalg.solve(Rr + B.T @ P @ B, B.T @ P @ A)
kp, kd = K.ravel()
print(f"  LQR discreto a 100 Hz: K = [k_p, k_d] = [{kp:.3f}, {kd:.3f}]")
w2 = (kd ** 2 + math.sqrt(kd ** 4 + 4 * kp ** 2)) / 2
wcx = math.sqrt(w2)
print(f"  |L(jw)| = 1 con L(s) = (k_p + k_d s)/s^2:  w^4 = k_p^2 + k_d^2 w^2 -> w^2 = (k_d^2 + sqrt(k_d^4 + 4 k_p^2))/2")
print(f"    k_d^2 = {kd**2:.2f}; k_d^4 = {kd**4:.1f}; 4 k_p^2 = {4*kp**2:.1f}; raíz = {math.sqrt(kd**4+4*kp**2):.2f}; w_c^2 = {w2:.2f}; w_c = {wcx:.3f} rad/s")
print(f"    comprobación |L(j w_c)| = {abs((kp + kd*1j*wcx)/(1j*wcx)**2):.6f}")
pm = math.atan(kd * wcx / kp)
print(f"  arg L(j w_c) = -180 + atan(k_d w_c / k_p) = -180 + atan({kd*wcx/kp:.4f}) = -180 + {math.degrees(pm):.2f} grados")
print(f"  margen de fase = {math.degrees(pm):.2f} grados = {pm:.4f} rad")
print(f"  retardo máximo = PM/w_c = {pm:.4f}/{wcx:.3f} = {pm/wcx*1e3:.2f} ms; menos T/2 del retenedor = {(pm/wcx - Ts/2)*1e3:.2f} ms")
print(f"  retardo d pasos -> retardo total d*T + T/2: inestable cuando d*10 + 5 >= {pm/wcx*1e3:.1f} ms -> d >= {(pm/wcx - Ts/2)/Ts:.2f} -> d = 9")


def radio_sin_predictor(d):
    nx = 2 * (d + 1)
    M = np.zeros((nx, nx))
    M[0:2, 0:2] = A
    M[0:2, 2 * d:2 * d + 2] += -B @ K
    for i in range(1, d + 1):
        M[2 * i:2 * i + 2, 2 * (i - 1):2 * (i - 1) + 2] = np.eye(2)
    return np.abs(np.linalg.eigvals(M)).max()


for dd in (0, 4, 7, 8, 9, 10):
    print(f"  discreto exacto, u_k = -K x_(k-d): d = {dd:2d} ({dd*10:3d} ms): radio espectral = {radio_sin_predictor(dd):.4f}")

titulo(10, "PREDICTOR A MANO: d = 2 PASOS EN EL EJE (valores de ejemplo)")
x_med = np.array([0.10, 0.0])           # medida de hace 2 pasos: 10 cm, en reposo
u_hist = [-0.5, -0.4]                   # aceleraciones mandadas en k-2 y k-1 (m/s^2)
xe = x_med.copy()
print(f"  x_(k-2) = {x_med.tolist()} (m, m/s); u_(k-2) = {u_hist[0]}, u_(k-1) = {u_hist[1]} m/s^2")
for i, u in enumerate(u_hist):
    nuevo = A @ xe + B.ravel() * u
    print(f"  paso {i+1}: p = {xe[0]:.6f} + {Ts}*{xe[1]:.6f} + {0.5*Ts**2:.5f}*({u}) = {nuevo[0]:.6f};  v = {xe[1]:.6f} + {Ts}*({u}) = {nuevo[1]:.6f}")
    xe = nuevo
A2_ = A @ A
direct = A2_ @ x_med + A @ B.ravel() * u_hist[0] + B.ravel() * u_hist[1]
print(f"  fórmula cerrada A^2 x + A B u_(k-2) + B u_(k-1) = {direct.round(6).tolist()}")
print(f"  A^2 = {A2_.tolist()}; A B = {(A @ B).ravel().round(6).tolist()}")
print(f"  mando con la medida vieja: -K x_(k-2) = {float(-(K @ x_med)[0]):.3f}; con la predicción: -K x^_k = {float(-(K @ xe)[0]):.3f} m/s^2")
G = B.copy(); SIG = 1.0
for dd in (1, 2, 5, 10, 15):
    S = sum(np.linalg.matrix_power(A, i) @ G @ G.T @ np.linalg.matrix_power(A, i).T for i in range(dd)) * SIG ** 2
    print(f"  error del predictor, d = {dd:2d} ({dd*10:3d} ms): desv. típica posición {math.sqrt(S[0,0])*100:.3f} cm, velocidad {math.sqrt(S[1,1])*100:.2f} cm/s")
print("  (sigma_w = 1 m/s^2 por paso, como en figuras_b4.py)")
print(f"  d = 1 a mano: G G^T sigma^2 = B B^T = [[{(B@B.T)[0,0]:.2e}, {(B@B.T)[0,1]:.2e}], [., {(B@B.T)[1,1]:.2e}]] -> desv. v = {Ts*100:.2f} cm/s")


def radio_predictor(d_real, d_mod):
    """Radio espectral del lazo lineal (sin saturar) con predictor que supone d_mod pasos y retardo real d_real.

    Estado aumentado: [x_k, x_(k-1), ..., x_(k-d_real), u_(k-1), ..., u_(k-H)], con H = max(d_mod, 1).
    El predictor parte de la medida x_(k-d_real) y la propaga d_mod pasos con u_(k-d_mod), ..., u_(k-1).
    """
    H = max(d_mod, 1)
    nx = 2 * (d_real + 1) + H

    def paso(z):
        xs_ = [z[2 * i:2 * i + 2] for i in range(d_real + 1)]
        us_ = z[2 * (d_real + 1):]           # us_[0] = u_(k-1), us_[i] = u_(k-1-i)
        xe_ = xs_[d_real].copy()
        for i2 in range(d_mod):
            xe_ = A @ xe_ + B.ravel() * us_[d_mod - 1 - i2]
        u = float(-(K @ xe_)[0])
        xn = A @ xs_[0] + B.ravel() * u
        return np.concatenate([xn] + xs_[:d_real] + [np.array([u]), us_[:H - 1]])

    M = np.column_stack([paso(e) for e in np.eye(nx)])
    return np.abs(np.linalg.eigvals(M)).max()


print("  predictor desajustado (lineal, sin saturación): radio espectral; retardo sin compensar (d - d^)*10 + 5 ms")
for dr, dm in ((0, 10), (10, 10), (12, 5), (13, 5), (14, 5), (15, 5), (12, 10), (15, 10)):
    print(f"    real {dr*10:3d} ms, supone {dm*10:3d} ms: radio = {radio_predictor(dr, dm):.4f}; sin compensar {max(dr-dm,0)*10+5:4d} ms")

# =============================================================================== PARTE 3
titulo(11, "EDAD MEDIA DE LA ÚLTIMA IMAGEN: CÁMARA A 30 Hz, 40 ms DE LATENCIA, CONTROL A 100 Hz")
L, Tc = 40.0, 1000 / 30
cap = [j * Tc for j in range(-3, 12)]
lleg = [c + L for c in cap]
eds = []
print("  control en t (ms) | última llegada (ms) | su captura (ms) | U = t - llegada | edad = t - captura")
for t in range(100, 200, 10):
    i = max(i for i, a in enumerate(lleg) if a <= t + 1e-9)
    U = t - lleg[i]
    ed = t - cap[i]
    eds.append(ed)
    print(f"    {t:3d} | {lleg[i]:7.2f} | {cap[i]:7.2f} | {U:6.2f} | {ed:6.2f}")
print(f"  media de las 10 edades = {np.mean(eds):.2f} ms; mín {min(eds):.2f}; máx {max(eds):.2f}")
print(f"  continuo (U uniforme en [0, T)): media L + T/2 = {L + Tc/2:.2f} ms; desv. típica T/sqrt(12) = {Tc/math.sqrt(12):.2f} ms")
delta = 10 / 3
print(f"  discreto: U toma 0, delta, ..., T - delta con delta = {delta:.3f} ms (rejilla común de 10 ms y 33,3 ms):"
      f" media (T - delta)/2 = {(Tc - delta)/2:.2f} -> edad {L + (Tc - delta)/2:.2f} ms")
print(f"  edad continua: entre L = {L:.1f} y L + T = {L + Tc:.1f} ms")

titulo(12, "JITTER FIFO A MANO: retardos de ejemplo en pasos de 10 ms")
dj = [8, 11, 7, 9, 6, 8]
a_prev = -math.inf
fil = []
for j, d_ in enumerate(dj):
    bruto = j + d_
    a = max(bruto, a_prev)
    fil.append((j, d_, bruto, a, a - j))
    a_prev = a
print("  j | d_j | j + d_j | a_j = max(j + d_j, a_(j-1)) | retardo efectivo a_j - j")
for r in fil:
    print("    " + " | ".join(str(v) for v in r))
print(f"  retardo medio pedido {np.mean(dj):.2f} pasos; efectivo FIFO {np.mean([r[4] for r in fil]):.2f} pasos")
rng = np.random.default_rng(0)
for sig in (1, 2, 4):
    n_ = 200_000
    d_s = np.maximum(0, np.round(8 + sig * rng.standard_normal(n_)))
    a_ = np.maximum.accumulate(np.arange(n_) + d_s)
    print(f"  sigma = {sig*10} ms: retardo efectivo medio FIFO = {np.mean(a_ - np.arange(n_))*10:.1f} ms (pedido 80 ms)")

titulo(13, "PÉRDIDAS: BERNOULLI FRENTE A GILBERT-ELLIOTT CON EL MISMO 20 %")
p_bg = 1 / 20
p_gb = 0.2 * p_bg / 0.8
Pm = np.array([[1 - p_gb, p_gb], [p_bg, 1 - p_bg]])
print(f"  ráfagas de 20 pasos (200 ms a 100 Hz): p_bg = 1/20 = {p_bg}; para un 20 %: p_gb = 0,2 p_bg / 0,8 = {p_gb:.4f}")
print(f"  P = [[1 - p_gb, p_gb], [p_bg, 1 - p_bg]] = {Pm.round(4).tolist()}  (filas: desde G, desde B)")
pi_b = p_gb / (p_gb + p_bg)
print(f"  estados: G (bueno, llega todo) y B (malo, se pierde todo); p_gb = P(G -> B), p_bg = P(B -> G)")
print(f"  estacionaria: pi_B = p_gb/(p_gb + p_bg) = {p_gb:.4f}/{p_gb + p_bg:.4f} = {pi_b:.3f}; pi_G = {1-pi_b:.3f}")
ev, evec = np.linalg.eig(Pm.T)
v = np.real(evec[:, np.argmin(abs(ev - 1))]); v = v / v.sum()
print(f"  comprobación: autovector de P^T con autovalor 1, normalizado (pi_G, pi_B): {v.round(4).tolist()}; pi P = {(v @ Pm).round(4).tolist()}")
print(f"  duración media de ráfaga 1/p_bg = {1/p_bg:.0f} pasos; de tramo bueno 1/p_gb = {1/p_gb:.0f} pasos")
for nn in (1, 2, 5, 20):
    print(f"    P(ráfaga de {nn:2d} pasos) = (1 - p_bg)^(n-1) p_bg = {(1-p_bg)**(nn-1)*p_bg:.4f}")
print(f"  Bernoulli del 20 %: ráfagas de 1/(1 - 0,2) = {1/0.8:.2f} pasos de media")
# simulación larga: pérdida media y longitud media de ráfaga
rng = np.random.default_rng(3)
N = 1_000_000
malo = False
perd = np.zeros(N, bool)
r = rng.random(N)
for i in range(N):
    malo = (r[i] < p_gb) if not malo else (r[i] >= p_bg)
    perd[i] = malo
cambios = np.diff(np.concatenate([[0], perd.astype(int), [0]]))
lr = np.flatnonzero(cambios == -1) - np.flatnonzero(cambios == 1)
print(f"  simulación de {N} pasos: pérdida {perd.mean()*100:.2f} %; ráfaga media {lr.mean():.2f} pasos; la más larga {lr.max()} pasos")
pb = rng.random(N) < 0.2
cb = np.diff(np.concatenate([[0], pb.astype(int), [0]]))
lb = np.flatnonzero(cb == -1) - np.flatnonzero(cb == 1)
print(f"  Bernoulli: pérdida {pb.mean()*100:.2f} %; ráfaga media {lb.mean():.2f} pasos; la más larga {lb.max()} pasos")
# tiras de la figura: 300 pasos (3 s a 100 Hz)
NS = 300


def tiras(seed):
    rng = np.random.default_rng(seed)
    tb = rng.random(NS) < 0.2
    tg = np.zeros(NS, bool); m_ = False
    for i in range(NS):
        u_ = rng.random()
        m_ = (u_ < p_gb) if not m_ else (u_ >= p_bg)
        tg[i] = m_
    return tb, tg


# primera semilla cuyas dos tiras de 300 pasos pierden entre el 19 y el 21 %, con 3 o más ráfagas en la de Gilbert-Elliott
for semilla in range(1000):
    tira_b, tira_g = tiras(semilla)
    nraf = int(np.sum(np.diff(np.concatenate([[0], tira_g.astype(int)])) == 1))
    if 0.19 <= tira_b.mean() <= 0.21 and 0.19 <= tira_g.mean() <= 0.21 and nraf >= 3:
        break
print(f"  semilla de las tiras: {semilla}")
print(f"  tiras de la figura ({NS} pasos): Bernoulli pierde {tira_b.sum()} ({tira_b.mean()*100:.0f} %), Gilbert-Elliott {tira_g.sum()} ({tira_g.mean()*100:.0f} %)")


def tramos(t):
    out, i = [], 0
    while i < len(t):
        if t[i]:
            j = i
            while j < len(t) and t[j]:
                j += 1
            out.append((i, j - i)); i = j
        else:
            i += 1
    return out


XS = lin(0, NS, 60, 1140)
for nombre, t in (("Bernoulli", tira_b), ("Gilbert-Elliott", tira_g)):
    tr = tramos(t)
    print(f"  {nombre}: {len(tr)} ráfagas, longitudes {[l_ for _, l_ in tr] if len(tr) < 25 else 'varias'}; más larga {max(l_ for _, l_ in tr)}")
    print("   rects:", " ".join(f"{XS(i):.1f}:{XS(i+l_)-XS(i):.1f}" for i, l_ in tr))

titulo(14, "PREDICTOR SIN MARCA DURANTE UNA RÁFAGA: CUÁNTO RETARDO QUEDA SIN COMPENSAR")
d_nom = 8
for rafaga in (5, 10, 20, 40):
    edad = d_nom + rafaga
    print(f"  ráfaga de {rafaga:2d} pasos: la última medida llega a tener {edad*10} ms; sin marca se compensan 80 ms;"
          f" quedan {(edad-d_nom)*10} ms + 5 ms sin compensar frente a los {94.7:.1f} ms del margen")

# =============================================================================== PARTE 4
titulo(15, "POR QUÉ s_(t-d) SOLO NO ES MARKOVIANO: s_(t+1) = s_t + a_t, d = 2")
for acc in ((+1, +1), (-1, -1), (+1, -1)):
    s = 0
    for a_ in acc:
        s += a_
    print(f"  s_(t-2) = 0, acciones en vuelo {acc}: s_t = {s}; z_t = (0, {acc[0]}, {acc[1]})")
print("  misma observación s_(t-2) = 0 y tres estados actuales distintos: el futuro depende de lo que la observación no lleva")
for d_ in (3, 15):
    for na in (1, 4):
        print(f"  dimensión de z_t con d = {d_:2d} y acciones de {na} componentes: dim(s) + {d_}*{na} = dim(s) + {d_*na}")
