"""Ejemplos numéricos de las hojas del tema A4b (percepción para estimación), para seguir a mano.

Uso:  python3 ejemplos_a4b.py > salida_ejemplos.txt
Requiere numpy; corre en CPU en unos segundos. Cada ejemplo lleva su número, que es el que citan
las hojas («ejemplo 4 del script»). Los trazados SVG se imprimen como 'path etiqueta: M...'
escalados al viewBox de la figura correspondiente.

Los intrínsecos y extrínsecos de EuRoC son los de cam0 y cam1 en la configuración de OpenVINS
(rpng/open_vins, config/euroc_mav/kalibr_imucam_chain.yaml), los mismos que usa figuras_a4b.py.
Las demás cifras (puntos, poses, ruidos, umbrales) son elegidas para los ejemplos.
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)

FX, FY, CX, CY = 458.654, 457.296, 367.215, 248.375      # cam0 de EuRoC: fu, fv, cu, cv
ANCHO, ALTO = 752, 480
P_C0 = np.array([-0.0216401454975, -0.064676986768, 0.00981073058949])
P_C1 = np.array([-0.0198435579556, 0.0453689425024, 0.00786212447038])
BASE = float(np.linalg.norm(P_C1 - P_C0))


def cab(n, titulo):
    print("=" * 72)
    print(f"EJEMPLO {n}. {titulo}")


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def hat(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def Exp(phi):
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.eye(3) + hat(phi)
    u = hat(phi / th)
    return np.eye(3) + math.sin(th) * u + (1 - math.cos(th)) * u @ u


def proy(pc, fx=FX, fy=FY, cx=CX, cy=CY):
    X, Y, Z = pc
    return np.array([fx * X / Z + cx, fy * Y / Z + cy])


def jac_proy(pc, fx=FX, fy=FY):
    X, Y, Z = pc
    return np.array([[fx / Z, 0, -fx * X / Z**2], [0, fy / Z, -fy * Y / Z**2]])


# ---------------------------------------------------------------------------------------------
cab(1, "Modelo estenopeico con la cam0 de EuRoC: del punto en ejes cámara al píxel")
print(f"  fx = {FX}, fy = {FY}, cx = {CX}, cy = {CY} px; imagen {ANCHO} x {ALTO}")
pc = np.array([0.4, 0.3, 2.0])
u = proy(pc)
print(f"  p_c = {pc} m  ->  X/Z = {pc[0]/pc[2]:.3f}, Y/Z = {pc[1]/pc[2]:.3f}")
print(f"  u = fx*X/Z + cx = {FX}*{pc[0]/pc[2]:.3f} + {CX} = {u[0]:.2f} px")
print(f"  v = fy*Y/Z + cy = {FY}*{pc[1]/pc[2]:.3f} + {CY} = {u[1]:.2f} px")
print(f"  el mismo rayo a doble distancia, p_c = {2*pc}: {proy(2*pc)} (igual: la cámara no ve la distancia)")
print(f"  campo de visión horizontal: 2*atan(({ANCHO}/2)/fx) = {2*math.degrees(math.atan(ANCHO/2/FX)):.1f} grados")
print(f"  (aprox., sin distorsión); vertical: {2*math.degrees(math.atan(ALTO/2/FY)):.1f} grados")

# ---------------------------------------------------------------------------------------------
cab(2, "Del mundo a la cámara: p_c = R_wc^T (p_w - t_wc), cámara mirando a +x del mundo")
# columnas de R_wc = ejes de la cámara en el mundo: x_c (derecha) = -y_w, y_c (abajo) = -z_w, z_c (delante) = +x_w
R_wc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
t_wc = np.array([1.0, 0.0, 1.5])
p_w = np.array([3.0, -0.4, 1.2])
d = p_w - t_wc
pc2 = R_wc.T @ d
print(f"  R_wc =\n{R_wc}\n  t_wc = {t_wc}, p_w = {p_w}")
print(f"  p_w - t_wc = {d}  ->  p_c = R_wc^T (p_w - t_wc) = {pc2}")
print(f"  píxel: {proy(pc2)}")

# ---------------------------------------------------------------------------------------------
cab(3, "Error de reproyección: cuánto mueve el píxel un error de pose, según la profundidad")
for Z in (2.0, 20.0):
    p = np.array([0.0, 0.0, Z])
    du_t = proy(p - np.array([0.01, 0, 0]))[0] - proy(p)[0]     # cámara desplazada 1 cm a su derecha
    Rg = Exp(np.array([0, 0.01, 0]))                              # cámara girada 0,01 rad sobre su eje y
    du_r = proy(Rg.T @ p)[0] - proy(p)[0]
    print(f"  punto centrado a Z = {Z:4.0f} m: 1 cm de traslación -> {du_t:+.2f} px; 0,01 rad de giro -> {du_r:+.2f} px")
z_med = np.array([461.0, 315.5])
r = z_med - proy(pc2)
print(f"  medida z = {z_med}, predicción {proy(pc2)} -> residuo r = z - pi = {r}, |r| = {np.linalg.norm(r):.2f} px")
print(f"  con sigma = 1 px, contribución al coste |r|^2/sigma^2 = {r @ r:.2f}")

# ---------------------------------------------------------------------------------------------
cab(4, "Jacobiano de la proyección en p_c = (0,4; 0,3; 2) y regla de la cadena hasta la pose")
J = jac_proy(pc2)
print(f"  d pi / d p_c =\n{J}")
eps = 1e-6
Jn = np.column_stack([(proy(pc2 + eps * e) - proy(pc2 - eps * e)) / (2 * eps) for e in np.eye(3)])
print(f"  diferencias finitas: error máximo {np.abs(J - Jn).max():.2e}")
J_th = J @ hat(pc2)                     # perturbación por la derecha de R_wc: d p_c / d theta = [p_c]_x
J_t = J @ (-R_wc.T)
J_pw = J @ R_wc.T


def pc_de(R, t, pw):
    return R.T @ (pw - t)


Jn_th = np.column_stack([(proy(pc_de(R_wc @ Exp(eps * e), t_wc, p_w)) - proy(pc_de(R_wc @ Exp(-eps * e), t_wc, p_w))) / (2 * eps) for e in np.eye(3)])
Jn_t = np.column_stack([(proy(pc_de(R_wc, t_wc + eps * e, p_w)) - proy(pc_de(R_wc, t_wc - eps * e, p_w))) / (2 * eps) for e in np.eye(3)])
print(f"  d pi / d theta = J [p_c]_x =\n{J_th}\n  (dif. finitas, error {np.abs(J_th - Jn_th).max():.1e})")
print(f"  d pi / d t_wc = -J R_wc^T =\n{J_t}\n  (dif. finitas, error {np.abs(J_t - Jn_t).max():.1e})")
print(f"  d pi / d p_w = J R_wc^T =\n{J_pw}")

# ---------------------------------------------------------------------------------------------
cab(5, f"Estéreo: Z = f b / d y su incertidumbre (f = {FX} px, base de EuRoC b = {100*BASE:.2f} cm)")
fb = FX * BASE
print(f"  f b = {fb:.2f} px m")
for Z in (1.0, 3.0, 10.0, 30.0):
    dd = fb / Z
    sZ = Z**2 * 0.5 / fb
    print(f"  Z = {Z:4.0f} m -> d = f b / Z = {dd:6.2f} px; |dZ/dd| = f b / d^2 = {fb/dd**2:.4f} m/px; "
          f"sigma_Z = {100*sZ:6.1f} cm; sigma_d/d = {100*0.5/dd:5.1f} %")
rng = np.random.default_rng(5)
for Z in (3.0, 10.0):
    dd = fb / Z + rng.normal(0, 0.5, 200000)
    Zs = fb / dd[dd > 0.3]
    print(f"  Monte Carlo a Z = {Z:.0f} m (200 000 disparidades con sigma 0,5 px): desviación típica de Z = "
          f"{100*np.std(Zs):.1f} cm; mediana {np.median(Zs):.3f} m (fórmula: {100*Z**2*0.5/fb:.1f} cm)")
print("  Curva sigma_Z(Z) para la figura (viewBox 0 0 520 280; Z de 0 a 15 m en x 60..500; sigma de 0 a 2 m en y 240..30):")
Xs = lin(0, 15, 60, 500); Ys = lin(0, 2.0, 240, 30)
Zg = np.linspace(0.25, 15, 60)
for b in (BASE, 0.30):
    sg = Zg**2 * 0.5 / (FX * b); k = sg <= 2.0
    print(f"path sigma b={100*b:.0f}cm:", path(Zg[k], sg[k], Xs, Ys))
for b in (BASE, 0.30):
    Zc = math.sqrt(2.0 * FX * b / 0.5)
    print(f"  la curva de b = {100*b:.0f} cm alcanza 2 m de sigma en Z = {Zc:.1f} m")

# ---------------------------------------------------------------------------------------------
cab(6, "Escala monocular: el mundo y la trayectoria multiplicados por s dan las mismas imágenes")
pw = np.array([2.0, 0.5, 0.3]); tA = np.zeros(3); tB = np.array([0.0, 0.3, 0.0])
Rlook = R_wc
for s in (1.0, 2.0):
    uA = proy(pc_de(Rlook, s * tA, s * pw)); uB = proy(pc_de(Rlook, s * tB, s * pw))
    print(f"  s = {s}: píxel desde la pose A {uA}, desde la pose B {uB}")
print("  la IMU sí ve la escala: si el dron acelera 1 m/s^2, la versión con s = 2 tendría que acelerar 2 m/s^2;")
print("  el acelerómetro mide 1 y descarta s = 2. Con aceleración nula, 0 = 2 * 0 y no descarta nada.")

# ---------------------------------------------------------------------------------------------
cab(7, "MSCKF: proyectar al espacio nulo izquierdo de H_f con un punto visto desde dos clones")
p_f = np.array([3.0, -0.4, 1.2])
clones = [(R_wc, np.array([1.0, 0.0, 1.5])), (R_wc @ Exp(np.array([0, 0.05, 0])), np.array([1.2, 0.3, 1.5]))]
Hf = np.vstack([jac_proy(pc_de(R, t, p_f)) @ R.T for R, t in clones])
print(f"  H_f (2m x 3 = 4 x 3), filas = d pi / d p_f de cada clon:\n{Hf}")
U, S, Vt = np.linalg.svd(Hf)
N = U[:, 3:]
print(f"  valores singulares de H_f: {S} -> rango 3; espacio nulo izquierdo de dimensión 4 - 3 = 1")
print(f"  N (4 x 1, unitario) = {N.ravel()}")
print(f"  N^T H_f = {(N.T @ Hf).ravel()} (cero numérico)")
print(f"  N^T N = {(N.T @ N).ravel()} -> el ruido proyectado sigue con covarianza sigma^2 I")
dp = np.array([0.05, -0.03, 0.02])
print(f"  un error del punto de {dp} m mueve los píxeles {Hf @ dp} pero el residuo proyectado {(N.T @ Hf @ dp).item():.2e}")
for m in (2, 5, 10, 20):
    print(f"  m = {m:2d} observaciones: {2*m} residuos -> {2*m-3} tras proyectar")

# ---------------------------------------------------------------------------------------------
cab(8, "Marginalizar con el complemento de Schur: tres poses 1D y un punto")
# variables (x1, x2, x3, l). Factores: prior en x1 (sigma 0,1), odometría x1->x2 y x2->x3 (sigma 0,1),
# el punto visto desde x1 y desde x3 (sigma 0,2). Cada factor lineal r = a^T x - z añade a a^T / sigma^2.
fact = [((1, 0, 0, 0), 0.1, 0.0, "prior x1"), ((-1, 1, 0, 0), 0.1, 1.0, "odom x1-x2"), ((0, -1, 1, 0), 0.1, 1.0, "odom x2-x3"),
        ((-1, 0, 0, 1), 0.2, 2.5, "punto desde x1"), ((0, 0, -1, 1), 0.2, 0.5, "punto desde x3")]
L = np.zeros((4, 4)); eta = np.zeros(4)
for a, s, z, nom in fact:
    a = np.array(a, float); L += np.outer(a, a) / s**2; eta += a * z / s**2
print(f"  Lambda (variables x1, x2, x3, l) =\n{L}\n  eta = {eta}")
mu = np.linalg.solve(L, eta)
print(f"  solución conjunta: {mu}")
A = L[:1, :1]; B = L[:1, 1:]; C = L[1:, 1:]
Lm = C - B.T @ np.linalg.inv(A) @ B
em = eta[1:] - B.T @ np.linalg.inv(A) @ eta[:1]
print(f"  A = Lambda_x1x1 = {A.ravel()}, B = {B.ravel()}")
print(f"  B^T A^-1 B =\n{B.T @ np.linalg.inv(A) @ B}")
print(f"  Lambda marginal (x2, x3, l) = C - B^T A^-1 B =\n{Lm}\n  eta marginal = {em}")
print(f"  solución con el sistema marginal: {np.linalg.solve(Lm, em)} (igual que la conjunta)")
print(f"  covarianza marginal inv(Lambda marg) =\n{np.linalg.inv(Lm)}")
print(f"  bloque (x2,x3,l) de inv(Lambda) =\n{np.linalg.inv(L)[1:, 1:]}")
print("  relleno: Lambda_x2,l pasa de", L[1, 3], "a", round(Lm[0, 2], 4))
# Si en vez de marginalizar se tiran los factores de x1:
L2 = np.zeros((3, 3))
for a, s, z, nom in fact:
    if a[0] == 0:
        a3 = np.array(a[1:], float); L2 += np.outer(a3, a3) / s**2
print(f"  si se tiran los factores de x1, Lambda (x2,x3,l) =\n{L2}\n  -> singular (det = {np.linalg.det(L2):.2e}): se pierde el anclaje")

# ---------------------------------------------------------------------------------------------
cab(9, "Preintegración: 10 muestras de IMU a 200 Hz entre dos fotogramas a 20 Hz")
dt = 0.005; n = 10
g = np.array([0, 0, -9.81])
w_m = np.array([0.0, 0.0, 1.0])                  # giróscopo: guiñada a 1 rad/s
a_m = np.array([1.0, 0.0, 9.81])                 # acelerómetro: empuje vertical que compensa g y 1 m/s^2 hacia delante
print(f"  cada muestra: omega = {w_m} rad/s, a = {a_m} m/s^2 (ejes cuerpo), dt = {dt} s; sesgos supuestos nulos")
dR = np.eye(3); dv = np.zeros(3); dpp = np.zeros(3)
for k in range(n):
    dpp = dpp + dv * dt + 0.5 * dR @ a_m * dt**2
    dv = dv + dR @ a_m * dt
    dR = dR @ Exp(w_m * dt)
    if k in (0, 1, 9):
        print(f"  tras la muestra {k+1:2d}: angulo de Delta R = {math.degrees(math.acos((np.trace(dR)-1)/2)):.3f} grados, "
              f"Delta v = {dv}, Delta p = {dpp}")
print(f"  Delta R_ij: giro de {math.degrees(n*dt*1.0):.3f} grados en z; Delta v_ij = {dv}; Delta p_ij = {dpp}")
for nombre, R0, v0 in (("estado A (guiñada 0, quieto)", np.eye(3), np.zeros(3)),
                        ("estado B (guiñada 60 grados, 3 m/s hacia y)", Exp(np.array([0, 0, math.radians(60)])), np.array([0, 3.0, 0]))):
    R = R0.copy(); v = v0.copy(); p = np.zeros(3)
    for k in range(n):                            # integración en el mundo, la que habría que repetir
        acc = R @ a_m + g
        p = p + v * dt + 0.5 * acc * dt**2; v = v + acc * dt; R = R @ Exp(w_m * dt)
    T = n * dt
    dv_rec = R0.T @ (v - v0 - g * T)
    dp_rec = R0.T @ (p - 0 - v0 * T - 0.5 * g * T**2)
    print(f"  {nombre}: v_j = {v}, p_j = {p}")
    print(f"     R_i^T (v_j - v_i - g Dt) = {dv_rec}; R_i^T (p_j - p_i - v_i Dt - g Dt^2/2) = {dp_rec}")
dba = np.array([0.05, 0, 0])
J_dv_ba = np.zeros((3, 3)); dR2 = np.eye(3)
for k in range(n):
    J_dv_ba -= dR2 * dt; dR2 = dR2 @ Exp(w_m * dt)
dv_lin = dv + J_dv_ba @ dba
dR3 = np.eye(3); dv3 = np.zeros(3)
for k in range(n):
    dv3 = dv3 + dR3 @ (a_m - dba) * dt; dR3 = dR3 @ Exp(w_m * dt)
print(f"  sesgo del acelerómetro corregido en {dba}: Delta v reintegrado {dv3}; con el jacobiano {dv_lin}")

# ---------------------------------------------------------------------------------------------
cab(10, "Desfase temporal: de dónde sale f * omega * t_d")
for w, td in ((2.0, 0.010), (0.5, 0.001), (5.0, 0.030)):
    ang = w * td
    exact = FX * math.tan(ang)
    print(f"  omega = {w} rad/s, t_d = {1000*td:.0f} ms: ángulo girado {ang:.4f} rad; "
          f"f*tan = {exact:.2f} px; f*omega*t_d = {FX*w*td:.2f} px")
# punto en el borde de la imagen
uX = (ANCHO - CX) / FX
print(f"  en el borde derecho (X/Z = {uX:.3f}) el desplazamiento a primer orden es f (1 + (X/Z)^2) omega t_d = "
      f"{FX*(1+uX**2)*2*0.01:.2f} px (centro: {FX*2*0.01:.2f})")
v = np.array([2.0, 0, 0]); td = 0.03
print(f"  traslación: v = {v} m/s, t_d = 30 ms -> p(t - t_d) ~ p(t) - v t_d: {v*td} m de corrección")
print(f"  jacobiano de h = p - v t_d: respecto a p, I; respecto a v, -t_d I = {-td}; respecto a t_d, -v = {-v}")

# ---------------------------------------------------------------------------------------------
cab(11, "Cámara de eventos: un borde claro que cruza un píxel y se va")
C = 0.2
I0, I1 = 50.0, 200.0
dL = math.log(I1) - math.log(I0)
print(f"  brillo de {I0} a {I1}: Delta log I = ln({I1}/{I0}) = {dL:.3f}; con C = {C}: floor({dL:.3f}/{C}) = {int(dL//C)} eventos ON")
print(f"  con 10 veces más luz ({10*I0} -> {10*I1}): Delta log I = {math.log(10*I1)-math.log(10*I0):.3f}: los mismos eventos")
print(f"  una cámara de fotogramas vería pasar el valor de {I0} a {I1} (diferencia 150) o de {10*I0} a {10*I1} (1500)")
# rampa de log I en 2 ms (borde que cruza el píxel) y vuelta
T = np.linspace(0, 4e-3, 401)
I2 = 45.0                                         # el borde pasa y el píxel queda algo más oscuro que al principio
Lt = np.log(I0) + dL * np.clip((T - 0.5e-3) / 1.5e-3, 0, 1) - (math.log(I1) - math.log(I2)) * np.clip((T - 2.8e-3) / 0.8e-3, 0, 1)
ref = Lt[0]; evs = []
for t, l in zip(T, Lt):
    while l - ref >= C:
        ref += C; evs.append((t, +1))
    while ref - l >= C:
        ref -= C; evs.append((t, -1))
print(f"  rampa de {I0:.0f} a {I1:.0f} entre 0,5 y 2 ms y de {I1:.0f} a {I2:.0f} entre 2,8 y 3,6 ms: {sum(1 for e in evs if e[1] > 0)} ON y {sum(1 for e in evs if e[1] < 0)} OFF; instantes (ms):",
      ", ".join(f"{1000*t:.2f}{'+' if p > 0 else '-'}" for t, p in evs))
print("  Figura (viewBox 0 0 520 280; t de 0 a 4 ms en x 60..500; log I de 3,8 a 5,4 en y 220..30):")
Xe = lin(0, 4e-3, 60, 500); Ye = lin(3.8, 5.4, 220, 30)
print("path logI:", path(T[::4], Lt[::4], Xe, Ye))
print("  niveles de referencia (y):", ", ".join(f"{Ye(math.log(I0) + k*C):.1f}" for k in range(0, 7)))
print("  eventos (x, polaridad):", ", ".join(f"{Xe(t):.1f}{'+' if p > 0 else '-'}" for t, p in evs))
G, vel = 0.5, 1000.0
rate = G * vel
print(f"  modelo linealizado: d log I/dt = -grad(log I) . velocidad; con gradiente {G} /px y {vel:.0f} px/s -> "
      f"{rate:.0f} /s, un evento cada C/{rate:.0f} = {1000*C/rate:.2f} ms")

# ---------------------------------------------------------------------------------------------
cab(12, "Filtro diferenciable: aprender el ruido de medida de un Kalman 1D por descenso de gradiente")
rng = np.random.default_rng(12)
Tn, q, r_true = 3000, 0.1**2, 0.5**2
x = np.cumsum(rng.normal(0, math.sqrt(q), Tn))
y = x + rng.normal(0, math.sqrt(r_true), Tn)


def kf_mse(log_r):
    r = math.exp(log_r); m, P = 0.0, 1.0; err = 0.0
    for k in range(Tn):
        P = P + q
        K = P / (P + r)
        m = m + K * (y[k] - m); P = (1 - K) * P
        err += (m - x[k]) ** 2
    return err / Tn


th = math.log(2.0**2); lr = 30.0
print(f"  verdad: sigma_medida = 0,5 (r = {r_true}); arranque: sigma = 2,0 (r = 4)")
for it in range(31):
    gr = (kf_mse(th + 1e-4) - kf_mse(th - 1e-4)) / 2e-4
    if it in (0, 1, 2, 5, 10, 20, 30):
        print(f"  iteración {it:2d}: sigma = {math.exp(th/2):.3f}, MSE = {kf_mse(th):.5f}, dMSE/dlog r = {gr:+.5f}")
    th -= lr * gr
rs = np.linspace(math.log(0.1**2), math.log(3.0**2), 200)
best = rs[np.argmin([kf_mse(t) for t in rs])]
print(f"  mínimo por búsqueda en rejilla: sigma = {math.exp(best/2):.3f}")
print(f"  MSE con la sigma verdadera: {kf_mse(math.log(r_true)):.5f}; sin filtrar (y - x): {np.mean((y-x)**2):.5f}")

# ---------------------------------------------------------------------------------------------
cab(13, "Cierre de bucle en un grafo de poses 2D: un cuadrado de 5 m de lado (20 pasos de 1 m) con odometría sesgada")
rng = np.random.default_rng(13)
nseg = 20
true = [np.zeros(2)]
dirs = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
for k in range(nseg):
    true.append(true[-1] + dirs[k // 5])
true = np.array(true)
meas = []
for k in range(nseg):
    d = true[k + 1] - true[k]
    meas.append(d + np.array([0.03, 0.04]) + rng.normal(0, 0.05, 2))   # sesgo de 5 cm por paso más ruido
dead = [np.zeros(2)]
for d in meas:
    dead.append(dead[-1] + d)
dead = np.array(dead)
s_od, s_lc = 0.05, 0.05
# mínimos cuadrados lineales sobre las posiciones x1..x20 (x0 fijo en el origen)
nv = nseg
A = []; b = []; W = []
for k in range(nseg):
    row = np.zeros(nv)
    if k > 0:
        row[k - 1] = -1
    row[k] = 1
    A.append(row); b.append(meas[k]); W.append(1 / s_od**2)
row = np.zeros(nv); row[nseg - 1] = 1                 # cierre: x20 - x0 = 0 (ha vuelto al inicio)
A.append(row); b.append(np.zeros(2)); W.append(1 / s_lc**2)
A = np.array(A); b = np.array(b); W = np.diag(W)
Lg = A.T @ W @ A
sol = np.linalg.solve(Lg, A.T @ W @ b)
opt = np.vstack([np.zeros(2), sol])
# sin cierre
Lo = A[:-1].T @ W[:-1, :-1] @ A[:-1]
sig_dead = np.sqrt(np.diag(np.linalg.inv(Lo)))
sig_opt = np.sqrt(np.diag(np.linalg.inv(Lg)))
e_dead = np.linalg.norm(dead - true, axis=1); e_opt = np.linalg.norm(opt - true, axis=1)
print(f"  error final sin cierre: {e_dead[-1]:.2f} m; error máximo {e_dead.max():.2f} m; RMS {np.sqrt(np.mean(e_dead**2)):.2f} m")
print(f"  con cierre de bucle: error final {e_opt[-1]:.2f} m; error máximo {e_opt.max():.2f} m; RMS {np.sqrt(np.mean(e_opt**2)):.2f} m")
print(f"  sigma por eje de la pose 10 (lado opuesto): sin cierre {sig_dead[9]:.3f} m, con cierre {sig_opt[9]:.3f} m")
print(f"  sigma de la pose 20: sin cierre {sig_dead[19]:.3f} m, con cierre {sig_opt[19]:.3f} m")
print("  Figura (viewBox 0 0 360 300; x de -1,5 a 6,5 m en 40..320; y de -1,5 a 6,5 en 280..20):")
Xg = lin(-1.5, 6.5, 40, 320); Yg = lin(-1.5, 6.5, 280, 20)
print("path verdad:", path(true[:, 0], true[:, 1], Xg, Yg))
print("path sin cierre:", path(dead[:, 0], dead[:, 1], Xg, Yg))
print("path con cierre:", path(opt[:, 0], opt[:, 1], Xg, Yg))
print(f"  punto final sin cierre: ({Xg(dead[-1,0]):.1f}, {Yg(dead[-1,1]):.1f})")

# ---------------------------------------------------------------------------------------------
cab(14, "Función robusta de Huber sobre el residuo al cuadrado s = |r|^2 / sigma^2, umbral kappa = 1")
kap = 1.0
for rr in (0.5, 1.0, 3.0, 10.0):
    s_ = rr**2
    rho = s_ if s_ <= kap**2 else 2 * kap * math.sqrt(s_) - kap**2
    w = 1.0 if s_ <= kap**2 else kap / math.sqrt(s_)
    print(f"  residuo de {rr:4.1f} sigma: cuadrático {s_:6.2f}; Huber {rho:6.2f}; peso rho'(s) = {w:.2f}")
