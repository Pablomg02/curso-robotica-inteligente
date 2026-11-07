"""Ejemplos numéricos de las hojas de B1 (panorama de simuladores).

Uso:  python3 ejemplos_b1.py > salida_ejemplos.txt      (numpy, CPU, menos de un segundo)

Cada ejemplo lleva un número que se cita en la hoja correspondiente («ejemplo 4 del script»).
Las cifras del x500 (masa, kT, c_d, ganancias del controlador) son las de figuras_b1.py; los números
de las cuentas del reloj, del contacto y del amortiguamiento rígido son números de ejemplo, elegidos
para que la cuenta se siga a mano, y así se dice en las hojas.
"""
import math
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# parámetros del x500 (los mismos que figuras_b1.py)
G = 9.81
M = 2.0 + 4 * 0.016076923076923075
JX = 0.02389
KT = 8.54858e-06
CD_ROT = 8.06428e-05
OM_HOVER = math.sqrt(M * G / (4 * KT))
KP, KD, KR, KW = 4.0, 3.5, 8.0, 25.0


def titulo(n, texto):
    print("=" * 78)
    print(f"EJEMPLO {n}. {texto}")


def svgpath(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


# --------------------------------------------------------------------------------------------- 1
titulo(1, "Un paso de simulación: cuántas evaluaciones de la dinámica por paso de control")
f_ctrl, n_sub = 250, 8
h = 1 / (f_ctrl * n_sub)
print(f"  control a {f_ctrl} Hz con {n_sub} subpasos de física: h = {h*1e3:.3f} ms ({1/h:.0f} Hz)")
print(f"  con RK4 (4 evaluaciones por subpaso): {4*n_sub} evaluaciones de la dinámica por paso de control")
print(f"  por segundo simulado: {f_ctrl} llamadas al controlador, {f_ctrl*n_sub} pasos de física, "
      f"{4*f_ctrl*n_sub} evaluaciones")
print(f"  SITL con Gazebo: h = 4 ms -> {1/0.004:.0f} Hz; IMU publicada en cada paso: {1/0.004:.0f} Hz")

# --------------------------------------------------------------------------------------------- 2
titulo(2, "Euler explícito y semiimplícito en el oscilador q'' = -w^2 q (w = 1 rad/s, h = 0,1 s)")
w, h = 1.0, 0.1
q0, v0 = 1.0, 0.0
E = lambda q, v: 0.5 * (w ** 2 * q ** 2 + v ** 2)
# explícito
q1e, v1e = q0 + h * v0, v0 - h * w ** 2 * q0
# semiimplícito: primero v, luego q con la v nueva
v1s = v0 - h * w ** 2 * q0
q1s = q0 + h * v1s
qx, vx = math.cos(w * h), -w * math.sin(w * h)
print(f"  exacto tras un paso:  q = {qx:.6f}, v = {vx:.6f}, E = {E(qx, vx):.6f}")
print(f"  explícito:            q = {q1e:.6f}, v = {v1e:.6f}, E = {E(q1e, v1e):.6f}  "
      f"(E1/E0 = {E(q1e, v1e)/E(q0, v0):.4f} = 1 + z^2)")
print(f"  semiimplícito:        q = {q1s:.6f}, v = {v1s:.6f}, E = {E(q1s, v1s):.6f}")
print(f"  error local en q del explícito: {abs(q1e-qx):.6f}  (h^2 w^2 / 2 = {h*h*w*w/2:.6f})")
z = h * w
Ae = np.array([[1, z], [-z, 1]])
As = np.array([[1 - z * z, z], [-z, 1]])
print(f"  z = h w = {z}; A_e = {Ae.tolist()}, det = {np.linalg.det(Ae):.4f}; "
      f"|autovalor| = {abs(np.linalg.eigvals(Ae)[0]):.6f} = sqrt(1+z^2) = {math.sqrt(1+z*z):.6f}")
print(f"  A_s = {As.tolist()}, det = {np.linalg.det(As):.4f}; |autovalores| = {np.abs(np.linalg.eigvals(As))}")
# varios pasos: deriva de energía
N = 120
st_e = np.array([w * q0, v0])
st_s = np.array([w * q0, v0])
tray_e, tray_s, en_e, en_s = [st_e.copy()], [st_s.copy()], [E(q0, v0)], [E(q0, v0)]
for _ in range(N):
    st_e = Ae @ st_e
    st_s = As @ st_s
    tray_e.append(st_e.copy())
    tray_s.append(st_s.copy())
    en_e.append(0.5 * (st_e @ st_e))
    en_s.append(0.5 * (st_s @ st_s))
en_s = np.array(en_s)
print(f"  tras {N} pasos ({N*h:.0f} s, casi dos vueltas): E/E0 explícito = {en_e[-1]/en_e[0]:.3f} "
      f"(1,01^{N} = {1.01**N:.3f}); semiimplícito entre {en_s.min()/en_s[0]:.4f} y {en_s.max()/en_s[0]:.4f}")
print(f"  radio final en el plano de fase: explícito {math.sqrt(2*en_e[-1]):.3f}; "
      f"semiimplícito {math.sqrt(2*en_s[-1]):.3f}")
print(f"  tras 1000 pasos: explícito E/E0 = {1.01**1000:.3e}")
# figura del plano de fase
X = lin(-2, 2, 20, 320)
Y = lin(-2, 2, 300, 0)   # y hacia arriba
tray_e = np.array(tray_e)
tray_s = np.array(tray_s)
print("  path explícito:", svgpath(tray_e[:, 0], tray_e[:, 1], X, Y))
print("  path semiimplícito:", svgpath(tray_s[:, 0], tray_s[:, 1], X, Y))
print(f"  ejes: x(-2)={X(-2):.0f}, x(0)={X(0):.0f}, x(2)={X(2):.0f}; y(-2)={Y(-2):.0f}, y(0)={Y(0):.0f}, y(2)={Y(2):.0f}")
print(f"  círculo exacto: centro ({X(0):.0f}, {Y(0):.0f}), radio {X(1)-X(0):.0f}")

# --------------------------------------------------------------------------------------------- 3
titulo(3, "Un paso de RK4 a mano en x' = -x (x0 = 1, h = 0,5)")
lam, h, x0 = -1.0, 0.5, 1.0
f = lambda x: lam * x
s1 = f(x0)
s2 = f(x0 + h / 2 * s1)
s3 = f(x0 + h / 2 * s2)
s4 = f(x0 + h * s3)
x1 = x0 + h / 6 * (s1 + 2 * s2 + 2 * s3 + s4)
print(f"  s1 = f(x0) = {s1:.6f}")
print(f"  s2 = f(x0 + h/2 s1) = f({x0 + h/2*s1:.4f}) = {s2:.6f}")
print(f"  s3 = f(x0 + h/2 s2) = f({x0 + h/2*s2:.4f}) = {s3:.6f}")
print(f"  s4 = f(x0 + h s3)   = f({x0 + h*s3:.5f}) = {s4:.6f}")
print(f"  suma ponderada s1 + 2 s2 + 2 s3 + s4 = {s1+2*s2+2*s3+s4:.6f}; x1 = x0 + h/6 * suma = {x1:.6f}")
print(f"  exacto e^-0,5 = {math.exp(-0.5):.6f}; error RK4 = {abs(x1-math.exp(-0.5)):.2e}; "
      f"Euler x1 = {x0 + h*s1:.6f}, error = {abs(x0+h*s1-math.exp(-0.5)):.4f}")
zz = lam * h
R = 1 + zz + zz ** 2 / 2 + zz ** 3 / 6 + zz ** 4 / 24
print(f"  R(z) con z = {zz}: 1 + z + z^2/2 + z^3/6 + z^4/24 = {R:.6f} (igual a x1/x0)")
print(f"  términos: {[1, zz, zz**2/2, zz**3/6, zz**4/24]}; siguiente término de e^z, z^5/120 = {zz**5/120:.2e}")

# --------------------------------------------------------------------------------------------- 4
titulo(4, "Orden global: dividir h entre 2 en x' = -x hasta t = 1")
for met in ("euler", "rk4"):
    errs = []
    for h in (0.1, 0.05, 0.025):
        x = 1.0
        for _ in range(int(round(1 / h))):
            if met == "euler":
                x = x + h * (-x)
            else:
                a1 = -x
                a2 = -(x + h / 2 * a1)
                a3 = -(x + h / 2 * a2)
                a4 = -(x + h * a3)
                x = x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        errs.append(abs(x - math.exp(-1)))
    print(f"  {met:5s}: error con h = 0,1 / 0,05 / 0,025: {errs[0]:.3e} / {errs[1]:.3e} / {errs[2]:.3e}; "
          f"cocientes {errs[0]/errs[1]:.2f} y {errs[1]/errs[2]:.2f}")
# error local frente a global para Euler
h = 0.1
loc = abs((1 - h) - math.exp(-h))
print(f"  Euler, error local de un paso con h = 0,1: {loc:.3e} (h^2/2 = {h*h/2:.3e}); "
      f"10 pasos: suma aproximada 10 x local = {10*loc:.3e}")

# --------------------------------------------------------------------------------------------- 5
titulo(5, "Estabilidad en el oscilador: factor de energía por paso y paso máximo")
for nombre, fz in (("Euler explícito", lambda z: 1 + z * z),
                   ("RK4", lambda z: 1 - z ** 6 / 72 + z ** 8 / 576)):
    for z in (0.1, 0.251, 1.0, 2.0, 2 * math.sqrt(2), 3.0):
        print(f"  {nombre:16s} z = {z:5.3f}: |R(iz)|^2 = {fz(z):.6f}")
# comprobación directa de la fórmula de RK4
for z in (0.5, 2.0):
    Rz = 1 + 1j * z + (1j * z) ** 2 / 2 + (1j * z) ** 3 / 6 + (1j * z) ** 4 / 24
    print(f"  comprobación: |R(i·{z})|^2 = {abs(Rz)**2:.6f} frente a la fórmula {1 - z**6/72 + z**8/576:.6f}")
for z in (0.1, 1.0, 3.0):
    print(f"  Euler implícito   z = {z:5.3f}: |R(iz)|^2 = 1/(1+z^2) = {1/(1+z*z):.6f} (pierde energía con cualquier paso)")
fm = 10.0
wm = 2 * math.pi * fm
print(f"  modo de {fm:.0f} Hz (w = {wm:.1f} rad/s): semiimplícito estable si h < 2/w = {2/wm*1e3:.1f} ms; "
      f"RK4 si h < 2 sqrt2 / w = {2*math.sqrt(2)/wm*1e3:.1f} ms")
z = 0.004 * wm
print(f"  Euler explícito con h = 4 ms: z = {z:.3f}, factor por paso {1+z*z:.4f}, en 250 pasos {(1+z*z)**250:.2e}")

# --------------------------------------------------------------------------------------------- 6
titulo(6, "Amortiguamiento rígido: explícito frente a implícito en v' = -(c/m) v (c/m = 500 1/s)")
a, h = 500.0, 0.01
fe = 1 - h * a
fi = 1 / (1 + h * a)
print(f"  h c/m = {h*a:.1f}; factor por paso: explícito 1 - h c/m = {fe:.3f}; implícito 1/(1 + h c/m) = {fi:.4f}; "
      f"exacto e^(-h c/m) = {math.exp(-h*a):.4f}")
print(f"  tras 5 pasos desde v = 1: explícito {fe**5:.0f}; implícito {fi**5:.2e}; exacto {math.exp(-5*h*a):.2e}")
print(f"  explícito estable solo si h < 2 m/c = {2/a*1e3:.1f} ms")

# --------------------------------------------------------------------------------------------- 7
titulo(7, "Por qué una saturación baja RK4 a orden 2: x' = clip(4 (1 - x), -1, 1), x(0) = 0, hasta t = 2")
# Saturación a 1: x = t hasta que 4 (1 - x) = 1, es decir x = 3/4, en t* = 3/4. Después, x' = 4 (1 - x).
# Los pasos h = 0,1; 0,1/3 y 0,1/9 dejan el codo siempre a mitad de paso (t*/h = 7,5; 22,5; 67,5), así que
# el cociente de errores entre dos pasos mide el orden sin el ruido de dónde cae el codo dentro del paso.
# (Con el codo a 1/3 o 2/3 del paso, la regla de Simpson que lleva dentro RK4 integra el codo sin error:
#  por eso se elige 1/2.)
ts = 0.75
KS = 4.0


def exacta_sat(t):
    return t if t <= ts else 1 - 0.25 * math.exp(-KS * (t - ts))


def f_sat(x):
    return max(-1.0, min(1.0, KS * (1 - x)))


def f_lis(x):
    return KS * (1 - x)


def rk4(fun, x, h):
    a1 = fun(x)
    a2 = fun(x + h / 2 * a1)
    a3 = fun(x + h / 2 * a2)
    a4 = fun(x + h * a3)
    return x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)


def euler(fun, x, h):
    return x + h * fun(x)


print(f"  la saturación se suelta cuando 4 (1 - x) = 1, es decir x = 3/4, en t* = {ts:.4f} s")
print(f"  exacta: x = t hasta t*, después x = 1 - (1/4) e^(-4 (t - t*)); x(2) = {exacta_sat(2):.8f}")
HS = (0.1, 0.1 / 3, 0.1 / 9)
for nombre, fun, exa in (("RK4 con saturación", f_sat, exacta_sat(2.0)),
                         ("RK4 sin saturación (x' = 4(1 - x))", f_lis, 1 - math.exp(-8)),
                         ("Euler con saturación", f_sat, exacta_sat(2.0))):
    met = euler if nombre.startswith("Euler") else rk4
    errs = []
    for h in HS:
        x = 0.0
        n = int(round(2 / h))
        for _ in range(n):
            x = met(fun, x, h)
        errs.append(abs(x - exa))
    pend = [math.log(errs[i] / errs[i + 1]) / math.log(3) for i in range(2)]
    print(f"  {nombre:36s}: error en t = 2 con h = 0,1 / 0,1/3 / 0,1/9: "
          f"{errs[0]:.2e} / {errs[1]:.2e} / {errs[2]:.2e}; cocientes {errs[0]/errs[1]:.1f} y {errs[1]/errs[2]:.1f}; "
          f"pendientes {pend[0]:.2f} y {pend[1]:.2f}")
for h in HS:
    print(f"    h = {h:.5f}: t*/h = {ts/h:.3f} (el codo cae a mitad del paso número {int(ts/h)+1})")
print("  error local de cada paso de RK4 (h = 0,1), arrancando de la solución exacta:")
h = 0.1
for k in range(10):
    t = k * h
    e = abs(rk4(f_sat, exacta_sat(t), h) - exacta_sat(t + h))
    marca = "   <- contiene t*" if t < ts < t + h else ""
    print(f"    paso {k+1:2d} [{t:.1f}, {t+h:.1f}]: {e:.2e}{marca}")
for h in HS:
    k = int(ts / h)
    t0 = k * h
    e_cruce = abs(rk4(f_sat, exacta_sat(t0), h) - exacta_sat(t0 + h))
    t1 = 1.5
    e_suave = abs(rk4(f_sat, exacta_sat(t1), h) - exacta_sat(t1 + h))
    print(f"    h = {h:.5f}: error local del paso con el codo {e_cruce:.2e} (error/h^2 = {e_cruce/h**2:.2e}); "
          f"de un paso suave en t = 1,5: {e_suave:.2e} (error/h^5 = {e_suave/h**5:.2e})")
# misma prueba con el codo a 2/3 del paso: el término h^2 se anula (caso especial de Simpson)
ts2 = 2 / 3
f2 = lambda x: max(-1.0, min(1.0, 3 * (1 - x)))
ex2 = lambda t: t if t <= ts2 else 1 - (1 / 3) * math.exp(-3 * (t - ts2))
e2 = abs(rk4(f2, ex2(0.6), 0.1) - ex2(0.7))
print(f"  caso especial: codo a 2/3 del paso (x' = clip(3(1-x), -1, 1), h = 0,1): error local {e2:.2e}, "
      f"{1.35e-3/e2:.0f} veces menor que con el codo a mitad de paso (1,35e-03): ahí la regla de Simpson "
      f"integra el codo sin error de orden h^2")
# figura: trayectoria con el codo
Xs = lin(0, 2, 40, 380)
Ys = lin(0, 1, 170, 20)
tt = np.linspace(0, 2, 81)
print("  path solución:", svgpath(tt, [exacta_sat(t) for t in tt], Xs, Ys))
print(f"  punto del codo: ({Xs(ts):.1f}, {Ys(ts):.1f}); ejes x(0)={Xs(0):.0f}, x(1)={Xs(1):.0f}, x(2)={Xs(2):.0f}; "
      f"y(0)={Ys(0):.0f}, y(0,5)={Ys(0.5):.0f}, y(1)={Ys(1):.0f}")
print(f"  paso de h = 0,1 que contiene el codo: t = 0,7 -> x = {Xs(0.7):.1f}; t = 0,8 -> x = {Xs(0.8):.1f}")
print("  derivada x' a lo largo del tiempo (para la figura: continua, con un codo en t*):")
Yd = lin(0, 1.05, 170, 20)
print("  path derivada:", svgpath(tt, [min(1.0, KS * (1 - exacta_sat(t))) for t in tt], Xs, Yd))

# --------------------------------------------------------------------------------------------- 8
titulo(8, "El paso frente al motor y al giro del rotor (Gazebo, x500)")
for hh in (0.004, 0.010):
    print(f"  tau = 12,5 ms con h = {hh*1e3:.0f} ms: {12.5/(hh*1e3):.2f} pasos por constante de tiempo")
print(f"  Omega de estacionario = sqrt(m g / (4 kT)) = {OM_HOVER:.1f} rad/s")
for Om in (OM_HOVER, 1000.0, 100.0):
    for hh in (0.004,):
        ang = Om * hh
        ap = (ang + math.pi) % (2 * math.pi) - math.pi
        print(f"  Omega = {Om:6.1f} rad/s, h = 4 ms: avance de la junta por paso {ang:.3f} rad "
              f"({math.degrees(ang):.0f}°); el muestreo ve {ap:+.3f} rad ({math.degrees(ap):+.0f}°)")
print(f"  límite de Nyquist de la junta con h = 4 ms: pi/h = {math.pi/0.004:.0f} rad/s")
print(f"  con rotorVelocitySlowdownSim = 10: Omega = 1000 -> la junta gira a 100 rad/s, "
      f"{100*0.004:.2f} rad por paso; límite real {10*math.pi/0.004:.0f} rad/s")

# --------------------------------------------------------------------------------------------- 9
titulo(9, "Contacto por penalización: rigidez y paso (dron de 2,064 kg apoyado en el suelo)")
for delta in (1e-3, 1e-4):
    k = M * G / delta
    wc = math.sqrt(k / M)
    Tp = 2 * math.pi / wc
    print(f"  penetración en reposo {delta*1e3:.1f} mm: k = m g / delta = {k:,.0f} N/m; "
          f"w_c = sqrt(k/m) = sqrt(g/delta) = {wc:.1f} rad/s; periodo {Tp*1e3:.1f} ms; "
          f"semiimplícito estable si h < 2/w_c = {2/wc*1e3:.2f} ms; diez pasos por periodo: h = {Tp/10*1e3:.1f} ms")
v, hh = 10.0, 0.01
print(f"  a {v:.0f} m/s con h = {hh*1e3:.0f} ms el dron avanza {v*hh*100:.0f} cm por paso")
for d, l in ((0.02, 0.05), (0.02, 0.10), (0.02, 0.5)):
    print(f"    obstáculo de {d*100:.0f} cm y cuerpo de colisión de {l*100:.0f} cm: "
          f"{'puede atravesarlo' if v*hh > d + l else 'se detecta'} (v h = {v*hh*100:.0f} cm frente a d + l = {(d+l)*100:.0f} cm)")

# --------------------------------------------------------------------------------------------- 10
titulo(10, "Discretizar el motor de primer orden: exacto e^(-h/tau) frente a Euler 1 - h/tau")
for tau, hh in ((0.0125, 0.004), (0.0125, 0.010), (0.0125, 0.025), (0.05, 0.005)):
    print(f"  tau = {tau*1e3:5.1f} ms, h = {hh*1e3:4.1f} ms: h/tau = {hh/tau:.3f}; "
          f"e^(-h/tau) = {math.exp(-hh/tau):.4f}; 1 - h/tau = {1-hh/tau:+.4f}")
Ok, Oc, tau = 700.0, 800.0, 0.0125
for hh in (0.004, 0.025):
    ex = Oc + (Ok - Oc) * math.exp(-hh / tau)
    eu = Ok + hh / tau * (Oc - Ok)
    print(f"  Omega_k = {Ok:.0f}, Omega_c = {Oc:.0f} rad/s, h = {hh*1e3:.0f} ms: exacto {ex:.1f}; Euler {eu:.1f}")

# --------------------------------------------------------------------------------------------- 11
titulo(11, "Lazo de rate con motor de primer orden: amortiguamiento (modelo lineal, K_w = 25 1/s)")
for nombre, tau in (("Gazebo al subir", 0.0125), ("Gazebo al bajar", 0.025), ("SIH", 0.05)):
    zeta = 1 / (2 * math.sqrt(KW * tau))
    wn = math.sqrt(KW / tau)
    sob = math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2)) * 100 if zeta < 1 else 0.0
    fase = math.degrees(math.atan(KW * tau))
    print(f"  {nombre:16s} tau = {tau*1e3:5.1f} ms: K_w tau = {KW*tau:.4f}; zeta = 1/(2 sqrt(K_w tau)) = {zeta:.3f}; "
          f"w_n = {wn:.1f} rad/s; sobreoscilación del rate {sob:.1f} %; retraso de fase a 25 rad/s {fase:.1f}°")
# comprobación por simulación del modelo lineal tau w'' + w' + K w = K w_d
for tau in (0.0125, 0.05):
    dt = 1e-5
    wv, wd_, wdot = 0.0, 1.0, 0.0
    mx = 0.0
    for _ in range(int(1.0 / dt)):
        wdd = (KW * (wd_ - wv) - wdot) / tau
        wdot += dt * wdd
        wv += dt * wdot
        mx = max(mx, wv)
    print(f"    simulación del modelo lineal, tau = {tau*1e3:.1f} ms: máximo del rate {mx:.4f} -> sobreoscilación {(mx-1)*100:.1f} %")

# lazo de actitud encima (K_R = 8), modelo lineal de un eje: phi' = w, w' = a, tau a' = K_w (K_R (phi_ref - phi) - w) - a
print("  lazo de actitud (K_R = 8) sobre el de rate, modelo lineal de un eje, escalón unitario:")
zeta_a = KW / (2 * math.sqrt(KW * KR))
print(f"    con tau = 0: phi'' + K_w phi' + K_w K_R phi = K_w K_R phi_ref -> zeta = K_w / (2 sqrt(K_w K_R)) = {zeta_a:.3f}; "
      f"sobreoscilación {math.exp(-math.pi*zeta_a/math.sqrt(1-zeta_a**2))*100:.2f} %")
for tau in (0.0, 0.0125, 0.05):
    dt = 1e-5
    phi = wv = acc = 0.0
    hist = []
    for _ in range(int(1.5 / dt)):
        ac = KW * (KR * (1 - phi) - wv)
        acc = ac if tau == 0 else acc + dt * (ac - acc) / tau
        wv += dt * acc
        phi += dt * wv
        hist.append(phi)
    hist = np.array(hist)
    fuera = np.where(np.abs(hist - 1) > 0.02)[0]
    print(f"    tau = {tau*1e3:4.1f} ms: sobreoscilación {(hist.max()-1)*100:5.2f} %; establecimiento (2 %) {fuera[-1]*dt*1e3:.0f} ms")
print("    (la simulación completa de figuras_b1.py da 0,2 % y 320 ms con empuje instantáneo; 0 % y 332 ms con Gazebo;")
print("     4,0 % y 576 ms con SIH, cuya planta completa añade, entre otras cosas, el amortiguamiento angular K_dw)")

# --------------------------------------------------------------------------------------------- 12
titulo(12, "Arrastre en crucero: inclinación y error del PD (x500, 5 m/s)")
v = 5.0
D_gz = 4 * CD_ROT * OM_HOVER
for nombre, D in (("Gazebo, 4 c_d Omega_h", D_gz), ("SIH, K_dv", 1.0)):
    th = math.degrees(math.atan(D * v / (M * G)))
    e = D * v / (M * KP)
    print(f"  {nombre:22s}: D = {D:.4f} N/(m/s); D v = {D*v:.3f} N; m g = {M*G:.3f} N; "
          f"tan(theta) = {D*v/(M*G):.4f} -> theta = {th:.2f}°; e = D v / (m K_p) = {e*100:.1f} cm; D/m = {D/M:.3f} 1/s")
print(f"  Omega_h = {OM_HOVER:.1f} rad/s; c_d = {CD_ROT:.4e}")

# --------------------------------------------------------------------------------------------- 13
titulo(13, "Lockstep: factor de tiempo real con números de ejemplo (h = 4 ms simulados)")
h = 0.004
print(f"  solo el simulador, c_sim = 1 ms: factor de tiempo real h / c_sim = {h/0.001:.2f}")
for c_sim, c_px4 in ((0.001, 0.0005), (0.006, 0.0005)):
    pared = c_sim + c_px4
    print(f"  c_sim = {c_sim*1e3:.1f} ms, c_PX4 = {c_px4*1e3:.1f} ms: {pared*1e3:.1f} ms de pared por paso; "
          f"factor de tiempo real h / (c_sim + c_PX4) = {h/pared:.2f}")
pared = 0.0065
retraso = (pared - h) * 100
print(f"  en tiempo real estricto con 6,5 ms por paso: cada paso llega {(pared-h)*1e3:.1f} ms tarde; "
      f"tras 100 pasos el simulador va {retraso*1e3:.0f} ms por detrás (o se saltan pasos)")

# --------------------------------------------------------------------------------------------- 14
titulo(14, "Física en GPU: rendimiento por entorno y tiempo para 10^9 pasos (cifras de cada artículo)")
cifras = [("gym-pybullet-drones, 1 dron, CPU", 240 * 16.8, 1),
          ("Flightmare, 150 drones, CPU", 2.0e5, 150),
          ("OmniDrones, 4096 entornos, RTX 4090", 732109, 4096),
          ("Aerial Gym, 2^16 robots, RTX 3090", 4.43e6, 2 ** 16)]
for nombre, tot, n in cifras:
    t = 1e9 / tot
    if t > 86400:
        ts_ = f"{t/86400:.1f} días"
    elif t > 3600:
        ts_ = f"{t/3600:.1f} h"
    else:
        ts_ = f"{t/60:.1f} min"
    print(f"  {nombre:38s}: {tot:12,.0f} pasos/s en total; {tot/n:8.1f} por entorno; 10^9 pasos en {t:10,.0f} s = {ts_}")

# --------------------------------------------------------------------------------------------- 15
titulo(15, "Qué es un paso: decimation (control a 250 Hz con 8 subpasos de física)")
fis = 1.0e6
print(f"  un simulador que hace {fis:,.0f} pasos de física por segundo da {fis/8:,.0f} pasos de control por segundo")

# --------------------------------------------------------------------------------------------- 16
titulo(16, "Datos de figuras: qué hace un integrador (Euler en x' = x, h = 0,5, de 0 a 2)")
Xi = lin(0, 2, 50, 410)
Yi = lin(0, 8, 230, 30)
tt = np.linspace(0, 2, 41)
print("  path exacta e^t:", svgpath(tt, np.exp(tt), Xi, Yi))
xe = [1.0]
for _ in range(4):
    xe.append(xe[-1] * 1.5)
for k, x in enumerate(xe):
    print(f"  Euler t = {k*0.5:.1f}: x = {x:.4f} -> ({Xi(k*0.5):.1f}, {Yi(x):.1f}); exacta {math.exp(k*0.5):.4f} -> y = {Yi(math.exp(k*0.5)):.1f}")
x15 = math.exp(1.5)
print(f"  paso local desde la exacta en t = 1,5: x = {x15:.4f} -> Euler {x15*1.5:.4f} (y = {Yi(x15*1.5):.1f}); "
      f"exacta en 2: {math.exp(2):.4f} (y = {Yi(math.exp(2)):.1f}); error local {math.exp(2)-1.5*x15:.3f}")
print(f"  error global en t = 2: {math.exp(2) - xe[-1]:.3f}; error local del primer paso: {math.exp(0.5)-1.5:.3f}")
print(f"  ejes: x(0)={Xi(0):.0f}, x(1)={Xi(1):.0f}, x(2)={Xi(2):.0f}; y(0)={Yi(0):.0f}, y(4)={Yi(4):.0f}, y(8)={Yi(8):.0f}")

# --------------------------------------------------------------------------------------------- 17
titulo(17, "Datos de figuras: la rueda de carro (junta del rotor muestreada cada h = 4 ms)")
cx, cy, r = 90, 100, 62
for nombre, Om in (("Omega = 1000 rad/s", 1000.0), ("con slowdown 10 (100 rad/s)", 100.0)):
    pts = []
    for k in range(4):
        a = k * Om * 0.004
        pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
    print(f"  {nombre}: " + "; ".join(f"k={k}: ({x:.1f}, {y:.1f})" for k, (x, y) in enumerate(pts)))

# --------------------------------------------------------------------------------------------- 18
titulo(18, "Datos de figuras: diagrama de fuerzas en crucero (SIH, theta = 13,9°)")
th = math.atan(1.0 * 5.0 / (M * G))
L = 110
ox, oy = 170, 150
print(f"  theta = {math.degrees(th):.2f}°; empuje desde ({ox}, {oy}) hasta ({ox + L*math.sin(th):.1f}, {oy - L*math.cos(th):.1f})")
print(f"  componentes del empuje: horizontal {L*math.sin(th):.1f} px, vertical {L*math.cos(th):.1f} px")
