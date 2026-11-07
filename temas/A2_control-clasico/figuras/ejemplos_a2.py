"""Ejemplos numéricos y datos de las figuras nuevas de A2 (control clásico y jerarquía de lazos).

Cada número de un ejemplo de la presentación sale de aquí; las cifras de las tablas y
figuras originales salen de figuras_a2.py (no hace falta relanzarlo).
Ejecutar:  python3 ejemplos_a2.py > salida_ejemplos.txt   (solo numpy, unos segundos en CPU)
"""
import math
import numpy as np

G = 9.81
np.set_printoptions(precision=4, suppress=True)


def titulo(n, s):
    print("\n" + "=" * 72 + f"\nEjemplo {n}. {s}\n" + "=" * 72)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


# ---------------------------------------------------------------------------
# Datos comunes del x500 (los mismos que figuras_a2.py, sección 6)
# ---------------------------------------------------------------------------
JX = 0.02389
KT = 8.54858e-06
M_X500 = 2.0 + 4 * 0.016076923076923075
ARM = 0.174
OM_MIN, OM_MAX = 150.0, 1000.0
P_, I_, D_, K_ = 0.15, 0.2, 0.003, 1.0
F_GYRO, F_DGYRO = 40.0, 20.0


def lazo_rate(w, tau_m=0.0125, f_loop=250, extra=0.0):
    """El mismo lazo abierto que figuras_a2.py: PID de PX4, filtros, motor, planta y retardo 1,5 T."""
    s = 1j * w
    wg, wdg = 2 * math.pi * F_GYRO, 2 * math.pi * F_DGYRO
    Fv = wg * wg / (s * s + math.sqrt(2) * wg * s + wg * wg)
    Fd = 1 / (1 + s / wdg)
    delay = np.exp(-s * (1.5 / f_loop + extra))
    C = K_ * ((P_ + I_ / s) * Fv + D_ * s * Fd)
    return C * B / s / (1 + tau_m * s) * delay


# ---------------------------------------------------------------------------
titulo(1, "La planta de alabeo del x500: de dónde sale b")
peso = M_X500 * G
OM0 = math.sqrt(peso / (4 * KT))
U0 = (OM0 - OM_MIN) / (OM_MAX - OM_MIN)
T0 = KT * OM0**2
DT_DOM = 2 * KT * OM0
DT_DU = DT_DOM * (OM_MAX - OM_MIN)
MIX = 1 / math.sqrt(2)
B = 4 * ARM * MIX * DT_DU / JX
U_SAT = (1 - U0) / MIX
print(f"  masa {M_X500:.3f} kg, peso {peso:.2f} N, empuje por motor en estacionario {T0:.3f} N")
print(f"  Omega0 = sqrt(m g / (4 kT)) = {OM0:.1f} rad/s")
print(f"  salida de motor en estacionario u0 = (Omega0 - 150)/850 = {U0:.4f}")
print(f"  dT/dOmega = 2 kT Omega0 = {DT_DOM:.5f} N s/rad;  dT/du = dT/dOmega * 850 = {DT_DU:.2f} N")
print(f"  par normalizado u: dos motores suben u/sqrt2, dos bajan u/sqrt2; brazo {ARM} m")
print(f"  par de alabeo por unidad de u: 4 * {ARM} * (1/sqrt2) * {DT_DU:.2f} = {4*ARM*MIX*DT_DU:.3f} N m")
print(f"  b = ese par / Jx = {4*ARM*MIX*DT_DU:.3f} / {JX} = {B:.1f} rad/s^2")
print(f"  saturación a empuje constante: u0 + u/sqrt2 = 1 -> u = {U_SAT:.4f} -> {U_SAT*B:.0f} rad/s^2")
print(f"  si el asignador baja el colectivo a 0,5: 0,5 + u/sqrt2 = 1 -> u = {0.5/MIX:.4f}")

# ---------------------------------------------------------------------------
titulo(2, "Forma ideal y forma paralela con las ganancias de PX4")
Kp, Ki, Kd = K_ * P_, K_ * I_, K_ * D_
Ti, Td = Kp / Ki, Kd / Kp
print(f"  paralela (MC_ROLLRATE_K = 1): Kp = {Kp}, Ki = {Ki} 1/s, Kd = {Kd} s")
print(f"  ideal equivalente: K = Kp = {Kp}, Ti = Kp/Ki = {Ti:.3f} s, Td = Kd/Kp = {Td:.4f} s")
e0 = 0.2
print(f"  error constante {e0} rad/s: término P = {Kp*e0:.3f}; el I iguala al P a los t = Ti = {Ti:.2f} s"
      f" (I(t) = Ki e0 t = {Ki*e0*Ti:.3f})")
# Un instante del lazo de rate
w_sp, w_m, int_e, wdot = 1.0, 0.8, 0.1, 5.0
tP, tI, tD = P_ * (w_sp - w_m), I_ * int_e, -D_ * wdot
print(f"  instante: omega_sp = {w_sp}, omega = {w_m} (e = {w_sp-w_m:.1f} rad/s), integral del error {int_e} rad,"
      f" omega_dot = {wdot} rad/s^2, FF = 0")
print(f"    P e = {tP:.3f}; I int = {tI:.3f}; -D omega_dot = {tD:.3f}; tau_sp = K (suma) = {K_*(tP+tI+tD):.3f}")

dpert = 0.3 / JX
ess = dpert / (B * Kp)
print(f"  solo P ante el par de 0,3 N m (centro de masas desplazado 1,5 cm): omega' = b u + {dpert:.2f} = 0 en régimen"
      f" -> u = -{dpert/B:.4f} -> error e = {dpert/B:.4f}/Kp = {ess:.3f} rad/s = {math.degrees(ess):.0f} °/s; con I, 0")

# ---------------------------------------------------------------------------
titulo(3, "Derivada filtrada y derivative kick")
wd = 2 * math.pi * F_DGYRO
print(f"  filtro de la D a {F_DGYRO} Hz: omega_d = {wd:.1f} rad/s; ganancia máxima Kd omega_d = {Kd*wd:.3f}")
for f in (1, 10, 20, 100, 400):
    w_ = 2 * math.pi * f
    pura = Kd * w_
    filt = Kd * w_ / math.sqrt(1 + (w_ / wd) ** 2)
    print(f"    {f:4d} Hz: |Kd s| = {pura:7.4f}, |Kd s/(1+s/wd)| = {filt:6.4f}   (P = {Kp})")
for fl in (250, 800):
    T = 1 / fl
    al = 1 / (wd * T)
    pico = Kd / (T * (1 + al))
    print(f"  escalón de 1 rad/s en la consigna con D sobre el error, lazo a {fl} Hz:"
          f" primer D_k = Kd/(T(1+alpha)) = {pico:.3f} (saturación del par: {U_SAT:.3f})")
print(f"  continuo: pico Kd omega_d * 1 = {Kd*wd:.3f}; con D sobre la medida el pico es 0")

# ---------------------------------------------------------------------------
titulo(4, "Discretización: integrador y filtro de la derivada")
for fl in (250, 800):
    T = 1 / fl
    al = 1 / (wd * T)
    a_px4 = T / (1 / wd + T)
    print(f"  {fl} Hz: T = {1000*T:.2f} ms, alpha = 1/(omega_d T) = {al:.3f}; 1/(1+alpha) = {1/(1+al):.4f}"
          f" = AlphaFilter dt/(tau+dt) = {a_px4:.4f}")
T = 1 / 800
al = 1 / (wd * T)
print(f"  integrador a 800 Hz con e = {e0} constante: cada muestra suma Ki T e = {Ki*T*e0:.2e}; en 1 s, {Ki*e0:.3f}")
# derivada sobre la medida ante una rampa de y de 10 rad/s^2 que empieza en k = 0
yk = [0.0] + [10.0 * T * k for k in range(1, 9)]
Dk = 0.0
out = []
for k in range(1, 9):
    Dk = (al * Dk - Kd * (yk[k] - yk[k - 1]) / T) / (1 + al)
    out.append(Dk)
print("  rampa de la medida a 10 rad/s^2 (800 Hz): D_k =", " ".join(f"{d:.4f}" for d in out),
      f"-> tiende a -Kd*10 = {-Kd*10:.3f}")

# ---------------------------------------------------------------------------
titulo(5, "Retenedor de orden cero: medio periodo de retardo")
for fc, fs in ((6.3, 250), (6.3, 800), (6.3, 50), (2.0, 30)):
    print(f"  cruce {fc} Hz con lazo a {fs} Hz: fase del retenedor omega_c T/2 = {180*fc/fs:.1f}°")
print(f"  regla: omega_c T/2 <= 10° = {math.radians(10):.4f} rad -> f_s >= pi/{math.radians(10):.4f} f_c"
      f" = {math.pi/math.radians(10):.1f} f_c")
# figura: senoide de 1 Hz muestreada a 10 Hz (para que se vea), en [0, 1,5] s
f0, fsz = 1.0, 10.0
tz = np.linspace(0, 1.5, 601)
yc = np.sin(2 * math.pi * f0 * tz)
yh = np.sin(2 * math.pi * f0 * np.floor(tz * fsz) / fsz)
ysh = np.sin(2 * math.pi * f0 * (tz - 0.5 / fsz))
XZ = lin(0, 1.5, 50, 500)
YZ = lin(-1.2, 1.2, 230, 20)
print("  figura ZOH: viewBox 0 0 520 260; t 0..1,5 s -> 50..500; y -1,2..1,2 -> 230..20")
print("path zoh continua:", path(tz, yc, XZ, YZ))
# escalera: vértices exactos
xs_, ys_ = [], []
for k in range(int(1.5 * fsz)):
    v = math.sin(2 * math.pi * f0 * k / fsz)
    xs_ += [k / fsz, (k + 1) / fsz]
    ys_ += [v, v]
print("path zoh escalera:", path(xs_, ys_, XZ, YZ))
print("path zoh retrasada T/2:", path(tz, ysh, XZ, YZ))
print("  muestras (círculos):", " ".join(f"{XZ(k/fsz):.1f},{YZ(math.sin(2*math.pi*k/fsz)):.1f}" for k in range(16)))

# ---------------------------------------------------------------------------
titulo(6, "Anti-windup en un lazo lento: velocidad horizontal con la inclinación limitada")
# Lazo PI de velocidad (P 1,8, I 0,4 de PX4) sobre un integrador v' = a, con a saturada por 45° de inclinación.
PV, IV = 1.8, 0.4
AMAX = G * math.tan(math.radians(45))
TT = 0.5
VSP = 12.0
print(f"  v' = sat(u), |a| <= g tan 45° = {AMAX:.2f} m/s^2; u = {PV} e + I; consigna 0 -> {VSP} m/s")
print(f"  remedios: ninguno, integración condicional, cálculo hacia atrás con Tt = {TT} s (elección nuestra)")


def sim_aw(modo, T=12.0, h=1e-3):
    v = integ = 0.0
    ts, vs, Is, us = [], [], [], []
    for k in range(int(T / h) + 1):
        t = k * h
        e = VSP - v
        u = PV * e + integ
        a = max(-AMAX, min(AMAX, u))
        if modo == "sin":
            dI = IV * e
        elif modo == "condicional":
            dI = 0.0 if (u != a and e * u > 0) else IV * e
        else:
            dI = IV * e + (a - u) / TT
        if k % 20 == 0:
            ts.append(t); vs.append(v); Is.append(integ); us.append(u)
        integ += h * dI
        v += h * a
    return np.array(ts), np.array(vs), np.array(Is), np.array(us)


res_aw = {}
for modo in ("sin", "condicional", "atras"):
    t, v, Iv, u = sim_aw(modo)
    res_aw[modo] = (t, v, Iv)
    k_sat = np.where(np.abs(u) > AMAX + 1e-9)[0]
    t_sat = t[k_sat[-1]] if len(k_sat) else 0.0
    sob = 100 * (v.max() - VSP) / VSP
    i2 = np.where(np.abs(v - VSP) > 0.02 * VSP)[0]
    t2 = t[i2[-1]] if len(i2) else 0.0
    print(f"  {modo:12s}: satura hasta t = {t_sat:.2f} s; integrador máximo {Iv.max():5.2f} m/s^2;"
          f" velocidad máxima {v.max():5.2f} m/s (sobrepaso {sob:4.1f} %); dentro del 2 % desde t = {t2:.2f} s")
XA = lin(0, 10, 60, 500)
YAv = lin(0, 16, 140, 20)
YAi = lin(-4, 4, 280, 170)
print("  figura anti-windup: viewBox 0 0 520 300; t 0..10 s -> 60..500; v 0..16 -> 140..20; I -4..4 -> 280..170")
for modo in res_aw:
    t, v, Iv = res_aw[modo]
    kk = t <= 10
    print(f"path aw v {modo}:", path(t[kk], v[kk], XA, YAv))
    print(f"path aw I {modo}:", path(t[kk], np.clip(Iv[kk], -4, 4), XA, YAi))
    print(f"  {modo}: integrador mínimo {Iv.min():.2f} m/s^2; integrador en t = 0,8 s: {Iv[np.argmin(np.abs(t-0.8))]:.2f}")

# ---------------------------------------------------------------------------
titulo(7, "Anti-windup del lazo de rate de PX4 en el x500")
err = math.radians(200)
fac = max(0, 1 - (200 / 400) ** 2)
dI = I_ * fac * err * 0.05
print(f"  error 200 °/s = {err:.3f} rad/s; factor 1-(200/400)^2 = {fac:.2f}")
print(f"  carga en 50 ms: I * factor * e * dt = {I_} * {fac:.2f} * {err:.3f} * 0,05 = {dI:.4f}")
print(f"  frente al límite MC_RR_INT_LIM 0,3 y a la saturación {U_SAT:.2f}: {100*dI/U_SAT:.0f} % de la saturación")
for e_deg in (50, 100, 200, 300, 400):
    print(f"    factor con error {e_deg} °/s: {max(0, 1-(e_deg/400)**2):.3f}")

# ---------------------------------------------------------------------------
titulo(8, "Sensibilidad S y sensibilidad complementaria T del lazo de rate del x500 (base)")
for f in (0.5, 1, 2, 6.3, 20, 50):
    L = lazo_rate(np.array([2 * math.pi * f]))[0]
    S, Tc = 1 / (1 + L), L / (1 + L)
    print(f"  {f:5.1f} Hz: |L| = {abs(L):8.3f}, |S| = {abs(S):.3f}, |T| = {abs(Tc):.3f}")
wq = np.logspace(-2, 5.5, 600000)
Lq = lazo_rate(wq)
Sq = np.abs(1 / (1 + Lq))
ims = np.argmax(Sq)
print(f"  pico de |S| (Ms) = {Sq[ims]:.3f} a {wq[ims]/2/math.pi:.1f} Hz; 1/Ms = {1/Sq[ims]:.3f} = distancia mínima de L a -1")
lnS = np.log(Sq)
integral = np.trapezoid(lnS, wq)
neg = np.trapezoid(np.minimum(lnS, 0), wq)
pos = np.trapezoid(np.maximum(lnS, 0), wq)
print(f"  integral de Bode hasta 3e5 rad/s: área negativa {neg:.1f}, positiva {pos:.1f}, total {integral:.2f} rad/s (debe ser ~0)")
# Figura colchón de agua: ln|S| frente a omega lineal 0..400 rad/s
wl = np.linspace(0.5, 400, 800)
lS = np.log(np.abs(1 / (1 + lazo_rate(wl))))
XW = lin(0, 400, 60, 500)
YW = lin(-3, 1, 230, 30)
print("  figura colchón: viewBox 0 0 520 270; omega 0..400 rad/s -> 60..500; ln|S| -3..1 -> 230..30")
print("path lnS:", path(wl, np.clip(lS, -3, 1), XW, YW))
# área positiva y negativa en el tramo dibujado
print(f"  en 0..400 rad/s: negativa {np.trapezoid(np.minimum(lS,0), wl):.1f}, positiva {np.trapezoid(np.maximum(lS,0), wl):.1f}")
cero = wl[np.where((lS[:-1] < 0) & (lS[1:] >= 0))[0][0]]
print(f"  |S| cruza 1 (ln = 0) en {cero:.1f} rad/s = {cero/2/math.pi:.1f} Hz")

# ---------------------------------------------------------------------------
titulo(9, "Nyquist del lazo del x500: margen de fase y de ganancia como distancias a -1")
wn_ = np.logspace(np.log10(15), 3, 1500)
Ln = lazo_rate(wn_)
k = (np.abs(Ln.real) < 2.2) & (np.abs(Ln.imag) < 2.2)
XN = lin(-2.2, 1.0, 40, 360)
YN = lin(-2.2, 1.0, 290, -30)
print("  figura Nyquist: viewBox 0 0 400 300; Re -2,2..1 -> 40..360; Im 1..-2,2 -> -30..290 (recortado en el SVG)")
print("path nyquist:", path(Ln.real[k], Ln.imag[k], XN, YN))
mag = np.abs(Ln)
i = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0][0]
Lc = Ln[i]
print(f"  cruce |L| = 1 en {wn_[i]/2/math.pi:.2f} Hz: L = {Lc.real:.3f} {Lc.imag:+.3f} j; ángulo desde -1: "
      f"{180+math.degrees(np.angle(Lc)):.1f}° -> punto ({XN(Lc.real):.1f}, {YN(Lc.imag):.1f})")
ph = np.unwrap(np.angle(Ln))
j = np.where((ph[:-1] > -math.pi) & (ph[1:] <= -math.pi))[0][0]
Lp = Ln[j]
print(f"  fase -180° en {wn_[j]/2/math.pi:.1f} Hz: L = {Lp.real:.3f}; MG = 1/|L| = {1/abs(Lp):.2f} = {20*math.log10(1/abs(Lp)):.1f} dB"
      f" -> punto ({XN(Lp.real):.1f}, {YN(0):.1f})")
print(f"  -1 en ({XN(-1):.1f}, {YN(0):.1f}); origen en ({XN(0):.1f}, {YN(0):.1f}); radio unidad = {XN(0)-XN(-1):.1f} px")

# ---------------------------------------------------------------------------
titulo(10, "Retardo y cruce máximo: la cuenta paso a paso")
for tau_ms, pm in ((5, 45), (20, 45), (5, 60)):
    tau = tau_ms / 1000
    wc = math.radians(90 - pm) / tau
    print(f"  tau = {tau_ms} ms, MF {pm}°: omega_c tau = {math.radians(90-pm):.4f} rad -> omega_c = {wc:.1f} rad/s"
          f" -> f_c = {wc/2/math.pi:.1f} Hz")
print(f"  x500: MF 71° en el cruce de 6,3 Hz -> retardo extra que aguanta = {math.radians(71)/(2*math.pi*6.3)*1000:.1f} ms")
print(f"  (con los valores exactos del script de figuras: 71,0° y 39,6 rad/s -> {1000*math.radians(71.0)/39.6:.1f} ms)")

# ---------------------------------------------------------------------------
titulo(11, "Retardo de un filtro: fase exacta frente a la aproximación lineal")
for nombre, fc, tipo in (("Butterworth 2.º orden", 40.0, "bw"), ("PT1", 20.0, "pt1")):
    wf = 2 * math.pi * fc
    td = (math.sqrt(2) if tipo == "bw" else 1.0) / wf
    for f in (2, 6.3, 10, 20):
        w_ = 2 * math.pi * f
        if tipo == "bw":
            ex = -math.degrees(math.atan2(math.sqrt(2) * wf * w_, wf * wf - w_ * w_))
        else:
            ex = -math.degrees(math.atan(w_ / wf))
        ap = -math.degrees(w_ * td)
        print(f"  {nombre} a {fc:.0f} Hz, f = {f:4.1f} Hz: fase exacta {ex:6.2f}°, aproximación -omega t_d = {ap:6.2f}°")
    print(f"    retardo equivalente t_d = {1000*td:.2f} ms")
fx = np.linspace(0, 40, 161)
wx = 2 * math.pi * fx
wf = 2 * math.pi * 40
ex_bw = -np.degrees(np.arctan2(math.sqrt(2) * wf * wx, wf * wf - wx * wx))
ap_bw = -np.degrees(wx * math.sqrt(2) / wf)
XF = lin(0, 40, 60, 500)
YF = lin(-100, 0, 250, 30)
print("  figura filtro: viewBox 0 0 520 290; f 0..40 Hz -> 60..500; fase -100..0° -> 250..30")
print("path bw exacta:", path(fx, ex_bw, XF, YF))
kk = ap_bw >= -100
print("path bw lineal:", path(fx[kk], ap_bw[kk], XF, YF))
print(f"  marca 6,3 Hz: x = {XF(6.3):.1f}")

# ---------------------------------------------------------------------------
titulo(12, "Separación de escalas: dos casos a mano")
for k_, wi in ((10.0, 40.0), (20.0, 40.0), (4.0, 39.6)):
    r = wi / k_
    z = 0.5 * math.sqrt(r)
    wn = math.sqrt(k_ * wi)
    print(f"  k = {k_}, omega_i = {wi}: r = {r:.2f}, omega_n = sqrt(k omega_i) = {wn:.2f} rad/s, zeta = {z:.3f}")

# ---------------------------------------------------------------------------
titulo(13, "De la aceleración deseada a la actitud deseada (x500, ENU)")
a_sp = np.array([3.0, 0.0, 0.0])
Tsp = M_X500 * (a_sp + G * np.array([0, 0, 1.0]))
zd = Tsp / np.linalg.norm(Tsp)
psi_d = 0.0
xc = np.array([math.cos(psi_d), math.sin(psi_d), 0.0])
yd = np.cross(zd, xc); yd /= np.linalg.norm(yd)
xd = np.cross(yd, zd)
Rd = np.column_stack([xd, yd, zd])
print(f"  a_sp = {a_sp} m/s^2 -> T_sp = m (a_sp + g e3) = {Tsp} N, |T_sp| = {np.linalg.norm(Tsp):.2f} N")
print(f"  z_d = {zd}; inclinación = acos(z_d . e3) = {math.degrees(math.acos(zd[2])):.1f}°")
print(f"  y_d = {yd}; x_d = {xd}")
print("  R_d =\n", Rd)
print(f"  det R_d = {np.linalg.det(Rd):.3f}")
print(f"  límite de 45°: aceleración horizontal máxima manteniendo altura = g tan 45° = {G*math.tan(math.radians(45)):.2f} m/s^2")
print(f"  empuje colectivo si el dron aún está nivelado (R = I): T = T_sp . e3 = {Tsp[2]:.2f} N")

# ---------------------------------------------------------------------------
titulo(14, "Error de actitud: geométrico y de cuaternión")


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def Exp(phi):
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.eye(3)
    K = hat(phi / th)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


for deg in (30, 90, 170):
    th = math.radians(deg)
    Rerr = Exp(th * np.array([1.0, 0, 0]))         # R_d^T R
    Psi = 0.5 * np.trace(np.eye(3) - Rerr)
    eR = 0.5 * np.array([(Rerr - Rerr.T)[2, 1], (Rerr - Rerr.T)[0, 2], (Rerr - Rerr.T)[1, 0]])
    print(f"  giro de {deg}° en x: tr = {np.trace(Rerr):.4f} (1+2cos = {1+2*math.cos(th):.4f}); Psi = {Psi:.4f}"
          f" (1-cos = {1-math.cos(th):.4f}); e_R = {eR} (sen = {math.sin(th):.4f}); 2 sen(th/2) = {2*math.sin(th/2):.4f}")
th = math.radians(30)
qe = np.array([math.cos(th / 2), math.sin(th / 2), 0, 0])
Kq = 4.0
print(f"  error de 30° en alabeo: q_e = {qe}; 2 q_e,v = {2*qe[1:]}; omega_sp = K o 2 q_e,v = {Kq*2*qe[1:]} rad/s (K = 4)")
print(f"  con -q_e (q_e,w < 0): omega_sp = {-Kq*2*qe[1:]} rad/s: gira al revés y recorre {360-30}°")

# ---------------------------------------------------------------------------
titulo(15, "Prioridad a la inclinación: separar inclinación y guiñada")


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def qaxis(n, a):
    n = np.asarray(n, float) / np.linalg.norm(n)
    return np.concatenate([[math.cos(a / 2)], math.sin(a / 2) * n])


q = np.array([1.0, 0, 0, 0])                           # actitud actual: nivelado, rumbo 0
qd = qmul(qaxis([0, 0, 1], math.radians(60)), qaxis([1, 0, 0], math.radians(20)))   # ZYX: guiñada 60°, alabeo 20°
z = q2R(q)[:, 2]
zd = q2R(qd)[:, 2]
ang = math.acos(np.clip(z @ zd, -1, 1))
ax = np.cross(z, zd); ax /= np.linalg.norm(ax)
qred_rel = qaxis(ax, ang)                       # giro mínimo de z a z_d, en ejes mundo
qd_red = qmul(qred_rel, q)
qyaw = qmul(qinv(qd_red), qd)
psi_e = 2 * math.atan2(qyaw[3], qyaw[0])
print(f"  actual: nivelado, rumbo 0. Deseada: guiñada 60°, alabeo 20° (ZYX). z_d = {zd}")
print(f"  giro mínimo z -> z_d: eje {ax}, ángulo {math.degrees(ang):.1f}°")
print(f"  resto de guiñada q_d,red^-1 q_d = {qyaw} -> psi_e = {math.degrees(psi_e):.1f}°")
w_y = 0.4
print(f"  con w = {w_y}: PX4 pide ahora {w_y*math.degrees(psi_e):.1f}° de guiñada además de la inclinación completa")
print(f"  brazo frente a k_Q/k_T: 0,174 / 0,016 = {0.174/0.016:.1f}")

# ---------------------------------------------------------------------------
titulo(16, "INDI en un paso, eje de alabeo del x500")
tau_d = 0.3
wdot_pert = tau_d / JX
print(f"  par de perturbación {tau_d} N m / Jx = {wdot_pert:.2f} rad/s^2 (G = b = {B:.1f} rad/s^2 por unidad de mando)")
u0_, nu_ = 0.0, 0.0
wdot0 = wdot_pert + B * u0_
u = u0_ + (nu_ - wdot0) / B
print(f"  instante anterior: u0 = {u0_}, omega_dot0 medida = {wdot0:.2f}; consigna nu = {nu_}")
print(f"  u = u0 + (nu - omega_dot0)/G = {u:.4f}; comprobación: b u + perturbación = {B*u + wdot_pert:.2e}")
wsp, wm, Kw = 1.0, 0.8, 45.0
nu2 = Kw * (wsp - wm)
u2 = u + (nu2 - 0.0) / B
print(f"  si además omega_sp = {wsp}, omega = {wm}: nu = K_w e = {nu2:.1f} rad/s^2;"
      f" siguiente u = {u:.4f} + ({nu2:.1f} - 0)/{B:.0f} = {u2:.4f}")

# ---------------------------------------------------------------------------
titulo(17, "Ala fija: modelo de diseño del alabeo del Aerosonde")
Jx_a, Jz_a, Jxz_a = 0.8244, 1.759, 0.1204
S_a, b_a, rho_a = 0.55, 2.8956, 1.2682
Clp, Cnp, Clda, Cnda = -0.51, 0.069, 0.17, -0.011
Gm = Jx_a * Jz_a - Jxz_a**2
g3, g4 = Jz_a / Gm, Jxz_a / Gm
Cpp = g3 * Clp + g4 * Cnp
Cpda = g3 * Clda + g4 * Cnda
Va = 25.0
qdin = 0.5 * rho_a * Va**2
a1 = -qdin * S_a * b_a * Cpp * b_a / (2 * Va)
a2 = qdin * S_a * b_a * Cpda
print(f"  Gamma = Jx Jz - Jxz^2 = {Gm:.4f} kg^2 m^4; Gamma3 = Jz/Gamma = {g3:.4f}, Gamma4 = Jxz/Gamma = {g4:.4f} 1/(kg m^2)")
print(f"  C_p_p = Gamma3 C_l_p + Gamma4 C_n_p = {Cpp:.4f};  C_p_da = Gamma3 C_l_da + Gamma4 C_n_da = {Cpda:.4f}")
print(f"  presión dinámica 1/2 rho Va^2 = {qdin:.1f} Pa; 1/2 rho Va^2 S b = {qdin*S_a*b_a:.1f} N m")
print(f"  a_phi1 = -(1/2 rho Va^2 S b) C_p_p b/(2Va) = {a1:.2f} 1/s;  a_phi2 = (1/2 rho Va^2 S b) C_p_da = {a2:.2f} 1/s^2")
da_max, ephi_max, zeta = math.radians(45), math.radians(15), 0.707
kp = da_max / ephi_max
wn_phi = math.sqrt(a2 * kp)
kd = (2 * zeta * wn_phi - a1) / a2
print(f"  kp = 45°/15° = {kp:.2f}; omega_n = sqrt(a2 kp) = {wn_phi:.2f} rad/s = {wn_phi/2/math.pi:.2f} Hz;"
      f" kd = (2 zeta omega_n - a1)/a2 = {kd:.4f} s")
pol = np.roots([1, a1 + a2 * kd, a2 * kp])
print(f"  polos del lazo cerrado s^2 + (a1 + a2 kd) s + a2 kp: {pol}; |polo| = {abs(pol[0]):.2f}, zeta = {-pol[0].real/abs(pol[0]):.3f}")
print(f"  sin derivada (kd = 0): zeta = a1/(2 omega_n) = {a1/(2*wn_phi):.3f}")
for W in (5, 10, 20):
    wn_chi = wn_phi / W
    kpc = 2 * zeta * wn_chi * Va / G
    kic = wn_chi**2 * Va / G
    print(f"  rumbo W = {W:2d}: omega_n_chi = {wn_chi:.3f} rad/s (periodo {2*math.pi/wn_chi:4.1f} s);"
          f" kp_chi = 2 zeta omega_n V_g/g = {kpc:.3f}, ki_chi = omega_n^2 V_g/g = {kic:.3f} 1/s")
phi = math.radians(20)
print(f"  giro coordinado a 25 m/s con 20° de alabeo: chi' = g tan(phi)/V = {G*math.tan(phi)/Va:.4f} rad/s"
      f" (lineal g phi/V = {G*phi/Va:.4f}); radio V^2/(g tan phi) = {Va**2/(G*math.tan(phi)):.0f} m")

# ---------------------------------------------------------------------------
titulo(18, "TECS: energía específica en el Aerosonde (m = 11 kg, V = 25 m/s)")
m_a = 11.0
gam = math.radians(5)
print(f"  subir a gamma = 5° sin cambiar V: Es' = V sen gamma = {Va*math.sin(gam):.2f} m/s;"
      f" (T - D)/(m g) = sen gamma = {math.sin(gam):.4f} -> T - D = {math.sin(gam)*m_a*G:.1f} N")
print(f"  tirar del mando a 5° sin tocar el motor (T = D): V' = -g sen gamma = {-G*math.sin(gam):.3f} m/s^2;"
      f" Es' = 0 y Ls'/V = 2 sen gamma = {2*math.sin(gam):.4f}")
print(f"  energía: 1 m/s de velocidad a 25 m/s vale V dV/g = {Va*1/G:.2f} m de altura")

# ---------------------------------------------------------------------------
titulo(19, "Guiado L1: geometría y linealización")
L1 = 80.0
d0 = 30.0
eta0 = math.asin(d0 / L1)
acmd = 2 * Va**2 / L1 * math.sin(eta0)
print(f"  V = {Va}, L1 = {L1}, separación {d0} m, rumbo paralelo: sen eta = d/L1 = {d0/L1:.3f} -> eta = {math.degrees(eta0):.1f}°")
print(f"  a_cmd = 2 V^2/L1 sen eta = {acmd:.2f} m/s^2 -> alabeo atan(a/g) = {math.degrees(math.atan(acmd/G)):.1f}°")
wn = math.sqrt(2) * Va / L1
print(f"  omega_n = sqrt2 V/L1 = {wn:.3f} rad/s (periodo propio {2*math.pi/wn:.1f} s), zeta = {1/math.sqrt(2):.3f}")
print(f"  aproximación lineal en el instante inicial: eta ~ d/L1 = {d0/L1:.3f} rad frente a {eta0:.3f}")

# ---------------------------------------------------------------------------
titulo(20, "Fase que cuesta el retardo de una política: Delta phi = omega_c tau")
for nombre, fc, tau in (("lazo de rate del x500", 6.3, 0.020), ("actitud de PX4 (k = 4)", 4 / (2 * math.pi), 0.057),
                        ("lazo de rate con 57 ms", 6.3, 0.057)):
    print(f"  {nombre}: f_c = {fc:.2f} Hz, tau = {1000*tau:.0f} ms -> Delta phi = {math.degrees(2*math.pi*fc*tau):.0f}°")
