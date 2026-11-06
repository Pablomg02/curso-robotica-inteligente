"""Datos de las figuras y tablas del tema A2 (control clásico y jerarquía de lazos).

Uso:  python3 figuras_a2.py > salida.txt
Requiere numpy. Imprime cifras para las tablas y trazados SVG (atributo d de <path>)
ya escalados al viewBox de cada figura.
"""
import math
import numpy as np

G = 9.81


def path(xs, ys, X, Y):
    pts = [f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys)]
    return "M" + " L".join(pts)


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def logmap(d0, d1, p0, p1):
    return lambda v: p0 + (math.log10(v) - math.log10(d0)) * (p1 - p0) / (math.log10(d1) - math.log10(d0))


def margenes(w, L):
    """Cruce de ganancia, margen de fase (grados) y margen de ganancia (dB) de L(jw) en una rejilla."""
    mag = np.abs(L)
    ph = np.unwrap(np.angle(L))
    i = np.where((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    wc = pm = None
    if len(i):
        k = i[0]
        a = (math.log(mag[k])) / (math.log(mag[k]) - math.log(mag[k + 1]))
        wc = math.exp(math.log(w[k]) + a * (math.log(w[k + 1]) - math.log(w[k])))
        phc = ph[k] + a * (ph[k + 1] - ph[k])
        pm = 180 + math.degrees(phc)
    j = np.where((ph[:-1] > -math.pi) & (ph[1:] <= -math.pi))[0]
    gm = wpc = None
    if len(j):
        k = j[0]
        a = (ph[k] + math.pi) / (ph[k] - ph[k + 1])
        wpc = w[k] + a * (w[k + 1] - w[k])
        m = mag[k] + a * (mag[k + 1] - mag[k])
        gm = -20 * math.log10(m)
    return wc, pm, wpc, gm


def sobrepaso(t, y, ref=1.0):
    return 100 * (np.max(y) - ref) / ref


print("=" * 70)
print("1. RETARDO Y CRUCE: lazo L = (wc/s) e^{-s tau}, MF = 90° - wc tau")
for tau_ms in (0.5, 1, 2.5, 5, 10, 20, 50):
    tau = tau_ms / 1000
    f45 = (math.pi / 4) / (2 * math.pi * tau)
    f60 = (math.pi / 6) / (2 * math.pi * tau)
    print(f"  tau = {tau_ms:5.1f} ms -> cruce máximo {f45:6.1f} Hz (MF 45°), {f60:6.1f} Hz (MF 60°)")

# Segundo orden estándar L = wn^2 / (s (s + 2 zeta wn)): margen de fase y sobrepaso exactos
for pm_obj in (45, 60, 68):
    lo, hi = 0.01, 2.0
    for _ in range(80):
        z = (lo + hi) / 2
        pm = math.degrees(math.atan(2 * z / math.sqrt(math.sqrt(1 + 4 * z**4) - 2 * z * z)))
        lo, hi = (z, hi) if pm < pm_obj else (lo, z)
    os_ = 100 * math.exp(-math.pi * z / math.sqrt(1 - z * z)) if z < 1 else 0.0
    print(f"  2.º orden estándar con MF {pm_obj}°: zeta = {z:.3f}, sobrepaso {os_:.1f} %")

print("=" * 70)
print("2. FILTRO DE BUTTERWORTH DE SEGUNDO ORDEN: retardo de grupo y fase")
for fc in (20, 30, 40, 60, 80, 100, 150, 200):
    wn = 2 * math.pi * fc
    tg = math.sqrt(2) / wn
    out = []
    for f in (5, 10, 20):
        w = 2 * math.pi * f
        ph = -math.degrees(math.atan2(math.sqrt(2) * wn * w, wn * wn - w * w))
        out.append(f"{ph:6.1f}° a {f} Hz")
    print(f"  fc = {fc:3d} Hz: retardo de grupo en baja frecuencia {1000*tg:4.2f} ms; fase " + ", ".join(out))

print("  Filtro PT1 (Betaflight): retardo de grupo en baja frecuencia 1/(2 pi fc)")
for fc in (20, 75, 150, 250, 500):
    print(f"  PT1 fc = {fc:3d} Hz: {1000/(2*math.pi*fc):4.2f} ms")
print("  Cadena de gyro por defecto de Betaflight: LPF1 dinámico 250-500 Hz + LPF2 500 Hz ->"
      f" {1000/(2*math.pi*250)+1000/(2*math.pi*500):.2f} ms (ralentí), {2000/(2*math.pi*500):.2f} ms (a tope)")

print("=" * 70)
print("3. SEPARACIÓN DE ESCALAS: lazo interno de 2.º orden (zeta_i = 0,7) con externo P de ganancia k")
# Externo: theta' = omega ; omega_sp = k (theta_sp - theta). Interno: omega/omega_sp = wi^2/(s^2 + 2 zi wi s + wi^2).
# Tiempo normalizado con k = 1; r = wi / k.
zi = 0.7
res3 = {}
for r in (1.0, 2.0, 4.0, 8.0, 1e6):
    wi = r
    dt = 1e-3
    T = 10.0
    n = int(T / dt)
    th = 0.0
    om = 0.0
    omd = 0.0
    ys = []
    ts = []
    # integración RK4 del sistema (th, om, omd) con om'' = wi^2 (sp - om) - 2 zi wi om'
    def rhs(s):
        th_, om_, omd_ = s
        sp = 1.0 * (1.0 - th_)
        if r > 1e5:
            return np.array([sp, 0.0, 0.0])
        return np.array([om_, omd_, wi * wi * (sp - om_) - 2 * zi * wi * omd_])
    s = np.zeros(3)
    for k in range(n + 1):
        if k % 20 == 0:
            ts.append(k * dt)
            ys.append(s[0])
        k1 = rhs(s); k2 = rhs(s + dt / 2 * k1); k3 = rhs(s + dt / 2 * k2); k4 = rhs(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ts = np.array(ts); ys = np.array(ys)
    w = np.logspace(-2, 3, 20000)
    sj = 1j * w
    if r > 1e5:
        L = 1.0 / sj
    else:
        L = (1.0 / sj) * wi * wi / (sj * sj + 2 * zi * wi * sj + wi * wi)
    wc, pm, _, _ = margenes(w, L)
    res3[r] = (ts, ys)
    tag = "inf" if r > 1e5 else f"{r:g}"
    print(f"  r = {tag:>4}: sobrepaso {max(0, sobrepaso(ts, ys)):5.1f} %, MF externo {pm:5.1f}°, cruce {wc:.3f} k")
# Figura: viewBox 0 0 520 300; ejes x 60..500 (t·k de 0 a 10), y 250..30 (0 a 1,8)
X3 = lin(0, 10, 60, 500)
Y3 = lin(0, 1.8, 250, 30)
for r in (1.0, 2.0, 4.0, 1e6):
    ts, ys = res3[r]
    tag = "inf" if r > 1e5 else f"{r:g}"
    print(f"path r={tag}:", path(ts, ys, X3, Y3))

print("=" * 70)
print("4. ERROR DE ACTITUD: magnitud frente al ángulo de giro")
th = np.linspace(0, math.pi, 91)
eR = np.sin(th)                    # Lee et al.: |e_R| = sin(theta)
eq = 2 * np.sin(th / 2)            # cuaternión con elección de hemisferio: 2|q_v| = 2 sin(theta/2)
ea = th                            # eje-ángulo: theta
psi = 1 - np.cos(th)               # función de error de Lee: Psi = 1 - cos(theta)
for d in (30, 90, 150, 170, 179):
    t_ = math.radians(d)
    print(f"  {d:3d}°: |e_R| = {math.sin(t_):.3f}, 2 sin(th/2) = {2*math.sin(t_/2):.3f}, th = {t_:.3f} rad")
X4 = lin(0, 180, 60, 500)
Y4 = lin(0, 3.2, 250, 30)
deg = np.degrees(th)
print("path eR:", path(deg, eR, X4, Y4))
print("path eq:", path(deg, eq, X4, Y4))
print("path ea:", path(deg, ea, X4, Y4))

print("=" * 70)
print("5. GUIADO L1 en recta: a = 2 V^2 / L1 sin(eta), V = 25 m/s, d0 = 30 m")
V = 25.0
res5 = {}
for L1 in (40.0, 80.0, 160.0):
    x, y, psi_ = 0.0, 30.0, 0.0
    dt = 0.01
    xs, ys = [x], [y]
    t = 0.0
    t5 = None
    amax_used = 0.0
    while t < 40:
        dx = math.sqrt(max(L1 * L1 - y * y, 1e-9))
        los = math.atan2(-y, dx)
        eta = los - psi_
        a = 2 * V * V / L1 * math.sin(eta)
        amax_used = max(amax_used, abs(a))
        psi_ += a / V * dt
        x += V * math.cos(psi_) * dt
        y += V * math.sin(psi_) * dt
        t += dt
        if t5 is None and abs(y) < 0.05 * 30:
            t5 = t
        if abs(t / 0.25 - round(t / 0.25)) < 1e-6:
            xs.append(x); ys.append(y)
    # predicción lineal: wn = sqrt(2) V / L1, zeta = 1/sqrt(2)
    wn = math.sqrt(2) * V / L1
    # primer cruce por el 5 % para zeta = 0,707 (respuesta libre desde d0 con d'0 = 0): se integra la lineal
    yl, vl, tl, t5l = 30.0, 0.0, 0.0, None
    while tl < 40:
        al = -wn * wn * yl - 2 * (1 / math.sqrt(2)) * wn * vl
        vl += al * 0.001; yl += vl * 0.001; tl += 0.001
        if t5l is None and abs(yl) < 1.5:
            t5l = tl
    res5[L1] = (np.array(xs), np.array(ys))
    print(f"  L1 = {L1:5.0f} m: wn = {wn:.3f} rad/s, periodo propio {2*math.pi/wn:5.1f} s;"
          f" llega al 5 % en {t5:5.2f} s (lineal {t5l:5.2f} s); |a|max = {amax_used:.2f} m/s^2"
          f" = {math.degrees(math.atan(amax_used/G)):.1f}° de alabeo")
# Figura: viewBox 0 0 520 220; x de 0 a 800 m -> 60..500; y de -10 a 40 m -> 190..20
X5 = lin(0, 800, 60, 500)
Y5 = lin(-10, 40, 190, 20)
for L1 in (40.0, 80.0, 160.0):
    xs, ys = res5[L1]
    k = xs <= 800
    print(f"path L1={L1:.0f}:", path(xs[k], ys[k], X5, Y5))

print("=" * 70)
print("6. LAZO DE RATE DEL x500 (Gazebo) CON LAS GANANCIAS POR DEFECTO DE PX4")
# Planta: x500 de PX4-gazebo-models (J, kT, tau motor) y airframe 4001_gz_x500 (CA_ROTOR, SIM_GZ_EC_MIN/MAX).
JX = 0.02389                      # inercia compuesta de Gazebo: cuerpo (0,02167) + 4 rotores
KT = 8.54858e-06
M_X500 = 2.0 + 4 * 0.016076923076923075
ARM = 0.174                       # |CA_ROTORn_PY|
OM_MIN, OM_MAX = 150.0, 1000.0    # SIM_GZ_EC_MIN / MAX: salida 0..1 -> 150..1000 rad/s
OM0 = math.sqrt(M_X500 * G / (4 * KT))
U0 = (OM0 - OM_MIN) / (OM_MAX - OM_MIN)
DT_DU = 2 * KT * OM0 * (OM_MAX - OM_MIN)          # N por unidad de salida de motor
MIX = 1 / math.sqrt(2)                            # columna de alabeo normalizada de PX4 (cuadricóptero X)
B = 4 * ARM * DT_DU * MIX / JX                    # rad/s^2 por unidad de par normalizado
U_SAT = (1 - U0) / MIX                            # par normalizado que satura el motor más cargado
print(f"  hover: Omega0 = {OM0:.1f} rad/s, salida u0 = {U0:.3f}; dT/du = {DT_DU:.2f} N")
print(f"  ganancia de planta b = {B:.1f} rad/s^2 por unidad de par normalizado")
print(f"  par normalizado de saturación en hover (motor a 1): {U_SAT:.3f} -> {U_SAT*B:.0f} rad/s^2")
# PX4 por defecto (mc_rate_control_params.yaml, imu_gyro_parameters.yaml)
P_, I_, D_, K_ = 0.15, 0.2, 0.003, 1.0
F_GYRO, F_DGYRO = 40.0, 20.0
print(f"  PX4: MC_ROLLRATE_P {P_}, I {I_}, D {D_}, K {K_}; IMU_GYRO_CUTOFF {F_GYRO} Hz, IMU_DGYRO_CUTOFF {F_DGYRO} Hz")


def lazo_rate(w, tau_m, f_loop, extra=0.0, P=P_, I=I_, D=D_, K=K_):
    s = 1j * w
    wg = 2 * math.pi * F_GYRO
    wdg = 2 * math.pi * F_DGYRO
    Fv = wg * wg / (s * s + math.sqrt(2) * wg * s + wg * wg)     # Butterworth 2.º orden (LowPassFilter2p)
    Fd = 1 / (1 + s / wdg)                                       # AlphaFilter de 1.er orden sobre la derivada
    T = 1 / f_loop
    delay = np.exp(-s * (1.5 * T + extra))                       # retenedor (T/2) + una muestra de cómputo (supuesto)
    C = K * ((P + I / s) * Fv + D * s * Fd)
    Pl = B / s / (1 + tau_m * s)
    return C * Pl * delay


w = np.logspace(0, 3.5, 6000)
casos = [("SITL 250 Hz, tau 12,5 ms", 0.0125, 250, 0.0), ("SITL 250 Hz, tau 25 ms", 0.025, 250, 0.0),
         ("800 Hz (rc.mc_defaults), tau 12,5 ms", 0.0125, 800, 0.0), ("SITL 250 Hz, tau 39 ms (Agilicious)", 0.039, 250, 0.0),
         ("250 Hz, tau 12,5 ms, +10 ms", 0.0125, 250, 0.010), ("250 Hz, tau 12,5 ms, +20 ms", 0.0125, 250, 0.020)]
for nombre, tm, fl, ex in casos:
    wc, pm, wpc, gm = margenes(w, lazo_rate(w, tm, fl, ex))
    print(f"  {nombre:42s}: cruce {wc:6.1f} rad/s = {wc/2/math.pi:5.2f} Hz, MF {pm:5.1f}°,"
          f" MG {gm:5.1f} dB a {wpc/2/math.pi:5.1f} Hz; margen de retardo {1000*math.radians(pm)/wc:5.1f} ms")
# Figura: Bode del caso base y con +10 ms. viewBox 0 0 520 300; dos paneles.
# Panel de magnitud: x log 1..1000 rad/s -> 60..500; y dB -30..30 -> 130..20
# Panel de fase:     y grados -270..-90 -> 270..160
Xb = logmap(1, 1000, 60, 500)
Ym = lin(-30, 30, 130, 20)
Yp = lin(-270, -90, 270, 160)
wf = np.logspace(0, 3, 400)
for nombre, ex in (("base", 0.0), ("+10ms", 0.010)):
    L = lazo_rate(wf, 0.0125, 250, ex)
    mag = 20 * np.log10(np.abs(L))
    ph = np.degrees(np.unwrap(np.angle(L)))
    km = (mag <= 30) & (mag >= -30)
    kp = (ph >= -270) & (ph <= -90)
    print(f"path bode mag {nombre}:", path(wf[km], mag[km], Xb, Ym))
    print(f"path bode fase {nombre}:", path(wf[kp], ph[kp], Xb, Yp))
wc0, pm0, _, _ = margenes(w, lazo_rate(w, 0.0125, 250, 0.0))
print(f"  marca cruce base: x = {Xb(wc0):.1f}; fase en el cruce = {pm0-180:.1f}° -> y = {Yp(pm0-180):.1f}")

print("=" * 70)
print("7. SIMULACIÓN EN EL TIEMPO DEL EJE DE ALABEO DEL x500 (lineal, mando a 250 Hz, planta a 10 kHz)")


class Biquad:
    """LowPassFilter2p de PX4: Butterworth de 2.º orden discretizado por la transformación bilineal."""

    def __init__(self, fs, fc):
        fr = fs / fc
        ohm = math.tan(math.pi / fr)
        c = 1 + 2 * math.cos(math.pi / 4) * ohm + ohm * ohm
        self.b0 = ohm * ohm / c
        self.b1 = 2 * self.b0
        self.b2 = self.b0
        self.a1 = 2 * (ohm * ohm - 1) / c
        self.a2 = (1 - 2 * math.cos(math.pi / 4) * ohm + ohm * ohm) / c
        self.d1 = self.d2 = 0.0

    def __call__(self, x):
        d0 = x - self.d1 * self.a1 - self.d2 * self.a2
        y = d0 * self.b0 + self.d1 * self.b1 + self.d2 * self.b2
        self.d2, self.d1 = self.d1, d0
        return y


def simula(ley, f_loop=250, extra=0.0, tau_m=0.0125, sp=lambda t: 0.0, tau_d=lambda t: 0.0, T=0.6):
    """Eje de alabeo: w' = (b_phys*m + tau_d)/J con m = estado del motor (par normalizado efectivo)."""
    h = 1e-4
    n_sub = int(round(1 / f_loop / h))
    n_ex = int(round(extra / h))
    lp = Biquad(f_loop, F_GYRO)
    a_d = (1 / f_loop) / (1 / (2 * math.pi * F_DGYRO) + 1 / f_loop)   # AlphaFilter de PX4: dt / (tau + dt)
    st = dict(w=0.0, m=0.0, integ=0.0, wprev=0.0, acc_f=0.0, m_f=0.0)
    cmd_buf = [0.0] * (n_ex + n_sub + 1)   # una muestra de cómputo + retardo extra
    ts, ws = [], []
    u_hold = 0.0
    for k in range(int(T / h)):
        t = k * h
        if k % n_sub == 0:                    # instante de muestreo del lazo
            wmeas = st["w"]
            wf = lp(wmeas)
            acc = (wmeas - st["wprev"]) * f_loop
            st["wprev"] = wmeas
            st["acc_f"] += a_d * (acc - st["acc_f"])
            st["m_f"] += a_d * (st["m"] - st["m_f"])
            u_new = ley(st, sp(t), wf, 1 / f_loop)
        else:
            u_new = None
        # cola de mandos: el mando calculado se aplica una muestra después (+ extra)
        cmd_buf.append(u_new if u_new is not None else cmd_buf[-1] if False else None)
        cmd_buf.pop(0)
        if cmd_buf[0] is not None:
            u_hold = cmd_buf[0]
        # motor de primer orden (par normalizado efectivo) y cuerpo rígido
        st["m"] += h * (max(-U_SAT, min(U_SAT, u_hold)) - st["m"]) / tau_m
        st["w"] += h * (B * st["m"] + tau_d(t) / JX)
        if k % 20 == 0:
            ts.append(t); ws.append(st["w"])
    return np.array(ts), np.array(ws)


def ley_pid(st, r, wf, dt):
    e = r - wf
    u = K_ * (P_ * e + st["integ"] - D_ * st["acc_f"])
    st["integ"] += K_ * I_ * e * dt
    return u


def ley_indi(st, r, wf, dt, kw=45.0):
    nu = kw * (r - wf)                        # aceleración angular deseada
    return st["m_f"] + (nu - st["acc_f"]) / B


paso = lambda t: 1.0 if t >= 0.05 else 0.0
res7 = {}
for nombre, ex in (("base", 0.0), ("+10ms", 0.010), ("+20ms", 0.020)):
    t, wv = simula(ley_pid, extra=ex, sp=paso, T=0.6)
    res7[nombre] = (t, wv)
    k = t >= 0.05
    tt = t[k] - 0.05
    yy = wv[k]
    i90 = np.argmax(yy >= 0.9)
    print(f"  escalón de 1 rad/s, {nombre:6s}: subida al 90 % en {1000*tt[i90]:5.1f} ms, sobrepaso {max(0,100*(yy.max()-1)):5.1f} %")
t20, w20 = res7["+20ms"]
k = t20 >= 0.05
yy = w20[k] - 1
tt = t20[k]
pk = [i for i in range(1, len(yy) - 1) if yy[i] > yy[i - 1] and yy[i] >= yy[i + 1] and yy[i] > 0.005]
per = np.diff(tt[pk]).mean()
print(f"  +20 ms: oscilación de {1/per:.2f} Hz; cociente entre picos sucesivos {np.mean(yy[pk][1:]/yy[pk][:-1]):.2f}")
# Figura: viewBox 0 0 520 300; x de 0 a 0,5 s -> 60..500; y de 0 a 1,6 -> 250..30
X7 = lin(0, 0.5, 60, 500)
Y7 = lin(0, 1.6, 250, 30)
for nombre in res7:
    t, wv = res7[nombre]
    k = (t >= 0.05) & (t <= 0.55)
    print(f"path paso {nombre}:", path(t[k] - 0.05, wv[k], X7, Y7))

print("  Perturbación: par de 0,3 N m (centro de masas desplazado 1,5 cm) en t = 0,05 s")
pert = lambda t: 0.3 if t >= 0.05 else 0.0
res8 = {}
for nombre, ley in (("PID", ley_pid), ("INDI", ley_indi)):
    t, wv = simula(ley, sp=lambda t: 0.0, tau_d=pert, T=1.05)
    res8[nombre] = (t, wv)
    k = t >= 0.05
    pico = np.max(np.abs(wv[k]))
    ang = np.cumsum(wv) * (t[1] - t[0])
    i_rec = np.where(np.abs(wv) > 0.05 * pico)[0][-1]
    print(f"  {nombre:4s}: pico de rate {math.degrees(pico):6.1f} °/s, ángulo final {math.degrees(ang[-1]):6.2f}°,"
          f" se queda en el 5 % del pico a los {1000*(t[i_rec]-0.05):5.0f} ms")
X8 = lin(0, 1.0, 60, 500)
Y8 = lin(-5, 25, 250, 30)
for nombre in res8:
    t, wv = res8[nombre]
    k = t >= 0.05
    print(f"path pert {nombre}:", path(t[k] - 0.05, np.degrees(wv[k]), X8, Y8))

print("=" * 70)
print("7b. PLANTA NO LINEAL: cuatro motores con k_T Omega^2 y constante asimétrica (12,5 ms al subir, 25 al bajar)")


def simula_nl(f_loop=250, tup=0.0125, tdn=0.025, sp=lambda t: 0.0, dinj=lambda t: 0.0, T=0.6, J=JX):
    h = 1e-4
    n_sub = int(round(1 / f_loop / h))
    lp = Biquad(f_loop, F_GYRO)
    a_d = (1 / f_loop) / (1 / (2 * math.pi * F_DGYRO) + 1 / f_loop)
    w_ = wprev = integ = acc_f = 0.0
    Om = np.array([OM0, OM0])          # par de motores que sube (+) y par que baja (-)
    cola = [None] * (n_sub + 1)
    c_hold = 0.0
    ts, ws, cs, us = [], [], [], []
    for k in range(int(T / h)):
        t = k * h
        if k % n_sub == 0:
            wf = lp(w_)
            acc = (w_ - wprev) * f_loop
            wprev = w_
            acc_f += a_d * (acc - acc_f)
            e = sp(t) - wf
            c = K_ * (P_ * e + integ - D_ * acc_f)
            integ += K_ * I_ * e / f_loop
            cola.append(c)
        else:
            cola.append(None)
        cola.pop(0)
        if cola[0] is not None:
            c_hold = cola[0]
        u = c_hold + dinj(t)
        m = MIX * u
        cmd = np.array([OM_MIN + (OM_MAX - OM_MIN) * min(1, max(0, U0 + m)),
                        OM_MIN + (OM_MAX - OM_MIN) * min(1, max(0, U0 - m))])
        Om = Om + h * (cmd - Om) / np.where(cmd > Om, tup, tdn)
        w_ += h * 2 * ARM * KT * (Om[0] ** 2 - Om[1] ** 2) / J
        if k % 10 == 0:
            ts.append(t); ws.append(w_); cs.append(c_hold); us.append(u)
    return np.array(ts), np.array(ws), np.array(cs), np.array(us)


def lazo_medido(tup, tdn, J=JX, fs_=(4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9)):
    Ls = []
    for f in fs_:
        t, wv, c, u = simula_nl(tup=tup, tdn=tdn, J=J, dinj=lambda t: 0.01 * math.sin(2 * math.pi * f * t),
                                T=max(1.0, 12 / f) + 0.5)
        k = t > 0.5
        ph = np.exp(-2j * math.pi * f * t[k])
        Ls.append(-np.sum(c[k] * ph) / np.sum(u[k] * ph))
    mags, phs = np.abs(Ls), np.degrees(np.angle(Ls))
    j = np.where((mags[:-1] >= 1) & (mags[1:] < 1))[0][0]
    a = math.log(mags[j]) / (math.log(mags[j]) - math.log(mags[j + 1]))
    return fs_[j] + a * (fs_[j + 1] - fs_[j]), 180 + phs[j] + a * (phs[j + 1] - phs[j])


J_GZ = None
for nombre, tup, tdn in (("simétrico 12,5 ms", 0.0125, 0.0125), ("asimétrico 12,5/25 ms (A1, Gazebo)", 0.0125, 0.025)):
    t, wv, _, _ = simula_nl(sp=paso, tup=tup, tdn=tdn)
    k = t >= 0.05
    i90 = np.argmax(wv[k] >= 0.9)
    fc_m, pm_m = lazo_medido(tup, tdn)
    print(f"  {nombre:36s}: subida al 90 % en {1000*(t[k][i90]-0.05):5.1f} ms, sobrepaso {100*(wv[k].max()-1):4.1f} %;"
          f" lazo medido: cruce {fc_m:.2f} Hz, MF {pm_m:.1f}°")

print("=" * 70)
print("8. ALA FIJA: lazo de alabeo y de rumbo del Aerosonde por cierre sucesivo (Beard y McLain, cap. 6)")
# Parámetros de mavsim_public (aerosonde_parameters.py), como en A1
Jx_a, Jz_a, Jxz_a = 0.8244, 1.759, 0.1204
S_a, b_a, rho_a = 0.55, 2.8956, 1.2682
Clp, Cnp, Clda, Cnda = -0.51, 0.069, 0.17, -0.011
Gm = Jx_a * Jz_a - Jxz_a**2
g3, g4 = Jz_a / Gm, Jxz_a / Gm
Cpp = g3 * Clp + g4 * Cnp
Cpda = g3 * Clda + g4 * Cnda
Va = 25.0
a1 = -0.5 * rho_a * Va**2 * S_a * b_a * Cpp * b_a / (2 * Va)
a2 = 0.5 * rho_a * Va**2 * S_a * b_a * Cpda
print(f"  Va = {Va} m/s: a_phi1 = {a1:.2f} 1/s, a_phi2 = {a2:.2f} 1/s^2  (phi''= -a1 phi' + a2 delta_a)")
da_max, ephi_max, zeta = math.radians(45), math.radians(15), 0.707   # elección de diseño nuestra
kp = da_max / ephi_max
wn_phi = math.sqrt(abs(a2) * kp)
kd = (2 * zeta * wn_phi - a1) / a2
print(f"  diseño: delta_a max 45°, error de alabeo que satura 15° -> kp = {kp:.2f}, wn_phi = {wn_phi:.2f} rad/s"
      f" ({wn_phi/2/math.pi:.2f} Hz), kd = {kd:.3f} s")
for W in (5, 10, 20):
    wn_chi = wn_phi / W
    kp_chi = 2 * zeta * wn_chi * Va / G
    ki_chi = wn_chi**2 * Va / G
    print(f"  W = {W:2d}: wn_chi = {wn_chi:.3f} rad/s (periodo {2*math.pi/wn_chi:5.1f} s), kp_chi = {kp_chi:.2f}, ki_chi = {ki_chi:.3f}")

print("=" * 70)
print("9. CASCADA POR DEFECTO DE PX4: anchos de banda aproximados y separación entre lazos")
# rate: cruce calculado en la sección 6; actitud: MC_ROLL_P; velocidad: MPC_XY_VEL_P_ACC; posición: MPC_XY_P
w_rate = margenes(w, lazo_rate(w, 0.0125, 250, 0.0))[0]
lazos = [("rate (cruce calculado)", w_rate), ("actitud MC_ROLL_P", 4.0), ("velocidad MPC_XY_VEL_P_ACC", 1.8),
         ("posición MPC_XY_P", 0.95)]
for i, (n, wb) in enumerate(lazos):
    txt = f"  {n:28s}: {wb:6.2f} rad/s = {wb/2/math.pi:5.2f} Hz"
    if i:
        r = lazos[i - 1][1] / wb
        txt += f"; r = {r:5.1f}; zeta (interno de 1.er orden) = {0.5*math.sqrt(r):.2f}"
    print(txt)
