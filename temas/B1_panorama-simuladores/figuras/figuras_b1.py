"""Datos de las figuras y tablas del tema B1 (panorama de simuladores).

Uso:  python3 figuras_b1.py > salida.txt      (numpy; unos dos minutos)

Planta: el x500 de PX4 en Gazebo (A1-A3): m = 2,064 kg (cuerpo + rotores), J = diag(0,0239; 0,0239; 0,044),
rotores en (±0,174; ±0,174) m, kT = 8,54858e-6 N/(rad/s)^2, km = 0,016 m, Omega en [150, 1000] rad/s.
Convenciones de planta leídas en el código (noviembre de 2026):
  - MuJoCo Menagerie (skydio_x2/x2.xml, bitcraze_crazyflie_2/cf2.xml): el actuador es una fuerza (o un par)
    instantánea; sin dinámica de motor ni arrastre.
  - Gazebo, MulticopterMotorModel (gz-sim) con el x500 de PX4-gazebo-models: Omega con filtro de primer orden
    (12,5 ms al subir, 25 ms al bajar), T = kT Omega^2 y arrastre de rotor -|Omega| c_d v_perp, c_d = 8,06428e-5.
  - SIH de PX4 (simulator_sih, sih_params.yaml): empuje lineal en el mando con primer orden de 50 ms
    (SIH_T_TAU), arrastre lineal SIH_KDV = 1 N/(m/s) y amortiguamiento angular SIH_KDW = 0,025 N m/(rad/s).
"""
import math
import time
import numpy as np

G = 9.81
M = 2.0 + 4 * 0.016076923076923075
J = np.array([0.02389, 0.02389, 0.044])
A_ = 0.174
KT = 8.54858e-06
KM = 0.016
OM_MIN, OM_MAX = 150.0, 1000.0
CD_ROT = 8.06428e-05
POS = np.array([[A_, A_], [-A_, -A_], [A_, -A_], [-A_, A_]])
SPIN = np.array([-1.0, -1.0, 1.0, 1.0])          # rotores 0 y 1 ccw (par de reacción -z), 2 y 3 cw
ALLOC = np.vstack([np.ones(4), POS[:, 1], -POS[:, 0], SPIN * KM])
ALLOC_INV = np.linalg.inv(ALLOC)
E3 = np.array([0.0, 0.0, 1.0])
OM_HOVER = math.sqrt(M * G / (4 * KT))


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qexp(phi):
    a = np.linalg.norm(phi)
    if a < 1e-12:
        return np.array([1.0, *(0.5 * phi)])
    return np.array([math.cos(a / 2), *(math.sin(a / 2) * phi / a)])


# ---------------------------------------------------------------- controlador (igual para todas las plantas)
KP, KD, KR, KW = 4.0, 3.5, 8.0, 25.0
SATURA = True     # recorte de los motores a [OM_MIN, OM_MAX]; se desactiva solo en el estudio de orden de RK4


def controlador(x, ref):
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    pr, vr, ar = ref
    R = q2R(q)
    acmd = -KP * (p - pr) - KD * (v - vr) + ar + G * E3
    T = M * acmd @ R[:, 2]
    b3 = acmd / np.linalg.norm(acmd)
    b2 = np.cross(b3, [1.0, 0, 0])
    b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    Rd = np.column_stack([b1, b2, b3])
    E = 0.5 * (Rd.T @ R - R.T @ Rd)
    eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
    wd = -KR * eR
    tau = J * (-KW * (w - wd)) + np.cross(w, J * w)
    f = ALLOC_INV @ np.array([T, *tau])
    if not SATURA:
        return np.maximum(f, 1e-9)
    return np.clip(f, KT * OM_MIN ** 2, KT * OM_MAX ** 2)


# ---------------------------------------------------------------- plantas
def fuerzas_rotor(estado_mot, modelo):
    """Empuje por rotor y velocidad de rotor efectiva según la convención."""
    if modelo == "gazebo":
        Om = estado_mot
        return KT * Om ** 2, Om
    return estado_mot, np.sqrt(np.maximum(estado_mot, 0) / KT)


def deriv(x, fcmd, modelo, tau_up=0.0125, tau_dn=0.025):
    """x = [p, v, q, w, motores(4)] con z hacia arriba (ENU/FLU)."""
    p, v, q, w, mot = x[0:3], x[3:6], x[6:10], x[10:13], x[13:17]
    R = q2R(q)
    if modelo == "instantaneo":
        f, Om = fcmd, np.sqrt(fcmd / KT)
        dmot = np.zeros(4)
    elif modelo == "gazebo":
        Omc = np.sqrt(fcmd / KT)
        tau = np.where(Omc > mot, tau_up, tau_dn)
        dmot = (Omc - mot) / tau
        f, Om = KT * mot ** 2, mot
    else:  # sih
        dmot = (fcmd - mot) / 0.05
        f, Om = mot, None
    T = f.sum()
    Fb = np.array([0, 0, T])
    tau_b = ALLOC[1:] @ f
    F = R @ Fb - M * G * E3
    if modelo == "gazebo":
        b3 = R[:, 2]
        vperp = v - (v @ b3) * b3
        F = F - CD_ROT * np.abs(Om).sum() * vperp
    elif modelo == "sih":
        F = F - 1.0 * v
        tau_b = tau_b - 0.025 * w
    dv = F / M
    dq = 0.5 * qmul(q, np.array([0.0, *w]))
    dw = (tau_b - np.cross(w, J * w)) / J
    return np.concatenate([v, dv, dq, dw, dmot])


def estado_inicial(modelo, p0=(0, 0, 0), v0=(0, 0, 0)):
    mot = np.full(4, OM_HOVER if modelo == "gazebo" else M * G / 4)
    return np.concatenate([p0, v0, [1, 0, 0, 0], [0, 0, 0], mot])


# ---------------------------------------------------------------- 1. integradores
print("=" * 70)
print("1. INTEGRADORES: x500 en bucle cerrado sobre un círculo (r = 4 m, 5 m/s), 4 s")
print("   Planta Gazebo con motor simétrico de 12,5 ms y arrastre de rotor; control evaluado en cada paso.")
RC, VC = 4.0, 5.0
WC = VC / RC


def ref_circ(t):
    c, s = math.cos(WC * t), math.sin(WC * t)
    return (np.array([RC * c, RC * s, 0.0]), np.array([-VC * s, VC * c, 0.0]),
            np.array([-VC * WC * c, -VC * WC * s, 0.0]))


def f_cl(x, t):
    return deriv(x, controlador(x, ref_circ(t)), "gazebo", 0.0125, 0.0125)


def paso(x, t, dt, metodo):
    if metodo == "rk4":
        k1 = f_cl(x, t)
        k2 = f_cl(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f_cl(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f_cl(x + dt * k3, t + dt)
        xn = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    elif metodo == "euler":
        xn = x + dt * f_cl(x, t)
    else:  # semiimplícito: velocidades primero, posiciones y actitud con las velocidades nuevas
        d = f_cl(x, t)
        xn = x.copy()
        xn[3:6] = x[3:6] + dt * d[3:6]
        xn[10:13] = x[10:13] + dt * d[10:13]
        xn[13:17] = x[13:17] + dt * d[13:17]
        xn[0:3] = x[0:3] + dt * xn[3:6]
        xn[6:10] = qmul(x[6:10], qexp(xn[10:13] * dt))
    xn[6:10] /= np.linalg.norm(xn[6:10])
    return xn


TF = 4.0
EVALS = {"euler": 1, "semi": 1, "rk4": 4}


def integra(metodo, dt, guardar_cada=None):
    n = int(round(TF / dt))
    x = estado_inicial("gazebo", p0=(RC, 0, 0), v0=(0, VC, 0))
    traj = [x[0:3].copy()]
    for k in range(n):
        x = paso(x, k * dt, dt, metodo)
        if guardar_cada and (k + 1) % guardar_cada == 0:
            traj.append(x[0:3].copy())
    return x, np.array(traj)


t0 = time.time()
DT_REF = 5e-5
T_CMP = 0.04   # instantes comunes a todos los pasos (múltiplo de 0,25; 0,5; 1; 2; 4; 8 y 10 ms)
xref, tref = integra("rk4", DT_REF, guardar_cada=int(round(T_CMP / DT_REF)))
print(f"  referencia: RK4 con dt = {DT_REF*1e3:.3f} ms ({time.time()-t0:.1f} s de cálculo)")
print(f"  posición final de referencia: {np.round(xref[0:3], 4)}")
DTS = [0.25e-3, 0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3, 10e-3]
res_int = {}
for met in ("euler", "semi", "rk4"):
    for dt in DTS:
        cada = int(round(T_CMP / dt))
        assert abs(cada * dt - T_CMP) < 1e-12
        xf, tr = integra(met, dt, guardar_cada=cada)
        m = min(len(tr), len(tref))
        err = np.max(np.linalg.norm(tr[:m] - tref[:m], axis=1))
        res_int[(met, dt)] = err
        print(f"  {met:6s} dt = {dt*1e3:5.2f} ms: error máx. de posición {err:.3e} m; "
              f"evaluaciones por segundo simulado {EVALS[met]/dt:8.0f}")
n_sat = 0
x = estado_inicial("gazebo", p0=(RC, 0, 0), v0=(0, VC, 0))
for k in range(int(TF / 1e-3)):
    fc = controlador(x, ref_circ(k * 1e-3))
    n_sat += int(np.any(fc <= KT * OM_MIN ** 2 * (1 + 1e-12)) or np.any(fc >= KT * OM_MAX ** 2 * (1 - 1e-12)))
    x = paso(x, k * 1e-3, 1e-3, "rk4")
print(f"  pasos de 1 ms con algún motor saturado: {n_sat} de {int(TF/1e-3)} (en el transitorio inicial)")
SATURA = False
xref2, tref2 = integra("rk4", DT_REF, guardar_cada=int(round(T_CMP / DT_REF)))
for dt in DTS:
    xf, tr = integra("rk4", dt, guardar_cada=int(round(T_CMP / dt)))
    res_int[("rk4ns", dt)] = np.max(np.linalg.norm(tr - tref2, axis=1))
    print(f"  rk4 sin saturación dt = {dt*1e3:5.2f} ms: error máx. de posición {res_int[('rk4ns', dt)]:.3e} m")
SATURA = True
EVALS["rk4ns"] = 4
for met in ("euler", "semi", "rk4", "rk4ns"):
    e1, e2 = res_int[(met, 1e-3)], res_int[(met, 4e-3)]
    print(f"  pendiente {met} entre 1 y 4 ms: {math.log(e2/e1)/math.log(4):.2f}")

# figura: error frente a evaluaciones por segundo simulado (log-log)
X1 = lin(1.5, 4.7, 60, 520)   # log10(evaluaciones/s)
Y1 = lin(-12, 0, 250, 20)     # log10(error en m)
for met in ("euler", "semi", "rk4", "rk4ns"):
    xs = [math.log10(EVALS[met] / dt) for dt in DTS]
    ys = [math.log10(res_int[(met, dt)]) for dt in DTS]
    print(f"path int {met}:", path(xs, ys, X1, Y1))
    for dt, xv, yv in zip(DTS, xs, ys):
        if dt in (1e-3, 4e-3, 10e-3):
            print(f"  punto {met} {dt*1e3:.0f} ms: ({X1(xv):.1f}, {Y1(yv):.1f})")
for e in range(-12, 1, 3):
    print(f"  marca y 1e{e}: y = {Y1(e):.1f}")
for e in (2, 3, 4):
    print(f"  marca x 1e{e}: x = {X1(e):.1f}")

# ---------------------------------------------------------------- 2. tres convenciones de planta
print("=" * 70)
print("2. MISMO DRON Y MISMO CONTROLADOR (250 Hz, retención de orden cero) CON TRES CONVENCIONES DE PLANTA")
print("   Planta integrada con RK4 a 2 kHz (8 subpasos por periodo de control): solo cambia el modelo.")
F_CTRL, SUB = 250, 8


def simula(modelo, ref, T, x0):
    dtc = 1.0 / F_CTRL
    h = dtc / SUB
    x = x0.copy()
    out = []
    for k in range(int(round(T * F_CTRL))):
        t = k * dtc
        fc = controlador(x, ref(t))
        for _ in range(SUB):
            k1 = deriv(x, fc, modelo)
            k2 = deriv(x + 0.5 * h * k1, fc, modelo)
            k3 = deriv(x + 0.5 * h * k2, fc, modelo)
            k4 = deriv(x + h * k3, fc, modelo)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x[6:10] /= np.linalg.norm(x[6:10])
        R = q2R(x[6:10])
        tilt = math.degrees(math.acos(max(-1, min(1, R[2, 2]))))
        out.append((t + dtc, *x[0:6], tilt))
    return np.array(out)


MODELOS = [("instantaneo", "Empuje instantáneo (Menagerie)"), ("gazebo", "Gazebo x500"), ("sih", "SIH de PX4")]
print("-" * 70)
print("   Escalón de alabeo de 20° (solo lazos de actitud y rate; empuje T = m g / cos(phi))")


def controlador_actitud(x, phi_ref):
    q, w = x[6:10], x[10:13]
    R = q2R(q)
    c, s = math.cos(phi_ref), math.sin(phi_ref)
    Rd = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    E = 0.5 * (Rd.T @ R - R.T @ Rd)
    eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
    wd = -KR * eR
    tau = J * (-KW * (w - wd)) + np.cross(w, J * w)
    T = M * G / max(0.5, R[2, 2])
    f = ALLOC_INV @ np.array([T, *tau])
    return np.clip(f, KT * OM_MIN ** 2, KT * OM_MAX ** 2)


def simula_act(modelo, phi_ref, T):
    dtc = 1.0 / F_CTRL
    h = dtc / SUB
    x = estado_inicial(modelo)
    out = []
    for k in range(int(round(T * F_CTRL))):
        fc = controlador_actitud(x, phi_ref)
        for _ in range(SUB):
            k1 = deriv(x, fc, modelo)
            k2 = deriv(x + 0.5 * h * k1, fc, modelo)
            k3 = deriv(x + 0.5 * h * k2, fc, modelo)
            k4 = deriv(x + h * k3, fc, modelo)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x[6:10] /= np.linalg.norm(x[6:10])
        R = q2R(x[6:10])
        out.append(((k + 1) * dtc, math.degrees(math.atan2(R[2, 1], R[2, 2]))))
    return np.array(out)


PHI = math.radians(20)
res_act = {}
for mod, nombre in MODELOS:
    r = simula_act(mod, PHI, 1.0)
    res_act[mod] = r
    ph = r[:, 1]
    sob = (ph.max() - 20) / 20 * 100
    fuera = np.where(np.abs(ph - 20) > 0.4)[0]
    ts = r[fuera[-1], 0] if len(fuera) else 0.0
    t90 = r[np.argmax(ph >= 18), 0]
    print(f"  {nombre:32s}: sobreoscilación {sob:5.1f} %; t(90 %) {t90*1e3:5.0f} ms; establecimiento (2 %) {ts*1e3:5.0f} ms")
X3 = lin(0, 1.0, 60, 560)
Y3 = lin(-2, 26, 250, 20)
for mod, _ in MODELOS:
    r = res_act[mod]
    print(f"path act {mod}:", path(np.r_[0, r[:, 0]], np.r_[0, r[:, 1]], X3, Y3))
print(f"  ejes: x(0,25 s)={X3(0.25):.0f}; y(0)={Y3(0):.1f}, y(10)={Y3(10):.1f}, y(20)={Y3(20):.1f}")

ESC = 2.0
ref_esc = lambda t: (np.array([ESC, 0, 0]), np.zeros(3), np.zeros(3))
res_esc = {}
for mod, nombre in MODELOS:
    r = simula(mod, ref_esc, 5.0, estado_inicial(mod))
    res_esc[mod] = r
    x = r[:, 1]
    sob = (x.max() - ESC) / ESC * 100
    fuera = np.where(np.abs(x - ESC) > 0.02 * ESC)[0]
    ts = r[fuera[-1], 0] if len(fuera) else 0.0
    t90 = r[np.argmax(x >= 0.9 * ESC), 0]
    print(f"  {nombre:32s}: sobreoscilación {sob:5.1f} %; t(90 %) {t90:.2f} s; establecimiento (2 %) {ts:.2f} s; "
          f"inclinación máx. {r[:, 7].max():5.1f}°; |z| máx. {np.abs(r[:, 3]).max()*100:.1f} cm")

X2 = lin(0, 5, 60, 560)
Y2 = lin(-0.2, 2.6, 250, 20)
for mod, _ in MODELOS:
    r = res_esc[mod]
    sel = slice(0, None, 5)
    print(f"path esc {mod}:", path(np.r_[0, r[sel, 0]], np.r_[0, r[sel, 1]], X2, Y2))
print(f"  ejes: x(0)={X2(0):.0f}, x(1)={X2(1):.0f} ... y(0)={Y2(0):.1f}, y(1)={Y2(1):.1f}, y(2)={Y2(2):.1f}")

print("-" * 70)
print("   Vuelo recto estabilizado a 5 m/s: inclinación y velocidad de rotor necesarias")
VF = 5.0
ref_v = lambda t: (np.array([VF * t, 0, 0]), np.array([VF, 0, 0]), np.zeros(3))
for mod, nombre in MODELOS:
    r = simula(mod, ref_v, 8.0, estado_inicial(mod, v0=(VF, 0, 0)))
    tilt = r[-50:, 7].mean()
    ex = (r[-50:, 1] - VF * r[-50:, 0]).mean()
    print(f"  {nombre:32s}: inclinación estacionaria {tilt:5.2f}°; error de seguimiento en x {ex*100:6.2f} cm")
# cuentas cerradas del arrastre
cdg = 4 * CD_ROT * OM_HOVER
print(f"  arrastre de rotor de Gazebo en hover: 4 c_d Omega_h = {cdg:.3f} N/(m/s) -> {cdg/M:.3f} 1/s por unidad de masa")
print(f"  inclinación con arrastre lineal D v: atan(D v / m g): Gazebo {math.degrees(math.atan(cdg*VF/(M*G))):.2f}°, "
      f"SIH {math.degrees(math.atan(1.0*VF/(M*G))):.2f}°")
print(f"  SIH con su masa por defecto (1 kg): KDV/m = {1.0/1.0:.2f} 1/s; aplicado al x500: {1.0/M:.3f} 1/s")
print("  Faessler et al. (2018), dron de 610 g: d_x = 0,544 y d_y = 0,386 1/s en un círculo (A1)")

# ---------------------------------------------------------------- 3. paso de física y frecuencias de lazo
print("=" * 70)
print("3. PASOS Y FRECUENCIAS POR DEFECTO (leídos en el código)")
for nombre, dt in [("Gazebo, mundo default.sdf de PX4 (ODE)", 0.004), ("MuJoCo Menagerie x2.xml", 0.01),
                   ("MuJoCo, timestep por defecto", 0.002), ("SIH de PX4 multirrotor: min(IMU_GYRO_RATEMAX=800, IMU_INTEG_RATE=200)", 1 / 200)]:
    print(f"  {nombre:70s}: {dt*1e3:5.1f} ms -> {1/dt:6.0f} Hz")
print(f"  tau del motor 12,5 ms -> pasos por constante de tiempo con 4 ms: {12.5/4:.1f}; con 10 ms: {12.5/10:.2f}")
print(f"  Omega_hover = {OM_HOVER:.1f} rad/s; aviso de aliasing de Gazebo si Omega_junta/(2 pi) > 1/(2 dt): "
      f"con dt = 4 ms, límite {2*math.pi/(2*0.004):.0f} rad/s de junta; con slowdown 10, Omega real hasta {10*2*math.pi/(2*0.004):.0f} rad/s")

# ---------------------------------------------------------------- 4. cifras de rendimiento publicadas
print("=" * 70)
print("4. RENDIMIENTO PUBLICADO (cada cifra en su hardware y con su definición de paso; no son comparables)")
FIS = [  # (etiqueta, valor, condiciones)
    ("gym-pybullet-drones", 240 * 16.8, "1 dron a 240 Hz, 16,8 veces el tiempo real (CPU de portátil); pasos = 240 x 16,8"),
    ("Flightmare", 2.0e5, "150 drones, acciones aleatorias, CPU i7-8850H"),
    ("MuJoCo (2012)", 4.0e5, "humanoide de 18 GDL, 12 núcleos (evaluaciones de la dinámica)"),
    ("Isaac Gym", 5.4e5, "Ant, 4096 agentes, A100, durante el entrenamiento"),
    ("OmniDrones", 732109, "Track, 4096 entornos, RTX 4090, con observaciones y recompensas"),
    ("Aerial Gym", 4.43e6, "2^16 robots, RPM constantes, RTX 3090"),
    ("Genesis: blog de Tao", 2.9e5, "Franka moviéndose, RTX 4090 (no revisado por pares)"),
    ("Genesis: README", 4.3e7, "Franka, RTX 4090 (afirmación del equipo, dic. 2024)"),
    ("L2F", 1.4063e10, "2^16, GPU de portátil (dado por sus autores; tabla de Aerial Gym)"),
]
REN = [
    ("gym-pybullet-drones", 16.70, "2^4 entornos con cámara (TinyRenderer)"),
    ("Flightmare", 230, "5 drones con RGB, escena Garage (Unity)"),
    ("OmniDrones", 307, "2^8 entornos, cámara nativa de Isaac Sim"),
    ("Aerial Gym", 3921, "2^10 entornos, profundidad y segmentación 270x480 por ray casting (Warp)"),
    ("MuJoCo Playground", 4.03e5, "Cartpole con píxeles, renderizador por lotes Madrona, A100"),
]
X4 = lin(1, 11, 250, 1180)
for i, (n, v, c) in enumerate(FIS):
    print(f"  fis {i}: {n:22s} {v:12.4g}  x = {X4(math.log10(v)):6.1f}   ({c})")
for i, (n, v, c) in enumerate(REN):
    print(f"  ren {i}: {n:22s} {v:12.4g}  x = {X4(math.log10(v)):6.1f}   ({c})")
for e in range(1, 12):
    print(f"  marca 1e{e}: x = {X4(e):.1f}")
print(f"  Aerial Gym: física / render = {4.43e6/3921:.0f} (mismo artículo, mismo hardware, distinto número de entornos)")
print(f"  Genesis: README / blog = {4.3e7/2.9e5:.0f}")
print(f"  L2F, '5 meses de vuelo por segundo': factor de tiempo real {5*30*86400:.2e}")
