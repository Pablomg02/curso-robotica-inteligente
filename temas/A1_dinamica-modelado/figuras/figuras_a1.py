"""Datos de las figuras y tablas del tema A1 (dinámica y modelado).

Uso:  python3 figuras_a1.py > salida.txt
Requiere numpy. Imprime cifras para las tablas y trazados SVG (atributo d de <path>
o coordenadas de <circle>) ya escalados al viewBox de cada figura.

Fuentes de parámetros (verificadas en su origen):
- x500 de PX4 (Gazebo): PX4-gazebo-models/models/x500_base y x500/model.sdf
- Cuadricóptero de NeuroBEM: Bauersfeld et al. (RSS 2021) y Readme del dataset
- Agilicious: Foehn et al. (Science Robotics 2022)
- Crazyflie 2.1: Eschmann et al. (IROS 2024) y tabla de plataformas de RAPTOR (2025)
- Crazyflie Brushless: Gräfe et al. (arXiv 2603.05944, 2026)
- Aerosonde: byu-magicc/mavsim_public, parameters/aerosonde_parameters.py (Beard y McLain)
- Arrastre de rotor: Faessler et al. (RA-L 2018)
"""
import math
import numpy as np

G = 9.81
RHO = 1.225


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def path(xs, ys, X, Y):
    """Convierte datos a un atributo d de SVG con las funciones de escala X, Y."""
    pts = [f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys)]
    return "M" + " L".join(pts)


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


print("=" * 70)
print("1. TEORÍA DE LA CANTIDAD DE MOVIMIENTO (figura y cifras)")
# Cuadricóptero de NeuroBEM: 752 g (artículo), hélices de 5 pulgadas
m_nb = 0.752
R_nb = 5 * 0.0254 / 2
T_h = m_nb * G / 4
A = math.pi * R_nb**2
v_h = math.sqrt(T_h / (2 * RHO * A))
P_id = T_h * v_h
print(f"NeuroBEM quad: T_h por rotor = {T_h:.3f} N, R = {R_nb*100:.2f} cm, A = {A*1e4:.1f} cm2")
print(f"  v_h = {v_h:.2f} m/s, P_ideal por rotor = {P_id:.1f} W, total = {4*P_id:.1f} W")
for fm in (0.5, 0.6, 0.7):
    print(f"  con FM = {fm}: potencia mecánica total = {4*P_id/fm:.0f} W")

# Figura: v_i/v_h frente a V_c/v_h. viewBox 0 0 520 300; área x 60..500, y 30..250
X = lin(-4, 3, 60, 500)
Y = lin(0, 1.6, 250, 30)
Vc = np.linspace(0, 3, 61)
climb = -Vc / 2 + np.sqrt((Vc / 2) ** 2 + 1)
Vd = np.linspace(-4, -2, 41)
wind = -Vd / 2 - np.sqrt(np.maximum((Vd / 2) ** 2 - 1, 0))
print("path subida:", path(Vc, climb, X, Y))
print("path molinete:", path(Vd, wind, X, Y))
print("zona sin solución x:", fmt(X(-2)), "a", fmt(X(0)))
print("ejes: x(-4,-3,...,3) ->", [fmt(X(v)) for v in range(-4, 4)])
print("ejes: y(0,0.4,...,1.6) ->", [fmt(Y(v)) for v in (0, 0.4, 0.8, 1.2, 1.6)])

print("=" * 70)
print("2. EFECTO SUELO (Cheeseman y Bennett, 1955)")
X = lin(0.5, 4, 60, 500)
Y = lin(1.0, 1.35, 250, 30)
zr = np.linspace(0.5, 4, 71)
ratio = 1 / (1 - (1 / (4 * zr)) ** 2)
print("path:", path(zr, ratio, X, Y))
for z in (0.5, 1, 1.5, 2, 3):
    r = 1 / (1 - (1 / (4 * z)) ** 2)
    print(f"  z/R = {z}: T_IGE/T_OGE = {r:.3f}  -> punto ({X(z):.1f}, {Y(r):.1f})")
print("ejes x:", [(v, fmt(X(v))) for v in (0.5, 1, 2, 3, 4)])
print("ejes y:", [(v, fmt(Y(v))) for v in (1.0, 1.1, 1.2, 1.3)])

print("=" * 70)
print("3. EJE INTERMEDIO: cuerpo rígido sin par (inercia de NeuroBEM)")
J = np.array([0.0025, 0.0021, 0.0043])  # Readme del dataset NeuroBEM


def euler_rhs(w):
    Jw = J * w
    return -np.cross(w, Jw) / J


w = np.array([20.0, 0.02, 0.02])
dt, T = 1e-4, 4.0
n = int(T / dt)
ts, W = [0.0], [w.copy()]
E0 = 0.5 * np.sum(J * w**2)
L0 = np.linalg.norm(J * w)
for k in range(n):
    k1 = euler_rhs(w)
    k2 = euler_rhs(w + 0.5 * dt * k1)
    k3 = euler_rhs(w + 0.5 * dt * k2)
    k4 = euler_rhs(w + dt * k3)
    w = w + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if (k + 1) % 200 == 0:
        ts.append((k + 1) * dt)
        W.append(w.copy())
W = np.array(W)
ts = np.array(ts)
E1 = 0.5 * np.sum(J * w**2)
L1 = np.linalg.norm(J * w)
lam = 20 * math.sqrt((J[0] - J[1]) * (J[2] - J[0]) / (J[1] * J[2]))
print(f"tasa de crecimiento lineal lambda = {lam:.2f} 1/s; e-folding = {1/lam*1000:.0f} ms")
print(f"conservación: dE/E = {(E1-E0)/E0:.2e}, d|L|/|L| = {(L1-L0)/L0:.2e}")
i_flip = np.argmax(W[:, 0] < 0)
print(f"primer cruce de omega_x por cero: t = {ts[i_flip]:.2f} s")
# viewBox 0 0 600 260; área x 50..580, y 20..220
X = lin(0, 4, 50, 580)
Y = lin(-22, 22, 220, 20)
for i, nm in enumerate(["wx", "wy", "wz"]):
    print(f"path {nm}:", path(ts, W[:, i], X, Y))
print("ejes x:", [(v, fmt(X(v))) for v in range(0, 5)])
print("ejes y:", [(v, fmt(Y(v))) for v in (-20, -10, 0, 10, 20)])
print("max |wy|, |wz|:", np.max(np.abs(W[:, 1])), np.max(np.abs(W[:, 2])))

print("=" * 70)
print("4. PLANITUD: demanda de empuje y velocidad de inclinación en una lemniscata")
# Lemniscata de Gerono horizontal: x = A cos(th), y = A sin(th) cos(th), th = W t
A_lem = 5.0
c_max = 4 * 9.5 / 0.752  # Agilicious: 4 x 9.5 N continuos, 752 g
print(f"c_max (Agilicious) = {c_max:.1f} m/s2 = {c_max/G:.2f} g")
d_iso = (0.544 + 0.386) / 2  # Faessler, círculo: media de dx y dy
print(f"arrastre isótropo d = {d_iso:.3f} 1/s")


def demands(vmax, d):
    th = np.linspace(0, 2 * math.pi, 4001)
    # velocidad máxima en función de W: calcula numéricamente
    sp = np.sqrt(np.sin(th) ** 2 + np.cos(2 * th) ** 2)  # |v| / (A W)
    Wr = vmax / (A_lem * sp.max())
    t = th / Wr
    x = A_lem * np.cos(th)
    y = A_lem * np.sin(th) * np.cos(th)
    vx = -A_lem * Wr * np.sin(th)
    vy = A_lem * Wr * np.cos(2 * th)
    ax = -A_lem * Wr**2 * np.cos(th)
    ay = -2 * A_lem * Wr**2 * np.sin(2 * th)
    tx, ty, tz = ax + d * vx, ay + d * vy, G + 0 * ax
    c = np.sqrt(tx**2 + ty**2 + tz**2)
    zb = np.stack([tx, ty, tz]) / c
    dz = np.gradient(zb, t, axis=1)
    rate = np.linalg.norm(dz, axis=0)  # velocidad angular de inclinación (rad/s)
    # error de actitud si se ignora el arrastre: ángulo entre z_B con y sin arrastre
    t0x, t0y = ax, ay
    c0 = np.sqrt(t0x**2 + t0y**2 + G**2)
    z0 = np.stack([t0x, t0y, G + 0 * ax]) / c0
    ang = np.degrees(np.arccos(np.clip(np.sum(z0 * zb, axis=0), -1, 1)))
    return c.max(), rate.max(), ang.max()


vs = np.linspace(1, 24, 47)
res0 = np.array([demands(v, 0.0) for v in vs])
resd = np.array([demands(v, d_iso) for v in vs])
for v, a, b in zip(vs[::4], res0[::4], resd[::4]):
    print(f"  v_max {v:5.1f} m/s: c sin arrastre {a[0]:6.1f}, con arrastre {b[0]:6.1f} m/s2;"
          f" giro {math.degrees(a[1]):6.0f} / {math.degrees(b[1]):6.0f} deg/s;"
          f" error de actitud por ignorar arrastre {b[2]:5.1f} deg")


def cross(vs, ys, lvl):
    for i in range(1, len(vs)):
        if ys[i - 1] < lvl <= ys[i]:
            return vs[i - 1] + (lvl - ys[i - 1]) * (vs[i] - vs[i - 1]) / (ys[i] - ys[i - 1])
    return None


v0 = cross(vs, res0[:, 0], c_max)
vd = cross(vs, resd[:, 0], c_max)
print(f"saturación de empuje: sin arrastre a {v0:.1f} m/s, con arrastre a {vd:.1f} m/s")
# panel A: viewBox total 0 0 700 300; cada panel área x 50..330, y 20..230 ; c de 0 a 100, v de 0 a 20
X = lin(0, 20, 50, 330)
Y = lin(0, 100, 230, 20)
sel = vs <= 20.0
print("A path c sin:", path(vs[sel], res0[sel, 0], X, Y))
print("A path c con:", path(vs[sel], resd[sel, 0], X, Y))
print("A y c_max:", fmt(Y(c_max)), " cruces x:", fmt(X(v0)), fmt(X(vd)))
print("A ejes x:", [(v, fmt(X(v))) for v in (0, 4, 8, 12, 16, 20)])
print("A ejes y:", [(v, fmt(Y(v))) for v in (0, 20, 40, 60, 80, 100)])
# panel B: error de actitud por ignorar el arrastre (deg) de 0 a 25, y velocidad de inclinación
Yb = lin(0, 60, 230, 20)
vs2 = np.linspace(0.2, 20, 100)
err = np.array([demands(v, d_iso)[2] for v in vs2])
ok = vs2 <= 4.0
print("B path error actitud (identificado, <= 4 m/s):", path(vs2[ok], err[ok], X, Yb))
print("B path error actitud (extrapolado):", path(vs2[~ok | (np.arange(len(vs2)) == np.where(ok)[0][-1])], err[~ok | (np.arange(len(vs2)) == np.where(ok)[0][-1])], X, Yb))
print("B ejes y:", [(v, fmt(Yb(v))) for v in (0, 15, 30, 45, 60)])
print("B punto 4 m/s:", fmt(X(4)), fmt(Yb(np.interp(4, vs2, err))), "err", np.interp(4, vs2, err))
for vq in (5, 10, 15):
    i = int(np.argmin(np.abs(vs - vq)))
    print(f"  a {vs[i]:.1f} m/s: error de actitud {resd[i,2]:.1f} deg; giro necesario {math.degrees(res0[i,1]):.0f} deg/s")

print("=" * 70)
print("5. MOTORES: ancho de banda y retardo de fase de un primer orden")
motors = [
    ("x500 Gazebo, subida", 0.0125),
    ("x500 Gazebo, bajada", 0.025),
    ("Agilicious (banco)", 0.0391),
    ("Crazyflie Brushless", 0.050),
    ("Crazyflie 2.1", 0.072),
]
for nm, tau in motors:
    fb = 1 / (2 * math.pi * tau)
    ph10 = math.degrees(math.atan(2 * math.pi * 10 * tau))
    ph30 = math.degrees(math.atan(2 * math.pi * 30 * tau))
    print(f"  {nm:24s} tau = {tau*1000:5.1f} ms  f_-3dB = {fb:5.1f} Hz  fase@10Hz = {ph10:4.0f}  fase@30Hz = {ph30:4.0f}")
for f in (10, 30):
    # retardo puro equivalente a baja frecuencia = tau
    pass

print("=" * 70)
print("6. ESCALA: aceleración angular máxima de alabeo (dos motores a tope, dos a cero)")
plats = {
    # nombre: (masa kg, empuje máx por rotor N, brazo de alabeo m, Ixx kg m2, fuente)
    "Crazyflie 2.1": (0.027, 0.0213 - 0.0112 + 0.1201, 0.080 / 2 / math.sqrt(2), 1.4e-5),
    "Crazyflie Brushless": (0.044, -0.23 + 0.562 - 0.043, 0.0707 / 2, 3.3e-5),
    "x500 (Gazebo)": (2.0 + 4 * 0.016077, 8.54858e-6 * 1000**2, 0.174, 0.021667),
}
for nm, (m, Tm, arm, Ixx) in plats.items():
    alpha = 2 * Tm * arm / Ixx
    tw = 4 * Tm / (m * G)
    print(f"  {nm:20s} m = {m*1000:6.0f} g  T/W = {tw:4.2f}  brazo = {arm*1000:5.1f} mm"
          f"  alpha_max = {alpha:6.0f} rad/s2  t_90 = {math.sqrt(2*(math.pi/2)/alpha)*1000:4.0f} ms")
L_ratio = (plats["x500 (Gazebo)"][0] / plats["Crazyflie 2.1"][0]) ** (1 / 3)
print(f"  cociente de tamaño (masa^(1/3)) x500/CF2.1 = {L_ratio:.2f};"
      f" Froude predice alpha x{L_ratio:.1f}, Mach x{L_ratio**2:.1f}")

print("=" * 70)
print("7. x500: vuelo estacionario y matriz de asignación (convenio de PX4, FRD)")
m500 = 2.0 + 4 * 0.016076923
kT = 8.54858e-6
Th = m500 * G / 4
Om = math.sqrt(Th / kT)
print(f"  masa total = {m500:.3f} kg; T_h = {Th:.3f} N por rotor; Omega_h = {Om:.0f} rad/s = {Om*60/2/math.pi:.0f} rpm")
print(f"  empuje máximo = {kT*1e6:.2f} N por rotor; T/W = {4*kT*1e6/(m500*G):.2f}")
P = [(0.174, 0.174, 0.05), (-0.174, -0.174, 0.05), (0.174, -0.174, -0.05), (-0.174, 0.174, -0.05)]
M = np.array([[1, 1, 1, 1],
              [-py for px, py, km in P],
              [px for px, py, km in P],
              [km for px, py, km in P]], dtype=float)
print("  M =\n", M)
print("  M^-1 =\n", np.round(np.linalg.inv(M), 4))
dm = 4 * 8.06428e-5 * Om / m500
print(f"  arrastre de rotor en Gazebo (masa-normalizado) en hover: d = {dm:.3f} 1/s")
print(f"  cociente KM asignador / momentConstant planta = {0.05/0.016:.2f}")

print("=" * 70)
print("8. ALA FIJA: Aerosonde (parámetros de mavsim_public) a Va = 25 m/s")
P = dict(mass=11.0, Jx=0.8244, Jy=1.135, Jz=1.759, Jxz=0.1204, S=0.55, b=2.8956, c=0.18994,
         rho=1.2682, e=0.9, CL0=0.23, CD0=0.0424, Cm0=0.0135, CLa=5.61, CDa=0.132, Cma=-2.74,
         CLq=7.95, CDq=0.0, Cmq=-38.21, CLde=0.13, CDde=0.0135, Cmde=-0.99, M=50.0, a0=0.47,
         CDp=0.043, CY0=0.0, Cl0=0.0, Cn0=0.0, CYb=-0.98, Clb=-0.13, Cnb=0.073, CYp=0.0,
         Clp=-0.51, Cnp=0.069, CYr=0.0, Clr=0.25, Cnr=-0.095, CYda=0.075, Clda=0.17,
         Cnda=-0.011, CYdr=0.19, Cldr=0.0024, Cndr=-0.069, D=20 * 0.0254, KV=145.0,
         R=0.042, i0=1.5, ncells=12, CQ2=-0.01664, CQ1=0.004970, CQ0=0.005230,
         CT2=-0.1079, CT1=-0.06044, CT0=0.09357)
P["AR"] = P["b"] ** 2 / P["S"]
Jx, Jy, Jz, Jxz = P["Jx"], P["Jy"], P["Jz"], P["Jxz"]
Gm = Jx * Jz - Jxz**2
g1 = Jxz * (Jx - Jy + Jz) / Gm
g2 = (Jz * (Jz - Jy) + Jxz**2) / Gm
g3, g4 = Jz / Gm, Jxz / Gm
g5, g6 = (Jz - Jx) / Jy, Jxz / Jy
g7 = ((Jx - Jy) * Jx + Jxz**2) / Gm
g8 = Jx / Gm
KQ = (1 / P["KV"]) * 60 / (2 * math.pi)


def prop(Va, dt):
    Vin = 3.7 * P["ncells"] * dt
    Dp, rho = P["D"], P["rho"]
    a = rho * Dp**5 * P["CQ0"] / (2 * math.pi) ** 2
    b = rho * Dp**4 * P["CQ1"] * Va / (2 * math.pi) + KQ**2 / P["R"]
    c = rho * Dp**3 * P["CQ2"] * Va**2 - KQ * Vin / P["R"] + KQ * P["i0"]
    Om = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    Jadv = 2 * math.pi * Va / (Om * Dp)
    CT = P["CT2"] * Jadv**2 + P["CT1"] * Jadv + P["CT0"]
    CQ = P["CQ2"] * Jadv**2 + P["CQ1"] * Jadv + P["CQ0"]
    nn = Om / (2 * math.pi)
    return rho * nn**2 * Dp**4 * CT, rho * nn**2 * Dp**5 * CQ


def f6(x, u):
    """x = [u, v, w, phi, theta, psi, p, q, r]; u = [de, da, dr, dt]. Sin viento."""
    uu, vv, ww, ph, th, ps, p, q, r = x
    de, da, dr, dth = u
    Va = math.sqrt(uu * uu + vv * vv + ww * ww)
    al = math.atan2(ww, uu)
    be = math.asin(vv / Va)
    qbar = 0.5 * P["rho"] * Va**2 * P["S"]
    s = (1 + math.exp(-P["M"] * (al - P["a0"])) + math.exp(P["M"] * (al + P["a0"]))) / (
        (1 + math.exp(-P["M"] * (al - P["a0"]))) * (1 + math.exp(P["M"] * (al + P["a0"]))))
    CL = (1 - s) * (P["CL0"] + P["CLa"] * al) + s * (2 * math.copysign(1, al) * math.sin(al) ** 2 * math.cos(al))
    CD = P["CDp"] + (P["CL0"] + P["CLa"] * al) ** 2 / (math.pi * P["e"] * P["AR"])
    cq = P["c"] / (2 * Va)
    bq = P["b"] / (2 * Va)
    FL = qbar * (CL + P["CLq"] * cq * q + P["CLde"] * de)
    FD = qbar * (CD + P["CDq"] * cq * q + P["CDde"] * de)
    Tp, Qp = prop(Va, dth)
    m = P["mass"]
    fx = -math.cos(al) * FD + math.sin(al) * FL + Tp - m * G * math.sin(th)
    fz = -math.sin(al) * FD - math.cos(al) * FL + m * G * math.cos(th) * math.cos(ph)
    fy = qbar * (P["CY0"] + P["CYb"] * be + P["CYp"] * bq * p + P["CYr"] * bq * r
                 + P["CYda"] * da + P["CYdr"] * dr) + m * G * math.cos(th) * math.sin(ph)
    Ml = qbar * P["b"] * (P["Cl0"] + P["Clb"] * be + P["Clp"] * bq * p + P["Clr"] * bq * r
                          + P["Clda"] * da + P["Cldr"] * dr) - Qp
    Mm = qbar * P["c"] * (P["Cm0"] + P["Cma"] * al + P["Cmq"] * cq * q + P["Cmde"] * de)
    Mn = qbar * P["b"] * (P["Cn0"] + P["Cnb"] * be + P["Cnp"] * bq * p + P["Cnr"] * bq * r
                          + P["Cnda"] * da + P["Cndr"] * dr)
    du = r * vv - q * ww + fx / m
    dv = p * ww - r * uu + fy / m
    dw = q * uu - p * vv + fz / m
    dph = p + q * math.sin(ph) * math.tan(th) + r * math.cos(ph) * math.tan(th)
    dth_ = q * math.cos(ph) - r * math.sin(ph)
    dps = (q * math.sin(ph) + r * math.cos(ph)) / math.cos(th)
    dp = g1 * p * q - g2 * q * r + g3 * Ml + g4 * Mn
    dq = g5 * p * r - g6 * (p * p - r * r) + Mm / Jy
    dr_ = g7 * p * q - g1 * q * r + g4 * Ml + g8 * Mn
    return np.array([du, dv, dw, dph, dth_, dps, dp, dq, dr_])


Va0 = 25.0


def trim_res(z):
    al, de, dt, da, dr, ph = z
    x = np.array([Va0 * math.cos(al), 0.0, Va0 * math.sin(al), ph, al, 0.0, 0.0, 0.0, 0.0])
    fx = f6(x, [de, da, dr, dt])
    return np.array([fx[0], fx[1], fx[2], fx[6], fx[7], fx[8]])


z = np.array([0.05, -0.1, 0.5, 0.0, 0.0, 0.0])
for it in range(50):
    r0 = trim_res(z)
    Jn = np.zeros((6, 6))
    for k in range(6):
        dz = np.zeros(6)
        dz[k] = 1e-6
        Jn[:, k] = (trim_res(z + dz) - r0) / 1e-6
    z = z - np.linalg.solve(Jn, r0)
    if np.linalg.norm(r0) < 1e-10:
        break
al, de, dt, da, dr, ph = z
print(f"  trim: alpha = {math.degrees(al):.2f} deg, de = {math.degrees(de):.2f} deg, dt = {dt:.3f},"
      f" da = {math.degrees(da):.3f} deg, dr = {math.degrees(dr):.3f} deg, phi = {math.degrees(ph):.3f} deg;"
      f" residuo {np.linalg.norm(trim_res(z)):.1e}")
Tp0, Qp0 = prop(Va0, dt)
print(f"  empuje de la hélice en trim = {Tp0:.1f} N; par = {Qp0:.2f} N m")
x0 = np.array([Va0 * math.cos(al), 0.0, Va0 * math.sin(al), ph, al, 0.0, 0.0, 0.0, 0.0])
u0 = np.array([de, da, dr, dt])
Ax = np.zeros((9, 9))
Bu = np.zeros((9, 4))
for k in range(9):
    dx = np.zeros(9)
    dx[k] = 1e-6
    Ax[:, k] = (f6(x0 + dx, u0) - f6(x0 - dx, u0)) / 2e-6
for k in range(4):
    du = np.zeros(4)
    du[k] = 1e-6
    Bu[:, k] = (f6(x0, u0 + du) - f6(x0, u0 - du)) / 2e-6
lon = [0, 2, 7, 4]  # u, w, q, theta
lat = [1, 6, 8, 3]  # v, p, r, phi
for nm, idx in (("longitudinal", lon), ("lateral", lat)):
    ev = np.linalg.eigvals(Ax[np.ix_(idx, idx)])
    print(f"  modos {nm}:")
    for e in sorted(ev, key=lambda c: -abs(c)):
        wn = abs(e)
        zeta = -e.real / wn if wn > 0 else float("nan")
        per = 2 * math.pi / abs(e.imag) if abs(e.imag) > 1e-9 else float("nan")
        tc = -1 / e.real if abs(e.imag) < 1e-9 else float("nan")
        print(f"    lambda = {e.real:8.4f} {e.imag:+8.4f}j  wn = {wn:6.3f} rad/s  zeta = {zeta:6.3f}"
              f"  periodo = {per:6.2f} s  tau = {tc:7.3f} s")
wl = math.sqrt(2) * G / Va0
print(f"  Lanchester: omega_ph = sqrt(2) g / V = {wl:.3f} rad/s; periodo = {2*math.pi/wl:.2f} s")
evf = np.linalg.eigvals(Ax)
print("  autovalores del modelo completo de 9 estados:", np.round(np.sort_complex(evf), 4))

# Respuesta a un doblete de profundidad (+-2 grados, 0,5 s cada medio ciclo), modelo lineal longitudinal
Al = Ax[np.ix_(lon, lon)]
Bl = Bu[lon, 0]
dtt = 0.002
Tsim = 60.0
N = int(Tsim / dtt)
xs = np.zeros(4)
out_t, out_q, out_V = [], [], []
for k in range(N + 1):
    t = k * dtt
    dlt = math.radians(2.0) if t < 0.5 else (-math.radians(2.0) if t < 1.0 else 0.0)
    if k % 10 == 0:
        uu = x0[0] + xs[0]
        ww = x0[2] + xs[1]
        out_t.append(t)
        out_q.append(math.degrees(xs[2]))
        out_V.append(math.sqrt(uu * uu + ww * ww) - Va0)

    def rhs(s):
        return Al @ s + Bl * dlt
    k1 = rhs(xs)
    k2 = rhs(xs + 0.5 * dtt * k1)
    k3 = rhs(xs + 0.5 * dtt * k2)
    k4 = rhs(xs + dtt * k3)
    xs = xs + dtt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
out_t, out_q, out_V = map(np.array, (out_t, out_q, out_V))
print(f"  doblete: q en [{out_q.min():.2f}, {out_q.max():.2f}] deg/s; dVa en [{out_V.min():.3f}, {out_V.max():.3f}] m/s")
# Fugoide: perturbación inicial de velocidad du = +2 m/s, sin mando
xs = np.array([2.0, 0.0, 0.0, 0.0])
out_V = []
for k in range(N + 1):
    if k % 10 == 0:
        uu = x0[0] + xs[0]
        ww = x0[2] + xs[1]
        out_V.append(math.sqrt(uu * uu + ww * ww) - Va0)
    k1 = Al @ xs
    k2 = Al @ (xs + 0.5 * dtt * k1)
    k3 = Al @ (xs + 0.5 * dtt * k2)
    k4 = Al @ (xs + dtt * k3)
    xs = xs + dtt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
out_V = np.array(out_V)
print(f"  fugoide (du0 = 2 m/s): dVa en [{out_V.min():.3f}, {out_V.max():.3f}] m/s")
# panel izquierdo: q, 0..4 s ; viewBox 0 0 430 250, área x 55..415, y 20..200
sel = out_t <= 4.0
qlim = math.ceil(max(abs(out_q.min()), abs(out_q.max())) / 5) * 5
X = lin(0, 4, 55, 415)
Y = lin(-qlim, qlim, 200, 20)
print("  qlim =", qlim)
print("  path q:", path(out_t[sel][::2], out_q[sel][::2], X, Y))
print("  ejes x:", [(v, fmt(X(v))) for v in range(0, 5)])
print("  ejes y:", [(v, fmt(Y(v))) for v in (-qlim, -qlim / 2, 0, qlim / 2, qlim)])
Vlim = math.ceil(max(abs(out_V.min()), abs(out_V.max())) * 10) / 10
X = lin(0, 60, 55, 415)
Y = lin(-Vlim, Vlim, 200, 20)
print("  Vlim =", Vlim)
print("  path V:", path(out_t[::25], out_V[::25], X, Y))
print("  ejes x:", [(v, fmt(X(v))) for v in range(0, 61, 10)])
print("  ejes y:", [(v, fmt(Y(v))) for v in (-Vlim, -Vlim / 2, 0, Vlim / 2, Vlim)])
