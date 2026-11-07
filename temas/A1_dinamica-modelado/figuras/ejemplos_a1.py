"""Ejemplos numéricos de las hojas de A1 (dinámica y modelado).

Uso:  python3 ejemplos_a1.py > salida_ejemplos.txt
Requiere numpy; corre en CPU en unos segundos. Cada bloque imprime «EJEMPLO n» y es el
que citan las hojas («ejemplos_a1.py, ejemplo n»). También imprime los datos de las
figuras nuevas (conjunto factible, curva de sustentación, plano complejo de los modos
y rayo de parámetros indistinguibles), ya escalados a su viewBox.

Solo usa parámetros que ya están en el tema (x500 de PX4 en Gazebo, dron de NeuroBEM,
arrastre de Faessler et al., Aerosonde de Beard y McLain). El modelo del Aerosonde, su
equilibrio y su linealización se toman de figuras_a1.py, que se importa en silencio.
"""
import contextlib
import io
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import figuras_a1 as F  # noqa: E402  (modelo del Aerosonde, trim y jacobianos)

G = 9.81
RHO = 1.225


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def cab(n, titulo):
    print("=" * 72)
    print(f"EJEMPLO {n}. {titulo}")


# Parámetros del x500 (tabla de la hoja «Un cuadricóptero son 17 estados»)
m500 = 2.0 + 4 * 0.016076923
kT = 8.54858e-6
Jx5, Jy5, Jz5 = 0.0239, 0.0239, 0.044
arm = 0.174
Tmax = kT * 1000**2
Th = m500 * G / 4
Om_h = math.sqrt(Th / kT)

# ---------------------------------------------------------------------------
cab(1, "Término giroscópico en el x500 (voltereta en alabeo con algo de guiñada)")
w = np.array([10.0, 0.0, 1.0])
J = np.diag([Jx5, Jy5, Jz5])
gyro = np.cross(w, J @ w)
print(f"  omega = {w} rad/s; J = diag({Jx5}, {Jy5}, {Jz5}) kg m2")
print(f"  J omega = {J @ w}")
print(f"  omega x J omega = {np.round(gyro, 4)} N m")
print(f"  sin par: J domega = -omega x J omega -> domega = {np.round(-gyro / np.diag(J), 3)} rad/s2")
tau_y_max = arm * 2 * Tmax
print(f"  par de cabeceo máximo (dos motores a tope, dos parados) = {tau_y_max:.3f} N m;"
      f" el giroscópico es el {abs(gyro[1]) / tau_y_max * 100:.1f} %")

# ---------------------------------------------------------------------------
cab(2, "Eje intermedio con la inercia de NeuroBEM")
Jn = np.array([0.0025, 0.0021, 0.0043])
w0 = 20.0
coef = (Jn[2] - Jn[0]) * (Jn[0] - Jn[1]) / (Jn[1] * Jn[2])
lam = w0 * math.sqrt(coef)
print(f"  (Jz-Jx)(Jx-Jy)/(Jy Jz) = ({Jn[2]-Jn[0]:.4f})({Jn[0]-Jn[1]:.4f})/({Jn[1]:.4f}*{Jn[2]:.4f}) = {coef:.4f}")
print(f"  lambda = omega0 sqrt(coef) = {w0} * {math.sqrt(coef):.4f} = {lam:.3f} 1/s;"
      f" se multiplica por e cada {1000/lam:.0f} ms; por 10 cada {1000*math.log(10)/lam:.0f} ms")
# giro alrededor del eje mayor (z): oscila
cz = (Jn[1] - Jn[2]) * (Jn[2] - Jn[0]) / (Jn[0] * Jn[1])
print(f"  giro en z (eje mayor): coeficiente = {cz:.4f} < 0 -> oscila a {w0*math.sqrt(-cz):.2f} rad/s")
cy = (Jn[2] - Jn[1]) * (Jn[1] - Jn[0]) / (Jn[0] * Jn[2])
print(f"  giro en y (eje menor): coeficiente = {(Jn[0]-Jn[1])*(Jn[1]-Jn[2])/(Jn[0]*Jn[2]):.4f} < 0 -> oscila")
print(f"  x500 (Jx = Jy): coeficiente en x = {(Jz5-Jx5)*(Jx5-Jy5)/(Jy5*Jz5):.4f}: ni crece ni oscila a primer orden")

# ---------------------------------------------------------------------------
cab(3, "Disco actuador en estacionario: dron de NeuroBEM (752 g, hélices de 5 pulgadas)")
m_nb = 0.752
R_nb = 5 * 0.0254 / 2
T1 = m_nb * G / 4
A = math.pi * R_nb**2
vh = math.sqrt(T1 / (2 * RHO * A))
mdot = RHO * A * vh
wfar = 2 * vh
print(f"  T por rotor = m g / 4 = {T1:.3f} N; R = {R_nb*100:.2f} cm; A = pi R^2 = {A:.5f} m2")
print(f"  v_h = sqrt(T / (2 rho A)) = sqrt({T1:.3f} / (2*{RHO}*{A:.5f})) = {vh:.2f} m/s")
print(f"  flujo másico = rho A v_h = {mdot:.4f} kg/s; estela w = 2 v_h = {wfar:.2f} m/s")
print(f"  comprobación T = mdot w = {mdot*wfar:.3f} N; potencia = T v_h = {T1*vh:.2f} W por rotor;"
      f" 1/2 mdot w^2 = {0.5*mdot*wfar**2:.2f} W")
print(f"  cuatro rotores: P_ideal = {4*T1*vh:.1f} W")

# ---------------------------------------------------------------------------
cab(4, "Subida y descenso axial con el mismo rotor")
for Vc in (2.0, 5.0):
    r = Vc / (2 * vh)
    vi = vh * (-r + math.sqrt(r * r + 1))
    print(f"  V_c = {Vc} m/s: V_c/(2 v_h) = {r:.3f}; v_i/v_h = {vi/vh:.3f}; v_i = {vi:.2f} m/s;"
          f" P = T (V_c + v_i) = {T1*(Vc+vi):.1f} W por rotor (hover {T1*vh:.1f} W)")
Vc = -5.0
r = Vc / (2 * vh)
vi = vh * (-r + math.sqrt(r * r + 1))
print(f"  V_c = {Vc} m/s (descenso): la fórmula daría v_i = {vi:.2f} m/s; aire lejano arriba {Vc:+.1f} m/s"
      f" (sube), estela {Vc+2*vi:+.1f} m/s (baja): no hay un tubo de corriente único")
print(f"  banda sin solución: -2 v_h = {-2*vh:.1f} m/s < V_c < 0")

# ---------------------------------------------------------------------------
cab(5, "T = k_T Omega^2 en el x500")
for Om in (500.0, Om_h, 1000.0):
    print(f"  Omega = {Om:6.0f} rad/s ({Om/1000*100:4.0f} % de Omega_max): T = {kT*Om**2:.2f} N"
          f" ({kT*Om**2/Tmax*100:4.1f} % del máximo)")
print(f"  (Omega_max / Omega_h)^2 = {(1000/Om_h)**2:.3f} = empuje/peso")

# ---------------------------------------------------------------------------
cab(6, "Arrastre de rotor: de un rotor a la matriz D (x500 de Gazebo y Faessler et al.)")
lam1 = 8.06428e-5
d500 = 4 * lam1 * Om_h / m500
print(f"  un rotor: f = -|Omega| lambda1 v_perp; cuatro a Omega_h = {Om_h:.0f} rad/s:"
      f" d = 4 lambda1 Omega_h / m = 4*{lam1:.3e}*{Om_h:.0f}/{m500:.3f} = {d500:.3f} 1/s")
for v in (5.0, 10.0):
    print(f"  v_B = ({v}, 0, 0) m/s: x500 a_x = -d v = {-d500*v:.2f} m/s2 (fuerza {m500*d500*v:.2f} N);"
          f" dron de Faessler (d_x = 0,544) a_x = {-0.544*v:.2f} m/s2")

# ---------------------------------------------------------------------------
cab(7, "Efecto suelo de Cheeseman y Bennett a potencia constante")
for zr in (0.5, 1.0, 2.0):
    dv = 1 / (16 * zr**2)
    print(f"  z/R = {zr}: dv_i/v_i = (R/4z)^2 = {dv:.4f}; T_IGE/T_OGE = 1/(1 - {dv:.4f}) = {1/(1-dv):.3f}")

# ---------------------------------------------------------------------------
cab(8, "Motor de primer orden: escalón y respuesta en frecuencia (x500, tau = 12,5 ms al subir)")
tau = 0.0125
for k in (1, 2.3, 3):
    print(f"  t = {k} tau = {k*tau*1000:.1f} ms: fracción del escalón = {1-math.exp(-k):.3f}")
print(f"  escalón de Omega_h = {Om_h:.0f} a 1000 rad/s: a los 12,5 ms va por"
      f" {1000 - (1000-Om_h)*math.exp(-1):.0f} rad/s")
for f in (6.4, 10.0, 12.7):
    wt = 2 * math.pi * f * tau
    print(f"  f = {f} Hz: omega tau = {wt:.3f}; |G| = {1/math.sqrt(1+wt*wt):.3f};"
          f" fase = {-math.degrees(math.atan(wt)):.1f} deg")

# ---------------------------------------------------------------------------
cab(9, "Asignación del x500 (PX4, FRD): un mando de alabeo y la eficacia de guiñada")
Pr = [(0.174, 0.174, 0.05), (-0.174, -0.174, 0.05), (0.174, -0.174, -0.05), (-0.174, 0.174, -0.05)]
M = np.array([[1, 1, 1, 1], [-py for px, py, km in Pr], [px for px, py, km in Pr],
              [km for px, py, km in Pr]], dtype=float)
Mi = np.linalg.inv(M)
wr = np.array([4 * Th, 0.5, 0.0, 0.0])
Ti = Mi @ wr
print(f"  llave pedida (T, tau_x, tau_y, tau_z) = {np.round(wr, 3)}")
print(f"  T_i = M^-1 llave = {np.round(Ti, 3)} N; Omega_i = {np.round(np.sqrt(Ti/kT), 0)} rad/s")
print(f"  comprobación M T_i = {np.round(M @ Ti, 4)}")
wz = np.array([4 * Th, 0.0, 0.0, 0.1])
Tz = Mi @ wz
print(f"  guiñada tau_z = 0,1 N m pedida con KM = 0,05: T_i = {np.round(Tz, 3)}")
s = np.array([1, 1, -1, -1])
print(f"  la planta (momentConstant 0,016) da tau_z = 0,016 * sum(s_i T_i) = {0.016*np.sum(s*Tz):.4f} N m"
      f" ({0.1/(0.016*np.sum(s*Tz)):.2f} veces menos)")

# ---------------------------------------------------------------------------
cab(10, "Conjunto factible en el plano (T, tau_x) del x500, con tau_y = tau_z = 0")
k_r = Mi[0, 1]  # -1,4368
print(f"  T_i = T/4 +- {abs(k_r):.4f} tau_x; 0 <= T_i <= {Tmax:.2f} N")
verts = [(0, 0), (2 * Tmax, Tmax / 2 / abs(k_r) * 1.0), (4 * Tmax, 0), (2 * Tmax, -Tmax / 2 / abs(k_r))]
# vértices exactos: T/4 + a tau = Tmax y T/4 - a tau = 0 -> T = 2 Tmax, tau = Tmax / (2a)
a = abs(k_r)
verts = [(0.0, 0.0), (2 * Tmax, Tmax / (2 * a)), (4 * Tmax, 0.0), (2 * Tmax, -Tmax / (2 * a))]
print("  vértices (T, tau_x):", [(round(float(t), 2), round(float(u), 3)) for t, u in verts])
marg = min(Tmax - Th, Th) / a
print(f"  en estacionario T = {4*Th:.2f} N: |tau_x| <= {marg:.3f} N m (lo limita el motor que sube)")
# figura: viewBox 0 0 520 330; T de 0 a 36 en x 60..480; tau de -3,2 a 3,2 en y 290..30
X = lin(0, 36, 60, 480)
Y = lin(-3.2, 3.2, 290, 30)
print("  polígono:", " ".join(f"{X(t):.1f},{Y(u):.1f}" for t, u in verts))
print(f"  hover: ({X(4*Th):.1f}, {Y(0):.1f}); margen arriba y = {Y(marg):.1f}, abajo y = {Y(-marg):.1f}")
print("  ejes x T:", [(v, round(X(v), 1)) for v in (0, 10, 20, 30)])
print("  ejes y tau:", [(v, round(Y(v), 1)) for v in (-3, -2, -1, 0, 1, 2, 3)])

# ---------------------------------------------------------------------------
cab(11, "Planitud, empuje y actitud: x500 acelerando 3 m/s2 hacia delante con psi = 0 (ENU)")
acc = np.array([3.0, 0.0, 0.0])
t = acc + np.array([0, 0, G])
nt = np.linalg.norm(t)
zB = t / nt
xC = np.array([1.0, 0.0, 0.0])
yB = np.cross(zB, xC)
yB /= np.linalg.norm(yB)
xB = np.cross(yB, zB)
print(f"  t = a + g e3 = {t}; |t| = {nt:.3f} m/s2; T = m |t| = {m500*nt:.2f} N (hover {m500*G:.2f} N)")
print(f"  z_B = {np.round(zB, 3)}; inclinación = {math.degrees(math.acos(zB[2])):.1f} deg")
print(f"  y_B = {np.round(yB, 3)}; x_B = {np.round(xB, 3)} (el morro baja)")
print(f"  por motor: {m500*nt/4:.2f} N, Omega = {math.sqrt(m500*nt/4/kT):.0f} rad/s")

# ---------------------------------------------------------------------------
cab(12, "Planitud, velocidades angulares y pares: x500 en estacionario")
j = np.array([2.0, 0.0, 0.0])
zB = np.array([0, 0, 1.0])
xB = np.array([1.0, 0, 0])
yB = np.array([0, 1.0, 0])
T = m500 * G
h = m500 / T * (j - (zB @ j) * zB)
print(f"  jerk j = {j} m/s3: T dot = m z_B.j = {m500*zB@j:.2f} N/s; h = (m/T)(j - (z.j)z) = {np.round(h, 4)}")
print(f"  omega_x = -h.y_B = {-h@yB:.4f}; omega_y = h.x_B = {h@xB:.4f} rad/s ({math.degrees(h@xB):.1f} deg/s)")
snap = np.array([10.0, 0, 0])
wd = snap[0] / G
tauy = Jy5 * wd
print(f"  snap = {snap} m/s4 con jerk nulo en ese instante: domega_y = s_x/g = {wd:.3f} rad/s2")
print(f"  tau_y = J_yy domega_y = {tauy:.4f} N m; diferencia por motor = {abs(Mi[0,2])*tauy:.4f} N")

# ---------------------------------------------------------------------------
cab(13, "Planitud con arrastre isótropo d = 0,465 1/s en vuelo recto y nivelado")
d = 0.465
for v in (4.0, 10.0):
    tx = d * v
    print(f"  v = {v} m/s: t = (d v, 0, g) = ({tx:.2f}, 0, {G}); inclinación = {math.degrees(math.atan2(tx, G)):.1f} deg"
          f" (sin modelo de arrastre: 0 deg); |t| = {math.hypot(tx, G):.2f} m/s2")

# ---------------------------------------------------------------------------
P = F.P
al0, de0, dt0, da0, dr0, ph0 = F.z
Va0 = F.Va0
cab(14, "Ala fija: velocidad relativa y ángulos del Aerosonde en el equilibrio a 25 m/s")
u0, w0_ = Va0 * math.cos(al0), Va0 * math.sin(al0)
print(f"  u = {u0:.3f} m/s, w = {w0_:.3f} m/s, v = 0: V_a = {math.hypot(u0, w0_):.2f} m/s;"
      f" alpha = atan2(w, u) = {math.degrees(math.atan2(w0_, u0)):.2f} deg; beta = 0")
for vw in (5.0, 12.5):
    print(f"  viento de cara de {vw} m/s: V_g = V_a - {vw} = {Va0-vw:.1f} m/s;"
          f" alcance por unidad de tiempo x{(Va0-vw)/Va0:.2f}")

# ---------------------------------------------------------------------------
cab(15, "Ala fija: fuerzas y momento en el equilibrio, término a término")
qbar = 0.5 * P["rho"] * Va0**2
qS = qbar * P["S"]
CL = P["CL0"] + P["CLa"] * al0 + P["CLde"] * de0
CDi = (P["CL0"] + P["CLa"] * al0) ** 2 / (math.pi * P["e"] * P["AR"])
CD = P["CDp"] + CDi + P["CDde"] * de0
Cm = P["Cm0"] + P["Cma"] * al0 + P["Cmde"] * de0
print(f"  presión dinámica 1/2 rho V^2 = 0,5*{P['rho']}*{Va0}^2 = {qbar:.1f} Pa; por S = {P['S']}: {qS:.1f} N")
print(f"  C_L = C_L0 + C_La alpha + C_Lde de = {P['CL0']} + {P['CLa']}*{al0:.4f} + {P['CLde']}*({de0:.4f})"
      f" = {CL:.4f} -> L = {qS*CL:.1f} N; peso = {P['mass']*G:.1f} N")
print(f"  AR = b^2/S = {P['AR']:.2f}; C_D = {P['CDp']} + {CDi:.4f} (inducida) {P['CDde']*de0:+.4f} (elevador) = {CD:.4f}"
      f" -> D = {qS*CD:.2f} N")
print(f"  C_m = {P['Cm0']} + ({P['Cma']})*{al0:.4f} + ({P['Cmde']})*({de0:.4f}) = {Cm:.5f} (equilibrio en cabeceo)")
Tp0, _ = F.prop(Va0, dt0)
fx = -math.cos(al0) * qS * CD + math.sin(al0) * qS * CL + Tp0 - P["mass"] * G * math.sin(al0)
fz = -math.sin(al0) * qS * CD - math.cos(al0) * qS * CL + P["mass"] * G * math.cos(al0)
print(f"  empuje de la hélice {Tp0:.2f} N; f_x = {fx:.3f} N, f_z = {fz:.3f} N (cero con theta = alpha)")

# Curva C_L(alpha) con el modelo de pérdida, para la figura
cab(16, "Curva de sustentación con el modelo de mezcla del Aerosonde (figura)")
als = np.radians(np.linspace(-10, 60, 141))


def sig(a):
    return (1 + np.exp(-P["M"] * (a - P["a0"])) + np.exp(P["M"] * (a + P["a0"]))) / (
        (1 + np.exp(-P["M"] * (a - P["a0"]))) * (1 + np.exp(P["M"] * (a + P["a0"]))))


lin_cl = P["CL0"] + P["CLa"] * als
plate = 2 * np.sign(als) * np.sin(als) ** 2 * np.cos(als)
s_ = sig(als)
blend = (1 - s_) * lin_cl + s_ * plate
for ad in (0, 10, 20, 27, 35, 45):
    a = math.radians(ad)
    print(f"  alpha = {ad:2d} deg: lineal {P['CL0']+P['CLa']*a:.3f}, placa {2*math.copysign(1,a)*math.sin(a)**2*math.cos(a):.3f},"
          f" sigma {float(sig(np.array(a))):.3f}, mezcla {float((1-sig(np.array(a)))*(P['CL0']+P['CLa']*a)+sig(np.array(a))*2*math.copysign(1,a)*math.sin(a)**2*math.cos(a)):.3f}")
ib = int(np.argmax(blend))
print(f"  máximo de la mezcla: C_L = {blend[ib]:.3f} a {math.degrees(als[ib]):.1f} deg")
# viewBox 0 0 520 300; alpha -10..60 en x 60..500; C_L -0,8..3,2 en y 260..30
X = lin(-10, 60, 60, 500)
Y = lin(-0.8, 3.2, 260, 30)
ad = np.degrees(als)
ok = lin_cl <= 3.2
print("  path lineal:", path(ad[ok], lin_cl[ok], X, Y))
print("  path placa:", path(ad, plate, X, Y))
print("  path mezcla:", path(ad, blend, X, Y))
print("  ejes x:", [(v, round(X(v), 1)) for v in (-10, 0, 10, 20, 30, 40, 50, 60)])
print("  ejes y:", [(v, round(Y(v), 1)) for v in (-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3)])
print(f"  alpha0 = 27 deg -> x = {X(27):.1f}; trim alpha = 2,85 deg -> ({X(2.85):.1f}, {Y(P['CL0']+P['CLa']*al0):.1f})")

# ---------------------------------------------------------------------------
cab(17, "Linealización: cuánto acopla A los subsistemas longitudinal y lateral")
Ax = F.Ax
lon, lat = F.lon, F.lat
names = ["u", "v", "w", "phi", "theta", "psi", "p", "q", "r"]
Alon = Ax[np.ix_(lon, lon)]
Alat = Ax[np.ix_(lat, lat)]
Ac1 = Ax[np.ix_(lon, lat)]
Ac2 = Ax[np.ix_(lat, lon)]
print("  A longitudinal (u, w, q, theta):\n", np.round(Alon, 3))
print("  A lateral (v, p, r, phi):\n", np.round(Alat, 3))
print(f"  máx |A_lon|, |A_lat| = {np.abs(Alon).max():.2f}, {np.abs(Alat).max():.2f};"
      f" máx |bloques cruzados| = {np.abs(Ac1).max():.4f}, {np.abs(Ac2).max():.4f}")
i, k = np.unravel_index(np.argmax(np.abs(Ac1)), Ac1.shape)
print(f"  mayor cruce lon<-lat: d({names[lon[i]]})/d({names[lat[k]]}) = {Ac1[i,k]:.4f}")
i, k = np.unravel_index(np.argmax(np.abs(Ac2)), Ac2.shape)
print(f"  mayor cruce lat<-lon: d({names[lat[i]]})/d({names[lon[k]]}) = {Ac2[i,k]:.4f}")
evs8 = np.linalg.eigvals(Ax[np.ix_(lon + lat, lon + lat)])
evb = np.concatenate([np.linalg.eigvals(Alon), np.linalg.eigvals(Alat)])
print("  autovalores de A (8 estados) frente a los de los dos bloques:")
print("   ", np.round(np.sort_complex(evs8), 4))
print("   ", np.round(np.sort_complex(evb), 4))

# ---------------------------------------------------------------------------
cab(18, "De autovalores a modos y qué estados se mueven en cada uno")


def modo(e):
    wn = abs(e)
    if abs(e.imag) > 1e-9:
        return (f"wn = |lambda| = sqrt({e.real:.3f}^2 + {e.imag:.3f}^2) = {wn:.3f} rad/s; "
                f"zeta = -Re/|lambda| = {-e.real/wn:.3f}; periodo = 2 pi/Im = {2*math.pi/abs(e.imag):.2f} s")
    if e.real < 0:
        return f"tau = -1/lambda = {-1/e.real*1000:.0f} ms"
    return f"inestable; dobla en ln2/lambda = {math.log(2)/e.real:.2f} s"


for nm, idx, A_ in (("longitudinal", lon, Alon), ("lateral", lat, Alat)):
    ev, V = np.linalg.eig(A_)
    lab = [names[i] for i in idx]
    for kk in np.argsort(-np.abs(ev)):
        e = ev[kk]
        if e.imag < -1e-9:
            continue
        vec = V[:, kk].copy()
        # escala: velocidades lineales / Va (ángulos), velocidades angulares / |lambda| (ángulos)
        sc = np.array([1 / Va0 if l in ("u", "v", "w") else (1 / abs(e) if l in ("p", "q", "r") else 1.0) for l in lab])
        mag = np.abs(vec * sc)
        mag /= mag.max()
        comp = ", ".join(f"{l} {m:.2f}" for l, m in zip(lab, mag))
        print(f"  {nm}: lambda = {e.real:.4f} {e.imag:+.4f}j -> {modo(e)}")
        print(f"      peso relativo en el autovector (u,v,w/Va; p,q,r/|lambda|): {comp}")
# plano complejo para la figura: viewBox 0 0 560 300
X = lin(-24, 2, 50, 530)
Y = lin(-11, 11, 270, 30)
allev = np.linalg.eigvals(Ax[np.ix_(lon + lat, lon + lat)])
print("  puntos (Re, Im) -> (x, y):")
for e in sorted(allev, key=lambda c: (c.real, c.imag)):
    print(f"    ({e.real:8.4f}, {e.imag:8.4f}) -> ({X(e.real):.1f}, {Y(e.imag):.1f})")
print("  ejes x:", [(v, round(X(v), 1)) for v in (-24, -20, -15, -10, -5, 0)])
print("  ejes y:", [(v, round(Y(v), 1)) for v in (-10, -5, 0, 5, 10)])
# recuadro ampliado cerca del origen (abajo a la izquierda): Re -0,3..0,2 , Im -0,6..0,6
Xz = lin(-0.3, 0.2, 110, 290)
Yz = lin(-0.6, 0.6, 262, 182)
for e in allev:
    if abs(e) < 0.6:
        print(f"    zoom ({e.real:.4f}, {e.imag:.4f}) -> ({Xz(e.real):.1f}, {Yz(e.imag):.1f})")
print("  zoom ejes:", [(v, round(Xz(v), 1)) for v in (-0.3, -0.2, -0.1, 0, 0.1, 0.2)],
      [(v, round(Yz(v), 1)) for v in (-0.5, 0, 0.5)])

# ---------------------------------------------------------------------------
cab(19, "Fugoide de Lanchester a 25 m/s")
wph = math.sqrt(2) * G / Va0
print(f"  omega_ph = sqrt(2) g / V0 = 1,414*9,81/25 = {wph:.4f} rad/s; T_ph = 2 pi/omega = {2*math.pi/wph:.2f} s")
print(f"  coeficiente 2 g^2 / V0^2 = {2*G**2/Va0**2:.4f} 1/s2")
dh = 10.0
print(f"  si sube {dh} m desde V0 = 25 m/s: V = sqrt(V0^2 - 2 g dh) = {math.sqrt(Va0**2-2*G*dh):.2f} m/s;"
      f" L/W = V^2/V0^2 = {(Va0**2-2*G*dh)/Va0**2:.3f}")

# ---------------------------------------------------------------------------
cab(20, "Leyes de escala: exponentes y comprobación con x500 frente a Crazyflie 2.1")
rows = {"Mach (nD cte)": -1.0, "Froude (n^2 D cte)": -0.5}
for nm, en in rows.items():
    eT = 2 * en + 4
    etau = eT + 1
    ea = etau - 5
    print(f"  {nm}: n ~ L^{en}; T ~ L^{eT}; T/W ~ L^{eT-3}; tau ~ L^{etau}; alpha ~ L^{ea}; t ~ L^{-ea/2}")
Lr = (m500 / 0.027) ** (1 / 3)
print(f"  L_x500 / L_CF = (m_x500/m_CF)^(1/3) = ({m500:.3f}/0,027)^(1/3) = {Lr:.2f}")
print(f"  Froude: alpha x{Lr:.2f}; Mach: alpha x{Lr**2:.1f}; datos 526/125 = {526/125:.2f}")
print(f"  mitad de tamaño con Froude: alpha x2, tiempo de giro x{math.sqrt(0.5):.3f}")

# ---------------------------------------------------------------------------
cab(21, "Identificabilidad: el x500 y su gemelo escalado por s = 2")
for s_ in (0.5, 1.0, 2.0):
    m_ = s_ * m500
    k_ = s_ * kT
    J_ = s_ * Jx5
    Omh = math.sqrt(m_ * G / (4 * k_))
    amax = 2 * k_ * 1000**2 * arm / J_
    print(f"  s = {s_}: m = {m_:.3f} kg, k_T = {k_:.3e}, J_xx = {J_:.4f}: k_T/m = {k_/m_:.4e};"
          f" Omega_h = {Omh:.0f} rad/s; alpha_max = {amax:.1f} rad/s2")
# figura: m de 0 a 5 kg en x 60..460, k_T de 0 a 2,2e-5 en y 260..30
X = lin(0, 5, 60, 460)
Y = lin(0, 2.2e-5, 260, 30)
print("  rayo k_T/m cte: de (0,0) a", (round(X(5), 1), round(Y(5 * kT / m500), 1)))
for s_ in (0.5, 1.0, 2.0):
    print(f"    punto s = {s_}: ({X(s_*m500):.1f}, {Y(s_*kT):.1f})")
print("  ejes x:", [(v, round(X(v), 1)) for v in range(0, 6)])
print("  ejes y:", [(v, round(Y(v * 1e-5), 1)) for v in (0, 0.5, 1, 1.5, 2)])
