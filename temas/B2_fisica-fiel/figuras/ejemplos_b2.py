"""Ejemplos numéricos de las hojas de B2 (física fiel) y datos de las figuras nuevas.

Uso:  python3 ejemplos_b2.py > salida_ejemplos.txt      (numpy, CPU, unos segundos)

Cada ejemplo lleva un número («Ejemplo N») que es el que citan las hojas. Usa los mismos parámetros
que figuras_b2.py (x500 de Gazebo y los valores SUPUESTOS de la pala, el motor y la batería), copiados
aquí para que el script se lea solo. Las figuras con datos que ya estaban en el tema salen de figuras_b2.py.
"""
import math
import numpy as np

G, RHO = 9.81, 1.225
M = 2.0 + 4 * 0.016076923076923075          # masa del x500 de Gazebo con sus 4 rotores
KT, KM = 8.54858e-06, 0.016
KQ = KM * KT
OM_H = math.sqrt(M * G / (4 * KT))
T_H = M * G / 4


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def cab(n, titulo):
    print("=" * 78)
    print(f"Ejemplo {n}. {titulo}")


# ----------------------------------------------------------------------------- PARTE 1 · ROTOR
R_P, NB, CH = 0.127, 2, 0.020               # SUPUESTOS (figuras_b2.py)
SIG = NB * CH / (math.pi * R_P)
A_L, TH_TW, CD0, KAPPA = 5.7, -0.20, 0.020, 1.15
AREA = math.pi * R_P ** 2
VT_H = OM_H * R_P
CT_H = T_H / (RHO * AREA * VT_H ** 2)
LAM_H = math.sqrt(CT_H / 2)
TH0 = 3 * (CT_H / (SIG * A_L / 2) + LAM_H / 2 - TH_TW / 4)

cab(1, "La pala supuesta y el estacionario del x500")
print(f"  R = {R_P} m, A = pi R^2 = {AREA:.5f} m^2, N = {NB}, c = {CH} m -> sigma = N c/(pi R) = {SIG:.4f}")
print(f"  T_h = m g/4 = {T_H:.3f} N; Omega_h = sqrt(m g/(4 k_T)) = {OM_H:.1f} rad/s; Omega_h R = {VT_H:.2f} m/s")
print(f"  C_T,h = T/(rho A (Omega R)^2) = {T_H:.3f}/({RHO}*{AREA:.5f}*{VT_H:.2f}^2) = {CT_H:.5f}")
print(f"  theta_tw = {TH_TW} rad = {math.degrees(TH_TW):.1f}°; theta_0 = {TH0:.4f} rad = {math.degrees(TH0):.1f}°")


def ct_lin(lam, mu=0.0, th0=TH0):
    return SIG * A_L / 2 * (th0 / 3 * (1 + 1.5 * mu ** 2) + TH_TW / 4 * (1 + mu ** 2) - lam / 2)


cab(2, "Un anillo de pala en estacionario (x = r/R = 0,75)")
x = 0.75
r = x * R_P
UT = OM_H * r
UP = LAM_H * VT_H
phi = UP / UT
th = TH0 + TH_TW * x
alf = th - phi
dLdr = 0.5 * RHO * UT ** 2 * CH * A_L * alf
print(f"  r = {r:.4f} m; U_T = Omega r = {UT:.1f} m/s; U_P = lambda_h Omega R = v_h = {UP:.2f} m/s")
print(f"  phi = U_P/U_T = {phi:.4f} rad = {math.degrees(phi):.1f}°; theta(0,75) = theta_0 + 0,75 theta_tw = {math.degrees(th):.1f}°")
print(f"  ángulo de ataque theta - phi = {math.degrees(alf):.1f}°; C_l = a (theta - phi) = {A_L*alf:.3f}")
print(f"  dL/dr = 1/2 rho U_T^2 c a (theta - phi) = {dLdr:.1f} N/m por pala; x N palas = {NB*dLdr:.1f} N/m")
# integral del empuje por anillos (comprobación numérica de la fórmula cerrada)
xs = np.linspace(1e-6, 1, 20001)
dT = NB * 0.5 * RHO * (OM_H * xs * R_P) ** 2 * CH * A_L * (TH0 + TH_TW * xs - LAM_H / xs) * R_P
Tnum = np.trapezoid(dT, xs) if hasattr(np, "trapezoid") else np.trapz(dT, xs)
print(f"  suma de todos los anillos (integral numérica): T = {Tnum:.3f} N (debe dar T_h = {T_H:.3f} N)")

cab(3, "Los tres términos de C_T en estacionario")
b1, b2, b3 = TH0 / 3, TH_TW / 4, LAM_H / 2
print(f"  sigma a/2 = {SIG*A_L/2:.4f}")
print(f"  theta_0/3 = {b1:.4f};  theta_tw/4 = {b2:.4f};  lambda/2 = {b3:.4f};  corchete = {b1+b2-b3:.4f}")
print(f"  C_T = {SIG*A_L/2:.4f} x {b1+b2-b3:.4f} = {SIG*A_L/2*(b1+b2-b3):.5f}")
print(f"  cantidad de movimiento: lambda_h = sqrt(C_T/2) = {LAM_H:.4f} -> v_h = lambda_h Omega R = {LAM_H*VT_H:.2f} m/s; "
      f"A1: sqrt(T/(2 rho A)) = {math.sqrt(T_H/(2*RHO*AREA)):.2f} m/s")


def resuelve(mu, lam_c, n=200, amort=0.5, lam0=None, traza=False):
    lam = LAM_H + lam_c if lam0 is None else lam0
    filas = []
    for k in range(n):
        ct = ct_lin(lam, mu)
        lam_new = lam_c + ct / (2 * math.sqrt(mu ** 2 + lam ** 2))
        filas.append((k, lam, ct, lam_new))
        lam = (1 - amort) * lam + amort * lam_new
    return ct, lam, filas


cab(4, "Iteración de punto fijo en subida a 2 m/s (Omega fijo)")
lam_c = 2 / VT_H
print(f"  lambda_c = V_c/(Omega R) = 2/{VT_H:.2f} = {lam_c:.4f}")
ct, lam, filas = resuelve(0, lam_c, n=60)
print("  k   lambda      C_T        lambda_nuevo = lambda_c + C_T/(2 lambda)")
for k, l_, c_, ln_ in filas[:6]:
    print(f"  {k}   {l_:.5f}    {c_:.6f}   {ln_:.5f}")
kconv = next(k for k, l_, c_, ln_ in filas if abs(l_ - lam) < 1e-6)
print(f"  error de lambda por debajo de 1e-6 desde la vuelta {kconv}")
print(f"  ...convergido: lambda = {lam:.5f}, C_T = {ct:.6f}, T/T_h = {ct/CT_H:.3f}; lambda_i = {lam-lam_c:.5f} ({(lam-lam_c)*VT_H:.2f} m/s)")
_, _, f_sin = resuelve(0, lam_c, n=12, amort=1.0)
print("  sin amortiguar (lambda <- lambda_nuevo):", ", ".join(f"{f[1]:.4f}" for f in f_sin[:8]))
# forma cerrada en subida pura: C_T = K - (sigma a/4) lambda = 2 lambda (lambda - lambda_c)
K = SIG * A_L / 2 * (TH0 / 3 + TH_TW / 4)
aa, bb, cc = 2.0, SIG * A_L / 4 - 2 * lam_c, -K
lam_ex = (-bb + math.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)
print(f"  forma cerrada: 2 lambda^2 + (sigma a/4 - 2 lambda_c) lambda - K = 0, K = {K:.5f} -> lambda = {lam_ex:.5f}; "
      f"C_T = {K - SIG*A_L/4*lam_ex:.6f}")

cab(5, "Avance con el disco horizontal: de dónde sale la sustentación de traslación")
for v in (4.0, 10.0):
    mu = v / VT_H
    ct_full, lam_f, _ = resuelve(mu, 0)
    ct_lamfijo = ct_lin(LAM_H, mu)
    print(f"  v = {v:4.1f} m/s: mu = {mu:.4f}, mu^2 = {mu*mu:.5f}; factores 1 + 1,5 mu^2 = {1+1.5*mu*mu:.4f}, 1 + mu^2 = {1+mu*mu:.4f}")
    print(f"     solo los factores (lambda fija en lambda_h): T/T_h = {ct_lamfijo/CT_H:.4f}")
    print(f"     con lambda resuelta: lambda = {lam_f:.4f} (lambda_h = {LAM_H:.4f}); T/T_h = {ct_full/CT_H:.3f}; sqrt(mu^2+lambda^2) = {math.sqrt(mu*mu+lam_f*lam_f):.4f}")
# promedio en una vuelta de (x + mu sin psi)^2
psi = np.linspace(0, 2 * math.pi, 100001)[:-1]
mu = 0.3
print(f"  comprobación del promedio: <(x + mu sin psi)^2> con x = 0,5, mu = 0,3: {np.mean((0.5+mu*np.sin(psi))**2):.5f}; x^2 + mu^2/2 = {0.25+0.045:.5f}")

cab(6, "Avance con el disco inclinado")
D_TRIM = 0.465
for v in (4.0, 10.0):
    a_ = math.atan(D_TRIM * v / G)
    mu, lc = v * math.cos(a_) / VT_H, v * math.sin(a_) / VT_H
    ct, lam, _ = resuelve(mu, lc)
    print(f"  v = {v:4.1f} m/s: tan alpha = d v/g = {D_TRIM*v/G:.4f} -> alpha = {math.degrees(a_):.1f}°; "
          f"V cos alpha = {v*math.cos(a_):.2f}, V sin alpha = {v*math.sin(a_):.2f} m/s")
    print(f"     mu = {mu:.4f}, lambda_c = {lc:.4f}; lambda = {lam:.4f}; T/T_h = {ct/CT_H:.3f}")

cab(7, "Potencia del rotor en estacionario")
ind = KAPPA * LAM_H * CT_H
prf = SIG * CD0 / 8
cp = ind + prf
P1 = cp * RHO * AREA * VT_H ** 3
print(f"  inducida kappa lambda_i C_T = {KAPPA} x {LAM_H:.4f} x {CT_H:.5f} = {ind:.3e}")
print(f"  perfil sigma C_d0/8 = {SIG:.4f} x {CD0}/8 = {prf:.3e}")
print(f"  C_P = {cp:.3e}; P = C_P rho A (Omega R)^3 = {P1:.1f} W por rotor, {4*P1:.0f} W los cuatro")
print(f"  ideal T v_h = {T_H*LAM_H*VT_H:.1f} W; FM = (C_T^1,5/sqrt2)/C_P = {CT_H**1.5/math.sqrt(2)/cp:.3f}")
print(f"  par en estacionario: Q = P/Omega = {P1/OM_H:.4f} N m;  k_Q Omega^2 de Gazebo = {KQ*OM_H**2:.4f} N m")

# ----------------------------------------------------------------------------- PARTE 2 · MOTOR Y BATERÍA
R_M, J_R = 0.12, 2.5e-5
V_FULL = 16.8
qa = KQ * 1000 ** 2 * R_M
KE = (V_FULL + math.sqrt(V_FULL ** 2 - 4 * 1000 * qa)) / (2 * 1000)
KTM = KE

cab(8, "Constante del motor equivalente a partir de la velocidad máxima")
print(f"  condición: con d = 1 y 16,8 V, k_Q Omega^2 = k_t (V - k_e Omega)/R en Omega = 1000 rad/s")
print(f"  con k_t = k_e: 1000 k_e^2 - 16,8 k_e + k_Q R 1000^2 = 0, k_Q R 1000^2 = {qa:.5f}")
print(f"  discriminante 16,8^2 - 4 x 1000 x {qa:.5f} = {V_FULL**2 - 4*1000*qa:.2f}; raíz grande k_e = {KE:.5f} V s/rad")
print(f"  la otra raíz, {(V_FULL - math.sqrt(V_FULL**2-4*1000*qa))/2000:.5f}, daría KV = {60/(2*math.pi*(V_FULL - math.sqrt(V_FULL**2-4*1000*qa))/2000):.0f} rpm/V y casi toda la tensión en R")
print(f"  KV = 60/(2 pi k_e) = {60/(2*math.pi*KE):.0f} rpm/V;  k_t = k_e = {KTM:.5f} N m/A")
i_h = KQ * OM_H ** 2 / KTM
print(f"  estacionario: par de la hélice k_Q Omega_h^2 = {KQ*OM_H**2:.4f} N m -> i = par/k_t = {i_h:.2f} A")
print(f"  fuerza contraelectromotriz k_e Omega_h = {KE*OM_H:.2f} V; caída R i = {R_M*i_h:.3f} V; d V = {KE*OM_H+R_M*i_h:.2f} V -> d = {(KE*OM_H+R_M*i_h)/V_FULL:.3f} con 16,8 V")
print(f"  potencias: eléctrica en la fcem e i = {KE*OM_H*i_h:.1f} W = mecánica k_t i Omega = {KTM*i_h*OM_H:.1f} W; pérdidas R i^2 = {R_M*i_h**2:.2f} W")

cab(9, "De dónde salen los 11 ms")
be, ba = KTM * KE / R_M, 2 * KQ * OM_H
tau = J_R / (be + ba)
print(f"  eléctrico k_t k_e/R = {KTM:.5f}^2/{R_M} = {be:.3e} N m s")
print(f"  aerodinámico 2 k_Q Omega_0 = 2 x {KQ:.4e} x {OM_H:.1f} = {ba:.3e} N m s   (cociente {be/ba:.1f})")
print(f"  tau = J_r/(suma) = {J_R:.1e}/{be+ba:.3e} = {tau*1e3:.2f} ms;  sin la hélice J_r R/k_e^2 = {J_R/be*1e3:.2f} ms")
print(f"  ganancia estática K = (k_t V/R)/(suma) = {KTM*V_FULL/R_M/(be+ba):.0f} rad/s por unidad de mando")
for om in (500, 1000):
    print(f"  a Omega_0 = {om}: tau = {J_R/(be+2*KQ*om)*1e3:.2f} ms")

cab(10, "Frenado activo: la corriente al bajar de 0,80 a 0,55")


def om_ss(duty, V):
    a, b, c = KQ * R_M, KTM * KE, -KTM * duty * V
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


w_hi, w_lo = om_ss(0.80, V_FULL), om_ss(0.55, V_FULL)
i0 = (0.55 * V_FULL - KE * w_hi) / R_M
print(f"  Omega antes {w_hi:.1f} rad/s, después {w_lo:.1f} rad/s")
print(f"  justo tras el cambio: d V = {0.55*V_FULL:.2f} V < k_e Omega = {KE*w_hi:.2f} V -> i = {i0:.1f} A (negativa)")
print(f"  par de frenado k_t i = {KTM*i0:.3f} N m frente al de la hélice k_Q Omega^2 = {KQ*w_hi**2:.4f} N m (x{abs(KTM*i0)/(KQ*w_hi**2):.1f})")
print(f"  sin frenado (i >= 0): J dOmega/dt = -k_Q Omega^2; J_r/k_Q = {J_R/KQ:.1f} rad")
t_libre = J_R / KQ * (1 / w_lo - 1 / w_hi)
print(f"  1/Omega_2 - 1/Omega_1 = 1/{w_lo:.1f} - 1/{w_hi:.1f} = {1/w_lo-1/w_hi:.4e} -> t = {t_libre*1e3:.1f} ms")
w63 = w_hi + 0.632 * (w_lo - w_hi)
print(f"  al 63 % ({w63:.1f} rad/s): t = {J_R/KQ*(1/w63-1/w_hi)*1e3:.1f} ms (la simulación con i >= 0 da 53,2 ms)")
w_on = 0.55 * V_FULL / KE
print(f"  cuándo vuelve a haber corriente sin frenado: d V = k_e Omega -> Omega = {w_on:.1f} rad/s, a los {J_R/KQ*(1/w_on-1/w_hi)*1e3:.1f} ms")
print(f"  la otra raíz de k_e (0,00104) pediría i = (16,8 - 0,00104 x 1000)/R = {(16.8-0.00104*1000)/R_M:.0f} A a 1000 rad/s")


def simula_i(freno, T=0.12, h=1e-5):
    w = w_hi
    out = []
    for k in range(int(T / h)):
        dd = 0.55 if k * h >= 0.02 else 0.80
        i = (dd * V_FULL - KE * w) / R_M
        if not freno:
            i = max(i, 0.0)
        w += h * (KTM * i - KQ * w * w) / J_R
        out.append((k * h - 0.02, i, w))
    return np.array(out)


Sf, Sl = simula_i(True)[::50], simula_i(False)[::50]
X10 = lin(-0.01, 0.10, 60, 520)
Y10 = lin(-32, 8, 230, 20)
sel = Sf[:, 0] >= -0.01
print("path corriente con freno:", path(Sf[sel, 0], Sf[sel, 1], X10, Y10))
print("path corriente sin freno:", path(Sl[sel, 0], Sl[sel, 1], X10, Y10))
print(f"  ejes corriente: y(0) = {Y10(0):.1f}, y(-10) = {Y10(-10):.1f}, y(-20) = {Y10(-20):.1f}, y(-30) = {Y10(-30):.1f}, y(5) = {Y10(5):.1f}; "
      f"x(0) = {X10(0):.1f}, x(0,02) = {X10(0.02):.1f}, x(0,05) = {X10(0.05):.1f}, x(0,1) = {X10(0.1):.1f}")
print(f"  corriente final {Sf[-1,1]:.2f} A (= i en estacionario con d = 0,55); mínimo con freno {Sf[:,1].min():.1f} A")

cab(11, "Del mando a la velocidad y al empuje (16,8 V)")
a2, b2_, = KQ * R_M, KTM * KE
print(f"  k_Q R Omega^2 + k_t k_e Omega - k_t d V = 0: coeficientes {a2:.3e}, {b2_:.3e}, -{KTM*V_FULL:.4f} d")
for dd in (0.25, 0.5, 0.759, 1.0):
    w = om_ss(dd, V_FULL)
    print(f"  d = {dd:5.3f}: Omega = {w:6.1f} rad/s (sin carga sería dV/k_e = {dd*V_FULL/KE:6.1f}); T = {KT*w*w:.2f} N; T/T_max = {KT*w*w/(KT*om_ss(1,V_FULL)**2):.3f}")
ds = np.linspace(0.05, 1, 96)
TT = np.array([om_ss(d_, V_FULL) ** 2 for d_ in ds]) / om_ss(1, V_FULL) ** 2
# ajuste de f en T/T_max = (1-f) u + f u^2 con u = d
A_ = ds ** 2 - ds
f_fit = float(A_ @ (TT - ds) / (A_ @ A_))
print(f"  ajuste por mínimos cuadrados de f en (1-f) d + f d^2: f = {f_fit:.2f}; error máximo {np.max(np.abs((1-f_fit)*ds+f_fit*ds**2-TT)):.3f}")
X11 = lin(0, 1, 60, 300)
Y11 = lin(0, 1, 250, 30)
print("path curva modelo:", path(ds, TT, X11, Y11))
print("path curva lineal:", path([0, 1], [0, 1], X11, Y11))
dq = np.linspace(0, 1, 41)
print("path curva cuadratica:", path(dq, dq ** 2, X11, Y11))
print("path curva px4 f=0,4:", path(dq, 0.6 * dq + 0.4 * dq ** 2, X11, Y11))
print(f"  ejes curva: x(0,5) = {X11(0.5):.0f}, x(1) = {X11(1):.0f}; y(0,5) = {Y11(0.5):.0f}, y(1) = {Y11(1):.0f}")

cab(12, "Batería: la caída en R0 y el punto fijo de la tensión de bus")
CAP, R0, V_EMPTY, V_CH = 5.0, 0.005, 3.6, 4.2


def ocv(soc):
    return V_EMPTY + soc * (V_CH - V_EMPTY)


def estacionario(Vb):
    i_m = KQ * OM_H ** 2 / KTM
    dd = (KE * OM_H + R_M * i_m) / Vb
    return dd, i_m, 4 * dd * i_m


for soc in (1.0, 0.0):
    Vb = 4 * ocv(soc)
    print(f"  SoC = {soc:.0f}: OCV por celda {ocv(soc):.2f} V, 4 x OCV = {Vb:.2f} V")
    for it in range(4):
        dd, im, ib = estacionario(Vb)
        Vn = 4 * (ocv(soc) - R0 * ib)
        print(f"     iteración {it}: V_bus = {Vb:.3f} V -> d = {dd:.4f}, I = 4 d i = {ib:.2f} A -> V_bus nuevo = 4({ocv(soc):.2f} - {R0} x {ib:.2f}) = {Vn:.3f} V")
        Vb = Vn
    wmax = om_ss(1.0, Vb)
    print(f"     margen: Omega_max(V_bus = {Vb:.2f}) = {wmax:.1f} rad/s; 4 k_T Omega_max^2 = {4*KT*wmax**2:.2f} N; / m g ({M*G:.2f} N) = {4*KT*wmax**2/(M*G):.3f}")
    print(f"     potencia de la batería V I = {Vb*ib:.0f} W = eje 4 k_Q Omega_h^2 Omega_h = {4*KQ*OM_H**3:.0f} W + bobinados 4 R i^2 = {4*R_M*im**2:.1f} W + R0 I^2 (por celda x4) = {4*R0*ib**2:.1f} W (esta última sale de la OCV, no del bus)")
    print(f"     potencia aerodinámica de la BEM (ejemplo 7): 207 W; 207/{Vb*ib:.0f} = {207/(Vb*ib)*100:.0f} %; el eje de Gazebo pide {4*KQ*OM_H**3:.0f} W porque su par k_Q Omega^2 = {KQ*OM_H**2:.4f} N m es mayor que el de la BEM (0,0672 N m)")
print(f"  autonomía aproximada Q / I medio = {CAP} Ah / 17,2 A = {CAP/17.2*60:.1f} min")

# curva de descarga (misma integración que figuras_b2.py)
soc, t, h = 1.0, 0.0, 1.0
F = []
while True:
    Vb = 4 * ocv(soc)
    for _ in range(20):
        dd, im, ib = estacionario(Vb)
        Vb = 4 * (ocv(soc) - R0 * ib)
    F.append((t, soc, 4 * ocv(soc), Vb, ib))
    if ocv(soc) <= V_EMPTY + 1e-9 or t > 3600:
        break
    soc = max(soc - ib * h / 3600 / CAP, 0.0)
    t += h
F = np.array(F)
X12 = lin(0, 18, 60, 520)
Y12 = lin(13.5, 17.0, 250, 20)
s = slice(0, None, 15)
print("path descarga ocv:", path(F[s, 0] / 60, F[s, 2], X12, Y12))
print("path descarga bus:", path(F[s, 0] / 60, F[s, 3], X12, Y12))
print(f"  final: t = {F[-1,0]/60:.2f} min, 4 OCV = {F[-1,2]:.2f}, V_bus = {F[-1,3]:.2f}, I = {F[-1,4]:.2f} A; caída R0 I total = {F[0,2]-F[0,3]:.2f} V al inicio y {F[-1,2]-F[-1,3]:.2f} V al final")
print(f"  ejes descarga: x(5) = {X12(5):.0f}, x(10) = {X12(10):.0f}, x(15) = {X12(15):.0f}; y(14) = {Y12(14):.1f}, y(15) = {Y12(15):.1f}, y(16) = {Y12(16):.1f}, y(17) = {Y12(17):.1f}")
print(f"  posición final en la figura: x = {X12(F[-1,0]/60):.0f}")

# ----------------------------------------------------------------------------- PARTE 3 · VIENTO
FT = 0.3048
h_ft = 10 / FT
W20 = 15 * 0.514444
sw = 0.1 * W20
su = sw / (0.177 + 0.000823 * h_ft) ** 0.4
Lw = h_ft * FT
Lu = h_ft / (0.177 + 0.000823 * h_ft) ** 1.2 * FT
U = 5.0
Tu = Lu / U

cab(13, "Densidad espectral de un proceso con memoria exponencial (la ráfaga u)")
print(f"  sigma_u = {su:.3f} m/s; T = L_u/V = {Lu:.1f}/{U} = {Tu:.2f} s")
print(f"  autocorrelación R(tau) = sigma^2 e^(-|tau|/T): R(0) = {su**2:.3f}; R(T) = {su**2*math.exp(-1):.3f} (m/s)^2; R(3T = {3*Tu:.0f} s) = {su**2*math.exp(-3):.3f}")
G0 = 4 * su ** 2 * Tu
fc = 1 / (2 * math.pi * Tu)
print(f"  G(f) = 4 sigma^2 T/(1 + (2 pi f T)^2): G(0) = {G0:.1f} (m/s)^2/Hz; corte f_c = 1/(2 pi T) = {fc:.4f} Hz")
for f_ in (0.01, 0.1, 1.0):
    print(f"   G({f_}) = {4*su**2*Tu/(1+(2*math.pi*f_*Tu)**2):.3g}")
for f_ in (0.012, 0.1, 1.0):
    frac = 2 / math.pi * math.atan(2 * math.pi * f_ * Tu)
    print(f"  fracción de la varianza por debajo de {f_} Hz: (2/pi) atan(2 pi f T) = {frac*100:.1f} %")
ff = np.logspace(-4, 4, 400001)
Gf = 4 * su ** 2 * Tu / (1 + (2 * math.pi * ff * Tu) ** 2)
integ = np.trapezoid(Gf, ff) if hasattr(np, "trapezoid") else np.trapz(Gf, ff)
print(f"  integral numérica de G entre 1e-4 y 1e4 Hz = {integ:.3f} (sigma^2 = {su**2:.3f})")
# figura: autocorrelación (lineal) y espectro (log-log)
lag = np.linspace(0, 60, 61)
XA = lin(0, 60, 50, 250)
YA = lin(0, 2.4, 230, 40)
print("path autocorr:", path(lag, su ** 2 * np.exp(-lag / Tu), XA, YA))
print(f"  ejes autocorr: x(T) = {XA(Tu):.1f}, y(R(T)) = {YA(su**2*math.exp(-1)):.1f}, y(sigma^2) = {YA(su**2):.1f}, y(1) = {YA(1):.1f}, y(2) = {YA(2):.1f}, x(20) = {XA(20):.0f}, x(40) = {XA(40):.0f}")
fl = np.logspace(-3, 1, 41)
XP = lin(-3, 1, 320, 520)
YP = lin(-3, 3, 230, 40)
print("path psd concepto:", path(np.log10(fl), np.log10(4 * su ** 2 * Tu / (1 + (2 * math.pi * fl * Tu) ** 2)), XP, YP))
print(f"  ejes psd: x(f_c) = {XP(math.log10(fc)):.1f}, y(G0) = {YP(math.log10(G0)):.1f}; x(1e-3) = {XP(-3):.0f}, x(1e-2) = {XP(-2):.0f}, x(1e-1) = {XP(-1):.0f}, x(1) = {XP(0):.0f}, x(10) = {XP(1):.0f}; "
      f"y(1e2) = {YP(2):.0f}, y(1) = {YP(0):.0f}, y(1e-2) = {YP(-2):.0f}")
print(f"  ruido blanco de igual varianza en 0-50 Hz: densidad {su**2/50:.4f} (m/s)^2/Hz -> y = {YP(math.log10(su**2/50)):.1f}")

cab(14, "Escalas e intensidades de Dryden a 10 m con turbulencia ligera")
base = 0.177 + 0.000823 * h_ft
print(f"  h = 10 m = {h_ft:.1f} ft; 0,177 + 0,000823 h = {base:.4f}")
print(f"  W20 = 15 kt = {W20:.2f} m/s; sigma_w = 0,1 W20 = {sw:.3f} m/s")
print(f"  {base:.4f}^0,4 = {base**0.4:.4f} -> sigma_u = sigma_v = {su:.3f} m/s")
print(f"  {base:.4f}^1,2 = {base**1.2:.4f} -> L_u = L_v = {h_ft:.1f}/{base**1.2:.4f} = {h_ft/base**1.2:.0f} ft = {Lu:.1f} m; L_w = h = {Lw:.1f} m")
print(f"  con V = 5 m/s: L_u/V = {Lu/U:.2f} s, L_w/V = {Lw/U:.2f} s")

cab(15, "El filtro conformador reproduce el espectro (comprobación de |H|^2)")
om = np.logspace(-6, 6, 2000001)
Hu2 = su ** 2 * 2 * Lu / (math.pi * U) / (1 + (Lu / U * om) ** 2)
Tw = Lw / U
Hw2 = sw ** 2 * Lw / (math.pi * U) * (1 + 3 * (Tw * om) ** 2) / (1 + (Tw * om) ** 2) ** 2
tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
print(f"  integral de |H_u(j omega)|^2 en omega de 0 a inf = {tz(Hu2, om):.3f} (sigma_u^2 = {su**2:.3f})")
print(f"  integral de |H_w(j omega)|^2 = {tz(Hw2, om):.4f} (sigma_w^2 = {sw**2:.4f})")
print(f"  ganancia de H_u a frecuencia 0: sigma_u sqrt(2 L_u/(pi V)) = {su*math.sqrt(2*Lu/(math.pi*U)):.3f}")
print(f"  en Hz: G(f) = 2 pi |H_u(j 2 pi f)|^2 -> G(0) = {2*math.pi*su**2*2*Lu/(math.pi*U):.1f} (= 4 sigma^2 T = {G0:.1f})")

cab(16, "Recursión exacta frente a Euler")
for nombre, tau_, sig_, dt in [("u, dt = 0,01 s", Tu, su, 0.01), ("w, dt = 0,01 s", Tw, sw, 0.01), ("w, dt = 0,1 s", Tw, sw, 0.1), ("w, dt = 0,5 s", Tw, sw, 0.5)]:
    a_ = math.exp(-dt / tau_)
    q = sig_ * math.sqrt(1 - a_ * a_)
    # Euler: x_k = (1 - dt/tau) x_{k-1} + sig sqrt(2 dt/tau) w_k  -> varianza estacionaria
    aE = 1 - dt / tau_
    qE2 = sig_ ** 2 * 2 * dt / tau_
    varE = qE2 / (1 - aE ** 2)
    print(f"  {nombre}: tau = {tau_:.2f} s; a = e^(-dt/tau) = {a_:.6f}; sqrt(1-a^2) = {math.sqrt(1-a_*a_):.5f}; "
          f"desv. del ruido por paso = {q:.4f} m/s; Euler: sigma_est/sigma = {math.sqrt(varE)/sig_:.4f}")
rng = np.random.default_rng(3)
a_ = math.exp(-0.5 / Tw)
xk, xs_ = 0.0, []
for k in range(200000):
    xk = a_ * xk + sw * math.sqrt(1 - a_ * a_) * rng.standard_normal()
    xs_.append(xk)
print(f"  comprobación (200 000 pasos de 0,5 s, recursión exacta): desv. típica {np.std(xs_):.4f} (sigma_w = {sw:.4f})")
print(f"  varianza: a^2 sigma^2 + sigma^2 (1 - a^2) = sigma^2 para cualquier a")

cab(17, "Cuánto empuja la ráfaga según el arrastre")
for nombre, D in [("Gazebo", 0.248), ("Faessler x masa", 0.465 * M), ("SIH", 1.0)]:
    print(f"  {nombre:16s}: D = {D:.3f} N s/m; D/m = {D/M:.3f} 1/s; aceleración típica de la ráfaga (D/m) sigma_u = {D/M*su:.3f} m/s^2")
print(f"  cociente SIH/Gazebo = {1/0.248:.2f} (la desviación de posición: 15,9/4,0 = {15.9/4.0:.2f})")

# ----------------------------------------------------------------------------- PARTE 4 · IDENTIFICACIÓN
cab(18, "Mínimos cuadrados con cuatro muestras: a = -d v")
v = np.array([-2.0, -1.0, 1.0, 2.0])
eps = np.array([0.03, -0.05, 0.04, -0.02])        # ruido elegido a mano para que se siga
a = -0.45 * v + eps
print(f"  v = {v}; ruido = {eps}; a = -0,45 v + ruido = {a}")
Phi = -v.reshape(-1, 1)                             # regresor: a = (-v) d
PtP, Pty = (Phi.T @ Phi).item(), (Phi.T @ a).item()
dh = Pty / PtP
res = a - Phi[:, 0] * dh
s2 = res @ res / (len(v) - 1)
print(f"  Phi = -v = {Phi[:,0]}; Phi^T Phi = sum v^2 = {PtP:.0f}; Phi^T y = -sum v a = {Pty:.3f}")
print(f"  d_hat = {Pty:.3f}/{PtP:.0f} = {dh:.4f}")
print(f"  residuos = {np.round(res,4)}; sigma_hat^2 = sum r^2/(N - 1) = {s2:.5f}; desv. de d_hat = sqrt(sigma^2/sum v^2) = {math.sqrt(s2/PtP):.4f}")

XS = lin(-2.5, 2.5, 40, 300)
YS = lin(-1.2, 1.2, 210, 20)
print("puntos ej18:", " ".join(f"({XS(vi):.1f},{YS(ai):.1f})" for vi, ai in zip(v, a)))
print("recta ej18:", path([-2.4, 2.4], [-dh * -2.4, -dh * 2.4], XS, YS))
print(f"  ejes ej18: x(-2) = {XS(-2):.0f}, x(0) = {XS(0):.0f}, x(2) = {XS(2):.0f}; y(1) = {YS(1):.1f}, y(0) = {YS(0):.1f}, y(-1) = {YS(-1):.1f}")

cab(19, "Dos parámetros y la identificabilidad: c = b + k_h v^2")
for nombre, vv in [("lento", np.array([0.25, 0.5, 0.75, 1.0])), ("rápido", np.array([1.0, 2.0, 3.0, 4.0]))]:
    Ph = np.column_stack([np.ones(4), vv ** 2])
    A2 = Ph.T @ Ph
    inv = np.linalg.inv(A2)
    ev = np.linalg.eigvalsh(A2)
    print(f"  {nombre}: v = {vv}; v^2 = {vv**2}")
    print(f"     Phi^T Phi = [[{A2[0,0]:.0f}, {A2[0,1]:.4g}], [{A2[1,0]:.4g}, {A2[1,1]:.4g}]]; det = {np.linalg.det(A2):.4g}; "
          f"Var(v^2) = {np.var(vv**2):.4g}")
    print(f"     [(Phi^T Phi)^-1]_22 = {inv[1,1]:.4g} -> con sigma = 0,05: desv. de k_h_hat = {0.05*math.sqrt(inv[1,1]):.4f} m^-1 ({0.05*math.sqrt(inv[1,1])/0.009*100:.0f} % de 0,009)")
    print(f"     comprobación: N Var(v^2) = {4*np.var(vv**2):.4g} = det/N = {np.linalg.det(A2)/4:.4g}")
# un caso numérico concreto, con el mismo ruido de la hoja
eps2 = np.array([0.03, -0.05, 0.04, -0.02])
for nombre, vv in [("lento", np.array([0.25, 0.5, 0.75, 1.0])), ("rápido", np.array([1.0, 2.0, 3.0, 4.0]))]:
    Ph = np.column_stack([np.ones(4), vv ** 2])
    c_ = G + 0.009 * vv ** 2 + eps2
    th = np.linalg.solve(Ph.T @ Ph, Ph.T @ c_)
    print(f"  {nombre}, con el ruido del ejemplo 18: c = {np.round(c_,4)} -> b_hat = {th[0]:.4f}, k_h_hat = {th[1]:.4f} (verdad 0,009; error {th[1]-0.009:+.4f})")
print("  con v cuatro veces mayor, v^2 es 16 veces mayor: el mismo ruido da un error de k_h_hat 16 veces menor")

cab(20, "Cómo escala la incertidumbre con la velocidad máxima (3000 muestras, sigma = 0,05)")
N, s_a = 3000, 0.05
print("  v_max   desv(d_hat)/d   fórmula sigma/sqrt(N v^2/3)/d   desv(k_h_hat)/k_h   fórmula sigma sqrt(45/4)/(sqrt(N) v^2)/k_h")
# desviaciones relativas (%) de 50 conjuntos de 3000 muestras: salida de figuras_b2.py, sección 4
sim_d = {0.25: 1.27, 0.5: 0.63, 1: 0.37, 2: 0.18, 4: 0.10, 8: 0.05}
sim_k = {0.25: 454.5, 0.5: 116.0, 1: 33.3, 2: 7.5, 4: 1.9, 8: 0.5}
for vm in (0.25, 0.5, 1, 2, 4, 8):
    fd = s_a / math.sqrt(N * vm ** 2 / 3)
    fk = s_a * math.sqrt(45 / 4) / (math.sqrt(N) * vm ** 2)
    print(f"  {vm:5.2f}   {sim_d[vm]:6.2f} %        {fd/0.45*100:6.2f} %                     {sim_k[vm]:7.1f} %        {fk/0.009*100:7.1f} %")
print("  momentos de v uniforme en [-v_max, v_max]: E[v^2] = v_max^2/3, E[v^4] = v_max^4/5, Var(v^2) = v_max^4/5 - v_max^4/9 = 4 v_max^4/45")
print(f"  errores en las variables, v_max = 0,25: Var(v) = {0.25**2/3:.4f}; sigma_v^2 = {0.02**2}; factor 1/(1 + {0.02**2/(0.25**2/3):.4f}) = {1/(1+0.02**2/(0.25**2/3)):.4f}")

cab(21, "Caja gris frente a caja negra: número de parámetros y lo que extrapolan")
print("  gris lineal -d v: 1 parámetro; gris -d v - c v|v|: 2; polinomio de grado 7: 8 coeficientes")
print("  a 12 m/s (figuras_b2.py, sección 4): verdad 5,98; gris lineal 5,55 (-7 %); gris cuadrático 6,07 (+1,6 %); polinomio 255 (x43)")
print(f"  errores relativos: {(5.551-5.976)/5.976*100:.1f} %, {(6.073-5.976)/5.976*100:.1f} %, x{254.589/5.976:.0f}")
print(f"  residuo que tendría que aprender una red sobre el gris lineal a 8 m/s: -3,856 - (-3,700) = {-3.856+3.700:.3f} m/s^2")
