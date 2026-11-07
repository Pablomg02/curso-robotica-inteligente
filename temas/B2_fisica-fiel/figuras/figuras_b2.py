"""Datos de las figuras y tablas del tema B2 (física fiel).

Uso:  python3 figuras_b2.py > salida.txt      (numpy; menos de un minuto)

Vehículo de referencia: el x500 de PX4 en Gazebo (A1-B1): m = 2,064 kg, kT = 8,54858e-6 N/(rad/s)^2,
km = 0,016 m (par = km * empuje), Omega_hover = 769,6 rad/s, Omega_max = 1000 rad/s.
Los parámetros que no salen de una fuente (geometría de la pala, motor eléctrico, batería, J del rotor)
están marcados como SUPUESTOS: son valores típicos elegidos para que el modelo reproduzca el empuje
y el par de Gazebo, no datos de un fabricante.
"""
import math
import numpy as np

G, RHO = 9.81, 1.225
M = 2.0 + 4 * 0.016076923076923075
KT, KM = 8.54858e-06, 0.016
KQ = KM * KT                     # par aerodinámico = KQ * Omega^2
OM_H = math.sqrt(M * G / (4 * KT))
T_H = M * G / 4


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


# ---------------------------------------------------------------- 1. rotor: elemento de pala + cantidad de movimiento
print("=" * 70)
print("1. ROTOR: TEORÍA DEL ELEMENTO DE PALA CON FLUJO INDUCIDO UNIFORME (Glauert)")
R_P = 0.127          # SUPUESTO: hélice de 10 pulgadas
NB, CH = 2, 0.020    # SUPUESTO: 2 palas de cuerda media 2 cm
SIG = NB * CH / (math.pi * R_P)
A_L = 5.7            # SUPUESTO: pendiente de sustentación (1/rad)
TH_TW = -0.20        # SUPUESTO: torsión lineal (rad, de raíz a punta)
CD0 = 0.020          # SUPUESTO: resistencia de perfil
KAPPA = 1.15         # factor de potencia inducida (típico)
AREA = math.pi * R_P ** 2
VT_H = OM_H * R_P
CT_H = T_H / (RHO * AREA * VT_H ** 2)
LAM_H = math.sqrt(CT_H / 2)
# paso colectivo que da el empuje de estacionario del x500 a Omega_hover
TH0 = 3 * (CT_H / (SIG * A_L / 2) + LAM_H / 2 - TH_TW / 4)
print(f"  sigma = {SIG:.4f}; Omega_h = {OM_H:.1f} rad/s; velocidad en punta {VT_H:.1f} m/s")
print(f"  C_T en estacionario = {CT_H:.5f}; lambda_h = {LAM_H:.4f}; v_h = {LAM_H*VT_H:.2f} m/s")
print(f"  paso en la raíz theta_0 = {math.degrees(TH0):.1f}°, torsión {math.degrees(TH_TW):.1f}°")


def ct_bem(mu, lam_c, th0=TH0):
    """C_T y lambda total para avance mu (en el plano del disco) y subida lam_c (perpendicular)."""
    lam = LAM_H + lam_c
    for _ in range(200):
        ct = SIG * A_L / 2 * (th0 / 3 * (1 + 1.5 * mu ** 2) + TH_TW / 4 * (1 + mu ** 2) - lam / 2)
        lam_new = lam_c + ct / (2 * math.sqrt(mu ** 2 + lam ** 2))
        lam = 0.5 * lam + 0.5 * lam_new
    return ct, lam


def cp_bem(mu, lam_c):
    ct, lam = ct_bem(mu, lam_c)
    lam_i = lam - lam_c
    return KAPPA * lam_i * ct + lam_c * ct + SIG * CD0 / 8 * (1 + 4.65 * mu ** 2), ct


cp0, _ = cp_bem(0, 0)
FM = CT_H ** 1.5 / math.sqrt(2) / cp0
print(f"  C_P en estacionario {cp0:.6f}; figura de mérito {FM:.2f} (A1: 0,5-0,6 en hélices pequeñas)")
print(f"  potencia en estacionario por rotor: {cp0*RHO*AREA*VT_H**3:.1f} W; total {4*cp0*RHO*AREA*VT_H**3:.0f} W")
VS = np.linspace(0, 15, 31)
rat_av = np.array([ct_bem(v / VT_H, 0)[0] / CT_H for v in VS])
# con el disco inclinado hacia delante el ángulo de equilibrio con arrastre lineal d = 0,465 1/s: tan(alpha) = d v / g
D_TRIM = 0.465
alf = np.arctan(D_TRIM * VS / G)
rat_tr = np.array([ct_bem(v * math.cos(a_) / VT_H, v * math.sin(a_) / VT_H)[0] / CT_H for v, a_ in zip(VS, alf)])
rat_sub = np.array([ct_bem(0, v / VT_H)[0] / CT_H for v in VS[:13]])   # subida hasta 6 m/s
faes = 1 + 0.009 * VS ** 2 / G
for v in (2, 4, 5, 10, 15):
    i = int(v * 2)
    print(f"  avance {v:2d} m/s, Omega fijo: disco horizontal T/T_h = {rat_av[i]:.3f}; disco inclinado {math.degrees(alf[i]):4.1f}°: {rat_tr[i]:.3f}; Faessler (k_h = 0,009 1/m, c/c_h): {faes[i]:.3f}")
for v in (2, 4, 6):
    i = int(v * 2)
    print(f"  subida {v} m/s, Omega fijo: T/T_h = {rat_sub[i]:.3f}")
X1 = lin(0, 15, 60, 520)
Y1 = lin(0.4, 1.6, 250, 20)
print("path rotor avance:", path(VS, rat_av, X1, Y1))
print("path rotor faessler:", path(VS[:9], faes[:9], X1, Y1))
print("path rotor faessler ext:", path(VS[8:], faes[8:], X1, Y1))
print("path rotor inclinado:", path(VS, rat_tr, X1, Y1))
print("path rotor subida:", path(VS[:13], rat_sub, X1, Y1))
print(f"  ejes: y(1) = {Y1(1):.1f}, y(0,6) = {Y1(0.6):.1f}, y(1,4) = {Y1(1.4):.1f}, y(0,8) = {Y1(0.8):.1f}, y(1,2) = {Y1(1.2):.1f}; x(4) = {X1(4):.0f}, x(5) = {X1(5):.0f}, x(10) = {X1(10):.0f}, x(15) = {X1(15):.0f}")
for v in (6, 15):
    i = int(v * 2)
    print(f"  extremos: v = {v}: avance ({X1(v):.0f}, {Y1(rat_av[i]):.0f}); inclinado ({X1(v):.0f}, {Y1(rat_tr[i]):.0f}); subida ({X1(min(v,6)):.0f}, {Y1(rat_sub[min(i,12)]):.0f}); Faessler ({X1(v):.0f}, {Y1(faes[i]):.0f})")
# derivada en subida: dT/dVc
d = (ct_bem(0, 0.5 / VT_H)[0] - CT_H) / CT_H / 0.5
print(f"  sensibilidad en subida: {d*100:.1f} % de empuje por m/s; teoría de cantidad de movimiento con theta fijo")

# ---------------------------------------------------------------- 2. motor, ESC y batería
print("=" * 70)
print("2. MOTOR SIN ESCOBILLAS, ESC Y BATERÍA")
R_M = 0.12          # SUPUESTO: resistencia de fase equivalente (ohm)
J_R = 2.5e-5        # SUPUESTO: inercia de rotor + hélice (kg m^2)
V_FULL = 4 * 4.2    # 4S cargada (BAT_V_CHARGED por defecto en PX4: 4,2 V por celda)
# k_e tal que, con 16,8 V y mando 1, el motor llega a 1000 rad/s con la carga de la hélice
qa = KQ * 1000 ** 2 * R_M
KE = (V_FULL + math.sqrt(V_FULL ** 2 - 4 * 1000 * qa)) / (2 * 1000)   # raíz física (la otra da un KV absurdo)
KTM = KE
print(f"  k_e = k_t = {KE:.5f} V/(rad/s) -> KV = {60/(2*math.pi*KE):.0f} rpm/V (motor equivalente, SUPUESTO R = {R_M} ohm, J = {J_R:.1e})")


def om_ss(duty, V):
    """Velocidad estacionaria: k_t (dV - k_e w)/R = KQ w^2."""
    a, b, c = KQ * R_M, KTM * KE, -KTM * duty * V
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


tau_lin = J_R / (KTM * KE / R_M + 2 * KQ * OM_H)
print(f"  constante de tiempo linealizada en estacionario: tau = J/(k_t k_e/R + 2 KQ Omega) = {tau_lin*1e3:.1f} ms")
print(f"    reparto: eléctrico {KTM*KE/R_M:.5f}, aerodinámico {2*KQ*OM_H:.5f} (N m s)")
print(f"  Gazebo x500: 12,5 ms al subir y 25 ms al bajar")
d_h = None
for dd in np.linspace(0.3, 1, 7001):
    if om_ss(dd, V_FULL) >= OM_H:
        d_h = dd
        break
print(f"  mando de estacionario con batería llena: {d_h:.3f}")


def simula_motor(d0, d1, V, freno, T=0.25, h=1e-5):
    w = om_ss(d0, V)
    out = []
    for k in range(int(T / h)):
        dd = d1 if k * h >= 0.02 else d0
        i = (dd * V - KE * w) / R_M
        if not freno:
            i = max(i, 0.0)      # sin frenado activo: la corriente no se invierte
        w += h * (KTM * i - KQ * w * w) / J_R
        out.append((k * h, w))
    return np.array(out)


w_lo, w_hi = om_ss(0.55, V_FULL), om_ss(0.80, V_FULL)
res_m = {}
for nombre, d0, d1, fr in [("sube", 0.55, 0.80, True), ("baja freno", 0.80, 0.55, True), ("baja libre", 0.80, 0.55, False)]:
    r = simula_motor(d0, d1, V_FULL, fr)
    res_m[nombre] = r
    w0, w1 = r[0, 1], r[-1, 1]
    obj = w0 + 0.632 * (w1 - w0)
    k = np.argmax((r[:, 1] - obj) * np.sign(w1 - w0) >= 0)
    print(f"  escalón {nombre:10s}: {w0:6.1f} -> {w1:6.1f} rad/s; tiempo al 63,2 %: {(r[k,0]-0.02)*1e3:5.1f} ms")
t_libre = J_R / KQ * (1 / w_lo - 1 / w_hi)
print(f"  sin frenado y sin corriente, J w' = -KQ w^2: de {w_hi:.0f} a {w_lo:.0f} rad/s en {t_libre*1e3:.0f} ms (forma cerrada)")
X2 = lin(0, 0.20, 60, 520)
Y2 = lin(w_lo - 40, w_hi + 40, 250, 20)
for nombre in res_m:
    r = res_m[nombre][::100]
    sel = r[:, 0] <= 0.22
    print(f"path motor {nombre}:", path(r[sel, 0] - 0.02, r[sel, 1], X2, Y2))
print(f"  ejes motor: y({w_lo:.0f}) = {Y2(w_lo):.1f}; y({w_hi:.0f}) = {Y2(w_hi):.1f}; x(0,05) = {X2(0.05):.0f}, x(0,1) = {X2(0.1):.0f}, x(0,15) = {X2(0.15):.0f}")

print("-" * 70)
print("   Batería 4S en estacionario: tensión, mando y margen de empuje")
CAP = 5.0            # SUPUESTO: 5 Ah
R0 = 0.005           # R_DEFAULT de PX4 (estimación inicial por celda, ohm)
V_EMPTY, V_CH = 3.6, 4.2   # BAT_V_EMPTY y BAT_V_CHARGED por defecto en PX4


def ocv(soc):          # modelo lineal de PX4 entre vacía y cargada (por celda)
    return V_EMPTY + soc * (V_CH - V_EMPTY)


def estacionario(Vb):
    """Mando, corriente de cada motor y corriente de batería para el estacionario con tensión de bus Vb."""
    i_m = KQ * OM_H ** 2 / KTM
    dd = (KE * OM_H + R_M * i_m) / Vb
    p = 4 * dd * Vb * i_m
    return dd, i_m, p / Vb


soc, t, h = 1.0, 0.0, 1.0
filas = []
while True:
    # tensión de bus con la caída en la resistencia interna (punto fijo)
    Vb = 4 * ocv(soc)
    for _ in range(20):
        dd, im, ib = estacionario(Vb)
        Vb = 4 * (ocv(soc) - R0 * ib)
    wmax = om_ss(1.0, Vb)
    marg = KT * wmax ** 2 * 4 / (M * G)
    filas.append((t, soc, Vb, dd, ib, marg, min(V_CH / (Vb / 4), 1.3)))
    if ocv(soc) <= V_EMPTY + 1e-9 or t > 3600:
        break
    soc -= ib * h / 3600 / CAP
    soc = max(soc, 0.0)
    t += h
F = np.array(filas)
print(f"  autonomía hasta {V_EMPTY} V por celda (OCV): {F[-1,0]/60:.1f} min con corriente media {F[:,4].mean():.1f} A")
for frac in (0.0, 0.5, 1.0):
    k = int(frac * (len(F) - 1))
    print(f"  t = {F[k,0]/60:5.1f} min: SoC {F[k,1]:.2f}, V_bus {F[k,2]:5.2f} V, mando {F[k,3]:.3f}, "
          f"I {F[k,4]:4.1f} A, empuje máx./peso {F[k,5]:.2f}, escala de PX4 {F[k,6]:.3f}")
X3 = lin(0, F[-1, 0] / 60, 60, 520)
Y3a = lin(0.7, 1.0, 250, 20)
Y3b = lin(1.0, 2.0, 250, 20)
sel = slice(0, None, 15)
print("path bat mando:", path(F[sel, 0] / 60, F[sel, 3], X3, Y3a))
print("path bat margen:", path(F[sel, 0] / 60, F[sel, 5], X3, Y3b))
print(f"  ejes bat: x(5 min) = {X3(5):.0f}, x(10) = {X3(10):.0f}; mando y(0,8) = {Y3a(0.8):.1f}, y(0,9) = {Y3a(0.9):.1f}, y(1,0) = {Y3a(1.0):.1f}; "
      f"margen y(1,2) = {Y3b(1.2):.1f}, y(1,4) = {Y3b(1.4):.1f}, y(1,6) = {Y3b(1.6):.1f}, y(1,8) = {Y3b(1.8):.1f}")

# ---------------------------------------------------------------- 3. viento: turbulencia de Dryden
print("=" * 70)
print("3. TURBULENCIA DE DRYDEN A BAJA ALTURA (h < 1000 ft) Y SU EFECTO EN EL ESTACIONARIO")
FT = 0.3048
h_ft = 10 / FT
W20 = 15 * 0.514444            # turbulencia ligera: W20 = 15 kt
sw = 0.1 * W20
su = sw / (0.177 + 0.000823 * h_ft) ** 0.4
Lw = h_ft * FT
Lu = h_ft / (0.177 + 0.000823 * h_ft) ** 1.2 * FT
U = 5.0                        # viento medio que transporta la turbulencia (hipótesis de Taylor)
print(f"  h = 10 m ({h_ft:.1f} ft), W20 = {W20:.2f} m/s: sigma_w = {sw:.3f}, sigma_u = sigma_v = {su:.3f} m/s; L_w = {Lw:.1f} m, L_u = L_v = {Lu:.1f} m")
print(f"  con viento medio U = {U} m/s: escalas de tiempo L_u/U = {Lu/U:.2f} s, L_w/U = {Lw/U:.2f} s")
rng = np.random.default_rng(0)
DT, TT = 0.01, 600.0
n = int(TT / DT)


def filtro_u(sig, L, V, w):
    """H(s) = sig sqrt(2L/(pi V)) / (1 + (L/V) s), discretizado exacto con ruido blanco de densidad unidad."""
    tau = L / V
    a = math.exp(-DT / tau)
    # varianza estacionaria del primer orden con entrada blanca de PSD unilateral 1: K^2/(4 tau)... normalizamos directamente
    x = np.zeros(len(w))
    b = sig * math.sqrt(1 - a * a)
    for k in range(1, len(w)):
        x[k] = a * x[k - 1] + b * w[k]
    return x


def filtro_w(sig, L, V, w):
    """Forma de H_w(s) = (1 + sqrt3 (L/V) s)/(1 + (L/V) s)^2 en espacio de estados (Euler con 20 subpasos); se reescala a sigma_w."""
    tau = L / V
    # realización: x1' = x2, x2' = (-x1 - 2 tau x2 + u)/tau^2, y = K (x1 + sqrt3 tau x2)
    K = 1.0
    sub = 20
    hh = DT / sub
    x1 = x2 = 0.0
    y = np.zeros(len(w))
    for k in range(len(w)):
        u = w[k] / math.sqrt(DT)   # ruido blanco; la amplitud se fija después reescalando a sigma_w
        for _ in range(sub):
            x1, x2 = x1 + hh * x2, x2 + hh * (-x1 - 2 * tau * x2 + u) / tau ** 2
        y[k] = K * (x1 + math.sqrt(3) * tau * x2)
    return y


wu = filtro_u(su, Lu, U, rng.standard_normal(n))
wv = filtro_u(su, Lu, U, rng.standard_normal(n))
wn = rng.standard_normal(n)
ww = filtro_w(sw, Lw, U, wn)
ww *= sw / ww.std()   # la realización del filtro vertical se reescala a sigma_w (la forma espectral se mantiene)
print(f"  desviaciones típicas simuladas (600 s): u {wu.std():.3f}, v {wv.std():.3f}, w {ww.std():.3f} m/s (w reescalada)")
# espectro de u frente al analítico
f = np.fft.rfftfreq(n, DT)
P = 2 * np.abs(np.fft.rfft(wu - wu.mean())) ** 2 * DT / n
nb = 40
edges = np.logspace(math.log10(0.005), math.log10(20), nb + 1)
fc, Pm = [], []
for i in range(nb):
    s = (f >= edges[i]) & (f < edges[i + 1])
    if s.sum() > 3:
        fc.append(math.sqrt(edges[i] * edges[i + 1]))
        Pm.append(P[s].mean())
fc, Pm = np.array(fc), np.array(Pm)
om = 2 * math.pi * fc
Pa = su ** 2 * 4 * (Lu / U) / (1 + (Lu / U * om) ** 2)   # PSD unilateral en Hz del primer orden
err = np.abs(np.log10(Pm / Pa))
print(f"  espectro de u: desviación media |log10(simulado/analítico)| = {err.mean():.3f} (en {len(fc)} bandas)")
X4 = lin(-2.3, 1.3, 60, 520)
Y4 = lin(-6, 2, 250, 20)
print("path dryden sim:", path(np.log10(fc), np.log10(Pm), X4, Y4))
fa = np.logspace(-2.3, 1.3, 80)
Pa2 = su ** 2 * 4 * (Lu / U) / (1 + (Lu / U * 2 * math.pi * fa) ** 2)
print("path dryden teo:", path(np.log10(fa), np.log10(Pa2), X4, Y4))
fcorte = U / (2 * math.pi * Lu)
print(f"  frecuencia de corte de u: U/(2 pi L_u) = {fcorte:.4f} Hz -> x = {X4(math.log10(fcorte)):.1f}")
for e in (-2, -1, 0, 1):
    print(f"  marca x 1e{e} Hz: {X4(e):.1f}")
for e in (-6, -4, -2, 0, 2):
    print(f"  marca y 1e{e}: {Y4(e):.1f}")
X4t = lin(0, 60, 60, 520)
Y4t = lin(-4, 4, 250, 20)
print("path dryden serie:", path(np.arange(0, 6000, 10) * DT, wu[:6000:10], X4t, Y4t))

print("-" * 70)
print("   Estacionario del x500 con esa turbulencia (plano horizontal, PD de A2/B1: Kp = 4, Kd = 3,5)")
for nombre, D in [("Gazebo x500 (0,248 N/(m/s))", 0.248), ("Faessler escalado al x500 (0,465 1/s x m)", 0.465 * M), ("SIH (1 N/(m/s))", 1.0)]:
    # x'' = -Kp x - Kd x' + (D/m)(w - x'), con la ráfaga respecto al viento medio (el medio lo compensa la integral)
    x = v = 0.0
    xs = []
    for k in range(n):
        a = -4 * x - 3.5 * v + D / M * (wu[k] - v)
        v += DT * a
        x += DT * v
        xs.append(x)
    xs = np.array(xs[1000:])
    print(f"  {nombre:44s}: desviación típica de posición {xs.std()*100:5.1f} cm; máx. {np.abs(xs).max()*100:5.1f} cm")

# ---------------------------------------------------------------- 4. identificación
print("=" * 70)
print("4. IDENTIFICACIÓN: ARRASTRE Y SUSTENTACIÓN DE TRASLACIÓN POR MÍNIMOS CUADRADOS")
D_TRUE, KH_TRUE = 0.45, 0.009
SIG_A = 0.05      # ruido del acelerómetro en m/s^2 (incluye vibración; SUPUESTO)
SIG_V = 0.02      # ruido de la velocidad de captura de movimiento (m/s; SUPUESTO)


def datos(vmax, n=3000, rng=None):
    v = rng.uniform(-vmax, vmax, n)
    a = -D_TRUE * v + rng.normal(0, SIG_A, n)              # aceleración horizontal en ejes cuerpo
    c = G + KH_TRUE * v ** 2 + rng.normal(0, SIG_A, n)     # empuje específico extra (dirección z)
    return v + rng.normal(0, SIG_V, n), a, c


rng = np.random.default_rng(1)
print("  vmax   d estimado (media ± desv.)   k_h estimado (media ± desv.)   [50 conjuntos de 3000 muestras]")
filas_id = []
for vmax in (0.25, 0.5, 1, 2, 4, 8):
    ds, ks = [], []
    for _ in range(50):
        vm, a, c = datos(vmax, rng=rng)
        ds.append(-(vm @ a) / (vm @ vm))
        Phi = np.column_stack([np.ones_like(vm), vm ** 2])
        th = np.linalg.lstsq(Phi, c, rcond=None)[0]
        ks.append(th[1])
    ds, ks = np.array(ds), np.array(ks)
    filas_id.append((vmax, ds.mean(), ds.std(), ks.mean(), ks.std()))
    print(f"  {vmax:4.2f}   {ds.mean():.4f} ± {ds.std():.4f}            {ks.mean():.5f} ± {ks.std():.5f}")
print(f"  valores verdaderos: d = {D_TRUE}, k_h = {KH_TRUE}")
X6 = lin(math.log10(0.2), math.log10(10), 60, 520)
Y6 = lin(-4, 1, 250, 20)
rel_k = [(math.log10(f[0]), math.log10(f[4] / KH_TRUE)) for f in filas_id]
rel_d = [(math.log10(f[0]), math.log10(f[2] / D_TRUE)) for f in filas_id]
print("path id kh:", path([r[0] for r in rel_k], [r[1] for r in rel_k], X6, Y6))
print("path id d:", path([r[0] for r in rel_d], [r[1] for r in rel_d], X6, Y6))
for f in filas_id:
    print(f"  punto vmax {f[0]}: x = {X6(math.log10(f[0])):.1f}; k_h y = {Y6(math.log10(f[4]/KH_TRUE)):.1f} ({f[4]/KH_TRUE*100:.1f} %); d y = {Y6(math.log10(f[2]/D_TRUE)):.1f} ({f[2]/D_TRUE*100:.2f} %)")
for e in (-4, -3, -2, -1, 0, 1):
    print(f"  marca y id 1e{e}: {Y6(e):.1f}")
print("  sesgo por errores en la variable (v con ruido): factor teórico 1/(1 + sigma_v^2/var(v)):")
for vmax in (0.25, 1, 4):
    print(f"    vmax {vmax}: {1/(1+SIG_V**2/(vmax**2/3)):.4f}")

print("-" * 70)
print("   Extrapolación: modelo de caja gris (lineal) frente a caja negra (polinomio de grado 7) ajustados hasta 4 m/s")
# verdad con un término cúbico pequeño (el arrastre parásito del cuerpo crece con v^2)
CQ_BODY = 0.004
verdad = lambda v: -D_TRUE * v - CQ_BODY * v * np.abs(v)
vm = rng.uniform(-4, 4, 3000)
a = verdad(vm) + rng.normal(0, SIG_A, 3000)
dg = -(vm @ a) / (vm @ vm)
Phi_g2 = np.column_stack([vm, vm * np.abs(vm)])
thg2 = np.linalg.lstsq(Phi_g2, a, rcond=None)[0]
pn = np.polyfit(vm, a, 7)
VE = np.linspace(0, 12, 49)
for v in (2, 4, 6, 8, 12):
    print(f"  v = {v:2d} m/s: verdad {verdad(v):7.3f}; gris lineal {-dg*v:7.3f}; gris con |v|v {thg2[0]*v+thg2[1]*v*abs(v):7.3f}; polinomio {np.polyval(pn, v):9.3f} m/s^2")
X5 = lin(0, 12, 60, 520)
Y5 = lin(-2, 8, 250, 20)       # se dibuja la deceleración (-a), positiva hacia arriba
print("path ext verdad:", path(VE, -verdad(VE), X5, Y5))
print("path ext gris:", path(VE, dg * VE, X5, Y5))
pv = np.clip(-np.polyval(pn, VE), -2, 8)
print("path ext negra:", path(VE, pv, X5, Y5))
print(f"  ejes ext: x(4) = {X5(4):.0f}, x(8) = {X5(8):.0f}, x(12) = {X5(12):.0f}; y(0) = {Y5(0):.1f}, y(2) = {Y5(2):.1f}, y(4) = {Y5(4):.1f}, y(6) = {Y5(6):.1f}, y(8) = {Y5(8):.1f}")
