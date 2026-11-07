"""Datos de las figuras y tablas del tema B3a (modelos de sensores).

Uso:  python3 figuras_b3a.py > salida.txt      (numpy; un par de minutos)

Parámetros de IMU de referencia:
  - TUM VI (Bosch BMI160), valores de Allan publicados en la página del dataset (secuencia imu-static, 111 h):
    acelerómetro 0,0014 m/s^2/sqrt(Hz) y 0,000086 m/s^3/sqrt(Hz); giróscopo 0,000080 rad/s/sqrt(Hz) y 0,0000022 rad/s^2/sqrt(Hz).
  - EuRoC (ADIS16448), configuración de OpenVINS/Kalibr: 2,0e-3; 3,0e-3; 1,6968e-4; 1,9393e-5.
  - x500 de Gazebo (PX4-gazebo-models): ruido blanco por muestra de 0,0008726646 rad/s y 0,00637 m/s^2 a 250 Hz, sin sesgo.
"""
import math
import numpy as np

G = 9.81


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def adev(x, fs, taus):
    """Desviación de Allan solapada a partir de la integral (ángulo o velocidad)."""
    th = np.concatenate([[0.0], np.cumsum(x) / fs])
    N = len(x)
    out = []
    for tau in taus:
        m = int(round(tau * fs))
        if m < 1 or 2 * m >= N:
            out.append(np.nan)
            continue
        d = th[2 * m:] - 2 * th[m:-m] + th[:-2 * m]
        out.append(math.sqrt(np.sum(d * d) / (2 * (m / fs) ** 2 * (N - 2 * m))))
    return np.array(out)


# ---------------------------------------------------------------- 1. Allan
print("=" * 70)
print("1. DESVIACIÓN DE ALLAN DE UN GIRÓSCOPO SIMULADO (parámetros de TUM VI) Y DEL x500 DE GAZEBO")
FS = 200.0
HORAS = 12.0
N = int(HORAS * 3600 * FS)
rng = np.random.default_rng(0)
NG, KG = 8.0e-5, 2.2e-6           # TUM VI, giróscopo
BI_G, TC_G = 3.0e-5, 300.0        # SUPUESTO: inestabilidad de sesgo como Gauss-Markov (rad/s, s)
dt = 1 / FS
w_white = NG / math.sqrt(dt) * rng.standard_normal(N)
rw = np.cumsum(KG * math.sqrt(dt) * rng.standard_normal(N))
a = math.exp(-dt / TC_G)
gm = np.empty(N)
e = BI_G * math.sqrt(1 - a * a) * rng.standard_normal(N)
gm[0] = 0.0
for k in range(1, N):
    gm[k] = a * gm[k - 1] + e[k]
gyro = w_white + rw + gm
taus = np.logspace(-2, math.log10(HORAS * 3600 / 9), 40)
ad = adev(gyro, FS, taus)
ad_w = adev(w_white, FS, taus)
# Gazebo x500: 0,0008726646 rad/s por muestra a 250 Hz, sin sesgo
FSZ = 250.0
NZ = int(HORAS * 3600 * FSZ)
gz = 0.0008726646 * rng.standard_normal(NZ)
ad_gz = adev(gz, FSZ, taus)
nd_gz = 0.0008726646 / math.sqrt(FSZ)
print(f"  Gazebo x500: densidad equivalente sigma/sqrt(f) = {nd_gz:.2e} rad/s/sqrt(Hz) ({nd_gz*180/math.pi*3600*60:.3f} °/sqrt(h))")
# lectura de los parámetros
i1 = np.argmin(abs(taus - 1))
print(f"  AD(1 s) simulada = {np.interp(0, np.log10(taus), ad):.3e}; N verdadera = {NG:.1e} rad/s/sqrt(Hz)")
imin = np.nanargmin(ad)
print(f"  mínimo de la AD: {ad[imin]:.3e} rad/s en tau = {taus[imin]:.0f} s; inestabilidad de sesgo B = mín/0,664 = {ad[imin]/0.664:.2e} rad/s "
      f"({ad[imin]/0.664*180/math.pi*3600:.2f} °/h)")
# ajuste de la recta +1/2 en las tau más largas
sel = (taus > 2000) & ~np.isnan(ad)
K_est = np.exp(np.mean(np.log(ad[sel]) - 0.5 * np.log(taus[sel]) - 0.5 * math.log(1 / 3)))
print(f"  K por la recta +1/2 leída en tau = 3 s: {K_est:.2e} (verdadera {KG:.1e}); tau máx. = {taus[-1]:.0f} s")
for tau in (0.01, 0.1, 1, 10, 100, 1000, 4000):
    j = np.argmin(abs(taus - tau))
    n_bloques = HORAS * 3600 / taus[j]
    print(f"  tau {taus[j]:8.2f} s: AD {ad[j]:.3e}; ruido blanco solo {ad_w[j]:.3e}; Gazebo {ad_gz[j]:.3e}; error relativo de la estimación ≈ {100/math.sqrt(2*(n_bloques-1)):.1f} %")
X1 = lin(-2, 3.8, 60, 520)
Y1 = lin(-7, -2.5, 250, 20)
ok = ~np.isnan(ad)
print("path allan sim:", path(np.log10(taus[ok]), np.log10(ad[ok]), X1, Y1))
print("path allan gz:", path(np.log10(taus[ok]), np.log10(ad_gz[ok]), X1, Y1))
tl = np.logspace(-2, 3.8, 30)
print("path allan blanco:", path(np.log10(tl), np.log10(NG / np.sqrt(tl)), X1, Y1))
tr = np.logspace(1.5, 3.8, 10)
print("path allan rw:", path(np.log10(tr), np.log10(KG * np.sqrt(tr / 3)), X1, Y1))
print(f"  punto N: ({X1(0):.1f}, {Y1(math.log10(NG)):.1f}); punto K: ({X1(math.log10(3)):.1f}, {Y1(math.log10(KG)):.1f}); "
      f"mínimo ({X1(math.log10(taus[imin])):.1f}, {Y1(math.log10(ad[imin])):.1f})")
for e_ in (-2, -1, 0, 1, 2, 3):
    print(f"  marca x 1e{e_} s: {X1(e_):.1f}")
print(f"  marca x 1 h: {X1(math.log10(3600)):.1f}")
for e_ in (-7, -6, -5, -4, -3):
    print(f"  marca y 1e{e_}: {Y1(e_):.1f}")

# ---------------------------------------------------------------- 2. navegación a ciegas
print("=" * 70)
print("2. DERIVA DE LA NAVEGACIÓN INERCIAL PURA (IMU en reposo, sin corrección)")


def deriva(t, Na, Ka, Ng, Kg):
    """Desviaciones típicas de posición por término (un eje horizontal)."""
    pa = Na * t ** 1.5 / math.sqrt(3)            # ruido blanco del acelerómetro
    pka = Ka * t ** 2.5 / math.sqrt(20)          # paseo del sesgo del acelerómetro
    pg = G * Ng * t ** 2.5 / math.sqrt(20)       # ruido blanco del giróscopo -> inclinación -> g
    pkg = G * Kg * t ** 3.5 / math.sqrt(252)     # paseo del sesgo del giróscopo
    return pa, pka, pg, pkg


IMUS = {"TUM VI (BMI160)": (0.0014, 0.000086, 0.000080, 0.0000022),
        "EuRoC (ADIS16448)": (2.0e-3, 3.0e-3, 1.6968e-4, 1.9393e-5)}
for nombre, (Na, Ka, Ng, Kg) in IMUS.items():
    for t in (1, 10, 60):
        pa, pka, pg, pkg = deriva(t, Na, Ka, Ng, Kg)
        tot = math.sqrt(pa ** 2 + pka ** 2 + pg ** 2 + pkg ** 2)
        print(f"  {nombre:18s} t = {t:3d} s: total {tot:9.3f} m | acel. blanco {pa:.3g}, sesgo acel. {pka:.3g}, "
              f"gir. blanco {pg:.3g}, sesgo gir. {pkg:.3g}")
# comprobación Monte Carlo de la fórmula del giróscopo blanco (la más sutil): theta = int n, a = g theta, p = int int a
Ng = 1.6968e-4
T, h, M = 10.0, 0.005, 2000
nst = int(T / h)
th = np.cumsum(Ng / math.sqrt(h) * rng.standard_normal((M, nst)) * h, axis=1)
v = np.cumsum(G * th * h, axis=1)
p = np.cumsum(v * h, axis=1)
print(f"  Monte Carlo (2000 ensayos, EuRoC, solo giróscopo blanco) a 10 s: {p[:, -1].std():.4f} m; fórmula g N t^2,5/sqrt(20) = {G*Ng*10**2.5/math.sqrt(20):.4f} m")
X2 = lin(0, 2, 60, 520)          # log10(t)
Y2 = lin(-4, 3, 250, 20)         # log10(m)
tt = np.logspace(0, 2, 40)
Na, Ka, Ng, Kg = IMUS["EuRoC (ADIS16448)"]
terms = deriva(tt, Na, Ka, Ng, Kg)
for nombre, arr in zip(["acc blanco", "acc sesgo", "gir blanco", "gir sesgo"], terms):
    print(f"path deriva {nombre}:", path(np.log10(tt), np.log10(arr), X2, Y2))
for e_ in (-4, -2, 0, 2):
    print(f"  marca y 1e{e_} m: {Y2(e_):.1f}")
for tv in (1, 10, 100):
    print(f"  marca x {tv} s: {X2(math.log10(tv)):.1f}")

# ---------------------------------------------------------------- 3. cámara: obturador rodante y desenfoque
print("=" * 70)
print("3. CÁMARA: OBTURADOR RODANTE Y DESENFOQUE POR MOVIMIENTO")
F_PX = 458.65    # EuRoC cam0 (A4b)
W, H = 752, 480
for w_rad, t_ro, t_exp in [(1.0, 0.020, 0.005), (2.0, 0.020, 0.005), (5.0, 0.030, 0.002), (5.0, 0.0, 0.002)]:
    print(f"  omega = {w_rad} rad/s, lectura {t_ro*1e3:.0f} ms, exposición {t_exp*1e3:.0f} ms: "
          f"cizalla entre primera y última fila f*omega*t_ro = {F_PX*w_rad*t_ro:5.1f} px; desenfoque f*omega*t_exp = {F_PX*w_rad*t_exp:4.1f} px")
# rejilla deformada por el obturador rodante (giro en guiñada a 5 rad/s, lectura de 30 ms)
w_rad, t_ro = 5.0, 0.030
Xc = lin(0, W, 20, 300)
Yc = lin(0, H, 20, 199)
lineas = []
for u0 in np.linspace(0, W, 9):
    vs = np.linspace(0, H, 25)
    us = u0 - F_PX * w_rad * t_ro * (vs / H)   # cada fila se expone t_ro*v/H más tarde
    lineas.append(path(us, vs, Xc, Yc))
for i, l in enumerate(lineas):
    print(f"path rs v{i}:", l)
for i, v0 in enumerate(np.linspace(0, H, 7)):
    us = np.linspace(0, W, 2)
    sh = F_PX * w_rad * t_ro * (v0 / H)
    print(f"path rs h{i}:", path(us - sh, [v0, v0], Xc, Yc))
print(f"  desplazamiento de la última fila: {F_PX*w_rad*t_ro:.1f} px = {F_PX*w_rad*t_ro/W*100:.1f} % del ancho")

# ---------------------------------------------------------------- 4. cámara de eventos
print("=" * 70)
print("4. CÁMARA DE EVENTOS: UN BORDE QUE CRUZA UN PÍXEL")
C = 0.2          # umbral de contraste en log-intensidad (SUPUESTO, típico)
I_osc, I_cla = 0.2, 1.0
salto = math.log(I_cla / I_osc)
print(f"  contraste del borde: ln({I_cla}/{I_osc}) = {salto:.3f} -> {int(salto // C)} eventos ON por píxel con C = {C}")
# borde suave (ancho 2 px) moviéndose a v px/s; píxel en x = 0
for v in (100, 1000, 10000):
    t = np.linspace(-0.02 * 1000 / v, 0.02 * 1000 / v, 20001)
    x = v * t
    L = np.log(I_osc + (I_cla - I_osc) / (1 + np.exp(-x / 0.5)))
    ref = L[0]
    ev = []
    for k in range(len(t)):
        while L[k] - ref >= C:
            ref += C
            ev.append(t[k])
    dur = (ev[-1] - ev[0]) * 1e6 if len(ev) > 1 else 0
    print(f"  v = {v:5d} px/s: {len(ev)} eventos en {dur:7.1f} µs (una cámara de 30 Hz vería el borde con {v/30:.0f} px de desplazamiento entre imágenes)")
X4 = lin(-2, 2.5, 60, 520)
Y4 = lin(math.log(0.15), math.log(1.1), 200, 30)
v = 1000
t = np.linspace(-0.002, 0.0025, 901)
L = np.log(I_osc + (I_cla - I_osc) / (1 + np.exp(-(v * t) / 0.5)))
print("path ev logI:", path(t * 1e3, L, X4, Y4))
ref = L[0]
evt = []
for k in range(len(t)):
    while L[k] - ref >= C:
        ref += C
        evt.append((t[k], ref))
for i, (te, le) in enumerate(evt):
    print(f"  evento {i}: t = {te*1e3:+.3f} ms -> x = {X4(te*1e3):.1f}; nivel y = {Y4(le):.1f}")
for tv in (-2, -1, 0, 1, 2):
    print(f"  marca x {tv} ms: {X4(tv):.1f}")
for lv in (math.log(0.2), math.log(1.0)):
    print(f"  nivel ln I = {lv:.2f}: y = {Y4(lv):.1f}")

# ---------------------------------------------------------------- 5. LiDAR: puntos sobre un obstáculo fino
print("=" * 70)
print("5. LiDAR: PUNTOS SOBRE UN POSTE DE 5 cm Y UN ÁRBOL DE 30 cm")
for nombre, dh, dv in [("0,2° x 2°", 0.2, 2.0), ("0,35° x 0,35°", 0.35, 0.35), ("0,1° x 1°", 0.1, 1.0)]:
    filas = []
    for r in (2, 5, 10, 20):
        nh_poste = 0.05 / (r * math.radians(dh))
        nh_arbol = 0.30 / (r * math.radians(dh))
        nv = 1.0 / (r * math.radians(dv))     # 1 m de altura
        filas.append(f"r = {r:2d} m: poste {nh_poste:4.2f} col., árbol {nh_arbol:5.1f} col.; filas en 1 m {nv:5.1f}")
    print(f"  resolución {nombre}:")
    for f_ in filas:
        print("    " + f_)
print("  distancia a la que un poste de 5 cm deja de recibir con seguridad un rayo por barrido (col. < 1):")
for dh in (0.1, 0.2, 0.35):
    print(f"    {dh}°: {0.05/math.radians(dh):.1f} m")

# ---------------------------------------------------------------- 6. GNSS
print("=" * 70)
print("6. GNSS: EL RUIDO SIMULADO DE PX4 FRENTE A LO QUE DECLARA")
phi, q, A = 0.76, 0.02, 0.8
sig_inn = q * A
sig_st = sig_inn / math.sqrt(1 - phi ** 2)
tau_gm = -0.125 / math.log(phi)
print(f"  sensor_gps_sim (8 Hz): n_k = 0,76 n_(k-1) + 0,02*0,8*w -> innovación {sig_inn:.3f} m; estacionaria {sig_st*100:.1f} cm (vertical x1,5: {1.5*sig_st*100:.1f} cm)")
print(f"  constante de tiempo equivalente: -Dt/ln(0,76) = {tau_gm:.2f} s; eph declarado 0,9 m (sin RTK), epv 1,78 m")
phi2 = 0.76 - 0.02
print(f"  gz_bridge: n_k = (0,76 - 0,02) n_(k-1) + ... -> estacionaria {sig_inn/math.sqrt(1-phi2**2)*100:.1f} cm (con su propia frecuencia de actualización)")
phi_v, qv, Av = 0.54, 0.4, 0.05
print(f"  velocidad: estacionaria {qv*Av/math.sqrt(1-phi_v**2)*100:.1f} cm/s; speed_accuracy declarado 0,4 m/s")
# un Gauss-Markov realista de ejemplo (SUPUESTO): sigma 1 m, tau 60 s; serie de 10 min a 8 Hz
rng = np.random.default_rng(3)
n = int(600 * 8)
def gmarkov(sig, tau, dt, n):
    a = math.exp(-dt / tau)
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + sig * math.sqrt(1 - a * a) * rng.standard_normal()
    return x
px4 = np.zeros(n)
for k in range(1, n):
    px4[k] = phi * px4[k - 1] + sig_inn * rng.standard_normal()
real = gmarkov(1.0, 60.0, 0.125, n)
blanco = rng.standard_normal(n) * 1.0
print(f"  series de 10 min: PX4 sim desv. {px4.std()*100:.1f} cm; Gauss-Markov 1 m/60 s desv. {real.std():.2f} m; blanco 1 m desv. {blanco.std():.2f} m")
# error de velocidad que induce cada modelo si se deriva la posición (dif. finita)
for nombre, s in (("PX4", px4), ("GM", real), ("blanco", blanco)):
    print(f"    derivada por diferencias a 8 Hz ({nombre}): desv. {np.std(np.diff(s)*8):.2f} m/s")
X6 = lin(0, 600, 60, 520)
Y6 = lin(-3.5, 3.5, 250, 20)
tt = np.arange(n) / 8
sel = slice(0, None, 8)
print("path gnss px4:", path(tt[sel], px4[sel], X6, Y6))
print("path gnss gm:", path(tt[sel], real[sel], X6, Y6))
print("path gnss blanco:", path(tt[::16], blanco[::16], X6, Y6))
for yv in (-3, -2, -1, 0, 1, 2, 3):
    print(f"  marca y {yv} m: {Y6(yv):.1f}")

# ---------------------------------------------------------------- 7. barómetro y magnetómetro
print("=" * 70)
print("7. BARÓMETRO Y MAGNETÓMETRO EN LOS SIMULADORES")
rho = 1.225
print(f"  1 Pa de ruido = {1/(rho*G)*100:.1f} cm de altura (dp = -rho g dh); 3 Pa (x500 de Gazebo) = {3/(rho*G)*100:.1f} cm")
B_tot = math.sqrt(6e-6 ** 2 + 2.3e-5 ** 2 + 4.2e-5 ** 2)
print(f"  campo del mundo de PX4 (6e-6, 2,3e-5, -4,2e-5) T: |B| = {B_tot*1e6:.1f} µT; ruido 3e-7 T = {3e-7/B_tot*100:.2f} % -> {math.degrees(3e-7/math.hypot(6e-6,2.3e-5)):.2f}° de rumbo por muestra")
