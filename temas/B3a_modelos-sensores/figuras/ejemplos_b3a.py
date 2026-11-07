"""Ejemplos numéricos y datos de las figuras nuevas de B3a (modelos de sensores).

Cada número de un ejemplo «a mano» de la presentación sale de aquí. Las figuras de
resultados ya existentes (Allan de 12 h, deriva, rejilla del obturador, borde de eventos,
series de GNSS) siguen saliendo de figuras_b3a.py, que no hace falta relanzar.
Los parámetros reales son los que ya cita el tema (TUM VI, EuRoC, PX4, EuRoC cam0);
los marcados como SUPUESTO son valores elegidos aquí para ilustrar.
Ejecutar:  python3 ejemplos_b3a.py > salida_ejemplos.txt   (numpy, CPU, segundos)
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)
G = 9.81
rng = np.random.default_rng(1)


def titulo(n, s):
    print("\n" + "=" * 72 + f"\nEjemplo {n}. {s}\n" + "=" * 72)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


# Parámetros reales que ya están en el tema
NG_TUM, KG_TUM = 8.0e-5, 2.2e-6          # giróscopo TUM VI (rad/s/sqrt(Hz), rad/s^2/sqrt(Hz))
NA_TUM, KA_TUM = 1.4e-3, 8.6e-5          # acelerómetro TUM VI (m/s^2/sqrt(Hz), m/s^3/sqrt(Hz))
EUROC = (2.0e-3, 3.0e-3, 1.6968e-4, 1.9393e-5)   # sigma_a, sigma_ba, sigma_g, sigma_bg
FS = 200.0
DT = 1 / FS

# ---------------------------------------------------------------------------
titulo(1, "Cómo se ve cada término del modelo de IMU en la señal de error (figura)")
SESGO0 = 5e-4          # SUPUESTO: sesgo constante de 0,5 mrad/s
ESCALA = 0.01          # SUPUESTO: factor de escala del 1 %
t10 = np.arange(0, 10, DT)
e_sesgo = np.full_like(t10, SESGO0)
e_blanco = NG_TUM / math.sqrt(DT) * rng.standard_normal(t10.size)
th = np.arange(0, 3600, 1.0)
e_paseo = np.cumsum(KG_TUM * math.sqrt(1.0) * rng.standard_normal(th.size))
omega = np.clip((t10 - 3) / 2, 0, 1) * 2.0 - np.clip((t10 - 7) / 1, 0, 1) * 2.0
e_escala = ESCALA * omega
print(f"  sesgo constante supuesto: {SESGO0*1e3:.1f} mrad/s")
print(f"  ruido blanco TUM VI a 200 Hz: sigma por muestra {NG_TUM/math.sqrt(DT)*1e3:.3f} mrad/s; desv. de la serie {e_blanco.std()*1e3:.3f} mrad/s")
print(f"  paseo del sesgo TUM VI en 1 h: desv. teórica K*sqrt(t) = {KG_TUM*60*1e3:.3f} mrad/s; esta realización acaba en {e_paseo[-1]*1e3:+.3f} mrad/s")
print(f"  factor de escala {ESCALA*100:.0f} % con omega = 2 rad/s: error {ESCALA*2*1e3:.0f} mrad/s mientras dura el giro")
X = lin(0, 10, 150, 480)
Xh = lin(0, 3600, 150, 480)
filas = [("sesgo", np.array([0.0, 10.0]), np.array([SESGO0, SESGO0]), 1e-3, 50, X), ("blanco", t10[::4], e_blanco[::4], 4e-3, 120, X),
         ("paseo", th[::20], e_paseo[::20], 4e-4, 190, Xh), ("escala", t10[::10], e_escala[::10], 25e-3, 260, X)]
for nombre, tt, ee, esc, yc, XX in filas:
    Y = lin(-esc, esc, yc + 26, yc - 26)
    print(f"path señal {nombre} (centro y = {yc}, escala ±{esc*1e3:g} mrad/s):", path(tt, ee, XX, Y))

# ---------------------------------------------------------------------------
titulo(2, "De densidad continua a desviación por muestra (TUM VI a 200 Hz) y comprobación")
sgd = NG_TUM / math.sqrt(DT)
sad = NA_TUM / math.sqrt(DT)
print(f"  giróscopo: sigma_g = {NG_TUM:.1e} rad/s/sqrt(Hz), Dt = {DT*1e3:.0f} ms -> sigma_g,d = sigma_g/sqrt(Dt) = {sgd:.3e} rad/s"
      f" (= sigma_g*sqrt(f_s) = {NG_TUM*math.sqrt(FS):.3e})")
print(f"  acelerómetro: sigma_a = {NA_TUM:.1e} -> sigma_a,d = {sad:.4f} m/s^2")
# comprobación: ruido «continuo» aproximado a 200 kHz, promediado en ventanas de 5 ms
FF = 200_000
n_fino = int(FF * 20)          # 20 s
fino = NG_TUM * math.sqrt(FF) * rng.standard_normal(n_fino)   # densidad sigma_g con paso 1/FF
medias = fino.reshape(-1, int(FF * DT)).mean(axis=1)
print(f"  Monte Carlo: ruido de densidad {NG_TUM:.1e} simulado a 200 kHz y promediado en ventanas de 5 ms -> desv. {medias.std():.3e} rad/s (teórica {sgd:.3e})")
sub = fino[::int(FF * DT)]
print(f"  submuestreado sin promediar (una de cada 1000 muestras): desv. {sub.std():.3e} rad/s: no baja, sigue siendo la de 200 kHz")
pb = KG_TUM * math.sqrt(DT)
print(f"  paso del sesgo: sigma_bg*sqrt(Dt) = {pb:.3e} rad/s por muestra; tras 1 h (720 000 pasos): {pb*math.sqrt(720000):.3e} rad/s"
      f" = sigma_bg*sqrt(3600) = {KG_TUM*60:.3e} rad/s = {KG_TUM*60*180/math.pi*3600:.1f} °/h")
arw = NG_TUM * 180 / math.pi * 60
print(f"  angle random walk: sigma_theta(t) = sigma_g*sqrt(t): en 1 h, {NG_TUM*60:.2e} rad = {NG_TUM*60*180/math.pi:.3f}°;"
      f" en unidades de hoja de datos, sigma_g = {arw:.3f} °/sqrt(h)")
print(f"  pregunta 1 del test: 0,01 °/s/sqrt(Hz) a 400 Hz -> {0.01*math.sqrt(400):.2f} °/s")

# ---------------------------------------------------------------------------
titulo(3, "Varianza de Allan a mano: 12 muestras a 1 Hz (mrad/s)")
x = np.array([2, 0, 1, 3, -1, 1, 0, 2, 4, 1, -2, 1], float)
print("  muestras:", x)
for m in (1, 2, 3):
    medias = x[: (len(x) // m) * m].reshape(-1, m).mean(axis=1)
    d = np.diff(medias)
    s2 = 0.5 * np.mean(d ** 2)
    print(f"  tau = {m} s: medias {medias}; diferencias {d}; d^2 {d**2}; sigma^2 = mitad de la media de d^2 = {s2:.4f}; sigma = {math.sqrt(s2):.4f} mrad/s")
# estimador solapado con la integral (theta)
thx = np.concatenate([[0.0], np.cumsum(x)])
for m in (2, 3):
    dd = thx[2 * m:] - 2 * thx[m:-m] + thx[:-2 * m]
    s2 = np.sum(dd ** 2) / (2 * m ** 2 * (len(x) - 2 * m + 1))
    print(f"  solapado, tau = {m} s: theta = {thx}; {len(dd)} términos theta_(k+2m) - 2 theta_(k+m) + theta_k = {dd}; sigma^2 = {s2:.4f}")
print("  (el estimador solapado del tema divide entre 2 tau^2 (N - 2m) con N el número de valores de theta menos 1; aquí N - 2m + 1 términos)")
# figura de cúmulos: 60 muestras, m = 10
y = 1.0 * rng.standard_normal(60) + np.cumsum(0.25 * rng.standard_normal(60))
Xc = lin(0, 60, 40, 540)
Yc = lin(-4, 4, 200, 30)
print("  figura de cúmulos: rango de la señal", y.min().round(2), y.max().round(2))
print("path cúmulos señal:", path(np.arange(60) + 0.5, y, Xc, Yc))
mc = y.reshape(6, 10).mean(axis=1)
for k, mk in enumerate(mc):
    print(f"  cúmulo {k}: media {mk:+.2f} -> línea x {Xc(10*k):.1f}–{Xc(10*k+10):.1f}, y {Yc(mk):.1f}")
print("  diferencias consecutivas:", np.diff(mc).round(2), "; sigma^2 de esta muestra", (0.5 * np.mean(np.diff(mc) ** 2)).round(3))

# ---------------------------------------------------------------------------
titulo(4, "Por qué cada ruido da su pendiente: núcleos en el tiempo e integrales en frecuencia")


def allan_mc(serie, m):
    mm = serie[: (len(serie) // m) * m].reshape(-1, m).mean(axis=1)
    return 0.5 * np.mean(np.diff(mm) ** 2)


Nn, Kk = 1.0, 1.0
for tau in (1, 4, 16):
    m = int(tau * 100)
    blanco = Nn * math.sqrt(100) * rng.standard_normal(8_000_000)
    paseo = np.cumsum(Kk * math.sqrt(0.01) * rng.standard_normal(8_000_000))
    print(f"  tau = {tau:2d} s (N = K = 1, 100 Hz, 8e6 muestras): blanco {allan_mc(blanco, m):.4f} (N^2/tau = {1/tau:.4f}); "
          f"paseo {allan_mc(paseo, m):.3f} (K^2 tau/3 = {tau/3:.3f})")
# núcleo triangular: integral de g^2
tau = 1.0
u = np.linspace(0, 2 * tau, 200001)
g = np.where(u < tau, u, 2 * tau - u)
print(f"  núcleo triangular: integral de g^2 en [0, 2 tau] = {np.trapezoid(g**2, u):.5f} (2 tau^3/3 = {2/3:.5f})")
# integrales de la regla espectral
XMAX = 1000.0
xx = np.linspace(1e-6, XMAX, 10_000_001)
s4 = np.sin(xx) ** 4
# cola analítica: sin^4 promedia 3/8 más allá de XMAX
i2 = np.trapezoid(s4 / xx**2, xx) + 3 / 8 / XMAX
i3 = np.trapezoid(s4 / xx**3, xx) + 3 / 8 / (2 * XMAX**2)
i4 = np.trapezoid(s4 / xx**4, xx)
print(f"  integral de sin^4 x / x^2 = {i2:.5f} (pi/4 = {math.pi/4:.5f})")
print(f"  integral de sin^4 x / x^3 = {i3:.5f} (ln 2 = {math.log(2):.5f})")
print(f"  integral de sin^4 x / x^4 = {i4:.5f} (pi/3 = {math.pi/3:.5f})")
del xx, s4
print(f"  inestabilidad del sesgo: sigma^2 = (2 ln2/pi) B^2 -> sigma = {math.sqrt(2*math.log(2)/math.pi):.4f} B")

# ---------------------------------------------------------------------------
titulo(5, "Ejemplo con TUM VI: las dos rectas, su cruce y lo que se lee en cada tau")
for tv in (0.01, 1, 100):
    print(f"  ruido blanco N/sqrt(tau), tau = {tv:g} s: {NG_TUM/math.sqrt(tv):.2e} rad/s")
for tv in (3, 300, 3000):
    print(f"  paseo K sqrt(tau/3), tau = {tv:g} s: {KG_TUM*math.sqrt(tv/3):.2e} rad/s")
tc = math.sqrt(3) * NG_TUM / KG_TUM
sc = NG_TUM / math.sqrt(tc)
print(f"  cruce: N/sqrt(tau) = K sqrt(tau/3) -> tau* = sqrt(3) N/K = {tc:.1f} s; cada recta vale {sc:.2e}; la suma, sqrt(2)*{sc:.2e} = {math.sqrt(2)*sc:.2e} rad/s")
print(f"  (en la figura de 12 h, con la inestabilidad de sesgo supuesta añadida, el mínimo baja a 1,73e-5 en 44 s)")
for horas, tv in ((111, 1000), (111, 3), (12, 3400), (1, 300)):
    nb = horas * 3600 / tv
    print(f"  {horas} h, tau = {tv} s: {nb:.0f} ventanas, error relativo 1/sqrt(2(n-1)) = {100/math.sqrt(2*(nb-1)):.1f} %")
Xa = lin(-2, 4, 70, 500)
Ya = lin(-6.5, -2.5, 240, 20)
tl = np.logspace(-2, 4, 40)
print("path TUM blanco:", path(np.log10(tl), np.log10(NG_TUM / np.sqrt(tl)), Xa, Ya))
tl2 = np.logspace(-1, 4, 34)
print("path TUM paseo:", path(np.log10(tl2), np.log10(KG_TUM * np.sqrt(tl2 / 3)), Xa, Ya))
print("path TUM suma:", path(np.log10(tl), np.log10(np.sqrt(NG_TUM**2 / tl + KG_TUM**2 * tl / 3)), Xa, Ya))
print(f"  punto N (1 s): ({Xa(0):.1f}, {Ya(math.log10(NG_TUM)):.1f}); punto K (3 s): ({Xa(math.log10(3)):.1f}, {Ya(math.log10(KG_TUM)):.1f}); cruce: ({Xa(math.log10(tc)):.1f}, {Ya(math.log10(sc)):.1f})")
for e_ in range(-2, 5):
    print(f"  marca x 1e{e_}: {Xa(e_):.1f}", end=";")
print()
for e_ in (-6, -5, -4, -3):
    print(f"  marca y 1e{e_}: {Ya(e_):.1f}", end=";")
print()

# ---------------------------------------------------------------------------
titulo(6, "Deriva con errores constantes: t^2 del acelerómetro y t^3 del giróscopo")
ba = KA_TUM * 60       # sesgo que acumula el paseo de TUM VI en 1 h (1 sigma)
bg = KG_TUM * 60
B_inst = 1.73e-5 / 0.664
print(f"  sesgo de acelerómetro tras 1 h de paseo (TUM VI, 1 sigma): {ba*1e3:.2f} mm/s^2; de giróscopo: {bg:.2e} rad/s")
for t in (1, 10, 60):
    print(f"  t = {t:2d} s: acel. b t^2/2 = {0.5*ba*t*t:8.3f} m; giróscopo: dtheta = b t = {bg*t:.2e} rad, g b t^3/6 = {G*bg*t**3/6:8.3f} m;"
          f" con la inestabilidad del ejemplo de Allan (B = {B_inst:.2e} rad/s): {G*B_inst*t**3/6:7.3f} m")

titulo(7, "Deriva con ruidos: integral repetida de ruido blanco (fórmula de Cauchy) y los cuatro términos")
for k in (1, 2, 3, 4):
    c = 1 / (math.factorial(k - 1) ** 2 * (2 * k - 1))
    print(f"  k = {k}: Var = q^2 t^{2*k-1} / ((k-1)!^2 (2k-1)) = q^2 t^{2*k-1} * {c:.6f} (1/{1/c:.0f})")
# Monte Carlo de la doble integral (k = 2)
M, h, T = 4000, 0.01, 10.0
w = rng.standard_normal((M, int(T / h))) / math.sqrt(h)
v = np.cumsum(w * h, axis=1)
p = np.cumsum(v * h, axis=1)
print(f"  Monte Carlo k = 2, q = 1, t = 10 s: desv. {p[:, -1].std():.3f} (sqrt(t^3/3) = {math.sqrt(1000/3):.3f})")
sa, sba, sg, sbg = EUROC
for t in (1, 10, 60):
    pa = sa * t ** 1.5 / math.sqrt(3)
    pba = sba * t ** 2.5 / math.sqrt(20)
    pg = G * sg * t ** 2.5 / math.sqrt(20)
    pbg = G * sbg * t ** 3.5 / math.sqrt(252)
    tot = math.sqrt(pa * pa + pba * pba + pg * pg + pbg * pbg)
    print(f"  EuRoC t = {t:2d} s: acel. blanco {pa:.4f} m; paseo acel. {pba:.4f} m; gir. blanco {pg:.4f} m; paseo gir. {pbg:.4f} m; total (raíz de la suma de cuadrados) {tot:.3f} m")
    print(f"      cuentas: sqrt(t^3) = {t**1.5:.2f}, sqrt(t^5) = {t**2.5:.1f}, sqrt(t^7) = {t**3.5:.1f}")

# ---------------------------------------------------------------------------
titulo(8, "Aliasing de la vibración: frecuencia aparente |f - k f_s|")
f_rot = 770 / (2 * math.pi)
for fs_ in (200.0, 250.0):
    for arm in (1, 2, 3):
        f = arm * f_rot
        k = round(f / fs_)
        print(f"  rotor a 770 rad/s = {f_rot:.2f} Hz, armónico {arm} ({f:.1f} Hz) muestreado a {fs_:.0f} Hz sin antialiasing: aparente {abs(f - k*fs_):.1f} Hz")
f_peor = 201 / 2
print(f"  si el 2.º armónico cae en 201 Hz (rotor a {f_peor:.1f} Hz = {2*math.pi*f_peor:.0f} rad/s) y se muestrea a 200 Hz: aparente 1 Hz, casi un sesgo")
# figura: versión a escala 21 Hz muestreada a 20 Hz -> 1 Hz
Xv = lin(0, 1, 50, 530)
Yv = lin(-1.2, 1.2, 190, 30)
tf = np.linspace(0, 1, 2101)
tf2 = np.linspace(0, 1, 841)
print("path alias continua 21 Hz:", path(tf2, np.sin(2 * math.pi * 21 * tf2), Xv, Yv))
ts = np.arange(0, 1.0001, 1 / 20)
print("  muestras a 20 Hz:", " ".join(f"({Xv(t_):.1f},{Yv(math.sin(2*math.pi*21*t_)):.1f})" for t_ in ts))
print("path alias aparente 1 Hz:", path(tf[::20], np.sin(2 * math.pi * 1 * tf[::20]), Xv, Yv))

# ---------------------------------------------------------------------------
titulo(9, "Cámara: modelo lineal de EMVA 1288 con números (parámetros SUPUESTOS)")
eta, Kg, sd, mud = 0.5, 0.1, 5.0, 0.0
sq2 = 1 / 12
for mup in (1000, 100, 20, 9000):
    mue = eta * mup
    muy = Kg * (mue + mud)
    s2y = Kg ** 2 * sd ** 2 + sq2 + Kg * (muy - Kg * mud)
    snr = mue / math.sqrt(sd ** 2 + sq2 / Kg ** 2 + mue)
    print(f"  mu_p = {mup:5d} fotones: {mue:6.0f} e-; mu_y = {muy:6.1f} DN; sigma_y^2 = {Kg**2*sd**2:.3f} + {sq2:.4f} + {Kg*muy:.2f} = {s2y:.3f} DN^2;"
          f" SNR = {snr:5.2f} (en DN: {muy/math.sqrt(s2y):5.2f}); solo disparo sqrt(eta mu_p) = {math.sqrt(mue):.2f}")
print("  SNR 10 -> 30 con disparo dominante: (30/10)^2 = 9 veces más luz")
F_PX = 458.65
for w_, te in ((2.0, 0.005), (2.0, 0.005 * 9), (1.0, 0.005)):
    print(f"  desenfoque f omega t_exp con omega = {w_} rad/s, t_exp = {te*1e3:.0f} ms: {F_PX*w_*te:.1f} px")

titulo(10, "Obturador rodante: tiempo de cada fila y desplazamiento (EuRoC cam0, f = 458,65 px, H = 480)")
H, Wd = 480, 752
for w_, tro in ((5.0, 0.030), (2.0, 0.020)):
    dt_fila = tro / H
    print(f"  omega = {w_} rad/s, t_ro = {tro*1e3:.0f} ms: cada fila empieza {dt_fila*1e6:.1f} µs después; en ese tiempo la cámara gira {w_*dt_fila*1e3:.4f} mrad"
          f" -> {F_PX*w_*dt_fila:.3f} px por fila; fila 240: {F_PX*w_*tro*240/H:.1f} px; fila 480: {F_PX*w_*tro:.1f} px")
fac = 1 + (376 / F_PX) ** 2
print(f"  lejos del centro: du/dalpha = f (1 + (u/f)^2); a 376 px del centro, factor {fac:.2f} -> {F_PX*5*0.03*fac:.0f} px en vez de {F_PX*5*0.03:.0f}")
print(f"  ángulo de giro durante la lectura: {5*0.03:.2f} rad = {math.degrees(5*0.03):.1f}°")
# figura de la línea temporal de exposición: 8 filas
Xt = lin(0, 40, 120, 520)
for i, v_ in enumerate(np.linspace(0, H, 7)):
    t0 = 30 * v_ / H
    print(f"  fila {v_:.0f}: exposición de {t0:.1f} a {t0+5:.1f} ms -> x {Xt(t0):.1f} a {Xt(t0+5):.1f}")
print(f"  marcas x: 0 ms {Xt(0):.1f}, 10 {Xt(10):.1f}, 20 {Xt(20):.1f}, 30 {Xt(30):.1f}, 40 {Xt(40):.1f}")

# ---------------------------------------------------------------------------
titulo(11, "Cámara de eventos: cuántos eventos, a qué ritmo y el paso de ESIM")
C = 0.2
I1, I2 = 0.2, 1.0
print(f"  contraste ln(I2/I1) = ln({I2/I1:.0f}) = {math.log(I2/I1):.3f}; entre C = {C}: {math.log(I2/I1)/C:.2f} -> {int(math.log(I2/I1)//C)} eventos")
for esc in (1.0, 0.01):
    print(f"  con la luz multiplicada por {esc}: ln({I2*esc:g}/{I1*esc:g}) = {math.log(I2/I1):.3f}: mismos eventos (el logaritmo mide contraste relativo)")
# borde de la figura (sigmoide de 0,5 px) y su gradiente de log-intensidad
xpx = np.linspace(-6, 6, 120001)
Lx = np.log(I1 + (I2 - I1) / (1 + np.exp(-xpx / 0.5)))
gL = np.gradient(Lx, xpx)
print(f"  gradiente máximo de ln I a lo largo del borde: {gL.max():.3f} por px (en x = {xpx[gL.argmax()]:+.2f} px)")
for vv in (100, 1000, 10000):
    rate = gL.max() * vv / C
    print(f"  v = {vv:5d} px/s: |dL/dt| máx = {gL.max()*vv:8.1f} /s -> ritmo máximo {rate:9.0f} eventos/s, intervalo mínimo {1e6/rate:8.1f} µs")
# extensión de los 8 eventos (desde ln 0,2)
niveles = math.log(I1) + C * np.arange(1, 9)
xs_ev = [xpx[np.argmax(Lx >= lv)] for lv in niveles]
print(f"  posiciones de los 8 eventos a lo largo del borde (px): {np.round(xs_ev, 2)}; extensión {xs_ev[-1]-xs_ev[0]:.2f} px -> a 1000 px/s, {xs_ev[-1]-xs_ev[0]:.2f} ms")
print("  en la figura del tema la ventana empieza en -2 ms, con el borde ya iniciado (ln I = ln 0,214): se ven 7 eventos")
lam = 0.5
for vv in (1000, 10000):
    dts = lam * C / (gL.max() * vv)
    print(f"  ESIM con lambda = {lam} (elegido aquí), v = {vv} px/s: paso de render lambda C / |dL/dt|max = {dts*1e6:.1f} µs -> {1/dts:.0f} imágenes/s")

# ---------------------------------------------------------------------------
titulo(12, "LiDAR: separación entre rayos, probabilidad de tocar un objeto fino y distorsión por movimiento")
for dth in (0.1, 0.2, 0.35):
    for r in (5, 9.8, 10, 20):
        sep = r * math.radians(dth)
        n = 0.05 / sep
        print(f"  {dth}° a {r:4.1f} m: separación r*dtheta = {sep*100:5.2f} cm; poste de 5 cm: n_h = {n:.2f}; P(al menos un rayo) = {min(1,n):.2f}")
p1 = 0.05 / (9.8 * math.radians(0.35))
print(f"  0,35° a 9,8 m: P = {p1:.3f}; si la fase cambia al azar entre barridos, P(ninguno en 3 barridos) = {(1-p1)**3:.4f}")
print(f"  cable de 1 cm con 0,2°: r = w/dtheta = {0.01/math.radians(0.2):.2f} m")
for v_, w_, Tb in ((10, 0, 0.1), (0, 1.0, 0.1), (2, 0, 0.1)):
    print(f"  barrido de {Tb*1e3:.0f} ms (10 Hz), v = {v_} m/s, giro {w_} rad/s: traslación {v_*Tb:.2f} m; un punto a 10 m se desplaza por el giro {10*w_*Tb:.2f} m")
# figura de rayos: sensor en (60,150), separación 4° (exagerada)
cx, cy = 60, 150
for ang in np.arange(-16, 16.1, 4):
    a = math.radians(ang)
    print(f"  rayo {ang:+.0f}°: hasta ({cx+420*math.cos(a):.1f}, {cy-420*math.sin(a):.1f}); a 300 px: ({cx+300*math.cos(a):.1f}, {cy-300*math.sin(a):.1f})")
print(f"  separación entre rayos a 300 px con 4°: {300*math.radians(4):.1f} px")

# figura de distorsión: pared a 10 m delante, el dron avanza hacia ella a 10 m/s; el barrido va de -40° a +40° en 100 ms
print("  distorsión: pared a 10 m; dron hacia la pared a 10 m/s; el sector de -40° a +40° se barre en 100 ms")
Xw = lin(-9, 9, 40, 520)
Yw = lin(0, 11, 250, 30)
pts = []
for ang in np.linspace(-40, 40, 17):
    ti = 0.1 * (ang + 40) / 80
    ys = 10 * ti                       # posición del dron en el instante del punto
    r_ = (10 - ys) / math.cos(math.radians(ang))
    px_, py_ = r_ * math.sin(math.radians(ang)), r_ * math.cos(math.radians(ang))   # como si el dron no se moviera
    pts.append((px_, py_))
    if ang in (-40, 0, 40):
        print(f"    ángulo {ang:+.0f}°: t = {ti*1e3:.0f} ms, dron en y = {ys:.2f} m, distancia medida {r_:.2f} m -> punto registrado ({px_:+.2f}, {py_:.2f}) m")
print("path pared registrada:", path([p[0] for p in pts], [p[1] for p in pts], Xw, Yw))
print(f"  pared verdadera y = 10: y_svg {Yw(10):.1f}; y = 9: {Yw(9):.1f}; dron en y=0: {Yw(0):.1f}, y=1: {Yw(1):.1f}; x = -8.39: {Xw(-8.39):.1f}, x = 0: {Xw(0):.1f}, x = 8.39: {Xw(8.39):.1f}")
for ang in (-40, 0, 40):
    ti = 0.1 * (ang + 40) / 80
    ys = 10 * ti
    r_ = (10 - ys) / math.cos(math.radians(ang))
    print(f"    rayo {ang:+.0f}° desde ({Xw(0):.1f},{Yw(ys):.1f}) hasta ({Xw(r_*math.sin(math.radians(ang))):.1f},{Yw(10):.1f})")

# ---------------------------------------------------------------------------
titulo(13, "GNSS: Gauss-Markov de primer orden, de continuo a discreto, varianza estacionaria y consecuencias")
for nombre, a_, q_ in (("PX4 sensor_gps_sim", 0.76, 0.016), ):
    s = q_ / math.sqrt(1 - a_ ** 2)
    print(f"  {nombre}: a = {a_}, q = {q_} m: sigma = q/sqrt(1-a^2) = {q_}/{math.sqrt(1-a_**2):.4f} = {s*100:.2f} cm; tau = -Dt/ln a = {-0.125/math.log(a_):.3f} s")
sig, tau = 1.0, 60.0
a_ = math.exp(-0.125 / tau)
q_ = sig * math.sqrt(1 - a_ ** 2)
print(f"  ejemplo 1 m y 60 s a 8 Hz: a = exp(-0,125/60) = {a_:.6f}; q = sigma sqrt(1-a^2) = {q_:.4f} m")
for lag in (0.125, 10, 60, 120):
    print(f"  autocorrelación a {lag:6.3f} s: exp(-lag/tau) = {math.exp(-lag/tau):.3f}; PX4 (0,46 s): {math.exp(-lag/0.4555):.3g}")
n = 480
idx = np.arange(n)
Rm = a_ ** np.abs(idx[:, None] - idx[None, :])
print(f"  media de {n} muestras (60 s): blanco 1 m -> {1/math.sqrt(n):.3f} m; Gauss-Markov -> sqrt(suma de a^|i-j|)/n = {math.sqrt(Rm.sum())/n:.3f} m")
print(f"  diferencia de dos muestras: Var = 2 sigma^2 (1 - a); velocidad a 8 Hz: GM {math.sqrt(2*(1-a_))*8:.3f} m/s; blanco sqrt(2)*8 = {math.sqrt(2)*8:.2f} m/s")
Xg = lin(0, 120, 70, 500)
Yg = lin(0, 1, 200, 30)
lags = np.linspace(0, 120, 121)
print("path autocorr GM 60 s:", path(lags, np.exp(-lags / 60), Xg, Yg))
lags2 = np.linspace(0, 3, 31)
print("path autocorr PX4:", path(lags2, np.exp(-lags2 / 0.4555), Xg, Yg))
print(f"  marcas x: 0 {Xg(0):.1f}, 30 {Xg(30):.1f}, 60 {Xg(60):.1f}, 90 {Xg(90):.1f}, 120 {Xg(120):.1f}; y 1 {Yg(1):.1f}, 0,5 {Yg(0.5):.1f}, e^-1 {Yg(math.exp(-1)):.1f}, 0 {Yg(0):.1f}")

# ---------------------------------------------------------------------------
titulo(14, "Barómetro: equilibrio hidrostático")
rho = 1.225
for dp in (1, 3, 100):
    print(f"  dp = {dp:3d} Pa -> dh = dp/(rho g) = {dp/(rho*G):.3f} m (rho = {rho} kg/m^3, g = {G})")
print(f"  rho g = {rho*G:.2f} Pa/m")

titulo(15, "Magnetómetro: rumbo, error por una perturbación perpendicular, hierro duro y blando")
Bx, By, Bz = 6e-6, 2.3e-5, -4.2e-5
Bh = math.hypot(Bx, By)
print(f"  mundo de PX4: B = ({Bx*1e6:.0f}, {By*1e6:.0f}, {Bz*1e6:.0f}) µT; |B| = {math.sqrt(Bx*Bx+By*By+Bz*Bz)*1e6:.1f} µT; horizontal B_h = {Bh*1e6:.2f} µT;"
      f" inclinación {math.degrees(math.atan2(-Bz, Bh)):.1f}°")
for db in (0.3e-6, 1e-6, 5e-6):
    print(f"  perturbación perpendicular de {db*1e6:.1f} µT: dpsi = atan(dB/B_h) = {math.degrees(math.atan(db/Bh)):.2f}° (lineal {math.degrees(db/Bh):.2f}°)")
bhi = 5e-6       # SUPUESTO
print(f"  hierro duro SUPUESTO de {bhi*1e6:.0f} µT fijo en el cuerpo: error de rumbo máximo asin(b/B_h) = {math.degrees(math.asin(bhi/Bh)):.1f}°, cambia con el rumbo")
psis = np.linspace(0, 2 * math.pi, 361)
err = []
for ps in psis:
    mb = np.array([Bh * math.cos(ps), -Bh * math.sin(ps)]) + np.array([bhi, 0])
    err.append(math.degrees(math.atan2(-mb[1], mb[0]) - ps))
err = (np.array(err) + 180) % 360 - 180
print(f"  error de rumbo con ese hierro duro al dar una vuelta: de {err.min():.1f}° a {err.max():.1f}°")
Xm = lin(-40, 40, 40, 280)
Ym = lin(-40, 40, 260, 20)
cu = Bh * 1e6 * np.cos(psis)
cv = Bh * 1e6 * np.sin(psis)
print("path mag círculo verdadero:", path(cu[::6], cv[::6], Xm, Ym))
print("path mag hierro duro:", path(cu[::6] + bhi * 1e6, cv[::6] + 0.3 * bhi * 1e6, Xm, Ym))
print("path mag hierro blando:", path(1.25 * cu[::6], 0.8 * cv[::6], Xm, Ym))
print(f"  centro hierro duro: ({Xm(bhi*1e6):.1f}, {Ym(0.3*bhi*1e6):.1f}); origen ({Xm(0):.1f}, {Ym(0):.1f}); radio {Xm(Bh*1e6)-Xm(0):.1f} px")
