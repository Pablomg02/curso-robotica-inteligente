"""Ejemplos numéricos de M0 (mapa del campo).

Todo en CPU con numpy, en segundos. Imprime cada ejemplo con su número;
la salida se guarda en figuras/salida_ejemplos.txt. Las cifras de sistemas
reales (250 ms de Gemini Robotics, 40 ms de Swift, 63,9 ms y 16 acciones de
GR00T N1...) son las que ya cita la presentación; aquí solo se hacen cuentas.

Uso:  python3 ejemplos_m0.py > salida_ejemplos.txt
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def cab(n, t):
    print(f"\n=== Ejemplo {n} · {t} ===")


# ---------------------------------------------------------------- 1
cab(1, "Retardo máximo tau_max = PM/(2 pi f_c) con PM = 45°")
PM = np.deg2rad(45.0)
print(f"PM = 45° = {PM:.4f} rad ;  PM/(2 pi) = {PM/(2*np.pi):.4f}  -> tau_max ~ 0,125/f_c")
for nombre, fc in [("Velocidad angular", 30.0), ("Actitud", 10.0),
                   ("Posición", 1.0), ("Misión", 0.1)]:
    tmax = PM / (2*np.pi*fc)
    print(f"{nombre:18s} f_c = {fc:5.1f} Hz  ->  tau_max = {tmax*1e3:8.2f} ms")

# ---------------------------------------------------------------- 2
cab(2, "Fase que pierde un lazo: omega_c * tau")
for caso, fc, tau in [("Swift 40 ms en un lazo de actitud a 10 Hz", 10.0, 0.040),
                      ("Pregunta 3: cruce a 5 Hz, 20 ms", 5.0, 0.020),
                      ("Bode de la hoja: cruce a 10 Hz, 12,5 ms", 10.0, 0.0125)]:
    perd = 2*np.pi*fc*tau
    print(f"{caso:45s} omega_c = {2*np.pi*fc:7.3f} rad/s, "
          f"omega_c*tau = {perd:.4f} rad = {np.rad2deg(perd):6.1f}°")
tau = 0.250
print(f"Gemini Robotics 250 ms: cruce máximo con PM 45° = 0,125/tau = {0.125/tau:.3f} Hz")

# ---------------------------------------------------------------- 3
cab(3, "Bode de L(s) = K / (s (s/p + 1)), cruce a 10 Hz con PM = 45°")
fc = 10.0
wc = 2*np.pi*fc
p = wc                      # polo en el cruce: aporta -45° allí
K = np.sqrt(2)*wc           # |L(j wc)| = K/(wc*sqrt(2)) = 1
print(f"p = {p:.3f} rad/s, K = {K:.3f} rad/s")


def L(f, tau=0.0):
    w = 2*np.pi*f
    s = 1j*w
    return K/(s*(s/p + 1))*np.exp(-s*tau)


Lc = L(fc)
print(f"|L(j wc)| = {abs(Lc):.4f},  fase = {np.rad2deg(np.angle(Lc)):.2f}°,  "
      f"PM = {180+np.rad2deg(np.angle(Lc)):.2f}°")
tau_b = 0.0125
fase_r = -90 - np.rad2deg(np.arctan(fc*2*np.pi/p)) - 360*fc*tau_b
print(f"con tau = 12,5 ms: |L_tau(j wc)| = {abs(L(fc, tau_b)):.4f}, "
      f"fase en el cruce = {fase_r:.2f}°  -> PM' = {180+fase_r:.2f}°")
# trazado: x = log10(f) de 1 a 100 Hz en [70, 560]; módulo en dB [+30, -40]
# en y [20, 160]; fase de -90° a -270° en y [200, 340]
fs = np.logspace(0, 2, 41)
X = 70 + (np.log10(fs) - 0)/2*490


def ydb(db):
    return 20 + (30 - db)/70*140


def yph(ph):
    return 200 + (-90 - ph)/180*140


mag = 20*np.log10(np.abs(L(fs)))
ph0 = -90 - np.rad2deg(np.arctan(2*np.pi*fs/p))
ph1 = ph0 - 360*fs*tau_b
print("Puntos de control:")
for f in [1, 2, 5, 10, 20, 50, 100]:
    print(f"  f = {f:5.1f} Hz  |L| = {20*np.log10(abs(L(f))):7.2f} dB   "
          f"fase sin retardo = {-90-np.rad2deg(np.arctan(2*np.pi*f/p)):8.2f}°   "
          f"con 12,5 ms = {-90-np.rad2deg(np.arctan(2*np.pi*f/p))-360*f*tau_b:8.2f}°")


def path(xs, ys, ymin=None, ymax=None):
    pts = []
    for x, y in zip(xs, ys):
        if ymax is not None and y > ymax:
            break
        pts.append(f"{x:.1f} {y:.1f}")
    return "M" + " L".join(pts)


print("path módulo :", path(X, ydb(mag)))
print("path fase 0 :", path(X, yph(ph0)))
# la curva con retardo se corta al salir del panel (fase < -270°)
fs2 = np.logspace(0, np.log10(29.0), 33)
X2 = 70 + np.log10(fs2)/2*490
ph2 = -90 - np.rad2deg(np.arctan(2*np.pi*fs2/p)) - 360*fs2*tau_b
print("path fase τ :", path(X2, yph(ph2)))
print(f"x(10 Hz) = {70+0.5*490:.1f}; y(0 dB) = {ydb(0):.1f}; "
      f"y(-135°) = {yph(-135):.1f}; y(-180°) = {yph(-180):.1f}")
print(f"fase con retardo a 29 Hz = {ph2[-1]:.1f}° (borde del panel, -270°)")

# ---------------------------------------------------------------- 4
cab(4, "Integrador con retardo: x'(t) = -K x(t - tau), el lazo del laboratorio 0")
K4 = 10.0
print(f"K = {K4} 1/s.  L(s) = K e^(-s tau)/s: |L(jw)| = K/w -> omega_c = K = {K4} rad/s "
      f"(f_c = {K4/(2*np.pi):.3f} Hz)")
print("fase de K/(jw) = -90° en todas las frecuencias -> PM = 90° = pi/2 rad")
tmax4 = (np.pi/2)/K4
print(f"tau_max = PM/omega_c = (pi/2)/{K4} = {tmax4:.4f} s = {tmax4*1e3:.1f} ms;  "
      f"periodo de la oscilación en el límite = 2 pi/K = {2*np.pi/K4:.4f} s")
dt = 1e-4
T = 3.0
n = int(T/dt)
tt = np.arange(n+1)*dt
resp = {}
for tau4 in [0.10, tmax4, 0.20]:
    d = int(round(tau4/dt))
    x = np.ones(n+1)
    for k in range(n):
        xr = x[k-d] if k-d >= 0 else 1.0     # historia x = 1 antes de t = 0
        x[k+1] = x[k] - dt*K4*xr             # Euler explícito, paso 0,1 ms
    resp[tau4] = x
    amp1 = np.max(np.abs(x[(tt > 1.0) & (tt <= 1.7)]))
    amp2 = np.max(np.abs(x[(tt > 2.3) & (tt <= 3.0)]))
    print(f"tau = {tau4*1e3:6.1f} ms (K tau = {K4*tau4:.3f}, PM' = "
          f"{np.rad2deg(np.pi/2-K4*tau4):6.1f}°): |x| máx en [1; 1,7] s = {amp1:.3f}, "
          f"en [2,3; 3] s = {amp2:.3f}")
# trazado: t de 0 a 2 s en x [60, 560]; x de -2 a 2 en y [20, 220]
# (la curva inestable se corta cuando sale de |x| <= 2)
idx = np.arange(0, int(2.0/dt)+1, 200)
for tau4, x in resp.items():
    xs = 60 + tt[idx]/2*500
    xv = x[idx]
    corte = np.argmax(np.abs(xv) > 2.0) if np.any(np.abs(xv) > 2.0) else len(xv)
    ys = 120 - xv[:corte]/2*100
    print(f"path tau={tau4*1e3:.0f} ms:", path(xs[:corte], ys))

# ---------------------------------------------------------------- 5
cab(5, "Action chunking: cuántas acciones por chunk y cuánto envejece la información")
for nombre, tinf, fctrl, H in [("Gemini Robotics", 0.250, 50.0, None),
                               ("GR00T N1", 0.0639, 120.0, 16)]:
    Hmin = tinf*fctrl
    print(f"{nombre}: latencia {tinf*1e3:.1f} ms, acciones a {fctrl:.0f} Hz "
          f"-> para no quedarse sin acciones, H >= tau*f = {Hmin:.2f}")
    if H is None:
        H = int(np.ceil(Hmin))
        print(f"   con H = {H} (mínimo entero): el chunk dura {H/fctrl*1e3:.0f} ms")
    else:
        print(f"   H = {H}: el chunk dura {H/fctrl*1e3:.1f} ms; inferir cada {tinf*1e3:.1f} ms "
              f"= {1/tinf:.2f} Hz")
    edad_min = tinf
    edad_max = tinf + (H-1)/fctrl
    print(f"   edad de la observación al ejecutar: de {edad_min*1e3:.0f} ms (primera acción) "
          f"a {edad_max*1e3:.0f} ms (última)")
