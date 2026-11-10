"""Cuentas a mano de las hojas de T1b (Taller II: simular la realidad en 2026).

Cada bloque reproduce un ejemplo numérico de la presentación. Solo necesita numpy.
Las cifras medidas salen de figuras/datos/ (ver comparar_real_sitl.py, extraer_neural.py y figuras/experimentos/).

    python3 ejemplos_t1b.py > salida_ejemplos.txt
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

DATOS = Path(__file__).parent / "datos"
RS = json.loads((DATOS / "real_sitl.json").read_text())
CAM = json.loads((DATOS / "camara_neuronal.json").read_text())
IMU1K = json.loads((DATOS / "imu_1khz.json").read_text())


def titulo(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------------------
titulo("1. Empuje de estacionario del x500 de Gazebo, a mano")
m = 2.0 + 4 * 0.016076923  # kg: base_link + 4 rotores (x500_base/model.sdf)
g = 9.8                    # m/s²: <gravity>0 0 -9.8</gravity> en worlds/default.sdf
kT = 8.54858e-6            # N/(rad/s)²: motorConstant (x500/model.sdf)
wmin, wmax = 150.0, 1000.0  # rad/s: SIM_GZ_EC_MIN y SIM_GZ_EC_MAX (airframe 4001_gz_x500)
W = m * g
T = W / 4
w = math.sqrt(T / kT)
u = (w - wmin) / (wmax - wmin)
print(f"masa {m:.4f} kg, peso {W:.3f} N, por rotor {T:.4f} N")
print(f"omega = sqrt(T/kT) = {w:.1f} rad/s ({w*60/(2*math.pi):.0f} rpm)")
print(f"mando u = (omega - {wmin:.0f})/({wmax:.0f} - {wmin:.0f}) = {u:.4f}")
print(f"medido en SITL: empuje medio {RS['sitl']['empuje_medio']}, estimador de empuje (hover_thrust_estimate) 0.7277")
u0 = 0.60
w0 = wmin + u0 * (wmax - wmin)
print(f"con MPC_THR_HOVER = 0.60: omega = {w0:.0f} rad/s, empuje {4*kT*w0**2:.2f} N = {4*kT*w0**2/W*100:.1f} % del peso")
print(f"pendiente dT/du por rotor en estacionario: 2·kT·omega·{wmax-wmin:.0f} = {2*kT*w*(wmax-wmin):.2f} N por unidad de mando")
print(f"real (X500 V2, f4f9b50c): estimador {RS['real']['hover_thrust_estimado']}, MPC_THR_HOVER {RS['real']['MPC_THR_HOVER']}")

# ---------------------------------------------------------------------------
titulo("2. Densidad espectral del ruido blanco del giróscopo de Gazebo")
sig = 0.0008726646  # rad/s por muestra: stddev del SDF (0,05 °/s)
fs = RS["sitl"]["fs_giro_hz"]
S = 2 * sig ** 2 / fs
print(f"sigma = {sig} rad/s = {math.degrees(sig):.3f} °/s por muestra a fs = {fs:.0f} Hz")
print(f"PSD unilateral plana: S = 2·sigma²/fs = {S:.3e} (rad/s)²/Hz")
print(f"comprobación: integrar S de 0 a fs/2 da S·fs/2 = {S*fs/2:.3e} = sigma² = {sig**2:.3e}")
print(f"densidad de ruido sqrt(S) = {math.sqrt(S):.3e} rad/s/√Hz = {math.degrees(math.sqrt(S)):.5f} °/s/√Hz")
print("lo que registra PX4 (sensor_combined) es la media de dos muestras seguidas (trapecio de Integrator.hpp):")
print("  H(f) = (1 + exp(-j 2 pi f/fs))/2  ->  |H(f)|² = cos²(pi f/fs): 1 en 0 Hz, 0,5 en fs/4, 0 en Nyquist")
for fx in (10, 30, 62.5, 100, 125):
    print(f"  f = {fx:5.1f} Hz: cos² = {math.cos(math.pi*fx/fs)**2:.3f} -> S0·cos² = {S*math.cos(math.pi*fx/fs)**2:.2e}")
print(f"  varianza de la media de dos muestras: sigma²/2 = {sig**2/2:.3e} (= integral de S0·cos² de 0 a fs/2 = S0·fs/4 = {S*fs/4:.3e})")
b20 = RS['sitl']['psd_banda_20_50']; c2 = RS['sitl']['cos2_medio_20_50']; bc = RS['sitl']['psd_banda_20_50_corregida']
print(f"medida en el SITL, banda 20-50 Hz: {b20:.3e}; cos² medio {c2:.3f}; corregida {bc:.3e} = {bc/S*100-100:+.0f} % sobre la teórica")
import csv as _csv
fil = list(_csv.DictReader(open(DATOS / "psd_giro.csv")))
fr_ = np.array([float(r["f_hz"]) for r in fil if r["fuente"] == "real"]); pr_ = np.array([float(r["psd_rad2_s2_hz"]) for r in fil if r["fuente"] == "real"])
fs2 = np.array([float(r["f_hz"]) for r in fil if r["fuente"] == "sitl"]); ps2 = np.array([float(r["psd_rad2_s2_hz"]) for r in fil if r["fuente"] == "sitl"])
kk = (fs2 >= 2) & (fs2 <= 100)
rat = np.interp(np.log(fs2[kk]), np.log(fr_), pr_) / ps2[kk]
print(f"cociente real/SITL de la PSD entre 2 y 100 Hz: de {rat.min():.0f} a {rat.max():.0f} (mediana {np.median(rat):.0f})")

# ---------------------------------------------------------------------------
titulo("3. Real frente a SITL: cocientes de la dispersión en estacionario")
r, s = RS["real"], RS["sitl"]
print(f"giróscopo (sensor_combined, eje x): real {r['giro_std_deg_s']} °/s, SITL {s['giro_std_deg_s']} °/s -> x{r['giro_std_deg_s']/s['giro_std_deg_s']:.1f}")
for i, e in enumerate("xyz"):
    print(f"acelerómetro {e}: real {r['acc_std_m_s2'][i]}, SITL {s['acc_std_m_s2'][i]} m/s² -> x{r['acc_std_m_s2'][i]/s['acc_std_m_s2'][i]:.0f}")
for i, e in enumerate("xyz"):
    print(f"posición {e}: real {r['pos_std_cm'][i]} cm, SITL {s['pos_std_cm'][i]} cm")
print(f"error de actitud RMS (roll, pitch): real {r['err_roll_rms_deg']}, {r['err_pitch_rms_deg']}°; SITL {s['err_roll_rms_deg']}, {s['err_pitch_rms_deg']}°")
print(f"giróscopo en bruto del FIFO (real, {RS['real_fifo']['fs_fifo_hz']} Hz): {RS['real_fifo']['giro_std_deg_s_fifo']} °/s")

# ---------------------------------------------------------------------------
titulo("4. De las RPM a las frecuencias de la vibración, y qué pasa al muestrear a 250 Hz")
f = RS["real_fifo"]
rpm = f["rpm_media"]
fr = rpm / 60
print(f"RPM media {rpm:.0f} -> giro {fr:.1f} Hz; paso de pala (2 palas) {2*fr:.1f} Hz; armónicos 2·, 4·, 6·giro: "
      f"{2*fr:.1f}, {4*fr:.1f}, {6*fr:.1f} Hz")
print(f"pico medido entre 40 y 400 Hz: {f['pico_hz']} Hz = {f['pico_hz']/fr:.2f} veces el giro medio")
for fsi in (250.0, 800.0, 1000.0):
    ny = fsi / 2
    fa = abs(2 * fr - round(2 * fr / fsi) * fsi)
    print(f"muestreo a {fsi:.0f} Hz: Nyquist {ny:.0f} Hz; el paso de pala ({2*fr:.1f} Hz) "
          + ("cabe" if 2 * fr < ny else f"se pliega a {fa:.1f} Hz"))
print(f"la IMU real muestrea a {f['fs_fifo_hz']} Hz con {f['muestras_por_mensaje']} muestras por mensaje: "
      f"{f['fs_fifo_hz']/f['muestras_por_mensaje']:.0f} mensajes/s")

# ---------------------------------------------------------------------------
titulo("5. IMU a 1 kHz en Gazebo: fase recuperada y coste")
fc = 6.3
for fsi in (250.0, 800.0, 1000.0):
    print(f"lazo de rate a {fsi:.0f} Hz: fase perdida en {fc} Hz = 540·{fc}/{fsi:.0f} = {540*fc/fsi:.2f}°")
rtf4, rtf1 = 34.76, IMU1K[1]["rtf"]
print(f"paso de 4 ms a x{rtf4}: {4e3/rtf4:.0f} µs de pared por paso; paso de 1 ms a x{rtf1}: {1e3/rtf1:.0f} µs por paso")
print(f"cada paso cuesta {1e3/rtf1/(4e3/rtf4)*100:.0f} % del de 4 ms, pero hay 4 veces más: el tope cae x{rtf4/rtf1:.2f}")
print("frecuencias medidas a 1 kHz:", IMU1K[0]["hz"])

# ---------------------------------------------------------------------------
titulo("6. La cámara neuronal: memoria, coste por gaussiana y presupuesto de latencia")
n, nf = CAM["n_gaussianas"], CAM["floats_por_gaussiana"]
print(f"floats por gaussiana: 3 (posición) + 4 (cuaternión) + 3 (escala) + 1 (opacidad) + 48 (armónicos esféricos de grado 3: 16 × 3) = {3+4+3+1+48}")
print(f"{n} gaussianas × {nf} floats × 4 B = {n*nf*4/1e6:.0f} MB (medido {CAM['memoria_parametros_mb']} MB)")
for W_ in (320, 640, 1280):
    pts = [(c["n"] / 1e6, c["ms_mediana"]) for c in CAM["coste"] if c["W"] == W_]
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    a, b = np.polyfit(x, y, 1)
    print(f"{W_}×{W_*3//4}: {a:.2f} ms por millón de gaussianas, ordenada {b:+.2f} ms")
v = CAM["vivo"]
print(f"bucle en vivo: {v['hz_logrados']} Hz, render {v['render_ms_mediana']} ms (p95 {v['render_ms_p95']})")
espera = 0.5 * 1000 / v["poses_recibidas_hz"]
print(f"modelo: espera desde la última pose U(0, {1000/v['poses_recibidas_hz']:.0f} ms), media {espera:.0f} ms, + render {v['render_ms_mediana']} ms "
      f"= {espera+v['render_ms_mediana']:.1f} ms; medido {v['edad_pose_ms_mediana']} ms")
print(f"ocupación de la GPU a 30 Hz: {v['render_ms_mediana']}/{1000/30:.1f} ms = {v['render_ms_mediana']/(1000/30)*100:.0f} %")
edad = v["edad_pose_ms_mediana"] / 1000
fpx = 0.9 * v["W"]
for vel in (2.0, 5.0):
    print(f"a {vel} m/s, en {edad*1e3:.1f} ms el dron se mueve {vel*edad*100:.1f} cm entre la pose y la imagen")
for gir in (30.0, 90.0):
    ang = gir * edad
    print(f"girando a {gir:.0f} °/s: {ang:.2f}° -> {fpx*math.tan(math.radians(ang)):.1f} px de desfase con f = {fpx:.0f} px")

# ---------------------------------------------------------------------------
titulo("7. La red de mc_nn_control: tamaño, cómputo y la cadena de la salida al motor")
capas = [(15, 64), (64, 32), (32, 4)]
params = sum(i * o + o for i, o in capas)
macs = sum(i * o for i, o in capas)
print(f"capas 15-64-32-4: {params} parámetros ({params*4} B en float32; el .tflite ocupa 15088 B); {macs} multiplicaciones-suma")
t_inf = 93.4e-6  # s, Hegre et al. (2025), STM32H743 a 460 MHz
print(f"a 93,4 µs por inferencia (Hegre et al.): {macs/t_inf/1e6:.1f} M MAC/s; {macs/(t_inf*460e6):.3f} MAC por ciclo a 460 MHz")
print(f"lazo completo 137,6 µs a 650 Hz (periodo {1e6/650:.0f} µs): {137.6/(1e6/650)*100:.1f} % del periodo")
# Inversa de RescaleActions (mc_nn_control.cpp): de un mando normalizado u a la salida de la red
coef, rmin, rmax, a = 1.2 / 1e5, 1000.0, 22000.0, 0.8
b_ = 1 - a; t1 = b_ / (2 * a); t2 = b_ * b_ / (4 * a * a)


def directa(o):
    fz = min(max(o, -1.0), 1.0) + 1.0
    rpm_ = math.sqrt(fz / coef) * 60
    rn = (rpm_ * 2 - rmax - rmin) / (rmax - rmin)
    return a * (((rn + 1) / 2 + t1) ** 2 - t2), rpm_


def inversa(u_):
    s_ = math.sqrt(u_ / a + t2) - t1
    rn = 2 * s_ - 1
    rpm_ = (rn * (rmax - rmin) + rmax + rmin) / 2
    fz = coef * (rpm_ / 60) ** 2
    return fz - 1.0, rpm_


for u_ in (0.728,):
    o, rp = inversa(u_)
    print(f"mando {u_} <- salida de la red {o:.3f} (rpm del modelo de la red {rp:.0f}); comprobación directa: {directa(o)[0]:.4f}")
for o in (-1.0, 0.0, 1.0):
    print(f"salida {o:+.1f} -> mando {directa(o)[0]:.3f}")
print(f"en el X500 V2 real, estacionario a {rpm:.0f} rpm (registro c8082ca8): las 'rpm' de la red no son las del motor, solo una escala")
rows = list(csv.DictReader(open(DATOS / "neural.csv")))
t = np.array([float(x["t_s"]) for x in rows])
for (a_, b2, nombre) in [(3, 6, "z = -5 m"), (19, 22, "z = -5 m tras dos escalones"), (27, 30, "z = -7 m"), (35, 38, "vuelta a -5 m")]:
    mm = (t >= a_) & (t <= b2)
    ez = np.mean([float(x["z"]) - float(x["sp_z"]) for x, k in zip(rows, mm) if k])
    print(f"{nombre}: z real - consigna = {ez*100:+.1f} cm (positivo = por debajo, en NED)")

# ---------------------------------------------------------------------------
titulo("8. SRCC: ¿ordena el simulador las políticas como la realidad? (ejemplo inventado)")
sim = np.array([0.95, 0.90, 0.85, 0.70, 0.60])
real = np.array([0.55, 0.70, 0.50, 0.45, 0.30])
print("éxito en simulación:", sim, " en real:", real)
ds, dr = sim - sim.mean(), real - real.mean()
print("desviaciones respecto a la media: sim", ds.round(3), " real", dr.round(3))
num = np.sum(ds * dr); den = math.sqrt(np.sum(ds ** 2) * np.sum(dr ** 2))
print(f"sum(ds·dr) = {num:.4f}; sqrt(sum ds² · sum dr²) = {den:.4f}; SRCC (Pearson) = {num/den:.3f}")
print(f"comprobación con numpy: {np.corrcoef(sim, real)[0,1]:.3f}")
rk = lambda v: np.argsort(np.argsort(-v)) + 1
print("rangos sim", rk(sim), " real", rk(real), f" Spearman = {np.corrcoef(rk(sim), rk(real))[0,1]:.3f}")
print(f"brecha media (sim - real) = {np.mean(sim-real):.3f}: el SRCC no la ve; mide si el orden se conserva")

# ---------------------------------------------------------------------------
titulo("9. Intervalo de Wilson para una tasa de éxito")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, c - h, c + h


for k, n_ in [(69, 70), (9, 10), (18, 20), (90, 100)]:
    p, lo, hi = wilson(k, n_)
    pw = 1.96 * math.sqrt(p * (1 - p) / n_)
    print(f"{k}/{n_}: p = {p:.3f}; Wilson 95 % [{lo:.3f}, {hi:.3f}]; Wald {p:.3f} ± {pw:.3f} -> [{p-pw:.3f}, {min(p+pw,1):.3f}]")
for n_ in (10, 20, 50, 100, 200):
    _, lo9, hi9 = wilson(round(0.9 * n_), n_)
    _, lo8, hi8 = wilson(round(0.8 * n_), n_)
    print(f"n = {n_:3d}: 90 % -> [{lo9:.3f}, {hi9:.3f}]; 80 % -> [{lo8:.3f}, {hi8:.3f}]; " + ("se separan" if lo9 > hi8 else "se solapan"))

# ---------------------------------------------------------------------------
titulo("10. El viento del mundo windy: inclinación de equilibrio con solo el arrastre de los rotores")
VI = json.loads((DATOS / "viento.json").read_text())
vw = math.hypot(5.0, 2.0)          # m/s: <wind><linear_velocity>5 2 0</linear_velocity> en worlds/windy.sdf
cd = 8.06428e-05                   # rotorDragCoefficient del x500 (model.sdf)
F = 4 * cd * w * vw                # N: cada rotor, |Omega|·c·v_perp, con el dron quieto
th = math.degrees(math.atan(F / W))
print(f"|v_viento| = sqrt(5² + 2²) = {vw:.3f} m/s; arrastre por rotor c·Omega·v = {cd}·{w:.1f}·{vw:.3f} = {F/4:.4f} N")
print(f"total {F:.3f} N; tan(theta) = F/(m·g) = {F:.3f}/{W:.3f} -> theta = {th:.2f}°")
print(f"medido: windy {VI['windy']['inclinacion_media_deg']}° (roll {VI['windy']['roll_medio_deg']}, pitch {VI['windy']['pitch_medio_deg']}); "
      f"default {VI['default']['inclinacion_media_deg']}°")
print(f"cociente roll/pitch medido {VI['windy']['roll_medio_deg']/VI['windy']['pitch_medio_deg']:.2f} frente a 2/5 = 0.40 de las componentes del viento")
