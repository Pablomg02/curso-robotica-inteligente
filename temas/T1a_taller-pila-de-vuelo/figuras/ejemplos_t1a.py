"""Cuentas a mano de las hojas de T1a (Taller I: la pila de vuelo simulada).

Cada bloque reproduce un ejemplo numérico de la presentación. Solo necesita numpy.
Las cifras medidas que se usan aquí (tasas, bytes por segundo, tiempos) salen de los ficheros de figuras/datos/,
generados con los scripts de figuras/experimentos/ contra PX4 v1.17.0 SITL + Gazebo Harmonic (gz-sim 8.15.0).

    python3 ejemplos_t1a.py > salida_ejemplos.txt
"""
import json
import math
from pathlib import Path

import numpy as np

DATOS = Path(__file__).parent / "datos"
np.set_printoptions(precision=4, suppress=True)


def titulo(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------------------
titulo("1. Ejes: de FLU (Gazebo) a FRD (PX4) con el cuaternión (0, 1, 0, 0)")
# GZBridge::imuCallback: q_FLU_to_FRD = gz::math::Quaterniond(0, 1, 0, 0)  (w, x, y, z)
w, x, y, z = 0.0, 1.0, 0.0, 0.0
R = np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
])
print("R(q) =\n", R, "\ndet =", round(np.linalg.det(R), 6), "(giro de 180° alrededor de x: una rotación propia)")
a_flu = np.array([0.0, 0.0, 9.81])  # IMU quieta en Gazebo: la fuerza específica apunta hacia arriba (+z de FLU)
print("acelerómetro quieto: FLU", a_flu, "-> FRD", R @ a_flu, "(PX4 espera -g en z hacia abajo)")
w_flu = np.array([0.10, 0.20, 0.30])
print("giróscopo: FLU", w_flu, "-> FRD", R @ w_flu)

titulo("2. Ejes del mundo: de ENU a NED")
M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
p_enu = np.array([1.0, 2.0, 3.0])  # 1 m al este, 2 m al norte, 3 m de altura
print("M =\n", M, "\ndet =", round(np.linalg.det(M), 6), "-> det = +1: rotación propia de 180° alrededor de (1, 1, 0)/√2")
u_eje = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
print("2·u·uᵀ − I =\n", 2 * np.outer(u_eje, u_eje) - np.eye(3), "(fórmula del giro de 180° alrededor de u: coincide con M)")
print("posición ENU", p_enu, "-> NED", M @ p_enu)
print("aplicar M dos veces devuelve el vector:", np.allclose(M @ M, np.eye(3)))

titulo("3. El magnetómetro: (x, y, z) -> (-y, -x, z) en GZBridge::magnetometerCallback")
Mmag = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
print("Mmag =\n", Mmag, "\ndet =", round(np.linalg.det(Mmag), 6),
      "-> det = -1: compensa un convenio zurdo del plugin (comentario FIXME del código, gz-sim PR 2460)")
b = np.array([0.21, 0.02, -0.43])  # gauss, campo de ejemplo
print("campo del plugin", b, "-> lo que publica PX4", Mmag @ b)

# ---------------------------------------------------------------------------
titulo("4. Fase que se come el retardo de muestreo: 250 Hz (SITL) frente a 800 Hz (Pixhawk)")
f_c = 6.3  # Hz, cruce del lazo de rate del x500 en A2
for f_s in (250.0, 800.0):
    T = 1.0 / f_s
    tau = 1.5 * T  # muestreo y cálculo: 1,5 periodos (supuesto de A2)
    phi = 360.0 * f_c * tau
    print(f"f_s = {f_s:5.0f} Hz  T = {T*1e3:6.3f} ms  tau = {tau*1e3:6.3f} ms  fase perdida en f_c: {phi:5.2f}°")
d = 360.0 * f_c * 1.5 * (1 / 250 - 1 / 800)
print(f"diferencia de margen de fase entre SITL y la placa: {d:.2f}° (de un margen de 71° en A2)")

# ---------------------------------------------------------------------------
titulo("5. ¿Cabe el flujo de MAVLink de SITL en una radio de telemetría?")
for puerto in ("14550", "14540"):
    m = json.loads((DATOS / f"mavlink_{puerto}.json").read_text())
    print(f"puerto {puerto}: {m['msgs_per_s']} mensajes/s, {m['bytes_per_s']} B/s")
baud = 57600
bps = baud / 10  # 8N1: 1 bit de inicio + 8 de datos + 1 de parada
m = json.loads((DATOS / "mavlink_14550.json").read_text())
print(f"radio SiK a {baud} baudios: {bps:.0f} B/s útiles; el flujo de la GCS en SITL es {m['bytes_per_s']/bps:.2f} veces eso")

# ---------------------------------------------------------------------------
titulo("6. Seguir un círculo con un lazo P de posición, con y sin velocidad de prealimentación")
K = 0.95  # MPC_XY_P (1/s), leído en el ulog
R_c, T_c = 5.0, 20.0
om = 2 * math.pi / T_c
v = R_c * om
print(f"v = R·omega = {R_c}·{om:.4f} = {v:.4f} m/s")
print(f"retraso en una recta sin prealimentación: e = v/K = {v/K:.3f} m")
G = K / complex(K, om)  # G(j omega) = K/(j omega + K)
print(f"|G(j omega)| = {abs(G):.4f} -> radio seguido {R_c*abs(G):.3f} m (se encoge {R_c*(1-abs(G))*100:.1f} cm)")
print(f"fase = {math.degrees(math.atan2(om, K)):.2f}° de retraso -> arco {R_c*math.atan2(om, K):.3f} m por detrás")
err = abs(1 - G) * R_c
print(f"error de posición |1 - G|·R = {err:.3f} m (constante a lo largo del círculo)")
print("con prealimentación v_ff = d(p_ref)/dt: G_ff = (j omega + K)/(j omega + K) = 1 -> error nulo en el modelo")

# ---------------------------------------------------------------------------
titulo("6b. El EKF2 va por delante: retardo de GNSS supuesto que Gazebo no tiene")
ur = json.loads((DATOS / "ulog_resumen.json").read_text())
tau_gps = ur["parametros"].get("EKF2_GPS_DELAY", 110.0) if "EKF2_GPS_DELAY" in ur["parametros"] else 110.0
print(f"EKF2_GPS_DELAY = 110 ms; ajuste e = v·tau sobre el vuelo: tau = {ur['ajuste_e_igual_v_tau']['tau_s']*1e3:.1f} ms, residuo {ur['ajuste_e_igual_v_tau']['residuo_cm']} cm")
print(f"en el círculo (v = {v:.3f} m/s): la estimación va {v*0.110*100:.1f} cm por delante de la verdad")
for b in ur["err_horiz_por_velocidad"]:
    print(f"  v media {b['v_media']:.2f} m/s: medido {b['err_medio_cm']} cm, v·0,110 = {b['v_media']*11.0:.1f} cm")

titulo("7. Cuánto debería durar la misión cuadrada (20 m por lado a 5 m/s)")
L, vmax, a, j = 20.0, 5.0, 3.0, 4.0  # MPC_ACC_HOR = 3 m/s², MPC_JERK_AUTO = 4 m/s³
t_trap = L / vmax + vmax / a
print(f"perfil trapezoidal (parar en cada esquina): {t_trap:.3f} s por lado, {4*t_trap:.2f} s en total")
tj = a / j
dv_j = a * a / j
ta = (vmax - dv_j) / a
t_acc = 2 * tj + ta
d_acc = vmax * t_acc / 2
t_crucero = (L - 2 * d_acc) / vmax
t_lado = 2 * t_acc + t_crucero
print(f"con jerk limitado: subir a 5 m/s tarda {t_acc:.3f} s y recorre {d_acc:.3f} m; crucero {t_crucero:.3f} s")
print(f"  -> {t_lado:.3f} s por lado, {4*t_lado:.2f} s en total")
mis = json.loads((DATOS / "mavsdk_vuelo.json").read_text())["mision"]["duracion_s"]
print(f"medido: {mis} s; el modelo con jerk explica {4*t_lado/mis*100:.0f} %")

# ---------------------------------------------------------------------------
titulo("8. Sincronización de relojes al estilo NTP (RFC 5905, §8)")
# t1: el agente envía; t2: PX4 recibe; t3: PX4 responde; t4: el agente recibe.
# Reloj de PX4 = reloj del agente + 500 µs; 300 µs de ida, 200 µs de vuelta, 50 µs de proceso.
off_real, ida, vuelta, proc = 500, 300, 200, 50
t1 = 10_000
t2 = t1 + ida + off_real
t3 = t2 + proc
t4 = t3 - off_real + vuelta
theta = ((t2 - t1) + (t3 - t4)) / 2
delta = (t4 - t1) - (t3 - t2)
print(f"t1={t1} t2={t2} t3={t3} t4={t4} (µs)")
print(f"theta = ((t2-t1)+(t3-t4))/2 = {theta:.0f} µs (real {off_real}); delta = (t4-t1)-(t3-t2) = {delta} µs")
print(f"error del offset = (ida - vuelta)/2 = {(ida-vuelta)/2:.0f} µs: la asimetría del canal no se puede ver")

# ---------------------------------------------------------------------------
titulo("9. Lo que tarda el ACK de un comando que pasa por commander")
lazo, poll = 10.0, 10.0  # ms: COMMANDER_MONITORING_INTERVAL y UXRCE_DEFAULT_POLL_INTERVAL_MS
print(f"espera hasta el siguiente lazo de commander: U(0, {lazo}) ms, media {lazo/2} ms")
print(f"espera hasta que el cliente XRCE sondea vehicle_command_ack: U(0, {poll}) ms, media {poll/2} ms")
print(f"suma: media {lazo/2+poll/2} ms, máximo {lazo+poll} ms, más el transporte")
r2 = json.loads((DATOS / "ros2_vuelo.json").read_text())
print("medido desde ROS 2:", r2["ack_ms"])

# ---------------------------------------------------------------------------
titulo("10. Factor de tiempo real y presupuesto de cada paso")
esc = json.loads((DATOS / "escala.json").read_text())
h = 0.004  # s, max_step_size del mundo default
for r in esc:
    if r["drones"] == 1 and r["pedido"] == 100:
        print(f"1 dron, pedido x100: alcanzado x{r['rtf']} -> {h/r['rtf']*1e6:.0f} µs de reloj de pared por paso de 4 ms")
for r in esc:
    if r["pedido"] == 100:
        print(f"{r['drones']} drones a todo lo que da: x{r['rtf']}; {r['rtf']*r['drones']:.1f} dron-segundos por segundo")

# ---------------------------------------------------------------------------
titulo("11. Tamaño del registro")
u = json.loads((DATOS / "ulog_resumen.json").read_text())
print(f"{u['bytes']/1e6:.1f} MB en {u['dur_s']} s -> {u['bytes']/u['dur_s']/1e3:.0f} kB/s, {u['bytes']/u['dur_s']*60/1e6:.1f} MB/min")
for n, bts in u["mayores_temas"][:4]:
    print(f"  {n:28s} {bts/1e6:6.2f} MB  {bts/u['bytes']*100:5.1f} %")
