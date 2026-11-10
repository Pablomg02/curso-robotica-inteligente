"""Vuelo completo del x500 en PX4 SITL + Gazebo con MAVSDK v4 (binding nativo, API síncrona).
Despegue, círculo en offboard (posición + velocidad a 20 Hz), escalón de 5 m, misión de 4 puntos y aterrizaje.
Guarda tiempos, tasas y trayectorias en out/."""
import time, json, math, threading, csv, statistics, sys
from mavsdk import Mavsdk, Configuration, ComponentType
from mavsdk.plugins.action import Action
from mavsdk.plugins.telemetry import Telemetry
from mavsdk.plugins.offboard import Offboard, PositionNedYaw, VelocityNedYaw
from mavsdk.plugins.mission import Mission, MissionItem, MissionPlan
from mavsdk.plugins.param import Param

OUT = sys.argv[1] if len(sys.argv) > 1 else "/work/exp/out"
res = {}; t0 = time.monotonic()
cfg = Configuration.create_with_component_type(ComponentType.GROUND_STATION)
mav = Mavsdk(cfg)
mav.add_any_connection("udpin://0.0.0.0:14540")
drone = mav.first_autopilot(30.0)
assert drone is not None, "no hay autopiloto"
res["t_descubrir_s"] = round(time.monotonic() - t0, 3)
tel, act, off, mis, par = Telemetry(drone), Action(drone), Offboard(drone), Mission(drone), Param(drone)
while not tel.health_all_ok():
    time.sleep(0.1)
res["t_listo_s"] = round(time.monotonic() - t0, 3)

# Ida y vuelta de un parámetro a través de MAVSDK
rtt = []
for _ in range(100):
    a = time.perf_counter(); par.get_param_float("MPC_XY_VEL_MAX"); rtt.append((time.perf_counter() - a) * 1e3)
q = statistics.quantiles(rtt, n=20)
res["param_rtt_ms"] = {"median": round(statistics.median(rtt), 3), "p5": round(q[0], 3), "p95": round(q[-1], 3)}

# Registro de telemetría: posición/velocidad NED y verdad de Gazebo no hay por MAVLink; usamos la estimación
lock = threading.Lock(); log = []; counts = {"pvn": 0, "att": 0}
def cb_pvn(pv, *a):
    with lock:
        counts["pvn"] += 1
        log.append((time.monotonic() - t0, pv.position.north_m, pv.position.east_m, pv.position.down_m,
                    pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s))
def cb_att(e, *a):
    with lock: counts["att"] += 1
h1 = tel.subscribe_position_velocity_ned(cb_pvn); h2 = tel.subscribe_attitude_euler(cb_att)
time.sleep(5.0)
with lock: res["tasa_por_defecto_hz"] = {"position_velocity_ned": counts["pvn"] / 5, "attitude_euler": counts["att"] / 5}
tel.set_rate_position_velocity_ned(50.0)
with lock: counts["pvn"] = 0
time.sleep(5.0)
with lock: res["tasa_pedida_50hz"] = counts["pvn"] / 5

# Despegue
act.set_takeoff_altitude(5.0)
ta = time.monotonic(); act.arm(); res["t_armar_s"] = round(time.monotonic() - ta, 3)
act.takeoff(); tk = time.monotonic()
while True:
    with lock: d = log[-1][3]
    if d < -4.75: break
    time.sleep(0.02)
res["t_despegue_a_4m75_s"] = round(time.monotonic() - tk, 2)
time.sleep(4.0)

# Offboard: hay que mandar consignas antes de entrar
with lock: n0, e0, d0 = log[-1][1:4]
sp = []
def mandar(n, e, d, vn=0.0, ve=0.0):
    off.set_position_velocity_ned(PositionNedYaw(n, e, d, 0.0), VelocityNedYaw(vn, ve, 0.0, 0.0))
    sp.append((time.monotonic() - t0, n, e, d))
for _ in range(20):
    mandar(n0, e0, -5.0); time.sleep(0.05)
off.start(); res["t_offboard_inicio_s"] = round(time.monotonic() - t0, 3)
# Círculo de radio 5 m y periodo 20 s (1,57 m/s), dos vueltas, consignas a 20 Hz
R, T = 5.0, 20.0; w = 2 * math.pi / T
tc0 = time.monotonic(); k = 0
while True:
    t = time.monotonic() - tc0
    if t > 2 * T: break
    n = n0 + R * math.sin(w * t); e = e0 + R * (1 - math.cos(w * t))
    mandar(n, e, -5.0, R * w * math.cos(w * t), R * w * math.sin(w * t))
    k += 1; time.sleep(max(0.0, tc0 + k * 0.05 - time.monotonic()))
res["circulo"] = {"t_ini": round(tc0 - t0, 3), "t_fin": round(time.monotonic() - t0, 3), "R": R, "T": T}
# Escalón de 5 m hacia el norte desde el punto de partida
for _ in range(60):
    mandar(n0, e0, -5.0); time.sleep(0.05)
te0 = time.monotonic()
while time.monotonic() - te0 < 8.0:
    mandar(n0 + 5.0, e0, -5.0); time.sleep(0.05)
res["escalon"] = {"t_ini": round(te0 - t0, 3), "n0": n0, "salto_m": 5.0}
off.stop(); act.hold(); time.sleep(2.0)

# Misión de 4 puntos (cuadrado de 20 m a 5 m/s)
home_v = []
hh = tel.subscribe_home(lambda h, *a: home_v.append((h.latitude_deg, h.longitude_deg)))
while not home_v or math.isnan(home_v[-1][0]):   # el getter devuelve NaN hasta que llega el primer HOME_POSITION
    time.sleep(0.1)
tel.unsubscribe_home(hh)
lat0, lon0 = home_v[-1]
dlat = 20.0 / 111_320.0; dlon = 20.0 / (111_320.0 * math.cos(math.radians(lat0)))
pts = [(lat0 + dlat, lon0), (lat0 + dlat, lon0 + dlon), (lat0, lon0 + dlon), (lat0, lon0)]
items = [MissionItem(latitude_deg=la, longitude_deg=lo, relative_altitude_m=5.0, speed_m_s=5.0, is_fly_through=True,
                     gimbal_pitch_deg=float("nan"), gimbal_yaw_deg=float("nan"), camera_action=0, loiter_time_s=float("nan"),
                     camera_photo_interval_s=float("nan"), acceptance_radius_m=1.0, yaw_deg=float("nan"),
                     camera_photo_distance_m=float("nan"), vehicle_action=0) for la, lo in pts]
tu = time.monotonic(); mis.upload_mission(MissionPlan(mission_items=items)); res["t_subir_mision_ms"] = round((time.monotonic() - tu) * 1e3, 1)
time.sleep(1.0)   # navigator valida la misión de forma asíncrona; sin esperar, commander rechaza el modo Mission
tm0 = time.monotonic(); mis.start_mission()
while not mis.is_mission_finished():
    time.sleep(0.1)
res["mision"] = {"t_ini": round(tm0 - t0, 3), "duracion_s": round(time.monotonic() - tm0, 2), "lado_m": 20.0, "v_m_s": 5.0}
act.land(); tl = time.monotonic()
while tel.armed():
    time.sleep(0.1)
res["t_aterrizar_y_desarmar_s"] = round(time.monotonic() - tl, 2)
tel.unsubscribe_position_velocity_ned(h1); tel.unsubscribe_attitude_euler(h2)
with lock:
    with open(f"{OUT}/mavsdk_telemetria.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["t", "n", "e", "d", "vn", "ve", "vd"]); wr.writerows(log)
with open(f"{OUT}/mavsdk_consignas.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["t", "n", "e", "d"]); wr.writerows(sp)
json.dump(res, open(f"{OUT}/mavsdk_vuelo.json", "w"), indent=1)
print(json.dumps(res, indent=1))
