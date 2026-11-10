"""Dos círculos en offboard (con y sin velocidad de prealimentación) y un fallo de GPS provocado en estacionario.
Guarda los instantes de cada tramo en out/circulo_fallo.json; las trayectorias se leen luego del ulog."""
import time, json, math, subprocess
from mavsdk import Mavsdk, Configuration, ComponentType
from mavsdk.plugins.action import Action
from mavsdk.plugins.telemetry import Telemetry
from mavsdk.plugins.offboard import Offboard, PositionNedYaw, VelocityNedYaw
BIN = "/work/px4/build/px4_sitl_default/bin"
def px4(c): return subprocess.run(f"cd /work/px4/build/px4_sitl_default/rootfs && {BIN}/px4-{c}", shell=True, capture_output=True, text=True).stdout
res = {}; t0 = time.monotonic(); ev = []
def marca(n): ev.append((n, round(time.monotonic() - t0, 3))); print(n, flush=True)
mav = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
mav.add_any_connection("udpin://0.0.0.0:14540"); d = mav.first_autopilot(30.0)
tel, act, off = Telemetry(d), Action(d), Offboard(d)
while not tel.health_all_ok(): time.sleep(0.1)
px4("param set SYS_FAILURE_EN 1")
pos = []; tel.set_rate_position_velocity_ned(50.0)
h = tel.subscribe_position_velocity_ned(lambda pv, *a: pos.append((pv.position.north_m, pv.position.east_m, pv.position.down_m)))
act.set_takeoff_altitude(5.0)
for intento in range(20):   # commander tarda un ciclo en ver la GCS: sin esperar, «Arming denied»
    try:
        act.arm(); break
    except Exception as e:
        print("armado rechazado, reintento:", e); time.sleep(1.0)
act.takeoff(); marca("despegue")
while not pos or pos[-1][2] > -4.75: time.sleep(0.05)
time.sleep(4.0)
n0, e0, _ = pos[-1]
def sp(n, e, vn=None, ve=None):
    if vn is None: off.set_position_ned(PositionNedYaw(n, e, -5.0, 0.0))
    else: off.set_position_velocity_ned(PositionNedYaw(n, e, -5.0, 0.0), VelocityNedYaw(vn, ve, 0.0, 0.0))
for _ in range(20): sp(n0, e0); time.sleep(0.05)
off.start(); marca("offboard")
R, T = 5.0, 20.0; w = 2 * math.pi / T
for nombre, ff in [("circulo_ff", True), ("circulo_sin_ff", False)]:
    for _ in range(60): sp(n0, e0); time.sleep(0.05)      # 3 s quieto en el punto de partida
    marca(nombre + "_ini"); tc0 = time.monotonic(); k = 0
    while time.monotonic() - tc0 < 2 * T:
        t = time.monotonic() - tc0
        n = n0 + R * math.sin(w * t); e = e0 + R * (1 - math.cos(w * t))
        if ff: sp(n, e, R * w * math.cos(w * t), R * w * math.sin(w * t))
        else: sp(n, e)
        k += 1; time.sleep(max(0.0, tc0 + k * 0.05 - time.monotonic()))
    marca(nombre + "_fin")
for _ in range(60): sp(n0, e0); time.sleep(0.05)
off.stop(); act.hold(); marca("hold_ini")
time.sleep(10.0)
open("/work/exp/out/uorb_top_hover.txt", "w").write(px4("uorb top -1"))
open("/work/exp/out/work_queue_hover.txt", "w").write(px4("work_queue status"))
time.sleep(35.0); marca("hold_fin")   # 45 s en estacionario para comparar con un vuelo real (T1b)
# Fallo de GPS en estacionario (modo Hold)
marca("gps_off"); print(px4("failure gps off"))
for i in range(40):
    time.sleep(1.0)
st = px4("listener vehicle_status -n 1"); marca("gps_ok"); print(px4("failure gps ok"))
time.sleep(10.0)
res["eventos"] = ev; res["nav_state_tras_fallo"] = [l.strip() for l in st.splitlines() if "nav_state:" in l]
act.land()
while tel.armed(): time.sleep(0.2)
marca("desarmado"); tel.unsubscribe_position_velocity_ned(h)
json.dump(res, open("/work/exp/out/circulo_fallo.json", "w"), indent=1); print(json.dumps(res, indent=1))
