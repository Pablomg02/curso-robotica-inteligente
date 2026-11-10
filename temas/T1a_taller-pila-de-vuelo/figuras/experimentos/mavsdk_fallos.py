"""Fallos provocados en SITL + Gazebo (PX4 v1.17): fusión de GNSS desactivada en estacionario y un motor apagado.
`failure gps off` no tiene efecto con Gazebo en v1.17 (solo lo atiende SimulatorMavlink); por eso el GNSS se quita
desactivando su fusión en EKF2 (EKF2_GPS_CTRL = 0), y el motor con `failure motor off` (FailureInjector de commander)."""
import time, json, subprocess
from mavsdk import Mavsdk, Configuration, ComponentType
from mavsdk.plugins.action import Action
from mavsdk.plugins.telemetry import Telemetry
BIN = "/work/px4/build/px4_sitl_default/bin"
def px4(c): return subprocess.run(f"cd /work/px4/build/px4_sitl_default/rootfs && {BIN}/px4-{c}", shell=True, capture_output=True, text=True).stdout
t0 = time.monotonic(); ev = []
def marca(n, extra=""): ev.append((n, round(time.monotonic() - t0, 3))); print(n, extra, flush=True)
mav = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
mav.add_any_connection("udpin://0.0.0.0:14540"); d = mav.first_autopilot(30.0)
tel, act = Telemetry(d), Action(d)
while not tel.health_all_ok(): time.sleep(0.1)
print(px4("param set SYS_FAILURE_EN 1")); gps_ctrl = px4("param show -q EKF2_GPS_CTRL").strip(); print("EKF2_GPS_CTRL =", gps_ctrl)
act.set_takeoff_altitude(10.0)
for _ in range(20):
    try: act.arm(); break
    except Exception as e: time.sleep(1.0)
act.takeoff(); marca("despegue")
z = []; h = tel.subscribe_position_velocity_ned(lambda pv, *a: z.append(pv.position.down_m))
while not z or z[-1] > -9.7: time.sleep(0.1)
time.sleep(5.0); act.hold(); time.sleep(10.0)
marca("gnss_fusion_off", px4("param set EKF2_GPS_CTRL 0"))
time.sleep(40.0)
marca("gnss_fusion_on", px4(f"param set EKF2_GPS_CTRL {gps_ctrl}"))
time.sleep(20.0)
try: act.hold()
except Exception as e: print("hold:", e)
time.sleep(8.0)
marca("motor1_off", px4("failure motor off -i 1"))
time.sleep(8.0); marca("fin")
try: act.kill()
except Exception as e: print("kill:", e)
json.dump({"eventos": ev}, open("/work/exp/out/fallos.json", "w"), indent=1)
