"""¿Sopla el viento del mundo windy sobre el x500? Despega a 5 m, Hold 25 s, aterriza; el análisis va aparte con el ulog.
Uso: python3 viento.py <etiqueta>  (con PX4 SITL ya en marcha en el mundo que se quiera probar)."""
import sys, time, json
from mavsdk import Mavsdk, Configuration, ComponentType
from mavsdk.plugins.action import Action
from mavsdk.plugins.telemetry import Telemetry
mav = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
mav.add_any_connection("udpin://0.0.0.0:14540"); d = mav.first_autopilot(30.0); tel, act = Telemetry(d), Action(d)
while not tel.health_all_ok(): time.sleep(0.1)
act.set_takeoff_altitude(5.0)
for _ in range(20):
    try: act.arm(); break
    except Exception: time.sleep(1.0)
act.takeoff(); time.sleep(12.0)
t0 = time.time(); time.sleep(25.0)
act.land()
while tel.armed(): time.sleep(0.2)
json.dump({"etiqueta": sys.argv[1], "hold_s": 25.0}, open(f"/work/exp/out/viento_{sys.argv[1]}.json", "w"))
print("hecho", sys.argv[1])
