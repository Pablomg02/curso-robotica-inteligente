"""¿Nota el x500 el viento del mundo windy? Compara un estacionario en el mundo default y en windy (<wind> de 5 m/s
este y 2 m/s norte): inclinación media de la verdad de Gazebo, empuje y deriva. Uso:
    uv run --with pyulog --with numpy python extraer_viento.py viento_default.ulg viento_windy.ulg
Los ulog salen de figuras/experimentos/viento.py (Hold a 5 m durante 25 s en cada mundo)."""
import json, sys
from pathlib import Path
import numpy as np
from pyulog import ULog


def g(u, n):
    return [d for d in u.data_list if d.name == n][0].data


out = {}
for ruta, mundo in zip(sys.argv[1:3], ["default", "windy"]):
    u = ULog(ruta); s0 = int(u.start_timestamp)
    st = g(u, "vehicle_status"); ts = (st["timestamp"].astype(np.int64) - s0) * 1e-6
    fin_despegue = ts[np.flatnonzero(st["nav_state"] == 17)[-1]]  # último instante en AUTO_TAKEOFF
    k = np.flatnonzero((st["nav_state"] == 4) & (ts > fin_despegue))  # AUTO_LOITER (Hold) tras el despegue
    a, b = ts[k[0]] + 3.0, ts[k[0]] + 18.0
    at = g(u, "vehicle_attitude_groundtruth"); ta = (at["timestamp"].astype(np.int64) - s0) * 1e-6; m = (ta > a) & (ta < b)
    w, x, y, z = (at[f"q[{i}]"][m] for i in range(4))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    sp = g(u, "vehicle_attitude_setpoint"); tsp = (sp["timestamp"].astype(np.int64) - s0) * 1e-6; ms = (tsp > a) & (tsp < b)
    gt = g(u, "vehicle_local_position_groundtruth"); tg = (gt["timestamp"].astype(np.int64) - s0) * 1e-6; mg = (tg > a) & (tg < b)
    out[mundo] = {"ventana_s": [round(a, 1), round(b, 1)], "roll_medio_deg": round(float(roll.mean()), 3), "pitch_medio_deg": round(float(pitch.mean()), 3),
                  "inclinacion_media_deg": round(float(np.hypot(roll.mean(), pitch.mean())), 3),
                  "empuje_medio": round(float(-sp["thrust_body[2]"][ms].mean()), 4),
                  "deriva_xy_cm": round(float(np.hypot(gt["x"][mg][-1] - gt["x"][mg][0], gt["y"][mg][-1] - gt["y"][mg][0]) * 100), 1)}
json.dump(out, open(Path(__file__).parent / "datos" / "viento.json", "w"), indent=1)
print(json.dumps(out, indent=1))
