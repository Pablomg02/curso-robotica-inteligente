"""Extrae del ulog del vuelo con mc_nn_control (build px4_sitl_neural) el tramo en el modo externo (nav_state 23):
posición real de Gazebo frente a la consigna. Uso: uv run --with pyulog --with numpy python extraer_neural.py <neural.ulg>"""
import csv, sys, json
from pathlib import Path
import numpy as np
from pyulog import ULog
u = ULog(sys.argv[1]); s0 = int(u.start_timestamp)
g = lambda n: [d for d in u.data_list if d.name == n][0].data
st, gt, ts = g("vehicle_status"), g("vehicle_local_position_groundtruth"), g("trajectory_setpoint")
t = (st["timestamp"].astype(np.int64) - s0) * 1e-6
k = np.flatnonzero(st["nav_state"] == 23); a, b = t[k[0]], t[k[-1]]
tg = (gt["timestamp"].astype(np.int64) - s0) * 1e-6; tt = (ts["timestamp"].astype(np.int64) - s0) * 1e-6
rej = np.arange(a, b, 0.1)
idx = np.clip(np.searchsorted(tt, rej) - 1, 0, len(tt) - 1)
with open(Path(__file__).parent / "datos" / "neural.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t_s", "x", "y", "z", "sp_x", "sp_y", "sp_z"])
    for r, i in zip(rej, idx):
        w.writerow([round(r - a, 2)] + [round(float(np.interp(r, tg, gt[c])), 3) for c in "xyz"] +
                   [round(float(ts[f"position[{j}]"][i]), 3) for j in range(3)])
p = u.initial_parameters
json.dump({"t_ini": round(a, 2), "t_fin": round(b, 2), "MC_NN_THRST_COEF": p.get("MC_NN_THRST_COEF"), "MC_NN_MAX_RPM": p.get("MC_NN_MAX_RPM"),
           "MC_NN_MIN_RPM": p.get("MC_NN_MIN_RPM"), "MC_NN_MANL_CTRL_inicial": p.get("MC_NN_MANL_CTRL")},
          open(Path(__file__).parent / "datos" / "neural.json", "w"), indent=1)
print(a, b)
