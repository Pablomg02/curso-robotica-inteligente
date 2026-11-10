"""Extrae de los ulog de SITL los datos pequeños con los que se dibujan las figuras de T1a.

Los .ulg no se suben al repositorio (37-60 MB cada uno): se generan con los scripts de figuras/experimentos/.
    uv run --with pyulog --with numpy python extraer_t1a.py <vuelo_mavsdk.ulg> <circulos.ulg> <fallo_gnss.ulg> <fallo_motor.ulg>
Escribe en figuras/datos/: ulog_resumen.json, escalon.csv, error_ekf.csv, circulos.csv, fallo_gnss.csv y fallo_motor.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from pyulog import ULog

DATOS = Path(__file__).parent / "datos"


def topic(u, name, inst=0):
    for d in u.data_list:
        if d.name == name and d.multi_id == inst:
            return d.data
    return None


def seg(u, t):  # tiempo del log en segundos desde el inicio del registro
    return (np.asarray(t, dtype=np.int64) - int(u.start_timestamp)) * 1e-6


def interp(t_new, t, y):
    return np.interp(t_new, t, y)


def tramos_nav(u, estado):
    st = topic(u, "vehicle_status")
    t, n = seg(u, st["timestamp"]), st["nav_state"]
    out, ini = [], None
    for ti, ni in zip(t, n):
        if ni == estado and ini is None:
            ini = ti
        if ni != estado and ini is not None:
            out.append((ini, ti)); ini = None
    if ini is not None:
        out.append((ini, t[-1]))
    return out


def resumen(u, ruta):
    gt, lp = topic(u, "vehicle_local_position_groundtruth"), topic(u, "vehicle_local_position")
    att, attgt = topic(u, "vehicle_attitude"), topic(u, "vehicle_attitude_groundtruth")
    res = {"dur_s": round((u.last_timestamp - u.start_timestamp) * 1e-6, 1), "bytes": Path(ruta).stat().st_size}
    # bytes por tema = número de muestras × tamaño del mensaje (como imprime ulog_info)
    tam = {}
    for d in u.data_list:
        size = sum(np.dtype(v.dtype).itemsize for k, v in d.data.items() if not k.startswith("_padding"))
        tam[d.name] = tam.get(d.name, 0) + len(d.data["timestamp"]) * size
    res["mayores_temas"] = sorted(tam.items(), key=lambda kv: -kv[1])[:8]
    rates = {}
    for n in ["sensor_combined", "vehicle_angular_velocity", "vehicle_attitude", "vehicle_local_position",
              "vehicle_local_position_groundtruth", "actuator_motors", "vehicle_rates_setpoint", "esc_status", "trajectory_setpoint"]:
        d = topic(u, n)
        if d is not None:
            tt = d["timestamp"]
            rates[n] = round((len(tt) - 1) / ((tt[-1] - tt[0]) * 1e-6), 1)
    res["tasas_en_log_hz"] = rates
    # Error de la estimación frente a la verdad de Gazebo, solo en vuelo (z < -1 m)
    te, tg = seg(u, lp["timestamp"]), seg(u, gt["timestamp"])
    m = (te > tg[0]) & (te < tg[-1]) & (lp["z"] < -1.0)
    e = {k: lp[k][m] - interp(te[m], tg, gt[k]) for k in "xyz"}
    res["err_pos_rms_cm"] = {k: round(float(np.sqrt(np.mean(v ** 2))) * 100, 1) for k, v in e.items()}
    # ¿Crece el error con la velocidad? EKF2 supone un retardo de GNSS (EKF2_GPS_DELAY) que Gazebo no simula
    vx, vy = interp(te[m], tg, gt["vx"]), interp(te[m], tg, gt["vy"])
    sp, eh = np.hypot(vx, vy), np.hypot(e["x"], e["y"])
    res["err_horiz_por_velocidad"] = []
    for a, b in [(0, 0.2), (0.2, 1), (1, 2), (2, 4), (4, 6)]:
        k = (sp >= a) & (sp < b)
        res["err_horiz_por_velocidad"].append({"v_min": a, "v_max": b, "n": int(k.sum()), "v_media": round(float(sp[k].mean()), 3),
                                              "err_medio_cm": round(float(eh[k].mean()) * 100, 1)})
    k = sp > 1
    tau = float(np.sum(e["x"][k] * vx[k] + e["y"][k] * vy[k]) / np.sum(sp[k] ** 2))
    res["ajuste_e_igual_v_tau"] = {"tau_s": round(tau, 4),
                                   "residuo_cm": round(float(np.mean(np.hypot(e["x"][k] - vx[k] * tau, e["y"][k] - vy[k] * tau))) * 100, 1),
                                   "EKF2_GPS_DELAY_ms": u.initial_parameters.get("EKF2_GPS_DELAY")}
    # Error de actitud: inclinación (ángulo entre los ejes z del cuerpo) y guiñada por separado
    ta, tag = seg(u, att["timestamp"]), seg(u, attgt["timestamp"])
    mm = (ta > tag[0]) & (ta < tag[-1])
    q = np.stack([att[f"q[{i}]"][mm] for i in range(4)], 1)
    qg = np.stack([interp(ta[mm], tag, attgt[f"q[{i}]"]) for i in range(4)], 1)
    qg /= np.linalg.norm(qg, axis=1, keepdims=True)

    def zaxis(qq):
        w, x, y, z = qq.T
        return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)

    def yaw(qq):
        w, x, y, z = qq.T
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    tilt = np.degrees(np.arccos(np.clip(np.sum(zaxis(q) * zaxis(qg), 1), -1, 1)))
    dyaw = np.degrees(np.angle(np.exp(1j * (yaw(q) - yaw(qg)))))
    ok = ta[mm] > ta[mm][0] + 5.0  # sin los primeros 5 s, mientras el EKF se alinea
    res["err_inclinacion_deg"] = {"rms": round(float(np.sqrt(np.mean(tilt[ok] ** 2))), 3), "max": round(float(tilt[ok].max()), 3)}
    res["err_guinada_deg"] = {"rms": round(float(np.sqrt(np.mean(dyaw[ok] ** 2))), 3), "max": round(float(np.abs(dyaw[ok]).max()), 3)}
    res["parametros"] = {k: u.initial_parameters.get(k) for k in
                         ["MPC_XY_P", "MPC_XY_VEL_P_ACC", "MPC_ACC_HOR", "MPC_JERK_AUTO", "MPC_XY_CRUISE", "NAV_ACC_RAD",
                          "COM_OF_LOSS_T", "IMU_GYRO_RATEMAX", "IMU_INTEG_RATE", "MPC_THR_HOVER", "SDLOG_PROFILE", "MAV_PROTO_VER"]}
    return res, (te[m], e), (ta[mm][ok], tilt[ok], dyaw[ok])


def escalon(u):
    ts, gt, lp = topic(u, "trajectory_setpoint"), topic(u, "vehicle_local_position_groundtruth"), topic(u, "vehicle_local_position")
    t = seg(u, ts["timestamp"]); x = ts["position[0]"]
    j = np.where(np.diff(x) > 4.0)[0]  # el salto de 5 m hacia el norte
    t0 = t[j[-1] + 1]
    tg = seg(u, gt["timestamp"]); te = seg(u, lp["timestamp"])
    rej = np.arange(t0 - 1.0, t0 + 8.0, 0.02)
    xsp = np.where(rej >= t0, x[j[-1] + 1], x[j[-1]])
    return rej - t0, xsp, interp(rej, tg, gt["x"]), interp(rej, te, lp["x"])


def circulos(u):
    off = tramos_nav(u, 14)  # NAVIGATION_STATE_OFFBOARD
    ts, gt = topic(u, "trajectory_setpoint"), topic(u, "vehicle_local_position_groundtruth")
    t, tg = seg(u, ts["timestamp"]), seg(u, gt["timestamp"])
    # dentro del tramo offboard, cada círculo empieza cuando la consigna se aleja del punto de partida
    a, b = off[0]
    m = (t > a) & (t < b)
    tt, xx, yy = t[m], ts["position[0]"][m], ts["position[1]"][m]
    x0, y0 = xx[0], yy[0]
    # los círculos son los huecos entre las esperas de 3 s en el punto de partida (la consigna pasa por él en cada vuelta,
    # así que no basta con mirar cuándo se aleja)
    cerca = np.hypot(xx - x0, yy - y0) < 0.05
    quietos, ini = [], None
    for k in range(len(tt)):
        if cerca[k] and ini is None:
            ini = tt[k]
        if not cerca[k] and ini is not None:
            if tt[k] - ini > 2.0:
                quietos.append((ini, tt[k - 1]))
            ini = None
    if ini is not None:
        quietos.append((ini, tt[-1]))
    tramos = [(quietos[k][1], quietos[k + 1][0]) for k in range(len(quietos) - 1) if quietos[k + 1][0] - quietos[k][1] > 30]
    filas = []
    for nombre, (ini, fin) in zip(["con_ff", "sin_ff"], tramos[:2]):
        rej = np.arange(ini, fin, 0.1)
        sx, sy = interp(rej, tt, xx), interp(rej, tt, yy)
        gx, gy = interp(rej, tg, gt["x"]), interp(rej, tg, gt["y"])
        for r, a1, a2, a3, a4 in zip(rej - ini, sx, sy, gx, gy):
            filas.append((nombre, round(r, 2), round(a1 - x0, 4), round(a2 - y0, 4), round(a3 - x0, 4), round(a4 - y0, 4)))
    return filas


def instante_parametro(u, nombre, valor):
    for t, n, v in u.changed_parameters:
        if n == nombre and abs(v - valor) < 1e-6 and t > u.start_timestamp:
            return (int(t) - int(u.start_timestamp)) * 1e-6
    raise ValueError(nombre)


def instante_mensaje(u, texto):
    for m in u.logged_messages:
        if texto in m.message:
            return (int(m.timestamp) - int(u.start_timestamp)) * 1e-6
    raise ValueError(texto)


def fallo_gnss(u):
    """EKF2_GPS_CTRL = 0 en estacionario: error real frente a eph y cambios de modo."""
    t_off = instante_parametro(u, "EKF2_GPS_CTRL", 0)
    gt, lp, st = topic(u, "vehicle_local_position_groundtruth"), topic(u, "vehicle_local_position"), topic(u, "vehicle_status")
    tg, te, tst = seg(u, gt["timestamp"]), seg(u, lp["timestamp"]), seg(u, st["timestamp"])
    rej = np.arange(t_off - 5.0, min(te[-1], t_off + 34.0), 0.25)
    ex = interp(rej, te, lp["x"]) - interp(rej, tg, gt["x"])
    ey = interp(rej, te, lp["y"]) - interp(rej, tg, gt["y"])
    zg = interp(rej, tg, gt["z"])
    nav = np.array([st["nav_state"][max(0, np.searchsorted(tst, r) - 1)] for r in rej])
    xyv = np.array([lp["xy_valid"][max(0, np.searchsorted(te, r) - 1)] for r in rej])
    eph = interp(rej, te, lp["eph"])
    msgs = [(round((int(m.timestamp) - int(u.start_timestamp)) * 1e-6 - t_off, 3), m.message) for m in u.logged_messages
            if t_off - 1 < (int(m.timestamp) - int(u.start_timestamp)) * 1e-6 < t_off + 40]
    return rej - t_off, ex, ey, eph, zg, nav, xyv, msgs


def fallo_motor(u):
    """Un motor apagado en estacionario a 10 m: inclinación y altura frente al tiempo."""
    t_f = instante_mensaje(u, "motor 1 off")
    attgt, gt = topic(u, "vehicle_attitude_groundtruth"), topic(u, "vehicle_local_position_groundtruth")
    ta, tg = seg(u, attgt["timestamp"]), seg(u, gt["timestamp"])
    m = (ta > t_f - 1) & (ta < t_f + 6)
    w, x, y, z = (attgt[f"q[{i}]"][m] for i in range(4))
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    alt = -interp(ta[m], tg, gt["z"])
    return ta[m] - t_f, tilt, alt


def main():
    v, c = sys.argv[1], sys.argv[2]
    u = ULog(v)
    res, (te, e), (ta, tilt, dyaw) = resumen(u, v)
    json.dump(res, open(DATOS / "ulog_resumen.json", "w"), indent=1)
    with open(DATOS / "error_ekf.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s", "ex_m", "ey_m", "ez_m"])
        for k in range(0, len(te), 25):  # 125 Hz -> 5 Hz
            w.writerow([round(te[k], 2), round(e["x"][k], 4), round(e["y"][k], 4), round(e["z"][k], 4)])
    t, xsp, xg, xe = escalon(u)
    with open(DATOS / "escalon.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s", "x_consigna", "x_verdad", "x_ekf"])
        w.writerows([(round(a, 3), round(b, 4), round(cc, 4), round(d, 4)) for a, b, cc, d in zip(t, xsp, xg, xe)])
    uc = ULog(c)
    filas = circulos(uc)
    with open(DATOS / "circulos.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["caso", "t_s", "sp_n", "sp_e", "gt_n", "gt_e"]); w.writerows(filas)
    ug = ULog(sys.argv[3])
    t, ex, ey, eph, zg, nav, xyv, msgs = fallo_gnss(ug)
    with open(DATOS / "fallo_gnss.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_rel_s", "ex_m", "ey_m", "eph_m", "z_verdad_m", "nav_state", "xy_valid"])
        for k in range(len(t)):
            w.writerow([round(t[k], 2), round(ex[k], 3), round(ey[k], 3), round(eph[k], 3), round(zg[k], 3), int(nav[k]), int(xyv[k])])
    json.dump(msgs, open(DATOS / "fallo_gnss_mensajes.json", "w"), indent=1, ensure_ascii=False)
    um = ULog(sys.argv[4])
    t, tilt, alt = fallo_motor(um)
    with open(DATOS / "fallo_motor.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_rel_s", "inclinacion_deg", "altura_m"])
        for k in range(0, len(t), 2):
            w.writerow([round(t[k], 3), round(tilt[k], 2), round(alt[k], 3)])
    print(json.dumps(res, indent=1))
    print(msgs)


if __name__ == "__main__":
    main()
