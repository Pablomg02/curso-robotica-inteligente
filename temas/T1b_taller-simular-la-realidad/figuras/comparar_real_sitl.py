"""El mismo x500 en estacionario: un vuelo real público (Flight Review) frente a PX4 SITL + Gazebo.

    uv run --with pyulog --with numpy python comparar_real_sitl.py <real_estacionario.ulg> <real_fifo.ulg> <sitl.ulg>

real_estacionario.ulg: registro público f4f9b50c-3b1a-47ea-b7ce-eb8fa46e0d3c (Holybro X500 V2, FMU-V6X, PX4 v1.17.0,
modo Position, 43 s quieto a 1,6 m). real_fifo.ulg: c8082ca8-c79b-4e9b-814a-4c1454546181 (mismo dron y versión, con el
giróscopo en bruto a 8,4 kHz en sensor_gyro_fifo y las RPM en esc_status). Descarga: https://logs.px4.io/download?log=<id>
sitl.ulg: el vuelo de figuras/experimentos de T1a con 45 s en Hold a 5 m (mavsdk_circulo_fallo.py).
Escribe figuras/datos/real_sitl.json y figuras/datos/psd_giro.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from pyulog import ULog

DATOS = Path(__file__).parent / "datos"


def topic(u, n, i=0):
    for d in u.data_list:
        if d.name == n and d.multi_id == i:
            return d.data
    return None


def seg(u, t):
    return (np.asarray(t, dtype=np.int64) - int(u.start_timestamp)) * 1e-6


def ventana_estacionaria(u, estados, dur=20.0):
    """Primer tramo de `dur` s en uno de los modos dados, en el aire y con |v| < 0,3 m/s."""
    lp, st = topic(u, "vehicle_local_position"), topic(u, "vehicle_status")
    t, ts = seg(u, lp["timestamp"]), seg(u, st["timestamp"])
    nav = st["nav_state"][np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 1)]
    ok = np.isin(nav, estados) & (np.hypot(lp["vx"], lp["vy"]) < 0.3) & (np.abs(lp["vz"]) < 0.2) & (lp["z"] < -1.0)
    ini = None
    for k in range(len(t)):
        if ok[k] and ini is None:
            ini = t[k]
        if not ok[k]:
            ini = None
        if ini is not None and t[k] - ini >= dur:
            return ini, t[k]
    raise RuntimeError("no hay un estacionario de %.0f s" % dur)


def welch(x, fs, nseg):
    """Densidad espectral de potencia unilateral por el método de Welch (Hann, solape 50 %)."""
    x = np.asarray(x, float) - np.mean(x)
    w = np.hanning(nseg); paso = nseg // 2
    U = np.sum(w ** 2)
    P = None; n = 0
    for a in range(0, len(x) - nseg + 1, paso):
        X = np.fft.rfft(x[a:a + nseg] * w)
        p = np.abs(X) ** 2 / (fs * U)
        P = p if P is None else P + p; n += 1
    P = P / n; P[1:-1] *= 2
    return np.fft.rfftfreq(nseg, 1 / fs), P, n


def errores(u, a, b):
    att, sp = topic(u, "vehicle_attitude"), topic(u, "vehicle_attitude_setpoint")
    ta, tsp = seg(u, att["timestamp"]), seg(u, sp["timestamp"])
    m = (ta > a) & (ta < b)

    def rp(q):
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        return np.degrees(roll), np.degrees(pitch)

    r, p = rp([att[f"q[{i}]"][m] for i in range(4)])
    qd = [np.interp(ta[m], tsp, sp[f"q_d[{i}]"]) for i in range(4)]
    rd, pd = rp(qd)
    thr = -np.interp(ta[m], tsp, sp["thrust_body[2]"])
    lp = topic(u, "vehicle_local_position"); tl = seg(u, lp["timestamp"]); ml = (tl > a) & (tl < b)
    av = topic(u, "vehicle_angular_velocity"); tv = seg(u, av["timestamp"]); mv = (tv > a) & (tv < b)
    return {"err_roll_rms_deg": round(float(np.sqrt(np.mean((r - rd) ** 2))), 3),
            "err_pitch_rms_deg": round(float(np.sqrt(np.mean((p - pd) ** 2))), 3),
            "empuje_medio": round(float(thr.mean()), 3),
            "pos_std_cm": [round(float(np.std(lp[k][ml])) * 100, 1) for k in "xyz"],
            "rate_std_deg_s": [round(float(np.degrees(np.std(av[f"xyz[{i}]"][mv]))), 2) for i in range(3)]}


def main():
    ur, uf, us = ULog(sys.argv[1]), ULog(sys.argv[2]), ULog(sys.argv[3])
    out = {}
    # --- real: estacionario limpio para errores, empuje y ruido ---
    a, b = ventana_estacionaria(ur, [2])  # POSCTL
    sc = topic(ur, "sensor_combined"); tsc = seg(ur, sc["timestamp"]); msc = (tsc > a) & (tsc < b)
    ht = topic(ur, "hover_thrust_estimate"); th = seg(ur, ht["timestamp"]); mh = (th > a) & (th < b)
    out["real"] = {"log": "f4f9b50c-3b1a-47ea-b7ce-eb8fa46e0d3c", "ventana_s": [round(a, 1), round(b, 1)],
                   "fs_sensor_combined_log_hz": round(float(1 / np.median(np.diff(tsc[msc]))), 1),
                   "IMU_GYRO_RATEMAX": ur.initial_parameters.get("IMU_GYRO_RATEMAX"), "IMU_INTEG_RATE": ur.initial_parameters.get("IMU_INTEG_RATE"),
                   "MPC_THR_HOVER": ur.initial_parameters.get("MPC_THR_HOVER"), "hover_thrust_estimado": round(float(ht["hover_thrust"][mh].mean()), 3),
                   "acc_std_m_s2": [round(float(np.std(sc[f"accelerometer_m_s2[{i}]"][msc])), 3) for i in range(3)],
                   "giro_std_deg_s": round(float(np.degrees(np.std(sc["gyro_rad[0]"][msc]))), 3)}
    # --- real con FIFO: espectro de vibración y RPM ---
    lp = topic(uf, "vehicle_local_position"); tl = seg(uf, lp["timestamp"]); aire = tl[lp["z"] < -0.5]
    af, bf = float(aire[0]) + 5.0, float(aire[0]) + 35.0
    fifo = topic(uf, "sensor_gyro_fifo"); tf = seg(uf, fifo["timestamp_sample"])
    m = (tf > af) & (tf < bf)
    n_s = int(np.median(fifo["samples"])); dt = float(np.median(fifo["dt"])) * 1e-6; fs_r = 1 / dt
    gx = np.stack([fifo[f"x[{j}]"][m] for j in range(n_s)], 1).ravel() * float(np.median(fifo["scale"]))
    fr, Pr, nr = welch(gx, fs_r, 8192)
    esc = topic(uf, "esc_status"); te = seg(uf, esc["timestamp"]); me = (te > af) & (te < bf)
    rpm = float(np.mean([esc[f"esc[{i}].esc_rpm"][me].mean() for i in range(4)]))
    k = (fr > 40) & (fr < 400)
    out["real_fifo"] = {"log": "c8082ca8-c79b-4e9b-814a-4c1454546181", "ventana_s": [round(af, 1), round(bf, 1)], "fs_fifo_hz": round(fs_r, 1),
                        "muestras_por_mensaje": n_s, "rpm_media": round(rpm, 0), "f_rotor_hz": round(rpm / 60, 1), "f_pala_hz": round(2 * rpm / 60, 1),
                        "pico_hz": round(float(fr[k][np.argmax(Pr[k])]), 1), "giro_std_deg_s_fifo": round(float(np.degrees(np.std(gx))), 2),
                        "IMU_GYRO_DNF_EN": uf.initial_parameters.get("IMU_GYRO_DNF_EN")}
    out["real"].update(errores(ur, a, b))
    # pico de vibración entre 40 y 400 Hz
    # --- SITL ---
    a2, b2 = ventana_estacionaria(us, [4], 20.0)  # AUTO_LOITER (Hold)
    sc2 = topic(us, "sensor_combined"); t2 = seg(us, sc2["timestamp"]); m2 = (t2 > a2) & (t2 < b2)
    fs_s = 1 / np.median(np.diff(t2[m2]))
    fs_, Ps, ns = welch(sc2["gyro_rad[0]"][m2], fs_s, 256)
    out["sitl"] = {"ventana_s": [round(a2, 1), round(b2, 1)], "fs_giro_hz": round(float(fs_s), 1),
                   "IMU_GYRO_RATEMAX": us.initial_parameters.get("IMU_GYRO_RATEMAX"), "IMU_INTEG_RATE": us.initial_parameters.get("IMU_INTEG_RATE"),
                   "MPC_THR_HOVER": us.initial_parameters.get("MPC_THR_HOVER"),
                   "acc_std_m_s2": [round(float(np.std(sc2[f"accelerometer_m_s2[{i}]"][m2])), 3) for i in range(3)],
                   "giro_std_deg_s": round(float(np.degrees(np.std(sc2["gyro_rad[0]"][m2]))), 3)}
    out["sitl"].update(errores(us, a2, b2))
    # Densidad del ruido blanco del SDF (stddev 0,0008726646 rad/s por muestra a 250 Hz): S = 2 σ² / fs (unilateral)
    sig = 0.0008726646
    out["sitl"]["psd_teorica_sdf"] = 2 * sig ** 2 / fs_s
    out["sitl"]["psd_media_medida"] = float(np.mean(Ps[(fs_ > 5) & (fs_ < 120)]))
    # sensor_combined no es la muestra del sensor: el integrador de PX4 (Integrator.hpp) usa la regla del trapecio, y con
    # una muestra por integración eso es la media de dos muestras seguidas: |H(f)|² = cos²(pi f / fs). Banda 20-50 Hz:
    # por encima del movimiento del dron y con cos² cerca de 1.
    kb = (fs_ >= 20) & (fs_ <= 50)
    c2 = np.cos(np.pi * fs_[kb] / fs_s) ** 2
    out["sitl"]["psd_banda_20_50"] = float(np.mean(Ps[kb]))
    out["sitl"]["cos2_medio_20_50"] = float(np.mean(c2))
    out["sitl"]["psd_banda_20_50_corregida"] = float(np.mean(Ps[kb] / c2))
    json.dump(out, open(DATOS / "real_sitl.json", "w"), indent=1)
    # PSD en bandas logarítmicas para la figura
    with open(DATOS / "psd_giro.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fuente", "f_hz", "psd_rad2_s2_hz"])
        for nombre, f_, P in [("real", fr, Pr), ("sitl", fs_, Ps)]:
            bordes = np.logspace(np.log10(1.0), np.log10(f_[-1]), 70)
            for lo, hi in zip(bordes[:-1], bordes[1:]):
                kk = (f_ >= lo) & (f_ < hi)
                if kk.any():
                    w.writerow([nombre, round(float(np.sqrt(lo * hi)), 2), float(P[kk].mean())])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
