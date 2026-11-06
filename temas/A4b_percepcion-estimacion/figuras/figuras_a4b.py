"""Datos de las figuras y tablas del tema A4b (percepción para estimación: VIO, SLAM, eventos, filtros aprendidos).

Uso:  python3 figuras_a4b.py > salida.txt      (numpy; menos de un minuto)
"""
import math
import numpy as np


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


print("=" * 70)
print("1. DESFASE TEMPORAL CÁMARA-IMU: error de reproyección equivalente = f * omega * td (cam0 de EuRoC, f = 458,65 px)")
# Intrínsecos y extrínsecos de EuRoC según rpng/open_vins, config/euroc_mav/kalibr_imucam_chain.yaml
f_px = 458.654
p_c0 = np.array([-0.0216401454975, -0.064676986768, 0.00981073058949])
p_c1 = np.array([-0.0198435579556, 0.0453689425024, 0.00786212447038])
BASE = float(np.linalg.norm(p_c1 - p_c0))
for w in (0.5, 2.0, 5.0):
    fila = [f"td = {td*1000:4.0f} ms -> {f_px*w*td:6.1f} px" for td in (0.001, 0.005, 0.01, 0.03)]
    print(f"  omega = {w:.1f} rad/s: " + "; ".join(fila))
for v in (2.0, 10.0):
    print(f"  traslación: a {v:.0f} m/s, 10 ms de desfase = {100*v*0.01:.0f} cm de posición mal asignada")

print("=" * 70)
print("2. ESTIMAR EL DESFASE EN LÍNEA: IMU (acelerómetro a 200 Hz) + posición visual a 20 Hz con desfase td")
dt, Tsim = 0.005, 40.0
TD_REAL = 0.030                                  # la cámara marca sus imágenes 30 ms más tarde de lo que las tomó
sa, sp = 0.05, 0.02                              # ruido del acelerómetro (m/s^2 por muestra) y de la posición visual (m)


def tray(t):
    # lemniscata 3D suave, velocidades de hasta unos 3 m/s
    w = 0.8
    p = np.array([3 * np.sin(w * t), 1.5 * np.sin(2 * w * t), 0.5 * np.sin(0.5 * w * t)])
    v = np.array([3 * w * np.cos(w * t), 3 * w * np.cos(2 * w * t), 0.25 * w * np.cos(0.5 * w * t)])
    a = np.array([-3 * w**2 * np.sin(w * t), -6 * w**2 * np.sin(2 * w * t), -0.125 * w**2 * np.sin(0.5 * w * t)])
    return p, v, a


def filtro(estimar_td, td_supuesto=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = 7 if estimar_td else 6                   # (p, v) y, si se estima, el desfase td
    x = np.zeros(n); p0, v0, _ = tray(0.0); x[0:3], x[3:6] = p0, v0
    if estimar_td:
        x[6] = td_supuesto
    P = np.diag([0.01] * 3 + [0.01] * 3 + ([0.05**2] if estimar_td else []))
    Fm = np.eye(n); Fm[0:3, 3:6] = np.eye(3) * dt
    Qd = np.zeros((n, n)); Qd[3:6, 3:6] = np.eye(3) * (sa**2) * dt**2 * 50; Qd[0:3, 0:3] = np.eye(3) * 1e-8
    if estimar_td:
        Qd[6, 6] = 1e-10
    errs, tds, ts = [], [], []
    for i in range(int(Tsim / dt)):
        t = i * dt
        _, _, a = tray(t)
        am = a + rng.normal(0, sa, 3)
        x[0:3] += x[3:6] * dt + 0.5 * am * dt**2; x[3:6] += am * dt
        P = Fm @ P @ Fm.T + Qd
        if i % 10 == 0 and t > TD_REAL:          # imagen a 20 Hz: se tomó en t - TD_REAL, pero llega marcada como t
            p_med = tray(t - TD_REAL)[0] + rng.normal(0, sp, 3)
            if estimar_td:
                h = x[0:3] - x[3:6] * x[6]       # p(t - td) ~ p(t) - v(t) td
                H = np.zeros((3, n)); H[:, 0:3] = np.eye(3); H[:, 3:6] = -np.eye(3) * x[6]; H[:, 6] = -x[3:6]
            else:
                h = x[0:3] - x[3:6] * td_supuesto
                H = np.zeros((3, n)); H[:, 0:3] = np.eye(3); H[:, 3:6] = -np.eye(3) * td_supuesto
            S = H @ P @ H.T + np.eye(3) * sp**2
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (p_med - h)
            IKH = np.eye(n) - K @ H
            P = IKH @ P @ IKH.T + K @ (np.eye(3) * sp**2) @ K.T
        if i % 20 == 0:
            errs.append(np.linalg.norm(x[0:3] - tray(t)[0])); ts.append(t)
            tds.append(x[6] if estimar_td else float("nan"))
    return np.array(ts), np.array(errs), np.array(tds)


res = {}
for nombre, est, tds0 in (("ignora el desfase", False, 0.0), ("estima el desfase", True, 0.0), ("desfase conocido", False, TD_REAL)):
    ts, e, tdh = filtro(est, tds0)
    res[nombre] = (ts, e, tdh)
    k = ts > 10
    msg = f"  {nombre:18s}: error de posición RMS (t > 10 s) {100*np.sqrt(np.mean(e[k]**2)):5.2f} cm"
    if est:
        i90 = np.argmax(np.abs(tdh - TD_REAL) < 0.1 * TD_REAL)
        msg += f"; td estimado final {1000*tdh[-1]:.1f} ms (real 30 ms), al 10 % a los {ts[i90]:.1f} s"
    print(msg)
X2 = lin(0, Tsim, 60, 500)
Y2 = lin(0, 12, 250, 30)
for nombre in ("ignora el desfase", "estima el desfase", "desfase conocido"):
    ts, e, _ = res[nombre]
    print(f"path err {nombre}:", path(ts, np.minimum(100 * e, 12), X2, Y2))
ts, _, tdh = res["estima el desfase"]
Ytd = lin(0, 40, 250, 30)
print("path td est:", path(ts, 1000 * np.clip(tdh, 0, 0.04), X2, Ytd))
print(f"  línea td real: y = {Ytd(30):.1f}")

print("=" * 70)
print(f"3. ESTÉREO: incertidumbre de profundidad sigma_z = z^2 sigma_d / (f b) (f = 458,65 px, sigma_d = 0,5 px); base de EuRoC = {100*BASE:.1f} cm")
for b in (0.05, BASE, 0.30):
    fila = [f"z = {z:4.1f} m -> {100*z**2*0.5/(f_px*b):6.1f} cm" for z in (1, 3, 10, 30)]
    print(f"  base {100*b:4.0f} cm: " + "; ".join(fila))
