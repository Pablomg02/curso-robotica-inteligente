"""Una «cámara neuronal» para PX4 SITL: la pose del x500 de Gazebo, leída por MAVSDK, se renderiza en una escena 3DGS con gsplat.

Mide (1) el coste del render frente a la resolución y al número de gaussianas, sin simulador, y (2) el bucle en vivo a 30 Hz:
tiempo de render y edad de la pose cuando la imagen está lista. No guarda imágenes (la escena preentrenada de 3DGS
solo permite uso de investigación no comercial): solo cifras.

    python camara_neuronal.py <point_cloud.ply> <salida.json> [--sin-vuelo]

Requisitos: torch 2.4 + CUDA 12.4, gsplat 1.5.3 (rueda precompilada), mavsdk 4.0.5, numpy; PX4 SITL + Gazebo en marcha
(gz_x500) escuchando en el puerto 14540. Escena: garden de los modelos preentrenados de 3DGS (ver el laboratorio de B3b).
"""
import json
import math
import sys
import threading
import time

import numpy as np
import torch
from gsplat import rasterization

DEV = "cuda"


def leer_ply(ruta):
    """Lee un point_cloud.ply de 3DGS (binario, little endian) sin dependencias extra."""
    with open(ruta, "rb") as f:
        cab = []
        while True:
            l = f.readline().decode().strip()
            cab.append(l)
            if l == "end_header":
                break
        n = int([l for l in cab if l.startswith("element vertex")][0].split()[-1])
        props = [l.split()[-1] for l in cab if l.startswith("property")]
        datos = np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    c = {p: i for i, p in enumerate(props)}
    xyz = datos[:, [c["x"], c["y"], c["z"]]]
    esc = np.exp(datos[:, [c["scale_0"], c["scale_1"], c["scale_2"]]])
    q = datos[:, [c["rot_0"], c["rot_1"], c["rot_2"], c["rot_3"]]]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    op = 1 / (1 + np.exp(-datos[:, c["opacity"]]))
    dc = datos[:, [c["f_dc_0"], c["f_dc_1"], c["f_dc_2"]]][:, None, :]
    resto = sorted([p for p in props if p.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    rest = datos[:, [c[p] for p in resto]].reshape(n, 3, -1).transpose(0, 2, 1)
    sh = np.concatenate([dc, rest], 1)
    t = lambda a: torch.tensor(a, device=DEV)
    return t(xyz), t(q), t(esc), t(op), t(sh), len(props)


def vista(R, p):
    """Matriz mundo→cámara 4×4 a partir de la orientación R (cámara→mundo) y la posición p."""
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R.T
    V[:3, 3] = -R.T @ p
    return V


def render(G, V, W, H, f):
    xyz, q, esc, op, sh = G
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=DEV, dtype=torch.float32)[None]
    img, alfa, _ = rasterization(xyz, q, esc, op, sh, torch.tensor(V, device=DEV)[None], K, W, H, sh_degree=3)
    return img


def cronometrar(G, V, W, H, f, n=50):
    for _ in range(5):
        render(G, V, W, H, f)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a = time.perf_counter(); render(G, V, W, H, f); torch.cuda.synchronize(); ts.append((time.perf_counter() - a) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ruta, salida = sys.argv[1], sys.argv[2]
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    t0 = time.perf_counter()
    xyz, q, esc, op, sh, nprop = leer_ply(ruta)
    torch.cuda.synchronize()
    res["n_gaussianas"] = int(xyz.shape[0]); res["floats_por_gaussiana"] = nprop - 3  # sin las normales nx, ny, nz
    res["carga_s"] = round(time.perf_counter() - t0, 2)
    res["memoria_parametros_mb"] = round(sum(t.numel() * 4 for t in (xyz, q, esc, op, sh)) / 1e6, 1)
    # Cámara de referencia: en el centro de la escena, a una distancia del orden de su extensión, mirando al centro
    cen = xyz.median(0).values.cpu().numpy(); ext = float((xyz - torch.tensor(cen, device=DEV)).norm(dim=1).median())
    p_ref = cen + np.array([0.0, -0.5 * ext, -1.0 * ext], np.float32)

    def mirar(p, objetivo):
        z = (objetivo - p); z /= np.linalg.norm(z)
        x = np.cross(z, np.array([0, -1.0, 0], np.float32)); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.stack([x, y, z], 1).astype(np.float32)

    V0 = vista(mirar(p_ref, cen), p_ref)
    # (1) Coste del render sin simulador
    res["coste"] = []
    for W, H in [(320, 240), (640, 480), (1280, 960)]:
        for frac in [0.25, 0.5, 1.0]:
            k = int(frac * xyz.shape[0])
            G = (xyz[:k], q[:k], esc[:k], op[:k], sh[:k])
            med, p95 = cronometrar(G, V0, W, H, 0.9 * W)
            res["coste"].append({"W": W, "H": H, "fraccion": frac, "n": k, "ms_mediana": round(med, 2), "ms_p95": round(p95, 2)})
            print(res["coste"][-1], flush=True)
    res["memoria_gpu_pico_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 0)
    if "--sin-vuelo" in sys.argv:
        json.dump(res, open(salida, "w"), indent=1); return
    # (2) Bucle en vivo con PX4 SITL: MAVSDK da la pose; el render va a 30 Hz con la última pose recibida
    from mavsdk import Mavsdk, Configuration, ComponentType
    from mavsdk.plugins.action import Action
    from mavsdk.plugins.offboard import Offboard, PositionNedYaw, VelocityNedYaw
    from mavsdk.plugins.telemetry import Telemetry
    mav = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
    mav.add_any_connection("udpin://0.0.0.0:14540"); d = mav.first_autopilot(30.0)
    tel, act, off = Telemetry(d), Action(d), Offboard(d)
    while not tel.health_all_ok():
        time.sleep(0.1)
    tel.set_rate_position_velocity_ned(50.0); tel.set_rate_attitude_quaternion(50.0)
    pose = {"p": None, "q": None, "t": 0.0, "n": 0}
    lock = threading.Lock()

    def cb_p(pv, *a):
        with lock:
            pose["p"] = np.array([pv.position.north_m, pv.position.east_m, pv.position.down_m], np.float32); pose["t"] = time.perf_counter(); pose["n"] += 1

    def cb_q(qq, *a):
        with lock:
            pose["q"] = np.array([qq.w, qq.x, qq.y, qq.z], np.float32)

    h1 = tel.subscribe_position_velocity_ned(cb_p); h2 = tel.subscribe_attitude_quaternion(cb_q)
    act.set_takeoff_altitude(5.0)
    for _ in range(20):
        try:
            act.arm(); break
        except Exception:
            time.sleep(1.0)
    act.takeoff()
    while pose["p"] is None or pose["p"][2] > -4.75:
        time.sleep(0.05)
    time.sleep(3.0)
    n0, e0 = float(pose["p"][0]), float(pose["p"][1])
    stop = threading.Event()

    def circulo():
        for _ in range(20):
            off.set_position_ned(PositionNedYaw(n0, e0, -5.0, 0.0)); time.sleep(0.05)
        off.start(); R, T = 5.0, 20.0; w = 2 * math.pi / T; tc = time.perf_counter(); k = 0
        while not stop.is_set():
            t = time.perf_counter() - tc
            off.set_position_velocity_ned(PositionNedYaw(n0 + R * math.sin(w * t), e0 + R * (1 - math.cos(w * t)), -5.0, 0.0),
                                          VelocityNedYaw(R * w * math.cos(w * t), R * w * math.sin(w * t), 0.0, 0.0))
            k += 1; time.sleep(max(0.0, tc + k * 0.05 - time.perf_counter()))

    hilo = threading.Thread(target=circulo, daemon=True); hilo.start()
    time.sleep(2.0)
    # mundo NED del dron → escena: 1 m de vuelo = 0,05·ext unidades de la escena (escala elegida, no medida)
    s = 0.05 * ext
    G = (xyz, q, esc, op, sh); W, H = 640, 480
    reg = []; tini = time.perf_counter(); k = 0; n_ini = pose["n"]
    while time.perf_counter() - tini < 30.0:
        with lock:
            p, qq, tp = pose["p"].copy(), pose["q"].copy(), pose["t"]
        a = time.perf_counter()
        pe = p_ref + s * np.array([p[1] - e0, -(p[2] + 5.0), p[0] - n0], np.float32)  # este→x, arriba→−y, norte→z
        V = vista(mirar(pe, pe + (cen - p_ref)), pe)
        img = render(G, V, W, H, 0.9 * W); torch.cuda.synchronize()
        b = time.perf_counter()
        reg.append({"render_ms": (b - a) * 1e3, "edad_pose_ms": (b - tp) * 1e3})
        k += 1; time.sleep(max(0.0, tini + k / 30.0 - time.perf_counter()))
    stop.set(); hilo.join(1.0)
    dur = time.perf_counter() - tini
    r = np.array([x["render_ms"] for x in reg]); e = np.array([x["edad_pose_ms"] for x in reg])
    res["vivo"] = {"W": W, "H": H, "imagenes": len(reg), "hz_logrados": round(len(reg) / dur, 2),
                   "render_ms_mediana": round(float(np.median(r)), 2), "render_ms_p95": round(float(np.percentile(r, 95)), 2),
                   "edad_pose_ms_mediana": round(float(np.median(e)), 2), "edad_pose_ms_p95": round(float(np.percentile(e, 95)), 2),
                   "edad_pose_ms_max": round(float(e.max()), 2), "poses_recibidas_hz": round((pose["n"] - n_ini) / dur, 1)}
    print(res["vivo"], flush=True)
    try:
        off.stop()
    except Exception:
        pass
    act.land()
    tel.unsubscribe_position_velocity_ned(h1); tel.unsubscribe_attitude_quaternion(h2)
    json.dump(res, open(salida, "w"), indent=1)


if __name__ == "__main__":
    main()
