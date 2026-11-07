"""Datos de las figuras y tablas del tema B3b (render y real2sim).

Uso:  python3 figuras_b3b.py > salida.txt      (numpy; unos segundos)
"""
import math
import numpy as np


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


# ---------------------------------------------------------------- 1. render volumétrico (NeRF) a lo largo de un rayo
print("=" * 70)
print("1. RENDER VOLUMÉTRICO A LO LARGO DE UN RAYO: CRISTAL SEMITRANSPARENTE EN t = 2 m Y PARED EN t = 4 m")


def sigma(t):
    vidrio = 1.2 * np.exp(-0.5 * ((t - 2.0) / 0.08) ** 2)      # integral ≈ 0,24 -> deja pasar ~79 %
    pared = 60.0 / (1 + np.exp(-(t - 4.0) / 0.02))               # opaca a partir de 4 m
    return vidrio + pared


def color(t):
    return np.where(t < 3.0, 0.9, 0.3)   # cristal claro (0,9), pared oscura (0,3)


def render(n, t0=0.5, t1=6.0, rng=None):
    if rng is None:
        tt = np.linspace(t0, t1, n + 1)
    else:
        tt = np.sort(np.concatenate([[t0, t1], rng.uniform(t0, t1, n - 1)]))
    tm = 0.5 * (tt[1:] + tt[:-1])
    dl = np.diff(tt)
    a = 1 - np.exp(-sigma(tm) * dl)
    T = np.concatenate([[1.0], np.cumprod(1 - a)[:-1]])
    w = T * a
    return (w * color(tm)).sum(), (w * tm).sum() / w.sum(), w.sum(), tm, w


C_ref, D_ref, A_ref, _, _ = render(200000)
print(f"  referencia (200 000 muestras): color {C_ref:.4f}, profundidad esperada {D_ref:.3f} m, opacidad acumulada {A_ref:.4f}")
trans_vidrio = math.exp(-1.2 * 0.08 * math.sqrt(2 * math.pi))
print(f"  transmitancia del cristal exp(-integral sigma) = {trans_vidrio:.3f}; color esperado = {(1-trans_vidrio)*0.9 + trans_vidrio*0.3:.4f}")
print(f"  profundidad esperada analítica = {(1-trans_vidrio)*2.0 + trans_vidrio*4.0:.3f} m (ni el cristal ni la pared: un punto en el aire)")
for n in (16, 32, 64, 128, 256):
    c, d, a, _, _ = render(n)
    print(f"  {n:4d} muestras uniformes: color {c:.4f} (error {abs(c-C_ref):.4f}), profundidad {d:.3f} m")
rng = np.random.default_rng(0)
errs = [abs(render(64, rng=rng)[0] - C_ref) for _ in range(200)]
print(f"  64 muestras al azar (uniformes, sin estratificar): error medio {np.mean(errs):.4f}")
tt = np.linspace(0.5, 6.0, 1101)
X1 = lin(0.5, 6.0, 60, 520)
s = sigma(tt)
Ys = lin(0, 3.0, 250, 30)
print("path rayo sigma:", path(tt, np.clip(s, 0, 3.0), X1, Ys))
Tt = np.exp(-np.concatenate([[0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(tt))]))
Yt = lin(0, 1.0, 250, 30)
print("path rayo T:", path(tt, Tt, X1, Yt))
_, _, _, tm, w = render(64)
print("  pesos con 64 muestras (barras): " + "; ".join(f"{X1(x):.1f},{Yt(min(1,y*4)):.1f}" for x, y in zip(tm, w) if y > 0.002))
print(f"  marcas: x(1) = {X1(1):.0f}, x(2) = {X1(2):.0f}, x(3) = {X1(3):.0f}, x(4) = {X1(4):.0f}, x(5) = {X1(5):.0f}, x(6) = {X1(6):.0f}")
print(f"  profundidad esperada x = {X1(D_ref):.1f}")

# ---------------------------------------------------------------- 2. proyección de gaussianas 3D (splatting)
print("=" * 70)
print("2. PROYECCIÓN DE GAUSSIANAS 3D: Sigma' = J W Sigma W^T J^T")
rng = np.random.default_rng(2)
F, W_IM, H_IM = 300.0, 400, 260
cx, cy = W_IM / 2, H_IM / 2


def quat_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


gaus = []
# suelo: gaussianas planas en z = 0 (mundo z arriba)
for i in range(60):
    p = np.array([rng.uniform(-6, 6), rng.uniform(4, 16), 0.0])
    S = np.diag([0.6, 0.6, 0.02])
    gaus.append((p, S, np.eye(3), "suelo", 0.55))
# dos árboles: tronco y copa
for (xt, yt) in [(-1.5, 8.0), (2.5, 11.0)]:
    for k in range(4):
        gaus.append((np.array([xt, yt, 0.4 + 0.6 * k]), np.diag([0.08, 0.08, 0.35]), np.eye(3), "tronco", 0.9))
    for k in range(25):
        p = np.array([xt, yt, 3.0]) + rng.normal(0, [0.6, 0.6, 0.5])
        q = rng.normal(size=4)
        S = np.diag(rng.uniform(0.15, 0.45, 3))
        gaus.append((p, S, quat_rot(q), "copa", 0.7))
# cámara en (0, 0, 1,5) mirando a +y, con cuerpo x derecha, y adelante, z arriba -> cámara: x der, y abajo, z adelante
Rcw = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
tcam = np.array([0.0, 0.0, 1.5])
elips = []
for p, S, Rg, tipo, op in gaus:
    pc = Rcw @ (p - tcam)
    if pc[2] < 0.5:
        continue
    Sig = Rg @ S @ S @ Rg.T            # Sigma = R S S^T R^T (parametrización de 3DGS)
    Sc = Rcw @ Sig @ Rcw.T
    x, y, z = pc
    J = np.array([[F / z, 0, -F * x / z ** 2], [0, F / z, -F * y / z ** 2]])
    S2 = J @ Sc @ J.T
    u, v = F * x / z + cx, F * y / z + cy
    ev, evec = np.linalg.eigh(S2)
    ang = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
    rx, ry = 1.5 * math.sqrt(ev[1]), 1.5 * math.sqrt(ev[0])
    if -60 < u < W_IM + 60 and -60 < v < H_IM + 60:
        elips.append((z, u, v, rx, ry, ang, tipo, op))
elips.sort(key=lambda e: -e[0])   # de atrás hacia delante
print(f"  {len(gaus)} gaussianas, {len(elips)} visibles; ordenadas por profundidad (de atrás adelante)")
for z, u, v, rx, ry, ang, tipo, op in elips:
    print(f"  elipse {tipo} z={z:5.2f}: cx={u:.1f} cy={v:.1f} rx={rx:.1f} ry={ry:.1f} ang={ang:.1f} op={op}")
P = 3 + 3 + 4 + 1 + 48
print(f"  parámetros por gaussiana: posición 3 + escala 3 + rotación 4 + opacidad 1 + SH de grado 3 (16 x 3 = 48) = {P}")
for n in (1e5, 1e6, 3e6):
    print(f"    {n:.0e} gaussianas: {n*P*4/1e6:.0f} MB en float32 (solo parámetros)")

# ---------------------------------------------------------------- 3. cobertura de vistas: captura a mano frente a vuelo
print("=" * 70)
print("3. ¿VE EL DRON LA ESCENA DESDE DONDE SE CAPTURÓ? (ángulo a la vista de captura más cercana)")
centro = np.array([0.0, 0.0, 1.0])
cap = []
for k in range(120):
    a = 2 * math.pi * k / 120
    cap.append(np.array([5 * math.cos(a), 5 * math.sin(a), 1.5]))     # captura a pie, 1,5 m de altura, radio 5 m
cap = np.array(cap)
dcap = (centro - cap) / np.linalg.norm(centro - cap, axis=1, keepdims=True)


def fuera(alturas, radio=5.0, umbral=15.0):
    res = []
    for h in alturas:
        ang = []
        for k in range(360):
            a = 2 * math.pi * k / 360
            p = np.array([radio * math.cos(a), radio * math.sin(a), h])
            d = (centro - p) / np.linalg.norm(centro - p)
            ang.append(math.degrees(math.acos(np.clip((dcap @ d).max(), -1, 1))))
        res.append(np.min(ang))
    return res


for h, a in zip((1.5, 3, 5, 8, 12), fuera((1.5, 3, 5, 8, 12))):
    print(f"  dron a {h:4.1f} m de altura, radio 5 m: ángulo mínimo a una vista de captura {a:5.1f}°")
for r in (2, 5, 10):
    p = np.array([r, 0, 1.5])
    d = (centro - p) / np.linalg.norm(centro - p)
    a = math.degrees(math.acos(np.clip((dcap @ d).max(), -1, 1)))
    print(f"  misma altura, radio {r} m: ángulo {a:.1f}° (pero escala {5/r:.1f}x: resolución distinta)")

# ---------------------------------------------------------------- 4. coste de render frente a lo que pide el aprendizaje
print("=" * 70)
print("4. ¿CUÁNTO TARDA EN RENDERIZARSE UN ENTRENAMIENTO? (cifras de cada artículo, una GPU, sin paralelizar)")
casos = [("NeRF original, V100 (30 s por imagen)", 1 / 30),
         ("Instant-NGP, RTX 3090, HD (60 FPS)", 60),
         ("3DGS-30K, A6000, Mip-NeRF 360 (134 FPS)", 134),
         ("FiGS de SOUS VIDE (hasta 130 imágenes/s)", 130),
         ("GRaD-Nav, paso completo de 0,07 s (render 55,7 %)", 1 / 0.07)]
for nombre, fps in casos:
    h7 = 1e7 / fps / 3600
    print(f"  {nombre:52s} {fps:8.3f} img/s -> 1e7 imágenes: {h7:9.1f} h ({h7/24/365:.2f} años); 1e5: {1e5/fps/3600:.2f} h")
print("  presupuesto por imagen a 30 Hz: 33,3 ms; con obturador rodante en 8 bandas y desenfoque con 4 subimágenes: 32 renders por imagen")
print(f"  -> cada render debe tardar {33.3/32:.2f} ms (≈ {32*30} renders/s)")
