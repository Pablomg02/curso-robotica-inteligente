"""Esquema de la DCT del tema A12a: un trozo de 1 s de la velocidad angular de cabeceo (a 50 Hz) y sus coeficientes.

Uso:  uv run --with numpy --with scipy python figura_dct.py > salida_dct.txt
Usa las mismas acciones CTBR que figuras_a12a.py (se importan sus funciones sin ejecutar el resto del script).
"""
import math
import numpy as np
from scipy.fft import dct, idct

G, M = 9.81, 2.0
T_MAX = 2.0 * M * G
W_MAX = np.array([4.0, 4.0, 2.0])


def referencia(t, k):
    r = np.random.default_rng(k)
    f = r.uniform(0.1, 0.8, (4, 3))
    a = r.uniform(0.2, 1.5, (4, 3)) / (1 + 4 * f)
    ph = r.uniform(0, 2 * np.pi, (4, 3))
    w = 2 * np.pi * f
    pos = lambda d: sum(a[:, i, None] * (w[:, i, None] ** d) * np.sin(w[:, i, None] * t + ph[:, i, None] + d * np.pi / 2) for i in range(3))
    return pos(0), pos(2), pos(3)


def acciones_ctbr(dur, hz, k):
    t = np.arange(0, dur, 1.0 / hz)
    p, a, j = referencia(t, k)
    acc = a[:3].T + np.array([0, 0, G])
    T = M * np.linalg.norm(acc, axis=1)
    zb = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    psi, dpsi = p[3], np.gradient(p[3], t)
    xc = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], 1)
    yb = np.cross(zb, xc)
    yb /= np.linalg.norm(yb, axis=1, keepdims=True)
    xb = np.cross(yb, zb)
    h = M / T[:, None] * (j[:3].T - np.sum(zb * j[:3].T, 1, keepdims=True) * zb)
    wx, wy = -np.sum(h * yb, 1), np.sum(h * xb, 1)
    wz = dpsi * zb[:, 2]
    return np.clip(np.stack([2 * T / T_MAX - 1, wx / W_MAX[0], wy / W_MAX[1], wz / W_MAX[2]], 1), -1, 1)


x = acciones_ctbr(1.0, 50, 50000)[:, 2]          # el primer trozo de prueba de figuras_a12a.py, dimensión de cabeceo
c = dct(x, norm="ortho")
q = np.round(10 * c).astype(int)
rec = idct(q / 10, norm="ortho")
print(f"trozo de 50 muestras; coeficientes con |c| > 0,05 (no nulos tras redondear con escala 10): {int(np.sum(q != 0))} de 50")
print(f"energía en los 5 primeros coeficientes: {100 * np.sum(c[:5] ** 2) / np.sum(c ** 2):.2f} %; RMSE tras redondear: {np.sqrt(np.mean((x - rec) ** 2)):.4f}")
print("primeros coeficientes:", " ".join(f"{v:.3f}" for v in c[:8]))
print("enteros tras redondear con escala 10:", " ".join(str(v) for v in q[:12]), "...")
# trazados: señal (x de 40 a 300 px; y = 130 - 180 px por unidad) y barras de coeficientes (x de 364 a 620; eje en y = 150)
X = lambda i: 40 + i / 49 * 260
Y = lambda v: 130 - v * 180
print("path senal:", "M" + " L".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(x)))
print("path reconstruida:", "M" + " L".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(rec)))
cmax = np.abs(c).max()
print("path barras:", "M" + " ".join(f"M{364 + i * 5.2:.1f},150 L{364 + i * 5.2:.1f},{150 - c[i] / cmax * 100:.1f}" for i in range(50))[1:])
print(f"escala de las barras: 100 px = {cmax:.2f}")
