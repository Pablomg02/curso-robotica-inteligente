"""Ejemplos numéricos y datos de las figuras nuevas del tema A12a (las hojas de notación, esquemas y derivaciones).

Uso:  ulimit -v 8000000; uv run --with numpy --with scipy python ejemplos_a12a.py > salida_ejemplos.txt
CPU, unos segundos. Usa las mismas acciones CTBR que figuras_a12a.py (las funciones se copian para no ejecutar el resto
de aquel script, que entrena BPE y llama a ollama). Las cifras de artículos que aparecen aquí (OpenVLA-OFT, π0, RTC)
son las del tema; este script solo hace las cuentas con ellas.
"""
import math
from collections import Counter

import numpy as np
from scipy.fft import dct, idct
from scipy.stats import norm

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


def acciones_ctbr(dur, hz, k, ruido=0.0):
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
    u = np.stack([2 * T / T_MAX - 1, wx / W_MAX[0], wy / W_MAX[1], wz / W_MAX[2]], 1)
    u = u + ruido * np.random.default_rng(k + 10 ** 6).normal(size=u.shape)
    return np.clip(u, -1, 1)


def trozos(hz, n, ruido, base):
    return np.stack([acciones_ctbr(1.0, hz, base + k, ruido) for k in range(n)])


def casillas(x, n=256):
    b = np.clip(np.floor((x + 1) / 2 * n), 0, n - 1).astype(int)
    return b, (b + 0.5) / n * 2 - 1


def cab(n, titulo):
    print("=" * 78)
    print(f"Ejemplo {n}. {titulo}")


# ---------------------------------------------------------------------------------------------------------------
cab(1, "Casillas: de un número a un token y vuelta")
for N, x in ((8, 0.3), (256, 0.3)):
    D = 2 / N
    b = min(math.floor((x + 1) / D), N - 1)
    xh = -1 + (b + 0.5) * D
    print(f"  N = {N}: Delta = 2/N = {D:.6f}; x = {x}: (x+1)/Delta = {(x + 1) / D:.4f} -> casilla b = {b}; "
          f"centro x^ = -1 + (b + 1/2) Delta = {xh:.6f}; error e = x - x^ = {x - xh:+.6f}")
print("  bordes de las 8 casillas:", " ".join(f"{-1 + i * 0.25:+.2f}" for i in range(9)))
print("  x = 1 exactamente: (1+1)/Delta = N -> se recorta a la casilla N-1 (el mín. de la fórmula)")
a = np.array([0.10, -0.25, 0.02, 0.0])
b, ah = casillas(a)
print(f"  acción CTBR de ejemplo (empuje, wx, wy, wz) = {a.tolist()} -> casillas {b.tolist()} -> texto '{' '.join(map(str, b))}'; "
      f"vuelta: {np.round(ah, 5).tolist()}")
# percentiles (OpenVLA): rango [x_1, x_99] en lugar de [-1, 1]
x1, x99 = -0.6, 0.8
print(f"  rango por percentiles (ilustrativo) [{x1}, {x99}]: Delta = (x99 - x1)/N = {(x99 - x1) / 256:.5f} con N = 256 "
      f"(frente a {2 / 256:.5f} en [-1, 1])")

# ---------------------------------------------------------------------------------------------------------------
cab(2, "Regla de la cadena: cuántas pasadas por el modelo cuesta escribir la acción")
for nombre, d, H in (("OpenVLA, una acción de 7 dimensiones", 7, 1), ("dron CTBR, una acción", 4, 1),
                     ("dron CTBR, trozo de 1 s a 50 Hz", 4, 50), ("dron CTBR, trozo de 1 s a 200 Hz", 4, 200)):
    print(f"  {nombre}: K = d H = {d} x {H} = {d * H} tokens -> {d * H} pasadas en serie")
p = [0.9, 0.8, 0.95, 0.7]
print(f"  probabilidades de cada token dado lo anterior {p}: producto = {np.prod(p):.4f}; "
      f"entropía cruzada = -sum log p = {-np.sum(np.log(p)):.4f} nats")

# ---------------------------------------------------------------------------------------------------------------
cab(3, "OpenVLA-OFT, tabla II (A100, acciones de 7 dimensiones): qué cuesta cada pasada")
t_ar, t_pd, t_ac = 0.2396, 0.0629, 0.0735
print(f"  autorregresivo: 7 tokens, 1 acción en {t_ar * 1e3:.1f} ms -> {1 / t_ar:.1f} acciones/s")
print(f"  paralelo, 1 acción: 7 salidas en una pasada, {t_pd * 1e3:.1f} ms -> {1 / t_pd:.1f} acciones/s")
print(f"  paralelo + trozo de 8: 56 salidas en una pasada, {t_ac * 1e3:.1f} ms -> {8 / t_ac:.1f} acciones/s")
pasada = (t_ar - t_pd) / 6
print(f"  si el autorregresivo es 'paralelo + 6 pasadas más': cada pasada extra ~ ({t_ar * 1e3:.1f} - {t_pd * 1e3:.1f})/6 = {pasada * 1e3:.1f} ms")
print(f"  pasar de 7 a 56 salidas en la pasada única añade {(t_ac - t_pd) * 1e3:.1f} ms ({100 * (t_ac - t_pd) / t_pd:.0f} %) por 8 veces más acciones")
print(f"  ganancias: paralelo {t_ar / t_pd:.1f} veces; trozo {(8 / t_ac) / (1 / t_pd):.1f} veces más; total {(8 / t_ac) / (1 / t_ar):.1f} veces")
for pasos, hz in ((50, 4.2), (10, 19.3), (5, 35.1), (2, 80.3), (1, 109.4)):
    print(f"  difusión con {pasos} pasos: {hz} acciones/s -> {8 / hz * 1e3:.0f} ms por trozo de 8")
print(f"  difusión: cada paso de quitar ruido más cuesta (1907,0 - 73,1)/49 = {(1907.0 - 73.1) / 49:.1f} ms")

# ---------------------------------------------------------------------------------------------------------------
cab(4, "Memoria de OpenVLA según los bytes por peso (unos 7e9 pesos)")
for nombre, by, med in (("bf16", 2, 16.8), ("int8", 1, 10.2), ("int4", 0.5, 7.0)):
    print(f"  {nombre}: 7e9 x {by} B = {7 * by:.1f} GB de pesos; medido en el artículo {med} GB")

# ---------------------------------------------------------------------------------------------------------------
cab(5, "Experto de flow matching: prefijo una vez, experto P veces (cifras de π0 en A10)")
t_tot, t_flujo, P = 73, 27, 10
print(f"  π0: {t_tot} ms por trozo, {t_flujo} ms las {P} pasadas del flujo -> {t_flujo / P:.1f} ms por pasada; resto {t_tot - t_flujo} ms (el prellenado de RTC: 46 ms)")
for t in (0.0, 0.1, 0.5, 0.9):
    print(f"  Euler: tau = {t:.1f} -> tau + delta = {t + 0.1:.1f}")
pre = 46
for ntok in (30, 60):
    print(f"  π0-FAST, unos 750 ms con {ntok} tokens: (750 - {pre})/{ntok} = {(750 - pre) / ntok:.0f} ms por token (si el prellenado es el mismo)")

# ---------------------------------------------------------------------------------------------------------------
cab(6, "La escalera de las casillas: un trozo de cabeceo a 50 y 200 Hz")
for hz in (50, 200):
    x = acciones_ctbr(1.0, hz, 50000)[:, 2]
    b, _ = casillas(x)
    n = hz // 10
    print(f"  {hz} Hz, primeros 0,1 s ({n} pasos): casillas {b[:n].tolist()}; repetidos en el trozo entero {100 * np.mean(b[1:] == b[:-1]):.0f} %")
x200 = acciones_ctbr(1.0, 200, 50000)[:, 2]
b200, _ = casillas(x200)
x50 = acciones_ctbr(1.0, 50, 50000)[:, 2]
b50, _ = casillas(x50)
# figura: 0,25 s de ese trozo alrededor del máximo (donde la señal apenas se mueve); eje x de 70 a 590 px
T0 = 0.10
j0, j1 = int(T0 * 200), int(T0 * 200) + 50
i0, i1 = int(b200[j0:j1].min()) - 1, int(b200[j0:j1].max()) + 1
print(f"  ventana de la figura: de {T0} a {T0 + 0.25} s; casillas de {i0} a {i1}")
X = lambda t: 70 + (t - T0) / 0.25 * 520
Y = lambda v: 230 - (v - i0) / (i1 - i0) * 190
print("  path escalera 200:", "M" + " L".join(f"{X(j / 200):.1f},{Y(b200[j]):.1f} L{X((j + 1) / 200):.1f},{Y(b200[j]):.1f}" for j in range(j0, j1)))
print("  path escalera 50:", "M" + " L".join(f"{X(j / 50):.1f},{Y(b50[j]):.1f} L{X((j + 1) / 50):.1f},{Y(b50[j]):.1f}" for j in range(int(T0 * 50), int(T0 * 50) + 12)))
print(f"  casillas a 200 Hz en la ventana: {b200[j0:j1].tolist()}; repetidos {100 * np.mean(b200[j0 + 1:j1] == b200[j0:j1 - 1]):.0f} %")
print(f"  casillas a 50 Hz en la ventana: {b50[int(T0 * 50):int(T0 * 50) + 12].tolist()}")
xc = acciones_ctbr(1.0, 400, 50000)[:, 2]
print("  path señal (400 Hz):", "M" + " L".join(f"{X(i / 400):.1f},{Y((xc[i] + 1) / (2 / 256) - 0.5):.1f}" for i in range(int(T0 * 400), int(T0 * 400) + 101)))
for v in range(i0, i1 + 1, 2):
    print(f"  marca y casilla {v}: {Y(v):.1f}")
# repetición predicha: un token cambia si la señal cruza un borde en el paso; con pendiente |x'| la probabilidad de
# cruzar un borde en un paso de 1/f es min(1, |x'|/(f Delta)) si la posición dentro de la casilla es uniforme
for hz in (50, 200):
    xx = trozos(hz, 300, 0.0, 50000)
    bb, _ = casillas(xx)
    xd = np.abs(np.diff(xx, axis=1))                     # |x'| / f por paso
    pred = 1 - np.mean(np.minimum(1, xd / (2 / 256)))
    print(f"  {hz} Hz, 300 trozos de prueba x 4 dim.: repetidos medidos {100 * np.mean(bb[:, 1:] == bb[:, :-1]):.1f} %; "
          f"predichos 1 - E[min(1, |dx|/Delta)] = {100 * pred:.1f} %; mediana de |dx|/Delta = {np.median(xd / (2 / 256)):.2f}")

# ---------------------------------------------------------------------------------------------------------------
cab(7, "BPE a mano sobre los enteros de la DCT de un trozo (una dimensión, 50 coeficientes)")
c = dct(x50, norm="ortho")
q = np.round(10 * c).astype(int)
seq = [str(v) for v in q]
print("  secuencia de partida:", " ".join(seq), f"({len(seq)} símbolos)")
vocab_nuevo = []
for paso in range(4):
    pares = Counter(zip(seq, seq[1:]))
    (a1, a2), cuenta = pares.most_common(1)[0]
    nuevo = a1 + "·" + a2
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a1 and seq[i + 1] == a2:
            out.append(nuevo); i += 2
        else:
            out.append(seq[i]); i += 1
    seq = out
    vocab_nuevo.append(nuevo)
    print(f"  fusión {paso + 1}: par más frecuente ({a1}, {a2}), {cuenta} veces -> símbolo nuevo '{nuevo}'; quedan {len(seq)} símbolos")
print("  secuencia final:", " ".join(seq))

# ---------------------------------------------------------------------------------------------------------------
cab(8, "La base de la DCT-II ortonormal con H = 8")
H = 8
Cm = np.array([[(math.sqrt(1 / H) if k == 0 else math.sqrt(2 / H)) * math.cos(math.pi / H * (n + 0.5) * k) for n in range(H)] for k in range(H)])
print(f"  max |C C^T - I| = {np.abs(Cm @ Cm.T - np.eye(H)).max():.2e}")
print(f"  alfa_0 = sqrt(1/8) = {math.sqrt(1 / 8):.4f}; alfa_k = sqrt(2/8) = {math.sqrt(2 / 8):.4f}")
for k in range(4):
    print(f"  fila k = {k}:", " ".join(f"{v:+.3f}" for v in Cm[k]))
# trazados: cada coseno en una franja de 70 px de alto; x de 60 a 360 px (n = 0..7 en el centro de cada celda)
for k in range(4):
    y0 = 55 + k * 72
    pts = [(60 + (u + 0.5) * 300 / 8, y0 - math.cos(math.pi / 8 * (u + 0.5) * k) * 24) for u in np.linspace(-0.5, 7.5, 81)]
    dots = [(60 + (n + 0.5) * 300 / 8, y0 - math.cos(math.pi / 8 * (n + 0.5) * k) * 24) for n in range(8)]
    print(f"  path coseno {k}: M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts))
    print(f"  puntos coseno {k}: " + " ".join(f"{px:.1f},{py:.1f}" for px, py in dots))

# ---------------------------------------------------------------------------------------------------------------
cab(9, "La DCT a mano: un trozo de 4 muestras (una rampa)")
x4 = np.array([0.1, 0.2, 0.3, 0.4])
H = 4
for k in range(4):
    al = math.sqrt(1 / H) if k == 0 else math.sqrt(2 / H)
    cs = [math.cos(math.pi / H * (n + 0.5) * k) for n in range(H)]
    terms = [x4[n] * cs[n] for n in range(H)]
    print(f"  k = {k}: alfa = {al:.4f}; cosenos " + ", ".join(f"{v:+.4f}" for v in cs) + "; x_n cos = " + ", ".join(f"{v:+.4f}" for v in terms)
          + f"; suma = {sum(terms):+.4f}; c_{k} = {al * sum(terms):+.4f}")
c4 = dct(x4, norm="ortho")
print("  scipy:", " ".join(f"{v:+.4f}" for v in c4))
q4 = np.round(10 * c4).astype(int)
r4 = idct(q4 / 10, norm="ortho")
print(f"  round(10 c) = {q4.tolist()}; c^ = q/10 = {(q4 / 10).tolist()}; x^ = IDCT(c^) = " + ", ".join(f"{v:.4f}" for v in r4))
print(f"  error x - x^ = " + ", ".join(f"{v:+.4f}" for v in x4 - r4) + f"; RMSE {np.sqrt(np.mean((x4 - r4) ** 2)):.4f}")
print(f"  energía: sum x^2 = {np.sum(x4 ** 2):.4f}; sum c^2 = {np.sum(c4 ** 2):.4f}")
print(f"  error en coeficientes c - c^ = " + ", ".join(f"{v:+.4f}" for v in c4 - q4 / 10) + f"; RMSE {np.sqrt(np.mean((c4 - q4 / 10) ** 2)):.4f} (igual: Parseval)")
x8 = np.linspace(0.1, 0.4, 8)
c8 = dct(x8, norm="ortho")
print("  la misma rampa con 8 muestras: c = " + " ".join(f"{v:+.4f}" for v in c8) + f"; no nulos con escala 10: {int(np.sum(np.round(10 * c8) != 0))} de 8")
x16 = np.linspace(0.1, 0.4, 16)
c16 = dct(x16, norm="ortho")
print(f"  con 16 muestras: no nulos con escala 10: {int(np.sum(np.round(10 * c16) != 0))} de 16; c_0 = {c16[0]:.4f}, c_1 = {c16[1]:.4f}")

# ---------------------------------------------------------------------------------------------------------------
cab(10, "Redondear coeficientes: el error en el tiempo es el error en los coeficientes (Parseval)")
for g in (2, 5, 10, 20, 50):
    print(f"  escala {g}: tope 1/(2 gamma) = {1 / (2 * g):.4f}; si todos sobreviven, RMSE = 1/(gamma raíz 12) = {1 / (g * math.sqrt(12)):.4f}")
for hz in (50, 200):
    for ruido in (0.0, 0.02):
        xx = trozos(hz, 300, ruido, 50000)
        cc = dct(xx, axis=1, norm="ortho")
        lin = f"  {hz} Hz, ruido {ruido}:"
        for g in (10, 50):
            qq = np.round(cc * g)
            rec = idct(qq / g, axis=1, norm="ortho")
            e_t = np.sqrt(np.mean((xx - rec) ** 2))
            e_c = np.sqrt(np.mean((cc - qq / g) ** 2))
            e_cl = np.sqrt(np.mean((xx - np.clip(rec, -1, 1)) ** 2))
            nz = np.mean(qq != 0)
            lin += f" | escala {g}: RMSE tiempo {e_t:.4f}, coeficientes {e_c:.4f}, recortado {e_cl:.4f}, no nulos {100 * nz:.1f} %"
        print(lin)

# ---------------------------------------------------------------------------------------------------------------
cab(11, "Un seno de 2 Hz en un trozo de 1 s: los coeficientes útiles no crecen con la frecuencia de muestreo")
for hz in (50, 200):
    t = np.arange(hz) / hz
    xs_ = 0.5 * np.sin(2 * np.pi * 2 * t)
    cs_ = dct(xs_, norm="ortho")
    qs = np.round(10 * cs_)
    big = np.argsort(-np.abs(cs_))[:3]
    print(f"  {hz} Hz: H = {hz}; 2 H f_max / f = {2 * hz * 2 / hz:.0f}; no nulos con escala 10: {int(np.sum(qs != 0))} de {hz}; "
          f"índices no nulos {np.nonzero(qs)[0].tolist()}; los tres mayores k = {sorted(big.tolist())}")

# ---------------------------------------------------------------------------------------------------------------
cab(12, "FAST, la tubería con un trozo de 50 Hz (4 dimensiones x 50 pasos)")
X4 = acciones_ctbr(1.0, 50, 50000)
C4 = dct(X4, axis=0, norm="ortho")
Q4 = np.round(10 * C4).astype(int)
print(f"  matriz de enteros 50 x 4: no nulos {int(np.sum(Q4 != 0))} de 200 ({100 * np.mean(Q4 != 0):.1f} %)")
for k in range(5):
    print(f"  fila k = {k} (empuje, wx, wy, wz): {Q4[k].tolist()}")
flat = Q4.reshape(-1)
print("  aplanado con las bajas frecuencias primero (primeros 20):", " ".join(map(str, flat[:20])))
print(f"  último no nulo en la posición {int(np.nonzero(flat)[0].max())} de 200; ceros a partir de ahí: {200 - int(np.nonzero(flat)[0].max()) - 1}")

# ---------------------------------------------------------------------------------------------------------------
cab(13, "El ruido en el espectro: un trozo de cabeceo a 200 Hz, con y sin ruido 0,02")
xa = acciones_ctbr(1.0, 200, 50000)[:, 2]
xb = acciones_ctbr(1.0, 200, 50000, 0.02)[:, 2]
ca, cb = dct(xa, norm="ortho"), dct(xb, norm="ortho")
for g in (10, 50):
    print(f"  escala {g}: no nulos sin ruido {int(np.sum(np.round(g * ca) != 0))}, con ruido {int(np.sum(np.round(g * cb) != 0))} de 200")
for umbral in (0.01, 0.001):
    kk = next(k for k in range(200) if np.all(np.abs(ca[k:]) < umbral))
    print(f"  sin ruido, |c_k| < {umbral} para todo k >= {kk}")
print(f"  desviación de los coeficientes k >= 40 con ruido: {cb[40:].std():.4f} (sigma = 0,02); sin ruido: {ca[40:].std():.5f}")
# figura: |c_k| en escala logarítmica de 1e-4 a 10, k de 0 a 199; x de 70 a 610, y de 30 (10) a 230 (1e-4)
XK = lambda k: 70 + k / 199 * 540
YL = lambda v: 230 - (math.log10(max(abs(v), 1e-4)) + 4) / 5 * 200
print("  path |c| con ruido:", "M" + " L".join(f"{XK(k):.1f},{YL(cb[k]):.1f}" for k in range(200)))
print("  path |c| sin ruido:", "M" + " L".join(f"{XK(k):.1f},{YL(ca[k]):.1f}" for k in range(200)))
for v in (10, 1, 0.1, 0.05, 0.02, 0.01, 0.001, 1e-4):
    print(f"  y({v}) = {YL(v):.1f}")
# predicción del número de no nulos con ruido: los de señal siguen, y de los demás sobrevive la fracción P
for hz in (50, 200):
    for g in (10, 50):
        xx0 = trozos(hz, 300, 0.0, 50000)
        ps = np.mean(np.round(g * dct(xx0, axis=1, norm="ortho")) != 0)
        P = 2 * (1 - norm.cdf(1 / (2 * g * 0.02)))
        print(f"  {hz} Hz, escala {g}: sin ruido {100 * ps:.1f} % no nulos; P = {100 * P:.1f} %; "
              f"predicho con ruido p_s + (1 - p_s) P = {100 * (ps + (1 - ps) * P):.1f} %")
for u in (2.5, 1.25, 0.5):
    print(f"  Phi({u}) = {norm.cdf(u):.4f}; 2 (1 - Phi) = {2 * (1 - norm.cdf(u)):.4f}")

# ---------------------------------------------------------------------------------------------------------------
cab(14, "Latencia de un modelo que escribe tokens (Gemma 4 de 12B, cifras de figuras_a12a.py)")
tpre, r = 0.156, 35.5
print(f"  un token cuesta 1/r = {1000 / r:.1f} ms")
for nombre, K in (("casillas 50 Hz", 200), ("DCT+BPE 50 Hz, con ruido", 26), ("DCT+BPE 200 Hz, con ruido", 44), ("casillas 200 Hz", 800)):
    print(f"  {nombre}: K = {K}; t = {tpre} + {K}/{r} = {tpre + K / r:.3f} s")
Kmax = r * (1 - tpre)
print(f"  K_max en T = 1 s: {r} x (1 - {tpre}) = {Kmax:.1f}; casillas d = 4: f_max = {Kmax / 4:.2f} Hz; 200/{Kmax:.1f} = {200 / Kmax:.2f}")
print(f"  con la cota de ancho de banda r = 41,3: K_max = {41.3 * (1 - tpre):.1f}; f_max = {41.3 * (1 - tpre) / 4:.2f} Hz")
# figura: 0 a 1,2 s en x de 230 a 1150 px
XT = lambda t: 230 + t / 1.2 * 920
print(f"  x(0,156) = {XT(0.156):.1f}; x(0,888) = {XT(0.888):.1f}; x(1) = {XT(1.0):.1f}; ancho de un token = {XT(1 / r) - XT(0):.2f} px")
print("  marcas de los tokens:", " ".join(f"{XT(tpre + j / r):.1f}" for j in range(1, 26)))

# ---------------------------------------------------------------------------------------------------------------
cab(15, "Ejecución en tiempo real (RTC): retraso en pasos y pesos de la guía, H = 50, s = 25, Delta t = 20 ms")
dt = 0.020
for nombre, lat in (("prellenado de π0 (solo)", 0.046), ("π0 completo", 0.073), ("SmolVLA en la GPU", 0.109),
                    ("300 ms (lo que aguanta RTC)", 0.300), ("SmolVLA en la CPU", 0.920)):
    D = math.floor(lat / dt + 1e-9)
    Hh = 50
    ok = D <= Hh - D
    print(f"  {nombre}: {lat * 1e3:.0f} ms -> D = floor({lat:.3f}/{dt}) = {D}; s posible en [{D}, {Hh - D}]" + ("" if ok else " -> vacío: no cabe con H = 50"))
H, s, D = 50, 25, 5
W = []
for i in range(H):
    if i < D:
        W.append(1.0)
    elif i < H - s:
        ci = (H - s - i) / (H - s - D + 1)
        W.append(ci * (math.exp(ci) - 1) / (math.e - 1))
    else:
        W.append(0.0)
print(f"  D = {D}: W_i para i = 0..{H - 1}:", " ".join(f"{w:.2f}" for w in W))
print(f"  rho_i = (H - s - i)/(H - s - D + 1): i = {D} -> {(H - s - D) / (H - s - D + 1):.3f}; i = {H - s - 1} -> {1 / (H - s - D + 1):.3f}")
# figura: i de 0 a 49 en x de 160 a 1110 px (19 px por acción)
print("  path W:", "M" + " L".join(f"{160 + i * 19:.0f},{200 - W[i] * 60:.1f} L{160 + (i + 1) * 19:.0f},{200 - W[i] * 60:.1f}" for i in range(H)))
