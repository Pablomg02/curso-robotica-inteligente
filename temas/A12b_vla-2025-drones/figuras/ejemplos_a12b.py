"""Ejemplos numéricos y datos de las figuras nuevas del tema A12b (numpy, CPU, menos de un segundo).

Uso:  python3 ejemplos_a12b.py > salida_ejemplos.txt

Cada ejemplo lleva su número; las hojas lo citan («ejemplo 7»). Las cifras de sistemas reales (GR00T N1, RDT-1B, RaceVLA,
QLoRA...) son las que ya cita el tema; aquí solo se hacen las cuentas con ellas. Las del experimento salen de
figuras/salida.txt (ajuste_smolvla.py, que no se relanza) o se recalculan con la misma dinámica y el mismo experto.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def titulo(n, s):
    print(f"\n=== Ejemplo {n}: {s}")


# ------------------------------------------------------------------ parte 1: dos sistemas
titulo(1, "dos sistemas: pasos rápidos por cada paso lento y trozos que cubren lo que tarda generarlos")
print(f"GR00T N1: 120 Hz / 10 Hz = {120 / 10:.0f} acciones rápidas por cada decisión del VLM")
print(f"Helix: 200 Hz / 9 Hz = {200 / 9:.1f};  200 Hz / 7 Hz = {200 / 7:.1f}")
dur = 16 / 120
print(f"GR00T N1: un trozo de 16 acciones a 120 Hz dura {dur * 1e3:.1f} ms; generarlo cuesta 63,9 ms -> margen {dur * 1e3 - 63.9:.1f} ms"
      f" (cociente {dur * 1e3 / 63.9:.2f})")
print(f"Gemini Robotics: para cubrir ~250 ms a 50 Hz hacen falta {0.25 * 50:.1f} -> al menos {int(np.ceil(0.25 * 50))} acciones por trozo")
print(f"RDT-1B: 381 acciones/s / 6 trozos/s = {381 / 6:.1f} acciones por trozo (el artículo usa trozos de T_a = 64)")

titulo(2, "π*0.6 (RECAP) de juguete: condicionar a «ventaja positiva» mejora la política de referencia")
pi_ref = np.array([0.5, 0.3, 0.2])
pasos = np.array([4, 6, 12])            # pasos hasta el éxito tras cada acción (recompensa -1 por paso, 0 al acabar)
R = -pasos.astype(float)
V = pi_ref @ R
A = R - V
print(f"pi_ref = {pi_ref};  retorno tras cada acción R = {R};  V = sum pi_ref R = {V:.2f}")
print(f"ventaja A = R - V = {A}")
for eps in (0.0, 1.0):
    I = (A > eps).astype(float)
    pi_hat = pi_ref * I / (pi_ref @ I)
    print(f"eps = {eps}: I = {I}, P(I=1) = {pi_ref @ I:.2f}, pi(a|I=1) = {pi_hat}, retorno esperado = {pi_hat @ R:.2f} (antes {V:.2f})")

# ------------------------------------------------------------------ parte 2: cuerpos
titulo(3, "relleno con máscara: pérdida y gradiente con D = 4 y un robot de d = 2")
a_rell = np.array([1.0, -0.5, 0.0, 0.0])
a_hat = np.array([0.8, -0.2, 0.3, -0.1])
m = np.array([1, 1, 0, 0], float)
err = a_hat - a_rell
L_m = (m * err ** 2).sum() / m.sum()
L_sin = (err ** 2).mean()
g_m = 2 * m * err / m.sum()
g_sin = 2 * err / len(err)
print(f"a~ = {a_rell}, a^ = {a_hat}, m = {m}, error = {err}")
print(f"con máscara: L = {L_m:.4f}; gradiente dL/da^ = {g_m}")
print(f"sin máscara: L = {L_sin:.4f}; gradiente dL/da^ = {g_sin}  (empuja las dimensiones 3 y 4 hacia 0)")
print(f"SmolVLA en el experimento: d = 2 de D = 32 -> {2 / 32 * 100:.2f} % de dimensiones reales, {32 - 2} de relleno")

titulo(4, "relleno en el tiempo: trozos que pasan del final del episodio")
T_EP, H = 120, 10
t = 115
n = min(H, T_EP - t)
print(f"muestra en t = {t} de {T_EP}: {n} acciones reales, {H - n} repetidas y enmascaradas; máscara temporal = {[1] * n + [0] * (H - n)}")
print(f"fracción de instantes de partida con algún relleno: {sum(1 for t in range(T_EP) if T_EP - t < H)}/{T_EP} = "
      f"{sum(1 for t in range(T_EP) if T_EP - t < H) / T_EP * 100:.1f} %")

titulo(5, "CrossFormer: error típico de una tasa medida con 3 ensayos")
for p in (0.68, 0.75, 0.82):
    print(f"p = {p}: sqrt(p(1-p)/3) = {np.sqrt(p * (1 - p) / 3):.3f}")
print(f"diferencia publicada: 0,82 - 0,68 = {0.82 - 0.68:.2f}")

# ------------------------------------------------------------------ parte 3: drones
titulo(6, "cuánto se mueve el dron mientras el VLA decide: e = v delta")
for nom, d, v in (("RaceVLA, velocidad media", 0.25, 1.04), ("RaceVLA, velocidad máxima", 0.25, 2.02),
                  ("RaceVLA a 10 m/s", 0.25, 10.0), ("SmolVLA (105 ms) a 3 m/s, el máximo del experto", 0.105, 3.0)):
    print(f"{nom}: delta = {d} s, v = {v} m/s -> e = {v * d:.3f} m")
print(f"entre dos decisiones a 4 Hz, a 2,02 m/s: v/f = {2.02 / 4:.3f} m")

titulo(7, "trozos: condición delta <= N_ejec / f")
f, N = 10, 5
print(f"experimento: N_ejec/f = {N}/{f} = {N / f:.2f} s frente a delta = 0,105 s -> sobra {N / f - 0.105:.3f} s")
print(f"sin trozos a 50 Hz: periodo {1 / 50 * 1e3:.0f} ms frente a 105 ms -> {0.105 * 50:.2f} periodos de retraso")
print(f"decisiones de SmolVLA por episodio de 12 s: {120 // N}")

titulo(8, "memoria de los pesos: bytes por parámetro")
for nom, Np in (("OpenVLA, 7B", 7e9), ("SmolVLA, 450 M", 450e6)):
    print(nom + ": " + ", ".join(f"{fmt} {Np * b / 1e9:.2f} GB" for fmt, b in (("fp32", 4), ("bf16", 2), ("int8", 1), ("int4", 0.5))))

# ------------------------------------------------------------------ parte 4: LoRA y QLoRA
titulo(9, "un producto de rango 1: B (4x1) por A (1x4)")
B = np.array([[1.0], [0.0], [2.0], [-1.0]])
Am = np.array([[0.5, 1.0, 0.0, -0.5]])
BA = B @ Am
print("BA =\n", BA)
print(f"rango de BA = {np.linalg.matrix_rank(BA)}; parámetros: B {B.size} + A {Am.size} = {B.size + Am.size} frente a {BA.size}")
x = np.ones(4)
print(f"x = {x}: A x = {(Am @ x)[0]:.2f} (un solo número, el cuello de botella); B (A x) = {(B @ (Am @ x))}")

titulo(10, "cuántos parámetros entrena LoRA en una proyección de 960 x 960 de SmolVLA")
d = k = 960
for r in (1, 8, 32):
    print(f"r = {r}: r(d+k) = {r * (d + k):,} frente a dk = {d * k:,} -> {r * (d + k) / (d * k) * 100:.2f} %")

titulo(11, "memoria del ajuste con Adam en fp32 (pesos, gradiente y dos momentos = 16 B por parámetro entrenable)")
full = d * k * 16
lora = d * k * 4 + 32 * (d + k) * 16
print(f"ajuste completo de la matriz: {d * k:,} x 16 B = {full / 1e6:.2f} MB")
print(f"LoRA r = 32: W0 congelada {d * k:,} x 4 B + {32 * (d + k):,} x 16 B = {lora / 1e6:.2f} MB (cociente {full / lora:.2f})")

titulo(12, "QLoRA: cuantizar un bloque con absmax y los 16 niveles de NF4 (valores del apéndice E del artículo)")
NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0])
w = np.array([0.12, -0.50, 0.30, 0.05])
c = np.abs(w).max()
wn = w / c
idx = np.abs(wn[:, None] - NF4[None]).argmin(1)
wq = NF4[idx] * c
print(f"bloque w = {w}; constante c = absmax = {c}")
print(f"w/c = {wn}; índice del nivel más cercano (0..15) = {idx}; nivel = {NF4[idx]}")
print(f"descuantizado c * nivel = {wq}; error = {wq - w}")

titulo(13, "QLoRA: bits por parámetro y memoria de 65B")
b1 = 4 + 32 / 64
b2 = 4 + 8 / 64 + 32 / (64 * 256)
print(f"4 bits + constante fp32 cada 64: {b1:.3f} bits; con doble cuantización (fp8 cada 64, fp32 cada 64*256): {b2:.3f} bits")
for nom, b in (("bf16", 16), ("NF4 sin DQ", b1), ("NF4 con DQ", b2)):
    print(f"65B en {nom}: {65e9 * b / 8 / 1e9:.1f} GB")
print(f"ahorro de la doble cuantización: {(b1 - b2):.3f} bits x 65e9 / 8 = {(b1 - b2) * 65e9 / 8 / 1e9:.2f} GB")

# ------------------------------------------------------------------ parte 6: experimento
titulo(14, "la dinámica del dron y el experto del experimento, paso a paso")
DT, TAU, VMAX, KP = 0.1, 0.3, 3.0, 1.5


def experto(p, obj):
    u = KP * (obj - p)
    nrm = np.linalg.norm(u)
    return u if nrm <= VMAX else u * VMAX / nrm


p, v, obj = np.array([4.0, 4.0]), np.zeros(2), np.array([12.0, 10.0])
print(f"p0 = {p}, objeto en {obj}, distancia {np.linalg.norm(obj - p):.1f} m; saturación hasta d = VMAX/KP = {VMAX / KP:.1f} m")
print(f"factor por paso de la velocidad: 1 - DT/TAU = {1 - DT / TAU:.4f}; tras 3 pasos queda {(1 - DT / TAU) ** 3:.3f} del error")
tray = []
for kk in range(120):
    u = experto(p, obj)
    if kk < 4:
        print(f"k = {kk}: u = {u}, |u| = {np.linalg.norm(u):.3f}", end="; ")
    v = v + (u - v) * DT / TAU
    p = p + v * DT
    if kk < 4:
        print(f"v = {v}, |v| = {np.linalg.norm(v):.3f}, p = {p}")
    tray.append((kk + 1, np.linalg.norm(obj - p), np.linalg.norm(v)))
tray = np.array(tray)
k15 = int(tray[tray[:, 1] < 1.5][0, 0])
print(f"llega a menos de 1,5 m en el paso {k15} ({k15 * DT:.1f} s); distancia final a los 12 s: {tray[-1, 1]:.4f} m")
print("puntos de la figura (t s, distancia m, rapidez m/s), cada 0,2 s hasta 6 s:")
for kk in range(1, 61, 2):
    print(f"  {kk * DT:.1f} {tray[kk - 1, 1]:.3f} {tray[kk - 1, 2]:.3f}")
# coordenadas SVG: eje x 0..6 s -> 60..540 px; distancia 0..10 m -> 230..30 px; rapidez 0..3 m/s -> 230..30 px
pts_d = " ".join(f"{60 + t * 80:.0f},{230 - dd * 20:.1f}" for t, dd in [(0.0, 10.0)] + [(kk * DT, tray[kk - 1, 1]) for kk in range(1, 61)])
pts_v = " ".join(f"{60 + t * 80:.0f},{230 - vv * 200 / 3:.1f}" for t, vv in [(0.0, 0.0)] + [(kk * DT, tray[kk - 1, 2]) for kk in range(1, 61)])
print("SVG distancia:", pts_d)
print("SVG rapidez:", pts_v)

titulo(15, "la política pequeña por dentro: tamaños y parámetros")
capas = [("conv 5x5, 5 -> 32, paso 2", 5 * 32 * 25 + 32, "64x64x5 -> 32x32x32"),
         ("conv 3x3, 32 -> 64, paso 2", 32 * 64 * 9 + 64, "32x32x32 -> 16x16x64"),
         ("conv 3x3, 64 -> 64, paso 2", 64 * 64 * 9 + 64, "16x16x64 -> 8x8x64"),
         ("FiLM: tabla de 4 colores x 2(32+64+64)", 4 * 2 * (32 + 64 + 64), "color -> gamma, beta"),
         ("lineal (4096 + 2) -> 256", (64 * 8 * 8 + 2) * 256 + 256, "4098 -> 256"),
         ("lineal 256 -> 20", 256 * 20 + 20, "256 -> 10 x 2")]
tot = sum(c[1] for c in capas)
for nom, n, forma in capas:
    print(f"{nom:42s} {forma:24s} {n:>10,}  {n / tot * 100:5.1f} %")
print(f"total {tot:,} (salida.txt: 1,12 M)")
print("FiLM en un canal con valor 0,8: color 1 (gamma 0,5, beta -0,2) -> relu(0,8*1,5 - 0,2) =",
      f"{max(0, 0.8 * 1.5 - 0.2):.2f}; color 2 (gamma -0,9, beta 0,1) -> {max(0, 0.8 * 0.1 + 0.1):.2f}")

titulo(16, "normalización de entradas y salidas en el experimento")
L = 20.0
for pp in (0.0, 10.0, 20.0):
    print(f"posición {pp} m -> (p - L/2)/(L/4) = {(pp - L / 2) / (L / 4):.1f}")
print(f"acción 3 m/s -> u / VMAX = {3 / VMAX:.1f}; 1,5 m/s -> {1.5 / VMAX:.1f}")

titulo(17, "cuántos ejemplos ve cada política durante el ajuste (de salida.txt)")
for n_d in (200, 1000):
    pares = n_d * 120
    print(f"{n_d} demostraciones = {pares:,} instantes de partida; SmolVLA 4000 x 16 = {4000 * 16:,} "
          f"({4000 * 16 / pares:.2f} pasadas); pequeña 8000 x 64 = {8000 * 64:,} ({8000 * 64 / pares:.2f} pasadas)")
print(f"tiempos: 4000 x 0,25 s = {4000 * 0.25 / 60:.1f} min (salida: 16,9 min con 0,2535 s); 8000 x 8,3 ms = {8000 * 8.3e-3 / 60:.2f} min")

titulo(18, "memoria del ajuste de SmolVLA (pesos en fp32, como en ajuste_smolvla.py)")
P, Pe = 450e6, 100e6
pesos, grad, adam = P * 4, Pe * 4, 2 * Pe * 4
tot = pesos + grad + adam
print(f"pesos 450 M x 4 B = {pesos / 2**30:.2f} GiB; gradientes 100 M x 4 B = {grad / 2**30:.2f} GiB; "
      f"AdamW 2 x 100 M x 4 B = {adam / 2**30:.2f} GiB; suma {tot / 2**30:.2f} GiB")
print(f"medido: 4,4 GiB -> resto {4.4 - tot / 2**30:.2f} GiB (activaciones, copias en bfloat16 del autocast...)")

titulo(19, "resultados (de salida.txt) para la figura de barras")
res = [("SmolVLA, 200", 0.24, 0.18, 0.05, 0.20), ("pequeña, 200", 0.38, 0.02, 0.00, 0.00),
       ("SmolVLA, 1000", 0.63, 0.05, 0.06, 0.53), ("pequeña, 1000", 0.95, 0.00, 0.00, 0.00)]
for nom, e1, o1, e2, o2 in res:
    print(f"{nom:15s} visto: éxito {e1:.2f}, otro {o1:.2f}, ninguno {1 - e1 - o1:.2f} | amarillo: éxito {e2:.2f}, otro {o2:.2f}, ninguno {1 - e2 - o2:.2f}")
print("azar si acabara junto a un objeto cualquiera: 1/3 =", f"{1 / 3:.3f}")

titulo(20, "error típico de una tasa medida con 100 escenas y coste de igualar las muestras vistas")
for p in (0.3, 0.6, 0.95):
    print(f"p = {p}: sqrt(p(1-p)/100) = {np.sqrt(p * (1 - p) / 100):.3f}")
print(f"ablación: 0,38 - 0,29 = {0.38 - 0.29:.2f}")
print(f"igualar muestras: 512 000 / 64 000 = {512000 / 64000:.0f} -> {8 * 16.9:.0f} min = {8 * 16.9 / 60:.2f} h de SmolVLA")
print(f"tamaño: 450 M / 1,12 M = {450 / 1.12:.0f}")
