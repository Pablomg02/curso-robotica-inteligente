"""Cuentas a mano de las hojas de R1a (los labs de robot learning en 2026).

Cada bloque reproduce un ejemplo numérico de la presentación. Solo usa la biblioteca estándar de Python.
Las cifras publicadas salen de figuras/datos/cifras.json, con su fuente y su estado (C, E o P).
Los SUPUESTOS del curso (salario del operador, horas al día, vida del equipo, precio de la GPU) van marcados.

    python3 ejemplos_r1a.py > salida_ejemplos.txt
"""
import json
import math
from pathlib import Path
from statistics import NormalDist

D = json.loads((Path(__file__).parent / "datos" / "cifras.json").read_text())
Z95 = NormalDist().inv_cdf(0.975)   # 1,959964
Z80 = NormalDist().inv_cdf(0.80)    # 0,841621


def titulo(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# --------------------------------------------------------------------------- utilidades estadísticas
def wilson(k, n, z=Z95):
    """Intervalo de Wilson (T1b): raíces de (p̂ − p)² = z² p(1 − p)/n."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def poisson_cdf(k, lam):
    """P(X <= k) para X ~ Poisson(lam), sumando en escala logarítmica."""
    if k < 0:
        return 0.0
    s, t = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i > 0:
            t *= lam / i
        s += t
    return s


def biseccion(f, a, b, it=200):
    fa = f(a)
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def poisson_exacto(k, conf=0.95):
    """IC exacto (Garwood) del parámetro de una Poisson con k eventos observados."""
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else biseccion(lambda L: (1 - poisson_cdf(k - 1, L)) - a, 1e-9, 10 * k + 10)
    hi = biseccion(lambda L: poisson_cdf(k, L) - a, 1e-9, 10 * k + 20)
    return lo, hi


def n_potencia(p1, p2, alfa=0.05, potencia=0.80):
    """Ensayos por política para detectar p1 frente a p2 (test z bilateral de dos proporciones)."""
    za, zb = NormalDist().inv_cdf(1 - alfa / 2), NormalDist().inv_cdf(potencia)
    pb = (p1 + p2) / 2
    return (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2


# ---------------------------------------------------------------------------
titulo("1. Coste de recoger N demostraciones (parte 2)")
r = D["umi_demos_por_hora"]
sup = D["supuestos_coste"]
eq = D["coste_equipo"]
N, R = 10_000, 10
print(f"N = {N} demostraciones útiles, R = {R} puestos en paralelo, h = {sup['h_dia']} h/día (SUPUESTO),")
print(f"c_o = {sup['c_o_usd_h']} US$/h por operador (SUPUESTO), vida del equipo A = {sup['A_dias']} días (SUPUESTO)")
print("ritmos publicados (UMI, Fig. 11, demostraciones útiles por hora y operador, con reinicios y fallos):")
equipo = {"teleoperacion": eq["aloha_usd"], "umi": eq["umi_pinza_usd"] + eq["umi_gopro_usd"], "mano": 0}
for metodo in ("teleoperacion", "umi", "mano"):
    rr = r[metodo]
    horas = N / rr                       # horas de operador
    dias = N / (rr * R * sup["h_dia"])   # días de calendario
    c_oper = sup["c_o_usd_h"] * horas
    c_equipo = R * equipo[metodo] * dias / sup["A_dias"]
    print(f"  {metodo:14s} r = {rr:3d}/h -> {horas:7.1f} h de operador, D = {dias:5.2f} días, "
          f"operadores {c_oper:8.0f} US$, equipo amortizado {c_equipo:7.0f} US$, total {c_oper + c_equipo:8.0f} US$ = {(c_oper + c_equipo) / N:5.2f} US$/demo")
print("  (las horas de operador N/r no dependen de R: paralelizar acorta el calendario, no la factura de operadores)")
dr = D["droid"]
print(f"DROID: {dr['trayectorias']} trayectorias, {dr['horas']} h -> {dr['horas'] * 3600 / dr['trayectorias']:.1f} s por trayectoria;")
print(f"  {dr['trayectorias'] / dr['recolectores']:.0f} por recolector en {dr['meses']} meses = {dr['trayectorias'] / dr['recolectores'] / dr['meses']:.0f} al mes")
print("GEN-0: +10 000 h de datos por semana; si 1 h de datos = 1 h de una persona (SUPUESTO),")
print(f"  a 5 días x 8 h por semana son {10_000 / 40:.0f} personas o dispositivos a tiempo completo")

# ---------------------------------------------------------------------------
titulo("2. Cómputo publicado: GR00T N1 (parte 3)")
c = D["gr00t_n1_computo"]
print(f"preentrenamiento: {c['h100_h']} H100·h / {c['gpus_max']} GPU = {c['h100_h'] / c['gpus_max']:.1f} h de reloj = {c['h100_h'] / c['gpus_max'] / 24:.1f} días")
precio = D["supuesto_precio_gpu_usd_h"]["H100"]
print(f"  a {precio} US$/H100·h (SUPUESTO): {c['h100_h'] * precio:,.0f} US$".replace(",", " "))
print(f"vídeo generado: {c['l40_h_video']} L40·h / {c['l40_n']} L40 = {c['l40_h_video'] / c['l40_n']:.1f} h de reloj; "
      f"{c['l40_h_video'] / c['video_h']:.0f} L40·h por hora de vídeo")
print(f"simulación: 6500 h de demostración equivalente en 11 h -> x{6500 / 11:.0f} más rápido que el tiempo real")

# ---------------------------------------------------------------------------
titulo("3. Latencia, frecuencia y trozos (parte 3)")
lat = D["gr00t_n17_latencia_ms"]
for dev in ("H100", "RTX Pro 6000", "L40", "DGX Spark", "AGX Thor", "Jetson Orin"):
    d = lat[dev]["trt"]
    print(f"  {dev:13s} TensorRT {d:6.1f} ms -> f_inf = {1000 / d:5.1f} Hz; a f_a = 50 Hz pasan "
          f"{d / 20:4.2f} periodos -> N_ejec >= {math.ceil(d / 20)}; a 30 Hz, N_ejec >= {math.ceil(d * 30 / 1000)}")
print(f"  AGX Thor con NVFP4 {lat['AGX Thor']['trt_nvfp4']} ms -> {1000 / lat['AGX Thor']['trt_nvfp4']:.1f} Hz (README: «~28 Hz»)")
print(f"  RTX Pro 6000 {lat['RTX Pro 6000']['trt']} ms -> {1000 / lat['RTX Pro 6000']['trt']:.1f} Hz (README: «~40 Hz»)")
print(f"  aceleración de TensorRT frente a PyTorch eager: Orin x{lat['Jetson Orin']['eager'] / lat['Jetson Orin']['trt']:.1f}, "
      f"Thor x{lat['AGX Thor']['eager'] / lat['AGX Thor']['trt']:.1f}, H100 x{lat['H100']['eager'] / lat['H100']['trt']:.1f}")
print(f"  Orin frente a H100 con TensorRT: x{lat['Jetson Orin']['trt'] / lat['H100']['trt']:.1f} más lento")
print("π0 (RTX 4090): 73 ms a bordo; ejecuta 25 acciones a 50 Hz = 0,5 s entre inferencias -> margen 0,5/0,073 = "
      f"{0.5 / 0.073:.1f}")
t2 = D["gr00t2_turbo_latencia_ms"]
for dev, v in t2.items():
    if dev.startswith("_"):
        continue
    print(f"GR00T 2 Turbo (2000 M), {dev}: {v['optimizado']} ms -> {1000 / v['optimizado']:.1f} Hz; "
          f"sin optimizar {v['eager_bf16']} ms -> x{v['eager_bf16'] / v['optimizado']:.1f} con TensorRT, FP8 y menos pasos")
print(f"  trozo de 1,6 s planificado ~1 vez por segundo: margen 1 s / 0,124 s = {1 / 0.12397:.1f}; "
      f"SONIC lo ejecuta a 50 Hz: {1.6 * 50:.0f} pasos por trozo")

# ---------------------------------------------------------------------------
titulo("4. Una tasa de éxito y su error (parte 4)")
n, p = 10, 0.8
se = math.sqrt(p * (1 - p) / n)
print(f"n = {n}, p̂ = {p}: SE = sqrt(p(1-p)/n) = {se:.3f}; normal ±1,96·SE = [{p - Z95 * se:.3f}; {p + Z95 * se:.3f}] (pasa de 1)")
for k, n, quien in [(8, 10, "π0, 10 episodios"), (24, 30, "humanos del estudio de π0.7"), (24, 25, "«96 %» con n = 25"),
                    (48, 50, "«96 %» con n = 50"), (96, 100, "«96 %» con n = 100"), (480, 500, "«96 %» con n = 500"),
                    (20, 20, "20/20"), (86, 86, "GEN-1, 86 seguidas")]:
    lo, hi = wilson(k, n)
    print(f"  Wilson {k:3d}/{n:<3d} ({quien:28s}) -> [{lo:.3f}; {hi:.3f}], anchura {hi - lo:.3f}")

# ---------------------------------------------------------------------------
titulo("5. Cuántos ensayos para distinguir 60 % de 70 % (parte 4)")
print(f"z_(1-α/2) = {Z95:.4f}, z_(1-β) = {Z80:.4f} (α = 0,05 bilateral, potencia 0,8)")
p1, p2 = 0.6, 0.7
pb = (p1 + p2) / 2
a = Z95 * math.sqrt(2 * pb * (1 - pb))
b = Z80 * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
print(f"p̄ = {pb}; z_a·sqrt(2p̄(1-p̄)) = {a:.4f}; z_b·sqrt(p1q1+p2q2) = {b:.4f}; suma² = {(a + b) ** 2:.4f}; Δ² = {(p2 - p1) ** 2:.4f}")
print(f"n = {n_potencia(p1, p2):.1f} -> {math.ceil(n_potencia(p1, p2))} por política ({2 * math.ceil(n_potencia(p1, p2))} en total)")
for q2 in (0.65, 0.75, 0.8, 0.9):
    print(f"  0,6 frente a {q2}: n = {n_potencia(0.6, q2):.0f}")
mde = biseccion(lambda d: n_potencia(0.5 - d / 2, 0.5 + d / 2) - 50, 0.05, 0.9)
print(f"con n = 50 por política (TRI), diferencia mínima detectable alrededor del 50 %: {mde * 100:.1f} puntos")
q = biseccion(lambda x: n_potencia(0.6, x) - 50, 0.61, 0.99)
print(f"con n = 50 y p1 = 0,6, la otra política tendría que llegar a {q:.3f}")
print("GR00T 2 (NVIDIA, 11-2026), nota 1: «two-sided two-proportion z-test at significance level 0.05 with 80% power»")
for p1, p2, dice in D["potencia_gr00t2"]["pares"]:
    print(f"  {p1} frente a {p2}: n = {n_potencia(p1, p2):.1f} -> {math.ceil(n_potencia(p1, p2))} por política "
          f"(el whitepaper: {'«more than 1,000»' if dice == 1000 else f'«about {dice}»'})")
for nn in (8, 15):
    p1, p2 = 0.09, 0.56
    pb = (p1 + p2) / 2
    print(f"  Helix 2.5, 9 % frente a 56 % con n = {nn} por brazo: z = {(p2 - p1) / math.sqrt(2 * pb * (1 - pb) / nn):.2f}")

# ---------------------------------------------------------------------------
titulo("6. La regla del tres, versión binomial (parte 4; R1b la usa por horas)")
alfa = 0.05
print(f"-ln α = -ln 0,05 = {-math.log(alfa):.4f} ≈ 3")
for nn in (20, 30, 86, 100, 300):
    print(f"  {nn}/{nn} éxitos: cota inferior exacta α^(1/n) = {alfa ** (1 / nn):.4f}; regla del tres 1 - 3/n = {1 - 3 / nn:.4f}")
print(f"ensayos sin fallo para afirmar p >= 0,99 al 95 %: exacto ln α/ln 0,99 = {math.log(alfa) / math.log(0.99):.1f} -> 299; regla del tres 3/0,01 = 300")

# ---------------------------------------------------------------------------
titulo("7. Leer la comparación de seguridad de Waymo (parte 6)")
w = D["waymo_hub"]
M = w["millas_M"]
for clave, nombre in (("grave_o_peor", "lesión grave o peor"), ("cualquier_lesion", "cualquier lesión"),
                      ("airbag_cualquier_vehiculo", "airbag en cualquier vehículo")):
    tasa, lo_h, hi_h = w[clave]["waymo"]
    k = round(tasa * M)
    lo, hi = poisson_exacto(k)
    print(f"  {nombre:30s} tasa {tasa} × {M} M mi = {tasa * M:.2f} -> k = {k}; IC exacto de Poisson [{lo:.3f}; {hi:.3f}] eventos")
    print(f"  {'':30s} -> [{lo / M:.5f}; {hi / M:.5f}] por millón de millas; el hub publica [{lo_h}; {hi_h}]")
h = w["grave_o_peor"]["humano"][0]
lo, hi = poisson_exacto(3)
print(f"razón de tasas, lesión grave, con la referencia humana tratada como conocida ({h}):")
print(f"  [{lo / M / h:.4f}; {hi / M / h:.4f}] -> reducción [{(lo / M / h - 1) * 100:.1f} %; {(hi / M / h - 1) * 100:.1f} %]; "
      f"el hub da {w['grave_o_peor']['reduccion'][1:]} (Clopper–Pearson con la incertidumbre de la referencia)")
lo, hi = poisson_exacto(7)
print(f"peatones: 7 frente a 93 esperados; IC de 7 eventos [{lo:.2f}; {hi:.2f}] -> razón [{lo / 93:.3f}; {hi / 93:.3f}], "
      f"reducción [{(lo / 93 - 1) * 100:.1f} %; {(hi / 93 - 1) * 100:.1f} %]")
print(f"con 0 lesiones graves, millas para acotar por debajo del 10 % de la tasa humana (Poisson, 3/T; R1b): {3 / (0.1 * h):.0f} M")
km = D["apollo_go"]["km_sin_conductor_M"]
print(f"Apollo Go: {km} M km sin conductor = {km / 1.609344:.1f} M millas (Waymo: {M} M millas a junio de 2026)")

# ---------------------------------------------------------------------------
titulo("8. Leer un anuncio con sus propias cifras: Figure en BMW (parte 5)")
f = D["figure_bmw"]
pph = f["piezas"] / f["horas"]
cph = pph / f["piezas_por_ciclo"]
print(f"{f['piezas']} piezas / {f['horas']} h = {pph:.1f} piezas/h = {cph:.1f} ciclos/h -> {3600 / cph:.0f} s por ciclo (requisito: {f['ciclo_req_s']} s)")
print(f"si todo el tiempo fuera a {f['ciclo_req_s']} s por ciclo: {f['horas'] * 3600 / f['ciclo_req_s'] * f['piezas_por_ciclo']:,.0f} piezas".replace(",", " "))
turnos = f["horas"] / f["turno_h"]
print(f"{f['horas']} h / {f['turno_h']} h por turno = {turnos:.0f} turnos = {turnos / 5:.0f} semanas de 5 días ≈ {turnos / 5 / 4.33:.1f} meses")
