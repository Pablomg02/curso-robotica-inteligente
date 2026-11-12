"""R1b · Drones autónomos en el mundo real: cuentas y figuras con datos.

Uso (desde esta carpeta):
    python3 figuras_r1b.py              # imprime los ejemplos numéricos (salida_ejemplos.txt)
    python3 figuras_r1b.py --insertar   # además escribe svg/*.svg y los inserta en ../index.html
                                        # entre <!--SVG:nombre--> y <!--/SVG:nombre-->

Solo numpy no hace falta: todo es aritmética de la biblioteca estándar.
Cada dato publicado lleva su fuente (la clave de referencia de index.html) y su estado:
C = confirmado en fuente primaria, E = según la empresa, P = prensa contrastada.
Lo que no publica nadie va marcado como SUPUESTO.
"""
import math
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
G = 9.80665                       # m/s², gravedad estándar
RHO = 1.225                       # kg/m³, aire a nivel del mar (SUPUESTO, como las pruebas de DJI)
C_LUZ = 299_792_458.0             # m/s
LB = 0.45359237                   # kg por libra
MPH = 0.44704                     # m/s por mph
KT = 0.514444                     # m/s por nudo
MILLA = 1609.344                  # m

# ---------------------------------------------------------------- datos publicados
MAVIC4 = dict(alcance=24.0, v_efectiva=18.0, inclinacion=35.0, v_sport=25.0,
              latencia_video=0.130)                     # djimavic4 (E)
X10 = dict(alcance=20.0, v_efectiva=16.0)               # skydiox10 (E)
M5TD = dict(vision=25.0, v_efectiva=15.0, lidar=110.0, radar=40.0)   # djidock4 (E)
FC30 = dict(m_carga=95.0, m_vacio=65.0, carga=30.0, d_rotor=1.375, ejes=4,
            e_bat=2 * 1984.4, t_est_carga=18.0, t_est_vacio=29.0,
            alcance_carga=16.0, alcance_vacio=28.0)     # djifc30, djifc30nota (E)
P1 = dict(m=50 * LB, v=60 * MPH, v_max=56 * KT, ida_vuelta=120 * MILLA)  # zipsparrow (C), zipfacts (E)
P2 = dict(m=63 * LB, v=47 * MPH)                        # zipdfwea (C)
MK30 = dict(m=83.2 * LB, v=52.4 * KT)                   # amzkc (C)
WING = dict(m=15 * LB, v=59 * MPH)                      # winghouston (C)
LD_SUP, ETA_SUP = 10.0, 0.6                             # SUPUESTOS: L/D y rendimiento propulsivo-eléctrico
LD_RANGO = (6.0, 15.0)                                  # SUPUESTO: barrido de L/D

# B3a (ejemplo 6 de su script): sesgos de TUM VI tras 1 h de paseo aleatorio, 1 sigma
B_ACEL = 5.16e-3          # m/s²
B_GIRO = 1.32e-4          # rad/s

# SPRIN-D, Tabla I de werner2026 (C): id, longitud (m), tiempo (s), RMSE odometría (m), RMSE método (m)
SPRIND = [(60, 830, 606, 8, 5), (61, 495, 343, 36, 13), (205, 939, 622, 31, 10),
          (206, 1023, 621, 27, 6), (208, 872, 500, 14, 11), (232, 1256, 1021, 53, 7),
          (239, 1371, 977, 8, 9), (242, 939, 622, 9, 11)]

# Objetivos de EASA, MoC Light-UAS.2510 (C), por hora de vuelo
OBJETIVOS = [(1e-4, "mayor (SAIL V y VI)"), (1e-5, "peligroso, SAIL V"),
             (1e-6, "peligroso, SAIL VI"), (1e-7, "catastrófico, SAIL V"),
             (1e-8, "catastrófico, SAIL VI")]
ZIPLINE_MILLAS = 150e6    # zip2026nov (E): «150 million commercial autonomous miles»
AMAZON_H = 908            # amz2024 (E, nota de mar-2025): 5166 vuelos, 908 h de prueba
AMAZON_H_DIC = 1070       # amz2024 (E, nota de dic-2024): «1,070 flight hours on more than 6,300 flights»
VBAT_H = 18_000           # reuters2026 (P): cifra dada por Shield AI, «18,000 flight hours since 2019»

# GNSS e interferencia
P_GPS_DBW = -158.5        # isgps200 (C): potencia mínima recibida de L1 C/A con antena de 3 dBi
F_L1 = 1575.42e6          # Hz
RC = 1.023e6              # chip/s, código C/A
CN0_UMBRAL = 30.0         # dB-Hz, umbral de seguimiento de fase usado por nrc1995 (C)
D_NRC_KM = 40.0           # nrc1995 (C): pérdida de portadora C/A a 40 km de un interferidor de 1 W
PIRE_DJI_DBM = 33.0       # djimavic4 (E): PIRE < 33 dBm a 2,4 GHz (FCC)

# Velocidades de las carreras autónomas (bahnam2026, introducción y discusión, C)
CARRERAS = [("2016", "IROS, Daejeon", 0.6), ("2017", "IROS, Vancouver", 0.7),
            ("2018", "IROS, Madrid", 2.0), ("2019", "IROS, Macao", 2.5),
            ("2019", "AlphaPilot, media", 6.8), ("2019", "AlphaPilot, pico", 9.2),
            ("2025", "MonoRace (A2RL), pico", 28.23)]

# NTSB WPR26LA001 (C): horas locales del 1-oct-2025 en Tolleson (AZ)
TOLLESON = [("05:30", "llega la grúa"), ("06:58", "escaneo de azoteas: nada"),
            ("07:30", "montaje de la grúa"), ("08:00", "grúa montada (~197 ft)"),
            ("09:24:40", "el dron se activa"), ("09:47:24", "despega"),
            ("09:49:03", "fin de los datos, a ~150 ft de la grúa")]


# ---------------------------------------------------------------- utilidades
def coma(x, d=2):
    s = f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", " ")
    return s


def f1(x):
    return f"{x:.1f}"


def db(x):
    return 10 * math.log10(x)


def fspl_db(d, f):
    """Pérdida en espacio libre (dB) entre antenas isótropas: (4 pi d f / c)^2."""
    return 20 * math.log10(4 * math.pi * d * f / C_LUZ)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def h(s, t):
    """Convierte 'hh:mm' o 'hh:mm:ss' a horas decimales."""
    p = [int(x) for x in t.split(":")]
    return p[0] + p[1] / 60 + (p[2] / 3600 if len(p) > 2 else 0)


# ---------------------------------------------------------------- ejemplos numéricos
def ejemplos():
    L = []
    w = L.append

    w("Ejemplo 1. Distancia de parada d = v·tau + v²/(2a), con a = g·tan(theta_max)")
    a = G * math.tan(math.radians(MAVIC4["inclinacion"]))
    w(f"  Mavic 4 Pro: theta_max = 35°, a = 9,807·tan 35° = {a:.3f} m/s²")
    for tau in (0.05, 0.1, 0.2, 0.3, 0.5):
        v = MAVIC4["v_efectiva"]
        d = v * tau + v * v / (2 * a)
        vmax = a * (-tau + math.sqrt(tau * tau + 2 * MAVIC4["alcance"] / a))
        w(f"  tau = {tau:.2f} s: a 18 m/s, d = {v*tau:.2f} + {v*v/(2*a):.2f} = {d:.2f} m; "
          f"v_max para parar en 24 m = {vmax:.2f} m/s")
    v = MAVIC4["v_sport"]
    w(f"  A 25 m/s (Sport, sin evitación) y tau = 0,1 s: d = {v*0.1 + v*v/(2*a):.2f} m")
    for tau in (0.05, 0.1, 0.2):
        v, R = X10["v_efectiva"], X10["alcance"]
        areq = v * v / (2 * (R - v * tau))
        w(f"  Skydio X10, tau = {tau:.2f} s: para parar desde 16 m/s en 20 m, a = {areq:.2f} m/s², "
          f"theta = {math.degrees(math.atan(areq / G)):.1f}°")
    v = M5TD["v_efectiva"]
    for tau in (0.1, 0.3):
        w(f"  Matrice 5TD a 15 m/s, tau = {tau} s, a = {a:.2f}: d = {v*tau + v*v/(2*a):.2f} m "
          f"(visión 25 m, radar 40 m, LiDAR 110 m)")
    w("")

    w("Ejemplo 2. Potencia en estacionario por cantidad de movimiento (FlyCart 30)")
    A = FC30["ejes"] * math.pi * (FC30["d_rotor"] / 2) ** 2
    w(f"  Área de disco: 4 discos coaxiales de 1,375 m: A = 4·pi·0,6875² = {A:.3f} m²")
    for m, t in ((FC30["m_carga"], FC30["t_est_carga"]), (FC30["m_vacio"], FC30["t_est_vacio"])):
        T = m * G
        vi = math.sqrt(T / (2 * RHO * A))
        Pi = T ** 1.5 / math.sqrt(2 * RHO * A)
        Pm = FC30["e_bat"] / (t / 60)
        w(f"  m = {m:.0f} kg: T = {T:.1f} N, v_i = {vi:.2f} m/s, P_i = T^1,5/sqrt(2 rho A) = {Pi:.0f} W; "
          f"batería {FC30['e_bat']:.1f} Wh en {t:.0f} min -> P = {Pm:.0f} W; eta_global = {Pi/Pm:.3f}; "
          f"{Pm/m:.1f} W/kg")
    w("")

    w("Ejemplo 3. Crucero de ala fija P = W·v/((L/D)·eta)  (L/D = 10 y eta = 0,6 SUPUESTOS)")
    for nombre, d in (("Zipline P1", P1), ("Zipline P2", P2), ("Amazon MK30", MK30), ("Wing 7000W-B", WING)):
        P = d["m"] * G * d["v"] / (LD_SUP * ETA_SUP)
        w(f"  {nombre}: m = {d['m']:.2f} kg, v = {d['v']:.2f} m/s -> P = {P:.0f} W ({P/d['m']:.1f} W/kg)")
    E_km = P1["m"] * G / (LD_SUP * ETA_SUP) * 1000 / 3600
    w(f"  P1, energía por km = m·g/((L/D)·eta) = {E_km:.2f} Wh/km; 120 millas = {P1['ida_vuelta']/1000:.1f} km "
      f"-> {E_km*P1['ida_vuelta']/1000:.0f} Wh; tiempo a 26,8 m/s = {P1['ida_vuelta']/P1['v']/3600:.2f} h")
    for LD in LD_RANGO:
        w(f"  P1 con L/D = {LD:.0f}: {P1['m']*G/(LD*ETA_SUP)*1000/3600:.2f} Wh/km")
    P_P1 = P1["m"] * G * P1["v"] / (LD_SUP * ETA_SUP)
    # Jetson AGX Thor T5000: «configurable 40- to 130-watt power range» (NVIDIA, nota del humanoide
    # de referencia, 31-05-2026; E). La misma cifra está en R1a.
    w(f"  Cómputo a bordo: un Jetson AGX Thor de 40–130 W (E) sería {40/P_P1:.1%}–{130/P_P1:.1%} "
      f"de los {P_P1:.0f} W de crucero del P1")
    w("  Energía por km y kg:")
    for etiqueta, e, km, m in (("FlyCart 30 con 30 kg", FC30["e_bat"], FC30["alcance_carga"], FC30["m_carga"]),
                               ("FlyCart 30 sin carga", FC30["e_bat"], FC30["alcance_vacio"], FC30["m_vacio"])):
        w(f"    {etiqueta}: {e:.1f} Wh / {km:.0f} km = {e/km:.1f} Wh/km -> {e/km/m:.2f} Wh/(km·kg)")
    for LD in (LD_RANGO[0], LD_SUP, LD_RANGO[1]):
        w(f"    ala fija, L/D = {LD:.0f}, eta = 0,6: g/((L/D)·eta) = {G/(LD*ETA_SUP)/3.6:.3f} Wh/(km·kg)")
    w("")

    w("Ejemplo 4. Deriva sin GNSS: inercial pura (sesgos de B3a) frente a odometría (SPRIN-D)")
    for t in (10, 60, 93.3):
        ea = 0.5 * B_ACEL * t * t
        eg = G * B_GIRO * t ** 3 / 6
        w(f"  t = {t:5.1f} s ({15*t:6.0f} m a 15 m/s): acel. {ea:7.2f} m, giróscopo {eg:7.2f} m, suma {ea+eg:7.2f} m")
    for fid, Lm, ts, ro, rm in SPRIND:
        w(f"  vuelo {fid}: {Lm} m en {ts} s ({Lm/ts:.2f} m/s); odometría {ro} m = {100*ro/Lm:.1f} %; "
          f"método {rm} m = {100*rm/Lm:.1f} %")
    ro = [100 * r[3] / r[1] for r in SPRIND]
    w(f"  deriva de la odometría: de {min(ro):.1f} a {max(ro):.1f} % de la distancia")
    w("")

    w("Ejemplo 5. Regla del tres por horas (Poisson): exp(-lambda·T) = alpha -> lambda_95 = ln 20 / T")
    w(f"  ln 20 = {math.log(20):.4f}")
    for lam, et in OBJETIVOS:
        w(f"  lambda = {lam:.0e} /h ({et}): T = ln 20/lambda = " + f"{math.log(20)/lam:,.0f} h".replace(",", " "))
    for mph in (47, 60):
        T = ZIPLINE_MILLAS / mph
        w(f"  Zipline: 150e6 millas a {mph} mph = {T:.3e} h -> lambda_95 = {math.log(20)/T:.2e} /h")
    for nombre, T in (("Amazon, prueba (mar-2025)", AMAZON_H), ("Amazon, prueba (dic-2024)", AMAZON_H_DIC)):
        w(f"  {nombre}: {T} h -> lambda_95 = {math.log(20)/T:.2e} /h (solo si no hubiera habido fallos)")
    w(f"  Con 0 fallos, la binomial exacta con p pequeño: (1-p)^n = alpha -> p ≈ -ln(alpha)/n (lo de R1a)")
    w("")

    w("Ejemplo 6. Interferencia: J/S con Friis (antenas isótropas, espacio libre)")
    S = P_GPS_DBW + 30
    w(f"  Señal GPS L1 C/A mínima: {P_GPS_DBW} dBW = {S:.1f} dBm")
    kTB = -174 + db(2 * RC)
    w(f"  Ruido térmico en 2,046 MHz: -174 + {db(2*RC):.1f} = {kTB:.1f} dBm -> la señal queda {kTB - S:.1f} dB "
      f"por debajo del ruido")
    for dkm in (1, 10, 40, 100):
        J = 30 - fspl_db(dkm * 1e3, F_L1)
        w(f"  Interferidor de 1 W a {dkm:3d} km: pérdida {fspl_db(dkm*1e3, F_L1):.1f} dB, J = {J:.1f} dBm, J/S = {J - S:.1f} dB")
    w(f"  Tolerancia del C/A: 10·log10(Rc) - (C/N0)_umbral = {db(RC):.2f} - {CN0_UMBRAL:.0f} = {db(RC)-CN0_UMBRAL:.1f} dB")
    dlim = 10 ** ((30 - S - (db(RC) - CN0_UMBRAL) - 20 * math.log10(4 * math.pi * F_L1 / C_LUZ)) / 20)
    w(f"  Distancia a la que J/S iguala esa tolerancia con 1 W: {dlim/1e3:.1f} km (NRC: 40 km)")
    for dS, dJ in ((5000, 500), (5000, 2000), (1000, 1000)):
        JS = (30 - PIRE_DJI_DBM) + 20 * math.log10(dS / dJ)
        w(f"  C2 a 2,4 GHz: mando de 33 dBm a {dS/1000:.0f} km, interferidor de 1 W a {dJ/1000:.1f} km: J/S = {JS:.1f} dB")
    w("")

    w("Ejemplo 7. Fiabilidad publicada de MonoRace (bahnam2026): Wilson al 95 %")
    for k, n in ((5, 7), (3, 3), (2, 4)):
        lo, hi = wilson(k, n)
        w(f"  {k}/{n}: p = {k/n:.3f}, Wilson [{lo:.3f}, {hi:.3f}]")
    w(f"  28,23 m/s = {28.23*3.6:.1f} km/h; 95,8 km/h = {95.8/3.6:.2f} m/s")
    w("")

    w("Ejemplo 8. Conversiones de las especificaciones")
    for nombre, d in (("Zipline P1", P1), ("Zipline P2", P2), ("Amazon MK30", MK30), ("Wing 7000W-B", WING)):
        w(f"  {nombre}: {d['m']:.2f} kg, {d['v']:.2f} m/s ({d['v']*3.6:.1f} km/h)")
    w(f"  Zipline P1, crucero máx. 56 kt = {P1['v_max']:.2f} m/s; MK30 crucero máx. 73 mph = {73*MPH:.2f} m/s")
    w(f"  Carga útil / masa máxima: Wing 7000W-B {2.7/15:.2f}; Wing 8000-A {5/25:.2f}; Zipline P2 {8/63:.2f}; "
      f"MK30 {5/83.2:.2f}; FlyCart 30 {30/95:.2f}; V-BAT {40/165:.2f}")
    return "\n".join(L)


# ---------------------------------------------------------------- figuras SVG
class Ejes:
    def __init__(self, x0, x1, y0, y1, X0, X1, Y0, Y1, logx=False, logy=False):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.X0, self.X1, self.Y0, self.Y1 = X0, X1, Y0, Y1
        self.logx, self.logy = logx, logy

    def X(self, v):
        if self.logx:
            v, a, b = math.log10(v), math.log10(self.x0), math.log10(self.x1)
        else:
            a, b = self.x0, self.x1
        return self.X0 + (v - a) / (b - a) * (self.X1 - self.X0)

    def Y(self, v):
        if self.logy:
            v, a, b = math.log10(v), math.log10(self.y0), math.log10(self.y1)
        else:
            a, b = self.y0, self.y1
        return self.Y0 - (v - a) / (b - a) * (self.Y0 - self.Y1)


def T(x, y, s, size=13, cls="", anchor="start"):
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{f1(x)}" y="{f1(y)}"{a} font-size="{size}"{c}>{s}</text>'


def linea(x1, y1, x2, y2, cls="ln", extra=""):
    return f'<line x1="{f1(x1)}" y1="{f1(y1)}" x2="{f1(x2)}" y2="{f1(y2)}" class="{cls}"{extra}/>'


def poli(pts, cls, extra=""):
    d = " ".join(f"{f1(x)},{f1(y)}" for x, y in pts)
    return f'<polyline points="{d}" fill="none" class="{cls}"{extra}/>'


def svg(nombre, w, hgt, cuerpo, aria, maxh=None):
    st = f' style="max-height:{maxh}px"' if maxh else ""
    return (f'<svg viewBox="0 0 {w} {hgt}"{st} role="img" aria-label="{aria}">\n'
            + "\n".join(cuerpo) + "\n</svg>")


def fig_parada():
    a = G * math.tan(math.radians(MAVIC4["inclinacion"]))
    E = Ejes(0, 26, 0, 60, 70, 540, 290, 20)
    s = [linea(E.X0, E.Y0, E.X1, E.Y0), linea(E.X0, E.Y0, E.X0, E.Y1)]
    for v in range(0, 27, 5):
        s += [linea(E.X(v), E.Y0, E.X(v), E.Y0 + 5), T(E.X(v), E.Y0 + 20, str(v), 12, "t-muted", "middle")]
    for d in range(0, 61, 10):
        s += [linea(E.X0 - 5, E.Y(d), E.X0, E.Y(d)), T(E.X0 - 9, E.Y(d) + 4, str(d), 12, "t-muted", "end")]
        if d:
            s.append(linea(E.X0, E.Y(d), E.X1, E.Y(d), "ln-rule"))
    s.append(T((E.X0 + E.X1) / 2, E.Y0 + 38, "velocidad v (m/s)", 13, "t-muted", "middle"))
    s.append(f'<text transform="translate(22,{f1((E.Y0+E.Y1)/2)}) rotate(-90)" text-anchor="middle" font-size="13" class="t-muted">distancia de parada d (m)</text>')
    for R, et, cls in ((MAVIC4["alcance"], "Mavic 4 Pro: mide hasta 24 m", "ln-accent"),
                       (X10["alcance"], "X10: percibe hasta 20 m", "ln-tesis")):
        s.append(linea(E.X0, E.Y(R), E.X1, E.Y(R), cls + " ln-dash"))
    s.append(T(E.X(0.6), E.Y(24) - 6, "Mavic 4 Pro: mide hasta 24 m", 12, "t-accent"))
    s.append(T(E.X(0.6), E.Y(20) + 15, "Skydio X10: percibe hasta 20 m", 12, "t-tesis"))
    for tau, cls, lab in ((0.05, "ln", "τ = 0,05 s"), (0.1, "ln-time", "τ = 0,1 s"), (0.3, "ln-accent", "τ = 0,3 s")):
        pts = [(E.X(v), E.Y(v * tau + v * v / (2 * a))) for v in [i * 0.25 for i in range(0, 105)]
               if v * tau + v * v / (2 * a) <= 60]
        s.append(poli(pts, cls, ' stroke-width="2.2"'))
        vlab = max(v for v in [i * 0.25 for i in range(0, 105)] if v * tau + v * v / (2 * a) <= 58)
        s.append(T(E.X(vlab) + 4, E.Y(58) + 2 + {0.05: 0, 0.1: 16, 0.3: 32}[tau], lab, 12,
                   {"ln": "t-muted", "ln-time": "t-time", "ln-accent": "t-accent"}[cls]))
    d18 = 18 * 0.1 + 18 * 18 / (2 * a)
    s.append(f'<circle cx="{f1(E.X(18))}" cy="{f1(E.Y(d18))}" r="5" class="fill-time"/>')
    s.append(T(E.X(18) + 8, E.Y(d18) + 18, f"18 m/s → {coma(d18,1)} m", 12, "t-time"))
    d25 = 25 * 0.1 + 25 * 25 / (2 * a)
    s.append(f'<circle cx="{f1(E.X(25))}" cy="{f1(E.Y(d25))}" r="5" class="fill-ink3"/>')
    s.append(T(E.X(25) - 8, E.Y(d25) + 18, f"25 m/s (Sport) → {coma(d25,0)} m", 12, "t-muted", "end"))
    s.append(T(E.X(0.6), E.Y(56), f"a = g·tan 35° = {coma(a,2)} m/s²", 12, "t-mono"))
    return svg("parada", 560, 335, s, "Distancia de parada frente a velocidad para tres latencias, con los alcances del Mavic 4 Pro y del Skydio X10", 300)


def fig_energia():
    filas = []
    for et, e, km, m in (("FlyCart 30, 30 kg de carga (95 kg)", FC30["e_bat"], FC30["alcance_carga"], FC30["m_carga"]),
                         ("FlyCart 30, sin carga (65 kg)", FC30["e_bat"], FC30["alcance_vacio"], FC30["m_vacio"])):
        filas.append((et, e / km / m, None, "fill-time", "dato de DJI: batería / alcance"))
    lo = G / (LD_RANGO[1] * ETA_SUP) / 3.6
    hi = G / (LD_RANGO[0] * ETA_SUP) / 3.6
    mid = G / (LD_SUP * ETA_SUP) / 3.6
    filas.append(("Ala fija (P1, P2, MK30, Wing en crucero)", mid, (lo, hi), "fill-accent", "supuestos: L/D 6–15, η = 0,6"))
    E = Ejes(0, 3.0, 0, 1, 290, 540, 0, 0)
    s = [linea(E.X(0), 30, E.X(0), 230)]
    for x in (0, 0.5, 1, 1.5, 2, 2.5, 3):
        s += [linea(E.X(x), 230, E.X(x), 235), T(E.X(x), 250, coma(x, 1), 12, "t-muted", "middle")]
        if x:
            s.append(linea(E.X(x), 30, E.X(x), 230, "ln-rule"))
    s.append(T((E.X(0) + E.X(3)) / 2, 272, "energía por kilómetro y kilogramo, Wh/(km·kg)", 13, "t-muted", "middle"))
    for i, (et, v, rng, cls, nota) in enumerate(filas):
        y = 48 + i * 64
        s.append(T(E.X(0) - 10, y + 14, et, 13, "", "end"))
        s.append(T(E.X(0) - 10, y + 31, nota, 11, "t-faint", "end"))
        s.append(f'<rect x="{f1(E.X(0))}" y="{y}" width="{f1(E.X(v)-E.X(0))}" height="26" rx="3" class="{cls}" opacity="0.85"/>')
        s.append(T(E.X(v) + 6, y + 18, coma(v, 2), 13, "t-mono"))
        if rng:
            s.append(linea(E.X(rng[0]), y + 13, E.X(rng[1]), y + 13, "ln", ' stroke-width="2"'))
            for r in rng:
                s.append(linea(E.X(r), y + 5, E.X(r), y + 21, "ln", ' stroke-width="2"'))
            s[-4] = T(E.X(rng[1]) + 6, y + 18, f"{coma(v,2)} ({coma(rng[0],2)}–{coma(rng[1],2)})", 13, "t-mono")
    return svg("energia", 560, 285, s, "Energía por kilómetro y kilogramo de un multirrotor de reparto frente a un ala fija", 270)


def fig_deriva():
    E = Ejes(0, 1400, 0, 120, 70, 540, 270, 20)
    s = [linea(E.X0, E.Y0, E.X1, E.Y0), linea(E.X0, E.Y0, E.X0, E.Y1)]
    for x in range(0, 1401, 200):
        s += [linea(E.X(x), E.Y0, E.X(x), E.Y0 + 5), T(E.X(x), E.Y0 + 19, str(x), 12, "t-muted", "middle")]
    for y in range(0, 121, 20):
        s += [linea(E.X0 - 5, E.Y(y), E.X0, E.Y(y)), T(E.X0 - 9, E.Y(y) + 4, str(y), 12, "t-muted", "end")]
        if y:
            s.append(linea(E.X0, E.Y(y), E.X1, E.Y(y), "ln-rule"))
    s.append(T((E.X0 + E.X1) / 2, E.Y0 + 36, "distancia recorrida (m); la inercial, a 15 m/s", 13, "t-muted", "middle"))
    s.append(f'<text transform="translate(22,{f1((E.Y0+E.Y1)/2)}) rotate(-90)" text-anchor="middle" font-size="13" class="t-muted">error de posición (m)</text>')
    # banda de la odometría 0,6 %–7,3 %
    lo, hi = 0.006, 0.073
    xs = [i * 10 for i in range(0, 141)]
    banda = [(E.X(x), E.Y(min(hi * x, 120))) for x in xs] + [(E.X(x), E.Y(lo * x)) for x in reversed(xs)]
    s.append('<polygon points="' + " ".join(f"{f1(a)},{f1(b)}" for a, b in banda) + '" class="fill-accent" opacity="0.13"/>')
    s.append(T(E.X(1110), E.Y(96), "odometría: 0,6–7,3 %", 12, "t-accent"))
    # inercial pura a 15 m/s
    pts = []
    for x in xs:
        t = x / 15
        e = 0.5 * B_ACEL * t * t + G * B_GIRO * t ** 3 / 6
        if e > 120:
            break
        pts.append((E.X(x), E.Y(e)))
    s.append(poli(pts, "ln-time", ' stroke-width="2.4"'))
    s.append(T(pts[-1][0] - 6, E.Y(112), "inercial pura (B3a)", 12, "t-time", "end"))
    # puntos SPRIN-D
    for fid, Lm, ts, ro, rm in SPRIND:
        s.append(f'<circle cx="{f1(E.X(Lm))}" cy="{f1(E.Y(ro))}" r="4.5" class="fill-accent"/>')
        s.append(f'<rect x="{f1(E.X(Lm)-4)}" y="{f1(E.Y(rm)-4)}" width="8" height="8" class="fill-tesis"/>')
    s.append(f'<circle cx="{f1(E.X(80))}" cy="{f1(E.Y(110))}" r="4.5" class="fill-accent"/>')
    s.append(T(E.X(80) + 10, E.Y(110) + 4, "SPRIN-D, odometría alineada con brújula", 12))
    s.append(f'<rect x="{f1(E.X(80)-4)}" y="{f1(E.Y(98)-4)}" width="8" height="8" class="fill-tesis"/>')
    s.append(T(E.X(80) + 10, E.Y(98) + 4, "SPRIN-D, con el mapa de alturas (5–13 m)", 12))
    return svg("deriva", 560, 315, s, "Error de posición sin GNSS: inercial pura frente a odometría y frente a la corrección por mapa de SPRIN-D", 290)


def fig_tres():
    E = Ejes(1e2, 1e9, 0, 1, 40, 540, 0, 0, logx=True)
    s = []
    y0 = 246
    s.append(linea(E.X(1e2), y0, E.X(1e9), y0))
    for k in range(2, 10):
        x = E.X(10 ** k)
        s += [linea(x, y0, x, y0 + 5), T(x, y0 + 20, f"10<tspan baseline-shift=\"super\" font-size=\"9\">{k}</tspan>", 12, "t-muted", "middle"),
              linea(x, 26, x, y0, "ln-rule")]
    s.append(T((E.X(1e2) + E.X(1e9)) / 2, y0 + 40, "horas de vuelo (escala logarítmica)", 13, "t-muted", "middle"))
    s.append(T(E.X(1e2), 20, "Horas sin fallos para acotar λ al 95 %:  T = ln 20 / λ", 13, "t-bold"))
    for i, (lam, et) in enumerate(OBJETIVOS):
        Tn = math.log(20) / lam
        y = 50 + i * 25
        s.append(linea(E.X(1e2), y, E.X(Tn), y, "ln-accent", ' stroke-width="7" opacity="0.55"'))
        s.append(T(E.X(Tn) - 2, y - 7, f"10<tspan baseline-shift=\"super\" font-size=\"8\">−{int(round(-math.log10(lam)))}</tspan> /h · {et}", 11, "", "end"))
    yf = 182
    s.append(T(E.X(1e2), yf - 8, "Experiencia publicada", 13, "t-bold"))
    marcas = [(AMAZON_H, "Amazon, pruebas (E)", "fill-time"), (VBAT_H, "V-BAT desde 2019 (E, vía Reuters)", "fill-time"),
              (ZIPLINE_MILLAS / 60, "", "fill-tesis"), (ZIPLINE_MILLAS / 47, "", "fill-tesis")]
    for Tn, et, cls in marcas:
        s.append(f'<circle cx="{f1(E.X(Tn))}" cy="{yf+12}" r="6" class="{cls}"/>')
    s.append(T(E.X(AMAZON_H) + 8, yf + 34, "Amazon, pruebas", 11, "t-time", "end"))
    s.append(T(E.X(AMAZON_H) + 8, yf + 47, "908 h (E)", 11, "t-time", "end"))
    s.append(T(E.X(VBAT_H) - 8, yf + 34, "V-BAT desde 2019", 11, "t-time"))
    s.append(T(E.X(VBAT_H) - 8, yf + 47, "18 000 h (E, vía Reuters)", 11, "t-time"))
    s.append(T(E.X(ZIPLINE_MILLAS / 60) - 6, yf + 34, "Zipline, 150 M millas", 11, "t-tesis"))
    s.append(T(E.X(ZIPLINE_MILLAS / 60) - 6, yf + 47, "≈ 2,5–3,2 M h (E)", 11, "t-tesis"))
    return svg("tres", 560, 294, s, "Horas necesarias para acotar las tasas de fallo de EASA frente a la experiencia publicada de varias flotas", 320)


def fig_js():
    S = P_GPS_DBW + 30
    E = Ejes(0.1, 200, 0, 90, 70, 540, 270, 20, logx=True)
    s = [linea(E.X0, E.Y0, E.X1, E.Y0), linea(E.X0, E.Y0, E.X0, E.Y1)]
    for d in (0.1, 1, 10, 100):
        s += [linea(E.X(d), E.Y0, E.X(d), E.Y0 + 5), T(E.X(d), E.Y0 + 19, coma(d, 1 if d < 1 else 0), 12, "t-muted", "middle"),
              linea(E.X(d), E.Y0, E.X(d), E.Y1, "ln-rule")]
    for y in range(0, 91, 15):
        s += [linea(E.X0 - 5, E.Y(y), E.X0, E.Y(y)), T(E.X0 - 9, E.Y(y) + 4, str(y), 12, "t-muted", "end")]
    s.append(T((E.X0 + E.X1) / 2, E.Y0 + 36, "distancia del interferidor al receptor (km)", 13, "t-muted", "middle"))
    s.append(f'<text transform="translate(22,{f1((E.Y0+E.Y1)/2)}) rotate(-90)" text-anchor="middle" font-size="13" class="t-muted">J/S (dB)</text>')
    pts = []
    for i in range(0, 101):
        d = 10 ** (-1 + i * (math.log10(200) + 1) / 100)
        js = 30 - fspl_db(d * 1e3, F_L1) - S
        if js <= 90:
            pts.append((E.X(d), E.Y(max(js, 0))))
    s.append(poli(pts, "ln-time", ' stroke-width="2.4"'))
    tol = db(RC) - CN0_UMBRAL
    s.append(linea(E.X0, E.Y(tol), E.X1, E.Y(tol), "ln-accent ln-dash"))
    s.append(T(E.X(0.12), E.Y(tol) + 16, f"tolerancia ≈ {coma(tol,1)} dB = 10·log Rc − 30 dB-Hz", 12, "t-accent"))
    for d in (1, 10, D_NRC_KM):
        js = 30 - fspl_db(d * 1e3, F_L1) - S
        s.append(f'<circle cx="{f1(E.X(d))}" cy="{f1(E.Y(js))}" r="5" class="fill-time"/>')
        s.append(T(E.X(d) + 8, E.Y(js) - 8, f"{coma(d,0)} km: {coma(js,0)} dB", 12, "t-time"))
    s.append(T(E.X(0.12), E.Y(84), "GPS L1 C/A, −128,5 dBm; interferidor de 1 W", 12, "t-mono"))
    return svg("js", 560, 315, s, "Relación interferencia a señal de GPS frente a la distancia a un interferidor de 1 W", 250)


def fig_carreras():
    E = Ejes(0, 1, 0.3, 40, 0, 0, 230, 30, logy=True)
    s = []
    n = len(CARRERAS)
    x0, paso = 80, 64
    for y in (0.5, 1, 2, 5, 10, 20):
        s += [linea(70, E.Y(y), 540, E.Y(y), "ln-rule"), T(64, E.Y(y) + 4, coma(y, 1 if y < 1 else 0), 12, "t-muted", "end")]
    s.append(linea(70, E.Y(0.3), 540, E.Y(0.3)))
    s.append(f'<text transform="translate(22,130) rotate(-90)" text-anchor="middle" font-size="13" class="t-muted">velocidad (m/s, log)</text>')
    for i, (anio, et, v) in enumerate(CARRERAS):
        x = x0 + i * paso
        cls = "fill-tesis" if anio == "2025" else ("fill-accent" if "AlphaPilot" in et else "fill-ink3")
        s.append(f'<rect x="{x}" y="{f1(E.Y(v))}" width="38" height="{f1(E.Y(0.3)-E.Y(v))}" rx="2" class="{cls}" opacity="0.85"/>')
        s.append(T(x + 19, E.Y(v) - 6, coma(v, 2 if v > 20 else 1), 12, "t-mono", "middle"))
        s.append(T(x + 19, E.Y(0.3) + 16, anio, 12, "t-muted", "middle"))
        corto = et.replace("IROS, ", "").replace("AlphaPilot, ", "AP ").replace("MonoRace (A2RL), pico", "MonoRace")
        s.append(T(x + 19, E.Y(0.3) + 31, corto, 11, "t-faint", "middle"))
    return svg("carreras", 560, 275, s, "Velocidades de las carreras de drones autónomos de 2016 a 2025", 260)


def fig_sprind():
    E = Ejes(0, 1, 0, 60, 0, 0, 240, 30)
    s = []
    for y in range(0, 61, 10):
        s += [linea(70, E.Y(y), 540, E.Y(y), "ln-rule" if y else "ln"), T(64, E.Y(y) + 4, str(y), 12, "t-muted", "end")]
    s.append(f'<text transform="translate(22,135) rotate(-90)" text-anchor="middle" font-size="13" class="t-muted">RMSE de posición (m)</text>')
    paso = 58
    for i, (fid, Lm, ts, ro, rm) in enumerate(SPRIND):
        x = 86 + i * paso
        s.append(f'<rect x="{x}" y="{f1(E.Y(ro))}" width="20" height="{f1(E.Y(0)-E.Y(ro))}" class="fill-accent" opacity="0.8"/>')
        s.append(f'<rect x="{x+22}" y="{f1(E.Y(rm))}" width="20" height="{f1(E.Y(0)-E.Y(rm))}" class="fill-tesis" opacity="0.85"/>')
        s.append(T(x + 10, E.Y(ro) - 5, str(ro), 11, "t-mono", "middle"))
        s.append(T(x + 32, E.Y(rm) - 5, str(rm), 11, "t-mono", "middle"))
        s.append(T(x + 21, E.Y(0) + 16, str(fid), 12, "t-muted", "middle"))
        s.append(T(x + 21, E.Y(0) + 30, f"{Lm} m", 11, "t-faint", "middle"))
    s.append(f'<rect x="92" y="38" width="12" height="12" class="fill-accent"/>')
    s.append(T(110, 49, "odometría alineada con brújula", 12))
    s.append(f'<rect x="92" y="58" width="12" height="12" class="fill-tesis"/>')
    s.append(T(110, 69, "con el mapa de alturas y el filtro", 12))
    return svg("sprind", 560, 278, s, "Error de posición de los ocho vuelos de SPRIN-D con odometría y con el método de corrección", 270)


def fig_tolleson():
    E = Ejes(5.0, 10.0, 0, 1, 40, 860, 0, 0)
    s = []
    y = 70
    s.append(linea(E.X(5), y, E.X(10), y, "ln", ' stroke-width="2"'))
    for hh in range(5, 11):
        s += [linea(E.X(hh), y - 5, E.X(hh), y + 5), T(E.X(hh), y + 22, f"{hh:02d}:00", 12, "t-muted", "middle")]
    s.append(f'<rect x="{f1(E.X(h(0,"07:30")))}" y="{y-14}" width="{f1(E.X(h(0,"08:00"))-E.X(h(0,"07:30")))}" height="28" class="fill-time" opacity="0.35"/>')
    s.append(f'<rect x="{f1(E.X(h(0,"08:00")))}" y="{y-6}" width="{f1(E.X(h(0,"09:49:03"))-E.X(h(0,"08:00")))}" height="12" class="fill-time" opacity="0.18"/>')
    eventos = [("05:30", "llega la grúa", -1, "t-muted"), ("06:58", "escaneo de azoteas: nada", 1, "t-accent"),
               ("07:30", "montaje de la grúa (07:30–08:00)", -1, "t-time"), ("09:24:40", "el dron se activa", 1, "t-muted"),
               ("09:49:03", "fin de datos a ~150 ft de la grúa", -1, "t-time")]
    for t, et, lado, cls in eventos:
        x = E.X(h(0, t))
        yy = y - 26 if lado < 0 else y + 44
        s.append(linea(x, y, x, yy + (8 if lado < 0 else -12), "ln-dash"))
        s.append(f'<circle cx="{f1(x)}" cy="{y}" r="5" class="{ "fill-time" if "time" in cls else ("fill-accent" if "accent" in cls else "fill-ink3")}"/>')
        anchor = "end" if x > 700 else "middle"
        s.append(T(x, yy, f"{t} · {et}", 12, cls, anchor))
    s.append(T(E.X(8.9), y + 70, "la grúa no está en el mapa del día: entre escaneo y vuelo pasan casi 3 h", 12, "t-muted", "middle"))
    return svg("tolleson", 900, 150, s, "Cronología del incidente de Tolleson según el informe preliminar del NTSB", 150)


def fig_poisson():
    """P(0 fallos en T horas) = exp(-lambda·T) frente a lambda·T, con alpha = 0,05."""
    E = Ejes(0, 5, 0, 1, 70, 540, 230, 20)
    s = [linea(E.X0, E.Y0, E.X1, E.Y0), linea(E.X0, E.Y0, E.X0, E.Y1)]
    for x in range(0, 6):
        s += [linea(E.X(x), E.Y0, E.X(x), E.Y0 + 5), T(E.X(x), E.Y0 + 19, str(x), 12, "t-muted", "middle")]
    for y in (0, 0.25, 0.5, 0.75, 1):
        s += [linea(E.X0 - 5, E.Y(y), E.X0, E.Y(y)), T(E.X0 - 9, E.Y(y) + 4, coma(y, 2), 12, "t-muted", "end")]
    s.append(T((E.X0 + E.X1) / 2, E.Y0 + 36, "λT (fallos esperados en T horas)", 13, "t-muted", "middle"))
    s.append(f'<text transform="translate(20,{f1((E.Y0+E.Y1)/2)}) rotate(-90)" text-anchor="middle" font-size="13" class="t-muted">P(0 fallos)</text>')
    s.append(poli([(E.X(i * 0.05), E.Y(math.exp(-i * 0.05))) for i in range(0, 101)], "ln-accent", ' stroke-width="2.4"'))
    s.append(linea(E.X0, E.Y(0.05), E.X1, E.Y(0.05), "ln-time ln-dash"))
    s.append(T(E.X(3.6), E.Y(0.05) - 7, "α = 0,05", 12, "t-time"))
    x3 = math.log(20)
    s.append(linea(E.X(x3), E.Y0, E.X(x3), E.Y(0.05), "ln-time ln-dash"))
    s.append(f'<circle cx="{f1(E.X(x3))}" cy="{f1(E.Y(0.05))}" r="5" class="fill-time"/>')
    s.append(T(E.X(x3) + 8, E.Y(0.25), f"λT = ln 20 = {coma(x3,3)} ≈ 3", 12, "t-time"))
    s.append(T(E.X(1.1), E.Y(0.75), "exp(−λT): cuanto mayor λ, más raro ver 0 fallos", 12, "t-accent"))
    return svg("poisson", 560, 275, s, "Probabilidad de no ver ningún fallo frente al número esperado de fallos", 225)


ANATOMIA = {
    # Un «~» delante marca lo que la empresa no publica (gris). (E) = según la empresa.
    "anat-dji": dict(
        sens=["visión binocular 360°", "LiDAR frontal (noche)", "IR hacia abajo", "GNSS: GPS, Galileo,", "BeiDou"],
        est=["posición visual", "±0,3 m horizontal", "satélite ±0,5 m", "~estimador: no publicado"],
        aut=["evita obstáculos", "a ≤ 18 m/s", "mide 0,5–24 m delante", "detecta hasta 200 m", "~planificador: no publ."],
        ctl=["inclinación máx. 35°", "25 m/s en Sport", "(sin evitación)", "~lazos: no publicados"],
        comp="~Cómputo de a bordo: no publicado",
        enl=["O4+: 30 km (FCC), 15 km (CE)", "urbano con interferencia: 1,5–6 km", "vídeo ≈ 130 ms; PIRE < 33 dBm"],
        op=["consumo, con piloto", "1063 g, 51 min máx.", "luz > 0,1 lux y textura"]),
    "anat-skydio": dict(
        sens=["6 cámaras de 32 MP", "trinocular arriba", "y abajo, 200°", "GNSS"],
        est=["VIO: ±10 cm", "GNSS: ±1 m", "X10D: sin GPS (E)", "~estimador: no publicado"],
        aut=["evitación 360°", "percibe hasta 20 m", "16 m/s con evitación", "~planificador: no publ."],
        ctl=["20 m/s máx.", "viento 12,8 m/s", "~lazos: no publicados"],
        comp="Cómputo: NVIDIA Jetson Orin + Snapdragon 865 (sin TOPS)",
        enl=["Connect SL: 12 km (WiFi 6)", "Connect 5G: donde haya cobertura", "~latencia: no publicada"],
        op=["docks: > 1000 en policía (E)", "≈ 100 000 misiones/mes (E)", "2,11 kg; 40 min (laboratorio)"]),
    "anat-zipline": dict(
        sens=["GNSS", "DAA a bordo (E)", "~resto: no publicado"],
        est=["~no publicado"],
        aut=["ruta generada", "justo antes del vuelo", "desconflicción V2V", "sigue sin enlace", "fuera de zona: termina"],
        ctl=["5 hélices; ala de 2,4 m", "47 mph a 330 ft", "droid por cable", "paracaídas («paraland»)", "~lazos: no publicados"],
        comp="~Cómputo de a bordo: no publicado",
        enl=["radio y red móvil", "V2V: posición, velocidad, intención", "piloto remoto (RPIC) que supervisa"],
        op=["hasta 400 vuelos/día por sitio", "radio de 10 millas", "< 63 lb con 8 lb de carga"]),
    "anat-wing": dict(
        sens=["GNSS", "ADS-B (para el DAA)", "~resto: no publicado"],
        est=["«múltiples sistemas", "de navegación» (E)", "mapa del nest", "(vuelos GeoBIT)"],
        aut=["rutas predefinidas", "asignación y", "desconflicción", "automáticas; DAA", "por ADS-B (E)"],
        ctl=["16 hélices (7000W-B)", "59 mph, 150–300 ft", "estacionario a 23 ft", "cable que se suelta"],
        comp="~Cómputo de a bordo: no publicado",
        enl=["~enlace: no publicado", "rutas que asigna el software de Wing", "software de rutas: cada mes"],
        op=["nests de hasta 24 aeronaves", "radio de 6 millas (energía)", "< 15 lb con 2,7 lb de carga"]),
    "anat-vbat": dict(
        sens=["cámara EO/IR (carga)", "magnetómetro", "(en investigación)", "GNSS cuando lo hay"],
        est=["visión: hitos contra", "imagen de satélite (E)", "ventana deslizante", "(de Nova 2) (E)"],
        aut=["Hivemind a bordo", "sin enlace: vuelve", "a cobertura (E)", "en equipo: comparte", "world models (E)"],
        ctl=["despegue y aterrizaje", "verticales", "motor de combustible", "pesado (JP5)", "~lazos: no publicados"],
        comp="~Cómputo: a bordo, sin especificar",
        enl=["MPU5: 130 km; banda C: 180 km", "operó con GPS y enlace", "interferidos, según la empresa"],
        op=["12+ h con EO/IR", "75 kg MTOW, 18,1 kg de carga", "aterriza en 4,6 × 4,6 m"]),
}


def fig_anatomia(nombre):
    d = ANATOMIA[nombre]
    s = []
    cols = [("Sensores", "sens", "box-accent", "t-accent"), ("Estimación", "est", "box-time", "t-time"),
            ("Autonomía", "aut", "box-tesis", "t-tesis"), ("Control", "ctl", "box", "")]
    x0, w, gap, y0, hb = 6, 140, 12, 8, 152
    for i, (tit, k, cls, tcls) in enumerate(cols):
        x = x0 + i * (w + gap)
        s.append(f'<rect x="{x}" y="{y0}" width="{w}" height="{hb}" rx="8" class="{cls}"/>')
        s.append(T(x + w / 2, y0 + 20, tit, 14, "t-bold " + tcls if tcls else "t-bold", "middle"))
        for j, it in enumerate(d[k]):
            faint = it.startswith("~")
            s.append(T(x + 8, y0 + 42 + j * 19, it.lstrip("~"), 12, "t-faint" if faint else ""))
        if i < 3:
            xa = x + w
            s.append(linea(xa + 1, y0 + hb / 2, xa + gap - 2, y0 + hb / 2, "ln"))
            s.append(f'<polygon points="{xa+gap-2},{y0+hb/2} {xa+gap-7},{y0+hb/2-4} {xa+gap-7},{y0+hb/2+4}" class="fill-ink3"/>')
    yc = y0 + hb + 8
    comp = d["comp"]
    s.append(f'<rect x="{x0}" y="{yc}" width="{4*w+3*gap}" height="26" rx="6" class="box-soft"/>')
    s.append(T(x0 + (4 * w + 3 * gap) / 2, yc + 18, comp.lstrip("~"), 12, "t-faint" if comp.startswith("~") else "t-bold", "middle"))
    yb = yc + 34
    wb = (4 * w + 3 * gap - gap) / 2
    for i, (tit, k) in enumerate((("Enlace", "enl"), ("Operación", "op"))):
        x = x0 + i * (wb + gap)
        s.append(f'<rect x="{f1(x)}" y="{yb}" width="{f1(wb)}" height="82" rx="8" class="box-soft"/>')
        s.append(T(x + 8, yb + 18, tit, 13, "t-bold"))
        for j, it in enumerate(d[k]):
            faint = it.startswith("~")
            s.append(T(x + 8, yb + 37 + j * 17, it.lstrip("~"), 12, "t-faint" if faint else ""))
    return svg(nombre, 600, yb + 88, s, f"Arquitectura publicada: sensores, estimación, autonomía, control, cómputo, enlace y operación ({nombre[5:]})", 330)


def fig_sprind_sistema():
    s = []
    def caja(x, y, w, hh, cls, lineas, tcls=""):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{hh}" rx="8" class="{cls}"/>')
        for j, (t_, c_) in enumerate(lineas):
            s.append(T(x + w / 2, y + 19 + j * 17, t_, 12, c_, "middle"))
    def flecha(x1, y1, x2, y2):
        s.append(linea(x1, y1, x2, y2, "ln"))
        ang = math.atan2(y2 - y1, x2 - x1)
        p1 = (x2 - 7 * math.cos(ang - 0.4), y2 - 7 * math.sin(ang - 0.4))
        p2 = (x2 - 7 * math.cos(ang + 0.4), y2 - 7 * math.sin(ang + 0.4))
        s.append(f'<polygon points="{f1(x2)},{f1(y2)} {f1(p1[0])},{f1(p1[1])} {f1(p2[0])},{f1(p2[1])}" class="fill-ink3"/>')
    caja(6, 10, 150, 58, "box-accent", [("LiDAR Livox Mid-360", "t-bold"), ("nube de puntos", "")])
    caja(6, 82, 150, 58, "box-accent", [("IMU + cámara D435", "t-bold"), ("VIO (odometría)", "")])
    caja(6, 154, 150, 58, "box-accent", [("brújula", "t-bold"), ("sesgo de hasta 30°", "t-faint")])
    caja(196, 10, 160, 58, "box-time", [("mapa de alturas local", "t-bold"), ("celdas de 1 m", "")])
    caja(196, 186, 160, 70, "box-soft", [("mapa de alturas previo", "t-bold"), ("de nubes de puntos", ""), ("abiertas (precalculado)", "t-faint")])
    caja(396, 30, 196, 110, "box-tesis", [("filtro de partículas", "t-bold t-tesis"), ("por grupos", "t-bold t-tesis"),
                                         ("predice con odometría", ""), ("y brújula; pesa con la", ""), ("correlación de bordes > 5 m", "")])
    caja(396, 176, 196, 44, "box", [("posición → planificador", "t-bold"), ("CPU solo (Intel NUC i7)", "t-faint")])
    flecha(156, 39, 196, 39)
    flecha(356, 39, 396, 60)
    flecha(156, 111, 396, 95)
    flecha(156, 183, 396, 118)
    flecha(356, 205, 396, 134)
    flecha(494, 140, 494, 176)
    return svg("sprind-sistema", 600, 262, s, "Sistema ganador de SPRIN-D: LiDAR, odometría y brújula fusionados con un mapa de alturas previo en un filtro de partículas", 250)


FIGURAS = {"poisson": fig_poisson, "sprind-sistema": fig_sprind_sistema,
           "anat-dji": lambda: fig_anatomia("anat-dji"), "anat-skydio": lambda: fig_anatomia("anat-skydio"),
           "anat-zipline": lambda: fig_anatomia("anat-zipline"), "anat-wing": lambda: fig_anatomia("anat-wing"),
           "anat-vbat": lambda: fig_anatomia("anat-vbat"),
           "parada": fig_parada, "energia": fig_energia, "deriva": fig_deriva, "tres": fig_tres,
           "js": fig_js, "carreras": fig_carreras, "sprind": fig_sprind, "tolleson": fig_tolleson}


def main():
    texto = ejemplos()
    print(texto)
    (AQUI / "salida_ejemplos.txt").write_text(texto + "\n")
    if "--insertar" in sys.argv:
        (AQUI / "svg").mkdir(exist_ok=True)
        html_path = AQUI.parent / "index.html"
        html = html_path.read_text()
        for nombre, fn in FIGURAS.items():
            cuerpo = fn()
            (AQUI / "svg" / f"{nombre}.svg").write_text(cuerpo + "\n")
            a, b = f"<!--SVG:{nombre}-->", f"<!--/SVG:{nombre}-->"
            if a in html and b in html:
                i, j = html.index(a) + len(a), html.index(b)
                html = html[:i] + "\n" + cuerpo + "\n" + html[j:]
            else:
                print(f"AVISO: no encuentro los marcadores de {nombre}")
        html_path.write_text(html)
        print("insertadas:", ", ".join(FIGURAS))


if __name__ == "__main__":
    main()
