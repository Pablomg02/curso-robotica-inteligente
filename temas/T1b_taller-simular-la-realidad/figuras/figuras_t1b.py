"""Figuras con datos de T1b: se calculan aquí y se escriben como SVG en figuras/svg/.

Se insertan en index.html entre <!--SVG:nombre--> y <!--/SVG:nombre--> con `python3 figuras_t1b.py --insertar`.
Datos: figuras/datos/ (comparar_real_sitl.py, extraer_neural.py y los scripts de figuras/experimentos/).
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).parent
DATOS, SVG = AQUI / "datos", AQUI / "svg"
SVG.mkdir(exist_ok=True)


def f(x, d=1):
    return f"{x:.{d}f}"


class Ejes:
    """Rectángulo de dibujo con escalas lineales o logarítmicas (igual que en T1a)."""

    def __init__(self, x0, y0, w, h, xlim, ylim, xlog=False, ylog=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _n(self, v, lim, log):
        a, b = lim
        if log:
            v, a, b = math.log10(v), math.log10(a), math.log10(b)
        return (v - a) / (b - a)

    def X(self, v):
        return self.x0 + self._n(v, self.xlim, self.xlog) * self.w

    def Y(self, v):
        return self.y0 + self.h - self._n(v, self.ylim, self.ylog) * self.h

    def marco(self, xt, yt, xlab, ylab, xfmt=str, yfmt=str):
        s = [f'<line x1="{f(self.x0)}" y1="{f(self.y0+self.h)}" x2="{f(self.x0+self.w)}" y2="{f(self.y0+self.h)}" class="ln"/>',
             f'<line x1="{f(self.x0)}" y1="{f(self.y0)}" x2="{f(self.x0)}" y2="{f(self.y0+self.h)}" class="ln"/>']
        for v in xt:
            x = self.X(v)
            s.append(f'<line x1="{f(x)}" y1="{f(self.y0)}" x2="{f(x)}" y2="{f(self.y0+self.h)}" class="ln-rule"/>')
            s.append(f'<text x="{f(x)}" y="{f(self.y0+self.h+18)}" text-anchor="middle" font-size="13" class="t-muted">{xfmt(v)}</text>')
        for v in yt:
            y = self.Y(v)
            s.append(f'<line x1="{f(self.x0)}" y1="{f(y)}" x2="{f(self.x0+self.w)}" y2="{f(y)}" class="ln-rule"/>')
            s.append(f'<text x="{f(self.x0-8)}" y="{f(y+4)}" text-anchor="end" font-size="13" class="t-muted">{yfmt(v)}</text>')
        if xlab:
            s.append(f'<text x="{f(self.x0+self.w/2)}" y="{f(self.y0+self.h+38)}" text-anchor="middle" font-size="14" class="t-muted">{xlab}</text>')
        if ylab:
            s.append(f'<text x="{f(self.x0-52)}" y="{f(self.y0+self.h/2)}" text-anchor="middle" font-size="14" class="t-muted" '
                     f'transform="rotate(-90 {f(self.x0-52)} {f(self.y0+self.h/2)})">{ylab}</text>')
        return "\n".join(s)

    def linea(self, xs, ys, clase="ln-accent", extra=""):
        pts = " ".join(f"{f(self.X(a))},{f(self.Y(b))}" for a, b in zip(xs, ys))
        return f'<polyline points="{pts}" class="{clase}" {extra}/>'


def coma(cuerpo):
    """Coma decimal en los textos (convenio del curso); no toca coordenadas ni atributos."""
    return re.sub(r"(<text[^>]*>)(.*?)(</text>)", lambda m: m.group(1) + re.sub(r"(\d)\.(\d)", r"\1,\2", m.group(2)) + m.group(3), cuerpo)


def svg(nombre, vb, cuerpo, aria, maxh=None):
    cuerpo = coma(cuerpo).replace("<polyline ", '<polyline fill="none" ')
    st = f' style="max-height:{maxh}px"' if maxh else ""
    s = f'<svg viewBox="0 0 {vb[0]} {vb[1]}"{st} role="img" aria-label="{aria}">\n{cuerpo}\n</svg>'
    (SVG / f"{nombre}.svg").write_text(s)
    return s


def pot10(v):
    e = int(round(math.log10(v)))
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"10{sup}"


RS = json.loads((DATOS / "real_sitl.json").read_text())


# ---------------------------------------------------------------------------
def fig_psd():
    """PSD del giróscopo (eje x): X500 V2 real (FIFO a 8,4 kHz) frente al x500 de SITL (250 Hz)."""
    filas = list(csv.DictReader(open(DATOS / "psd_giro.csv")))
    E = Ejes(100, 24, 760, 270, (1, 4200), (1e-10, 1e-2), xlog=True, ylog=True)
    s = [E.marco([1, 10, 100, 1000], [1e-10, 1e-8, 1e-6, 1e-4, 1e-2], "frecuencia (Hz)", "PSD ((rad/s)²/Hz)", yfmt=pot10)]
    fr = RS["real_fifo"]["f_rotor_hz"]
    for v, txt, dy in [(fr, "giro", 0), (2 * fr, "paso de pala", 0), (RS["real_fifo"]["pico_hz"], "pico 332 Hz", 0), (125, "Nyquist del SITL", 1)]:
        x = E.X(v)
        s.append(f'<line x1="{f(x)}" y1="{f(E.y0)}" x2="{f(x)}" y2="{f(E.y0+E.h)}" class="ln-dash"/>')
    # etiquetas de las marcas, escalonadas para que no se pisen
    for v, txt, y in [(fr, "giro 80 Hz", 40), (2 * fr, "pala 160 Hz", 58), (RS["real_fifo"]["pico_hz"], "pico 332 Hz", 40), (125, "Nyquist SITL 125 Hz", 290)]:
        s.append(f'<text x="{f(E.X(v)+4)}" y="{y}" font-size="12" class="t-muted">{txt}</text>')
    for fuente, clase, etiqueta in [("real", "ln-time", "X500 V2 real (giróscopo en bruto, 8,4 kHz)"), ("sitl", "ln-accent", "x500 de SITL (250 Hz)")]:
        xs = [float(r["f_hz"]) for r in filas if r["fuente"] == fuente]
        ys = [max(float(r["psd_rad2_s2_hz"]), 1e-10) for r in filas if r["fuente"] == fuente]
        s.append(E.linea(xs, ys, clase, 'stroke-width="2"'))
    S = RS["sitl"]["psd_teorica_sdf"]
    s.append(f'<line x1="{f(E.X(1))}" y1="{f(E.Y(S))}" x2="{f(E.X(125))}" y2="{f(E.Y(S))}" class="ln-dash" stroke-width="2"/>')
    s.append(f'<text x="{f(E.X(1.2))}" y="{f(E.Y(S)+18)}" font-size="12" class="t-accent">sensor del SDF: 2σ²/fs = {S:.2e}</text>'.replace("e-09", "·10⁻⁹"))
    # lo que registra PX4: media de dos muestras (trapecio) -> S0·cos²(pi f / fs), que cae a 0 en Nyquist
    fx = np.logspace(0, math.log10(124.0), 80)
    s.append(E.linea(fx, np.maximum(S * np.cos(np.pi * fx / 250.0) ** 2, 1e-10), "ln-tesis", 'stroke-width="2" stroke-dasharray="5 4"'))
    s.append(f'<text x="{f(E.X(1.2))}" y="{f(E.Y(S)+36)}" font-size="12" class="t-tesis">tras el trapecio de PX4: × cos²(πf/fs)</text>')
    s.append(f'<text x="{f(E.X(4)) }" y="{f(E.Y(3e-3))}" font-size="13" class="t-time">X500 V2 real (8,4 kHz, en bruto)</text>')
    s.append(f'<text x="{f(E.X(4))}" y="{f(E.Y(2e-8)-6)}" font-size="13" class="t-accent">x500 de SITL (250 Hz)</text>')
    return svg("psd", (900, 340), "\n".join(s), "Densidad espectral del giróscopo, dron real frente a SITL", 262)


def fig_brecha():
    """Dispersión en estacionario, real frente a SITL, en escala logarítmica (una fila por magnitud)."""
    r, s_ = RS["real"], RS["sitl"]
    filas = [("giróscopo x (°/s)", r["giro_std_deg_s"], s_["giro_std_deg_s"]),
             ("acelerómetro x (m/s²)", r["acc_std_m_s2"][0], s_["acc_std_m_s2"][0]),
             ("acelerómetro y (m/s²)", r["acc_std_m_s2"][1], s_["acc_std_m_s2"][1]),
             ("acelerómetro z (m/s²)", r["acc_std_m_s2"][2], s_["acc_std_m_s2"][2]),
             ("velocidad angular y (°/s)", r["rate_std_deg_s"][1], s_["rate_std_deg_s"][1]),
             ("posición z (cm)", r["pos_std_cm"][2], s_["pos_std_cm"][2]),
             ("posición x (cm)", r["pos_std_cm"][0], s_["pos_std_cm"][0]),
             ("error de cabeceo RMS (°)", r["err_pitch_rms_deg"], s_["err_pitch_rms_deg"])]
    E = Ejes(250, 20, 560, 34 * len(filas), (0.003, 30), (0, len(filas)), xlog=True)
    s = [E.marco([0.01, 0.1, 1, 10], [], "desviación típica en estacionario (escala logarítmica)", "", xfmt=lambda v: f"{v:g}")]
    for i, (nom, a, b) in enumerate(filas):
        y = E.y0 + 17 + 34 * i
        s.append(f'<text x="{f(E.x0-12)}" y="{f(y+4)}" text-anchor="end" font-size="13">{nom}</text>')
        s.append(f'<line x1="{f(E.X(min(a,b)))}" y1="{f(y)}" x2="{f(E.X(max(a,b)))}" y2="{f(y)}" class="ln" stroke-width="2"/>')
        s.append(f'<circle cx="{f(E.X(b))}" cy="{f(y)}" r="6" class="fill-accent"/>')
        s.append(f'<circle cx="{f(E.X(a))}" cy="{f(y)}" r="6" class="fill-time"/>')
        k = a / b
        s.append(f'<text x="{f(E.X(max(a,b))+12)}" y="{f(y+4)}" font-size="13" class="t-mono">{"×" + f(k, 0) if k >= 9.5 else "×" + f(k, 1)}</text>')
    yl = E.y0 + E.h + 58
    s.append(f'<circle cx="260" cy="{yl}" r="6" class="fill-time"/><text x="272" y="{yl+4}" font-size="13">X500 V2 real (registro público)</text>')
    s.append(f'<circle cx="520" cy="{yl}" r="6" class="fill-accent"/><text x="532" y="{yl+4}" font-size="13">x500 de SITL</text>')
    return svg("brecha", (900, E.y0 + E.h + 72), "\n".join(s), "Dispersión en estacionario del dron real frente al de SITL", 290)


def fig_empuje():
    """Empuje por rotor frente al mando en el x500 de Gazebo: T = kT (150 + 850 u)²."""
    kT, wmin, wmax = 8.54858e-6, 150.0, 1000.0
    u = np.linspace(0, 1, 101)
    T = kT * (wmin + (wmax - wmin) * u) ** 2
    E = Ejes(80, 20, 420, 230, (0, 1), (0, 9))
    s = [E.marco([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 2, 4, 6, 8], "mando normalizado u", "empuje por rotor (N)", xfmt=lambda v: f"{v:g}")]
    s.append(E.linea(u, T, "ln-accent", 'stroke-width="2.5"'))
    Th = 2.0643 * 9.8 / 4
    s.append(f'<line x1="{f(E.X(0))}" y1="{f(E.Y(Th))}" x2="{f(E.X(1))}" y2="{f(E.Y(Th))}" class="ln-dash"/>')
    s.append(f'<text x="{f(E.X(0.02))}" y="{f(E.Y(Th)-6)}" font-size="13" class="t-muted">peso / 4 = 5,06 N</text>')
    for uu, txt, cl, dy in [(0.60, "MPC_THR_HOVER = 0,60", "t-time", 20), (0.728, "medido y calculado: 0,728", "t-accent", -10)]:
        TT = kT * (wmin + (wmax - wmin) * uu) ** 2
        s.append(f'<line x1="{f(E.X(uu))}" y1="{f(E.Y(0))}" x2="{f(E.X(uu))}" y2="{f(E.Y(TT))}" class="ln-dash"/>')
        s.append(f'<circle cx="{f(E.X(uu))}" cy="{f(E.Y(TT))}" r="5" class="{"fill-time" if cl=="t-time" else "fill-accent"}"/>')
        s.append(f'<text x="{f(E.X(uu)-8)}" y="{f(E.Y(TT)+dy)}" text-anchor="end" font-size="13" class="{cl}">{txt}</text>')
    return svg("empuje", (540, 300), "\n".join(s), "Empuje por rotor frente al mando en el x500 de Gazebo", 280)


def fig_coste():
    """Tope del factor de tiempo real según lo que se simula (un dron, sin pantalla)."""
    cam = json.loads((DATOS / "camaras.json").read_text())
    imu = json.loads((DATOS / "imu_1khz.json").read_text())
    tope = {c["modelo"]: c["rtf"] for c in cam if c["pedido"] != 1}
    filas = [("x500 base (250 Hz)", 34.76),
             ("IMU a 1 kHz", imu[1]["rtf"]),
             ("cámara 1280×960", tope.get("gz_x500_mono_cam")),
             ("OAK-D Lite", tope.get("gz_x500_depth"))]
    E = Ejes(150, 16, 330, 40 * len(filas), (0, 40), (0, len(filas)))
    s = [E.marco([0, 10, 20, 30, 40], [], "factor de tiempo real máximo", "", xfmt=lambda v: f"×{v:g}")]
    for i, (nom, v) in enumerate(filas):
        y = E.y0 + 8 + 40 * i
        s.append(f'<rect x="{f(E.x0)}" y="{f(y)}" width="{f(E.X(v)-E.x0)}" height="24" class="{"box-accent" if i == 0 else "box-time"}"/>')
        s.append(f'<text x="{f(E.x0-10)}" y="{f(y+17)}" text-anchor="end" font-size="13">{nom}</text>')
        s.append(f'<text x="{f(E.X(v)+8)}" y="{f(y+17)}" font-size="13" class="t-mono">×{v:.2f}</text>')
    print("coste:", filas)
    return svg("coste", (560, E.y0 + E.h + 46), "\n".join(s), "Factor de tiempo real máximo según los sensores simulados", 220)


def fig_gsplat():
    """Tiempo de render de gsplat frente al número de gaussianas, para tres resoluciones."""
    cam = json.loads((DATOS / "camara_neuronal.json").read_text())
    E = Ejes(80, 20, 440, 230, (0, 6), (0, 40))
    s = [E.marco([0, 1, 2, 3, 4, 5, 6], [0, 10, 20, 30, 40], "millones de gaussianas", "ms por imagen (mediana)")]
    s.append(f'<line x1="{f(E.X(0))}" y1="{f(E.Y(33.33))}" x2="{f(E.X(6))}" y2="{f(E.Y(33.33))}" class="ln-dash"/>')
    s.append(f'<text x="{f(E.X(0.1))}" y="{f(E.Y(33.33)-6)}" font-size="13" class="t-muted">33,3 ms: presupuesto a 30 Hz</text>')
    clases = {320: "ln-tesis", 640: "ln-accent", 1280: "ln-time"}
    for W, cl in clases.items():
        pts = [(c["n"] / 1e6, c["ms_mediana"]) for c in cam["coste"] if c["W"] == W]
        s.append(E.linea([0] + [p[0] for p in pts], [0] + [p[1] for p in pts], cl, 'stroke-width="2"'))
        for x, y in pts:
            s.append(f'<circle cx="{f(E.X(x))}" cy="{f(E.Y(y))}" r="3.5" class="{cl.replace("ln", "fill")}"/>')
    for i, (W, cl) in enumerate(clases.items()):
        y = E.Y(11) + 20 * i
        s.append(f'<line x1="{f(E.X(3.9))}" y1="{f(y)}" x2="{f(E.X(4.3))}" y2="{f(y)}" class="{cl}" stroke-width="2"/>')
        s.append(f'<text x="{f(E.X(4.4))}" y="{f(y+4)}" font-size="13">{W}×{W*3//4}</text>')
    s.append(f'<text x="{f(E.X(5.83))}" y="{f(E.Y(27))}" text-anchor="end" font-size="13" class="t-muted">escena «garden»: 5,83 M</text>')
    return svg("gsplat", (560, 300), "\n".join(s), "Coste del render de splats gaussianos frente al número de gaussianas", 270)


def fig_lazo_camara():
    """Esquema temporal del bucle de la cámara neuronal: poses a 50 Hz, imágenes a 30 Hz, render de 21 ms."""
    cam = json.loads((DATOS / "camara_neuronal.json").read_text())["vivo"]
    r = cam["render_ms_mediana"]
    X = lambda t: 60 + t * 7.0
    s = []
    for fila, y, txt in [(0, 50, "poses (MAVSDK, 50 Hz)"), (1, 120, "render (GPU)")]:
        s.append(f'<text x="{f(X(0))}" y="{y-22}" font-size="13" class="t-muted">{txt}</text>')
        s.append(f'<line x1="{f(X(0))}" y1="{y}" x2="{f(X(110))}" y2="{y}" class="ln-rule"/>')
    for k in range(6):
        t = 20 * k
        s.append(f'<line x1="{f(X(t))}" y1="38" x2="{f(X(t))}" y2="62" class="ln-accent" stroke-width="2"/>')
    for k in range(4):
        t0 = 100 / 3 * k
        if t0 + r > 110:
            continue
        s.append(f'<rect x="{f(X(t0))}" y="108" width="{f(X(t0+r)-X(t0))}" height="24" class="box-time"/>')
        tp = 20 * math.floor(t0 / 20)
        s.append(f'<polyline points="{f(X(tp))},62 {f(X(tp))},80 {f(X(t0+r))},80 {f(X(t0+r))},108" class="ln-dash"/>')
        s.append(f'<text x="{f((X(tp)+X(t0+r))/2)}" y="76" text-anchor="middle" font-size="12" class="t-mono">{t0+r-tp:.0f} ms</text>')
    for t in range(0, 111, 20):
        s.append(f'<text x="{f(X(t))}" y="160" text-anchor="middle" font-size="12" class="t-muted">{t} ms</text>')
    s.append(f'<text x="{f(X(55))}" y="186" text-anchor="middle" font-size="13">edad de la pose al terminar la imagen = espera desde la última pose + render ({r:.0f} ms)</text>')
    return svg("lazo_camara", (860, 200), "\n".join(s), "Esquema temporal del bucle de la cámara neuronal", 165)


def fig_neural():
    """La red de mc_nn_control volando el x500 de Gazebo: posición real frente a la consigna en el modo neuronal."""
    filas = list(csv.DictReader(open(DATOS / "neural.csv")))
    t = np.array([float(r["t_s"]) for r in filas])
    s = []
    paneles = [("x", "norte x (m)", (-0.5, 2.5), [0, 1, 2], 1), ("y", "este y (m)", (-0.5, 2.5), [0, 1, 2], 1), ("z", "altura −z (m)", (4.5, 7.5), [5, 6, 7], -1)]
    for i, (c, lab, lim, ticks, sg) in enumerate(paneles):
        E = Ejes(90, 14 + 100 * i, 720, 72, (0, 38), lim)
        s.append(E.marco([0, 10, 20, 30] if i == 2 else [], ticks, "tiempo en el modo neuronal (s)" if i == 2 else "", lab))
        v = np.array([float(r[c]) for r in filas]) * sg
        sp = np.array([float(r["sp_" + c]) for r in filas]) * sg
        s.append(E.linea(t, sp, "ln-dash", 'stroke-width="1.6"'))
        s.append(E.linea(t, v, "ln-accent", 'stroke-width="2"'))
    s.append('<text x="820" y="30" text-anchor="end" font-size="13" class="t-accent">posición real (Gazebo)</text>')
    s.append('<text x="820" y="48" text-anchor="end" font-size="13" class="t-muted">consigna (ROS 2)</text>')
    return svg("neural", (840, 340), "\n".join(s), "Seguimiento de consignas del controlador neuronal de PX4 en SITL", 272)


def fig_wilson():
    """Intervalos de Wilson al 95 % para 90 % y 80 % de éxito frente al número de ensayos."""
    def wilson(k, n, z=1.96):
        p = k / n
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return c - h, c + h
    ns = [10, 20, 50, 100, 200, 500]
    E = Ejes(80, 20, 460, 220, (7, 700), (0.4, 1.0), xlog=True)
    s = [E.marco(ns, [0.4, 0.6, 0.8, 1.0], "número de vuelos n", "tasa de éxito", yfmt=lambda v: f"{v:.1f}")]
    for p, cl, dx in [(0.9, "accent", -5), (0.8, "time", 5)]:
        for n in ns:
            lo, hi = wilson(round(p * n), n)
            x = E.X(n) + dx
            s.append(f'<line x1="{f(x)}" y1="{f(E.Y(lo))}" x2="{f(x)}" y2="{f(E.Y(hi))}" class="ln-{cl}" stroke-width="3"/>')
            s.append(f'<circle cx="{f(x)}" cy="{f(E.Y(p))}" r="3.5" class="fill-{cl}"/>')
    s.append(f'<text x="{f(E.X(250))}" y="{f(E.Y(0.55))}" font-size="13" class="t-accent">90 % de éxito</text>')
    s.append(f'<text x="{f(E.X(250))}" y="{f(E.Y(0.55)+18)}" font-size="13" class="t-time">80 % de éxito</text>')
    return svg("wilson", (580, 290), "\n".join(s), "Intervalos de Wilson al 95 por ciento frente al número de vuelos", 270)


FIGS = [fig_psd, fig_brecha, fig_empuje, fig_coste, fig_gsplat, fig_lazo_camara, fig_neural, fig_wilson]

if __name__ == "__main__":
    salida = {}
    for fn in FIGS:
        r = fn()
        nombre = fn.__name__[4:]
        salida[nombre] = r if isinstance(r, str) else r[0]
    if "--insertar" in sys.argv:
        p = AQUI.parent / "index.html"
        h = p.read_text()
        for nombre, cuerpo in salida.items():
            a, b = f"<!--SVG:{nombre}-->", f"<!--/SVG:{nombre}-->"
            if a in h:
                i, j = h.index(a) + len(a), h.index(b)
                h = h[:i] + "\n" + cuerpo + "\n" + h[j:]
            else:
                print("sin marcador:", nombre)
        p.write_text(h)
    print("figuras:", list(salida))
