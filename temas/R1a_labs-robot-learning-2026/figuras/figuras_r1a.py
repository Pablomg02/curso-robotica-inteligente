"""Figuras con datos de R1a: se calculan aquí y se escriben como SVG en figuras/svg/.

Se insertan en index.html entre <!--SVG:nombre--> y <!--/SVG:nombre--> con `python3 figuras_r1a.py --insertar`.
Datos: figuras/datos/cifras.json (cifras publicadas con su fuente y su estado). Solo usa la biblioteca estándar.
"""
import json
import math
import re
import sys
from pathlib import Path
from statistics import NormalDist

AQUI = Path(__file__).parent
SVG = AQUI / "svg"
SVG.mkdir(exist_ok=True)
D = json.loads((AQUI / "datos" / "cifras.json").read_text())


def f(x, d=1):
    return f"{x:.{d}f}"


class Ejes:
    """Rectángulo de dibujo con escalas lineales o logarítmicas (el mismo patrón que en T1a y T1b)."""

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
    # los nombres de modelo (GR00T N1.7, π0.7) conservan el punto: no se toca un número pegado a una letra
    return re.sub(r"(<text[^>]*>)(.*?)(</text>)", lambda m: m.group(1) + re.sub(r"(?<![\w.π])(\d+)\.(\d)", r"\1,\2", m.group(2)) + m.group(3), cuerpo)


def svg(nombre, vb, cuerpo, aria, maxh=None):
    cuerpo = coma(cuerpo).replace("<polyline ", '<polyline fill="none" ')
    st = f' style="max-height:{maxh}px"' if maxh else ""
    s = f'<svg viewBox="0 0 {vb[0]} {vb[1]}"{st} role="img" aria-label="{aria}">\n{cuerpo}\n</svg>'
    (SVG / f"{nombre}.svg").write_text(s)
    return s


def miles(v):
    """10000 -> «10 000» (separador de miles del curso)."""
    return f"{v:,.0f}".replace(",", " ")


def potencias(a, b):
    out, e = [], math.ceil(math.log10(a))
    while 10 ** e <= b:
        out.append(10 ** e)
        e += 1
    return out


def etiqueta_pot(v):
    return {1: "1", 10: "10", 100: "100", 1000: "1000"}.get(v, "10" + str(int(round(math.log10(v)))).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")))


def separar(ys, minimo):
    """Reparte etiquetas en vertical para que no se pisen (mantiene el orden)."""
    ys = list(ys)
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < minimo:
            ys[i] = ys[i - 1] + minimo
    return ys


# ---------------------------------------------------------------------------
CAPA = {"robot": ("accent", "robot real"), "sintetico": ("time", "simulación"), "humanos": ("tesis", "humanos con sensores"),
        "mixto": ("muted", "humanos y robots, sin desglose"), "video": ("muted", "vídeo general")}


def fig_piramide():
    """Horas de datos publicadas por capa de la pirámide (escala logarítmica)."""
    filas = sorted(D["horas_datos"], key=lambda r: r["horas"])
    E = Ejes(330, 16, 470, 26 * len(filas), (10, 2e6), (0, 1), xlog=True)
    s = [E.marco([10, 100, 1000, 1e4, 1e5, 1e6], [], "horas (escala logarítmica)", "",
                 xfmt=lambda v: miles(v) if v < 1e5 else etiqueta_pot(v))]
    for i, r in enumerate(filas):
        y = E.y0 + 26 * i + 5
        cl = CAPA[r["capa"]][0]
        fill = f"fill-{cl}" if cl != "muted" else "fill-ink3"
        x1 = E.X(r["horas"])
        s.append(f'<rect x="{f(E.x0)}" y="{f(y)}" width="{f(x1-E.x0)}" height="16" rx="3" class="{fill}" opacity="0.85"/>')
        s.append(f'<text x="{f(E.x0-8)}" y="{f(y+13)}" text-anchor="end" font-size="13">{r["nombre"]}</text>')
        et = ("&gt; " if r["nombre"].startswith(("π0", "GEN", "V-JEPA")) else "") + miles(r["horas"]) + (" (E)" if r["estado"] == "E" else "")
        s.append(f'<text x="{f(x1+6)}" y="{f(y+13)}" font-size="12" class="t-mono">{et}</text>')
    yl = E.y0 + E.h + 62
    x = 40
    for k in ("robot", "sintetico", "humanos", "mixto"):
        cl, nom = CAPA[k]
        fill = f"fill-{cl}" if cl != "muted" else "fill-ink3"
        s.append(f'<rect x="{x}" y="{yl-11}" width="14" height="14" rx="2" class="{fill}" opacity="0.85"/>')
        s.append(f'<text x="{x+20}" y="{yl}" font-size="13">{nom}{" / vídeo" if k == "mixto" else ""}</text>')
        x += {"robot": 120, "sintetico": 120, "humanos": 200, "mixto": 0}[k]
    return svg("piramide", (900, int(yl + 12)), "\n".join(s), "Horas de datos publicadas por capa de la pirámide, en escala logarítmica", 400)


def fig_coste():
    """Horas de operador para 10 000 demostraciones útiles con los ritmos medidos en UMI."""
    r = D["umi_demos_por_hora"]
    N = 10_000
    filas = [("teleoperación (space mouse)", r["teleoperacion"], "accent"), ("pinza UMI en la mano", r["umi"], "time"),
             ("mano humana", r["mano"], "tesis")]
    E = Ejes(210, 14, 330, 120, (0, 300), (0, 1))
    s = [E.marco([0, 100, 200, 300], [], "horas de operador para 10 000 demostraciones", "")]
    for i, (nom, rr, cl) in enumerate(filas):
        y = E.y0 + 12 + 36 * i
        h = N / rr
        s.append(f'<rect x="{f(E.x0)}" y="{f(y)}" width="{f(E.X(h)-E.x0)}" height="20" rx="3" class="fill-{cl}" opacity="0.85"/>')
        s.append(f'<text x="{f(E.x0-8)}" y="{f(y+15)}" text-anchor="end" font-size="13">{nom}</text>')
        s.append(f'<text x="{f(E.X(h)+6)}" y="{f(y+15)}" font-size="12" class="t-mono">{h:.0f} h ({rr}/h)</text>')
    return svg("coste", (600, 196), "\n".join(s), "Horas de operador para recoger 10 000 demostraciones con tres métodos", 200)


def fig_latencia():
    """Latencia mediana de GR00T N1.7 por dispositivo (README de despliegue de Isaac-GR00T).

    Thor con pesos NVFP4 va en su propia fila: es otra configuración, no una barra que se suma a la de TensorRT."""
    L = D["gr00t_n17_latencia_ms"]
    filas = [("H100", "H100", "trt"), ("RTX Pro 6000", "RTX Pro 6000", "trt"), ("L40", "L40", "trt"),
             ("AGX Thor", "AGX Thor", "trt"), ("AGX Thor", "AGX Thor + NVFP4", "trt_nvfp4"),
             ("DGX Spark", "DGX Spark", "trt"), ("Jetson Orin", "Jetson Orin", "trt")]
    paso = 30
    E = Ejes(150, 34, 540, len(filas) * paso, (0, 380), (0, 1))
    s = [E.marco([0, 50, 100, 150, 200, 250, 300, 350], [], "latencia mediana de una llamada completa (ms)", "")]
    s.append(f'<text x="{f(E.x0+E.w+44)}" y="{f(E.y0-12)}" text-anchor="end" font-size="12" class="t-muted">TensorRT</text>')
    # 50 y 30 Hz están a 13 ms: etiquetas ancladas a lados opuestos para que no se pisen
    for ms, et, ancla, dx in ((20, "50 Hz", "end", -3), (100 / 3, "30 Hz", "start", 3), (100, "10 Hz", "middle", 0)):
        x = E.X(ms)
        s.append(f'<line x1="{f(x)}" y1="{f(E.y0-6)}" x2="{f(x)}" y2="{f(E.y0+E.h)}" class="ln-time" stroke-dasharray="5 4"/>')
        s.append(f'<text x="{f(x+dx)}" y="{f(E.y0-12)}" text-anchor="{ancla}" font-size="12" class="t-time">{et}</text>')
    for i, (d, etiqueta, clave) in enumerate(filas):
        y = E.y0 + 5 + paso * i
        v = L[d]
        if clave == "trt":
            s.append(f'<rect x="{f(E.x0)}" y="{f(y)}" width="{f(E.X(v["eager"])-E.x0)}" height="8" rx="2" class="fill-ink3" opacity="0.45"/>')
            clase = "fill-accent"
        else:
            clase = "fill-tesis"
        s.append(f'<rect x="{f(E.x0)}" y="{f(y+10)}" width="{f(E.X(v[clave])-E.x0)}" height="11" rx="2" class="{clase}"/>')
        s.append(f'<text x="{f(E.x0-8)}" y="{f(y+17)}" text-anchor="end" font-size="13">{etiqueta}</text>')
        # valores en una columna a la derecha: junto a la barra los cruzarían las líneas de 50, 30 y 10 Hz
        s.append(f'<text x="{f(E.x0+E.w+44)}" y="{f(y+20)}" text-anchor="end" font-size="13" class="t-mono">{v[clave]:.1f}</text>')
    yl = E.y0 + E.h + 58
    s.append(f'<rect x="150" y="{yl-10}" width="16" height="9" rx="2" class="fill-ink3" opacity="0.45"/><text x="172" y="{yl}" font-size="13">PyTorch sin compilar</text>')
    s.append(f'<rect x="330" y="{yl-11}" width="16" height="11" rx="2" class="fill-accent"/><text x="352" y="{yl}" font-size="13">TensorRT</text>')
    s.append(f'<rect x="450" y="{yl-11}" width="16" height="11" rx="2" class="fill-tesis"/><text x="472" y="{yl}" font-size="13">TensorRT, pesos NVFP4 (solo Thor)</text>')
    return svg("latencia", (740, int(yl + 10)), "\n".join(s), "Latencia de GR00T N1.7 por dispositivo frente a los periodos de 50, 30 y 10 hercios", 290)


def fig_param_freq():
    """Parámetros frente a frecuencia: llenos, medidos con método publicado (C); huecos, declarados (E)."""
    M = [m for m in D["modelos_param_freq"] if m.get("en_figura", True)]
    E = Ejes(90, 20, 470, 270, (20, 2e4), (2, 400), xlog=True, ylog=True)
    s = [E.marco([100, 1000, 10000], [2, 5, 10, 20, 50, 100, 200],
                 "parámetros (millones, escala logarítmica)", "Hz (1/latencia o declarada)",
                 xfmt=lambda v: miles(v), yfmt=lambda v: str(v))]
    for hz, et in ((50, "50 Hz"),):
        y = E.Y(hz)
        s.append(f'<line x1="{f(E.x0)}" y1="{f(y)}" x2="{f(E.x0+E.w)}" y2="{f(y)}" class="ln-time" stroke-dasharray="5 4"/>')
    # rangos verticales (π0.7, GR00T N1.7, GR00T 2 Turbo, Helix S2 7–9 Hz)
    rangos = [(5000, 7.87, 26.3, "accent"), (3000, 6.71, 43.67, "accent"), (2000, 8.07, 26.04, "accent"), (7000, 7, 9, "tesis")]
    for x, a, b, cl in rangos:
        s.append(f'<line x1="{f(E.X(x))}" y1="{f(E.Y(a))}" x2="{f(E.X(x))}" y2="{f(E.Y(b))}" class="ln-{cl}" stroke-width="2"/>')
    pts = []
    for m in M:
        x, y = E.X(m["param_M"]), E.Y(m["hz"])
        if m["estado"] == "C":
            s.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="5" class="fill-accent"/>')
        else:
            s.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="5" fill="none" class="ln-tesis" stroke-width="2"/>')
        pts.append((m, x, y))
    # etiquetas: las del grupo denso (1000–10000 M) en una columna a la derecha con líneas guía
    derecha = sorted([p for p in pts if p[0]["param_M"] >= 1000], key=lambda p: p[2])
    ys = separar([p[2] for p in derecha], 19)
    off = (ys[0] + ys[-1]) / 2 - (derecha[0][2] + derecha[-1][2]) / 2
    ys = [y - max(0, off) for y in ys]
    for (m, x, y), yl in zip(derecha, ys):
        s.append(f'<line x1="{f(x+6)}" y1="{f(y)}" x2="{f(E.x0+E.w+8)}" y2="{f(yl-4)}" class="ln-rule"/>')
        txt = m.get("etiqueta", m["nombre"] + f" · {m['hz']:.1f} Hz")
        s.append(f'<text x="{f(E.x0+E.w+12)}" y="{f(yl)}" font-size="14" class="{"t-accent" if m["estado"] == "C" else "t-tesis"}">{txt}</text>')
    for m, x, y in pts:
        if m["param_M"] < 1000:
            ancla, dx = ("start", 9) if m["nombre"] != "LBM de Atlas" else ("start", 9)
            s.append(f'<text x="{f(x+dx)}" y="{f(y+4)}" text-anchor="{ancla}" font-size="14" '
                     f'class="{"t-accent" if m["estado"] == "C" else "t-tesis"}">{m.get("etiqueta", m["nombre"] + f" · {m['hz']:g} Hz")}</text>')
    s.append(f'<text x="{f(E.x0+6)}" y="{f(E.Y(50)-6)}" font-size="13" class="t-time">50 Hz: control habitual de un brazo</text>')
    yl = E.y0 + E.h + 62
    s.append(f'<circle cx="100" cy="{yl-4}" r="5" class="fill-accent"/><text x="112" y="{yl}" font-size="13">medida con método publicado (C)</text>')
    s.append(f'<circle cx="370" cy="{yl-4}" r="5" fill="none" class="ln-tesis" stroke-width="2"/><text x="382" y="{yl}" font-size="13">frecuencia que declara la empresa (E)</text>')
    return svg("param_freq", (840, int(yl + 10)), "\n".join(s), "Parámetros frente a frecuencia de varios modelos de robot", 330)


def fig_wilson():
    """Intervalos de Wilson al 95 % de resultados publicados o hipotéticos."""
    def wilson(k, n, z=1.959964):
        p = k / n
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, c - h), min(1.0, c + h)
    filas = [(8, 10, "π0: 8 de 10", "accent"), (24, 30, "humanos en π0.7: 24 de 30", "accent"),
             (20, 20, "20 de 20", "time"), (86, 86, "GEN-1: 86 seguidas", "time"),
             (24, 25, "«96 %» con n = 25", "tesis"), (48, 50, "«96 %» con n = 50", "tesis"),
             (96, 100, "«96 %» con n = 100", "tesis"), (480, 500, "«96 %» con n = 500", "tesis")]
    E = Ejes(230, 10, 380, 26 * len(filas), (0.4, 1.0), (0, 1))
    s = [E.marco([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [], "tasa de éxito (intervalo de Wilson al 95 %)", "", xfmt=lambda v: f"{v:.1f}")]
    for i, (k, n, nom, cl) in enumerate(filas):
        y = E.y0 + 13 + 26 * i
        lo, hi = wilson(k, n)
        s.append(f'<line x1="{f(E.X(lo))}" y1="{f(y)}" x2="{f(E.X(hi))}" y2="{f(y)}" class="ln-{cl}" stroke-width="3"/>')
        s.append(f'<circle cx="{f(E.X(k/n))}" cy="{f(y)}" r="4" class="fill-{cl}"/>')
        s.append(f'<text x="{f(E.x0-8)}" y="{f(y+4)}" text-anchor="end" font-size="13">{nom}</text>')
        s.append(f'<text x="{f(min(E.X(hi), E.X(0.97))+8 if hi < 0.97 else E.X(lo)-8)}" y="{f(y+4)}" font-size="11" '
                 f'class="t-mono" text-anchor="{"start" if hi < 0.97 else "end"}">[{lo:.2f}; {hi:.2f}]</text>')
    return svg("wilson", (640, int(E.y0 + E.h + 46)), "\n".join(s), "Intervalos de Wilson al 95 por ciento de varios resultados", 270)


def fig_potencia():
    """Ensayos por política para separar p1 = 0,6 de p2 (α = 0,05 bilateral, potencia 0,8)."""
    nd = NormalDist()
    za, zb = nd.inv_cdf(0.975), nd.inv_cdf(0.8)

    def n(p1, p2):
        pb = (p1 + p2) / 2
        return (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2
    E = Ejes(80, 14, 430, 200, (0.6, 0.95), (10, 3000), ylog=True)
    s = [E.marco([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], [10, 30, 100, 300, 1000, 3000],
                 "tasa de la segunda política, p₂ (la primera, p₁ = 0,6)", "ensayos por política", xfmt=lambda v: f"{v:.2f}")]
    # la curva empieza donde n baja de 3000 (el tope del eje), sin tramo plano recortado
    a0, b0 = 0.601, 0.7
    for _ in range(80):
        m = (a0 + b0) / 2
        a0, b0 = (m, b0) if n(0.6, m) > 3000 else (a0, m)
    xs = [b0] + [0.6 + 0.0025 * i for i in range(1, 141) if 0.6 + 0.0025 * i > b0]
    s.append(E.linea(xs, [n(0.6, x) for x in xs], "ln-accent", 'stroke-width="2.5"'))
    n7 = n(0.6, 0.7)
    s.append(f'<circle cx="{f(E.X(0.7))}" cy="{f(E.Y(n7))}" r="5" class="fill-accent"/>')
    s.append(f'<text x="{f(E.X(0.7)+9)}" y="{f(E.Y(n7)-6)}" font-size="13" class="t-accent">0,6 frente a 0,7: {math.ceil(n7)}</text>')
    y50 = E.Y(50)
    s.append(f'<line x1="{f(E.x0)}" y1="{f(y50)}" x2="{f(E.x0+E.w)}" y2="{f(y50)}" class="ln-time" stroke-dasharray="5 4"/>')
    # p2 que se detecta con n = 50
    a, b = 0.61, 0.99
    for _ in range(80):
        m = (a + b) / 2
        a, b = (m, b) if n(0.6, m) > 50 else (a, m)
    s.append(f'<circle cx="{f(E.X(a))}" cy="{f(y50)}" r="5" class="fill-time"/>')
    # bajo la línea y a la izquierda del punto no pasa la curva: ahí va la etiqueta
    s.append(f'<text x="{f(E.X(a)-10)}" y="{f(y50+20)}" text-anchor="end" font-size="13" class="t-time">'
             f'<tspan font-style="italic">n</tspan> = 50 (TRI): solo se ve si <tspan font-style="italic">p</tspan>₂ ≥ {a:.2f}</text>')
    return svg("potencia", (560, 270), "\n".join(s), "Ensayos necesarios por política frente a la diferencia de tasas", 290)


def fig_waymo():
    """Tasas por millón de millas con sus IC al 95 % (hub de Waymo, datos a junio de 2026).

    Cada grupo lleva su título encima y las cifras en una columna a la derecha del eje, para que se lean a tamaño de hoja."""
    w = D["waymo_hub"]
    filas = [("lesión grave o peor", w["grave_o_peor"]), ("airbag (cualquier vehículo)", w["airbag_cualquier_vehiculo"]),
             ("cualquier lesión", w["cualquier_lesion"])]
    paso = 64
    E = Ejes(20, 4, 330, 3 * paso, (0.001, 10), (0, 1), xlog=True)
    s = [E.marco([0.001, 0.01, 0.1, 1, 10], [], "incidentes por millón de millas (escala logarítmica)", "",
                 xfmt=lambda v: f"{v:g}")]
    xc = E.x0 + E.w + 22
    s.append(f'<text x="{f(xc)}" y="{f(E.y0+E.h+18)}" font-size="12" class="t-muted">tasa [IC al 95 %]</text>')
    for i, (nom, d) in enumerate(filas):
        y0 = E.y0 + paso * i
        s.append(f'<text x="{f(E.x0+6)}" y="{f(y0+16)}" font-size="14" class="t-bold">{nom}</text>')
        for j, (clave, cl, et) in enumerate((("waymo", "accent", "Waymo  "), ("humano", "time", "humanos"))):
            v, lo, hi = d[clave]
            y = y0 + 30 + 17 * j
            s.append(f'<line x1="{f(E.X(lo))}" y1="{f(y)}" x2="{f(E.X(hi))}" y2="{f(y)}" class="ln-{cl}" stroke-width="3"/>')
            s.append(f'<circle cx="{f(E.X(v))}" cy="{f(y)}" r="4.5" class="fill-{cl}"/>')
            s.append(f'<text x="{f(xc)}" y="{f(y+4.5)}" font-size="13" class="t-mono t-{cl}" xml:space="preserve">{et} {v:.3g} [{lo:.3g}; {hi:.3g}]</text>')
    return svg("waymo", (650, int(E.y0 + E.h + 46)), "\n".join(s), "Tasas de incidentes por millón de millas de Waymo y de la referencia humana con intervalos al 95 por ciento", 270)


FIGS = [fig_piramide, fig_coste, fig_latencia, fig_param_freq, fig_wilson, fig_potencia, fig_waymo]

if __name__ == "__main__":
    salida = {}
    for fn in FIGS:
        salida[fn.__name__[4:]] = fn()
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
