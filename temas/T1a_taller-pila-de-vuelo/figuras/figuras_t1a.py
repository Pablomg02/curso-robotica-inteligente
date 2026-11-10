"""Figuras con datos de T1a: se calculan aquí y se escriben como SVG en figuras/svg/.

Las figuras se insertan en index.html entre los marcadores <!--SVG:nombre--> y <!--/SVG:nombre-->
con `python3 figuras_t1a.py --insertar` (sin el argumento solo escribe los SVG y las cifras).
Datos de entrada: figuras/datos/ (ver extraer_t1a.py y figuras/experimentos/).
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
    """Rectángulo de dibujo con escalas lineales o logarítmicas."""

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
        s.append(f'<text x="{f(self.x0+self.w/2)}" y="{f(self.y0+self.h+38)}" text-anchor="middle" font-size="14" class="t-muted">{xlab}</text>')
        s.append(f'<text x="{f(self.x0-48)}" y="{f(self.y0+self.h/2)}" text-anchor="middle" font-size="14" class="t-muted" '
                 f'transform="rotate(-90 {f(self.x0-48)} {f(self.y0+self.h/2)})">{ylab}</text>')
        return "\n".join(s)

    def linea(self, xs, ys, clase="ln-accent", extra=""):
        pts = " ".join(f"{f(self.X(a))},{f(self.Y(b))}" for a, b in zip(xs, ys))
        return f'<polyline points="{pts}" class="{clase}" {extra}/>'


def coma(cuerpo):
    """Coma decimal en los textos de las figuras (convenio del curso); no toca coordenadas ni atributos."""
    return re.sub(r"(<text[^>]*>)(.*?)(</text>)", lambda m: m.group(1) + re.sub(r"(\d)\.(\d)", r"\1,\2", m.group(2)) + m.group(3), cuerpo)


def svg(nombre, vb, cuerpo, aria, maxh=None):
    cuerpo = coma(cuerpo).replace("<polyline ", '<polyline fill="none" ')
    st = f' style="max-height:{maxh}px"' if maxh else ""
    s = f'<svg viewBox="0 0 {vb[0]} {vb[1]}"{st} role="img" aria-label="{aria}">\n{cuerpo}\n</svg>'
    (SVG / f"{nombre}.svg").write_text(s)
    return s


def leer_csv(p):
    with open(p) as fh:
        r = csv.DictReader(fh)
        return list(r)


# ---------------------------------------------------------------------------
def fig_arranque():
    """Línea temporal del arranque de PX4 SITL + Gazebo sin interfaz (registro con marcas de tiempo)."""
    log = (DATOS / "arranque.log").read_text().splitlines()
    eventos = [("px4 starting", "PX4 arranca y lee rcS", 0),
               ("Starting gazebo with world", "lanza gz sim -s (servidor)", 1),
               ("Gazebo world is ready", "mundo listo (scene/info)", 0),
               ("lockstep_scheduler", "primer reloj de Gazebo", 1),
               ("uxrce_dds_client] init", "cliente uXRCE-DDS (8888)", 0),
               ("mavlink] mode: Normal", "MAVLink a la GCS (14550)", 1),
               ("Startup script returned", "rcS termina", 0),
               ("Ready for takeoff", "listo para despegar (con GCS)", 1)]
    t = {}
    for clave, _, _ in eventos:
        for l in log:
            if clave in l:
                t[clave] = float(l.split()[0]); break
    tmax = 16.5
    X = lambda v: 40 + v / tmax * 1100
    s = [f'<line x1="{f(X(0))}" y1="150" x2="{f(X(tmax))}" y2="150" class="ln-time"/>']
    for k in range(0, 17, 2):
        s.append(f'<line x1="{f(X(k))}" y1="145" x2="{f(X(k))}" y2="155" class="ln-time"/>')
        s.append(f'<text x="{f(X(k))}" y="174" text-anchor="middle" font-size="13" class="t-time">{k} s</text>')
    a, b = t["Starting gazebo with world"], t["Gazebo world is ready"]
    s.append(f'<rect x="{f(X(a))}" y="141" width="{f(X(b)-X(a))}" height="18" class="box-time" opacity="0.6"/>')
    # etiquetas: arriba y abajo, separadas al menos 190 px dentro de cada fila
    for fila in (1, 0):
        evs = sorted([(t[c], txt) for c, txt, ar in eventos if ar == fila])
        pos = [X(v) for v, _ in evs]
        for k in range(1, len(pos)):
            pos[k] = max(pos[k], pos[k - 1] + 190)
        exceso = pos[-1] - 1100
        if exceso > 0:
            pos = [p_ - exceso for p_ in pos]
            for k in range(len(pos) - 2, -1, -1):
                pos[k] = min(pos[k], pos[k + 1] - 190)
        for (v, txt), xl in zip(evs, pos):
            x = X(v)
            yl = 62 if fila else 236
            s.append(f'<circle cx="{f(x)}" cy="150" r="4" class="fill-accent"/>')
            if fila:
                s.append(f'<polyline points="{f(x)},141 {f(x)},112 {f(xl)},96" class="ln"/>')
                s.append(f'<text x="{f(xl)}" y="{yl}" text-anchor="middle" font-size="13" class="t-mono t-accent">{v:.2f} s</text>')
                s.append(f'<text x="{f(xl)}" y="{yl+18}" text-anchor="middle" font-size="13">{txt}</text>')
            else:
                s.append(f'<polyline points="{f(x)},159 {f(x)},190 {f(xl)},204" class="ln"/>')
                s.append(f'<text x="{f(xl)}" y="{yl-14}" text-anchor="middle" font-size="13" class="t-mono t-accent">{v:.2f} s</text>')
                s.append(f'<text x="{f(xl)}" y="{yl+4}" text-anchor="middle" font-size="13">{txt}</text>')
    print("arranque:", {k: t[k] for k in t})
    return svg("arranque", (1180, 260), "\n".join(s), "Línea temporal del arranque de PX4 SITL con Gazebo", 240), t


def fig_escala():
    esc = json.loads((DATOS / "escala.json").read_text())
    uno = [r for r in esc if r["drones"] == 1]
    E = Ejes(90, 30, 430, 250, (1, 100), (1, 100), xlog=True, ylog=True)
    s = [E.marco([1, 2, 4, 8, 16, 32, 100], [1, 2, 4, 8, 16, 32, 100], "factor pedido (PX4_SIM_SPEED_FACTOR)", "factor alcanzado")]
    s.append(E.linea([1, 100], [1, 100], "ln-dash"))
    s.append(f'<text x="{f(E.X(40))}" y="{f(E.Y(55))}" font-size="13" class="t-muted" text-anchor="end">pedido = alcanzado</text>')
    xs = [r["pedido"] for r in uno]; ys = [r["rtf"] for r in uno]
    s.append(E.linea(xs, ys, "ln-accent"))
    for a, b in zip(xs, ys):
        s.append(f'<circle cx="{f(E.X(a))}" cy="{f(E.Y(b))}" r="4" class="fill-accent"/>')
    tope = max(ys)
    s.append(f'<text x="{f(E.X(100))}" y="{f(E.Y(tope)+20)}" text-anchor="end" font-size="14" class="t-accent t-bold">tope ×{tope:.1f}</text>')
    # máximo frente al número de drones (barras)
    mx = {r["drones"]: r["rtf"] for r in esc if r["pedido"] == 100}
    B = Ejes(640, 30, 360, 250, (0, 4), (0, 40))
    s.append(B.marco([], [0, 10, 20, 30, 40], "drones en el mismo mundo", "factor máximo"))
    for i, n in enumerate(sorted(mx)):
        x = B.X(i + 0.2); w = B.X(0.6) - B.X(0)
        s.append(f'<rect x="{f(x)}" y="{f(B.Y(mx[n]))}" width="{f(w)}" height="{f(B.Y(0)-B.Y(mx[n]))}" class="box-accent"/>')
        s.append(f'<text x="{f(x+w/2)}" y="{f(B.Y(mx[n])-6)}" text-anchor="middle" font-size="13" class="t-mono">×{mx[n]:.1f}</text>')
        s.append(f'<text x="{f(x+w/2)}" y="{f(B.Y(0)+18)}" text-anchor="middle" font-size="13" class="t-muted">{n}</text>')
        s.append(f'<text x="{f(x+w/2)}" y="{f(B.Y(mx[n])+18)}" text-anchor="middle" font-size="12" class="t-muted">{mx[n]*n:.0f} d·s/s</text>')
    print("escala:", [(r["pedido"], r["drones"], r["rtf"]) for r in esc])
    return svg("escala", (1040, 330), "\n".join(s), "Factor de tiempo real alcanzado frente al pedido y frente al número de drones", 285), mx


def fig_circulos():
    filas = leer_csv(DATOS / "circulos.csv")
    out = {}
    s = []
    for j, caso in enumerate(["con_ff", "sin_ff"]):
        d = [r for r in filas if r["caso"] == caso]
        t = np.array([float(r["t_s"]) for r in d])
        sp = np.array([[float(r["sp_n"]), float(r["sp_e"])] for r in d])
        gt = np.array([[float(r["gt_n"]), float(r["gt_e"])] for r in d])
        e = np.linalg.norm(gt - sp, axis=1)
        m = t > 5.0  # sin el arranque del primer cuarto de vuelta
        cen = np.array([0.0, 5.0])
        rad = np.linalg.norm(gt - cen, axis=1)
        out[caso] = {"err_medio_m": round(float(e[m].mean()), 3), "err_max_m": round(float(e[m].max()), 3),
                     "radio_medio_m": round(float(rad[m].mean()), 3)}
        E = Ejes(60 + j * 300, 30, 240, 240, (-6.5, 6.5), (-1.5, 11.5))
        # norte hacia arriba, este hacia la derecha
        X = lambda n, e_: E.x0 + (e_ - (-1.5)) / 13.0 * E.w
        Y = lambda n, e_: E.y0 + E.h - (n - (-6.5)) / 13.0 * E.h
        s.append(f'<rect x="{E.x0}" y="{E.y0}" width="{E.w}" height="{E.h}" class="box-soft"/>')
        ang = np.linspace(0, 2 * np.pi, 120)
        s.append('<polyline points="' + " ".join(f"{f(X(5*np.sin(a), 5*(1-np.cos(a))))},{f(Y(5*np.sin(a), 5*(1-np.cos(a))))}" for a in ang)
                 + '" class="ln-dash"/>')
        s.append('<polyline points="' + " ".join(f"{f(X(n, e_))},{f(Y(n, e_))}" for n, e_ in gt[::3]) + '" class="ln-accent"/>')
        nom = "con velocidad de prealimentación" if caso == "con_ff" else "solo posición"
        s.append(f'<text x="{E.x0+E.w/2}" y="{E.y0-10}" text-anchor="middle" font-size="14" class="t-bold">{nom}</text>')
        s.append(f'<text x="{E.x0+E.w/2}" y="{E.y0+E.h+22}" text-anchor="middle" font-size="13" class="t-mono">error medio {out[caso]["err_medio_m"]*100:.0f} cm</text>')
        s.append(f'<text x="{E.x0+E.w/2}" y="{E.y0+E.h+40}" text-anchor="middle" font-size="13" class="t-mono">radio {out[caso]["radio_medio_m"]:.2f} m (pedido 5)</text>')
    # error frente al tiempo
    E = Ejes(700, 30, 300, 240, (0, 40), (0, 2.0))
    s.append(E.marco([0, 10, 20, 30, 40], [0, 0.5, 1.0, 1.5, 2.0], "tiempo en el círculo (s)", "|p − p_ref| (m)"))
    for caso, cl in [("con_ff", "ln-accent"), ("sin_ff", "ln-time")]:
        d = [r for r in filas if r["caso"] == caso]
        t = [float(r["t_s"]) for r in d]
        e = [math.hypot(float(r["gt_n"]) - float(r["sp_n"]), float(r["gt_e"]) - float(r["sp_e"])) for r in d]
        s.append(E.linea(t, e, cl))
    s.append(f'<text x="{f(E.X(39))}" y="{f(E.Y(out["sin_ff"]["err_medio_m"])-8)}" text-anchor="end" font-size="13" class="t-time">solo posición</text>')
    s.append(f'<text x="{f(E.X(39))}" y="{f(E.Y(0.12)-6)}" text-anchor="end" font-size="13" class="t-accent">con prealimentación</text>')
    print("circulos:", out)
    return svg("circulos", (1040, 330), "\n".join(s), "Trayectoria del x500 en dos círculos en offboard y error de seguimiento", 285), out


def fig_escalon():
    d = leer_csv(DATOS / "escalon.csv")
    t = np.array([float(r["t_s"]) for r in d]); sp = np.array([float(r["x_consigna"]) for r in d])
    xg = np.array([float(r["x_verdad"]) for r in d]); xe = np.array([float(r["x_ekf"]) for r in d])
    x0, x1 = sp[0], sp[-1]
    y = (xg - x0) / (x1 - x0)
    t10 = t[np.argmax(y >= 0.1)]; t90 = t[np.argmax(y >= 0.9)]
    sob = (xg.max() - x1) / (x1 - x0) * 100
    dentro = np.flatnonzero(np.abs(xg - x1) > 0.05 * (x1 - x0))
    t5 = t[dentro[-1] + 1] if len(dentro) else t[0]
    E = Ejes(80, 30, 600, 240, (-1, 8), (-0.5, 6))
    s = [E.marco([0, 2, 4, 6, 8], [0, 1, 2, 3, 4, 5, 6], "tiempo desde el escalón (s)", "norte (m)")]
    s.append(E.linea(t, sp - x0, "ln-dash"))
    s.append(E.linea(t, xe - x0, "ln-time"))
    s.append(E.linea(t, xg - x0, "ln-accent"))
    s.append(f'<text x="{f(E.X(7.9))}" y="{f(E.Y(5.0)-8)}" text-anchor="end" font-size="13" class="t-muted">consigna</text>')
    s.append(f'<line x1="{f(E.X(t10))}" y1="{f(E.Y(0))}" x2="{f(E.X(t10))}" y2="{f(E.Y(0.5))}" class="ln"/>')
    s.append(f'<line x1="{f(E.X(t90))}" y1="{f(E.Y(4.5))}" x2="{f(E.X(t90))}" y2="{f(E.Y(5.6))}" class="ln"/>')
    s.append(f'<text x="{f(E.X(t90)+6)}" y="{f(E.Y(5.6))}" font-size="13" class="t-mono">t₉₀ = {t90:.2f} s</text>')
    s.append(f'<text x="{f(E.X(t10)+6)}" y="{f(E.Y(0.5)-4)}" font-size="13" class="t-mono">t₁₀ = {t10:.2f} s</text>')
    s.append(f'<text x="710" y="80" font-size="14" class="t-accent">verdad (Gazebo)</text>')
    s.append(f'<text x="710" y="104" font-size="14" class="t-time">estimación (EKF2)</text>')
    s.append(f'<text x="710" y="150" font-size="14" class="t-mono">subida 10–90 %: {t90-t10:.2f} s</text>')
    s.append(f'<text x="710" y="174" font-size="14" class="t-mono">sobreoscilación: {sob:.1f} %</text>')
    s.append(f'<text x="710" y="198" font-size="14" class="t-mono">dentro del 5 %: {t5:.2f} s</text>')
    out = {"t10": round(float(t10), 2), "t90": round(float(t90), 2), "subida": round(float(t90 - t10), 2), "sobre_pct": round(float(sob), 1), "t5": round(float(t5), 2)}
    print("escalon:", out)
    return svg("escalon", (960, 320), "\n".join(s), "Respuesta del x500 a un escalón de 5 m en offboard", 300), out


def fig_fallo_gnss():
    d = leer_csv(DATOS / "fallo_gnss.csv")
    msgs = json.loads((DATOS / "fallo_gnss_mensajes.json").read_text())
    t = np.array([float(r["t_rel_s"]) for r in d])
    e = np.hypot([float(r["ex_m"]) for r in d], [float(r["ey_m"]) for r in d])
    eph = np.array([float(r["eph_m"]) for r in d]); z = -np.array([float(r["z_verdad_m"]) for r in d])
    nav = [int(r["nav_state"]) for r in d]
    tfin = float(np.ceil(t.max()))
    E = Ejes(80, 30, 700, 220, (-5, tfin), (0, 14))
    s = [E.marco([v for v in [-5, 0, 5, 10, 15, 20, 25, 30] if v <= tfin], [0, 2, 4, 6, 8, 10, 12, 14], "tiempo desde EKF2_GPS_CTRL = 0 (s)", "metros")]
    s.append(f'<line x1="{f(E.X(0))}" y1="{E.y0}" x2="{f(E.X(0))}" y2="{E.y0+E.h}" class="ln-time"/>')
    s.append(f'<text x="{f(E.X(0)+5)}" y="{E.y0+14}" font-size="13" class="t-time">GNSS fuera</text>')
    t_fs = [m[0] for m in msgs if "Failsafe activated" in m[1]][0]
    t_land = [m[0] for m in msgs if "Landing detected" in m[1]][0]
    for tt, txt in [(t_fs, f"failsafe: aterrizaje a ciegas ({t_fs:.1f} s)"), (t_land, f"en tierra ({t_land:.1f} s)")]:
        s.append(f'<line x1="{f(E.X(tt))}" y1="{E.y0}" x2="{f(E.X(tt))}" y2="{E.y0+E.h}" class="ln-dash"/>')
        s.append(f'<text x="{f(E.X(tt)+5)}" y="{E.y0+34 if tt < 10 else E.y0+54}" font-size="13">{txt}</text>')
    s.append(E.linea(t, z, "ln", 'stroke-dasharray="2 3"'))
    s.append(E.linea(t, eph, "ln-time"))
    s.append(E.linea(t, e * 5, "ln-accent"))
    s.append('<text x="800" y="70" font-size="14" class="t-muted">altura real (m)</text>')
    s.append('<text x="800" y="94" font-size="14" class="t-time">eph del EKF2 (m)</text>')
    s.append('<text x="800" y="118" font-size="14" class="t-accent">error real ×5 (m)</text>')
    k_fs = int(np.argmin(np.abs(t - t_fs)))
    out = {"t_failsafe_s": t_fs, "t_tierra_s": t_land, "eph_en_failsafe_m": round(float(eph[k_fs]), 2),
           "err_en_failsafe_m": round(float(e[k_fs]), 3), "err_max_m": round(float(e[t > 0].max()), 3), "eph_max_m": round(float(eph[t > 0].max()), 2),
           "nav": [(round(float(t[k]), 2), nav[k - 1], nav[k]) for k in range(1, len(nav)) if nav[k] != nav[k - 1]]}
    print("fallo_gnss:", out)
    return svg("fallo_gnss", (1000, 300), "\n".join(s), "Error real, eph y altura al quitar el GNSS al EKF2 en estacionario", 290), out


def fig_motor():
    d = leer_csv(DATOS / "fallo_motor.csv")
    t = np.array([float(r["t_rel_s"]) for r in d]); tilt = np.array([float(r["inclinacion_deg"]) for r in d])
    alt = np.array([float(r["altura_m"]) for r in d])
    m = t <= 2.0
    E = Ejes(70, 30, 360, 200, (-0.5, 2.0), (0, 180))
    s = [E.marco([-0.5, 0, 0.5, 1.0, 1.5, 2.0], [0, 45, 90, 135, 180], "tiempo desde «motor 1 off» (s)", "inclinación (°)")]
    s.append(E.linea(t[m], tilt[m], "ln-accent"))
    A = Ejes(560, 30, 360, 200, (-0.5, 2.0), (0, 11))
    s.append(A.marco([-0.5, 0, 0.5, 1.0, 1.5, 2.0], [0, 2, 4, 6, 8, 10], "tiempo desde «motor 1 off» (s)", "altura (m)"))
    s.append(A.linea(t[m], alt[m], "ln-time"))
    t45 = float(t[np.argmax(tilt > 45)]); t90 = float(t[np.argmax(tilt > 90)])
    k = np.flatnonzero((t > 0) & (alt < 1.0)); t_suelo = float(t[k[0]]) if len(k) else float("nan")
    s.append(f'<text x="{f(E.X(t45)+6)}" y="{f(E.Y(45))}" font-size="13" class="t-mono">45° a {t45:.2f} s</text>')
    s.append(f'<text x="{f(A.X(t_suelo)-6)}" y="{f(A.Y(1.0)-8)}" text-anchor="end" font-size="13" class="t-mono">&lt; 1 m a {t_suelo:.2f} s</text>')
    out = {"t45": t45, "t90": t90, "t_suelo": t_suelo}
    print("motor:", out)
    return svg("motor", (960, 290), "\n".join(s), "Inclinación y altura del x500 tras apagar un motor a 10 m", 235), out


def fig_ekf_velocidad():
    r = json.loads((DATOS / "ulog_resumen.json").read_text())
    tau = r["ajuste_e_igual_v_tau"]["tau_s"]; dl = r["ajuste_e_igual_v_tau"]["EKF2_GPS_DELAY_ms"] / 1000
    E = Ejes(80, 30, 460, 220, (0, 5), (0, 60))
    s = [E.marco([0, 1, 2, 3, 4, 5], [0, 10, 20, 30, 40, 50, 60], "velocidad real (m/s)", "error horizontal medio (cm)")]
    s.append(E.linea([0, 5], [0, 500 * dl], "ln-dash"))
    s.append(f'<text x="{f(E.X(3.1))}" y="{f(E.Y(100*dl*3.1)-10)}" font-size="13" class="t-muted" text-anchor="end">v · EKF2_GPS_DELAY</text>')
    for b in r["err_horiz_por_velocidad"]:
        s.append(f'<circle cx="{f(E.X(b["v_media"]))}" cy="{f(E.Y(b["err_medio_cm"]))}" r="5" class="fill-accent"/>')
        s.append(f'<text x="{f(E.X(b["v_media"])+8)}" y="{f(E.Y(b["err_medio_cm"])+15)}" font-size="12" class="t-mono">{b["err_medio_cm"]:.0f}</text>')
    return svg("ekf_velocidad", (600, 300), "\n".join(s), "Error horizontal del EKF2 frente a la velocidad en SITL", 250), {"tau": tau}


def fig_fase():
    fc = 6.3
    fs = np.linspace(100, 1000, 181)
    phi = 360 * fc * 1.5 / fs
    E = Ejes(80, 30, 520, 220, (100, 1000), (0, 35))
    s = [E.marco([100, 250, 400, 600, 800, 1000], [0, 5, 10, 15, 20, 25, 30, 35], "frecuencia del lazo de rate (Hz)", "fase perdida en f_c (°)")]
    s.append(E.linea(fs, phi, "ln-accent"))
    for v, cl, txt in [(250, "fill-time", "SITL"), (800, "fill-accent", "Pixhawk")]:
        p = 360 * fc * 1.5 / v
        s.append(f'<circle cx="{f(E.X(v))}" cy="{f(E.Y(p))}" r="5" class="{cl}"/>')
        s.append(f'<text x="{f(E.X(v)+8)}" y="{f(E.Y(p)-8)}" font-size="14" class="t-mono">{txt}: {p:.1f}°</text>')
    return svg("fase", (640, 300), "\n".join(s), "Fase perdida en el cruce frente a la frecuencia del lazo", 280)


def fig_flujo():
    """Cadena de un dato de la IMU al motor, con las frecuencias medidas (work_queue status en estacionario)."""
    wq = (DATOS / "work_queue_hover.txt").read_text()
    def hz(nombre):
        m = re.search(rf"\)\s+{re.escape(nombre)}\s+([\d.]+) Hz", wq)
        return float(m.group(1)) if m else float("nan")
    cad = [("IMU de Gazebo", "250 Hz (SDF)", None), ("gz_bridge", "sensor_gyro/accel", None),
           ("vehicle_imu", "integra", "vehicle_imu"), ("vehicle_angular_velocity", "filtra", "vehicle_angular_velocity"),
           ("mc_rate_control", "PID de rate", "mc_rate_control"), ("control_allocator", "mezcla", "control_allocator"),
           ("gz_bridge-actuators-esc", "→ motor_speed", "gz_bridge-actuators-esc")]
    s = ['<defs><marker id="t1aflecha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" class="fill-ink3"/></marker></defs>']
    x = 10; w = 152; g = 14
    medidas = {}
    for i, (nom, sub, clave) in enumerate(cad):
        cl = "box-accent" if clave in ("mc_rate_control",) else ("box-time" if i == 0 else "box")
        s.append(f'<rect x="{x}" y="40" width="{w}" height="78" rx="8" class="{cl}"/>')
        corto = nom.replace("gz_bridge-actuators-esc", "gz_bridge (ESC)").replace("vehicle_angular_velocity", "vehicle_angular_vel.")
        s.append(f'<text x="{x+w/2}" y="66" text-anchor="middle" font-size="13" class="t-bold">{corto}</text>')
        s.append(f'<text x="{x+w/2}" y="86" text-anchor="middle" font-size="12" class="t-muted">{sub}</text>')
        if clave:
            v = hz(clave); medidas[clave] = v
            s.append(f'<text x="{x+w/2}" y="108" text-anchor="middle" font-size="13" class="t-mono t-time">{v:.1f} Hz</text>')
        if i < len(cad) - 1:
            s.append(f'<line x1="{x+w}" y1="79" x2="{x+w+g}" y2="79" class="ln" marker-end="url(#t1aflecha)"/>')
        x += w + g
    otros = [("ekf2", "EKF2"), ("mc_att_control", "actitud"), ("mc_pos_control", "posición"), ("flight_mode_manager", "modos"),
             ("vehicle_gps_position", "GNSS"), ("vehicle_air_data", "baro"), ("vehicle_magnetometer", "mag")]
    x = 10
    for clave, nom in otros:
        v = hz(clave); medidas[clave] = v
        s.append(f'<rect x="{x}" y="160" width="152" height="52" rx="8" class="box-soft"/>')
        s.append(f'<text x="{x+76}" y="182" text-anchor="middle" font-size="13">{nom}</text>')
        s.append(f'<text x="{x+76}" y="201" text-anchor="middle" font-size="13" class="t-mono t-time">{v:.1f} Hz</text>')
        x += 166
    s.append('<text x="10" y="148" font-size="13" class="t-muted">En paralelo, en otras colas de trabajo:</text>')
    print("flujo:", medidas)
    return svg("flujo", (1170, 225), "\n".join(s), "Cadena de un dato desde la IMU simulada hasta el motor, con frecuencias medidas", 230), medidas


def insertar(html_path, piezas):
    h = Path(html_path).read_text()
    for nombre, contenido in piezas.items():
        pat = re.compile(rf"(<!--SVG:{nombre}-->).*?(<!--/SVG:{nombre}-->)", re.S)
        if not pat.search(h):
            print("AVISO: falta el marcador", nombre); continue
        h = pat.sub(lambda m: m.group(1) + "\n" + contenido + "\n" + m.group(2), h)
    Path(html_path).write_text(h)


if __name__ == "__main__":
    cifras = {}
    piezas = {}
    for fn in [fig_arranque, fig_escala, fig_circulos, fig_escalon, fig_fallo_gnss, fig_motor, fig_ekf_velocidad, fig_flujo]:
        r = fn()
        piezas[fn.__name__[4:]] = r[0]
        cifras[fn.__name__[4:]] = r[1]
    piezas["fase"] = fig_fase()
    json.dump(cifras, open(AQUI / "salida_figuras.json", "w"), indent=1, default=float)
    if "--insertar" in sys.argv:
        insertar(AQUI.parent / "index.html", piezas)
