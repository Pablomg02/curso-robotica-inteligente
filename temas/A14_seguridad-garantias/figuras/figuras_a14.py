"""Datos de las figuras y tablas del tema A14 (seguridad y garantías): un filtro de seguridad CBF sobre una política agresiva.

Uso:  uv run --with numpy python -u figuras_a14.py > salida.txt        (CPU, un par de minutos; A14_RAPIDO=1 para probar)

Un dron (doble integrador en el plano, aceleración limitada a A_MAX por eje) vuela de izquierda a derecha por un campo de
30 x 20 m con 7 obstáculos circulares (copas o postes, radio de 0,8 a 1,8 m) y una zona prohibida circular de 3 m de
radio. La política nominal, que hace de "política aprendida" agresiva, es un PD hacia la meta que no ve nada.

Filtro de seguridad: para cada obstáculo i, h_i(p) = |p - c_i|^2 - (r_i + r_dron)^2 >= 0 define el conjunto seguro. Como
la entrada es la aceleración, h tiene grado relativo 2 y se usa una CBF de orden alto (exponencial):
    psi_1 = hdot + a1 h,     condición:  psi_1dot + a2 psi_1 >= 0,
que es lineal en la aceleración: 2 (p - c)·u + 2 |v|^2 + (a1 + a2) hdot + a1 a2 h >= 0. El filtro resuelve en cada paso
    min |u - u_nom|^2   sujeto a las condiciones de todos los obstáculos y |u_x|, |u_y| <= A_MAX
con el método de Hildreth (ascenso por coordenadas en el dual), vectorizado sobre todos los episodios.

Experimentos (200 escenas de 12 s, paso de 10 ms):
  1. Sin filtro / con filtro y estado perfecto, para varias ganancias a1 = a2 = alfa; y con un margen de un paso (el
     radio se infla con lo que se recorre en un paso más 2 cm) contra el error de discretizar la condición continua.
  2. Con ruido en la posición y la velocidad estimadas (sigma de 0,1 y 0,3 m): con el margen de un paso y con un
     margen robusto además (2 sigma más lo que se recorre en 2 pasos a la velocidad estimada).
  3. Con ráfagas de viento (aceleración de Ornstein-Uhlenbeck, desviación 1 m/s^2) que el filtro no conoce: con el
     filtro normal y con una CBF robusta que supone una perturbación acotada por w_max (resta 2 |p - c| w_max).
  4. Con aceleración máxima baja (2 m/s^2): la condición CBF puede ser imposible de cumplir con la entrada disponible.
Métricas: choques (episodios que entran en un obstáculo o en la zona prohibida), llegada a la meta, distancia mínima a
un obstáculo, intervención media |u - u_nom| y pasos en que el QP no se pudo satisfacer.
"""
import math
import os
import time

import numpy as np

RAPIDO = os.environ.get("A14_RAPIDO") == "1"
N_ESC = 20 if RAPIDO else 200
DT, T_FIN = 0.01, 12.0
A_MAX, V_MAX, KP, KD = 6.0, 6.0, 2.0, 2.2
R_DRON = 0.3
META = np.array([28.0, 10.0])


def escenas(n, sem):
    rng = np.random.default_rng(sem)
    C, Rr = np.zeros((n, 8, 2)), np.zeros((n, 8))
    P0 = np.zeros((n, 2))
    for e in range(n):
        obs = []
        # zona prohibida: un círculo de 3 m en la mitad del campo, cerca de la línea recta
        obs.append((np.array([rng.uniform(12, 18), rng.uniform(7, 13)]), 3.0))
        while len(obs) < 8:
            c, r = np.array([rng.uniform(5, 25), rng.uniform(2, 18)]), rng.uniform(0.8, 1.8)
            if all(np.linalg.norm(c - o) > r + ro + 1.5 for o, ro in obs):
                obs.append((c, r))
        for i, (c, r) in enumerate(obs):
            C[e, i], Rr[e, i] = c, r
        P0[e] = [2.0, rng.uniform(6, 14)]
    return C, Rr, P0


def nominal(p, v):
    """La política agresiva: PD hacia la meta, saturada, que no ve los obstáculos."""
    u = KP * (META - p) - KD * v
    return np.clip(u, -A_MAX, A_MAX)


def filtro(p, v, u_nom, C, R, alfa, amax, margen=0.0, iters=60, w_max=0.0):
    """QP de la CBF de orden alto con el método de Hildreth. Devuelve (u, pasos imposibles)."""
    d = p[:, None, :] - C                                   # (E, m, 2)
    Rm = R + R_DRON + margen if np.ndim(margen) == 0 else R + R_DRON + margen[:, None]
    h = np.sum(d * d, -1) - Rm ** 2
    hd = 2 * np.sum(d * v[:, None, :], -1)
    # restricciones G u <= g:  -2 d·u <= 2|v|^2 + 2 alfa hd + alfa^2 h
    G = np.concatenate([-2 * d, np.broadcast_to(np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]]), (p.shape[0], 4, 2))], 1)
    # con una perturbación de aceleración acotada por w_max, el peor caso resta 2 |d| w_max (CBF robusta)
    g = np.concatenate([2 * np.sum(v * v, -1)[:, None] + 2 * alfa * hd + alfa ** 2 * h - 2 * np.linalg.norm(d, axis=-1) * w_max,
                        np.full((p.shape[0], 4), amax)], 1)
    nG = np.sum(G * G, -1) + 1e-12
    lam = np.zeros(g.shape)
    u = u_nom.copy()
    for _ in range(iters):
        for j in range(G.shape[1]):
            r = np.sum(G[:, j] * u, -1) - g[:, j]            # violación de la restricción j
            nuevo = np.maximum(0.0, lam[:, j] + r / nG[:, j])
            u = u - (nuevo - lam[:, j])[:, None] * G[:, j]
            lam[:, j] = nuevo
    viol = np.max(np.sum(G * u[:, None, :], -1) - g, 1) > 1e-3
    return np.clip(u, -amax, amax), viol


def simular(C, R, P0, filtrar=True, alfa=2.0, amax=A_MAX, ruido=0.0, robusto=False, retardo=0.0, prediccion=False, sem=3,
            paso_margen=False, viento=0.0, w_max=0.0):
    rng = np.random.default_rng(sem)
    n = len(P0)
    p, v = P0.copy(), np.zeros((n, 2))
    choque = np.zeros(n, bool)
    pen = np.zeros(n)
    w = np.zeros((n, 2))
    llega = np.zeros(n, bool)
    dmin = np.full(n, np.inf)
    interv, imposible, pasos = 0.0, 0, int(T_FIN / DT)
    k_ret = int(round(retardo / DT))
    hist_p, hist_v, hist_u = [p.copy()] * (k_ret + 1), [v.copy()] * (k_ret + 1), [np.zeros((n, 2))] * (k_ret + 1)
    for k in range(pasos):
        # estado que ve el controlador: con latencia y con ruido
        pe, ve = hist_p[0].copy(), hist_v[0].copy()
        if prediccion and k_ret > 0:
            for j in range(k_ret):                            # predecir con los mandos ya enviados
                ve = ve + hist_u[j + 1] * DT
                pe = pe + ve * DT
        if ruido > 0:
            pe = pe + rng.normal(0, ruido, pe.shape)
            ve = ve + rng.normal(0, ruido, ve.shape)
        u_nom = nominal(pe, ve)
        if filtrar:
            margen = (2 * ruido + 2 * DT * np.linalg.norm(ve, axis=1)) if robusto else 0.0
            if paso_margen:
                margen = margen + DT * np.linalg.norm(ve, axis=1) + 0.02
            u, viol = filtro(pe, ve, u_nom, C, R, alfa, amax, margen, w_max=w_max)
            imposible += int(np.sum(viol & ~choque & ~llega))
        else:
            u = np.clip(u_nom, -amax, amax)
        u = np.clip(u, -amax, amax)
        act = ~choque & ~llega
        interv += float(np.sum(np.linalg.norm(u - u_nom, axis=1)[act]))
        if viento > 0:                                        # ráfagas: proceso de Ornstein-Uhlenbeck con tau = 1 s
            w = w - w * DT + viento * math.sqrt(2 * DT) * rng.normal(size=w.shape)
        v[act] = v[act] + (u[act] + w[act]) * DT
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(sp > V_MAX, v * V_MAX / np.maximum(sp, 1e-9), v)
        p[act] = p[act] + v[act] * DT
        dist = np.linalg.norm(p[:, None] - C, axis=-1) - R - R_DRON
        dmin = np.where(act, np.minimum(dmin, dist.min(1)), dmin)
        pen = np.where(act, np.maximum(pen, -dist.min(1)), pen)
        choque |= act & (dist.min(1) < 0)
        llega |= act & (np.linalg.norm(p - META, axis=1) < 0.5)
        hist_p = hist_p[1:] + [p.copy()]
        hist_v = hist_v[1:] + [v.copy()]
        hist_u = hist_u[1:] + [u.copy()]
    return dict(choque=choque.mean(), llega=llega.mean(), dmin=float(np.median(dmin)), interv=interv / (n * pasos),
                imposible=imposible / (n * pasos), pen=float(np.median(pen[choque]) * 1000) if choque.any() else 0.0)


def linea(nombre, r):
    print(f"  {nombre:52s}: choques {r['choque']:.3f} (penetración mediana {r['pen']:.2f} mm) | llegan {r['llega']:.3f} | distancia mínima mediana {r['dmin']:.2f} m | "
          f"intervención media {r['interv']:.2f} m/s2 | pasos con QP imposible {100 * r['imposible']:.2f} %", flush=True)


C, R, P0 = escenas(N_ESC, 0)
print(f"{N_ESC} escenas de {T_FIN:.0f} s; 7 obstáculos y una zona prohibida de 3 m; aceleración máxima {A_MAX} m/s2 por eje")
t0 = time.perf_counter()
linea("1. sin filtro", simular(C, R, P0, filtrar=False))
for alfa in (0.5, 1.0, 2.0, 4.0):
    linea(f"1. con filtro, alfa = {alfa}", simular(C, R, P0, alfa=alfa))
for alfa in (2.0, 4.0):
    linea(f"1. con filtro y margen de un paso, alfa = {alfa}", simular(C, R, P0, alfa=alfa, paso_margen=True))
for s in (0.1, 0.3):
    linea(f"2. ruido {s} m, filtro con margen de un paso", simular(C, R, P0, ruido=s, paso_margen=True))
    linea(f"2. ruido {s} m, filtro con margen robusto", simular(C, R, P0, ruido=s, robusto=True, paso_margen=True))
linea("3. viento de 1 m/s2, filtro con margen de un paso", simular(C, R, P0, viento=1.0, paso_margen=True))
for wm in (1.0, 2.5):
    linea(f"3. viento de 1 m/s2, filtro robusto con w_max = {wm}", simular(C, R, P0, viento=1.0, paso_margen=True, w_max=wm))
linea("4. aceleración máxima 2 m/s2, sin filtro", simular(C, R, P0, filtrar=False, amax=2.0))
for alfa in (2.0, 0.5):
    linea(f"4. aceleración máxima 2 m/s2, filtro, alfa = {alfa}", simular(C, R, P0, amax=2.0, alfa=alfa, paso_margen=True))
# coste del filtro: tiempo de una llamada para una escena
tt = []
for _ in range(200):
    t1 = time.perf_counter()
    filtro(P0[:1], np.zeros((1, 2)), nominal(P0[:1], np.zeros((1, 2))), C[:1], R[:1], 2.0, A_MAX)
    tt.append(time.perf_counter() - t1)
print(f"una llamada al filtro (8 restricciones de seguridad + 4 de entrada, 60 iteraciones, numpy) para una escena: {np.median(tt) * 1e3:.2f} ms")
print(f"tiempo total: {time.perf_counter() - t0:.0f} s")

# trayectorias de una escena para la figura (x de 0 a 30 m -> 20 a 500 px; y de 0 a 20 m -> 300 a 20 px)
X = lambda x: 20 + x * 16
Y = lambda y: 300 - y * 14
e = 0
for i in range(8):
    print(f"obstaculo {i}: {X(C[e, i, 0]):.1f},{Y(C[e, i, 1]):.1f} r {R[e, i] * 16:.1f} r_seg {(R[e, i] + R_DRON) * 16:.1f} {'zona' if i == 0 else 'obs'}")
for nombre, kw in (("sin filtro", dict(filtrar=False)), ("con filtro", dict(alfa=2.0))):
    p, v = P0[e:e + 1].copy(), np.zeros((1, 2))
    pts = [p[0].copy()]
    for k in range(int(T_FIN / DT)):
        u_nom = nominal(p, v)
        u = filtro(p, v, u_nom, C[e:e + 1], R[e:e + 1], kw.get("alfa", 2.0), A_MAX)[0] if kw.get("filtrar", True) else u_nom
        v = v + u * DT
        p = p + v * DT
        if k % 10 == 0:
            pts.append(p[0].copy())
        if np.linalg.norm(p[0] - META) < 0.5:
            break
    print(f"path {nombre}:", "M" + " L".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts))
print(f"meta: {X(META[0]):.1f},{Y(META[1]):.1f}")
