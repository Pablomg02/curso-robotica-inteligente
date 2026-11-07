"""Datos de las figuras y tablas del tema A13 (arquitecturas jerárquicas y fusión de modelos).

Uso:  uv run --with numpy python -u figuras_a13.py > salida.txt        (CPU, unos minutos; A13_RAPIDO=1 para probar)

Un dron sigue a una persona que camina entre árboles. Plano horizontal de 40 x 40 m con 8 copas de árbol circulares
(radio de 1,5 a 3 m) a la altura de vuelo; la persona camina a 1,5 m/s con un rumbo que cambia al azar y pasa por debajo de
los árboles. El dron tiene que estar cerca de ella (a menos de 2,5 m en horizontal) sin chocar con las copas.

Dinámica real (100 Hz): la aceleración sigue a la pedida con una constante de tiempo de 0,08 s y hay arrastre
cuadrático (c = 0,05 1/m), que el planificador no conoce. Aceleración pedida limitada a 6 m/s^2.

Sistema lento (System 2): MPPI con 256 trayectorias de 2 s (20 pasos de 0,1 s) sobre un doble integrador sin arrastre,
que predice a la persona con velocidad constante desde su última observación y penaliza acercarse a las copas. Tarda
delta segundos en dar su plan (la latencia se simula: es la variable del experimento) y se vuelve a llamar en cuanto
termina, así que replanifica cada delta.
Sistema rápido (System 1): un seguidor PD a 100 Hz de la trayectoria del plan (posición, velocidad y aceleración de
referencia) con el estado actual.

Arquitecturas:
  A. Solo lento: el primer mando del plan se mantiene hasta el plan siguiente (un modelo grande que manda directo).
  B. Lento con trozos en lazo abierto: se ejecutan los mandos del plan uno tras otro, sin realimentación.
  C. Jerárquica sin compensar: el seguidor rápido sigue el plan, que se calculó con el estado de cuando se pidió y se
     empieza a seguir desde el principio cuando llega (desfasado delta).
  D. Jerárquica con compensación: el plan se calcula desde el estado previsto para cuando llegue (el actual más lo que el
     plan en curso prevé moverse durante delta) y se sigue en tiempo absoluto (la idea de RTC y RoboDual).
  E. Solo rápido: un PD hacia la posición actual de la persona, sin planificador (no ve los árboles).
Métricas en 100 escenas de 20 s: fracción del tiempo (de 3 a 20 s) a menos de 2,5 m de la persona, contando como lejos
todo lo que queda tras un choque, y fracción de episodios con choque.
"""
import math
import os
import time

import numpy as np

RAPIDO = os.environ.get("A13_RAPIDO") == "1"
N_ESC = 8 if RAPIDO else 100
T_FIN, DT = 20.0, 0.01
DT_P, HP, K_MPPI = 0.1, 20, 256
A_MAX, TAU_A, C_D, R_DRON = 6.0, 0.08, 0.05, 0.3
V_PER, CERCA = 1.5, 2.5
KP, KD = 4.0, 3.0
LAT = [0.05, 0.1, 0.2, 0.4, 0.8]


def escenas(n, sem):
    rng = np.random.default_rng(sem)
    E = []
    for _ in range(n):
        obs = []
        while len(obs) < 8:
            c, r = rng.uniform(4, 36, 2), rng.uniform(1.5, 3.0)
            if all(np.linalg.norm(c - o[:2]) > r + o[2] + 2.0 for o in obs):
                obs.append(np.array([*c, r]))
        obs = np.array(obs)
        while True:
            p0 = rng.uniform(5, 35, 2)
            if np.all(np.linalg.norm(obs[:, :2] - p0, axis=1) > obs[:, 2] + 1.5):
                break
        g0 = p0 + rng.uniform(-3, 3, 2)
        # trayectoria de la persona: rumbo con giros aleatorios, rebota en los bordes
        n_t = int(T_FIN / DT) + int(2 / DT) + 2
        th = rng.uniform(0, 2 * np.pi)
        g = np.zeros((n_t, 2))
        vg = np.zeros((n_t, 2))
        g[0], w = g0, 0.0
        for k in range(1, n_t):
            w += -w * DT / 1.0 + rng.normal(0, 0.8) * math.sqrt(DT)
            th += w * DT
            v = V_PER * np.array([math.cos(th), math.sin(th)])
            q = g[k - 1] + v * DT
            for i in range(2):
                if not 2 < q[i] < 38:
                    v[i] = -v[i]
                    th = math.atan2(v[1], v[0])
                    q[i] = g[k - 1][i] + v[i] * DT
            g[k], vg[k] = q, v
        vg[0] = vg[1]
        E.append(dict(obs=obs, p0=p0, g=g, vg=vg))
    return E


def mppi(p, v, a_prev, g_obs, vg_obs, t_obs, t0, obs, rng, desp=1):
    """Plan para E escenas a la vez desde (p, v) en el instante t0. Devuelve aceleraciones (E, HP, 2).
    Arranque en caliente: el plan anterior desplazado desp pasos (los que han pasado desde su inicio hasta t0)."""
    E = p.shape[0]
    desp = int(min(max(desp, 1), HP - 1))
    a_nom = np.concatenate([a_prev[:, desp:], np.repeat(a_prev[:, -1:], desp, 1)], 1)
    a = np.clip(a_nom[:, None] + rng.normal(0, 3.0, (E, K_MPPI, HP, 2)), -A_MAX, A_MAX)
    a[:, 0] = a_nom                                                             # incluye el plan anterior
    pp = np.repeat(p[:, None], K_MPPI, 1)
    vv = np.repeat(v[:, None], K_MPPI, 1)
    J = np.zeros((E, K_MPPI))
    for k in range(HP):
        pp = pp + vv * DT_P + 0.5 * a[:, :, k] * DT_P ** 2
        vv = vv + a[:, :, k] * DT_P
        tk = t0 + (k + 1) * DT_P
        gk = g_obs + vg_obs * (tk - t_obs)                                      # persona a velocidad constante
        d = np.linalg.norm(pp - gk[:, None], axis=-1)
        J += np.maximum(d - 1.0, 0) ** 2
        dobs = np.linalg.norm(pp[:, :, None] - obs[:, None, :, :2], axis=-1) - obs[:, None, :, 2] - R_DRON - 0.5
        J += 400.0 * np.sum(np.maximum(-dobs, 0) ** 2 + 2.0 * (dobs < -0.5), -1)
        J += 0.01 * np.sum(a[:, :, k] ** 2, -1)
    w = np.exp(-(J - J.min(1, keepdims=True)) / 2.0)
    w /= w.sum(1, keepdims=True)
    return np.einsum("ek,ekhd->ehd", w, a)


def referencia(p, v, a_plan):
    """Integra el doble integrador nominal: posiciones y velocidades de referencia en los nodos del plan."""
    P, Vv = [p], [v]
    for k in range(HP):
        P.append(P[-1] + Vv[-1] * DT_P + 0.5 * a_plan[:, k] * DT_P ** 2)
        Vv.append(Vv[-1] + a_plan[:, k] * DT_P)
    return np.stack(P, 1), np.stack(Vv, 1)


def en_plan(Pr, Vr, Ar, tau):
    """Referencia del plan en el instante tau (s desde su inicio), por tramos de aceleración constante."""
    k = np.clip((tau / DT_P).astype(int), 0, HP - 1)
    s = np.clip(tau - k * DT_P, 0, None)[:, None]
    i = np.arange(len(k))
    a = Ar[i, k]
    return Pr[i, k] + Vr[i, k] * s + 0.5 * a * s ** 2, Vr[i, k] + a * s, a


def simular(E, arq, delta, sem=7):
    rng = np.random.default_rng(sem)
    n = len(E)
    obs = np.stack([e["obs"] for e in E])
    G = np.stack([e["g"] for e in E])
    VG = np.stack([e["vg"] for e in E])
    p = np.stack([e["p0"] for e in E]).astype(float)
    v = np.zeros((n, 2))
    acc = np.zeros((n, 2))
    choque = np.zeros(n, bool)
    cerca = np.zeros(n)
    pasos = int(T_FIN / DT)
    # plan en curso (empieza en t = 0 con mandos nulos) y plan pedido
    Ar = np.zeros((n, HP, 2))
    Pr, Vr = referencia(p, v, Ar)
    t_ini = np.zeros(n)
    t_pedido, pend = 0.0, None
    cada = max(1, int(round(delta / DT)))
    for k in range(pasos):
        t = k * DT
        if arq != "E":
            if pend is None:
                # se pide un plan: con lo observado ahora (persona incluida) y desde el estado de partida
                kg = k
                if arq == "D":
                    # estado previsto para cuando llegue el plan: el actual más lo que el plan en curso prevé moverse en delta
                    pa, va, _ = en_plan(Pr, Vr, Ar, t - t_ini)
                    pb, vb, _ = en_plan(Pr, Vr, Ar, t + delta - t_ini)
                    p0, v0, t0 = p + (pb - pa), v + (vb - va), t + delta
                else:
                    p0, v0, t0 = p.copy(), v.copy(), t
                desp = round(float(np.mean(t0 - t_ini)) / DT_P)
                nuevo = mppi(p0, v0, Ar, G[:, kg], VG[:, kg], t, t0, obs, rng, desp)
                pend = (nuevo, p0, v0, t0, k + cada)
            if k >= pend[4]:
                Ar = pend[0]
                Pr, Vr = referencia(pend[1], pend[2], Ar)
                t_ini = np.full(n, pend[3] if arq == "D" else t)
                pend = None
        # mando
        if arq == "A":
            a_cmd = Ar[:, 0]
        elif arq == "B":
            _, _, a_cmd = en_plan(Pr, Vr, Ar, t - t_ini)
        elif arq in ("C", "D"):
            pr, vr, ar = en_plan(Pr, Vr, Ar, t - t_ini)
            a_cmd = ar + KP * (pr - p) + KD * (vr - v)
        else:
            a_cmd = KP * (G[:, k] - p) + KD * (VG[:, k] - v)
        nrm = np.linalg.norm(a_cmd, axis=1, keepdims=True)
        a_cmd = np.where(nrm > A_MAX, a_cmd * A_MAX / np.maximum(nrm, 1e-9), a_cmd)
        # dinámica real
        acc = acc + (a_cmd - acc) * DT / TAU_A
        vivo = ~choque
        v[vivo] += (acc[vivo] - C_D * np.linalg.norm(v[vivo], axis=1, keepdims=True) * v[vivo]) * DT
        p[vivo] += v[vivo] * DT
        dob = np.linalg.norm(p[:, None] - obs[:, :, :2], axis=-1) - obs[:, :, 2]
        choque |= np.any(dob < R_DRON, 1)
        if t >= 3.0:
            cerca += (~choque) & (np.linalg.norm(p - G[:, k + 1], axis=1) < CERCA)
    return cerca / (pasos - int(3.0 / DT)), choque


E = escenas(N_ESC, 0)
print(f"{N_ESC} escenas de {T_FIN:.0f} s; persona a {V_PER} m/s; MPPI con {K_MPPI} trayectorias de {HP * DT_P:.0f} s")
# coste real de una llamada a MPPI para una escena
rng = np.random.default_rng(1)
e = E[0]
ts = []
for _ in range(20):
    t0 = time.perf_counter()
    mppi(e["p0"][None], np.zeros((1, 2)), np.zeros((1, HP, 2)), e["g"][None, 0], e["vg"][None, 0], 0.0, 0.0, e["obs"][None], rng)
    ts.append(time.perf_counter() - t0)
print(f"una llamada a MPPI para una escena en CPU (numpy): {np.median(ts) * 1e3:.1f} ms")

RES = {}
t_ini = time.perf_counter()
fr, ch = simular(E, "E", 0.05)
RES[("E", 0)] = (fr.mean(), ch.mean())
print(f"E solo rápido (PD hacia la persona, sin planificador): cerca {fr.mean():.3f}, choques {ch.mean():.2f}", flush=True)
for arq in ("A", "B", "C", "D"):
    for d in LAT:
        fr, ch = simular(E, arq, d)
        RES[(arq, d)] = (fr.mean(), ch.mean())
        print(f"{arq} latencia {d * 1e3:.0f} ms: cerca {fr.mean():.3f}, choques {ch.mean():.2f}", flush=True)
print(f"tiempo de simulación: {time.perf_counter() - t_ini:.0f} s")

# trazados para la figura: latencia en escala logarítmica de 50 a 800 ms (x de 60 a 500); fracción cerca de 0 a 1 (y de 230 a 30)
X = lambda d: 60 + (math.log2(d) - math.log2(0.05)) / (math.log2(0.8) - math.log2(0.05)) * 440
Y = lambda f: 230 - f * 200
for arq in ("A", "B", "C", "D"):
    print(f"path cerca {arq}:", "M" + " L".join(f"{X(d):.1f},{Y(RES[(arq, d)][0]):.1f}" for d in LAT))
print(f"path cerca E: M60,{Y(RES[('E', 0)][0]):.1f} L500,{Y(RES[('E', 0)][0]):.1f}")
