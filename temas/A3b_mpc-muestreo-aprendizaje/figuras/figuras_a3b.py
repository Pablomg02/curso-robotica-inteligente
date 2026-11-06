"""Datos de las figuras y tablas del tema A3b (MPC por muestreo, MPC con aprendizaje, MPC frente a RL).

Uso:  python3 figuras_a3b.py > salida.txt      (numpy; unos minutos en CPU)
Escenario: punto con aceleración acotada (|a_i| <= 4 m/s^2) que va de (0, 0) a (8, 0) entre dos obstáculos
circulares, con un planificador por muestreo replanificando a 20 Hz con horizonte de 1,5 s.
"""
import math
import time
import numpy as np

DT, H, AMAX = 0.05, 30, 4.0
GOAL = np.array([8.0, 0.0])
OBS = [(np.array([3.0, 0.2]), 1.0), (np.array([5.5, -0.6]), 0.8)]
SIG = 2.0                        # desviación típica del ruido de muestreo (m/s^2)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def logmap(d0, d1, p0, p1):
    return lambda v: p0 + (math.log10(v) - math.log10(d0)) * (p1 - p0) / (math.log10(d1) - math.log10(d0))


def rollout_cost(x0, U):
    """U: (K, H, 2) aceleraciones. Devuelve coste (K,) y trayectorias (K, H+1, 2)."""
    K = U.shape[0]
    p = np.repeat(x0[None, :2], K, 0); v = np.repeat(x0[None, 2:], K, 0)
    S = np.zeros(K); P = np.zeros((K, H + 1, 2)); P[:, 0] = p
    for k in range(H):
        a = np.clip(U[:, k], -AMAX, AMAX)
        v = v + DT * a; p = p + DT * v
        P[:, k + 1] = p
        S += DT * (np.sum((p - GOAL) ** 2, 1) + 0.01 * np.sum(a**2, 1))
        for c, r in OBS:
            S += 1e3 * (np.linalg.norm(p - c, axis=1) < r + 0.15)      # choque (con 15 cm de margen)
    S += 10 * np.sum((p - GOAL) ** 2, 1) + 1.0 * np.sum(v**2, 1)
    return S, P


LAM = 10.0                       # temperatura de MPPI (ver sección 3)


def plan(metodo, x0, U, K, rng, lam=LAM):
    if metodo == "aleatorio":
        E = rng.normal(0, SIG, (K, H, 2))
        Uk = U[None] + E
        S, _ = rollout_cost(x0, Uk)
        return Uk[np.argmin(S)]
    if metodo == "cem":
        mu, sd = U.copy(), np.full_like(U, SIG)
        it, n = 4, max(4, K // 4)
        for _ in range(it):
            Uk = mu[None] + sd[None] * rng.normal(0, 1, (n, H, 2))
            S, _ = rollout_cost(x0, Uk)
            el = Uk[np.argsort(S)[:max(2, n // 10)]]
            mu, sd = el.mean(0), el.std(0) + 0.05
        return mu
    if metodo == "mppi":
        E = rng.normal(0, SIG, (K, H, 2))
        S, _ = rollout_cost(x0, U[None] + E)
        w = np.exp(-(S - S.min()) / lam); w /= w.sum()
        return U + np.einsum("k,khd->hd", w, E)


def episodio(metodo, K, seed, T=12.0, guarda=False):
    rng = np.random.default_rng(seed)
    x = np.zeros(4); U = np.zeros((H, 2))
    tray = [x[:2].copy()]
    for i in range(int(T / DT)):
        U = plan(metodo, x, U, K, rng)
        a = np.clip(U[0], -AMAX, AMAX)
        x[2:] += DT * a; x[:2] += DT * x[2:]
        tray.append(x[:2].copy())
        if any(np.linalg.norm(x[:2] - c) < r for c, r in OBS):
            return False, None, np.array(tray)
        if np.linalg.norm(x[:2] - GOAL) < 0.3 and np.linalg.norm(x[2:]) < 1.0:
            return True, (i + 1) * DT, np.array(tray)
        U = np.vstack([U[1:], U[-1:]])
    return False, None, np.array(tray)


print("=" * 70)
print("1. ÉXITO Y TIEMPO FRENTE AL NÚMERO DE MUESTRAS POR PASO (20 semillas por caso)")
Ks = [16, 64, 256, 1024]
res = {}
t0 = time.perf_counter()
for metodo in ("aleatorio", "cem", "mppi"):
    for K in Ks:
        ok, tiempos = 0, []
        for s in range(20):
            exito, tl, _ = episodio(metodo, K, s)
            ok += exito
            if exito:
                tiempos.append(tl)
        res[(metodo, K)] = (ok / 20, np.mean(tiempos) if tiempos else float("nan"))
        print(f"  {metodo:9s} K = {K:5d}: éxito {100*ok/20:5.0f} %, tiempo medio a la meta {res[(metodo,K)][1]:5.2f} s")
print(f"  (cálculo total {time.perf_counter()-t0:.0f} s)")

print("=" * 70)
print("2. TRAYECTORIAS DE EJEMPLO (K = 256, semilla 0)")
X2 = lin(-0.5, 8.5, 40, 500)
Y2 = lin(-2.1, 2.1, 232, 18)
for metodo in ("aleatorio", "cem", "mppi"):
    ex, tl, tr = episodio(metodo, 256, 0)
    print(f"  {metodo}: {'llega' if ex else 'no llega'}" + (f" en {tl:.2f} s" if ex else ""))
    print(f"path tray {metodo}:", path(tr[:, 0], tr[:, 1], X2, Y2))
for (c, r), nm in zip(OBS, ("1", "2")):
    print(f"  obstáculo {nm}: cx = {X2(c[0]):.1f}, cy = {Y2(c[1]):.1f}, r = {r*(X2(1)-X2(0)):.1f} (x) / {r*(Y2(0)-Y2(1)):.1f} (y)")
print(f"  inicio ({X2(0):.1f}, {Y2(0):.1f}); meta ({X2(8):.1f}, {Y2(0):.1f})")

rng = np.random.default_rng(0)
for K in (256, 1024, 4096):
    t0 = time.perf_counter()
    for _ in range(20):
        plan("mppi", np.zeros(4), np.zeros((H, 2)), K, rng)
    print(f"  MPPI, un paso de planificación con K = {K} en numpy (CPU): {1000*(time.perf_counter()-t0)/20:.1f} ms")
# Figura de éxito: x log K 16..1024 -> 60..500 ; y 0..100 % -> 250..30
X1 = logmap(16, 1024, 60, 500)
Y1 = lin(0, 100, 250, 30)
for metodo in ("aleatorio", "cem", "mppi"):
    print(f"path exito {metodo}:", path(Ks, [100 * res[(metodo, K)][0] for K in Ks], X1, Y1))

print("=" * 70)
print("3. TEMPERATURA DE MPPI: TAMAÑO EFECTIVO DE MUESTRA (K = 1024, desde el inicio)")
rng = np.random.default_rng(1)
E = rng.normal(0, SIG, (1024, H, 2))
S, _ = rollout_cost(np.zeros(4), E)
lams = np.logspace(-2, 3, 41)
ess = []
for lam in lams:
    w = np.exp(-(S - S.min()) / lam); w /= w.sum()
    ess.append(1 / np.sum(w**2))
ess = np.array(ess)
for lam in (0.01, 0.1, 1, 10, 100, 1000):
    w = np.exp(-(S - S.min()) / lam); w /= w.sum()
    print(f"  lambda = {lam:7.2f}: ESS = {1/np.sum(w**2):7.1f} de 1024")
print(f"  coste de las muestras: mínimo {S.min():.1f}, mediana {np.median(S):.1f}")
X3 = logmap(0.01, 1000, 60, 500)
Y3 = logmap(1, 1024, 250, 30)
print("path ess:", path(lams, np.maximum(ess, 1), X3, Y3))
