"""Modelo de Vicsek: orden del grupo frente al ruido (figura de la parte 1 de A15b).

Uso:  python3 vicsek_a15b.py > salida_vicsek.txt      (numpy, CPU, unos minutos)

Partículas con velocidad constante v0 en una caja periódica de lado L; en cada paso, cada una toma la dirección media de
sus vecinas a menos de r = 1 (incluida ella) más un ruido uniforme en [-eta/2, eta/2]. Parámetro de orden
v_a = |media de e^{i theta}|: 1 si todas van en la misma dirección, cerca de 0 si van al azar. Densidad 4, como en una de
las series de Vicsek et al. (1995), con dos tamaños para ver el efecto del tamaño finito.
"""
import numpy as np

V0, R, RHO = 0.03, 1.0, 4.0
ETAS = np.round(np.arange(0.0, 5.01, 0.5), 2)
PASOS, DESCARTE, SEMILLAS = 1500, 750, 3


def simular(n, eta, semilla):
    rng = np.random.default_rng(semilla)
    L = np.sqrt(n / RHO)
    x = rng.uniform(0, L, (n, 2))
    th = rng.uniform(-np.pi, np.pi, n)
    orden = []
    for t in range(PASOS):
        d = x[:, None, :] - x[None, :, :]
        d -= L * np.round(d / L)                                  # distancias con la caja periódica
        vec = (d ** 2).sum(-1) < R ** 2
        s, c = vec @ np.sin(th), vec @ np.cos(th)
        th = np.arctan2(s, c) + rng.uniform(-eta / 2, eta / 2, n)
        x = (x + V0 * np.stack([np.cos(th), np.sin(th)], 1)) % L
        if t >= DESCARTE:
            orden.append(np.abs(np.exp(1j * th).mean()))
    return float(np.mean(orden))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


X, Y = lin(0, 5, 60, 520), lin(0, 1, 250, 30)
for n in (100, 400):
    va = [np.mean([simular(n, eta, s) for s in range(SEMILLAS)]) for eta in ETAS]
    print(f"N = {n} (L = {np.sqrt(n / RHO):.1f}): " + " | ".join(f"eta {e:.1f}: v_a {v:.3f}" for e, v in zip(ETAS, va)), flush=True)
    print(f"path vicsek {n}:", "M" + " L".join(f"{X(e):.1f},{Y(v):.1f}" for e, v in zip(ETAS, va)))
