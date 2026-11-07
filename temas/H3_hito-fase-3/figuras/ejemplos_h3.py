"""Ejemplos numéricos de las hojas nuevas del enunciado de H3.

numpy, CPU, segundos. Uso:  python3 ejemplos_h3.py > salida_ejemplos.txt
Las cuentas del examen siguen en calculos_h3.py (salida.txt). Las cifras de
partida (P_D = 0,7, el ejemplo de dos celdas de A15a, 31,9 pasos de A7b,
2 de 64 de A8b, 10,0 y 0,021 ms de A10, 0,910 y 0,879 de A15b) son las de
los temas; aquí solo se hacen cuentas.
"""
import numpy as np

rng = np.random.default_rng(0)


def cab(n, t):
    print(f"\n=== Ejemplo {n} · {t} ===")


PD = 0.7
q = 1 - PD

cab(1, "Test del criterio 1: dos celdas, p0 = (0,5; 0,5), una mirada a A sin detección")
m = np.array([0.5, 0.5])
m = m*np.array([q, 1.0])
print(f"m = p0 * q^n = {m}  ->  b = m / sum(m) = {m/m.sum()}  (b(A) = {m[0]/m.sum():.4f})")

cab(2, "Recompensa del belief-MDP: ejemplo de A15a (p0 = (0,6; 0,4); paso 0 un dron mira A, paso 1 dos miran B)")
m = np.array([0.6, 0.4])
ks = [np.array([1, 0]), np.array([0, 2])]
tot = 0.0
for t, k in enumerate(ks):
    r = np.sum(m*(1 - q**k))
    print(f"t = {t}: m = {m}, k = {k}, r_t = {r:.3f}")
    tot += r
    m = m*q**k
print(f"suma r_t = {tot:.3f} = 1 - sum(m_h) = {1 - m.sum():.3f}")
N = 10000
cel = rng.choice(2, size=N, p=[0.6, 0.4])
det0 = (cel == 0) & (rng.random(N) < PD)
det1 = (cel == 1) & (1 - (rng.random(N) > PD)*(rng.random(N) > PD)).astype(bool)
enc = det0 | det1
p = enc.mean()
hw = 1.96*np.sqrt(p*(1 - p)/N)
print(f"simulación con N = {N}: p_hat = {p:.4f}, IC 95 % = [{p-hw:.4f}, {p+hw:.4f}] "
      f"(semianchura {hw:.4f}); ¿contiene 0,784? {p-hw <= 0.784 <= p+hw}")

cab(3, "Semianchura del intervalo al 95 %: 1,96 sqrt(p(1-p)/N)")
for pp in (0.78, 0.9):
    for NN in (1000, 10000):
        print(f"  p = {pp}, N = {NN}: ± {1.96*np.sqrt(pp*(1-pp)/NN):.4f}")

cab(4, "Umbrales de los criterios frente a lo medido en los temas")
print(f"criterio 2: 5 % sobre 31,9 pasos = {31.9*1.05:.1f}; A7b mide 32,3 ({100*(32.3/31.9-1):.1f} % más)")
print(f"criterio 4: 2 de 64 = {100*2/64:.1f} % (< 5 %)")
print(f"criterio 5: 2 ms / 20 ms (50 Hz) = {2/20*100:.0f} %")
print(f"criterio 8: 10,0 / 0,021 = {10.0/0.021:.0f} veces (umbral 100)")
print(f"criterio 9: 0,910 - 0,879 = {100*(0.910-0.879):.1f} puntos con mensajes de hace 10 pasos (umbral 5)")
print(f"error a 0,5 s: 25 pasos a 50 Hz = {25/50:.1f} s")
