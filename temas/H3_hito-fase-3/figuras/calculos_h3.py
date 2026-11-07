"""Cifras del examen y de los criterios del hito 3. Uso: python3 calculos_h3.py > salida.txt (numpy, segundos)."""
import math
import numpy as np

print("Pregunta 3 (A7a): belief con detecciones negativas, dos celdas, P_D = 0,7, sin falsas alarmas")
pA, PD = 0.5, 0.7
for k in range(1, 6):
    pA = pA * (1 - PD) / (pA * (1 - PD) + (1 - pA))
    print(f"  tras {k} miradas a A sin detección: P(A) = {pA:.4f}")

print("Pregunta 8 (A9b): ACC y BWT con errores (positivo = olvido)")
R = np.array([[0.10, 0.80, 0.90], [0.30, 0.12, 0.85], [0.45, 0.20, 0.11]])
acc = R[-1].mean()
bwt = np.mean([R[-1, j] - R[j, j] for j in range(len(R) - 1)])
print(f"  R = {R.tolist()}\n  ACC = {acc:.4f}; BWT = {bwt:.4f}")

print("Pregunta 9 (A10): flow matching con el camino lineal")
x0, x1, t = np.array([0.5, -1.0]), np.array([1.0, 1.0]), 0.25
print(f"  x_t = {((1 - t) * x0 + t * x1).tolist()}; objetivo x1 - x0 = {(x1 - x0).tolist()}")

print("Pregunta 10 (A15a): políticas deterministas con |A| = 3, |Omega| = 2, h = 3, dos agentes")
nodos = (2 ** 3 - 1) // (2 - 1)
print(f"  nodos {nodos}; por agente 3^{nodos} = {3 ** nodos}; conjuntas {(3 ** nodos) ** 2}")

print("Pregunta 11 (A15b): consenso con tres drones")
for nombre, A in (("en línea 1-2-3", np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])), ("todos con todos", np.ones((3, 3)) - np.eye(3))):
    L = np.diag(A.sum(1)) - A
    ev = np.sort(np.linalg.eigvalsh(L))
    print(f"  {nombre}: autovalores del laplaciano {np.round(ev, 4).tolist()}; tiempo para dividir el desacuerdo por 100: {math.log(100) / ev[1]:.3f} s")

print("Pregunta 1 (A6a): término recortado de PPO con A = -1, r = 0,7, eps = 0,2")
r, A, eps = 0.7, -1.0, 0.2
print(f"  r A = {r * A:.2f}; clip(r) A = {min(max(r, 1 - eps), 1 + eps) * A:.2f}; min = {min(r * A, min(max(r, 1 - eps), 1 + eps) * A):.2f}")
