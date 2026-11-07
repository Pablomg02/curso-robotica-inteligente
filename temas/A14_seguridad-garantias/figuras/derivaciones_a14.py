"""Ejemplos numéricos de las derivaciones del tema A14 (CPU, instantáneo).

Uso:  uv run --with numpy python derivaciones_a14.py > salida_derivaciones.txt

1. Con hdot >= -alfa h, h(t) >= h(0) exp(-alfa t): nunca llega a cero.
2. Solución cerrada del CBF-QP con una sola restricción (sin límites de entrada).
3. La condición de orden 2 para un dron (doble integrador) frente a un poste: qué aceleración exige.
4. Cuánto puede entrar un dron en un obstáculo en un paso si la condición solo se impone en los instantes de muestreo.
"""
import math
import numpy as np

print("1. COTA EXPONENCIAL")
for alfa in (0.5, 2.0):
    print(f"  alfa = {alfa}, h(0) = 1: h(1 s) >= {math.exp(-alfa):.3f}, h(3 s) >= {math.exp(-3 * alfa):.4f}  (positivo siempre)")

print("2. CBF-QP CON UNA RESTRICCIÓN")
# restricción a + b·u >= 0, con a = Lf h + alfa h y b = Lg h
u_nom = np.array([3.0, 0.0])
b = np.array([-2.0, -1.0])
a = 2.0
viol = a + b @ u_nom
lam = max(0.0, -viol) / (b @ b)
u = u_nom + lam * b
print(f"  u_nom = {u_nom.tolist()}, a = {a}, b = {b.tolist()}: a + b·u_nom = {viol:.2f} (< 0: inseguro)")
print(f"  lambda = max(0, -(a + b·u_nom)) / |b|^2 = {lam:.3f}; u = u_nom + lambda b = {np.round(u, 3).tolist()}; a + b·u = {a + b @ u:.3f}")

print("3. CONDICIÓN DE ORDEN 2 PARA UN DRON")
c, rs = np.array([10.0, 0.0]), 1.3            # poste en (10, 0) con radio de seguridad 1,3 m
p, v = np.array([4.0, 0.0]), np.array([3.0, 0.0])
d = p - c
h = d @ d - rs ** 2
hd = 2 * d @ v
for alfa in (1.0, 2.0):
    # 2 d·u + 2|v|^2 + 2 alfa hd + alfa^2 h >= 0  ->  con u = (ux, 0): ux <= (...)/(-2 dx) si dx < 0
    rhs = 2 * v @ v + 2 * alfa * hd + alfa ** 2 * h
    ux_max = rhs / (-2 * d[0])
    print(f"  p = {p.tolist()}, v = {v.tolist()}, poste {c.tolist()} r = {rs}: h = {h:.2f}, hdot = {hd:.2f}; alfa = {alfa}: "
          f"2|v|^2 + 2 alfa hdot + alfa^2 h = {rhs:.2f}, así que u_x <= {ux_max:.2f} m/s2")

print("4. DISCRETIZACIÓN")
for v_, dt in ((6.0, 0.01), (6.0, 0.05)):
    print(f"  a {v_} m/s con paso de {dt * 1e3:.0f} ms, en un paso se recorren {v_ * dt * 100:.0f} cm")
