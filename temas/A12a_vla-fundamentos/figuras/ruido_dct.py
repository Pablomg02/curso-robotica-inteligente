"""Cuentas de las derivaciones del tema A12a: error de cuantización de las casillas y ruido blanco en la DCT.

Uso:  uv run --with numpy --with scipy python ruido_dct.py > salida_ruido.txt
"""
import numpy as np
from scipy.fft import dct
from scipy.stats import norm

N = 256
delta = 2 / N
print(f"casillas: N = {N}, ancho = {delta:.7f}, desviación del error = ancho / raíz de 12 = {delta / np.sqrt(12):.5f}")
sigma = 0.02
for g in (2, 5, 10, 20, 50):
    u = 1 / (2 * g * sigma)
    print(f"DCT con escala {g}: umbral de redondeo 1/(2 escala) = {1 / (2 * g):.4f} = {u:.2f} sigma; "
          f"probabilidad de que un coeficiente de solo ruido no se anule = {100 * 2 * (1 - norm.cdf(u)):.1f} %")
# comprobación numérica: la DCT ortonormal conserva la desviación del ruido blanco
rng = np.random.default_rng(0)
n = rng.normal(0, sigma, (20000, 200))
c = dct(n, axis=1, norm="ortho")
print(f"ruido blanco de desviación {sigma}: desviación de los coeficientes de la DCT ortonormal = {c.std():.4f}; "
      f"fracción no nula con escala 10 = {100 * np.mean(np.round(10 * c) != 0):.1f} %, con escala 50 = {100 * np.mean(np.round(50 * c) != 0):.1f} %")
