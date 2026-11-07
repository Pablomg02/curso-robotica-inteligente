"""Ejemplos numéricos de las derivaciones del tema A13 (CPU, instantáneo).

Uso:  uv run --with numpy python derivaciones_a13.py > salida_derivaciones.txt

1. Un retardo puro tau añade una fase de -omega*tau: cuánto margen de fase se come a la frecuencia de cruce.
2. La aproximación de Padé de primer orden y la cota de ancho de banda que sugiere (omega_c < 0,6 z con z = 2/tau).
3. El ensamblado temporal de ACT: pesos w_i = exp(-m i) para las predicciones de una misma acción.
4. Lo que deriva un dron que mantiene un mando de aceleración durante la latencia del planificador.
"""
import math
import numpy as np

print("1. FASE DE UN RETARDO")
for fc, tau in ((2.0, 0.02), (2.0, 0.1), (2.0, 0.25), (0.5, 0.25), (0.5, 0.8)):
    wc = 2 * math.pi * fc
    print(f"  cruce a {fc} Hz (omega_c = {wc:.2f} rad/s), retardo {tau * 1e3:.0f} ms: fase perdida omega_c*tau = {wc * tau:.3f} rad = {math.degrees(wc * tau):.1f} grados")
print("2. PADÉ Y COTA DE ANCHO DE BANDA")
for tau in (0.02, 0.1, 0.25, 0.4, 0.8):
    z = 2 / tau
    wmax = 0.6 * z
    print(f"  tau = {tau * 1e3:.0f} ms: cero equivalente z = 2/tau = {z:.1f} rad/s; cruce máximo 0,6 z = {wmax:.2f} rad/s = {wmax / (2 * math.pi):.2f} Hz")
w = np.array([1.0, 5.0, 10.0])
for tau in (0.1,):
    exacta = -w * tau
    pade = -2 * np.arctan(w * tau / 2)
    print(f"  fase con tau = 100 ms a {w.tolist()} rad/s: exacta {np.round(np.degrees(exacta), 1).tolist()} grados, Padé {np.round(np.degrees(pade), 1).tolist()} grados")
print("3. ENSAMBLADO TEMPORAL DE ACT")
for m in (0.01, 0.1, 0.5):
    wi = np.exp(-m * np.arange(4))
    print(f"  m = {m}: pesos de 4 predicciones (de la más antigua a la más nueva) w_i = exp(-m i) = {np.round(wi, 3).tolist()}; "
          f"normalizados {np.round(wi / wi.sum(), 3).tolist()}")
preds = np.array([0.50, 0.44, 0.40, 0.38])
wi = np.exp(-0.1 * np.arange(4))
print(f"  ejemplo: predicciones {preds.tolist()} con m = 0,1 -> acción = {np.sum(wi * preds) / wi.sum():.3f} (media simple {preds.mean():.3f})")
print("4. MANTENER UN MANDO DURANTE LA LATENCIA")
for a, d in ((6.0, 0.1), (6.0, 0.2), (6.0, 0.4), (6.0, 0.8)):
    print(f"  aceleración {a} m/s2 mantenida {d * 1e3:.0f} ms de más: error de velocidad a*delta = {a * d:.2f} m/s; de posición a*delta^2/2 = {a * d * d / 2:.2f} m")
v, d = 1.5, 0.8
print(f"  seguir un plan desfasado delta = {d} s con la persona a {v} m/s: retraso de unos v*delta = {v * d:.1f} m")
