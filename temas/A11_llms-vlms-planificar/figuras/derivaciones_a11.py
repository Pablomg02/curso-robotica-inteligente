"""Ejemplos numéricos de las derivaciones del tema A11 (CPU, segundos).

Uso:  uv run --with numpy python derivaciones_a11.py > salida_derivaciones.txt

1. Por qué se divide por raíz de d en la atención: varianza del producto escalar y peso máximo del softmax.
2. La cota del ancho de banda al decodificar con los datos de la RTX 4060 Ti y de Gemma 4 de 12B en 4 bits.
3. Un ejemplo ilustrativo (inventado) de la regla de SayCan: utilidad según el LLM por factibilidad según el robot.
4. Predicción conforme (KnowNo) con un "LLM" de juguete: cobertura y tamaño de los conjuntos.
"""
import math
import numpy as np

rng = np.random.default_rng(0)
print("1. ATENCIÓN")
for d in (16, 64, 256):
    q = rng.normal(size=(20000, d))
    k = rng.normal(size=(20000, 16, d))
    s = np.einsum("nd,nkd->nk", q, k)
    def wmax(x):
        x = x - x.max(1, keepdims=True)
        w = np.exp(x)
        return (w / w.sum(1, keepdims=True)).max(1).mean()
    print(f"  d = {d}: desviación de q·k = {s.std():.2f} (raíz de d = {math.sqrt(d):.2f}); con 16 claves, peso máximo medio del "
          f"softmax sin escalar = {wmax(s):.3f}, dividiendo por raíz de d = {wmax(s / math.sqrt(d)):.3f}")

print("2. ANCHO DE BANDA")
bus, gbps, M = 128, 18.0, 6.98
B = bus / 8 * gbps
print(f"  bus de {bus} bits a {gbps:.0f} Gbps: B = {bus} / 8 x {gbps:.0f} = {B:.0f} GB/s; pesos M = {M} GB; cota B / M = {B / M:.1f} tokens/s")

print("3. SAYCAN (ejemplo ilustrativo)")
hab = ["coger la esponja", "ir a la encimera", "coger la lata"]
p_llm = np.array([0.6, 0.3, 0.1])
afor = np.array([0.05, 0.9, 0.8])
prod = p_llm * afor
for h, a, b, c in zip(hab, p_llm, afor, prod):
    print(f"  {h}: LLM {a:.2f} x afordancia {b:.2f} = {c:.3f}")
print(f"  solo el LLM elige: {hab[int(np.argmax(p_llm))]}; SayCan elige: {hab[int(np.argmax(prod))]}")

print("4. PREDICCIÓN CONFORME (LLM de juguete con 4 opciones)")
def casos(n):
    """Probabilidades del LLM: la opción correcta (la 0) tiene más peso de media, pero no siempre."""
    z = rng.normal(size=(n, 4)) * 1.2
    z[:, 0] += 1.5
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return p
eps, n_cal = 0.15, 400
pc = casos(n_cal)
sc = 1 - pc[:, 0]                                   # puntuación de no conformidad de la opción correcta
nivel = math.ceil((n_cal + 1) * (1 - eps)) / n_cal
qh = np.quantile(sc, nivel, method="higher")
pt = casos(20000)
conj = (1 - pt) <= qh
cob = conj[:, 0].mean()
tam = conj.sum(1)
print(f"  epsilon = {eps}; {n_cal} casos de calibración; nivel del cuantil = ceil(({n_cal}+1)(1-{eps}))/{n_cal} = {nivel:.4f}; q = {qh:.3f}")
print(f"  20000 casos nuevos: cobertura {cob:.3f} (garantía: al menos {1 - eps:.2f}); tamaño medio del conjunto {tam.mean():.2f}; "
      f"una sola opción (no hace falta preguntar) {100 * np.mean(tam == 1):.0f} %; conjunto vacío {100 * np.mean(tam == 0):.1f} %")
top1 = (pt.argmax(1) == 0).mean()
print(f"  comparación: elegir siempre la opción más probable acierta el {100 * top1:.0f} %")
