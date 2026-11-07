"""Ejemplos numéricos de las hojas del tema A15a (desarrollo de la matemática).

Uso:  python3 ejemplos_a15a.py > salida_ejemplos.txt      (numpy, CPU, unos segundos)

Cada ejemplo lleva el número con el que se cita en las hojas («ejemplo N del script»).
Las cifras del experimento de búsqueda (PPO en GPU) NO se recalculan aquí: las escribió figuras_a15a.py en
salida_gpu.txt. Lo de aquí son cuentas pequeñas que se siguen a mano y modelos de juguete construidos para
explicar un mecanismo; no reproducen ningún sistema real.
"""
import itertools
import math

import numpy as np

np.set_printoptions(precision=4, suppress=True)
Q_FALLO = 0.3                       # probabilidad de no detectar al objetivo presente (detección 0,7), como en el tema
PAGO = np.array([[8.0, -12.0, -12.0], [-12.0, 0.0, 0.0], [-12.0, 0.0, 0.0]])   # juego matricial del tema


def titulo(n, texto):
    print("=" * 78)
    print(f"EJEMPLO {n}. {texto}")


# --------------------------------------------------------------------------------------------- 1
titulo(1, "Dos drones, dos celdas: cada uno acaba con su propio belief")
p0 = np.array([0.5, 0.5])
m = p0 * np.array([Q_FALLO, 1.0])       # el dron 1 mira A y no ve nada
b1 = m / m.sum()
print(f"  mapa previo (A, B) = {p0}; el dron 1 mira A y no detecta: masa sin normalizar {m}, suma {m.sum():.2f}")
print(f"  belief del dron 1: A = {b1[0]:.3f}, B = {b1[1]:.3f}  (0,15 / 0,65 y 0,5 / 0,65)")
print(f"  belief del dron 2 (no sabe nada de esa mirada): A = {p0[0]:.3f}, B = {p0[1]:.3f}")
print("  con comunicación instantánea, los dos tienen el del dron 1: un belief común")

# --------------------------------------------------------------------------------------------- 2
titulo(2, "Contar políticas deterministas: árboles de política")
for h in range(1, 7):
    nodos = (2 ** h - 1) // (2 - 1)
    print(f"  |A| = |Omega| = 2, n = 2, h = {h}: nodos = {nodos:2d}; por agente 2^{nodos} = 10^{nodos * math.log10(2):.1f}; "
          f"conjuntas 2^{2 * nodos} = 10^{2 * nodos * math.log10(2):.1f}")
# comprobación por fuerza bruta con h = 2: una acción en la raíz y una por cada observación posible
pols = list(itertools.product([0, 1], repeat=3))       # (acción raíz, acción si o = 0, acción si o = 1)
print(f"  h = 2 por enumeración: {len(pols)} políticas por agente, {len(pols) ** 2} conjuntas; primeras: {pols[:3]}")
print(f"  h = 3 a mano: 1 + 2 + 4 = 7 nodos; 2^7 = {2 ** 7} por agente; {2 ** 7} ^ 2 = {(2 ** 7) ** 2} conjuntas")
nod = 1 + 2 + 4
print(f"  búsqueda: 5 acciones, observación detecta/no (|Omega| = 2), h = 3, 3 drones: 5^{nod} = {5 ** nod} por dron; "
      f"conjuntas 5^{3 * nod} = {5 ** (3 * nod):.2e}")
for h in (2, 3, 4):
    print(f"    h = {h}: nodos {2 ** h - 1}; log10(conjuntas, 3 drones, 5 acciones) = {3 * (2 ** h - 1) * math.log10(5):.1f}")

# --------------------------------------------------------------------------------------------- 3
titulo(3, "No estacionariedad en el juego matricial: lo que vale una acción del agente 1 depende de pi_2")
for p in (1 / 3, 0.375, 0.8):
    pi2 = np.array([p, (1 - p) / 2, (1 - p) / 2])
    q1 = PAGO @ pi2
    print(f"  pi_2 = ({p:.3f}, {(1 - p) / 2:.3f}, {(1 - p) / 2:.3f}): Q_1(a_1) = sum_a2 pi_2(a_2) R(a_1, a_2) = {q1}; "
          f"mejor a_1 = {int(np.argmax(q1)) if abs(q1[0] - q1[1]) > 1e-9 else 'empate 0/1'}")
print("  Q_1(0) = 8p - 12(1 - p) = 20p - 12;  Q_1(1) = -12p;  iguales en 32p = 12, p = 0,375")
print("  el crítico conjunto Q(a_1, a_2) = R(a_1, a_2) no cambia con pi_2: es la tabla de pagos")

# --------------------------------------------------------------------------------------------- 4
titulo(4, "COMA en el juego matricial: línea base contrafactual y varianza del gradiente")
pi = np.array([0.5, 0.3, 0.2])
for u_otro in (0, 1):
    Qc = PAGO[:, u_otro]
    b = pi @ Qc
    print(f"  el otro agente juega {u_otro}: Q(u', {u_otro}) = {Qc}; b = 0,5·{Qc[0]:g} + 0,3·{Qc[1]:g} + 0,2·{Qc[2]:g} = {b:.2f}; "
          f"ventajas A = Q - b = {Qc - b}")
    # gradiente respecto a los logits de una softmax: grad log pi(u) = e_u - pi
    G = np.eye(3) - pi[None]
    media_b = sum(pi[u] * G[u] * b for u in range(3))
    g_sin = np.array([G[u] * Qc[u] for u in range(3)])
    g_con = np.array([G[u] * (Qc[u] - b) for u in range(3)])
    m_sin, m_con = pi @ g_sin, pi @ g_con
    v_sin = pi @ np.sum((g_sin - m_sin) ** 2, 1)
    v_con = pi @ np.sum((g_con - m_con) ** 2, 1)
    print(f"    E[grad log pi · b] = {media_b}  (cero);  gradiente medio sin línea base {m_sin}, con ella {m_con}")
    print(f"    varianza total del estimador de una muestra: sin línea base {v_sin:.2f}, con ella {v_con:.2f}")

# --------------------------------------------------------------------------------------------- 5
titulo(5, "Cuántas acciones conjuntas hay que comparar para elegir la mejor")
for n in (2, 3, 10, 32):
    print(f"  5 acciones, n = {n:2d}: conjuntas 5^n = {5 ** n:.3g}; con IGM basta n·5 = {5 * n} evaluaciones")

# --------------------------------------------------------------------------------------------- 6
titulo(6, "IGM con la suma (VDN) y con una mezcla monótona no lineal, f(x, y) = ln(e^x + e^y)")
Q1, Q2 = np.array([1.0, 3.0]), np.array([2.0, -1.0])
f = lambda x, y: np.log(np.exp(x) + np.exp(y))
T6 = np.array([[f(Q1[i], Q2[j]) for j in range(2)] for i in range(2)])
print(f"  Q_1 = {Q1}, Q_2 = {Q2}: argmax individuales (1, 0)")
S6 = Q1[:, None] + Q2[None, :]
print(f"  VDN, Q_tot = Q_1 + Q_2 = {S6.tolist()}; máximo {S6.max():g} en "
      f"{tuple(int(v) for v in np.unravel_index(np.argmax(S6), S6.shape))} = max Q_1 + max Q_2 = {Q1.max():g} + {Q2.max():g}")
print(f"  Q_tot = f(Q_1(a_1), Q_2(a_2)) = {T6.round(3).tolist()}; máximo en {tuple(int(v) for v in np.unravel_index(np.argmax(T6), T6.shape))}")
print(f"  en cada columna, la fila 1 gana a la 0: {T6[1] > T6[0]} (el orden de las filas no depende de a_2)")

# --------------------------------------------------------------------------------------------- 7
titulo(7, "VDN a mano en el juego matricial: mínimos cuadrados con datos uniformes")
fila, col, tot = PAGO.mean(1), PAGO.mean(0), PAGO.mean()
print(f"  medias de fila {fila.round(3)}, de columna {col.round(3)}, total {tot:.3f}  (-16/3, -4; -40/9)")
Qv = fila[:, None] + col[None] - tot
print(f"  ajuste aditivo Q_tot = fila + columna - total = {Qv.round(2).tolist()}")
print(f"  (figuras_a15a.py, entrenado con Adam: [[-6.2, -4.9, -4.9], [-4.9, -3.6, -3.6], [-4.9, -3.6, -3.6]])")
print(f"  error cuadrático medio de VDN: {np.mean((Qv - PAGO) ** 2):.2f}  (el script: 50,57)")
mez = np.array([[-8.0, -8, -8], [-8, 0, 0], [-8, 0, 0]])
print(f"  error de la mezcla monótona aprendida [[-8,-8,-8],[-8,0,0],[-8,0,0]]: {np.mean((mez - PAGO) ** 2):.2f}  (el script: 35,56)")
print(f"  a mano: (8 + 8)^2 = 256 en (0,0) y 4^2 = 16 en las cuatro casillas de -12: (256 + 64) / 9 = {320 / 9:.2f}")
print("  q_1 = fila - c: la acción 0 vale -5,33 + c' y las otras -4 + c': 1 y 2 empatan; el entrenamiento eligió (1, 1)")
# comprobación: el gradiente de la pérdida en la solución es cero
g1 = np.array([np.sum(Qv[i] - PAGO[i]) for i in range(3)])
print(f"  ecuaciones normales: sum_a2 (Q_tot - R) por fila = {g1.round(10)}")

# --------------------------------------------------------------------------------------------- 8
titulo(8, "MAPPO: el objetivo recortado con parámetros compartidos, tres drones en un paso")
eps = 0.2
rt = np.array([1.3, 0.9, 1.1])
A = np.array([1.0, -2.0, 0.5])
t1 = rt * A
t2 = np.clip(rt, 1 - eps, 1 + eps) * A
mn = np.minimum(t1, t2)
for i in range(3):
    print(f"  dron {i + 1}: r = {rt[i]}, A = {A[i]:+.1f}: r·A = {t1[i]:+.2f}, clip(r)·A = {t2[i]:+.2f}, min = {mn[i]:+.2f}")
print(f"  media sobre los tres drones: {mn.mean():+.4f}; un único theta recibe el gradiente de los tres términos")

# --------------------------------------------------------------------------------------------- 9
titulo(9, "Mensajes con retraso: lo que cree un dron de una celda que otro ya miró")
p_c = 0.10
print(f"  celda c con p_0(c) = {p_c}; el dron 2 la miró en t-3 y t-2 sin detectar")
print(f"  mensajes al instante: el dron 1 cuenta n = 2 miradas: m = {p_c} · 0,3^2 = {p_c * Q_FALLO ** 2:.3f}")
print(f"  mensajes de hace 10 pasos: aún no le han llegado: m = {p_c:.3f}, {p_c / (p_c * Q_FALLO ** 2):.1f} veces más")

# --------------------------------------------------------------------------------------------- 10
titulo(10, "La recompensa del belief-MDP a mano: dos celdas, dos pasos")
p0 = np.array([0.6, 0.4])
k = [np.array([1, 0]), np.array([0, 2])]     # paso 0: un dron mira A; paso 1: dos drones miran B
mt = p0.copy()
r = []
masas = []
for t in range(2):
    masas.append(mt.sum())
    rt_ = np.sum(mt * (1 - Q_FALLO ** k[t]))
    print(f"  paso {t}: m_t = {mt}, k_t = {k[t]}: r_t = sum m (1 - 0,3^k) = {rt_:.3f}", end="")
    mt = mt * Q_FALLO ** k[t]
    r.append(rt_)
    print(f"; m_t+1 = {mt.round(3)}")
print(f"  suma de r_t = {sum(r):.3f} = 1 - masa final {mt.sum():.3f};  pasos esperados = sum_t sum_c m_t(c) = "
      f"{' + '.join(f'{v:.2f}' for v in masas)} = {sum(masas):.2f}")
rng = np.random.default_rng(0)
NS = 200_000
celda = (rng.random(NS) < p0[1]).astype(int)        # 0 = A, 1 = B
det0 = (celda == 0) & (rng.random(NS) < 0.7)
det1 = (celda == 1) & (rng.random((NS, 2)) < 0.7).any(1)
hallado = det0 | det1
tau = np.where(det0, 1, np.where(det1, 2, 2))      # min(paso de detección, 2)
print(f"  Monte Carlo con {NS} episodios: P(encontrarlo) = {hallado.mean():.4f}; E[min(tau, 2)] = {tau.mean():.4f}")
print(f"  varianza del retorno realizado (0 o 1): {hallado.var():.4f}; del retorno del belief-MDP con estas acciones: 0")
