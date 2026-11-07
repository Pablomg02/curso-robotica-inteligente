"""Ejemplos numéricos hechos a mano y datos de las figuras nuevas de A11 (LLMs y VLMs para razonar y planificar).

Cada número de un ejemplo pequeño de la presentación sale de aquí. Las cifras del experimento con Gemma 4 de 12B
(aciertos, tiempos, tokens, latencia) salen de figuras_a11.py, que necesita la GPU y ollama y no hace falta relanzar:
su salida está en salida.txt, salida_diagnostico.txt y respuestas_a11.json. Los ejemplos de derivaciones_a11.py
(simulaciones de sqrt(d), SayCan y KnowNo) siguen en salida_derivaciones.txt.

Ejecutar:  python3 ejemplos_a11.py > salida_ejemplos.txt     (solo numpy, CPU, unos segundos)
"""
import json
import math
import os

import numpy as np

np.set_printoptions(precision=3, suppress=True)
AQUI = os.path.dirname(os.path.abspath(__file__))


def titulo(s):
    print("\n" + "=" * 76 + "\n" + s + "\n" + "=" * 76)


def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max())
    return e / e.sum()


# ===========================================================================
titulo("1. La atención a mano: tres tokens, d = 2")
Q = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
d = Q.shape[1]
S = Q @ K.T
print("  Q =\n", Q, "\n  K =\n", K, "\n  V =\n", V)
print("  Q K^T =\n", S)
Ss = S / math.sqrt(d)
print(f"  dividido por raíz de d = {math.sqrt(d):.3f}:\n", Ss)
mask = np.triu(np.ones((3, 3), bool), 1)
Sm = np.where(mask, -np.inf, Ss)
print("  con la máscara causal (-inf donde j > i):\n", Sm)
W = np.vstack([softmax(r) for r in Sm])
print("  pesos (softmax por filas):\n", W)
print("  salida = pesos V:\n", W @ V)
r3 = Ss[2]
print(f"  fila 3 paso a paso: exp(1,414) = {math.exp(r3[0]):.3f}, exp(0) = 1, exp(1,414) = {math.exp(r3[2]):.3f}; "
      f"suma = {np.exp(r3).sum():.3f}; pesos = {np.exp(r3) / np.exp(r3).sum()}")
print(f"  salida 3 = {W[2, 0]:.3f}·(1, 0) + {W[2, 1]:.3f}·(0, 1) + {W[2, 2]:.3f}·(0,5; 0,5) = {W[2] @ V}")

# ===========================================================================
titulo("2. Por qué sqrt(d): el softmax saturado y su gradiente")
for z in ([16.0, 0.0, -16.0], [1.0, 0.0, -1.0]):
    w = softmax(z)
    print(f"  z = {z}: pesos = {np.array2string(w, precision=7)}; derivada dw1/dz1 = w1 (1 - w1) = {w[0] * (1 - w[0]):.2e}")
rng = np.random.default_rng(3)
dd, nk = 64, 8
q = rng.normal(size=dd)
k = rng.normal(size=(nk, dd))
s_att = k @ q
print(f"  figura: d = {dd}, {nk} claves; q·k_j = {np.round(s_att, 1)}")
print(f"  pesos sin escalar        = {np.round(softmax(s_att), 3)}")
print(f"  pesos dividiendo por {math.sqrt(dd):.0f}  = {np.round(softmax(s_att / math.sqrt(dd)), 3)}")

# ===========================================================================
titulo("3. La cota r <= B/M con los números de la RTX 4060 Ti y Gemma 4 de 12B en 4 bits")
bus, gbps, M = 128, 18.0, 6.98
B = bus / 8 * gbps
t_min = M / B
r_med = 33.9
print(f"  B = {bus} bits / 8 x {gbps:.0f} Gbit/s = {B:.0f} GB/s;  M = {M} GB")
print(f"  tiempo mínimo por token = M / B = {M} / {B:.0f} = {t_min * 1e3:.1f} ms  ->  r <= {1 / t_min:.1f} tokens/s")
print(f"  medido: {r_med} tokens/s = {1e3 / r_med:.1f} ms por token = {100 * r_med * t_min:.0f} % de la cota")
print(f"  parámetros x 4 bits: 12e9 x 0,5 bytes = {12e9 * 0.5 / 1e9:.1f} GB (el fichero ocupa {M} GB)")
n_pre, t_pre, n_gen = 1218, 0.65, 146
print(f"  prellenado: {n_pre} tokens en {t_pre} s = {n_pre / t_pre:.0f} tokens/s; si lo limitara la memoria, una lectura "
      f"de los pesos, {t_min * 1e3:.1f} ms: tarda {t_pre / t_min:.0f} veces más -> lo limita el cálculo")
print(f"  generación de {n_gen} tokens a {r_med} tokens/s: {n_gen / r_med:.2f} s")
print(f"  intensidad aritmética: 2 operaciones por parámetro y token / 0,5 bytes por parámetro = {2 / 0.5:.0f} op/byte al "
      f"decodificar; en el prellenado de {n_pre} tokens, {2 * n_pre / 0.5:.0f} op/byte")
print(f"  pregunta del test: 400 GB/s / 8 GB = {400 / 8:.0f} tokens/s")
for nom, tok, t in (("con dos ejemplos", 132, 4.5), ("verificador que exige un plan", 369, 12.1),
                    ("razonamiento, temperatura 0", 6000, 184.4), ("razonamiento con muestreo", 10726, 331.1)):
    print(f"  {nom}: {tok} tokens / {r_med} tokens/s = {tok / r_med:.1f} s (medido {t} s)")
print(f"  razonamiento con muestreo frente a dos ejemplos: {10726 / 132:.0f} veces más tokens, {331.1 / 4.5:.0f} veces más tiempo")

# ===========================================================================
titulo("4. Elegir el siguiente token: temperatura, top-k y top-p")
toks = ["Wait", "Then", "So", "The", "{"]
z = np.array([2.0, 1.5, 0.5, 0.0, -1.0])
for T in (1.0, 0.5, 0.1):
    print(f"  T = {T}: softmax(z / T) = {np.round(softmax(z / T), 3)}")
print(f"  T -> 0: todo el peso en el máximo ({toks[int(np.argmax(z))]})")
p1 = softmax(z)
kk = 2
top = np.argsort(-p1)[:kk]
pk = np.zeros_like(p1)
pk[top] = p1[top] / p1[top].sum()
print(f"  top-k con k = {kk}: se quedan {[toks[i] for i in top]}, renormalizados {np.round(pk, 3)}")
orden = np.argsort(-p1)
acum = np.cumsum(p1[orden])
print(f"  probabilidades ordenadas {np.round(p1[orden], 3)}, acumuladas {np.round(acum, 3)}")
for P in (0.8, 0.95):
    ncut = int(np.searchsorted(acum, P) + 1)
    keep = orden[:ncut]
    pp = np.zeros_like(p1)
    pp[keep] = p1[keep] / p1[keep].sum()
    print(f"  top-p con p = {P}: el conjunto más pequeño cuya suma llega a {P}: {[toks[i] for i in keep]}, renormalizados {np.round(pp, 3)}")

# ===========================================================================
titulo("5. RL con recompensa verificable: la ventaja relativa al grupo (GRPO)")
for r in ([1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]):
    r = np.array(r, float)
    sd = r.std()
    A = (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)
    print(f"  recompensas {r.astype(int)}: media {r.mean():.2f}, desviación {sd:.3f} -> ventajas {np.round(A, 2)}")

# ===========================================================================
titulo("6. Una imagen en tokens (cifras de Gemma 3)")
px, nt = 896, 256
lado = int(math.isqrt(nt))
print(f"  {nt} tokens = rejilla de {lado} x {lado}; cada token resume {px / lado:.0f} x {px / lado:.0f} píxeles de media")
print(f"  un mapa de 100 m a {px} píxeles: {100 / lado:.2f} m por token; el margen del verificador es de 2 m")

# ===========================================================================
titulo("7. Un problema de planificación clásica, a mano: dos bloques")


def aplicar(estado, acc):
    nombre, pre, add, dele = acc
    falta = pre - estado
    if falta:
        return None, falta
    return (estado - dele) | add, set()


def accion(tipo, x, y=None):
    if tipo == "desapilar":   # quitar x de encima de y
        return (f"desapilar({x},{y})", {f"sobre({x},{y})", f"libre({x})", "mano_vacia"},
                {f"en_mano({x})", f"libre({y})"}, {f"sobre({x},{y})", f"libre({x})", "mano_vacia"})
    if tipo == "dejar":       # dejar x en la mesa
        return (f"dejar({x})", {f"en_mano({x})"}, {f"en_mesa({x})", f"libre({x})", "mano_vacia"}, {f"en_mano({x})"})
    if tipo == "coger":       # coger x de la mesa
        return (f"coger({x})", {f"en_mesa({x})", f"libre({x})", "mano_vacia"}, {f"en_mano({x})"},
                {f"en_mesa({x})", f"libre({x})", "mano_vacia"})
    if tipo == "apilar":      # poner x encima de y
        return (f"apilar({x},{y})", {f"en_mano({x})", f"libre({y})"}, {f"sobre({x},{y})", f"libre({x})", "mano_vacia"},
                {f"en_mano({x})", f"libre({y})"})


s0 = {"sobre(A,B)", "en_mesa(B)", "libre(A)", "mano_vacia"}
meta = {"sobre(B,A)"}
plan = [accion("desapilar", "A", "B"), accion("dejar", "A"), accion("coger", "B"), accion("apilar", "B", "A")]
s = set(s0)
print(f"  s0 = {sorted(s0)}; meta = {sorted(meta)}")
for a in plan:
    print(f"  {a[0]}: pre {sorted(a[1])}; añade {sorted(a[2])}; borra {sorted(a[3])}")
    s, falta = aplicar(s, a)
    print(f"     -> {sorted(s)}")
print(f"  ¿meta en el estado final? {meta <= s}")
s, falta = aplicar(set(s0), accion("coger", "B"))
print(f"  plan erróneo, empezar por coger(B): precondiciones que faltan {sorted(falta)}")

# ===========================================================================
titulo("8. LLM-Modulo con las cifras del experimento: tiempo del bucle y tabla de aceptados")
print(f"  verificador que exige un plan: 2,47 intentos de media x 4,5 s por intento = {2.47 * 4.5:.1f} s (medido 12,1 s)")
print("  sin verificador (dos ejemplos): 25 posibles -> 8 correctos, 17 con un cruce que se volarían; 5 imposibles -> 5 planes que se volarían")
print("  con el verificador que exige un plan: 16 correctos aceptados; 9 sin plan válido tras 4 intentos (no se vuelan); "
      "5 imposibles rechazadas; planes incorrectos aceptados: 0, porque el verificador define qué es correcto")

# ===========================================================================
titulo("9. SayCan a mano (ilustrativo, inventado)")
hab = ["coger la esponja", "ir a la encimera", "coger la lata", "hecho"]
p_llm = np.array([0.55, 0.30, 0.10, 0.05])
afor = np.array([0.05, 0.90, 0.80, 1.00])
prod = p_llm * afor
for h_, a_, b_, c_ in zip(hab, p_llm, afor, prod):
    print(f"  paso 1, {h_}: {a_:.2f} x {b_:.2f} = {c_:.3f}")
print(f"  paso 1: solo el LLM -> {hab[int(np.argmax(p_llm))]}; SayCan -> {hab[int(np.argmax(prod))]}")
p_llm2 = np.array([0.70, 0.05, 0.10, 0.15])
afor2 = np.array([0.85, 0.95, 0.80, 1.00])
prod2 = p_llm2 * afor2
for h_, a_, b_, c_ in zip(hab, p_llm2, afor2, prod2):
    print(f"  paso 2 (ya en la encimera, la esponja a la vista), {h_}: {a_:.2f} x {b_:.2f} = {c_:.3f}")
print(f"  paso 2: SayCan -> {hab[int(np.argmax(prod2))]}")
pt = [0.60, 0.90, 0.95]
print(f"  puntuar un texto de 3 tokens: {pt[0]} x {pt[1]} x {pt[2]} = {np.prod(pt):.3f}")

# ===========================================================================
titulo("10. Predicción conforme a mano: 9 casos de calibración")
p_cal = np.array([0.90, 0.80, 0.75, 0.70, 0.60, 0.55, 0.40, 0.30, 0.15])
s_cal = 1 - p_cal
n, eps = len(s_cal), 0.2
kq = math.ceil((n + 1) * (1 - eps))
qh = np.sort(s_cal)[kq - 1]
print(f"  p del LLM a la opción correcta: {p_cal}")
print(f"  sorpresas s_j = 1 - p: {np.round(s_cal, 2)}; ordenadas: {np.round(np.sort(s_cal), 2)}")
print(f"  n = {n}, eps = {eps}: k = ceil((n+1)(1-eps)) = ceil({(n + 1) * (1 - eps):.1f}) = {kq}; q = la {kq}.ª menor = {qh:.2f}")
print(f"  regla: una opción entra si 1 - p <= {qh:.2f}, es decir, si p >= {1 - qh:.2f}")
print(f"  garantía: k / (n+1) = {kq}/{n + 1} = {kq / (n + 1):.2f} >= 1 - eps = {1 - eps:.2f}")
for nom, p in (("caso nuevo 1", [0.45, 0.35, 0.15, 0.05]), ("caso nuevo 2", [0.80, 0.10, 0.06, 0.04])):
    C = [l for l, v in zip("ABCD", p) if 1 - v <= qh + 1e-12]
    print(f"  {nom}: p = {p} -> conjunto {C} -> {'actúa' if len(C) == 1 else 'pregunta'}")
for n_ in (9, 19, 99, 400):
    k_ = math.ceil((n_ + 1) * (1 - eps))
    print(f"  n = {n_}: k = {k_}, k/(n+1) = {k_ / (n_ + 1):.3f}")
print("  -- el mismo 'LLM' de juguete de derivaciones_a11.py, recalibrando 500 veces con 400 casos (eps = 0,15):")
rng2 = np.random.default_rng(1)


def casos(nn):
    zz = rng2.normal(size=(nn, 4)) * 1.2
    zz[:, 0] += 1.5
    return np.exp(zz) / np.exp(zz).sum(1, keepdims=True)


pt_test = casos(200000)
covs = []
for _ in range(500):
    pc = casos(400)
    sc = np.sort(1 - pc[:, 0])
    q_ = sc[math.ceil(401 * 0.85) - 1]
    covs.append(((1 - pt_test[:, 0]) <= q_).mean())
covs = np.array(covs)
print(f"  cobertura media {covs.mean():.3f}; percentiles 5 y 95: {np.percentile(covs, 5):.3f} y {np.percentile(covs, 95):.3f}; "
      f"por debajo de 0,85 en el {100 * (covs < 0.85).mean():.0f} % de las calibraciones")
bordes = np.arange(0.79, 0.911, 0.01)
cuenta, _ = np.histogram(covs, bins=bordes)
print("  histograma de la cobertura (figura): " + "; ".join(f"[{a:.2f}, {b:.2f}) {c}" for a, b, c in zip(bordes[:-1], bordes[1:], cuenta)))
print(f"  fuera del histograma: {int((covs < bordes[0]).sum())} por debajo, {int((covs >= bordes[-1]).sum())} por encima")

# ===========================================================================
titulo("11. El verificador del experimento con la misión 1")
src = open(os.path.join(AQUI, "figuras_a11.py"), encoding="utf-8").read()
src = src[src.index("NOMBRES = "):src.index("# ---------------------------------------------------------------------- textos")]
ns = {"math": math, "itertools": __import__("itertools"), "random": __import__("random"), "json": json}
exec(src, ns)
m = ns["generar"](1)
proh = [m["zonas"][i] for i in m["prohibidas"]]
print(f"  autonomía {m['autonomia']} m; máximo con 20 % de reserva {0.8 * m['autonomia']:.0f} m; energía óptima {m['e_opt']:.1f} m")
for z_ in m["zonas"]:
    r = z_["rect"]
    print(f"  {z_['id']} {z_['nombre']} {r}: centro {ns['centro'](r)}, área {(r[2] - r[0]) * (r[3] - r[1])} m², barrido área/5 = {ns['coste_busqueda'](r):.0f} m")
p, qq = (0.0, 0.0), (46.5, 35.0)
for z_ in proh:
    x0, y0, x1, y1 = z_["rect"]
    x0, y0, x1, y1 = x0 - 2, y0 - 2, x1 + 2, y1 + 2
    tx = (x0 / qq[0], x1 / qq[0])
    ty = (y0 / qq[1], y1 / qq[1])
    t0, t1 = max(tx[0], ty[0], 0), min(tx[1], ty[1], 1)
    print(f"  tramo (0,0)->(46,5; 35) frente a {z_['id']} inflada 2 m [{x0},{x1}]x[{y0},{y1}]: t en x {np.round(tx, 3)}, t en y {np.round(ty, 3)}"
          f" -> intervalo [{t0:.3f}, {t1:.3f}] -> {'cruza' if t1 > t0 else 'no cruza'} (función del script: {ns['cruza'](p, qq, z_['rect'], 2.0)})")
resp = json.load(open(os.path.join(AQUI, "respuestas_a11.json"), encoding="utf-8"))


def plan_de(cond):
    for f in resp[cond]:
        if f["sem"] == 1:
            return f["resp"]


for nom, pj in (("con dos ejemplos", plan_de("con dos ejemplos")), ("con verificador", plan_de("verificador que exige un plan")),
                ("óptimo", json.dumps({"plan": m["plan_opt"]}))):
    tipo, err, e = ns["verificar"](m, pj)
    print(f"  {nom}: {tipo}; energía {e:.1f} m (/ óptima {e / m['e_opt']:.3f}); errores {err}")
    pts = [(0.0, 0.0)]
    tram = []
    for a in json.loads(pj)["plan"]:
        if a["accion"] == "ir_a":
            pts.append((a["x"], a["y"]))
        elif a["accion"] == "buscar":
            pts.append(ns["centro"]([z_ for z_ in m["zonas"] if z_["id"] == a["zona"]][0]["rect"]))
        elif a["accion"] == "volver_base":
            pts.append((0.0, 0.0))
    tram = [math.dist(a_, b_) for a_, b_ in zip(pts[:-1], pts[1:])]
    print(f"     tramos {np.round(tram, 1)} suman {sum(tram):.1f} m")

# ===========================================================================
titulo("12. Coordenadas de las figuras nuevas (SVG)")
# a) pesos del softmax (ejemplo 2): barras con 90 px = peso 1
w_sin = softmax(s_att)
w_con = softmax(s_att / math.sqrt(dd))
print("  a) barras, x = 70 + 52 j; alto sin escalar:", np.round(90 * w_sin, 1), "; alto escalado:", np.round(90 * w_con, 1))
# b) línea temporal del prellenado y la decodificación, 220 px por segundo desde x = 40
X = lambda t: 40 + 220 * t
t_tok = 1 / r_med
print(f"  b) prellenado de x = {X(0):.1f} a {X(t_pre):.1f}; decodificación de {X(t_pre):.1f} a {X(t_pre + n_gen * t_tok):.1f}; "
      f"un token = {220 * t_tok:.2f} px; marcas de 1 s en " + ", ".join(f"{X(k):.0f}" for k in range(6)))
print("     rayas de los tokens: M" + " M".join(f"{X(t_pre + i * t_tok):.1f},64 v42" for i in range(1, n_gen)))
# c) escalas de tiempo, eje logarítmico de 1e-4 s a 1e3 s
L = lambda t: 60 + 154.3 * (math.log10(t) + 4)
for nom, a, b in (("planificador exacto", 5e-4, None), ("actitud 200-800 Hz", 1 / 800, 1 / 200), ("trayectoria 10-50 Hz", 1 / 50, 1 / 10),
                  ("búsqueda 1-10 Hz", 1 / 10, 1.0), ("LLM, dos ejemplos", 4.5, None), ("LLM con verificador", 12.1, None),
                  ("LLM razonando con muestreo", 331.1, None)):
    print(f"  c) {nom}: x = {L(a):.1f}" + (f" a {L(b):.1f}" if b else ""))
print("     marcas: " + ", ".join(f"{10.0 ** e:g} s -> {L(10.0 ** e):.1f}" for e in range(-4, 4)))
print(f"     de 1,25 ms a 331,1 s: {math.log10(331.1 / 1.25e-3):.1f} órdenes de magnitud")
# d) predicción conforme: recta de 0 a 1 en x = 60 + 600 s
print("  d) sorpresas de calibración en x =", [round(float(60 + 600 * v), 1) for v in np.sort(s_cal)], f"; q = {60 + 600 * qh:.1f}")
# e) histograma de la cobertura: x = 60 + (c - 0,79) * 3500, 1 calibración = 1,2 px
print("  e) barras del histograma (x, alto):", [(round(float(60 + (a - 0.79) * 3500), 1), round(float(1.2 * c), 1)) for a, c in zip(bordes[:-1], cuenta)],
      f"; media en x = {60 + (covs.mean() - 0.79) * 3500:.1f}; 0,85 en x = {60 + 0.06 * 3500:.1f}")
# f) misión 1 cerca de Z5: 7 px por metro, origen en (40, 300)
P = lambda x, y: (round(40 + 7 * x, 1), round(300 - 7 * y, 1))
print("  f) Z5", P(12, 31), "a", P(30, 17), "; inflada", P(10, 33), "a", P(32, 15), "; Z4", P(37, 40), "a", P(56, 30),
      "; centro Z4", P(46.5, 35), "; entrada", P(0.4286 * 46.5, 0.4286 * 35), "; salida", P(0.688 * 46.5, 0.688 * 35),
      "; punto de paso del verificador", P(46.5, 15))
