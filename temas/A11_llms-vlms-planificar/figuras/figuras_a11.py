"""Datos de las figuras y tablas del tema A11 (LLMs y VLMs para razonar y planificar).

Uso:  uv run --with numpy --with matplotlib python -u figuras_a11.py > salida.txt
      Necesita un servidor de ollama en local (http://127.0.0.1:11434) con un modelo multimodal; por defecto
      gemma4:12b-it-qat (A11_MODELO para otro). A11_N fija el número de misiones (30 por defecto).
      A11_RESPUESTAS es el fichero donde se guardan las respuestas (respuestas_a11.json por defecto); se reescribe tras
      cada condición. Con A11_DE_RESPUESTAS=respuestas_a11.json, las condiciones que estén en ese fichero se rehacen desde
      las respuestas guardadas y solo las que falten se piden al modelo; con A11_SIN_MODELO=1 además no se llama nunca al
      modelo (ni se mide la latencia): sirve para comprobar la salida sin GPU.

Un modelo de lenguaje y visión local hace de planificador de misión para un dron. Cada misión es un mapa de 100 x 100 m
con seis zonas (una o dos prohibidas de sobrevolar), una base en (0, 0), una autonomía en metros y una orden en lenguaje
natural ("busca en el patio y en el almacén; no sobrevueles la pista; vuelve con al menos un 20 % de batería"). El modelo
devuelve un plan en JSON con cinco acciones (despegar, ir_a, buscar, volver_base, aterrizar) o dice que es imposible.
Un verificador externo comprueba el plan (formato, zonas, tramos que cruzan zonas prohibidas, energía) y un planificador
exacto (orden óptimo de las zonas y rutas por un grafo de visibilidad) da la referencia. Una de cada seis misiones es
imposible a propósito (la autonomía no basta).
Condiciones: sin ejemplos; con dos ejemplos; con dos ejemplos y el verificador devolviendo los errores (hasta 3 intentos
más, como en LLM-Modulo), en dos versiones: su mensaje recuerda que se puede responder que la misión es imposible, o solo
pide corregir el plan; el mapa como imagen en lugar de coordenadas; con dos ejemplos y razonamiento ("thinking").
Dos de cada tres misiones exigen rodear una zona prohibida. Temperatura 0 y semilla fija: las respuestas son repetibles con
el mismo modelo y la misma máquina. Con razonamiento no se fuerza el formato JSON (forzarlo lo hacía entrar en bucle).
"""
import base64
import io
import itertools
import json
import math
import os
import random
import time
import urllib.request

import numpy as np

MODELO = os.environ.get("A11_MODELO", "gemma4:12b-it-qat")
N_MIS = int(os.environ.get("A11_N", "30"))
URL = "http://127.0.0.1:11434/api/generate"
NOMBRES = ["aparcamiento", "patio", "tejado norte", "almacén", "jardín", "muelle de carga", "pista", "huerto", "plaza", "invernadero"]
BARRIDO = 5.0          # ancho de barrido de la cámara (m): buscar en una zona cuesta área / BARRIDO metros de vuelo
MARGEN = 2.0           # distancia mínima a una zona prohibida (m)
RESERVA = 0.8          # hay que volver con al menos un 20 % de batería


# ---------------------------------------------------------------------- geometría y planificador exacto
def cruza(p, q, r, m=0.0):
    """¿El segmento p-q entra en el rectángulo r = (x0, y0, x1, y1) inflado m? (Liang-Barsky)."""
    x0, y0, x1, y1 = r[0] - m, r[1] - m, r[2] + m, r[3] + m
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return t1 - t0 > 1e-9


def ruta(p, q, prohibidas):
    """Camino más corto de p a q que no entra en las zonas prohibidas (grafo de visibilidad con esquinas infladas)."""
    nodos = [p, q] + [(x, y) for r in prohibidas for x in (r[0] - MARGEN - 0.5, r[2] + MARGEN + 0.5)
                      for y in (r[1] - MARGEN - 0.5, r[3] + MARGEN + 0.5)]
    n = len(nodos)
    libre = lambda a, b: not any(cruza(a, b, r, MARGEN) for r in prohibidas)
    dist = [math.inf] * n
    prev = [-1] * n
    dist[0] = 0.0
    hecho = [False] * n
    for _ in range(n):
        u = min((i for i in range(n) if not hecho[i]), key=lambda i: dist[i])
        if dist[u] == math.inf:
            break
        hecho[u] = True
        for v in range(n):
            if not hecho[v] and libre(nodos[u], nodos[v]):
                d = dist[u] + math.dist(nodos[u], nodos[v])
                if d < dist[v]:
                    dist[v], prev[v] = d, u
    camino, v = [], 1
    while v != -1:
        camino.append(nodos[v])
        v = prev[v]
    return dist[1], camino[::-1]


def centro(r):
    return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)


def coste_busqueda(r):
    return (r[2] - r[0]) * (r[3] - r[1]) / BARRIDO


def plan_optimo(m):
    """Orden óptimo de las zonas pedidas (todas las permutaciones) y rutas sin cruzar zonas prohibidas."""
    proh = [m["zonas"][i]["rect"] for i in m["prohibidas"]]
    mejor = (math.inf, None)
    for orden in itertools.permutations(m["pedidas"]):
        pos, total, pasos = (0.0, 0.0), 0.0, [{"accion": "despegar"}]
        for i in orden:
            c = centro(m["zonas"][i]["rect"])
            d, cam = ruta(pos, c, proh)
            for w in cam[1:-1]:
                pasos.append({"accion": "ir_a", "x": round(w[0], 1), "y": round(w[1], 1)})
            pasos.append({"accion": "buscar", "zona": m["zonas"][i]["id"]})
            total += d + coste_busqueda(m["zonas"][i]["rect"])
            pos = c
        d, cam = ruta(pos, (0.0, 0.0), proh)
        for w in cam[1:-1]:
            pasos.append({"accion": "ir_a", "x": round(w[0], 1), "y": round(w[1], 1)})
        pasos += [{"accion": "volver_base"}, {"accion": "aterrizar"}]
        total += d
        if total < mejor[0]:
            mejor = (total, pasos)
    return mejor


def ruta_directa_cruza(m, orden):
    pts = [(0.0, 0.0)] + [centro(m["zonas"][i]["rect"]) for i in orden] + [(0.0, 0.0)]
    return any(cruza(a, b, m["zonas"][i]["rect"], MARGEN) for a, b in zip(pts[:-1], pts[1:]) for i in m["prohibidas"])


def generar(semilla):
    """Dos de cada tres misiones exigen rodear una zona prohibida (la ruta recta en el orden óptimo la cruza)."""
    rng = random.Random(semilla)
    while True:
        m = _generar(rng, semilla)
        ids = [z["id"] for z in m["zonas"]]
        orden = [ids.index(p["zona"]) for p in m["plan_opt"] if p["accion"] == "buscar"]
        if (semilla % 3 == 0) or ruta_directa_cruza(m, orden):
            return m


def _generar(rng, semilla):
    zonas = []
    while len(zonas) < 6:
        w, h = rng.randint(10, 20), rng.randint(10, 20)
        x, y = rng.randint(8, 95 - w), rng.randint(8, 95 - h)
        r = (x, y, x + w, y + h)
        if all(r[0] > z[2] + 6 or r[2] < z[0] - 6 or r[1] > z[3] + 6 or r[3] < z[1] - 6 for z in (zz["rect"] for zz in zonas)):
            zonas.append({"rect": r})
    for i, (z, nom) in enumerate(zip(zonas, rng.sample(NOMBRES, 6))):
        z["id"], z["nombre"] = f"Z{i + 1}", nom
    idx = list(range(6))
    rng.shuffle(idx)
    n_proh = rng.choice([1, 2])
    n_ped = rng.choice([2, 3, 4]) if n_proh == 1 else rng.choice([2, 3])
    m = {"semilla": semilla, "zonas": zonas, "prohibidas": idx[:n_proh], "pedidas": idx[n_proh:n_proh + n_ped]}
    e_opt, pasos = plan_optimo(m)
    imposible = semilla % 6 == 5
    u = rng.uniform(1.15, 1.4) if imposible else rng.uniform(0.6, 0.92)
    m["autonomia"] = int(round(e_opt / (RESERVA * u) / 10) * 10)
    m["e_opt"], m["plan_opt"] = e_opt, pasos
    m["factible"] = e_opt <= RESERVA * m["autonomia"]
    return m


# ---------------------------------------------------------------------- verificador
ACCIONES = {"despegar", "ir_a", "buscar", "volver_base", "aterrizar"}


def verificar(m, resp):
    """Devuelve (tipo, errores, energía). tipo: 'ok', 'rechazo' (dice que es imposible) o 'fallo'."""
    try:
        d = json.loads(resp)
    except Exception:
        return "fallo", ["formato: la respuesta no es JSON válido"], None
    if isinstance(d, dict) and d.get("factible") is False:
        return "rechazo", [], None
    plan = d.get("plan") if isinstance(d, dict) else None
    if not isinstance(plan, list) or not plan:
        return "fallo", ["formato: falta la lista 'plan'"], None
    ids = {z["id"]: z for z in m["zonas"]}
    proh = {m["zonas"][i]["id"] for i in m["prohibidas"]}
    err, pos, energia, buscadas, volado = [], (0.0, 0.0), 0.0, [], False
    for k, p in enumerate(plan, 1):
        a = p.get("accion") if isinstance(p, dict) else None
        if a not in ACCIONES:
            err.append(f"formato: acción {k} desconocida ({a!r})")
            continue
        if a == "despegar":
            volado = True
            continue
        if not volado and a != "aterrizar":
            err.append("estructura: el plan no empieza por despegar")
            volado = True
        destino = None
        if a == "ir_a":
            try:
                destino = (float(p["x"]), float(p["y"]))
            except Exception:
                err.append(f"formato: la acción {k} (ir_a) necesita x e y numéricos")
                continue
        elif a == "buscar":
            z = ids.get(p.get("zona"))
            if z is None:
                err.append(f"formato: la acción {k} busca en una zona que no existe ({p.get('zona')!r})")
                continue
            if z["id"] in proh:
                err.append(f"zonas: la acción {k} busca en la zona prohibida {z['id']} ({z['nombre']})")
            destino = centro(z["rect"])
            buscadas.append(z["id"])
        elif a == "volver_base":
            destino = (0.0, 0.0)
        elif a == "aterrizar":
            if k != len(plan):
                err.append("estructura: aterrizar no es la última acción")
            if pos != (0.0, 0.0):
                err.append("estructura: aterriza fuera de la base")
            continue
        for i in m["prohibidas"]:
            z = m["zonas"][i]
            if a == "buscar" and z["id"] == p.get("zona"):
                continue
            if cruza(pos, destino, z["rect"], MARGEN):
                err.append(f"cruce: el tramo {k}, de ({pos[0]:.0f}, {pos[1]:.0f}) a ({destino[0]:.0f}, {destino[1]:.0f}), "
                           f"pasa por la zona prohibida {z['id']} ({z['nombre']}) o a menos de {MARGEN:.0f} m")
        energia += math.dist(pos, destino) + (coste_busqueda(ids[p["zona"]]["rect"]) if a == "buscar" else 0.0)
        pos = destino
    if plan and (not isinstance(plan[-1], dict) or plan[-1].get("accion") != "aterrizar"):
        err.append("estructura: el plan no termina aterrizando en la base")
    for i in m["pedidas"]:
        if m["zonas"][i]["id"] not in buscadas:
            err.append(f"zonas: no busca en {m['zonas'][i]['id']} ({m['zonas'][i]['nombre']})")
    if energia > RESERVA * m["autonomia"] + 1e-6:
        err.append(f"energía: el plan vuela {energia:.0f} m y el máximo con un 20 % de reserva es {RESERVA * m['autonomia']:.0f} m")
    return ("ok" if not err else "fallo"), err, energia


# ---------------------------------------------------------------------- textos
API = """Acciones disponibles (un objeto JSON por acción, en orden):
  {"accion": "despegar"}
  {"accion": "ir_a", "x": <m>, "y": <m>}        vuela en línea recta a ese punto
  {"accion": "buscar", "zona": "<id>"}           vuela en línea recta al centro de la zona y la barre entera
  {"accion": "volver_base"}                      vuela en línea recta a la base (0, 0)
  {"accion": "aterrizar"}                        solo en la base
Reglas: ningún tramo recto puede pasar por una zona prohibida ni a menos de 2 m de ella. Barrer una zona cuesta
(área / 5) metros de vuelo. La energía total (tramos + barridos) no puede superar el 80 % de la autonomía.
Responde solo con JSON: {"plan": [ ... ]} si la misión es posible, o {"factible": false, "motivo": "..."} si no lo es."""


def describir(m, coordenadas=True):
    lin = ["Mapa de 100 x 100 m (x hacia el este, y hacia el norte). Base del dron en (0, 0)."]
    for i, z in enumerate(m["zonas"]):
        r = z["rect"]
        tipo = " PROHIBIDO SOBREVOLAR" if i in m["prohibidas"] else ""
        if coordenadas:
            lin.append(f"- {z['id']} {z['nombre']}: rectángulo x de {r[0]} a {r[2]}, y de {r[1]} a {r[3]}{tipo}")
        else:
            lin.append(f"- {z['id']} {z['nombre']}{tipo} (coordenadas en la imagen)")
    lin.append(f"Autonomía del dron: {m['autonomia']} m de vuelo.")
    ped = [m["zonas"][i]["nombre"] for i in m["pedidas"]]
    nop = [m["zonas"][i]["nombre"] for i in m["prohibidas"]]
    lin.append(f"Misión: busca a una persona perdida en {', '.join(ped[:-1])} y {ped[-1]}. No sobrevueles {' ni '.join(nop)}. "
               f"Vuelve a la base con al menos un 20 % de batería.")
    return "\n".join(lin)


def imagen(m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    for i, z in enumerate(m["zonas"]):
        r = z["rect"]
        proh = i in m["prohibidas"]
        ax.add_patch(plt.Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], color="#d62728" if proh else "#1f77b4", alpha=0.35))
        ax.text(*centro(r), f"{z['id']}\n{z['nombre']}" + ("\nPROHIBIDA" if proh else ""), ha="center", va="center", fontsize=8)
    ax.plot(0, 0, "k^", ms=10)
    ax.text(2, 2, "base", fontsize=8)
    ax.set_xlim(0, 100), ax.set_ylim(0, 100), ax.set_xticks(range(0, 101, 10)), ax.set_yticks(range(0, 101, 10))
    ax.grid(alpha=0.4), ax.set_aspect("equal"), ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


EJEMPLOS = [generar(1000 + k) for k in range(6)]
EJEMPLOS = [e for e in EJEMPLOS if e["factible"]][:2]


def prompt(m, ejemplos, coordenadas=True):
    partes = ["Eres el planificador de misión de un dron de búsqueda.", API]
    for e in ejemplos:
        partes.append("EJEMPLO\n" + describir(e) + "\nRespuesta:\n" + json.dumps({"plan": e["plan_opt"]}, ensure_ascii=False))
    partes.append("MISIÓN\n" + describir(m, coordenadas) + "\nRespuesta:")
    return "\n\n".join(partes)


def extraer_json(s):
    """Con razonamiento no se fuerza el formato (forzarlo hacía que el razonamiento se colara en el JSON y entrara en bucle):
    se toma el bloque entre la primera llave y la última."""
    i, j = s.find("{"), s.rfind("}")
    return s[i:j + 1] if 0 <= i < j else s


def llamar(texto, think=False, images=None, muestreo=False, tope=6000):
    """Temperatura 0 por defecto. Con muestreo=True, el muestreo que trae el modelo en ollama (temperatura 1, top_k 64,
    top_p 0,95), con semilla fija."""
    opts = {"temperature": 1.0, "top_k": 64, "top_p": 0.95} if muestreo else {"temperature": 0}
    body = {"model": MODELO, "prompt": texto, "stream": False, "think": think,
            "options": dict(opts, seed=0, num_ctx=16384, num_predict=tope)}
    if not think:
        body["format"] = "json"
    if images:
        body["images"] = images
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    r["_t"] = time.perf_counter() - t0
    if think:
        r["response"] = extraer_json(r.get("response", ""))
    return r


# ---------------------------------------------------------------------- experimento
DE = os.environ.get("A11_DE_RESPUESTAS")
GUARDADAS = json.load(open(DE, encoding="utf-8")) if DE and os.path.exists(DE) else {}
SIN_MODELO = os.environ.get("A11_SIN_MODELO") == "1"
SALIDA_JSON = os.environ.get("A11_RESPUESTAS", "respuestas_a11.json")
print("modelo:", MODELO, f"(condiciones guardadas en {DE}: {', '.join(GUARDADAS)})" if GUARDADAS else "")
if not SIN_MODELO:
    llamar("Responde {\"ok\": true}")                  # carga el modelo antes de medir
MIS = [generar(s) for s in range(N_MIS)]
print(f"{N_MIS} misiones; imposibles: {sum(not m['factible'] for m in MIS)}; "
      f"zonas pedidas por misión: {np.mean([len(m['pedidas']) for m in MIS]):.1f}; "
      f"prohibidas: {np.mean([len(m['prohibidas']) for m in MIS]):.1f}")
t0 = time.perf_counter()
for m in MIS:
    plan_optimo(m)
print(f"planificador exacto: {(time.perf_counter() - t0) / N_MIS * 1e3:.1f} ms por misión (CPU, Python puro)")
for e in EJEMPLOS:
    assert verificar(e, json.dumps({"plan": e["plan_opt"]}))[0] == "ok"
assert all(verificar(m, json.dumps({"plan": m["plan_opt"]}))[0] == "ok" for m in MIS if m["factible"])
print("comprobación: el plan óptimo de cada misión posible pasa el verificador")

CONDS = [("sin ejemplos", dict(ej=0)), ("con dos ejemplos", dict(ej=2)), ("con ejemplos y verificador", dict(ej=2, bucle=3, rendirse=True)),
         ("verificador que exige un plan", dict(ej=2, bucle=3, rendirse=False)),
         ("mapa como imagen", dict(ej=2, img=True)), ("con ejemplos y razonamiento", dict(ej=2, think=True)),
         ("razonamiento con muestreo", dict(ej=2, think=True, muestreo=True, tope=15000, n=12))]
# «con ejemplos y razonamiento» es a temperatura 0 con un tope de 6000 tokens: el razonamiento no termina (ver
# diagnostico_razonamiento.py). «razonamiento con muestreo» usa el muestreo que trae el modelo y un tope de 15 000 tokens,
# en las 12 primeras misiones (unos 4 min por misión).
RES = {}
for nombre, cfg in CONDS:
    if nombre not in GUARDADAS and SIN_MODELO:
        print(f"  {nombre:28s}: sin respuestas guardadas; se omite")
        continue
    filas = GUARDADAS.get(nombre, [])
    for m in (MIS[:cfg.get("n", N_MIS)] if nombre not in GUARDADAS else []):
        ej = EJEMPLOS[:cfg.get("ej", 0)]
        texto = prompt(m, ej, coordenadas=not cfg.get("img"))
        imgs = [imagen(m)] if cfg.get("img") else None
        r = llamar(texto, think=cfg.get("think", False), images=imgs, muestreo=cfg.get("muestreo", False), tope=cfg.get("tope", 6000))
        t_tot, tok, intentos = r["_t"], r.get("eval_count", 0), 1
        tipo, err, energia = verificar(m, r.get("response", ""))
        while tipo == "fallo" and intentos <= cfg.get("bucle", 0):
            salida = " (o responde que la misión no es posible si no hay forma de cumplirla)" if cfg.get("rendirse") else ""
            texto2 = texto + "\n" + r.get("response", "") + "\n\nEl verificador ha encontrado estos problemas:\n- " + "\n- ".join(err) + \
                f"\nCorrige el plan{salida}. Respuesta:"
            r = llamar(texto2, images=imgs)
            t_tot += r["_t"]
            tok += r.get("eval_count", 0)
            intentos += 1
            tipo, err, energia = verificar(m, r.get("response", ""))
        cats = sorted({e.split(":")[0] for e in err})
        filas.append(dict(sem=m["semilla"], factible=m["factible"], tipo=tipo, cats=cats, energia=energia, e_opt=m["e_opt"],
                          t=t_tot, tok=tok, intentos=intentos, resp=r.get("response", ""), pev=r.get("prompt_eval_count", 0)))
    RES[nombre] = filas
    if not SIN_MODELO:
        with open(SALIDA_JSON, "w", encoding="utf-8") as f:
            json.dump(RES, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    fac = [f for f in filas if f["factible"]]
    imp = [f for f in filas if not f["factible"]]
    ok = [f for f in fac if f["tipo"] == "ok"]
    ratio = [f["energia"] / f["e_opt"] for f in ok]
    cnt = {c: sum(c in f["cats"] for f in fac if f["tipo"] == "fallo") for c in ("formato", "estructura", "zonas", "cruce", "energía")}
    print(f"  {nombre:28s}: posibles bien planificadas {len(ok)}/{len(fac)} ({100 * len(ok) / len(fac):.0f} %) | "
          f"imposibles rechazadas {sum(f['tipo'] == 'rechazo' for f in imp)}/{len(imp)} | rechazos indebidos {sum(f['tipo'] == 'rechazo' for f in fac)} | "
          f"energía / óptima (mediana) {np.median(ratio) if ratio else float('nan'):.2f} | "
          f"tiempo medio {np.mean([f['t'] for f in filas]):.1f} s | tokens generados {np.mean([f['tok'] for f in filas]):.0f} | "
          f"intentos medios {np.mean([f['intentos'] for f in filas]):.2f}", flush=True)
    print(f"    fallos en misiones posibles por tipo (una misión puede tener varios): " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
    # ¿acierta solo donde basta ir en línea recta? (ruta recta en el orden óptimo sin cruzar zonas prohibidas)
    recta = {m["semilla"] for m in MIS if m["semilla"] in {f["sem"] for f in filas} and m["factible"] and not ruta_directa_cruza(
        m, [[z["id"] for z in m["zonas"]].index(p["zona"]) for p in m["plan_opt"] if p["accion"] == "buscar"])}
    ok_r = sum(f["sem"] in recta for f in ok)
    print(f"    aciertos donde basta la ruta recta {ok_r}/{len(recta)}; donde hay que rodear una zona prohibida {len(ok) - ok_r}/{len(fac) - len(recta)}")

# latencia del modelo: prefijo (prompt) y generación
r = None if SIN_MODELO else llamar(prompt(MIS[0], EJEMPLOS))
if r is None:
    print("latencia: no se mide sin el modelo")
else:
    print(f"latencia: prompt de {r['prompt_eval_count']} tokens en {r['prompt_eval_duration'] / 1e9:.2f} s "
          f"({r['prompt_eval_count'] / (r['prompt_eval_duration'] / 1e9):.0f} tokens/s); "
          f"generación de {r['eval_count']} tokens a {r['eval_count'] / (r['eval_duration'] / 1e9):.1f} tokens/s")

# ejemplo para la figura: la primera misión posible en la que el modelo con dos ejemplos falla por cruce y el verificador
# que exige un plan acaba con uno válido
def puntos(plan_json, m):
    try:
        plan = json.loads(plan_json).get("plan", [])
    except Exception:
        return []
    ids = {z["id"]: z for z in m["zonas"]}
    pts = [(0.0, 0.0)]
    for p in plan:
        a = p.get("accion") if isinstance(p, dict) else None
        try:
            if a == "ir_a":
                pts.append((float(p["x"]), float(p["y"])))
            elif a == "buscar" and p.get("zona") in ids:
                pts.append(centro(ids[p["zona"]]["rect"]))
            elif a == "volver_base":
                pts.append((0.0, 0.0))
        except Exception:
            pass
    return pts


CORR = {f["sem"]: f for f in RES.get("verificador que exige un plan", [])}
ej = next((f for f in RES.get("con dos ejemplos", []) if f["factible"] and "cruce" in f["cats"] and CORR[f["sem"]]["tipo"] == "ok"), None)
if ej is not None:
    m = MIS[ej["sem"]]
    corr = CORR[ej["sem"]]
    print(f"figura: misión {m['semilla']}; con dos ejemplos -> {ej['cats']}; con verificador -> {corr['tipo']} en {corr['intentos']} intentos")
    print(f"figura: errores del verificador al plan con dos ejemplos: {' | '.join(verificar(m, ej['resp'])[1])}")
    print(f"figura: energía del plan corregido / óptima = {corr['energia'] / m['e_opt']:.3f}")
    for z_i, z in enumerate(m["zonas"]):
        print(f"zona {z['id']} {z['nombre']} {z['rect']} {'prohibida' if z_i in m['prohibidas'] else 'pedida' if z_i in m['pedidas'] else 'otra'}")
    for nombre, pj in (("con dos ejemplos", ej["resp"]), ("con verificador", corr["resp"]), ("óptimo", json.dumps({"plan": m["plan_opt"]}))):
        print(f"puntos {nombre}: " + " ".join(f"{x:.1f},{y:.1f}" for x, y in puntos(pj, m)))

