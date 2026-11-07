"""Por qué el modo de razonamiento no termina a temperatura 0 (tema A11).

Uso:  uv run --with numpy --with matplotlib python -I diagnostico_razonamiento.py > salida_diagnostico.txt   (ollama con el modelo; unos 12 min)

Misión 0 con dos ejemplos y razonamiento, con un tope de 15 000 tokens: (a) temperatura 0, como el resto del experimento;
(b) el muestreo que trae el modelo en ollama (temperatura 1, top_k 64, top_p 0,95), con semilla fija. Para cada una
imprime cuánto tarda, cuántos tokens genera, si termina y el final del razonamiento, y cuenta cuántas veces vuelve a
empezar una comprobación («Wait, let me re-...» o «Wait, let's re-...»).
"""
import json, os, re, time, urllib.request

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras_a11.py"), encoding="utf-8").read()
g = {"__name__": "diagnostico"}
exec(src.split("# ---------------------------------------------------------------------- experimento")[0], g)
m = g["generar"](0)
texto = g["prompt"](m, g["EJEMPLOS"])
for nombre, opts in (("temperatura 0", {"temperature": 0}), ("muestreo del modelo", {"temperature": 1.0, "top_k": 64, "top_p": 0.95})):
    body = {"model": g["MODELO"], "prompt": texto, "stream": False, "think": True,
            "options": dict(opts, seed=0, num_ctx=16384, num_predict=15000)}
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(g["URL"], data=json.dumps(body).encode(),
                                                                 headers={"Content-Type": "application/json"}), timeout=1800).read())
    th, resp = r.get("thinking", ""), r.get("response", "")
    tipo = g["verificar"](m, g["extraer_json"](resp))[0] if resp else "sin respuesta"
    PAT = r"Wait, let(?:'s| me) re-"
    rech = len(re.findall(PAT, th))
    rech_fin = len(re.findall(PAT, th[len(th) // 2:]))
    print(f"{nombre}: {time.perf_counter() - t0:.0f} s; {r.get('eval_count')} tokens generados; termina: {r.get('done_reason') == 'stop'}; "
          f"plan: {tipo}; «Wait, let me/let's re-...» {rech} veces ({rech_fin} en la segunda mitad del razonamiento)")
    print("  final del razonamiento:", " ".join(th[-700:].split()))
