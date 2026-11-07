"""Datos de las figuras y tablas del tema A12a (VLAs: fundamentos, arquitecturas, tokenización de acciones, datos).

Uso:  uv run --with numpy --with scipy --with tokenizers python -u figuras_a12a.py > salida.txt
      La parte 3 necesita un servidor de ollama en local con un modelo multimodal (A12A_MODELO, por defecto
      gemma4:12b-it-qat); sin él, se salta.

1. Acciones de un cuadricóptero en CTBR (empuje colectivo normalizado y tres velocidades angulares) a 50 y 200 Hz,
   calculadas por planitud diferencial a partir de referencias suaves aleatorias, con y sin ruido.
2. Tokenizar trozos de 1 s: por casillas (256 por dimensión y paso, como RT-1, RT-2 y OpenVLA), por casillas con BPE, y
   con la transformada discreta del coseno, redondeo y BPE (como FAST). Tokens por trozo frente a error de reconstrucción.
3. Latencia de un modelo de visión y lenguaje local que generara esos tokens: prellenado con y sin imagen y
   decodificación; cuántas acciones por segundo daría cada tokenización.
"""
import json
import math
import os
import time
import urllib.request

import numpy as np
from scipy.fft import dct, idct

rng = np.random.default_rng(0)
G, M = 9.81, 2.0
T_MAX = 2.0 * M * G                         # empuje máximo (N): acción 1 = T_MAX, -1 = 0
W_MAX = np.array([4.0, 4.0, 2.0])           # velocidades angulares máximas (rad/s), como el x500 de A6b


def referencia(t, k):
    """Posición y guiñada suaves: suma de tres senos por eje con frecuencias de 0,1 a 0,8 Hz (aceleraciones moderadas)."""
    r = np.random.default_rng(k)
    f = r.uniform(0.1, 0.8, (4, 3))
    a = r.uniform(0.2, 1.5, (4, 3)) / (1 + 4 * f)          # menos amplitud a más frecuencia
    ph = r.uniform(0, 2 * np.pi, (4, 3))
    w = 2 * np.pi * f
    pos = lambda d: sum(a[:, i, None] * (w[:, i, None] ** d) * np.sin(w[:, i, None] * t + ph[:, i, None] + d * np.pi / 2) for i in range(3))
    return pos(0), pos(2), pos(3)                            # p, aceleración, sobreaceleración (filas: x, y, z, guiñada)


def acciones_ctbr(dur, hz, k, ruido=0.0):
    """Planitud diferencial (Mellinger y Kumar): empuje y velocidades angulares a partir de aceleración y sobreaceleración."""
    t = np.arange(0, dur, 1.0 / hz)
    p, a, j = referencia(t, k)
    acc = a[:3].T + np.array([0, 0, G])
    T = M * np.linalg.norm(acc, axis=1)
    zb = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    psi, dpsi = p[3], np.gradient(p[3], t)
    xc = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], 1)
    yb = np.cross(zb, xc)
    yb /= np.linalg.norm(yb, axis=1, keepdims=True)
    xb = np.cross(yb, zb)
    h = M / T[:, None] * (j[:3].T - np.sum(zb * j[:3].T, 1, keepdims=True) * zb)
    wx, wy = -np.sum(h * yb, 1), np.sum(h * xb, 1)
    wz = dpsi * zb[:, 2]
    u = np.stack([2 * T / T_MAX - 1, wx / W_MAX[0], wy / W_MAX[1], wz / W_MAX[2]], 1)
    u = u + ruido * np.random.default_rng(k + 10 ** 6).normal(size=u.shape)
    return np.clip(u, -1, 1)


def trozos(hz, n, ruido, base):
    return np.stack([acciones_ctbr(1.0, hz, base + k, ruido) for k in range(n)])      # (n, hz, 4)


# ---------------------------------------------------------------------- tokenizadores
def casillas(x, n=256):
    q = np.clip(np.floor((x + 1) / 2 * n), 0, n - 1).astype(int)
    return q, (q + 0.5) / n * 2 - 1


def dct_cuant(x, escala):
    c = dct(x, axis=1, norm="ortho")                         # por dimensión, a lo largo del tiempo
    q = np.round(c * escala).astype(int)
    return q, idct(q / escala, axis=1, norm="ortho")


def a_texto(q, orden_frecuencia=True):
    """Secuencia de enteros como caracteres (un símbolo por valor); FAST ordena por frecuencia: primero los coeficientes
    de baja frecuencia de todas las dimensiones."""
    seq = q.transpose(0, 1, 2).reshape(q.shape[0], -1) if not orden_frecuencia else q.reshape(q.shape[0], -1)
    return ["".join(chr(0x4E00 + int(v) + 2048) for v in fila) for fila in seq]


def entrenar_bpe(textos, vocab=1024):
    from tokenizers import Tokenizer, models, trainers
    tok = Tokenizer(models.BPE())
    alfabeto = sorted(set("".join(textos)))
    tr = trainers.BpeTrainer(vocab_size=max(vocab, len(alfabeto) + 1), initial_alphabet=alfabeto, show_progress=False, special_tokens=[])
    tok.train_from_iterator(textos, trainer=tr)
    return tok


def n_tokens(tok, textos):
    return np.array([len(tok.encode(s).ids) for s in textos])


print("=" * 70)
print("1-2. TOKENIZAR TROZOS DE 1 s DE ACCIONES CTBR (4 dimensiones)")
muestra = trozos(50, 300, 0.0, 90000)
print(f"  desviación típica de las acciones (empuje, wx, wy, wz): {', '.join(f'{v:.3f}' for v in muestra.reshape(-1, 4).std(0))}; "
      f"error de cuantización teórico de 256 casillas en [-1, 1]: {2 / 256 / math.sqrt(12):.4f}")
RES = {}
for hz in (50, 200):
    for ruido in (0.0, 0.02):
        tr_x, te_x = trozos(hz, 1500, ruido, 0), trozos(hz, 300, ruido, 50000)
        rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
        q_tr, _ = casillas(tr_x)
        q_te, rec = casillas(te_x)
        # redundancia: fracción de tokens iguales al del paso anterior en la misma dimensión
        rep = float(np.mean(q_te[:, 1:] == q_te[:, :-1]))
        n_bin = hz * 4
        tok_b = entrenar_bpe(a_texto(q_tr.transpose(0, 2, 1)))          # por dimensión, en orden temporal
        nb = n_tokens(tok_b, a_texto(q_te.transpose(0, 2, 1)))
        lin = f"  {hz} Hz, ruido {ruido}: casillas {n_bin} tokens, RMSE {rmse(te_x, rec):.4f}, repetidos {100 * rep:.0f} % | casillas + BPE {nb.mean():.0f} tokens"
        RES[(hz, ruido, "casillas")] = (n_bin, rmse(te_x, rec), rep)
        RES[(hz, ruido, "casillas+bpe")] = (float(nb.mean()), rmse(te_x, rec))
        for esc in (2, 5, 10, 20, 50):
            q_tr, _ = dct_cuant(tr_x, esc)
            q_te, rec = dct_cuant(te_x, esc)
            tok = entrenar_bpe(a_texto(q_tr.transpose(0, 1, 2)))
            nt = n_tokens(tok, a_texto(q_te.transpose(0, 1, 2)))
            nz = float(np.mean(q_te != 0))
            RES[(hz, ruido, f"dct{esc}")] = (float(nt.mean()), rmse(te_x, np.clip(rec, -1, 1)), nz)
            lin += f" | DCT×{esc}+BPE {nt.mean():.0f} tokens, RMSE {rmse(te_x, np.clip(rec, -1, 1)):.4f}, coef. no nulos {100 * nz:.1f} %"
        print(lin, flush=True)

# puntos de la figura: tokens por trozo (eje logarítmico de 5 a 1000) frente a RMSE (de 0 a 0,06), sin ruido
X = lambda n: 60 + (math.log10(n) - math.log10(5)) / (math.log10(1000) - math.log10(5)) * 460
Y = lambda e: 250 - e / 0.06 * 220
for hz in (50, 200):
    pts = [RES[(hz, 0.0, f"dct{e}")][:2] for e in (2, 5, 10, 20, 50)]
    print(f"path pareto dct {hz}:", "M" + " L".join(f"{X(n):.1f},{Y(e):.1f}" for n, e in pts))
    for nombre in ("casillas", "casillas+bpe"):
        n, e = RES[(hz, 0.0, nombre)][:2]
        print(f"punto {nombre} {hz}: {X(n):.1f},{Y(e):.1f}")

# ---------------------------------------------------------------------- 3. latencia con un VLM local
print("=" * 70)
print("3. LATENCIA DE UN VLM LOCAL QUE GENERARA LOS TOKENS DE ACCIÓN")
MODELO = os.environ.get("A12A_MODELO", "gemma4:12b-it-qat")


def llamar(prompt, n, imagen=None):
    body = {"model": MODELO, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": n, "num_ctx": 4096}}
    if imagen:
        body["images"] = [imagen]
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def imagen_prueba():
    """Una imagen sintética de 224 x 224 (PNG) como cámara del dron; el contenido no importa para la latencia."""
    import base64, struct, zlib
    w = h = 224
    img = (np.indices((h, w)).sum(0) % 32 < 16).astype(np.uint8) * 200
    raw = b"".join(b"\x00" + bytes(np.repeat(row, 3)) for row in img)
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


try:
    llamar("Hola", 4)                                       # carga el modelo
    texto = "Escribe los números enteros del 1 al 400 separados por espacios, sin nada más."
    img = imagen_prueba()
    filas = []
    for rep in range(3):
        r0 = llamar(texto, 300)
        r1 = llamar(texto, 300, img)
        filas.append((r0["prompt_eval_count"], r0["prompt_eval_duration"] / 1e9, r1["prompt_eval_count"], r1["prompt_eval_duration"] / 1e9,
                      r0["eval_count"] / (r0["eval_duration"] / 1e9), r1["eval_count"] / (r1["eval_duration"] / 1e9)))
    f = np.median(np.array(filas), 0)
    print(f"  prellenado sin imagen: {f[0]:.0f} tokens en {f[1] * 1e3:.0f} ms | con imagen de 224 x 224: {f[2]:.0f} tokens en {f[3] * 1e3:.0f} ms")
    print(f"  decodificación: {f[4]:.1f} tokens/s sin imagen, {f[5]:.1f} con imagen")
    vel, pre = f[5], f[3]
    for hz in (50, 200):
        for nombre, n in (("casillas", RES[(hz, 0.02, "casillas")][0]), ("DCT×10+BPE", RES[(hz, 0.02, "dct10")][0])):
            t = pre + n / vel
            print(f"  {hz} Hz, {nombre}: {n:.0f} tokens por trozo de 1 s -> {t:.2f} s por trozo (prellenado + decodificación); "
                  f"{'tiempo real' if t < 1.0 else f'{t:.1f} veces más lento que el tiempo real'}")
except Exception as e:
    print(f"  sin servidor de ollama ({type(e).__name__}): parte 3 omitida")
