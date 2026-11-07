"""Ajuste fino ligero de un VLA (SmolVLA, 450 M) con datos de un simulador de dron (tema A12b).

Uso:  uv run --with "lerobot[smolvla]" python -u ajuste_smolvla.py > salida.txt
      A12B_RAPIDO=1 hace una prueba corta (en CPU, en unos minutos). Descarga lerobot/smolvla_base.

Tarea: un dron vuela a altura fija sobre una zona de 20 x 20 m con tres objetos de colores distintos (discos de 1 m de
radio). Una cámara cenital ve toda la zona (imagen de 256 x 256) y una orden en inglés ("fly to the red object") dice a
cuál ir. Estado: posición horizontal; acción: velocidad deseada (vx, vy) a 10 Hz, que el vehículo sigue con una constante
de tiempo de 0,3 s. Experto: ir en línea recta al objeto con una ganancia proporcional y 3 m/s como máximo; su acción solo
depende de la posición. Cada episodio dura 12 s. Éxito: acabar a menos de 1,5 m del objeto pedido.

Colores: rojo, verde, azul y amarillo; cada escena tiene tres distintos. En las demostraciones el objetivo nunca es el
amarillo (solo aparece como distractor). Se evalúa con objetivos de color visto y con el amarillo.

Condiciones (una semilla cada una; las demostraciones llevan ruido en la acción ejecutada, como DART, y la etiqueta es la
acción limpia del experto):
  1. SmolVLA base sin ajustar (su salida es la de un brazo SO-100: sirve de control).
  2. SmolVLA con ajuste fino del experto de acciones y la proyección del estado, con el VLM congelado (la receta por
     defecto de LeRobot), con 200 y con 1000 demostraciones.
  3. Una política pequeña entrenada desde cero con las mismas demostraciones: CNN sobre la imagen a 64 x 64 con canales de
     coordenadas y FiLM (la palabra del color modula cada capa), más la posición; predice el trozo por regresión.
  4. Ablación: la política pequeña con la velocidad también en la entrada (200 demostraciones), para ver la confusión
     causal: la velocidad predice la acción del experto durante la aproximación y la red puede copiarla en vez de mirar.
Se imprime: parámetros entrenables, tiempo por paso, memoria, éxito en lazo cerrado, distancia final y latencia.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAPIDO = os.environ.get("A12B_RAPIDO") == "1"
DEV = "cuda" if torch.cuda.is_available() and not RAPIDO else "cpu"
torch.manual_seed(0)
L, IMG, DT, TAU, VMAX, KP = 20.0, 256, 0.1, 0.3, 3.0, 1.5
T_EP = 120                                  # 12 s a 10 Hz
H, N_EJEC = 10, 5                           # trozo de 1 s; se ejecutan 5 acciones y se replanifica
COLORES = {"red": (0.85, 0.1, 0.1), "green": (0.1, 0.7, 0.15), "blue": (0.1, 0.25, 0.9), "yellow": (0.95, 0.85, 0.1)}
NOMBRES = list(COLORES)
VISTOS = ["red", "green", "blue"]
R_OBJ, R_EXITO = 1.0, 1.5
N_DEMOS = [4] if RAPIDO else [200, 1000]
PASOS = 3 if RAPIDO else 4000
PASOS_P = 3 if RAPIDO else 8000
LOTE = 2 if RAPIDO else 16
N_EVAL = 4 if RAPIDO else 100


# ---------------------------------------------------------------------- simulador
def escena(rng, objetivo_amarillo):
    """Tres objetos separados al menos 4 m, dentro de [2, 18] m; el dron empieza a más de 6 m del objetivo."""
    while True:
        cols = rng.choice(4, 3, replace=False)
        if objetivo_amarillo != (3 in cols):
            continue
        pos = rng.uniform(2, L - 2, (3, 2))
        if min(np.linalg.norm(pos[i] - pos[j]) for i in range(3) for j in range(i)) < 4:
            continue
        objetivo = int(np.where(cols == 3)[0][0]) if objetivo_amarillo else int(rng.choice([i for i in range(3) if cols[i] != 3]))
        p0 = rng.uniform(1, L - 1, 2)
        if np.linalg.norm(p0 - pos[objetivo]) > 6 and min(np.linalg.norm(p0 - q) for q in pos) > 2:
            return dict(cols=cols, pos=pos, obj=objetivo, p0=p0)


def experto(p, e):
    d = e["pos"][e["obj"]] - p
    u = KP * d
    n = np.linalg.norm(u)
    return u if n <= VMAX else u * VMAX / n


def paso(p, v, u):
    v = v + (u - v) * DT / TAU
    return p + v * DT, v


def final(p, e):
    """Objeto junto al que acaba el dron (a menos de R_EXITO), o None."""
    d = np.linalg.norm(e["pos"] - p, axis=1)
    return int(np.argmin(d)) if d.min() < R_EXITO else None


def demos(n, rng):
    """Episodios de T_EP pasos con el experto; se ejecuta u + ruido (desviación 0,3 m/s) y se guarda u."""
    E, S, A = [], np.zeros((n, T_EP, 4), np.float32), np.zeros((n, T_EP, 2), np.float32)
    for k in range(n):
        e = escena(rng, False)
        p, v = e["p0"].copy(), np.zeros(2)
        for t in range(T_EP):
            u = experto(p, e)
            S[k, t], A[k, t] = (*p, *v), u
            p, v = paso(p, v, np.clip(u + rng.normal(0, 0.3, 2), -VMAX, VMAX))
        E.append(e)
    return E, S, A


# ---------------------------------------------------------------------- imagen cenital
YY, XX = torch.meshgrid(torch.arange(IMG, dtype=torch.float32), torch.arange(IMG, dtype=torch.float32), indexing="ij")


def render(escenas, ps, res=IMG):
    """Imágenes (B, 3, res, res) en [0, 1]: fondo, tres discos de color y el dron como un disco negro pequeño."""
    B = len(escenas)
    esc = res / L
    yy, xx = (YY, XX) if res == IMG else torch.meshgrid(torch.arange(res, dtype=torch.float32), torch.arange(res, dtype=torch.float32), indexing="ij")
    img = torch.empty(B, 3, res, res)
    img[:, 0], img[:, 1], img[:, 2] = 0.55, 0.6, 0.5
    for b, (e, p) in enumerate(zip(escenas, ps)):
        for c, q in zip(e["cols"], e["pos"]):
            m = (xx - q[0] * esc) ** 2 + (yy - (L - q[1]) * esc) ** 2 < (R_OBJ * esc) ** 2
            for ch in range(3):
                img[b, ch][m] = COLORES[NOMBRES[c]][ch]
        m = (xx - p[0] * esc) ** 2 + (yy - (L - p[1]) * esc) ** 2 < (0.35 * esc) ** 2
        img[b, :, m] = 0.05
    return img


def orden(e):
    return f"fly to the {NOMBRES[e['cols'][e['obj']]]} object"


def norm_estado(s, con_vel=False):
    """Posición normalizada; con con_vel, también la velocidad (solo para la ablación de la confusión causal)."""
    s = torch.as_tensor(s, dtype=torch.float32)
    pos = (s[..., :2] - L / 2) / (L / 4)
    return torch.cat([pos, s[..., 2:] / VMAX], -1) if con_vel else pos


# ---------------------------------------------------------------------- SmolVLA
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from transformers import AutoTokenizer


def cargar_smolvla(propia):
    cfg = SmolVLAConfig.from_pretrained("lerobot/smolvla_base")
    if propia:
        cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                              "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG, IMG))}
        cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
        cfg.chunk_size, cfg.n_action_steps = H, N_EJEC
    cfg.device = DEV
    # los pesos se distribuyen casi todos en bfloat16; se pasan a float32 para ajustar (con autocast a bfloat16 en GPU)
    return SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=cfg).to(DEV).float()


TOK = AutoTokenizer.from_pretrained(SmolVLAConfig().vlm_model_name)
TOK.padding_side = "right"


def tokens(textos):
    t = TOK([s + "\n" for s in textos], padding="max_length", max_length=16, truncation=True, return_tensors="pt")
    return t["input_ids"].to(DEV), t["attention_mask"].bool().to(DEV)


def lote_smolvla(escenas, ps, estados, camaras):
    ids, mask = tokens([orden(e) for e in escenas])
    img = render(escenas, ps).to(DEV)
    b = {OBS_LANGUAGE_TOKENS: ids, OBS_LANGUAGE_ATTENTION_MASK: mask, "observation.state": norm_estado(estados).to(DEV)}
    for k in camaras:
        b[k] = img
    return b


# ---------------------------------------------------------------------- política pequeña desde cero
class Pequena(nn.Module):
    """CNN con canales de coordenadas y FiLM: la incrustación de la palabra del color escala y desplaza cada capa."""
    def __init__(self, con_vel=False):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
        self.register_buffer("coord", torch.stack([xx, yy])[None])
        self.c1, self.c2, self.c3 = nn.Conv2d(5, 32, 5, 2, 2), nn.Conv2d(32, 64, 3, 2, 1), nn.Conv2d(64, 64, 3, 2, 1)
        self.film = nn.Embedding(4, 2 * (32 + 64 + 64))
        self.con_vel = con_vel
        self.mlp = nn.Sequential(nn.Linear(64 * 8 * 8 + (4 if con_vel else 2), 256), nn.ReLU(), nn.Linear(256, H * 2))

    def forward(self, img, col, est):
        g1, b1, g2, b2, g3, b3 = torch.split(self.film(col), [32, 32, 64, 64, 64, 64], -1)
        x = torch.cat([img, self.coord.expand(img.shape[0], -1, -1, -1)], 1)
        for c, g, b in ((self.c1, g1, b1), (self.c2, g2, b2), (self.c3, g3, b3)):
            x = torch.relu(c(x) * (1 + g[:, :, None, None]) + b[:, :, None, None])
        return self.mlp(torch.cat([x.flatten(1), est], -1)).view(-1, H, 2)


def entradas_pequena(escenas, ps, estados, con_vel=False):
    img = render(escenas, ps, 64).to(DEV)
    col = torch.tensor([int(e["cols"][e["obj"]]) for e in escenas], device=DEV)
    return img, col, norm_estado(estados, con_vel).to(DEV)


# ---------------------------------------------------------------------- entrenamiento y evaluación
def muestras(E, S, A, rng):
    """Lote de (escena, posición, estado, trozo de acciones normalizado, máscara de relleno)."""
    k = rng.integers(0, len(E), LOTE)
    t = rng.integers(0, T_EP, LOTE)
    trozo = np.zeros((LOTE, H, 2), np.float32)
    pad = np.zeros((LOTE, H), bool)
    for i in range(LOTE):
        n = min(H, T_EP - t[i])
        trozo[i, :n] = A[k[i], t[i]:t[i] + n] / VMAX
        trozo[i, n:] = trozo[i, n - 1]
        pad[i, n:] = True
    return [E[j] for j in k], S[k, t, :2], S[k, t], torch.tensor(trozo, device=DEV), torch.tensor(pad, device=DEV)


def entrenar_smolvla(E, S, A, rng):
    pol = cargar_smolvla(True)
    pol.train()
    params = [p for p in pol.parameters() if p.requires_grad]
    n_ent = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * (0.5 * (1 + math.cos(math.pi * s / PASOS)) * 0.9 + 0.1))
    cams = list(pol.config.image_features)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, perdidas = time.perf_counter(), []
    for s in range(PASOS):
        esc, ps, est, trozo, pad = muestras(E, S, A, rng)
        b = lote_smolvla(esc, ps, est, cams)
        b["action"], b["action_is_pad"] = trozo, pad
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            loss, _ = pol.forward(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()
        perdidas.append(loss.item())
        if (s + 1) % max(1, PASOS // 5) == 0:
            print(f"    paso {s + 1}: pérdida de flow matching {np.mean(perdidas[-50:]):.4f}", flush=True)
    dt = (time.perf_counter() - t0) / PASOS
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if DEV == "cuda" else float("nan")
    return pol, n_ent, dt, mem


def entrenar_pequena(E, S, A, rng, con_vel=False):
    pol = Pequena(con_vel).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    n_ent = sum(p.numel() for p in pol.parameters())
    pasos = PASOS_P
    t0 = time.perf_counter()
    global LOTE
    lote0, LOTE = LOTE, (2 if RAPIDO else 64)
    for s in range(pasos):
        esc, ps, est, trozo, pad = muestras(E, S, A, rng)
        pred = pol(*entradas_pequena(esc, ps, est, con_vel))
        loss = (((pred - trozo) ** 2).sum(-1) * (~pad)).sum() / (~pad).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    LOTE = lote0
    return pol, n_ent, (time.perf_counter() - t0) / pasos


def evaluar(politica, tipo, amarillo, semilla, cams=None):
    """Lazo cerrado en lotes de hasta 50 episodios de T_EP pasos; replanifica cada N_EJEC pasos.
    Devuelve (éxito, acaba junto a otro objeto, mediana de la distancia final al objeto pedido)."""
    rng = np.random.default_rng(semilla)
    escs = [escena(rng, amarillo) for _ in range(N_EVAL)]
    exito, mal, dist = 0, 0, []
    for i0 in range(0, N_EVAL, 50):
        es = escs[i0:i0 + 50]
        p = np.array([e["p0"] for e in es])
        v = np.zeros_like(p)
        for t in range(T_EP):
            if t % N_EJEC == 0:
                est = np.concatenate([p, v], 1)
                with torch.no_grad():
                    if tipo == "smolvla":
                        politica.reset()
                        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                            a = politica.predict_action_chunk(lote_smolvla(es, p, est, cams))
                        cola = a[:, :, :2].float().cpu().numpy() * VMAX
                    else:
                        cola = politica(*entradas_pequena(es, p, est, politica.con_vel)).cpu().numpy() * VMAX
            u = np.clip(cola[:, t % N_EJEC], -VMAX, VMAX)
            for b in range(len(es)):
                p[b], v[b] = paso(p[b], v[b], u[b])
        for e, q in zip(es, p):
            f = final(q, e)
            exito += f == e["obj"]
            mal += f is not None and f != e["obj"]
            dist.append(np.linalg.norm(e["pos"][e["obj"]] - q))
    return exito / N_EVAL, mal / N_EVAL, float(np.median(dist))


def latencia(pol, cams):
    rng = np.random.default_rng(5)
    e = [escena(rng, False)]
    b = lote_smolvla(e, np.array([e[0]["p0"]]), np.zeros((1, 4)), cams)
    ts = []
    with torch.no_grad():
        for k in range(23):
            pol.reset()
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                pol.predict_action_chunk(b)
            if DEV == "cuda":
                torch.cuda.synchronize()
            if k >= 3:
                ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ---------------------------------------------------------------------- experimento
t_ini = time.perf_counter()
print("dispositivo:", torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU")
rng = np.random.default_rng(0)
E_all, S_all, A_all = demos(max(N_DEMOS), rng)
print(f"demostraciones: {max(N_DEMOS)} de {T_EP} pasos; las de ruido en la ejecución acaban junto al objetivo en el "
      f"{100 * np.mean([final(S_all[k, -1, :2], E_all[k]) == E_all[k]['obj'] for k in range(len(E_all))]):.0f} %")


def experto_eval(amarillo, semilla):
    """El experto (que conoce el objetivo) en las mismas escenas de evaluación: el techo."""
    rng = np.random.default_rng(semilla)
    ok = 0
    for _ in range(N_EVAL):
        e = escena(rng, amarillo)
        p, v = e["p0"].copy(), np.zeros(2)
        for t in range(T_EP):
            p, v = paso(p, v, experto(p, e))
        ok += final(p, e) == e["obj"]
    return ok / N_EVAL


print(f"experto en las escenas de evaluación: color visto {experto_eval(False, 101):.2f}, amarillo {experto_eval(True, 202):.2f}")
print("azar (acabar junto a un objeto al azar): 0.33")

base = cargar_smolvla(False)
n_tot = sum(p.numel() for p in base.parameters())
print(f"SmolVLA: {n_tot / 1e6:.0f} M parámetros")
qp = [(n, tuple(w.shape)) for n, w in base.named_parameters() if "q_proj" in n and "vlm" in n and "text_model" in n][:1]
print(f"  dimensión oculta del modelo de texto: {base.model.vlm_with_expert.config.text_config.hidden_size}; primera proyección q del VLM: {qp}")
r = evaluar(base, "smolvla", False, 101, list(base.config.image_features))
print(f"  SmolVLA base sin ajustar, color visto: éxito {r[0]:.2f}, otro objeto {r[1]:.2f}, distancia final mediana {r[2]:.1f} m", flush=True)
del base
if DEV == "cuda":
    torch.cuda.empty_cache()

for n in N_DEMOS:
    E, S, A = E_all[:n], S_all[:n], A_all[:n]
    print(f"--- {n} demostraciones ({n * T_EP / 10 / 60:.1f} min de vuelo)")
    pol, n_ent, dt, mem = entrenar_smolvla(E, S, A, np.random.default_rng(1))
    cams = list(pol.config.image_features)
    print(f"  SmolVLA ajustado: {n_ent / 1e6:.0f} M parámetros entrenables; {PASOS} pasos de lote {LOTE} a {dt:.2f} s por paso "
          f"({dt * PASOS / 60:.1f} min); memoria máxima {mem:.1f} GiB", flush=True)
    for am, sem, nom in ((False, 101, "color visto"), (True, 202, "amarillo (nunca objetivo)")):
        r = evaluar(pol, "smolvla", am, sem, cams)
        print(f"  SmolVLA ajustado, {nom}: éxito {r[0]:.2f}, otro objeto {r[1]:.2f}, distancia final mediana {r[2]:.1f} m", flush=True)
    if n == max(N_DEMOS):
        print(f"  latencia de SmolVLA ajustado (un trozo, lote 1, {'bfloat16' if DEV == 'cuda' else 'float32'}): {latencia(pol, cams) * 1e3:.0f} ms")
    del pol
    if DEV == "cuda":
        torch.cuda.empty_cache()
    peq, n_p, dtp = entrenar_pequena(E, S, A, np.random.default_rng(1))
    print(f"  política pequeña: {n_p / 1e6:.2f} M parámetros; {PASOS_P} pasos de lote {2 if RAPIDO else 64} a {dtp * 1e3:.1f} ms por paso ({dtp * PASOS_P / 60:.1f} min)")
    for am, sem, nom in ((False, 101, "color visto"), (True, 202, "amarillo (nunca objetivo)")):
        r = evaluar(peq, "pequena", am, sem)
        print(f"  política pequeña, {nom}: éxito {r[0]:.2f}, otro objeto {r[1]:.2f}, distancia final mediana {r[2]:.1f} m", flush=True)
    if n == min(N_DEMOS):
        peq_v, _, _ = entrenar_pequena(E, S, A, np.random.default_rng(1), con_vel=True)
        r = evaluar(peq_v, "pequena", False, 101)
        print(f"  ablación, política pequeña con la velocidad en la entrada, color visto: éxito {r[0]:.2f}, otro objeto {r[1]:.2f}, "
              f"distancia final mediana {r[2]:.1f} m", flush=True)
print(f"tiempo total: {(time.perf_counter() - t_ini) / 60:.1f} min")
