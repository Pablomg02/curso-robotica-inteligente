"""Latencia de un VLA pequeño real: SmolVLA (LeRobot) en GPU y en CPU.

Uso:  uv run --with "lerobot[smolvla]" python -u latencia_smolvla.py > salida_smolvla.txt
      Descarga de Hugging Face los pesos de lerobot/smolvla_base (y lo que necesite su modelo de visión y lenguaje).

Mide el tiempo de predecir un trozo de acciones (predict_action_chunk: codificar la imagen, el texto y el estado con el
VLM y generar el trozo con el experto de acciones por flow matching) con entradas sintéticas del tamaño correcto: el
contenido de la imagen y del texto no cambia el cálculo. También cuenta parámetros y la memoria de GPU máxima.
"""
import time

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

torch.manual_seed(0)
pol = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
cfg = pol.config
n_par = sum(p.numel() for p in pol.parameters())
n_ent = sum(p.numel() for p in pol.parameters() if p.requires_grad)
print(f"parámetros: {n_par / 1e6:.0f} M en total; {n_ent / 1e6:.0f} M entrenables")
tipos = {}
for prm in pol.parameters():
    tipos[str(prm.dtype)] = tipos.get(str(prm.dtype), 0) + prm.numel()
print("tipos de los pesos al cargar (M parámetros):", {k: round(v / 1e6) for k, v in tipos.items()}, "")
print(f"trozo de {cfg.chunk_size} acciones; {cfg.num_steps} pasos de flow matching; imágenes a {cfg.resize_imgs_with_padding}; "
      f"VLM {cfg.vlm_model_name}")
print("entradas de imagen:", {k: tuple(f.shape) for k, f in cfg.image_features.items()}, "| estado:", cfg.robot_state_feature.shape if cfg.robot_state_feature else None,
      "| acción:", cfg.action_feature.shape if cfg.action_feature else None)


def lote(dev):
    b = {}
    for k, f in cfg.image_features.items():
        b[k] = torch.rand((1,) + tuple(f.shape), device=dev)
    if cfg.robot_state_feature is not None:
        b["observation.state"] = torch.randn((1,) + tuple(cfg.robot_state_feature.shape), device=dev)
    b[OBS_LANGUAGE_TOKENS] = torch.randint(1, 1000, (1, cfg.tokenizer_max_length), device=dev)
    b[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones((1, cfg.tokenizer_max_length), dtype=torch.bool, device=dev)
    return b


def medir(dev, n, bf16=False, como_carga=False):
    p = pol.to(dev) if como_carga else pol.to(dev).float()
    b = lote(dev)
    ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=bf16)
    with torch.no_grad():
        for _ in range(3):
            p.reset()
            with ac():
                p.predict_action_chunk(b)
        if dev == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(n):
            p.reset()
            t0 = time.perf_counter()
            with ac():
                a = p.predict_action_chunk(b)
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    ts = sorted(ts)
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else float("nan")
    return ts[len(ts) // 2], ts[int(0.9 * len(ts))], tuple(a.shape), mem


if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    for bf, cc, nom in ((False, True, "pesos como se distribuyen (casi todos en bfloat16)"), (False, False, "pesos en float32"),
                        (True, False, "pesos en float32 con autocast a bfloat16")):
        try:
            med, p90, forma, mem = medir("cuda", 30, bf, cc)
            print(f"  GPU, {nom}: trozo {forma} en {med * 1e3:.0f} ms (mediana; percentil 90 {p90 * 1e3:.0f} ms); memoria máxima {mem:.2f} GiB; "
                  f"a 50 Hz, un trozo de {cfg.chunk_size} acciones dura {cfg.chunk_size / 50:.1f} s")
        except Exception as e:
            print(f"  GPU, {nom}: falló ({type(e).__name__}: {str(e)[:120]})")
torch.set_num_threads(8)
med, p90, forma, _ = medir("cpu", 5)
print(f"  CPU (8 hilos), pesos en float32: trozo {forma} en {med * 1e3:.0f} ms (mediana; percentil 90 {p90 * 1e3:.0f} ms)")
