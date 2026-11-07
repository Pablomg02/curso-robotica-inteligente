# Laboratorio A15a · MAPPO con tres drones que buscan

**Duración:** unas 4 h.
**Requisitos:** JAX con optax. Con una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`), el script entero tarda unos 25 minutos (12 entrenamientos de PPO). En CPU, usa `A15A_RAPIDO=1` o menos actualizaciones (`A15A_UPD`). Usa un límite de memoria (`ulimit -v`) y no cierres un `jax.jit` sobre arrays grandes: pásalos como argumentos (el script pasa así los parámetros de la red al evaluar).

## Objetivo

Plantear la búsqueda con varios drones como un Dec-POMDP, entrenar políticas descentralizadas con parámetros compartidos y medir lo que aportan el crítico centralizado y la comunicación, y lo que cuesta un canal imperfecto. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a15a.py`, salida en `figuras/salida_gpu.txt`):

| Resultado de la presentación (determinista en la misma GPU) | Valor | Paso |
|---|---|---|
| Juego matricial: acción conjunta de VDN / mezcla monótona / Q conjunta | (1, 1) en las 5 / (1, 1) en 4 de 5; (2, 1) en 1 de 5 / (0, 0) en las 5 | 5 |
| P(encontrarlo en 80 pasos): aleatorio / barrido por franjas / voraz centralizada | 0,282 / 0,798 / 0,894 | 2 |
| IPPO sin comunicación / MAPPO sin comunicación / MAPPO con comunicación | 0,878 / 0,892 / 0,910 | 3 |
| MAPPO con comunicación con mensajes de hace 10 pasos / sin mensajes | 0,880 / 0,820 | 4 |
| Solapamiento con mensajes al instante / de hace 10 pasos / sin mensajes | 0,36 / 0,66 / 0,82 | 4 |
| MAPPO con comunicación con mapas sin centrar | 0,893 | 3 |

## 1. El entorno (45 min)

1. Rejilla de 12 × 12, mapa previo con dos gaussianas en sitios al azar más un 20 % uniforme, tres drones que salen de la esquina y un sensor que detecta con probabilidad 0,7 en su celda.
2. Escribe la recompensa del belief-MDP, \(r_t = \sum_c m_t(c)(1 - q^{k_t(c)})\), y comprueba con 10 000 episodios con objetivo y detecciones simulados que su suma estima bien la probabilidad de encontrarlo.
3. Observación de cada dron en mapas centrados en él: su belief (o el del equipo), las posiciones de los demás y la máscara de la rejilla.

## 2. Referencias (30 min)

Aleatorio, barrido en zigzag por franjas y la heurística voraz centralizada (cada dron, por turnos, va hacia la celda con más masa descontada por la distancia). ¿Cuál solapa menos? ¿Cuál encuentra antes?

## 3. IPPO y MAPPO (75 min)

1. PPO con parámetros compartidos e identidad del dron en la observación. IPPO: el crítico ve lo mismo que el actor. MAPPO: el crítico ve el belief del equipo y las posiciones de todos.
2. Con y sin comunicación, 3 semillas. Curvas de aprendizaje y resultados finales.
3. Repite con mapas sin centrar. ¿Cuánto más tarda en aprender?

## 4. El canal (45 min)

1. Evalúa la política con comunicación con mensajes de hace 10 pasos y sin mensajes. Compara con la entrenada sin comunicación.
2. Elige la acción más probable en lugar de muestrear. ¿Por qué empeora con comunicación?
3. Opcional: entrena con mensajes que se pierden con probabilidad 0,2 y vuelve a evaluar (es el punto de partida de A15b).

## 5. Juego matricial (30 min)

VDN, una mezcla monótona (pesos en valor absoluto) y una Q conjunta en el juego con un óptimo aislado. Cambia la distribución de los datos (más peso en (0, 0)): ¿cambia la elección de VDN?

## Entregables

- `marl/entorno_busqueda.py`, `marl/mappo.py` y `experimentos/a15a.py`.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Qué información del crítico centralizado tendrías en tu simulador y no en vuelo?
2. ¿Qué se contarían tus drones, y con qué frecuencia, en una radio real?
3. ¿Exige tu tarea de búsqueda observar de verdad, o la resolvería una política de lazo abierto?
