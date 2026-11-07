# Laboratorio A12b · Ajuste fino ligero de un VLA con datos del simulador

**Duración:** unas 4 h, de las que unos 42 minutos son de cálculo.
**Requisitos:** una GPU con 8 GB o más (la memoria máxima medida al ajustar fue de 4,4 GiB en una RTX 4060 Ti) y `uv run --with "lerobot[smolvla]"`; descarga `lerobot/smolvla_base` (unos 450 M parámetros). En CPU solo es práctica la prueba corta (`A12B_RAPIDO=1`).

## Objetivo

Medir qué aporta un VLA preentrenado frente a una política pequeña entrenada desde cero en una tarea de dron sencilla, con pocos y con más datos, y si generaliza a una palabra que nunca fue objetivo. Comprueba con tus números lo que mide la presentación (`figuras/ajuste_smolvla.py`, salida en `figuras/salida.txt`):

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| SmolVLA base, sin ajustar (color visto) | 0 % | 2 |
| SmolVLA ajustado, 200 / 1000 demostraciones (color visto) | 24 % / 63 % | 3 |
| Política pequeña, 200 / 1000 demostraciones (color visto) | 38 % / 95 % | 4 |
| Amarillo, 1000 demostraciones: SmolVLA / pequeña | 6 % / 0 % | 5 |
| Pequeña con la velocidad en la entrada, 200 demostraciones | 29 % | 4 |
| Parámetros entrenables de SmolVLA / memoria máxima al ajustar | 100 M / 4,4 GiB | 3 |
| Tiempo por paso / ajuste completo de SmolVLA | 0,25 s / 16,9 min | 3 |
| Un trozo de acciones de SmolVLA (lote 1, bfloat16) | 105 ms | 3 |

## 1. Simulador y experto (45 min)

1. Zona de 20 × 20 m con tres discos de 1 m de radio y colores distintos (de rojo, verde, azul y amarillo); imagen cenital de 256 × 256; orden en inglés («fly to the red object»).
2. Dron que sigue la velocidad pedida con una constante de tiempo de 0,3 s; acción (vx, vy) a 10 Hz en trozos de 1 s, de los que se ejecuta la mitad.
3. Experto proporcional, con 3 m/s como máximo; demostraciones con ruido en la acción ejecutada y la acción limpia como etiqueta. El amarillo nunca es objetivo.

## 2. SmolVLA base (20 min)

Evalúa el modelo sin ajustar. ¿Por qué no sirve de nada? Mira qué representa su salida.

## 3. Ajuste fino (60 min)

1. Congela el VLM y entrena el experto de acciones y la proyección del estado, con 200 y con 1000 demostraciones (4000 pasos de lote 16, bfloat16).
2. Mide parámetros entrenables, tiempo por paso, memoria y latencia de un trozo.
3. Calcula cuántos parámetros entrenaría LoRA de rango 8 y 32 en una proyección de 960 × 960 y compáralo con lo que entrenas tú.

## 4. Política pequeña (45 min)

1. CNN sobre la imagen a 64 × 64 con canales de coordenadas y FiLM (la palabra del color modula cada capa), más la posición; regresión del trozo.
2. Repite con la velocidad en la entrada. ¿Qué pasa y por qué? Relaciónalo con la confusión causal (A10).

## 5. Generalizar (30 min)

Evalúa las dos políticas pidiendo el amarillo. Mira no solo el éxito sino a qué objeto van. ¿Qué tendrías que cambiar en los datos o en el ajuste para que SmolVLA fuera al amarillo?

## Entregables

- `sim/escena.py`, `politicas/pequena.py`, `experimentos/a12b.py`.
- Una tabla con tus valores frente a los de la presentación y una figura con trayectorias de las dos políticas.

## Preguntas para cerrar

1. Para tu dron de búsqueda, ¿ajustarías un VLA o entrenarías una política pequeña? ¿De qué depende?
2. ¿Qué pondrías en la entrada para que la política tenga en cuenta lo ya explorado?
3. ¿Cabe la latencia medida en un lazo a 10 Hz? ¿Y a 50 Hz?
