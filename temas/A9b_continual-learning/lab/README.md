# Laboratorio A9b · Olvido y plasticidad en un modelo de la aerodinámica

**Duración:** unas 3,5 h.
**Requisitos:** JAX con optax. Basta la CPU con menos tareas; con una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`), todo el script tarda unos 10 minutos. Usa un límite de memoria (`ulimit -v`) y no cierres un `jax.jit` sobre arrays grandes: pásalos como argumentos.

## Objetivo

Medir por separado los dos fallos del aprendizaje continuo con un modelo de la fuerza aerodinámica residual de un dron: el olvido, cuando cambia la misión pero no la dinámica, y la pérdida de plasticidad, cuando la dinámica cambia una y otra vez. Probar los remedios de cada uno. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a9b.py`, salida en `figuras/salida_gpu.txt`):

| Resultado de la presentación (determinista en la misma GPU) | Valor | Paso |
|---|---|---|
| Ajuste fino: error en el régimen agresivo justo después de aprenderlo / al final | 0,0059 / 0,2371 N² | 1 |
| BWT (con errores; positivo es olvido): ajuste fino / EWC λ = 1 / repetición / conjunto | +0,0660 / +0,0031 / +0,0016 / −0,0016 N² | 2 |
| Error en el régimen recién aprendido: ajuste fino / EWC λ = 1 | 0,0012 / 0,0022 N² | 2 |
| 300 vientos y cargas, Adam: error normalizado en las tareas 1–30 / 271–300; unidades muertas al final | 0,016 / 0,007; 28 % | 3 |
| 600 maestros aleatorios, Adam con tasa 1e-3: tareas 1–60 / 541–600 | 0,159 / 0,156 | 3 |
| Ídem con tasa 1e-2: Adam / retropropagación continua | 0,110 → 0,159 / 0,099 → 0,118 | 4 |

## 1. Regímenes en secuencia (45 min)

1. Escribe la fuerza residual supuesta (resistencia, arrastre de rotor, pérdida de empuje) y cinco generadores de entradas: flotar, agresivo, avance lento, subidas y bajadas, lateral rápido.
2. Entrena una red de 2×64 ReLU régimen tras régimen y construye la matriz \(R_{i,j}\). Calcula ACC, BWT y FWT con errores (y explica los signos).
3. Cambia el orden de los regímenes. ¿Qué pasa si el agresivo, el más amplio, va el último?

## 2. Contra el olvido (45 min)

1. EWC con la Fisher diagonal calculada con el jacobiano de la salida. Repite con el gradiente del error y compara: ¿por qué subestima la importancia?
2. Barre λ y representa olvido (BWT) frente al error en el régimen recién aprendido: la frontera de estabilidad y plasticidad.
3. Repetición con 50, 200 y 1000 puntos por régimen. ¿Cuántos hacen falta?

## 3. Flujo de tareas (45 min)

1. 300 tareas de viento y carga y 600 tareas de maestros aleatorios, 500 pasos cada una, con las mismas entradas. Error normalizado al final de cada tarea.
2. Mide la fracción de unidades τ-latentes (τ = 0 y τ = 0,1) y la norma de los pesos cada 10 tareas.
3. ¿Hay pérdida de plasticidad? ¿Y con tasa 1e-2? ¿Y con 32 unidades por capa?

## 4. Contra la pérdida de plasticidad (45 min)

1. Adam + L2, encoger y perturbar (\(\theta \leftarrow \lambda\theta + p\)), retropropagación continua (utilidad con \(\eta = 0{,}99\), madurez 100, pesos de salida a cero) y reiniciar en cada tarea.
2. Añade normalización por capas. ¿Cambia algo?
3. Ajusta la tasa de reemplazo de la retropropagación continua a la tasa de aprendizaje alta.

## 5. Protocolo de actualización (30 min)

Escribe la función que acepta o rechaza una actualización del modelo después de un vuelo: entrenar una copia con datos nuevos y repetidos, evaluar en todos los regímenes, comparar con umbrales y guardar el modelo anterior.

## Entregables

- `continual/metricas.py` (matriz, ACC, BWT y FWT con tests), `continual/metodos.py` y `experimentos/a9b.py`.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Qué regímenes de vuelo de tu tesis podrían quedar sin datos durante semanas?
2. ¿Qué parte de tu world model actualizarías después de cada vuelo, y cuál dejarías fija?
3. ¿Qué indicador vigilarías en vuelo para detectar que el modelo ya no se adapta?
