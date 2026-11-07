# Laboratorio A12a · Inferencia y latencia de un VLA pequeño

**Duración:** unas 3,5 h.
**Requisitos:** Python con numpy, scipy y `tokenizers` para los pasos 1 y 2 (segundos en CPU). Para el paso 3, [ollama](https://ollama.com) con un modelo multimodal como en A11. Para el paso 4, `lerobot[smolvla]` (PyTorch y unos 2,7 GB de pesos que se descargan de Hugging Face) y una GPU de consumo o, más despacio, la CPU.

## Objetivo

Medir lo que cuesta representar las acciones de un dron como tokens y si un VLA cabe en el presupuesto de tiempo de su bucle de control. Comprueba con tus números lo que calculan los scripts de la presentación (`figuras/figuras_a12a.py` y `figuras/latencia_smolvla.py`, con sus salidas en `figuras/`):

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Tokens por trozo de 1 s a 50 / 200 Hz con casillas | 200 / 800 | 2 |
| Con casillas + BPE, sin ruido, 50 / 200 Hz | 141 / 421 | 2 |
| Con DCT (escala 10) + BPE, sin ruido, 50 / 200 Hz | 22 / 31 (RMSE 0,0149 / 0,0089) | 2 |
| Con DCT (escala 10) + BPE, con ruido, 50 / 200 Hz | 26 / 44 | 2 |
| Prellenado con imagen y decodificación del VLM de 12B | 156 ms / 35,5 tokens/s | 3 |
| Trozo de 1 s a 50 Hz: casillas / DCT + BPE | 5,79 / 0,88 s | 3 |
| SmolVLA, un trozo de 50 acciones: GPU con pesos en bfloat16 / en float32 / CPU | 109 / 155 / 920 ms | 4 |

## 1. Acciones de un cuadricóptero (40 min)

1. Genera referencias suaves de posición y guiñada (sumas de senos de 0,1 a 0,8 Hz) y calcula por planitud diferencial el empuje y las velocidades angulares (CTBR, como en A6b), normalizados en [−1, 1].
2. Muestrea trozos de 1 s a 50 y a 200 Hz, con y sin ruido gaussiano de desviación 0,02.

## 2. Tres tokenizadores (60 min)

1. Casillas: 256 por dimensión y paso. Comprueba que el RMSE es Δ/√12 y mide qué fracción de tokens repite el anterior.
2. Casillas + BPE (vocabulario de 1024) entrenado con 1500 trozos.
3. DCT por dimensión, escala y redondeo (2, 5, 10, 20 y 50), aplanado con las bajas frecuencias primero y BPE: dibuja tokens frente a error.
4. ¿Por qué el ruido rompe la compresión de la DCT a escalas altas?

## 3. Latencia de decodificación (30 min)

1. Con las estadísticas de ollama, mide el prellenado con y sin una imagen y la velocidad de decodificación.
2. Calcula cuánto tardaría en generarse un trozo de 1 s con cada tokenización y compáralo con 1 s.

## 4. SmolVLA en tu máquina (45 min)

1. Carga `lerobot/smolvla_base`, cuenta parámetros y mide `predict_action_chunk` en GPU (float32 y bfloat16) y en CPU con entradas sintéticas del tamaño correcto.
2. Con trozos de 50 acciones a 50 Hz (1 s), ¿cubre el trozo su propia latencia? ¿Y a bordo?

## 5. Diseño (25 min)

Escribe media página: dónde pondrías un VLA en tu dron, qué generaría, a qué frecuencia y qué iría debajo.

## Entregables

- `tokenizar/acciones.py`, `tokenizar/tokenizadores.py`, `experimentos/a12a.py`.
- Una tabla con tus valores frente a los de la presentación y la figura de tokens frente a error.

## Preguntas para cerrar

1. ¿Qué escala de la DCT elegirías para las acciones de tu dron y por qué?
2. Si tu ordenador de a bordo decodifica 10 veces más despacio que la GPU de la presentación, ¿qué representación de la acción usarías?
3. ¿Qué perderías al mandar referencias de velocidad a 5 Hz en vez de CTBR a 50 Hz?
