# Laboratorio A7b · Recurrente, belief explícito y belief aprendido en tu búsqueda

**Duración:** unas 4 h.
**Requisitos:** tu PPO en JAX de A6a y el entorno de búsqueda de A7a. Una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`) o en la nube; en CPU funciona con menos entornos (por ejemplo, 256) y más paciencia. Lanza las pruebas largas con un límite de memoria (`ulimit -v`) y con `python -u` para ver el progreso.

## Objetivo

Entrenar políticas para la búsqueda con un dron de A7a con distintas formas de memoria, compararlas con las heurísticas y POMCP de A7a con el mismo protocolo, y comprobar qué hay en el estado oculto de una política recurrente. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a7b.py`, salida en `figuras/salida_gpu.txt`):

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Pasos medios hasta detectar, solo la casilla actual (3 semillas) | 35,0 (34,5–35,7) | 2 |
| Las 8 últimas casillas | 32,9 (32,6–33,6) | 2 |
| GRU separada en actor y crítico | 32,3 (31,4–32,9) | 2 |
| Belief exacto y casilla | 31,9 (31,5–32,3) | 2 |
| Referencias de A7a: barrido, máximo del belief, ganancia por distancia | 37,9; 34,0; 32,4 | 2 |
| Sondeo lineal del belief: variación total con la GRU entrenada | 0,064 (R² 0,74) | 3 |
| Con una GRU sin entrenar y las mismas observaciones | 0,073 (R² 0,70) | 3 |
| Con solo la casilla actual; con el prior fijo | 0,151; 0,250 | 3 |

## 1. PPO recurrente (60 min)

1. Extiende tu PPO: el despliegue guarda el estado oculto al inicio de cada tramo; al terminar un episodio, el estado se pone a cero.
2. La pérdida recorre cada tramo desde el estado guardado, con los mismos reinicios. Los minilotes son conjuntos de entornos, no de pasos.
3. **Tests:** (a) con una política cuya salida depende solo del estado oculto, la log-probabilidad recalculada en la pérdida coincide con la del despliegue antes de actualizar; (b) un episodio que termina a mitad del tramo no deja rastro en el siguiente.
4. Actor y crítico con GRU separadas, como recomiendan Ni et al. Prueba también un codificador compartido y mira la norma del gradiente.

## 2. Cuatro entradas (45 min)

1. Entrena con: solo la casilla actual; las 8 últimas casillas; GRU; belief exacto y casilla. Tres semillas cada una.
2. Evalúa con el bucle de A7a (mismos 600 objetivos, misma semilla de detección) y compara con barrido, máximo del belief, ganancia por distancia y POMCP.
3. Amplía la rejilla a 10 × 10 y repite. ¿Crece la diferencia entre memoria y sin memoria? ¿Y entre GRU y belief exacto?

## 3. Sondeo del estado oculto (45 min)

1. Registra estados ocultos de la GRU entrenada y, en cada paso, el belief exacto y qué casillas se han mirado ya.
2. Ajusta sondeos lineales con el 70 % de los episodios y evalúa en el resto: variación total al belief exacto y acierto en «casilla mirada».
3. Referencias: una GRU sin entrenar recorriendo **las mismas observaciones**, solo la casilla actual y el prior fijo.
4. Repite con una MLP pequeña como sondeo. ¿Cambia la conclusión?

## 4. Pérdidas auxiliares (40 min)

1. Añade a la GRU una cabeza que prediga el belief exacto (supervisión privilegiada, solo en entrenamiento) y otra que prediga la siguiente observación.
2. Compara rendimiento, velocidad de aprendizaje y sondeo con la GRU sin pérdidas auxiliares.
3. Relaciónalo con la cota del estado de información aproximado de A7a.

## 5. Crítico asimétrico (30 min)

1. Crítico con la posición verdadera del objetivo (\(V(s)\)) y crítico con el belief y la posición del dron (\(V(b, s)\)), actor con GRU.
2. ¿Cuál aprende antes? Relaciónalo con el ejemplo de las dos puertas de A6b.

## Entregables

- `rl/ppo_recurrente.py` con tests; `experimentos/a7b.py` que genere la tabla de las cuatro entradas, las curvas y el sondeo.
- Una tabla con tus valores frente a los de la presentación y una nota sobre el sondeo.

## Preguntas para cerrar

1. En tu tarea de tesis, ¿podrás calcular un belief explícito? ¿Con qué modelo de observación?
2. Si no, ¿qué pérdidas auxiliares usarías para que el estado de la red se parezca a un estado de información?
3. ¿Cómo comprobarías en vuelo real que la política mantiene un belief coherente?
