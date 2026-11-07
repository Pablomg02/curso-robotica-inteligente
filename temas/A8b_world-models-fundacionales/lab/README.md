# Laboratorio A8b · Planificar con MPPI dentro de un world model del x500

**Duración:** unas 4 h.
**Requisitos:** tu simulador del x500 (B2), tu RSSM de A8a y JAX con optax. Una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`) o en la nube; en CPU, con menos episodios. Lanza los entrenamientos con `python -u` y un límite de memoria (`ulimit -v`).
**Ojo con la memoria:** no cierres un `jax.jit` sobre arrays grandes, porque XLA los copia como constantes al compilar. Pásalos como argumentos.

## Objetivo

Planificar con MPPI usando como modelo la física, tu RSSM, un latente sin reconstrucción y una MLP sin memoria. Medir cuánto se pierde, comprobar si el planificador explota el modelo, probar dos remedios y ver si el paso de control cabe en 20 ms. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a8b.py`; salida en `figuras/salida_gpu.txt` y `figuras/salida_cpu.txt`):

| Resultado de la presentación (una ejecución en GPU, determinista en la misma tarjeta) | Valor | Paso |
|---|---|---|
| MPPI con la física nominal: error final (mediana, último segundo) | 7,6 cm (horizontal 1,3, vertical 7,5) | 1 |
| Error horizontal sin / con el término de velocidad, con ruido independiente | 16,8 / 1,8 cm (con ruido correlado: 1,7 / 1,3) | 1 |
| Episodios que divergen con MPPI puro: RSSM, latente, MLP | 59 de 64, 63 de 64, 64 de 64 | 2 |
| Error en bucle abierto del RSSM a 0,5 s: acciones de los datos / del MPPI puro / del MPPI + política | 26 / 193 / 30 cm | 3 |
| Política imitada desde el RSSM: R² en prueba; error final; divergen | 0,96; 47,1 cm; 0 de 64 | 4 |
| MPPI + política con el RSSM: error final; divergen | 40,6 cm; 2 de 64 | 4 |
| Error vertical con masa > +7 %: política / MPPI + política | −56,6 / −29,4 cm | 4 |
| Una iteración de datos: episodios útiles de 1000; RSSM con MPPI puro, divergen | 20; 60 de 64 | 4 |
| Latencia por paso, MPPI con el RSSM: GPU / CPU | 3,5 / 67,1 ms (política sola: 0,17 / 0,07 ms) | 5 |

## 1. MPPI con la física (45 min)

1. Tarea: ir a un punto aleatorio a ±1 m y mantenerse 3 s, con masa desconocida (±15 %). Usa la misma observación que en A8a.
2. Implementa MPPI con 256 secuencias de 25 pasos (0,5 s), 3 iteraciones con ruido decreciente, 32 élites y ruido correlado en el tiempo.
3. Coste: posición, más un campo de velocidades \(v = -\lambda (p - p^\star)\) calculado con diferencias de las posiciones predichas, más acciones pequeñas y suaves. Compara con y sin el término de velocidad, con ruido de muestreo independiente y correlado. ¿Cuándo hace falta el término, y por qué depende del ruido?
4. Mide el error vertical frente a la masa. ¿De dónde sale el sesgo con la física nominal?

## 2. MPPI con modelos aprendidos (60 min)

1. Usa tu RSSM de A8a como modelo: filtra con el posterior, imagina con el prior y decodifica la posición.
2. Entrena un latente sin reconstrucción al estilo de TD-MPC2: un codificador de una ventana de observaciones, una dinámica latente con SimNorm, consistencia con el codificador del paso siguiente (gradiente detenido) y una cabeza que solo predice la posición.
3. Compara en bucle cerrado con la física y con la MLP de un paso de A8a.

## 3. Diagnóstico: ¿explota el planificador el modelo? (40 min)

1. Para cada modelo, mide el error en bucle abierto a 0,5 s con las acciones de los datos de prueba y con las acciones que eligió el planificador en bucle cerrado.
2. Representa las trayectorias de un episodio que diverge: lo que el modelo predecía al planificar frente a lo que pasó.

## 4. Remedios (60 min)

1. **Política previa:** imita al controlador de los datos, sin su ruido, desde la creencia del modelo y el objetivo relativo (maestro con estado verdadero, alumno con creencia, como en A6b). Evalúala sola.
2. **MPPI + política:** optimiza un residuo acotado sobre la política, aplicando la política también dentro de la imaginación. Prueba cotas de ±0,2, ±0,5 y ±1. ¿Dónde está el equilibrio entre corregir y explotar?
3. **Iteración de datos:** recoge episodios con tu planificador, reentrena el modelo y repite tres veces. Recoge con MPPI + política, que no diverge, y evalúa el MPPI puro. ¿Converge?

## 5. Latencia (15 min)

1. Mide el tiempo de un paso de control para un solo dron, en GPU y en CPU (`JAX_PLATFORMS=cpu`, `A8B_SOLO_TIEMPOS=1`).
2. ¿Cuántas secuencias y cuánto horizonte caben en 20 ms en CPU? ¿Cuánto empeora el control?

## Entregables

- `planificacion/mppi.py` con tests: con un modelo lineal conocido, el MPPI debe acercarse al óptimo del LQR.
- `experimentos/a8b.py`, que genere la tabla de bucle cerrado, la de explotación y la de latencias.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. Si tu modelo de A8a se equivoca 25 cm a 0,5 s con las acciones de los datos, ¿qué precisión necesitarías para que el MPPI puro funcione en esta tarea? ¿Cómo lo comprobarías?
2. ¿Qué parte de este laboratorio cambiaría si la creencia tuviera que incluir un objetivo oculto, como en la búsqueda de A7a?
3. Un codificador fundacional de vídeo congelado, ¿te ayudaría aquí? ¿En qué parte del sistema lo pondrías?
