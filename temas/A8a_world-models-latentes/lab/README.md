# Laboratorio A8a · Un RSSM mínimo en JAX con el x500

**Duración:** unas 4 h.
**Requisitos:** tu simulador del x500 (B2) y JAX con optax. Una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`) o en la nube; en CPU, con menos datos y pasos. Lanza los entrenamientos con `python -u` y un límite de memoria (`ulimit -v`).

## Objetivo

Construir un world model latente que aprenda la dinámica del dron a partir de observaciones parciales, medir cómo crece su error al predecir en bucle abierto, comprobar qué guarda su estado latente y usarlo para imaginar. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a8a.py`, salida en `figuras/salida_gpu.txt`):

| Resultado de la presentación (una ejecución en GPU; varía entre ejecuciones) | Valor | Paso |
|---|---|---|
| Error de posición del RSSM (β_dyn = 1) en bucle abierto a 0,1 / 0,5 / 1 s | 8,5 / 28,2 / 63,2 cm (en otras ejecuciones, 59 y 75 cm a 1 s) | 3 |
| RSSM determinista | 7,9 / 40,0 / 81,6 cm | 3 |
| MLP de un paso; quieto en la última posición | 91,4 y 90,0 cm a 1 s | 3 |
| Correlación entre la dispersión de las muestras y el error, a 0,5 s | 0,14 | 3 |
| R² de la velocidad desde el estado del RSSM a 0,5 s | 0,94 · 0,94 · 0,92 | 4 |
| Ídem desde el RSSM determinista; desde dos observaciones | 0,81 · 0,77 · 0,16; 0,78 · 0,76 · 0,13 | 4 |
| R² de la masa a 3 s (RSSM) | 0,32 | 4 |

## 1. Datos (30 min)

1. Genera 3000 episodios de entrenamiento y 1000 de prueba de 3 s a 50 Hz: un controlador de posición hacia un punto aleatorio con ruido de exploración (Ornstein-Uhlenbeck), masa distinta en cada episodio (±15 %).
2. Observación: posición con 2 cm de ruido, matriz de rotación con 1°, giróscopo con 0,05 rad/s. Sin velocidad. Guarda también el estado verdadero, solo para evaluar.
3. Normaliza las observaciones con la media y la desviación del conjunto de entrenamiento.

## 2. El RSSM (75 min)

1. Camino determinista \(h_t = \mathrm{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])\); prior \(p(z_t\mid h_t)\) y posterior \(q(z_t\mid h_t, o_t)\) gaussianos; decodificador \(p(o_t\mid h_t, z_t)\).
2. Pérdida: reconstrucción más los dos términos de KL de DreamerV3 (\(\beta_{\text{dyn}} = 1\) sobre el prior y \(\beta_{\text{rep}} = 0{,}1\) sobre el posterior, con el otro lado detenido; la primera versión de arXiv usaba \(\beta_{\text{dyn}} = 0{,}5\): prueba las dos) y bits libres de 1 nat.
3. **Tests:** (a) con \(o_t\) constante y acciones nulas, el KL debe bajar hacia los bits libres; (b) el filtrado sobre una secuencia de prueba no debe depender del orden de los episodios del lote.
4. Entrena 6000 pasos con secuencias de 50 pasos y lotes de 64.

## 3. Bucle abierto y referencias (45 min)

1. Filtra 0,5 s con observaciones y predice 1 s solo con las acciones. Representa el error de posición frente al horizonte.
2. Compara con: la versión determinista del mismo modelo, una MLP de un paso sobre la última observación y «el dron se queda quieto».
3. Saca 20 muestras del RSSM por episodio: ¿la dispersión de las muestras se corresponde con el error en cada episodio? Representa una frente a otro.
4. Entrena con predicción a varios pasos (overshooting latente) y compara la curva de error.

## 4. ¿Qué hay en el estado latente? (30 min)

1. Sondeo lineal de la velocidad y de la masa desde \((h_t, \mu_q)\) a 0,5, 1,5 y 3 s, con episodios separados para ajustar y para probar.
2. Referencias: la última observación y dos observaciones (diferencia finita).
3. ¿Por qué la versión determinista guarda menos velocidad? Prueba a quitarle al posterior el acceso directo a la observación (o a reducir su dimensión) y repite.

## 5. Imaginar para actuar (opcional, 40 min)

1. Con el RSSM congelado, entrena un actor y un crítico solo en trayectorias imaginadas de 15 pasos desde estados filtrados, con retornos λ, para llevar el dron a un punto.
2. Evalúa la política en el simulador real. ¿Explota errores del modelo? ¿Cuánto mejora si reentrenas el modelo con los datos de la política?

## Entregables

- `modelos/rssm.py` con tests; `experimentos/a8a.py` que genere la curva de error, la tabla de sondeos y la figura de muestras.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Qué parte del belief de tu tarea de tesis podría aprender un world model, y cuál calcularías con un filtro explícito?
2. ¿Cómo comprobarías que la incertidumbre del modelo sirve para decidir dónde explorar?
3. Si tu modelo se equivoca un 30 % a 1 s, ¿qué horizonte de imaginación usarías para entrenar la política, y por qué?
