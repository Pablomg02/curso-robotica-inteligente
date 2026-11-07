# Laboratorio A10 · Destilar un MPPI en una política de flow matching

**Duración:** unas 3,5 h.
**Requisitos:** JAX con optax. Con una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`), el script entero tarda unos 2 minutos. En CPU funciona igual, pero el experto MPPI (30 720 pasos de la física por decisión) es lento: genera menos demostraciones (`A10_RAPIDO=1`). Usa un límite de memoria (`ulimit -v`) y no cierres un `jax.jit` sobre arrays grandes: pásalos como argumentos.

## Objetivo

Imitar a un experto caro y bimodal con alumnos baratos y medir dos cosas por separado: si llegan (éxito) y si reproducen las distintas maneras de llegar del experto (reparto de lados). Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a10.py`, salida en `figuras/salida_gpu.txt` y, con `A10_Y0=0.15`, en `figuras/salida_gpu_y015.txt`):

| Resultado de la presentación (determinista en la misma GPU) | Valor | Paso |
|---|---|---|
| Experto (MPPI): reparto de lados en las 300 demostraciones | 48 % izquierda / 52 % derecha | 1 |
| BC: éxito y reparto de lados en 200 vuelos de prueba | 100 %; 0 / 100 | 2 |
| Flow matching (trozos de 8, 10 pasos de Euler): éxito y reparto | 100 %; 51 / 49 | 3 |
| DAgger con BC: éxito y reparto | 99 %; 99 / 1 | 4 |
| Con ±15 cm de desviación inicial (`A10_Y0=0.15`), BC: éxito y reparto | 100 %; 49 / 51 | 2 |
| Tiempo por paso: experto / BC; flow matching por trozo | 10,0 / 0,021 ms; 0,081 ms | 5 |

## 1. El experto (40 min)

1. Reutiliza la física del x500 y el MPPI de A8b (ruido correlado, élites, 3 iteraciones). Añade un poste de 0,6 m de radio en el origen y un término de coste que castigue acercarse a menos de 0,95 m de su eje.
2. Genera 300 demostraciones desde \(x = -2{,}5\) m con \(|y_0| \le 2\) cm. ¿Qué fracción rodea el poste por cada lado? ¿Qué decide el lado?
3. Guarda, en cada paso, la acción ejecutada y las 8 primeras acciones del plan.

## 2. BC y BC con trozos (40 min)

1. Una red de 3 capas de 256 unidades que predice \(a_t\) con error cuadrático. Despliégala en 200 vuelos nuevos: éxito y reparto de lados.
2. La misma red prediciendo 8 acciones; ejecuta 4 y vuelve a predecir. ¿Cambia el reparto?
3. Repite con \(|y_0| \le 15\) cm. Explica la diferencia con la multimodalidad de \(p(a\mid o)\).

## 3. Flow matching (50 min)

1. Normaliza las acciones. Entrena \(v_\theta(x_t, t, o)\) con \(x_t = (1-t)x_0 + t x_1\) y objetivo \(x_1 - x_0\), con \(x_1\) un trozo de 8 acciones.
2. Muestrea con 10 pasos de Euler. Mide éxito, reparto y actividad \(\sum|\Delta a|\). Prueba con 1, 3 y 30 pasos.
3. Opcional: añade ensamblado temporal (ACT) y compara la actividad.

## 4. DAgger (40 min)

1. Despliega el alumno con el experto corriendo a la sombra (MPPI con su plan caliente desde el estado del alumno) y añade sus etiquetas a los datos. 4 rondas de 100 despliegues.
2. ¿Cambia el éxito? ¿Cambia el reparto de lados? Hazlo con BC y con flow matching.

## 5. Tiempos y destilación (30 min)

1. Mide el tiempo por decisión del experto, de BC y del flow matching (por trozo) en GPU y en CPU.
2. Mueve el poste 0,5 m o cambia la meta: ¿qué alumnos siguen funcionando? ¿Y el experto?

## Entregables

- `imitacion/experto.py`, `imitacion/alumnos.py` (BC, trozos, flow matching) y `experimentos/a10.py`.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Qué decisiones de tu búsqueda con belief tienen varias respuestas igual de buenas?
2. ¿Qué tamaño de trozo y qué frecuencia de nueva inferencia elegirías para un dron a 50 Hz, y por qué?
3. ¿Qué observaría tu alumno a bordo, y qué brecha de imitación (A6b) esperas frente al experto con belief exacto?
