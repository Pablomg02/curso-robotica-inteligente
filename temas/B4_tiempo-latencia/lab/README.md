# Laboratorio B4 · Multifrecuencia, búferes de retardo y jitter en el simulador JAX

**Duración:** unas 4 h (el paso 5 depende de la descarga; el 6 prepara A6a).
**Requisitos:** tu simulador JAX del x500 (B1, B2) con los modelos de sensor de B3a, el ESKF de A4a y la cascada de A2. Todo corre en CPU; con GPU, la vectorización del paso 1 se nota a partir de miles de entornos. Para el paso 5, `pyulog` (licencia BSD-3).

## Objetivo

Dar a tu simulador un reloj multifrecuencia y canales con retardo, jitter y pérdidas, medir qué hacen con la cascada y con el estimador, compararlo con latencias reales de registros de PX4 y dejar un entorno con retardos listo para el aprendizaje por refuerzo. Comprueba con tus números lo que calcula la presentación:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Cámara a 30 Hz sobre 1 ms: disparando cada 33 pasos frente a acumulando el instante | 1818 frente a 1800 imágenes por minuto | 1 |
| Edad media de la última imagen (30 Hz, 40 ms de latencia, control a 100 Hz) | 55 ms, entre 40 y 70 ms | 2 |
| Eje con LQR de 8 rad/s a 100 Hz sin predictor | inestable desde 90 ms (margen de fase 65,5°, cruce 12,1 rad/s) | 3 |
| Coste con predictor exacto a 50, 100 y 150 ms | ×1,74; ×2,87; ×4,50 | 3 |
| Predictor que supone 50 ms con 120 ms reales | ×17 | 3 |
| 20 % de pérdidas en ráfagas de 200 ms: predictor con 80 ms frente a predictor con la edad real | ×487 frente a ×4,8 | 4 |

El script de las figuras está en `../figuras/figuras_b4.py` (numpy, unos 30 s).

## 1. Reloj multifrecuencia (40 min)

En `src/proyecto_hilo/sim/reloj.py`:

1. Un paso base de física (1 ms) y una lista de tareas con su periodo: IMU (200 Hz), lazo de *rate* (500 Hz), actitud (250 Hz), posición (50 Hz), cámara (30 Hz) y GNSS (10 Hz). Cada tarea guarda su siguiente instante programado y le suma el periodo exacto al dispararse.
2. Implementa el bucle con `lax.scan` y un contador global `k`; cada tarea con `lax.cond(k_dispara, ...)`.
3. Vectoriza con `vmap` sobre 4096 entornos. Mide el tiempo por paso en dos versiones: (a) contador común, (b) fase de la cámara aleatoria por entorno. Explica la diferencia con lo que hace `vmap` con `lax.cond`.

**Comprobaciones:** intervalos de cámara de 33 y 34 ms con 33,345 ms de media; 1800 imágenes en 60 s simulados (y 1818 con la versión que cuenta pasos).

## 2. Búferes de retardo (45 min)

En `src/proyecto_hilo/sim/canales.py`, un canal con:

1. Búfer circular de `D` casillas a la frecuencia del canal, inicializado con el valor inicial.
2. Retardo fraccionario por interpolación lineal. Test: comprueba la respuesta en frecuencia de la derivación de la presentación (módulo \(|1-2\alpha|\) en Nyquist).
3. Jitter por medida con orden de llegada (FIFO) y pérdidas independientes y a ráfagas (cadena de dos estados). Test: la pérdida media y la longitud media de ráfaga coinciden con \(p_{gb}/(p_{gb}+p_{bg})\) y \(1/p_{bg}\).
4. Cada lectura devuelve el dato, su marca de tiempo de medida y su edad.
5. La edad de la cámara de la presentación: 30 Hz, 40 ms de latencia y control a 100 Hz. Reproduce la figura y la media de 55 ms.

## 3. Retardo, predictor y desajuste en tu cascada (50 min)

1. Reproduce el experimento del eje con `figuras_b4.py` y después repítelo en tu simulador completo: retrasa la posición y la velocidad que recibe el lazo de posición de tu cascada (A2).
2. Busca el retardo con el que el lazo se vuelve inestable sin predictor y compáralo con \(\text{PM}/\omega_c\) de tu lazo (A2).
3. Añade el predictor con la historia de acciones (la ecuación de la presentación, con tu modelo) y rehaz la tabla de desajuste: supón 50 y 100 ms con retardos reales de 0 a 150 ms.

**Piensa:** tu lazo de posición tiene un ancho de banda menor que el del ejemplo. ¿Cuánto retardo tolera? ¿Y el de actitud?

## 4. Marcas de tiempo en el ESKF (35 min)

Con el ESKF de A4a y la posición de referencia de un vuelo de EuRoC degradada como «GNSS» a 10 Hz con 100 ms de retardo:

1. Añade jitter FIFO de σ = 20 y 40 ms, y pérdidas a ráfagas del 20 % con ráfagas de 0,5 s.
2. Compara tres versiones: fundir al llegar como si la medida fuera de ahora; fundir con el retardo nominal; fundir en el instante de la marca de tiempo (horizonte retrasado o reprocesado).
3. Mide RMSE de posición y NEES.

**Comprueba:** ¿cuál de las tres mantiene el NEES dentro de su intervalo de confianza con jitter y pérdidas? ¿Cuánto se alejan las otras dos, y en qué dirección (sobreconfianza o subconfianza)?

## 5. Latencias reales en registros de PX4 (40 min)

1. La lista de registros públicos de Flight Review está en `https://logs.px4.io/dbinfo` (redirige a un JSON comprimido de unos 33 MB): guárdalo en su propia carpeta de datos. Elige tres registros de multirrotor recientes (versión de PX4 1.15 o posterior) y descárgalos uno a uno con `https://logs.px4.io/download?log=<id>`, cada uno en su carpeta. Son datos externos: léelos con `pyulog` desde un script que esté en otra carpeta.
2. Para cada registro: la distribución de `timestamp − timestamp_sample` en `actuator_motors` (de la muestra del giróscopo a la orden a los motores) y en `vehicle_angular_velocity` (filtrado). El registrador guarda `actuator_motors` cada 100 ms por defecto: tendrás pocas muestras por segundo, suficientes para una distribución.
3. Si el registro trae visión externa u odometría (`vehicle_visual_odometry`), compara su `timestamp_sample` con `timestamp` para ver cuánto tarda en llegar a la controladora.
4. Correlación cruzada entre la consigna de velocidad angular y el giróscopo en un tramo con maniobras: retardo total del lazo.

**Entregable:** una tabla de latencias medidas (mediana y percentil 95) que alimente los rangos de aleatorización del paso 6.

## 6. Un entorno con retardos para el aprendizaje (30 min)

En `src/proyecto_hilo/rl/entorno_retardos.py`, un envoltorio de tu entorno con:

- canales de observación y acción con retardo, jitter y pérdidas, sorteados por episodio en rangos que cubran con holgura lo medido en el paso 5;
- observación del actor: cada medida con su edad y la historia de acciones de al menos el retardo máximo;
- observación del crítico: el estado verdadero y actual;
- un modo «sin retardos» para comparar.

Pruébalo con un controlador LQR con y sin predictor. En A6a entrenarás PPO sobre él; en A6b, el esquema de actor y crítico asimétricos.

## Entregables

- `sim/reloj.py`, `sim/canales.py` y `rl/entorno_retardos.py`, con tests (intervalos del reloj, respuesta en frecuencia del retardo fraccionario, estadística de las pérdidas, orden FIFO).
- `experimentos/b4_retardos.py`, que genere: la edad de la cámara, el coste frente al retardo con y sin predictor, la tabla de desajuste y la de marcas de tiempo, y la tabla de latencias reales.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. Con lo medido en el paso 5, ¿qué parte del retardo total de tu dron viene del cómputo, cuál de los sensores y cuál de los motores? ¿Cuál puedes reducir?
2. Tu política de tesis recibirá imágenes con una edad entre 40 y 100 ms. ¿Le das la edad como entrada, la compensas con un predictor o dejas que el belief recurrente aprenda? Argumenta con los experimentos.
3. ¿Qué cambiaría en el paso 6 si las observaciones pudieran llegar fuera de orden?
