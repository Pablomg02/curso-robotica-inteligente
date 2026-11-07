# Laboratorio A4a · Kalman, ESKF con datos reales de EuRoC y filtro de partículas

**Duración:** unas 3,5 h (el paso 5 es opcional).
**Requisitos:** `dinamica/so3.py` de M1 (Exp, Log, hat) y el simulador de A1–A2. Todo en CPU.

## Objetivo

Añadir al proyecto hilo un módulo `estimacion/` con filtro de Kalman, ESKF y filtro de partículas, comprobar con tus números lo que calcula la presentación y probar el ESKF con datos reales:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| NEES medio (200 ensayos, velocidad constante, posición a 10 Hz) | 1,86 bien ajustado; 95 con \(Q\) cien veces menor | 1 |
| RMSE en el modelo bimodal (50 ensayos): EKF, UKF, PF 50, PF 500 | 20,6; 11,1; 5,6; 4,7 | 2 |
| ESKF de actitud sin magnetómetro, error inicial de 10° en guiñada | error sin corregir (de 1° a 28° a los 120 s según la semilla) con σ del filtro de unos 3,2° | 3 |
| Kalman estacionario de un eje con acelerómetro: ganancias | \((\sqrt2\,\omega_0, \omega_0^2)\), \(\omega_0 = (q/r)^{1/4}\) | 1 |

El script de las figuras está en `../figuras/figuras_a4a.py`. Úsalo para comparar al final de cada paso. Los ejemplos que la presentación hace a mano (rejilla de cinco celdas, Joseph, \(Q_d\), forma de información, NEES, puntos sigma, pesos de partículas, jacobianos del ESKF) están en `../figuras/ejemplos_a4a.py`, con su salida en `../figuras/salida_ejemplos.txt`: sirven como tests unitarios de tus funciones.

## 0. Datos (20 min, se puede hacer antes)

Descarga una secuencia de EuRoC (por ejemplo `MH_01_easy` o `V1_01_easy`) desde el ETH Research Collection, [doi:10.3929/ethz-b-000690084](https://doi.org/10.3929/ethz-b-000690084). Descomprímela en su propia carpeta de datos, fuera del código.

Antes de escribir una línea de filtro, lee en la carpeta `mav0/`:
- `imu0/sensor.yaml`: frecuencia de la IMU, densidades de ruido y paseo aleatorio de giróscopo y acelerómetro, y la transformación cuerpo–IMU;
- `imu0/data.csv`: unidades y orden de las columnas;
- `state_groundtruth_estimate0/data.csv`: la referencia (posición, cuaternión, velocidad y sesgos) que usarás para evaluar y para fabricar medidas de «GNSS».

Anota el convenio del cuaternión de los ficheros y conviértelo al del proyecto hilo (Hamilton w, x, y, z; mundo ENU).

## 1. Kalman y consistencia (40 min)

En `estimacion/kalman.py`:

1. Un filtro de Kalman genérico (predicción, actualización con la forma de Joseph) y las funciones `nees(e, P)` y `nis(nu, S)`.
2. El experimento de la presentación: velocidad constante, \(q\) = 0,5 m²/s³, posición con σ = 0,5 m a 10 Hz, \(Q_d\) de la hoja de ruido de proceso. 200 ensayos de 10 s.
3. Repite con \(Q\) cien veces menor y cien veces mayor.

**Comprobaciones:** NEES medio dentro de 1,73–2,29 (1,86 en el script); con \(Q/100\), del orden de 95. Con \(Q\cdot100\), ¿qué pasa con el NEES y con el error real?

4. Kalman de un eje con el acelerómetro como entrada y una posición ruidosa: integra la Riccati continua hasta el estacionario y comprueba que \(K = (\sqrt2\,\omega_0, \omega_0^2)\) con \(\omega_0 = (q/r)^{1/4}\).

## 2. EKF, UKF y partículas (40 min)

En `estimacion/no_lineal.py`, para el modelo \(x_k = x/2 + 25x/(1+x^2) + 8\cos(1{,}2k) + w\) (\(Q\) = 10), \(y = x^2/20 + v\) (\(R\) = 1):

1. EKF con jacobianos analíticos (o `jax.jacfwd`).
2. UKF con \(\alpha\) = 1, \(\beta\) = 0, \(\kappa\) = 2.
3. Filtro de partículas *bootstrap* con remuestreo sistemático en cada paso, con 50 y 500 partículas, vectorizado con `jax.vmap` o numpy.

**Comprobaciones:** RMSE medio de 50 ensayos parecido a la tabla: ±25 % para EKF y UKF, cuya media varía mucho entre conjuntos de semillas, y ±10 % para el filtro de partículas (o usa 200 ensayos). Representa un ensayo con el histograma de partículas en tres instantes: ¿ves los dos modos \(\pm x\)?

## 3. ESKF de actitud (40 min)

En `estimacion/eskf_actitud.py`: estado nominal \((R, \mathbf b_g)\), error \((\delta\boldsymbol\theta, \delta\mathbf b_g)\), perturbación por la derecha, IMU a 200 Hz y actualizaciones a 50 Hz con la gravedad (acelerómetro) y, opcionalmente, un vector magnético.

1. Implementa propagación, actualización con \(H = [\,[\hat R^\top\mathbf v]_\times\ \ 0\,]\), inyección y reinicio.
2. Reproduce la figura de la guiñada: error inicial (3°, −3°, 10°), con y sin magnetómetro, 120 s.

**Comprobaciones:** con magnetómetro, error de guiñada por debajo de 1°; sin magnetómetro, el error no disminuye (en 6 semillas acaba entre 1° y 28°) mientras la σ del filtro baja a unos 3° en todas. Calcula el NEES de la actitud en los dos casos, promediando varias semillas.

3. Prueba un remedio: con solo acelerómetro, no dejes que la covarianza de la guiñada en el mundo baje (por ejemplo, proyectando la ganancia para que no corrija esa dirección). ¿Se vuelve consistente?

## 4. ESKF de 15 estados con EuRoC (60 min)

En `estimacion/eskf_ins.py`, el filtro de la hoja «ESKF con IMU»: posición, velocidad, actitud y sesgos de acelerómetro y giróscopo, con las densidades de ruido de `sensor.yaml` convertidas a varianzas por paso.

1. Propaga con la IMU de EuRoC. Inicializa con la referencia en \(t_0\) (posición, actitud y velocidad), sesgos a cero.
2. «GNSS»: toma la posición de la referencia a 10 Hz, súmale ruido gaussiano de 0,3 m y retrásala 100 ms.
3. Funde las posiciones de dos formas: (a) ignorando el retardo; (b) con horizonte retrasado o reprocesando desde el instante de la medida.
4. Añade una prueba de innovación con el umbral \(\chi^2_3\) del 99 % (11,34) y mete a mano algunas medidas atípicas de 5 m.

**Mide:** error de posición y de actitud frente a la referencia, NEES de posición, sesgos estimados frente a los de la referencia, y cuántas medidas atípicas rechaza la prueba. ¿Cuánto empeora (a) frente a (b) en los tramos rápidos?

## 5. Filtro de partículas para el mismo problema (opcional, 30 min)

Con la actitud de la referencia como conocida, el problema se reduce a posición y velocidad (6 estados) con la aceleración en el mundo como entrada. Implementa un filtro de partículas y compáralo con el Kalman del mismo problema: error, NEES y tiempo de cálculo con 100, 1000 y 10 000 partículas. ¿Cuántas necesitas para igualar al Kalman? Es óptimo solo para su modelo lineal gaussiano: con la IMU real (sesgos no modelados, errores no gaussianos) no tiene por qué ganar.

## Entregables

- `estimacion/kalman.py`, `no_lineal.py`, `eskf_actitud.py` y `eskf_ins.py`, con tests: un Kalman con \(Q = 0\), medidas sin ruido y \(R > 0\) pequeño en el filtro converge al estado real (asintóticamente, como \(1/k\); fija una tolerancia); el ESKF con la IMU sin ruido ni sesgos reproduce la actitud de la referencia; los pesos del filtro de partículas suman 1 tras el remuestreo.
- `experimentos/a4a_filtros.py`, que genere: NEES con distintos \(Q\), comparación no lineal, figura de la guiñada y error del ESKF con EuRoC con y sin compensar el retardo.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. Tu dron de tesis tendrá un estimador clásico (actitud, sesgos, velocidad) y un belief aprendido (entorno, tarea). ¿Qué le pasas de uno al otro: la media, la covarianza o las medidas crudas?
2. En el paso 3, el EKF se cree que sabe la guiñada. ¿Qué haría un planificador que confía en esa covarianza?
3. ¿Qué cambiaría en el paso 4 si la posición llegase de una cámara a 20 Hz con 50 ms de retardo variable?
