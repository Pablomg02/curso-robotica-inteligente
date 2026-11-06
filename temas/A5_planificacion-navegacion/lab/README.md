# Laboratorio A5 · Minimum snap y planificador local con obstáculos

**Duración:** unas 3,5 h (el paso 5 es opcional).
**Requisitos:** el modelo del x500 de A1, la cascada de A2 y, para el paso 4, el ESKF de A4a si quieres volar con el estado estimado. Todo en CPU, en segundos; la GPU solo ayuda si vectorizas muchas semillas o muchos planes con `jax.vmap`.

## Objetivo

Añadir al proyecto hilo un módulo `planificacion/` con búsqueda, generación de trayectorias y un planificador local que replanifica con un sensor de alcance limitado, y comprobar con tus números lo que calcula la presentación:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| A* en rejilla 8-conexa de 0,1 m y camino recortado por visibilidad | 21,65 m (6000 nodos expandidos); 21,07 m | 1 |
| Mejor camino de RRT y RRT* (semilla 0; 500 / 1000 / 3000 iteraciones) | 26,94 / 26,94 / 26,94 m; 24,40 / 23,98 / 23,98 m | 1 |
| Coste de snap con 2 s por tramo, \(T\) = 6, 8 y 12 s | 8483; 1132; 66,3 (cociente 128) | 2 |
| Reparto de tiempos optimizado con \(T\) = 8 s | (2,87; 1,13; 1,13; 2,87) s, coste 3,1 veces menor, \(a_\max\) de 5,96 a 4,34 m/s² | 2 |
| Optimización local sobre el campo de distancias | la recta entra 1,58 m en un obstáculo; la optimizada queda a 0,55 m y mide 21,6 m | 3 |
| Distancia de parada con 0,1 s de latencia y \(a = g\tan30°\) | 2,7 m a 5 m/s; 9,8 m a 10 m/s | 4 |
| Caminos de Dubins con \(R\) = 110 m, de (0, 0, 0°) a (400, 150, 180°) | RSL 753 m; LSR 911 m; LRL 1217 m | 5 |

El script de las figuras, en numpy, está en `../figuras/figuras_a5.py`. Úsalo como referencia: tu versión debe dar las mismas cifras, salvo las de RRT y RRT*, que dependen de la semilla y de los detalles del muestreo.

## 1. A*, RRT y RRT* (45 min)

En `src/proyecto_hilo/planificacion/busqueda.py`:

1. El mapa de la presentación: 20 m × 10 m, obstáculos circulares en (5, 3) con radio 1,8; (9, 7) con 2,0; (13; 3,5) con 2,2 y (16,5; 7,5) con 1,5; margen de 0,3 m; de (1, 1) a (19, 9).
2. A* en una rejilla 8-conexa de 0,1 m con heurística euclídea. Cuenta los nodos expandidos. Repite con Dijkstra (\(h = 0\)) y con \(h\) inflada por 1,5: ¿cuántos nodos expande cada uno y cuánto se alarga el camino?
3. Recorte por visibilidad: une cada vértice con el más lejano que se vea sin chocar.
4. RRT y RRT* con paso de 1 m, 5 % de muestras en la meta y radio de vecinos \(\min(2{,}5;\ 6\sqrt{\log n/n})\). En RRT*, al recablear un nodo, propaga el nuevo coste a todos sus descendientes.

**Comprobaciones:** A* y el recorte, como en la tabla (deben salir iguales). Con 20 semillas, representa la mediana y los percentiles 10 y 90 del coste frente a las iteraciones para RRT y RRT*. ¿Converge RRT* hacia los 21,07 m? ¿A qué velocidad?

## 2. Minimum snap (60 min)

En `planificacion/trayectorias.py`, en JAX:

1. `q_snap(T)` y `fila(k, tau, T)` en tiempo normalizado \(\tau = t/T\) (es mucho mejor condicionado que usar \(t\) directamente). Polinomios de grado 7, continuidad impuesta hasta el snap y reposo en los extremos.
2. Resuelve el sistema KKT para los waypoints de la presentación: x = (0, 4, 8, 12, 16), y = (0, 3, 0, −3, 0), con 2 s por tramo. Comprueba que las restricciones se cumplen y que las derivadas 5 y 6 salen continuas sin imponerlo.
3. Implementa la forma cerrada de Richter et al. (derivadas libres \(\mathbf d_P^\star = -R_{PP}^{-1}R_{FP}^\top\mathbf d_F\)) y comprueba que da la misma trayectoria.
4. Ley de escala: \(T\) = 6, 8 y 12 s. Después, con \(T\) = 8 s, optimiza el reparto de tiempos con `jax.grad` del coste respecto a las duraciones (con \(T_i = e^{\tau_i}\) y la suma fija, o con una penalización \(k_T\sum T_i\) como en Richter et al.).
5. Factibilidad: escala \(T\) para que la aceleración horizontal no pase de 5 m/s² y la inclinación no pase de 30°.

**Comprobaciones:** costes de la tabla con 4 cifras; cociente 128; reparto óptimo cerca de (2,87; 1,13; 1,13; 2,87) s. Si el KKT te da restricciones incumplidas con tramos cortos, estás usando \(t\) sin normalizar.

## 3. Planificador local sobre un campo de distancias (40 min)

En `planificacion/local.py`:

1. Rasteriza el mapa en una rejilla de 0,05 m y calcula el campo de distancias con signo con `scipy.ndimage.distance_transform_edt` (distancia fuera menos distancia dentro). Interpólalo bilinealmente para tener \(d(\mathbf p)\) y \(\nabla d\).
2. Optimiza 40 puntos con el coste de la presentación (suavidad + colisión con \(d_s\) = 0,6 m y \(w\) = 10), con gradiente de `jax.grad` y descenso por gradiente con paso 0,02, desde la línea recta.
3. Arranca desde el camino recortado de A* en vez de la recta. ¿Cambia el lado por el que se esquiva cada obstáculo?

**Comprobaciones:** con el campo analítico (círculos), las cifras de la tabla; con el rasterizado, parecidas (±5 cm). Mide el tiempo por iteración y compáralo con lo que cuesta calcular el campo: ¿qué domina?

## 4. Bucle cerrado con replanificación (60 min)

En `experimentos/a5_navegacion.py`:

1. El x500 de A1 con la cascada de A2 a frecuencias reales. Sigue la trayectoria de minimum snap con *feedforward* de planitud: posición, velocidad y aceleración de referencia al lazo de posición, y la guiñada fija.
2. Un sensor de profundidad 2D simulado: rayos en un abanico de 90° con alcance \(r_s\) y ruido gaussiano que crezca con el cuadrado de la distancia. Con él, un mapa de ocupación log-odds local.
3. Replanifica a 10 Hz con el planificador local del paso 3 sobre el mapa que va construyendo, tratando lo desconocido como libre, pero exigiendo que exista una trayectoria de frenado dentro del espacio visto.
4. Un bosque aleatorio de columnas (por ejemplo, 1 árbol cada 20 m² con radios de 0,2 a 0,5 m) y velocidades de crucero de 2 a 10 m/s.

**Mide:** tasa de éxito con 20 bosques por velocidad, distancia mínima a los obstáculos y error de seguimiento. Representa la tasa de éxito frente a la velocidad para \(r_s\) = 5, 10 y 20 m. ¿Dónde cae cada curva? Compáralo con la distancia de parada \(v t_\ell + v^2/2a\).

## 5. Dubins y viento (opcional, 30 min)

1. Los seis caminos de Dubins en forma cerrada; comprueba cada uno integrando el camino y midiendo el error de cierre.
2. El ejemplo de la presentación y la tabla del radio mínimo frente a la velocidad y el alabeo.
3. Con viento de 10 m/s, vuela el camino RSL con un guiado L1 de A2 sobre el suelo y con el radio de aire en calma. ¿Dónde se sale? Repite planificando con \(R = (V+W)^2/(g\tan\phi)\), el peor caso.

## Entregables

- `planificacion/busqueda.py`, `trayectorias.py` y `local.py`, con tests: A* con \(h = 0\) da el mismo coste que Dijkstra; la trayectoria de minimum snap pasa por todos los waypoints con derivadas continuas hasta la 6; el coste escala como \(T^{-7}\); el gradiente de colisión apunta hacia fuera del obstáculo.
- `experimentos/a5_navegacion.py`, que genere: coste de RRT y RRT* frente a las iteraciones, trayectorias y perfiles de velocidad con los dos repartos de tiempos, y éxito frente a velocidad para tres alcances de sensor.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. En el paso 4, ¿qué te limita más la velocidad: el alcance del sensor, la latencia o la deceleración? ¿Cómo lo cambiaría una cámara de eventos (A4b)?
2. Tratar lo desconocido como libre o como ocupado: ¿cómo lo decidiría un planificador que razona sobre el belief del mapa, en lugar de sobre el mapa más probable?
3. Si tu world model da un campo de distancias con incertidumbre (media y varianza por punto), ¿cómo cambiarías el término de colisión del paso 3?
