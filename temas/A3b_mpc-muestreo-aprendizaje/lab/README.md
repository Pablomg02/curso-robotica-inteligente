# Laboratorio A3b · MPPI vectorizado en JAX sobre el modelo del x500

**Duración:** unas 3,5 h (el paso 5 es opcional).
**Requisitos:** el modelo del x500 de A1, la cascada de A2 (lazo de *rate*) y el NMPC de A3a para comparar. Funciona en CPU; con una GPU NVIDIA de consumo, el paso 4 llega a decenas de miles de trayectorias por paso.

## Objetivo

Construir en el proyecto hilo un planificador por muestreo (CEM y MPPI) vectorizado en JAX, comprobar lo que mide la presentación y usarlo para que el x500 esquive obstáculos con mandos CTBR:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Éxito con \(K\) = 256 (20 semillas): aleatorio, CEM, MPPI | 55 %, 80 %, 95 % | 1 |
| Éxito con \(K\) = 1024 | 75 %, 90 %, 100 % | 1 |
| Tiempo medio a la meta con \(K\) = 1024: CEM frente a MPPI | 3,1 s frente a 5,4 s | 1 |
| Tamaño efectivo de muestra con 1024 trayectorias desde el inicio | 1 con \(\lambda \le 1\); 2,8 con 10; 611 con 100 | 2 |
| Ejemplo a mano de cinco muestras con \(\lambda\) = 1: pesos de MPPI y tamaño efectivo | 0,490 · 0,180 · 0,297 · 0,024 · 0,009; 2,77 | 1 |

El script de las figuras, en numpy, está en `../figuras/figuras_a3b.py`. Los ejemplos a mano de las hojas (cinco muestras, CEM, MPPI, temperatura, ruido correlado, modelo aprendido) están en `../figuras/ejemplos_a3b.py`. Úsalo como referencia: tu versión en JAX debe dar las mismas cifras (salvo el ruido de las semillas).

## 1. CEM y MPPI en el escenario 2D (60 min)

En `src/proyecto_hilo/planificacion/muestreo.py`:

1. `rollout(x0, U, params)`: simula en lote con `jax.vmap` sobre las \(K\) muestras y `jax.lax.scan` sobre el horizonte. Devuelve el coste de cada muestra. Escenario: punto con aceleración acotada (\(|a_i| \le 4\) m/s²), \(\Delta t\) = 0,05 s, horizonte de 30 pasos, de (0, 0) a (8, 0), con obstáculos en (3; 0,2) de radio 1 y en (5,5; −0,6) de radio 0,8, y el coste del script (distancia a la meta, mando, choque con 15 cm de margen, coste terminal).
2. `cem_paso` y `mppi_paso` con la misma firma `(clave, x0, U) -> U`: CEM con 4 iteraciones y el 10 % de élites; MPPI con una iteración y \(\lambda\) = 10. Calcula los pesos restando el mínimo, con `jax.nn.softmax(-S / lam)` (es lo mismo y es estable).
3. Bucle cerrado a 20 Hz con arranque en caliente (desplazar \(U\) un paso). Todo dentro de un `jax.jit`.

**Comprobaciones:** con 20 semillas, las tasas de éxito de la tabla con ±15 puntos. MPPI más fiable que CEM y CEM más rápido. Si MPPI te sale peor que el aleatorio, revisa el signo del coste en los pesos.

## 2. La temperatura (20 min)

1. Con 1024 muestras desde el inicio, representa el tamaño efectivo de muestra \(1/\sum_k w_k^2\) frente a \(\lambda\) de \(10^{-2}\) a \(10^3\). Compáralo con la figura de la presentación.
2. Barre \(\lambda\) en el bucle cerrado (1, 10, 100) con \(K\) = 256 y mide el éxito.
3. Implementa un \(\lambda\) adaptativo que fije un tamaño efectivo objetivo (por ejemplo, el 5 % de \(K\)) por bisección en cada paso. ¿Mejora la robustez frente al escalado del coste? Prueba a multiplicar todo el coste por 100.

## 3. MPPI sobre el x500 (60 min)

En `planificacion/mppi_x500.py`:

1. Modelo de planificación: el x500 de A1 **con el lazo de *rate* de A2 cerrado dentro**, de modo que el mando del planificador es CTBR (empuje colectivo y velocidades angulares). Malla de 0,05 s y horizonte de 1 s (20 pasos), con una sola consigna por paso.
2. Coste: distancia a un punto objetivo, velocidad terminal, penalización de mandos y un indicador de choque grande (\(10^4\)–\(10^6\)) con dos columnas cilíndricas en el camino.
3. Simulación de la planta a 1 kHz con tu simulador multifrecuencia de A2 y el planificador a 50 Hz.

**Mide:** éxito y tiempo en llegar con \(K\) = 256, 1024 y 4096; distancia mínima a los obstáculos; tiempo de cálculo por paso.

**Compara** con el NMPC de A3a en un vuelo sin obstáculos (seguir una lemniscata): ¿cuál sigue mejor, y a qué coste de cómputo? Recuerda lo que vieron Minařík et al.: en seguimiento puro, el controlador geométrico era más preciso que MPPI.

## 4. Escalado en GPU (20 min)

Mide el tiempo de un paso de MPPI del paso 3 con \(K\) de 64 a 16 384, en CPU y, si tienes, en GPU. Representa el tiempo frente a \(K\) en escala logarítmica. ¿Dónde está el codo en el que la GPU deja de ir gratis? ¿Cuántas muestras te caben en 10 ms (100 Hz)?

## 5. Modelo aprendido (opcional, 40 min)

1. Genera datos con tu simulador (vuelos aleatorios con el lazo de *rate*) y entrena un modelo de red pequeño de la dinámica CTBR (estado, mando) → siguiente estado, o un residuo sobre un modelo nominal sin arrastre.
2. Sustituye el modelo físico del planificador por el aprendido y repite el paso 3.
3. Busca el error: dibuja las trayectorias que el planificador cree que va a seguir frente a las reales. ¿Explota el planificador alguna región donde el modelo se equivoca? Entrena un conjunto de 5 modelos y penaliza su desacuerdo en el coste: ¿cambia el comportamiento?

## Entregables

- `planificacion/muestreo.py` y `planificacion/mppi_x500.py`, con tests: los pesos de MPPI suman 1 y son invariantes a sumar una constante al coste; con \(\lambda\to\infty\) la actualización tiende a la media del ruido; con las cinco secuencias del ejemplo a mano, `mppi_paso` devuelve \(U = (1{,}14;\ 0{,}50)\) con \(\lambda = 1\) y el CEM con dos élites, \(\mu = (1;\ 0{,}5)\), \(\sigma = (0;\ 0{,}5)\).
- `experimentos/a3b_muestreo.py`, que genere: éxito frente a \(K\) para los tres métodos, tamaño efectivo frente a \(\lambda\), trayectorias del x500 entre obstáculos y tiempo por paso frente a \(K\).
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. Tu planificador del paso 3 da CTBR a 50 Hz. ¿Qué ocurre si quitas el lazo de *rate* del modelo de planificación y planificas directamente sobre los mandos de motor? Piensa en la dimensión del muestreo y en la latencia (A2).
2. En el paso 5, ¿la incertidumbre que mide el conjunto de modelos es aleatoria o epistémica? ¿Qué harías distinto con cada una?
3. ¿Cómo cambiaría el planificador si el estado inicial no fuera conocido, sino un belief con covarianza? Esboza un MPPI que muestree también el estado inicial.
