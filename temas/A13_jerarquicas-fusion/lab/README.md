# Laboratorio A13 · Sistema lento y rápido con inferencia asíncrona

**Duración:** unas 3,5 h.
**Requisitos:** Python con numpy. Solo CPU: el barrido completo de la presentación tardó unos 5 minutos.

## Objetivo

Medir qué pasa cuando un planificador lento controla un dron directamente y cuando mueve la referencia de un seguidor rápido, en función de su latencia. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a13.py`, salida en `figuras/salida.txt`; las cuentas de las derivaciones, en `figuras/derivaciones_a13.py` y `figuras/ejemplos_a13.py`, con su salida en `figuras/salida_ejemplos.txt`):

| Resultado de la presentación (100 escenas de 20 s) | Valor | Paso |
|---|---|---|
| Una llamada a MPPI (256 trayectorias de 2 s) en CPU | 1,4 ms | 2 |
| Fracción del tiempo cerca, solo rápido (E) / choques | 0,52 / 0,69 | 3 |
| Cerca a 400 ms: A / B / C / D | 0,14 / 0,30 / 0,90 / 0,91 | 4 |
| Choques a 400 ms: A / B / C / D | 0,82 / 0,67 / 0,00 / 0,00 | 4 |
| Cerca a 800 ms: C / D | 0,57 / 0,38 | 5 |

Las cifras dependen del montaje (dinámica, métrica, árboles); lo que deberías reproducir es la forma de las curvas y la ausencia de choques con un lazo rápido.

## 1. Escenas y dron (40 min)

1. Plano de 40 × 40 m con 8 copas de árbol circulares que no se solapan y una persona que camina a 1,5 m/s con giros aleatorios (rebota en los bordes).
2. Dron a 100 Hz: aceleración pedida limitada a 6 m/s², seguida con una constante de tiempo de 0,08 s, y arrastre cuadrático que el planificador no conoce.

## 2. Planificador y seguidor (50 min)

1. MPPI con 256 trayectorias de 2 s (20 pasos de 0,1 s) sobre un doble integrador; coste: distancia a la persona prevista con velocidad constante, copas y esfuerzo. Arranque en caliente con el plan anterior desplazado.
2. Seguidor PD a 100 Hz de la posición, la velocidad y la aceleración de referencia del plan. Mide cuánto tarda una llamada a MPPI.

## 3. Latencia simulada y arquitecturas (50 min)

1. Simula una latencia δ: el plan pedido en t llega en t + δ y se vuelve a pedir en cuanto llega.
2. Implementa las cinco arquitecturas: solo lento, lento con trozos, jerárquica, jerárquica con compensación y solo rápido.

## 4. Barrido (30 min)

Barre δ = 50, 100, 200, 400 y 800 ms en 100 escenas y dibuja la fracción del tiempo cerca y los choques.

## 5. Mejorar la compensación (40 min)

La compensación (D) ayuda poco y empeora a 800 ms. Prueba otra forma de prever el estado (por ejemplo, con el modelo del dron y los mandos ya decididos) o haz que el planificador tenga en cuenta su propia latencia al predecir a la persona.

## Entregables

- `sim/escenas.py`, `sim/planificador.py`, `sim/arquitecturas.py` y `experimentos/a13.py`.
- La figura de fracción cerca frente a latencia y una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Qué latencia toleraría tu dron de búsqueda en el nivel del belief, y por qué?
2. ¿Qué pasaría si el seguidor rápido también tuviera latencia (por ejemplo, 35 ms de la controladora, como en Agilicious)?
3. ¿Cómo entrenarías una política rápida para que use bien planes que llegan tarde?
