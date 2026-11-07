# Laboratorio A14 · Filtro de seguridad CBF sobre una política agresiva

**Duración:** unas 3,5 h.
**Requisitos:** Python con numpy. Solo CPU: el experimento completo tardó unos 2 minutos.

## Objetivo

Construir un filtro de seguridad con funciones de barrera de control (CBF) para un dron que manda aceleraciones, y medir qué garantiza y qué cuesta cuando el estado, el modelo o el tiempo no son perfectos. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a14.py`, salida en `figuras/salida.txt`; los ejemplos hechos a mano y los datos de las figuras, en `figuras/ejemplos_a14.py`, salida en `figuras/salida_ejemplos.txt`):

| Resultado de la presentación (200 escenas de 12 s) | Valor | Paso |
|---|---|---|
| Choques sin filtro | 100 % | 1 |
| Choques / llegadas con filtro, α = 0,5 | 0 % / 0 % | 3 |
| Choques / llegadas con filtro, α = 2 | 21 % / 77 % | 3 |
| Choques / llegadas con margen de un paso, α = 2 | 0 % / 81 % | 3 |
| Ruido de 0,3 m: choques normal / con margen robusto | 28 % / 0 % | 4 |
| Viento: choques normal / robusto con w_max = 2,5 | 44 % / 0 % | 4 |
| Una llamada al filtro (12 restricciones, numpy) | 2,9 ms | 2 |

## 1. Escenas, dron y política nominal (30 min)

1. Campo de 30 × 20 m con 7 obstáculos circulares (radio de 0,8 a 1,8 m) y una zona prohibida de 3 m; dron como doble integrador con aceleración de hasta 6 m/s² por eje y paso de 10 ms.
2. Política nominal: un PD hacia la meta que no ve nada. Comprueba que choca casi siempre.

## 2. CBF de orden 2 y QP (60 min)

1. Para cada obstáculo, h = |p − c|² − r² y la condición 2 d·u + 2 |v|² + 2α ḣ + α² h ≥ 0 (con d = p − c).
2. Resuelve el QP (acción más parecida a la nominal que cumple todas las condiciones y los límites de la entrada) con el método de Hildreth, vectorizado sobre todas las escenas. Compara con la solución cerrada cuando solo hay una restricción.

## 3. Elegir α y el margen de un paso (40 min)

1. Barre α = 0,5, 1, 2 y 4: choques, llegadas, intervención. ¿Por qué con α pequeño nadie llega?
2. Mide cuánto penetran los «choques» con α grande e infla el radio con lo que se recorre en un paso.

## 4. Ruido y viento (40 min)

1. Añade ruido a la posición y la velocidad estimadas (0,1 y 0,3 m) y prueba un margen robusto (2σ más dos pasos).
2. Añade viento como una aceleración de Ornstein-Uhlenbeck y una CBF robusta que reste 2 |p − c| w_max. ¿Qué w_max necesitas?

## 5. Retardo (40 min)

Añade un retardo entre la medida y la actuación y evalúa la CBF en el estado predicho con las entradas ya enviadas (el método de Singletary, Chen y Ames). Sin viento, la predicción es exacta; con viento, ¿cuánto ayuda?

## Entregables

- `seguridad/cbf.py`, `seguridad/qp.py`, `experimentos/a14.py`.
- Una tabla con tus valores frente a los de la presentación y una figura con trayectorias con y sin filtro.

## Preguntas para cerrar

1. ¿Qué margen tendría que tener el filtro de tu dron con el estimador de A4a?
2. ¿Qué harías cuando el QP no tenga solución?
3. ¿Cómo evitarías que el filtro bloquee al dron delante de un obstáculo?
