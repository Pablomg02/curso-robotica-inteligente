# Laboratorio A6b · Privilegio, imitación y espacios de acción con tu x500

**Duración:** unas 4 h.
**Requisitos:** tu PPO en JAX de A6a y tu simulador vectorizado con aleatorización (B2–B4). Una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`) o en la nube; en CPU funciona con menos entornos (por ejemplo, 512) y bastante más tiempo. La Pixhawk solo para el HIL opcional del paso 6: nunca se vuela.

## Objetivo

Entrenar una política cuya entrada sea solo lo que tendría el dron en vuelo, aprovechando en el entrenamiento todo lo que sabe el simulador, y comprobar con varias semillas qué funciona. Comprueba con tus números lo que calcula la presentación (una ejecución de `figuras/figuras_a6b.py` en una RTX 4060 Ti, unos 15 min; en GPU los resultados cambian algo entre ejecuciones aunque la semilla sea la misma):

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Actor con la observación realista y crítico simétrico, 5 semillas | 87 % de choques; 152 cm de error en los que no chocan | 2 |
| Lo mismo con crítico asimétrico (estado y parámetros) | 0,2 % de choques; 21 cm (de 8 a 32 según la semilla) | 2 |
| Maestro privilegiado | 0 % de choques; 1,7 cm | 3 |
| Alumno por clonación, 3 semillas | pérdida de imitación 0,0013; 100 % de choques | 3 |
| Alumno por DAgger, 3 semillas | pérdida 0,12–0,13; 0–9 % de choques; 24–66 cm | 3 |
| CTBR y SRT con 20 ms de retardo de acción no modelado | error entre 4 y 13 veces el nominal | 5 |
| Término de suavidad con ruido de sensores | actividad de los rotores 1,6 veces menor (CTBR) y 3,2 veces menor (SRT) | 5 |
| CTBR con término de suavidad | 3 de 5 semillas oscilan a 5–7 Hz sin ruido | 5 |
| Filtro paso bajo con τ = 20 ms | casi 20 ms de retardo por debajo de unos 5 Hz | 5 |

## 1. Observación realista (30 min)

1. Añade a tu entorno la observación del actor de la presentación: error de posición con ruido de 2 cm y 40 ms de retardo, matriz de rotación con 1° de ruido, giróscopo con 0,05 rad/s, la acción anterior y la historia de las 4 últimas. Sin velocidad.
2. Y la privilegiada: estado verdadero y parámetros físicos (masa ±20 %, ganancia de empuje ±10 %, constante del motor de 20 a 60 ms, distintos en cada entorno).
3. **Test:** con el ruido a cero y sin retardo, la observación realista debe contener exactamente la posición y la actitud verdaderas.

## 2. Crítico simétrico frente a asimétrico (40 min)

1. Entrena 5 semillas con el crítico sobre la observación del actor y 5 con el crítico sobre la privilegiada. Mismo PPO, misma red.
2. Evalúa desde 400 arranques aleatorios: choques y error en régimen. Representa las curvas de aprendizaje de las 10 ejecuciones.
3. ¿Se acerca el simétrico si le das una red más grande o una historia más larga? Prueba una de las dos.
4. Calcula, para una política entrenada, el error cuadrático del crítico frente al retorno real en cada caso.

## 3. Maestro y alumno (40 min)

1. Entrena un maestro que ve la observación privilegiada.
2. Clonación: datos solo con el maestro volando (con algo de ruido en la acción) y regresión de sus acciones. DAgger: después de la primera iteración vuela el alumno y el maestro etiqueta.
3. Mide la pérdida de imitación en dos sitios: en los estados del maestro y en los del alumno. ¿Cuál predice si vuela?
4. Compara el alumno de DAgger con la política del paso 2 (crítico asimétrico): error, choques y tiempo total de cálculo.

## 4. Brecha de imitación (30 min)

1. Cambia la tarea: el dron sale del centro y la consigna está a 2 m a la izquierda o a la derecha, al azar; el actor no la ve hasta estar a menos de 0,5 m (la «encuentra»), el maestro sí.
2. Destila el maestro con DAgger. ¿Qué hace el alumno al principio de cada episodio? Compáralo con \(\mathbb E[\pi^*(s)\mid o]\).
3. Entrena con PPO y crítico asimétrico sobre la misma tarea, con \(V(s)\) y con \(V(h, s)\) (el crítico recibe también la historia del actor). ¿Aprende a buscar? ¿Cuál aprende antes?

## 5. Espacios de acción y suavidad (40 min)

1. Entrena con CTBR y con empujes de rotor (SRT), con y sin el término \(0{,}1\,\|\Delta a\|^2\) en el coste, 5 semillas por variante.
2. Evalúa en nominal, con el ruido de sensores en la entrada y con 20 ms de retardo de acción no modelado. Mide la actividad de los rotores (\(\sum|\Delta f|\) por paso).
3. Implementa el término espacial de CAPS (\(\|\pi(s) - \pi(\bar s)\|\), \(\bar s\sim\mathcal N(s, \sigma)\) con el ruido de sensores) y compáralo.
4. Pon un filtro paso bajo de primer orden con \(\tau\) = 20 ms en la acción de la política sin suavidad. ¿Qué pasa con el error? Entrena con el filtro incluido y compara.

## 6. Hacia PX4 (opcional, 30 min)

1. Escribe la conversión de tu acción CTBR a `VehicleRatesSetpoint`: velocidades angulares en FRD y empuje normalizado (`thrust_body[2]` negativo), con la curva de empuje que identificaste en B2.
2. Escribe la conversión de la estimación de PX4 (NED, FRD) a la observación de tu política.
3. **Test:** un estado y una acción conocidos, convertidos ida y vuelta, deben dar lo mismo.
4. Si tienes el entorno de SITL de B5, prueba la política en estacionario y comprueba qué hace PX4 si tu nodo deja de publicar.

## Entregables

- `rl/observaciones.py` (realista y privilegiada, con tests), `rl/ppo.py` con la opción de crítico asimétrico, `rl/dagger.py`.
- `experimentos/a6b.py`, que genere las tablas de los pasos 2, 3 y 5 y las curvas del paso 2.
- Una tabla con tus valores frente a los de la presentación, y una nota sobre lo que hace el alumno en el paso 4.

## Preguntas para cerrar

1. En tu tarea de tesis, ¿qué parte de lo oculto se deduce de la historia en poco tiempo y qué parte solo se descubre buscando? ¿Qué crítico usarías para cada una?
2. Si tuvieras un maestro que conoce la posición del objetivo, ¿cómo lo usarías sin que el alumno herede la brecha de imitación?
3. ¿Qué retardo de punta a punta esperas en tu sistema (percepción, estimación, política, autopiloto)? ¿Cómo lo meterías en el entrenamiento?
