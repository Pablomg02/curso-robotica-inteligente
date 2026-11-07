# Laboratorio A15b · Compañeros desconocidos en la búsqueda

**Duración:** unas 4 h.
**Requisitos:** numpy para Vicsek (CPU, unos 3 minutos) y JAX con optax para los drones. Con una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`), `figuras_a15b.py` tarda unos 36 minutos (14 entrenamientos de PPO). En CPU, usa `A15B_RAPIDO=1` o menos actualizaciones (`A15B_UPD`). Usa un límite de memoria (`ulimit -v`) y pasa los parámetros de las redes como argumentos de las funciones compiladas.

## Objetivo

Medir qué le pasa a un equipo de drones aprendido cuando cambia lo que dio por hecho al entrenar: sus compañeros, el número de drones que funcionan y la fiabilidad y puntualidad de los mensajes. Probar dos remedios. Comprueba con tus números lo que calcula la presentación (`figuras/vicsek_a15b.py` y `figuras/figuras_a15b.py`, salidas en `figuras/salida_vicsek.txt` y `figuras/salida_gpu.txt`; los ejemplos a mano de boids, Vicsek, consenso, ORCA, CBBA, other-play y el mentiroso salen de `figuras/ejemplos_a15b.py`, numpy en segundos, con su salida en `figuras/salida_ejemplos.txt`):

| Resultado de la presentación (determinista en la misma GPU) | Valor | Paso |
|---|---|---|
| Vicsek, 400 partículas: orden con ruido 0, 2 y 4 rad | 1,000 / 0,782 / 0,199 | 1 |
| Sin comunicación: autojuego / juego cruzado / peor pareja | 0,894 / 0,879 / 0,859 | 2 |
| Con comunicación: autojuego / juego cruzado | 0,910 / 0,910 | 2 |
| MAPPO con comunicación: normal / un dron cae / un compañero miente / mensajes de hace 10 pasos | 0,910 / 0,864 / 0,898 / 0,879 | 3 |
| Ídem con entrenamiento variado | 0,909 / 0,863 / 0,898 / 0,894 | 4 |
| Dron 0 con compañeros nuevos: autojuego / entrenado con la población | 0,886 / 0,887 | 5 |

## 1. Vicsek (30 min)

1. Implementa el modelo con caja periódica, densidad 4 y radio 1. Parámetro de orden frente al ruido para 100 y 400 partículas.
2. Cambia la densidad a 1 y a 8. ¿Dónde cae la transición?

## 2. Autojuego y juego cruzado (60 min)

1. Entrena 4 equipos de MAPPO sin comunicación y 4 con ella (el entorno de A15a).
2. Matriz de juego cruzado: dron 0 de un equipo con los drones 1 y 2 de otro. ¿Qué semilla aprendió otra convención? Dibuja los recorridos de esa semilla y de otra.
3. Mezcla un dron entrenado con compañeros voraces descentralizados.

## 3. Fallos y mentiras (45 min)

1. Un dron que deja de moverse y de mirar en el paso 30; uno que anuncia cada paso haber mirado la celda más probable sin mirarla; mensajes con 10 pasos de retraso.
2. Evalúa los equipos y los voraces. Explica por qué los voraces mejoran con el mentiroso.

## 4. Entrenamiento variado (45 min)

1. Entrena con una condición al azar en cada episodio. ¿Qué mejora y qué no?
2. Cambia la observación para que cada mensaje llegue con la identidad de quien lo envía (un mapa por compañero). ¿Aprende ahora a desconfiar del mentiroso?

## 5. Población de compañeros (45 min)

1. Construye una población con los equipos sin comunicación de 3 semillas en tres momentos del entrenamiento y entrena solo el dron 0 con compañeros elegidos al azar.
2. Evalúa con los drones de la cuarta semilla y con voraces. Añade voraces a la población y repite.

## Entregables

- `marl/vicsek.py`, `marl/condiciones.py` (fallos, mentirosos y retrasos) y `experimentos/a15b.py`.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Cómo atribuirías cada mensaje a su emisor y estimarías su fiabilidad dentro del belief?
2. ¿Qué compañeros distintos esperas en tu tesis (otras versiones, otros grupos, personas) y cómo construirías la población?
3. ¿Qué parte de la pérdida por un dron caído es recuperable reasignando la zona, y cómo lo medirías?
