# Laboratorio A11 · Un VLM local como planificador de misión

**Duración:** unas 3,5 h.
**Requisitos:** [ollama](https://ollama.com) con un modelo abierto multimodal de 4 a 12B (la presentación usa `gemma4:12b-it-qat`, unos 7 GB) y una GPU con 8 GB o más; en CPU funciona, mucho más despacio. Python con numpy y matplotlib. El script entero tardó unos 176 minutos en una RTX 4060 Ti, casi todo en el modo de razonamiento.

## Objetivo

Medir qué hace bien y qué hace mal un modelo de lenguaje y visión como planificador de misión de un dron, qué arregla un verificador externo y cuánto cuesta en tiempo. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a11.py`, salida en `figuras/salida.txt` y respuestas completas en `figuras/respuestas_a11.json`):

| Resultado de la presentación (Gemma 4 de 12B en 4 bits, RTX 4060 Ti, temperatura 0 salvo donde se indica) | Valor | Paso |
|---|---|---|
| Misiones posibles bien planificadas sin ejemplos / con dos ejemplos | 8/25 (32 %) / 8/25 (32 %) | 2 |
| Fallos por tramos que cruzan una zona prohibida, con dos ejemplos | 17 | 2 |
| Con verificador que admite «imposible»: bien planificadas / posibles dadas por imposibles | 10/25 (40 %) / 13 | 3 |
| Con verificador que exige un plan: bien planificadas / posibles dadas por imposibles | 16/25 (64 %) / 0 | 3 |
| Mapa como imagen: bien planificadas | 8/25 (32 %) | 4 |
| Razonamiento a temperatura 0 / con muestreo (12 misiones): bien planificadas | 0/25 (0 %) / 7/10 (70 %) | 4 |
| Tiempo por misión con dos ejemplos / razonando con muestreo | 4,5 / 331,1 s | 4 |
| Prellenado / decodificación | 1871 / 33,9 tokens/s | 5 |

Con otro modelo, otra cuantización u otro prompt las cifras cambiarán; lo que deberías reproducir es el tipo de fallo y lo que el verificador arregla y lo que no.

## 1. Misiones, verificador y planificador exacto (45 min)

1. Generador de misiones: seis zonas que no se solapan, una o dos prohibidas, base en (0, 0) y una autonomía calculada a partir del plan óptimo; una de cada seis, imposible.
2. Verificador: formato, estructura, zonas pedidas y prohibidas, tramos que pasan a menos de 2 m de una zona prohibida (Liang-Barsky) y energía. Cada error, con un mensaje concreto.
3. Planificador exacto: todas las órdenes de las zonas y rutas por un grafo de visibilidad. Comprueba que su plan pasa siempre el verificador.

## 2. El LLM sin ayuda y con ejemplos (40 min)

1. Prompt con las reglas, la API, el mapa en coordenadas y la orden; salida en JSON (formato forzado).
2. Repite con dos ejemplos resueltos por el planificador exacto. Clasifica los fallos por tipo.

## 3. El bucle con el verificador (40 min)

1. Devuelve los errores al modelo hasta 3 veces más. Prueba dos mensajes: uno que recuerda que puede responder «imposible» y otro que solo pide corregir.
2. ¿Cuántas misiones posibles da por imposibles en cada caso? ¿Detecta las imposibles?

## 4. Imagen y razonamiento (40 min)

1. Quita las coordenadas del texto y da el mapa como imagen. ¿Cuánto empeora y por qué fallos?
2. Activa el modo de razonamiento a temperatura 0 y mira el final del razonamiento: ¿termina? Repite con el muestreo que trae el modelo (temperatura 1, top_k 64, top_p 0,95) y un tope de tokens mayor. Mide aciertos, tokens y tiempo. Si fuerzas el formato JSON con razonamiento, ¿qué pasa?

## 5. Latencia (25 min)

1. Mide el prellenado y la decodificación con las estadísticas de ollama.
2. Calcula la cota del ancho de banda para tu GPU (bus × velocidad de la memoria / bytes de los pesos) y compárala con lo medido.

## Entregables

- `mision/generador.py`, `mision/verificador.py`, `mision/planificador.py` y `experimentos/a11.py`.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Qué parte de la orden de búsqueda de tu tesis dejarías al LLM y qué parte al planificador?
2. ¿Cómo detectarías que una misión es imposible sin preguntarle al LLM?
3. ¿Qué latencia aceptarías para replanificar en vuelo, y con qué modelo?
