# Laboratorio R1a · Audita los datos y las afirmaciones de un lab

**Duración:** unas 3 h.
**Requisitos:** cualquier ordenador con Python 3.10 o posterior. Solo CPU. Para la figura, `matplotlib` (`uv run --with matplotlib`). No hace falta GPU, ni cuenta en Hugging Face, ni aceptar condiciones: los seis conjuntos del paso 1 son públicos y sin acceso restringido (comprobado el 12-11-2026). No se descarga vídeo, solo ficheros de metadatos de pocos kilobytes.
**Material:** las cuentas de la presentación están en `../figuras/ejemplos_r1a.py` (salida en `../figuras/salida_ejemplos.txt`), las figuras en `../figuras/figuras_r1a.py` y las cifras con su fuente y su estado en `../figuras/datos/cifras.json`. Úsalos de referencia; puedes copiar las funciones `wilson`, `poisson_exacto` y `n_potencia`. Escribe tus scripts en el proyecto hilo y guarda las salidas en `experimentos/r1a_*`.

## Objetivo

Hacer con tus manos las tres comprobaciones que la jornada aplica a cualquier lab:

- cuántos datos hay de verdad en los datasets abiertos, leídos de sus propios ficheros y no de un titular;
- qué dice (y qué no dice) una tasa de éxito publicada, con su intervalo y los ensayos que habría hecho falta;
- qué demuestra un vídeo oficial, con la lista de seis preguntas de la parte 5;
- y una cifra de seguridad real reproducida desde cero: el intervalo de Poisson del hub de Waymo.

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| DROID: segundos por trayectoria en el artículo | 16,6 s | 1 |
| Ensayos por política para separar 0,6 de 0,7 (α = 0,05, potencia 0,8) | 356 | 2 |
| Cota inferior de la tasa con 20 de 20 (95 %, una cola) | 0,861 | 2 |
| Waymo, lesión grave o peor, IC al 95 % por millón de millas | [0,00228; 0,0323] | 4 |

## 1. Lo que hay en los datasets abiertos (60 min)

1. Lee el fichero `meta/info.json` de estos seis repositorios en formato LeRobot. Solo con la biblioteca estándar:

   ```python
   import json, urllib.request
   REPOS = ["lerobot/droid_1.0.1", "lerobot/aloha_static_coffee", "lerobot/aloha_mobile_cabinet",
            "IPEC-COMMUNITY/bridge_orig_lerobot", "IPEC-COMMUNITY/fractal20220817_data_lerobot",
            "lerobot/libero_10_image"]
   def info(repo):
       url = f"https://huggingface.co/datasets/{repo}/resolve/main/meta/info.json"
       return json.load(urllib.request.urlopen(url))
   ```

   De cada uno saca `robot_type`, `total_episodes`, `total_frames`, `fps`, la forma de `features["action"]` y cuántas entradas de `features` son cámaras (`dtype` igual a `video` o `image`). Con `https://huggingface.co/api/datasets/<repo>` saca además `gated` (debe ser `False`), la licencia (`cardData.license`) y `lastModified`.
2. Calcula para cada uno las horas, \(\text{horas} = \text{total\_frames}/(\text{fps}\cdot 3600)\), y la duración media de un episodio, \(\text{total\_frames}/(\text{fps}\cdot\text{total\_episodes})\).
3. Haz una tabla y una figura de barras en escala logarítmica con tus seis conjuntos junto a las horas publicadas de la hoja 16 (88 h de teleoperación propia de GR00T N1, 350 h de DROID, más de 10 000 h de π0, un millón de horas de vídeo de V-JEPA 2; están en `cifras.json`, clave `horas_datos`).
4. Compara con lo que dicen los artículos: DROID cuenta 76 000 trayectorias y 350 h; RT-1, cuyos datos son Fractal, 130 000 episodios (A12a). Mira en `features` de DROID si hay un campo que explique la diferencia.

**Comprueba** (valores leídos el 12-11-2026; si un repositorio cambia, anota la fecha y el valor nuevo):

| Repositorio | Robot | Episodios | Horas | s por episodio | fps | Dim. acción |
|---|---|---|---|---|---|---|
| `droid_1.0.1` | Franka | 95 658 | 511,7 | 19,3 | 15 | 8 |
| `fractal20220817_data_lerobot` | google_robot | 87 212 | 350,6 | 14,5 | 3 | 7 |
| `bridge_orig_lerobot` | widowx | 53 192 | 105,2 | 7,1 | 5 | 7 |
| `libero_10_image` | panda | 379 | 2,8 | 26,8 | 10 | 7 |
| `aloha_mobile_cabinet` | aloha | 85 | 0,71 | 30,0 | 50 | 14 |
| `aloha_static_coffee` | aloha | 50 | 0,31 | 22,0 | 50 | 14 |

En DROID, 350 h entre 76 000 trayectorias son 16,6 s (la cuenta de la hoja 18), y la copia de Hugging Face da 19,3 s por episodio con 95 658 episodios: el campo `is_episode_successful` indica que la copia incluye también los fallidos. Escribe en una frase por qué «el dataset tiene X episodios» no basta sin decir qué versión y qué cuenta.

## 2. Una tasa de éxito con su n (45 min)

1. Elige **tres resultados publicados con su \(n\)**: los de la presentación (π0 con 8 de 10, los humanos de π0.7 con 24 de 30, la racha de 86 de GEN-1) u otros que encuentres en artículos de 2025–2026 con el número de ensayos escrito.
2. Para cada uno, el intervalo de Wilson al 95 % (T1b). Si todos los ensayos salen bien (\(k = n\)), la cota inferior exacta a una cola, \(p_L = \alpha^{1/n}\), y la regla del tres, \(1 - 3/n\).
3. Con `n_potencia`, cuántos ensayos por política harían falta para separar dos tasas que te interesen (por ejemplo, tu política frente a la de referencia de A6b). Y al revés: con 50 ensayos por política y una tasa cerca del 50 %, cuál es la diferencia mínima detectable (resuelve \(n(\Delta) = 50\) por bisección).

**Comprueba:** 8 de 10 da [0,49; 0,94]; 24 de 30, [0,63; 0,90]; 20 de 20, \(p_L = 0{,}05^{1/20} = 0{,}861\) (la regla del tres da 0,85); 86 de 86, 0,966; 0,6 frente a 0,7, 356 por política; con 50 por política alrededor del 50 %, unos 27,7 puntos.

## 3. Seis preguntas a dos vídeos (45 min)

1. Elige **dos vídeos oficiales de 2025–2026** de empresas distintas, publicados en sus propias páginas (los de las referencias de la presentación sirven: Helix, Gemini Robotics 2.5, π0.7, GEN-1, el LBM de Atlas o NEO).
2. Responde a las seis preguntas de la hoja 37 para cada vídeo: ¿autónomo o teleoperado?, ¿a qué velocidad?, ¿cuántos intentos y cortes?, ¿entorno preparado o nuevo?, ¿cuántos ensayos y fallos?, ¿quién lo mide? En cada respuesta, la prueba: el minuto del vídeo o la frase de la página. Si no hay prueba, escribe «no lo dice».
3. Escribe un veredicto de tres líneas con la escala de la hoja: ¿demuestra que algo **puede pasar** una vez, da **una tasa con su \(n\)** o describe **un despliegue** con horas e intervenciones?

**Comprueba:** al menos una de las seis respuestas de cada vídeo será «no lo dice». Si las seis tienen prueba, el vídeo es la excepción: guárdalo como ejemplo de buena práctica.

## 4. El intervalo de Waymo desde cero (30 min)

1. Del hub de seguridad de Waymo (datos a junio de 2026): 271,3 M de millas sin conductor y una tasa de lesión grave o peor de 0,01106 por millón de millas. Calcula cuántos sucesos son: \(x = 0{,}01106 \cdot 271{,}3\).
2. Intervalo exacto de Poisson (Garwood) para \(x\) sucesos con `poisson_exacto` o con cuantiles de \(\chi^2\): \(\mu_L = \tfrac12\chi^2_{0{,}025,\,2x}\), \(\mu_U = \tfrac12\chi^2_{0{,}975,\,2x+2}\). Divídelo por las millas.
3. Peatones: 7 sucesos frente a 93 esperados con la referencia humana. Trata los 93 como conocidos y calcula el intervalo de la razón \(RR = x/93\).
4. Regla del tres por distancia: con cero lesiones graves, cuántas millas harían falta para acotar la tasa por debajo del 10 % de la humana (0,21335 por millón de millas). La versión por horas de vuelo está en R1b.

**Comprueba:** \(x = 3\); el intervalo de los sucesos, [0,619; 8,767]; por millón de millas, [0,00228; 0,03232], que reproduce el [0,00228; 0,03231] del hub; peatones, de 0,030 a 0,155; unos 141 M de millas.

## Entregables

- `experimentos/r1a_metadatos.py` y su tabla y figura de escalas (paso 1), con la fecha de lectura de cada repositorio.
- `experimentos/r1a_estadistica.py`: los intervalos, las cotas y los tamaños de muestra del paso 2, y el del paso 4.
- `experimentos/r1a_videos.md`: las dos fichas de seis preguntas con sus pruebas y sus veredictos.
- Una tabla con tus valores frente a los de la presentación (la de arriba).

## Preguntas de cierre

1. Para tu tesis harían falta datos de dron reales. Con la tabla del paso 1, ¿qué frecuencia, qué dimensión de acción y cuántas horas tendría un «DROID de drones» para tu tarea de búsqueda? ¿Cuánto costaría recogerlo con la cuenta de la hoja 17 y por qué casi nadie lo ha hecho?
2. Si comparas dos variantes de tu *belief* con 20 vuelos reales cada una y salen 15 y 18 éxitos, ¿puedes decir que la segunda es mejor? ¿Cuántos vuelos te harían falta y cuántos episodios de simulador te los sustituyen sin perder lo que solo se ve en el dron real?
3. El hub de Waymo publica método, referencia, intervalos y limitaciones. Escribe en cinco líneas qué publicarías tú de tu sistema para pasar el filtro de credibilidad de M0, con la misma estructura.
