# Cómo continuar el curso

Manual de trabajo para producir los temas que faltan con la misma calidad que M0 y M1. Está pensado para Claude Code, pero sirve igual para una persona. Si vas a usar Claude, el prompt listo para pegar está en [`00_plan/PROMPT_CONTINUAR.md`](00_plan/PROMPT_CONTINUAR.md).

## 1. Contexto en un minuto

- **Para quién:** Pablo Magariños, doctorando (desde octubre de 2026) en Tecnología Aeroespacial en la Universidade de Vigo. Su tesis: *Belief-state and world-model-based learning for autonomous aerial systems in partially observable environments*.
- **Qué es:** un curso intensivo autodidacta para ponerse al día en robótica inteligente, con tres bloques "ultra sólidos": control con IA, simulación, y hardware y despliegue. Todo orientado a sim-to-real.
- **Su nivel:** fuerte en RL (incluidos MARL y POMDP), dinámica de vuelo, ML y matemáticas. Débil en la pila del robot real (ROS 2, tiempo real, autopilotos, inferencia en el borde), MPC, estimación, planificación, imitation learning, VLAs y simuladores. Programa en Python y JAX; no en C++. Detalle en [`00_plan/diagnostico.md`](00_plan/diagnostico.md).
- **Lo que pidió:** curso intenso, exigente y en profundidad; derivaciones completas; diapositivas mixtas (texto y esquema a la par); español con términos en inglés; laboratorio por tema, proyecto integrador, tests y lecturas guiadas. Dio la última palabra sobre importancia y profundidad al diseño del curso.

## 2. Qué leer antes de empezar

1. [`CLAUDE.md`](CLAUDE.md): reglas cortas.
2. [`00_plan/ESTADO.md`](00_plan/ESTADO.md): qué está hecho, qué toca y dónde parar.
3. [`00_plan/TEMARIO.md`](00_plan/TEMARIO.md): orden, profundidad y laboratorio de cada tema.
4. [`00_plan/diagnostico.md`](00_plan/diagnostico.md): nivel de Pablo y ajustes de votación.
5. [`00_plan/ANALISIS.md`](00_plan/ANALISIS.md): la petición original y los criterios de "curso superado".
6. Las dos presentaciones hechas, como modelo de estilo: [`temas/M0_mapa-del-campo/index.html`](temas/M0_mapa-del-campo/index.html) (tema de panorama) y [`temas/M1_herramientas-matematicas/index.html`](temas/M1_herramientas-matematicas/index.html) (tema con derivaciones).

## 3. Flujo de trabajo por tema

Para cada tema, en el orden de `TEMARIO.md`:

1. **Marcar el tema como ⏳** en `ESTADO.md`.
2. **Investigar a fondo antes de escribir.** Buscar en la web:
   - Revisiones (surveys) de los últimos 2–3 años y los papers seminales.
   - Trabajo de 2025–2026 (la fecha actual manda; buscar novedades recientes).
   - Documentación oficial y repositorios (PX4, ROS 2, simuladores, librerías).
   - Resultados en hardware real, con cifras: Hz, ms, W, g, parámetros, tasas de éxito.

   Verificar cada referencia (autores, año, venue, enlace) y cada cifra en la fuente primaria; si solo hay fuentes secundarias, decirlo en la diapositiva. Nunca inventar referencias, autores, páginas ni cifras: si algo no se puede confirmar, se omite o se marca.
3. **Guion.** Ordenar el material para enseñar, no para enumerar. Estructura de la sección 4.
4. **Figuras.** Diagramas en SVG dentro del HTML. Cualquier figura con datos (curvas, nubes de puntos, escalas) se **calcula con un script** y se pega el resultado: nada de curvas dibujadas a ojo. Desde A1, el script se guarda en `temas/<ID>_<slug>/figuras/` para poder regenerar cada cifra y cada curva (numpy es suficiente).
5. **Escribir** `temas/<ID>_<slug>/index.html` con el motor compartido (sección 5).
6. **Laboratorio** en `temas/<ID>_<slug>/lab/README.md` (sección 7).
7. **Verificar en el navegador** (sección 6). No se da un tema por terminado sin esto.
8. **Actualizar** el índice `index.html` (marcar el tema como "disponible" con su enlace) y `ESTADO.md` (✅, fecha y notas: nº de hojas y cualquier cifra sin verificar).
9. **Commit y push** con un mensaje del tipo `Tema A1: dinámica y modelado`.
10. **Si el tema lleva 🛑**, parar: resumir a Pablo qué se ha hecho, qué queda dudoso y esperar su respuesta. Lo que pida se anota en el registro de revisiones de `ESTADO.md` y se aplica también a los temas siguientes.

### 3.1 Trabajo en paralelo con subagentes (desde A2)

Pablo pidió paralelizar siempre que no reste calidad. El reparto que la mantiene:

- **Investigación en paralelo y por adelantado, con moderación.** Pablo pidió (2026-11-06) cuidar el consumo: como mucho **un** agente de investigación por tema, con un encargo corto y acotado (pocas fuentes, solo cifras de artículos difíciles de verificar, sin descargas innecesarias). Las consultas a código y documentación oficial (parámetros de PX4, ficheros de un repositorio) se hacen directamente con `curl`, `gh` y `grep`, que es mucho más barato. Cada uno escribe un dossier en el scratchpad (`dossiers/<ID>/*.md`): cita completa con DOI o URL, la cifra o la frase exacta con su ubicación (tabla, sección, línea de código) y su estado (confirmado o no). La investigación del tema siguiente se lanza mientras se escribe el actual.
- **Redacción secuencial por un solo autor.** La presentación, el script de figuras y el laboratorio los escribe el agente principal, un tema detrás de otro, para mantener la coherencia de notación, convenios y referencias cruzadas. Nada del dossier entra en una hoja sin estar marcado como confirmado.
- **Revisión independiente antes del commit, solo en temas con muchas derivaciones** (A3a, A4a y similares), limitada a derivaciones y cifras. Un agente que no ha visto el razonamiento del autor recibe el HTML, el script y el laboratorio. Rehace las derivaciones y las cuentas, contrasta con su fuente una muestra de cifras y referencias, y busca errores de física, de convenio y de texto. Solo se corrige lo que se comprueba.
- **No se paralelizan** la redacción de temas que dependen entre sí, ni los commits: un commit por tema, en orden.

#### Temas de actualidad y casos reales: dossier y punto de control (desde R1a y R1b)

Cuando un tema vive de casos reales (empresas, despliegues, accidentes, regulación) y no de derivaciones, se puede encargar cada tema a un subagente propio. Así se hicieron R1a y R1b, en paralelo, el 2026-11-12, con permiso de Pablo. El agente principal prepara, revisa, integra y hace los commits. El subagente trabaja en dos fases separadas por un punto de control.

1. **Encargo.** Recibe solo la orden: tema, carpeta, puerto del servidor de comprobación (8765, 8766…; uno distinto por agente, con Playwright y no con el panel del navegador, que es compartido), ruta del dossier en el scratchpad y frontera con los otros temas. No toca `assets/`, el `index.html` raíz, `00_plan/`, otros temas ni git, no lanza agentes y no borra nada fuera de su carpeta y de su dossier.
2. **Fase 1 · Investigación**, sin escribir ninguna hoja:
   - **Fuentes primarias primero:** papers, blogs técnicos y documentación de la empresa, informes de accidentes (NTSB, AAIB, CIAIAC, BEA), documentos regulatorios (EASA, FAA, Federal Register), *think tanks* (RUSI, CSIS) y fichas de datasets.
   - **Actualidad obligatoria:** para cada caso, qué ha pasado en el último año y medio, con la búsqueda extendida.
   - **Fotos candidatas** con la licencia leída en su página (§4, «Fotos»), sin descargar nada.
   - **El dossier** en `dossiers/<ID>/`: un fichero por caso más `indice.md`. Cada dato lleva la cifra o la frase exacta, la cita completa, dónde está (sección, tabla, página, minuto), la fecha de la fuente y su estado:
     - **C**: confirmado en la fuente primaria;
     - **E**: según la empresa, porque solo la publica ella (los preprints de una empresa valen como C para especificaciones medidas con método, y como E para sus tasas de éxito);
     - **P**: prensa contrastada, para un hecho sin fuente primaria accesible que publican al menos dos medios fiables e independientes, o el texto íntegro de una agencia de referencia (Reuters, AP) leído en el original o en una sindicación literal;
     - **S**: sin confirmar.
   - **Mínimo por caso:** una fuente primaria, al menos tres datos C o E y lo más reciente, con su fecha. Por debajo, el caso es «débil».
   - **`indice.md`:** la tabla de casos, las cuentas con las especificaciones que usarán, los casos que propone añadir y quitar, las contradicciones entre fuentes y lo que no pudo consultar.
3. **Punto de control** (agente principal), en cuanto llega cada dossier:
   - contrastar con su fuente al menos dos datos por caso, empezando por los más recientes y los de empresa;
   - comprobar la licencia de cada foto candidata en su página de origen (API `extmetadata` de Commons, página de arXiv, Crossref) y, si la foto viene de otra web, también allí;
   - rehacer las cuentas;
   - decidir qué casos entran, qué datos se descartan o se vuelven a comprobar y los límites de los apartados delicados;
   - escribirlo en `dossiers/<ID>/decisiones.md` y mandárselo al subagente con `SendMessage`, para que escriba con su contexto intacto.
4. **Fase 2 · Escritura:** solo entran en una hoja los datos C, E o P del dossier, con E y P marcados como tales. Todo dato nuevo se añade antes al dossier. Al final, una **pasada de actualidad** de las dos o tres últimas semanas.
5. **Revisión independiente** (agente principal), al llegar el informe final:
   - todas las fotos cotejadas con su página, con `img/CREDITOS.md` y con el pie;
   - al menos 12 cifras y 8 referencias contrastadas con su fuente (más los identificadores de arXiv y los DOI);
   - todas las cuentas rehechas contra el script;
   - solapes con otros temas;
   - las comprobaciones del navegador de §6.

   Solo se corrige lo comprobado. Si el subagente se queda corto, se le continúa con `SendMessage`; no se relanza.

## 4. Estructura de cada presentación

Basada en M0 y M1, entre 40 y 60 hojas según la profundidad (más si el tema es muy matemático; M1 tiene 84).

**Cómo se presenta la matemática** (petición de Pablo del 2026-11-07, con M1 como modelo): ninguna fórmula entra sin que quede claro qué es cada símbolo, de dónde sale y por qué tiene esa forma. Para cada bloque:

- una hoja de **notación** al empezar cada parte (símbolo, qué es, ejemplo en un dron), y avisos cuando una letra cambia de significado (la \(R\) de rotación frente a la del ruido en Kalman);
- el **origen** de la fórmula en palabras, antes de la fórmula;
- la **derivación paso a paso** en una tabla «# · paso · por qué», visible en la hoja (no solo plegada) cuando la derivación es el contenido central;
- un **ejemplo numérico** con números pequeños que se puedan seguir a mano, calculado en el script de `figuras/`;
- mejor partir en dos o tres hojas que comprimir: una idea por hoja.

**Segunda petición del 2026-11-07** (tras revisar A2, A4a y A7a; se aplica a todos los temas):

- **Ningún símbolo sin definir en la hoja donde aparece** (o en la hoja de notación inmediatamente anterior). El glosario de la tecla `G` es una ayuda, no un sustituto: una hoja con \(K, T_i, T_d, FF\) sin decir qué es cada uno no se puede seguir.
- **Antes de una formulación, un esquema**: qué problema resolvemos, qué entra y qué sale, con un dibujo (diagrama de bloques, línea temporal, árbol…) y un ejemplo concreto. La fórmula llega después.
- **Las fórmulas fundamentales del tema** (las que el resto del tema usa) se derivan en la hoja, con su tabla «paso / por qué» y su ejemplo numérico, repartidas en las hojas que hagan falta.
- **Nada plegado salvo las respuestas** de la autoevaluación y de los exámenes. Lo secundario va en un `div.aparte` (recuadro punteado, siempre abierto, texto gris a tamaño de lectura) y sin comprimir: si no cabe, otra hoja.
- **Sin letra pequeña para el contenido**: `.small` (16 px) solo para pies y notas; el texto que hay que leer para entender va a tamaño normal. Si la hoja no cabe, se parte.

En HTML, el signo «menor que» dentro de una fórmula se escribe `\lt` (o `&lt;`): `a_{<t}` lo interpreta el navegador como una etiqueta y se come el resto de la fórmula sin dar error.

1. **Portada** (`title-slide`): kicker con fase y jornada, título, subtítulo y meta (duración, prerrequisitos, qué prepara).
2. **Objetivos**: 4–6, verificables ("derivar", "explicar", "calcular").
3. **Por qué importa**: el problema real y su conexión con lo anterior.
4. **Partes** separadas por hojas `divider`, cada una con:
   - historia breve con fechas (componente `.tl`),
   - fundamentos con ecuaciones clave y **derivaciones completas** en `details.deriv`,
   - estado del arte 2025–2026 con cifras verificadas,
   - cómo se hace en drones (multirrotor y, cuando aplique, ala fija),
   - limitaciones y errores frecuentes.
5. **Hacia dónde va**: previsiones marcadas como tales, no como hechos.
6. **Conexión con la tesis**: callouts `tesis` repartidos, más una hoja de preguntas abiertas.
7. **Resumen**.
8. **Autoevaluación**: 3 hojas, unas 6 preguntas (5 de opción múltiple + 1 abierta con respuesta plegada). Al menos una debe ser una cuenta.
9. **Lecturas guiadas**: 3 esenciales (qué leer, qué buscar, qué responder) y una línea de profundización.
10. **Laboratorio**: resumen en una hoja; el enunciado completo, en `lab/README.md`.
11. **Referencias**: varias hojas, unas 14–17 por hoja, con `start` correlativo.
12. **Glosario** (petición de Pablo del 2026-11-07): bloque `<aside id="glosario" hidden>` después de `</main>`, que el motor muestra con la tecla `G` o el botón «Glosario». Recoge **todos** los símbolos del tema (letras con sus subíndices y decoraciones, operadores, constantes con nombre) y, al final, las siglas. Formato: un `<h3>` por grupo (las partes del tema, en orden), un `<p>` opcional con convenciones y un `<dl>` de pares `<dt>\(símbolo\)</dt><dd data-hojas="12,30">qué es</dd>`. El `<dt>` copia el TeX tal como sale en las hojas; el `<dd>` dice qué es, tipo y dimensión, unidades si es físico y con qué se relaciona, sin datos que no estén en el tema; `data-hojas` es la hoja donde se define (y una o dos donde se usa mucho). Si una letra cambia de significado entre partes, una entrada por significado. **Si se añaden o quitan hojas, se renumeran los `data-hojas`.** M1 es el modelo.

Hilos que deben aparecer en todos los temas: latencia y frecuencia (el color `.hz`), observabilidad parcial, sim-to-real y cómputo limitado.

**«Así lo hacen en la realidad»** (petición de Pablo del 2026-11-12): desde C1a, cada tema lleva al menos dos hojas sobre un sistema real (empresa, lab o despliegue) relacionado con su contenido. Cada una con cifras verificadas (las de la empresa, como afirmaciones suyas), su esquema y su foto si tiene licencia libre comprobada, y con lo que ese caso enseña del tema. Ejemplos: C1b con el cómputo de Skydio y de DJI; C2b con la arquitectura de Zipline; C5 con la inferencia de un VLA en un robot de producto; C7 con un informe de accidente; B7b con un despliegue sim-to-real comercial. Se puede remitir a R1a y R1b en lugar de repetirlos.

**Fotos e imágenes** (desde R1a y R1b). El repositorio es público, así que cada imagen incrustada se redistribuye:

| Fuente | ¿Se incrusta? | Condición |
|---|---|---|
| Wikimedia Commons | Sí | Licencia leída en la página del fichero: CC0, dominio público, CC BY o CC BY-SA. Con CC BY-NC, solo anotando que el curso no es comercial. Con ND, sin modificar ni recortar. Si la foto viene de otra web, la licencia tiene que estar clara también allí (las de ArmyInform quedaron fuera porque su web dice «CC BY 4.0» y enlaza a CC BY-NC-ND). |
| Obras del Gobierno de EE. UU. (DoD/DVIDS, NASA, FAA) | Sí | Si la página dice dominio público y no hay aviso de copyright de terceros. |
| Figuras de papers | Sí | Solo si el paper es CC BY, CC BY-SA o CC0, comprobado en arXiv, en la revista o en Crossref. Se cita el paper y la figura. |
| Capturas propias (SITL, Gazebo, gráficas del curso) | Sí | — |
| Notas de prensa, webs de empresa, fotogramas de YouTube, X o LinkedIn, informes «All rights reserved» | **No** | Se enlazan («vídeo oficial: …», con fecha de consulta). |

- **Formato:** `img/<slug>.webp` o `.jpg`, como mucho 1600 px de ancho, 300 KB por imagen y 6 MB por tema. Se convierte con Pillow (`uv run --with pillow`) y se comprueba con `file` que lo descargado es una imagen.
- **Pie:** qué se ve y por qué importa, y el crédito en `span.xsmall`: «Foto: autor, licencia (enlace), vía fuente», más «recortada» o «redimensionada» si se ha modificado. Una versión modificada de una foto CC BY-SA queda bajo CC BY-SA.
- **`img/CREDITOS.md`:** una fila por imagen con fichero, qué muestra, autor, licencia con enlace, página de origen, fecha de consulta y qué se ha modificado.
- **Las fotos no sustituyen a los esquemas.** Cada sistema importante lleva su diagrama de arquitectura dibujado por el curso; la foto acompaña.
- **Una foto sin licencia clara no entra**, aunque sea la mejor: se busca otra o se enlaza.

## 5. El motor de presentaciones

Cada tema es un HTML que enlaza `../../assets/curso.css?v=N` y `../../assets/curso.js?v=N`, carga las fuentes de Google Fonts y MathJax 3.2.2 (ver la cabecera de M1 y copiarla tal cual). Hojas de 1280×720; modo presentación (flechas, `L` lectura, `I` índice, `G` glosario, `T` tema) y modo lectura automático en pantallas estrechas.

**Si cambias `assets/`**, sube el número `?v=` en **todas** las páginas que lo enlazan (los navegadores cachean). Ahora mismo: `curso.css?v=13`, `curso.js?v=5`.

Componentes disponibles:

| Componente | Uso |
|---|---|
| `<section class="slide" data-section="…">` | Una hoja. `data-section` solo en la primera hoja de cada sección (aparece en el índice y el cajetín). |
| `slide title-slide`, `slide divider` | Portada y separador de parte (`p.num`, `h2`, `p`). |
| `p.kicker` + `h2` | Encabezado de cada hoja. Títulos de una línea si es posible. |
| `.cols`, `.cols.c3`, `.cols.c4`, `.cols.w60`, `.cols.w40`, `.center` | Rejillas de 2, 3, 4 columnas o 60/40 y 40/60. |
| `.stack`, `.body` | Columna con separación; `.body` ocupa el alto sobrante. Con `.cols` (p. ej. `cols c3 body`) sigue siendo rejilla desde `curso.css?v=7`. |
| `.callout.clave / .realidad / .tesis / .error` con `<span class="tag">` | Avisos: idea clave, cifras reales, tu tesis, error frecuente. |
| `.card` (`.meta`, `h3`, `p`) y `.cards-sm` en el contenedor | Tarjetas; `cards-sm` reduce la letra si no caben. |
| `.table-wrap > table.t`, `td.n` | Tablas; `td.n` para cifras en monoespaciada. |
| `figure.foto > img` + `figcaption` (con `span.xsmall` para el crédito) | Fotos (desde `curso.css?v=13`). Alto máximo con `style="--foto-h:280px"` en el `figure` (340 px por defecto) o `style="max-height:…"` en la `img`. Por defecto `object-fit: contain` (la foto entera); `.cubre` recorta a la caja (nunca con licencias ND) y `object-position` se ajusta con `style`. `.foto.duo`: dos `img` lado a lado a la misma altura y un solo pie (una encima de otra en móvil). La `img` lleva siempre `width`, `height` y `alt`, y nunca `loading="lazy"`: así reserva su sitio antes de cargar, `checkOverflow()` mide bien y la comprobación de imágenes cargadas de §6 no da falsos fallos. Pie: qué se ve y por qué importa, y debajo «Foto: autor, licencia (enlace), vía fuente». |
| `figure.fig > svg` + `figcaption` | Figuras. Clases SVG: `box`, `box-accent`, `box-time`, `box-tesis`, `box-soft`, `ln`, `ln-accent`, `ln-time`, `ln-tesis`, `ln-dash`, `ln-rule`, `fill-*`, `t-mono`, `t-muted`, `t-faint`, `t-accent`, `t-time`, `t-tesis`, `t-bold`. Todo color sale de las variables del tema: así funcionan claro y oscuro. |
| `.tl > .ev > .yr` | Línea temporal. |
| `details.deriv > summary` | Plegable. **Solo para respuestas** de la autoevaluación y de los exámenes (desde el 2026-11-07). |
| `div.aparte > span.rotulo + …` | Material secundario siempre visible: recuadro punteado, texto gris a 18 px (desde `curso.css?v=10`). El rótulo dice qué es («Por qué funciona», «Comprobación», «Caso particular»). |
| `.quiz[data-answer="i"]` con `ol.opts > li > button` y `p.explain` | Test de opción múltiple; `i` es el índice (desde 0) de la respuesta correcta. |
| `<a class="cite" data-ref="clave1,clave2"></a>` | Cita; el número se calcula solo. |
| `ol.refs[start=N] > li#ref-clave` | Referencias. Cada clave citada debe existir y cada referencia debe citarse al menos una vez. |
| `span.hz` | Frecuencias, latencias y cifras de tiempo, en el color del hilo temporal. |
| `.small`, `.xsmall`, `.muted`, `.lead` | Tamaños y tonos de texto. |
| `aside#glosario[hidden] > h3 + p? + dl > dt + dd[data-hojas]` | Glosario del tema (tecla `G`, desde `curso.js?v=4`): panel con buscador (busca también por el código TeX, p. ej. `theta`) y enlaces a las hojas. Ver §4, punto 12. |

Matemáticas con MathJax: `\( … \)` en línea y `\[ … \]` en bloque. Las fórmulas muy largas se parten con `aligned`.

## 6. Verificación en el navegador

1. Servir el repositorio:

   ```bash
   python3 -m http.server 8765 --bind 127.0.0.1
   ```

   En Claude Code de escritorio existe la configuración `curso` en `.claude/launch.json`, que hace lo mismo y abre el panel del navegador. Abrir el HTML como archivo suelto en el panel de Claude no carga `assets/`; en un navegador normal sí funciona.
2. Abrir `http://localhost:8765/temas/<ID>_<slug>/index.html` con una ventana de unos **1440×900** en modo presentación y ejecutar en la consola:

   ```js
   curso.checkOverflow()   // debe devolver []
   ```

   Devuelve las hojas que no caben en 1280×720 y, desde `curso.js?v=3`, también las que recortan contenido por dentro: tablas (`.table-wrap`), bloques `.body` y fórmulas en bloque que se salen de su caja. Arreglar recortando texto, usando `cards-sm`, partiendo hojas o fórmulas, o limitando la altura de una figura (`style="max-height:…px"` en el `svg`).

   `checkOverflow()` no ve solapes entre bloques ni etiquetas que se pisan dentro de una figura: **mira cada hoja**, en tema claro y oscuro. Sin un navegador a mano, Playwright con el Chrome del sistema sirve para todo este apartado y para hacer capturas (`uv run --with playwright python …`, `channel="chrome"`).
3. Comprobar que no hay citas sin referencia (aparecerían como `[?]`), ni errores de MathJax (`document.querySelectorAll('mjx-merror').length` debe ser 0), ni errores en consola.
4. Con ancho de móvil (unos 375 px), comprobar que ninguna hoja desborda en horizontal:

   ```js
   [...document.querySelectorAll('.slide')].map((s,i)=>[i+1, s.scrollWidth - s.clientWidth]).filter(x=>x[1]>2)   // debe ser []
   ```

5. Mirar las figuras en tema claro y oscuro: etiquetas que no se pisen y que todo quede dentro del dibujo.
6. Glosario: `uv run --with playwright python 00_plan/verificar_glosario.py <carpeta> [--shots]` (con el servidor del paso 1 en marcha) comprueba que el panel abre con `G`, que no hay errores de MathJax ni desbordes, que cada `data-hojas` está en rango y lista las entradas cuyo TeX no aparece literal en la hoja citada, para revisarlas a mano. También repite `checkOverflow()` y la prueba de móvil.

Si no hay ningún navegador disponible, decirlo en la siguiente parada y pedir a Pablo que ejecute `curso.checkOverflow()`.

## 7. Laboratorios

- **Independientes del hardware.** Un PC con GPU NVIDIA de consumo o la nube, con CPU como alternativa. Si algo necesita más memoria, indicarlo y dar una alternativa (modelo más pequeño, cuantización, servicio en la nube).
- **Nunca se vuela un dron.** Los datos reales salen de datasets y logs públicos. Una Pixhawk solo como hardware-in-the-loop opcional.
- **Todo en Python + JAX** dentro del *proyecto hilo* (un simulador y pipeline de dron en JAX que crece con los labs; estructura en el lab de M0), salvo los labs de ROS 2, PX4 y despliegue, que usan sus herramientas.
- Formato del README: objetivo, duración, pasos numerados con lo mínimo de código para no bloquearse, tests o comprobaciones con resultado esperado, entregables y preguntas de cierre. Cada lab debe verificar numéricamente al menos un resultado de la presentación.

## 8. Hitos (H1, H3, H4 dentro de C7 y examen final)

Una presentación corta con el mismo motor, en `temas/<ID>_<slug>/`, que contiene:

- el enunciado de integración del proyecto hilo para ese hito, con criterios de aceptación;
- un examen de fase de 10–15 preguntas, mezclando opción múltiple, cuentas y preguntas abiertas con respuesta plegada.

El examen final evalúa los 7 criterios de "curso superado" de `ANALISIS.md` §3.

## 9. Git

- Un commit por tema, que incluya el tema, `index.html` y `ESTADO.md`. Push al terminar cada tema.
- Antes de empezar en un ordenador: `git pull`.
- No subir datasets ni binarios grandes: los labs explican de dónde descargarlos.
