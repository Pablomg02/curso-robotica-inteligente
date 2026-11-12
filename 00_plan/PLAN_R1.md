# Plan: R1a y R1b, «Así lo hacen en la realidad»

> 2026-11-12. Petición de Pablo: «Echo en falta leer sobre la actualidad de la industria y el research de labs estilo DeepMind, PI, etc. […] un tema que enseñe cómo lo hacen en la realidad con ejemplos reales, fotos, figuras». El prompt para ejecutar este plan está en §9. Este fichero se borra al terminar R1b: lo que siga valiendo pasa a `CONTINUAR.md` y `TEMARIO.md` (§4.3, punto 5).

## 1. Por qué y qué hueco cubre

Una búsqueda en los 35 temas hechos (2026-11-12) dio este resultado:

- **Lo que hay:** los labs aparecen como **métodos**. A12a y A12b explican π0, π0.5 y π\*0.6 (Physical Intelligence), Gemini Robotics (Google DeepMind), GR00T N1 (NVIDIA), RDT-1B y Open X-Embodiment. A8b explica V-JEPA 2, Genie, Cosmos y GAIA. M0 tiene la hoja «Quién hace qué». Swift sale en M0, A6a y A13.
- **Lo que falta:**
  - Cómo trabajan los labs: datos, flotas, cómputo, evaluación, despliegue, qué es demo y qué es producto.
  - Toda la industria de drones en producción: DJI tiene 0 menciones. Skydio sale solo como modelo de MuJoCo. Zipline, Wing, Prime Air, Shield AI, Anduril y Ucrania tampoco aparecen.
  - Las empresas de robótica general: Boston Dynamics, Tesla, Figure, Agility, Skild y Waymo.
  - **Fotos:** ninguna. Los 35 temas no tienen ni una `<img>`.

Los temas planificados tampoco lo cubren. C1b y C2b tratan plataformas y arquitecturas reales desde el hardware. D2 trata de cómo se investiga, no de qué se está construyendo.

## 2. Decisiones

| Decisión | Elegido | Por qué |
|---|---|---|
| Formato | Dos temas nuevos de una jornada cada uno, **R1a** y **R1b**, más una regla fija para los temas siguientes (§7) | Un solo tema no da para labs y drones a la vez sin quedarse en una lista. La regla evita que la realidad quede encerrada en dos jornadas. |
| Sitio | **J36–37**, en la fase 5, justo después del taller T1 y antes de C1a | Pablo los quiere ya. Encajan ahí: R1a cierra la visión de las fases 3 y 4 (lo que hacen los labs con lo aprendido) y R1b abre el bloque de despliegue con sistemas reales que los temas C desmontan después pieza a pieza. |
| Renumeración | C1a…C7 pasan a J38–48 y la fase 6 a J49–55. El curso queda en **52 presentaciones y 55 jornadas** | Hecho ya en `TEMARIO.md`, `ESTADO.md`, `index.html` y `README.md`. Falta la figura del plan en M0 (hoja 3, ver §6.3). |
| Profundidad | «A fondo», de 50 a 65 hojas cada uno, como M0 (tema de panorama) | Son temas de casos, pero con las cuentas que permiten **leer** un caso: presupuestos, estadística de evaluación y fiabilidad, derivadas como en M1. |
| Fotos | Sí, con **licencia libre verificada**. Lo que no tenga licencia va enlazado (§5) | El repositorio es **público**: una foto de prensa incrustada sin licencia es redistribuirla. |
| Paralelismo | **Un subagente por tema**, en paralelo. El agente principal prepara, revisa, integra y hace los commits (§4) | Pablo lo autoriza para estos dos temas. Son independientes entre sí, con la frontera fijada en §3, así que la excepción a «un solo autor» de `CONTINUAR.md` §3.1 no resta coherencia. |
| Parada | 🛑 tras R1b | Pablo revisa los dos antes de seguir con C1a. |

## 3. Contenido

Ninguno de los dos temas repite arquitecturas ya explicadas (π0, GR00T, V-JEPA, Swift…): las cita con su tema («la arquitectura está en A12b») y cuenta lo que allí falta. **Todas las empresas, cifras y fechas de esta sección son candidatas que hay que investigar y verificar.** Las que no se confirmen se quitan. Las que solo publica la empresa van como «según X». La lista no es cerrada: si en noviembre de 2026 hay un caso más importante, entra ese.

### R1a · Los labs de robot learning en 2026: cómo se hace de verdad

Pregunta central: **¿cómo convierte un lab un método de las fases 3 y 4 en un robot que funciona fuera de la demo, y cómo se sabe si lo ha conseguido?**

1. **El mapa, con fechas y tamaños.**
   - Labs de grandes empresas: Google DeepMind (Gemini Robotics), NVIDIA (GEAR, Isaac, Cosmos), Meta FAIR, Amazon (tras incorporar el equipo de Covariant en 2024), Toyota Research Institute, OpenAI (vuelta a la robótica).
   - Startups de modelos fundacionales: Physical Intelligence, Skild AI, Generalist, Field AI, Dyna…
   - Humanoides y robots de producto: Tesla Optimus, Figure (Helix), 1X (NEO y su world model), Agility (Digit), Apptronik, Boston Dynamics (Atlas eléctrico con los Large Behavior Models de TRI), Unitree, AgiBot.
   - Los grupos académicos que marcan el ritmo: Stanford, Berkeley, CMU, ETH RSL y UZH RPG, MIT.
   - Una línea temporal 2015–2026 (`.tl`) y una tabla de quién publica pesos, código y datos y quién no.
2. **Los datos: la parte cara.**
   - Teleoperación: ALOHA, UMI, guantes, VR.
   - Flotas de recogida y sus cifras: horas, episodios, robots, operadores. Datasets abiertos como Open X-Embodiment, DROID y AgiBot World.
   - Vídeo humano y egocéntrico. Datos sintéticos: GR00T, DreamGen, Cosmos.
   - La «pirámide de datos», con cifras verificadas.
   - Una **cuenta**: el coste de recoger N episodios con R robots y operadores (modelo sencillo con sus símbolos, derivado y con ejemplo del script).
3. **Entrenar y desplegar.**
   - Cómputo publicado: GPU·h y parámetros, solo lo que esté publicado.
   - Ciclo de datos: el despliegue alimenta el entrenamiento (RECAP y el *data flywheel*, enlazando con A12b).
   - Inferencia en el robot frente a la nube y su latencia (enlaza con A12b y C5): una figura de parámetros frente a frecuencia de control, calculada con cifras verificadas.
4. **Evaluar: cómo se sabe que funciona.**
   - Ensayos en robot real: A/B, evaluación ciega, evaluaciones distribuidas como RoboArena.
   - Simuladores y world models como evaluadores (con la cautela de la correlación de A8b).
   - Las **cuentas**, derivadas paso a paso:
     - Cuántos ensayos hacen falta para distinguir un 60 % de éxito de un 70 % (binomial, aproximación normal y potencia).
     - El intervalo de Wilson de una tasa de éxito publicada (derivado en T1b: se cita y se aplica).
     - Qué dice «20/20 éxitos».
   - Tres resultados publicados reales, recalculados con su n.
5. **Demo frente a producto.**
   - Cómo leer un vídeo: teleoperado o autónomo, velocidad, cortes, número de intentos, condiciones del entorno.
   - Casos documentados por fuentes fiables de demos teleoperadas o editadas, y de despliegues reales con cifras: horas de operación continua, robots en fábricas, pilotos comerciales.
   - El filtro de credibilidad de M0, aplicado a cuatro anuncios de 2025–2026.
6. **Conducción autónoma, el caso más maduro.**
   - Waymo y Wayve: escala de flota y kilómetros sin conductor, simulación y world models, comparativas de seguridad publicadas y cómo se leen.
   - Es el único sector con despliegue masivo de autonomía aprendida bajo observabilidad parcial. Lo que un dron puede copiar de él y lo que no.
7. **Y los drones, ¿dónde están en estos labs?** Casi ausentes, y por qué: datos, riesgo, peso y cómputo. Puente a R1b.
8. **Conexión con la tesis.** Qué haría falta para que la tesis de Pablo produjera un resultado creíble según los estándares de esta parte: datos, evaluación y n.

### R1b · Drones autónomos en el mundo real: cómo lo hacen

Pregunta central: **¿qué autonomía vuela hoy de verdad, con qué sensores, cómputo y enlace, y qué la limita?** El hilo es la observabilidad parcial en el mundo real: sin GNSS, con el enlace interferido o perdido, de noche, con niebla, en interiores.

1. **El mapa del sector.**
   - Aplicaciones: consumo, inspección, seguridad y emergencias, reparto, agricultura, defensa, carreras.
   - Escala: unidades, vuelos y horas, según fuentes fiables.
   - Quién hace qué: DJI, Skydio, Zipline, Wing, Amazon Prime Air, Flyability, Percepto, Shield AI, Anduril, Helsing, los fabricantes ucranianos.
   - Línea temporal 2006–2026.
2. **Anatomía de cinco sistemas.** Para cada uno, un dibujo SVG de su arquitectura (sensores, cómputo, autonomía, enlace, operación) con cifras verificadas, la foto si hay licencia y lo que publica y lo que calla:
   - **DJI** (navegación visual omnidireccional, transmisión O4…);
   - **Skydio** (de Skydio 2 a X10 y Dock; autonomía visual con varias cámaras; salida del consumo y paso a empresas y defensa);
   - **Zipline** (P1 de ala fija y P2 con descenso por cable; detección y evitación);
   - **Wing** o **Prime Air** (el que tenga mejores fuentes);
   - **Shield AI V-BAT / Hivemind** (autonomía sin GNSS).

   C1b y C2b retomarán el cómputo y la arquitectura de estos casos. Aquí basta con el sistema entero y lo que enseña.
3. **Investigación que llegó a la realidad.** Swift frente a campeones humanos (enlaza con M0 y A6a, sin repetirlo), las competiciones A2RL de Abu Dabi, Agilicious, MIT y ETH. Qué pasó de ahí a producto y qué no.
4. **Ucrania: autonomía bajo interferencia electrónica.** Solo a nivel de sistema y con fuentes públicas fiables (RUSI, CSIS, prensa especializada contrastada):
   - Por qué la guerra electrónica empuja la autonomía: el enlace se pierde y el GNSS se interfiere, lo que es observabilidad parcial en estado puro.
   - Qué partes se automatizan: navegación sin GNSS, guiado terminal. Drones de fibra óptica como respuesta sin autonomía.
   - Cifras de producción y de eficacia, con su fuente y su incertidumbre.
   - **Sin detalles de armamento, de selección de blancos ni de cómo construir o mejorar un sistema ofensivo.** Interesa la arquitectura y la lección de observabilidad.
5. **Lo que falla.**
   - Informes de accidentes públicos (NTSB, AAIB, CIAIAC, BEA, FAA), por ejemplo los incidentes de pruebas de reparto.
   - Qué falló: percepción, estimación, enlace, energía, software, operación.
   - Una tabla de causas con su fuente.
6. **Demostrar que es seguro.**
   - Regulación como requisito técnico: categorías de EASA y SORA, U-space, BVLOS en EE. UU. (Part 108: comprobar su estado en noviembre de 2026), detección y evitación.
   - La **cuenta**: con cero fallos en n horas, la tasa de fallo se acota por 3/n al 95 % (regla del tres, derivada). Cuántas horas hacen falta para acotar 10⁻⁴, 10⁻⁶ y 10⁻⁷ por hora, y cómo se compara con la experiencia publicada de una flota.
7. **Las cuentas de un sistema real.** Cada una con notación, derivación «paso / por qué» y ejemplo del script, usando especificaciones publicadas:
   - **Distancia de parada** frente a alcance de percepción y latencia, \(d = v\tau + v^2/(2a)\): a qué velocidad puede ir un dron con obstáculos visuales a X m y Y ms (hilo de latencia).
   - **Presupuesto de energía y alcance** de un dron de reparto: multirrotor por teoría de cantidad de movimiento (A1) frente a ala fija con \(P = W v / ((L/D)\,\eta)\).
   - **Deriva sin GNSS:** navegación inercial pura con el sesgo de B3a frente a odometría visual con su deriva por distancia recorrida.
   - **Interferencia del enlace:** por qué un interferidor cercano gana a un enlace lejano. La relación J/S a partir de Friis, en una versión sencilla que no invada C3b.
8. **Conexión con la tesis.** Qué problemas abiertos de la industria son exactamente los de la tesis (belief bajo GNSS degradado, pérdida de enlace, cómputo a bordo) y qué datos públicos de sistemas reales puede usar Pablo.

## 4. Ejecución

### 4.1 Preparación (agente principal, antes de lanzar nada)

1. `git pull` y leer `CLAUDE.md`, `CONTINUAR.md`, `00_plan/ESTADO.md`, `00_plan/TEMARIO.md`, `00_plan/diagnostico.md` y este plan. Mirar M0 (modelo de tema de panorama), M1 (modelo de matemática) y T1b (el tema más reciente).
2. Marcar R1a y R1b como ⏳ en `ESTADO.md`.
3. **Componente de foto en `assets/curso.css`** (no existe ninguno):
   - `figure.foto` con `img` (`object-fit: cover` o `contain`, `max-height` controlable con `style`, bordes del tema) y `figcaption` con la línea de créditos en `.xsmall`.
   - Variante `.foto.duo` para dos fotos lado a lado.
   - Que funcione en tema claro y oscuro y en modo lectura y móvil (`max-width: 100%`).
   - Subir `curso.css` a la versión siguiente a la que haya (el 2026-11-12 era `?v=12`, así que `?v=13`) en **todas** las páginas (los 35 temas, `index.html` y las nuevas). Comprobar con `grep -rL "curso.css?v=13" temas/*/index.html index.html`, que no debe listar nada.
   - Documentar el componente en `CONTINUAR.md` §5.
4. Crear las carpetas `temas/R1a_labs-robot-learning-2026/` y `temas/R1b_drones-autonomos-reales/`, cada una con `img/`, `figuras/` y `lab/`, y un `img/CREDITOS.md` vacío con su cabecera (§5).
5. Preparar las dos órdenes de encargo (§4.2) y lanzar **los dos subagentes en el mismo mensaje**, para que corran en paralelo.

### 4.2 Orden de encargo de cada subagente

Cada subagente recibe solo esto, sin el razonamiento del agente principal:

> Escribe el tema `<ID>` del curso de robótica inteligente de este repositorio (`<ruta absoluta>`).
>
> 1. Lee `CLAUDE.md`, `CONTINUAR.md` (entero), `00_plan/PLAN_R1.md` (entero, sobre todo §3 `<ID>`, §5 y §6) y `00_plan/diagnostico.md`. Usa M0 como modelo de estructura y M1 como modelo de matemática. La parte de `<ID>` en §3 es el contenido mínimo, no un índice cerrado.
> 2. **Trabaja solo en `temas/<carpeta>/`**. No toques `assets/`, `index.html`, `00_plan/`, otros temas ni git (nada de commit, push ni stash). El otro tema lo escribe otro agente a la vez.
> 3. **Investiga antes de escribir** (`CONTINUAR.md` §3, paso 2). Primero las fuentes primarias: papers, blogs técnicos de cada empresa, documentación, informes de accidentes, documentos regulatorios, informes de *think tanks*. Las cifras que solo publica una empresa van como afirmaciones suyas. Guarda el dossier en `<scratchpad>/dossiers/<ID>/`, con cada cifra, su fuente exacta y su estado (confirmado o no). Nada sin confirmar entra en una hoja.
> 4. **Fotos** según `PLAN_R1.md` §5: licencia comprobada en la página del fichero, crédito en el pie y en `img/CREDITOS.md`, comprimidas. Lo que no tenga licencia libre va como enlace.
> 5. Escribe `index.html` (50–65 hojas, con `curso.css?v=13` y `curso.js?v=5`; la cabecera, copiada de T1b), `figuras/` (script que calcula cada figura con datos y cada ejemplo numérico), `lab/README.md` (§6.1 para R1a, §6.2 para R1b), glosario y referencias, con las reglas de `CONTINUAR.md` §4.
> 6. **Comprueba en el navegador** según `CONTINUAR.md` §6, con un servidor en el puerto `<8765 para R1a, 8766 para R1b>`: `checkOverflow()` vacío a 1440×900 en claro y oscuro, sin desborde a 375 px, MathJax sin errores, citas completas, `verificar_glosario.py`, todas las imágenes cargadas (`[...document.images].filter(i=>!i.complete||i.naturalWidth===0)` vacío). Mira cada hoja con fotos en una captura.
> 7. Al acabar, devuelve un informe breve: número de hojas; lista de fotos con su licencia; cifras que se quedaron fuera por no poder verificarse; afirmaciones que van como «según la empresa»; decisiones de diseño; qué enlaces dejas hacia otros temas (C1b, C2b, C5, C7, B7b, D1, D2). No devuelvas el contenido de las hojas.

### 4.3 Revisión, integración y commits (agente principal)

1. **Revisión de cada tema al llegar su informe**, como revisor independiente: no reescribir, comprobar.
   - Todas las fotos: abrir la página de la licencia de cada una y cotejarla con `CREDITOS.md` y el pie.
   - Una muestra de al menos 12 cifras y 8 referencias por tema, contrastadas con su fuente.
   - Rehacer a mano todas las cuentas (cuentas de §3) contra el script.
   - Que no haya solapes con A12a, A12b, A8b ni M0, ni contenido de §3.4 fuera de los límites marcados.
   - Repetir las comprobaciones del navegador.
   - Corregir solo lo comprobado.
2. **Coherencia entre los dos:** R1a debe acabar con el puente a R1b y R1b debe citar R1a donde toque. Las cifras compartidas (por ejemplo, las de Swift o las de cómputo a bordo) tienen que coincidir.
3. **M0, hoja 3:** la figura del plan dice `FASE 5 · J34–46`, `FASE 6 · J47–53` y «50 presentaciones en 53 jornadas». Corregirla a `J34–48`, `J49–55` y 52 en 55, comprobando que las etiquetas no se pisan. Con `checkOverflow()` repetido en M0.
4. **`index.html`:** R1a y R1b pasan a «disponible» con su enlace. **`ESTADO.md`:** ✅, fecha y notas (hojas, fotos, cifras sin verificar), y la revisión anotada.
5. **Limpieza antes del último commit:**
   - Pasar las reglas de fotos (§5) a `CONTINUAR.md` §4, y el componente a §5.
   - Resumir el contenido de §3 en «Contenido obligatorio» de `TEMARIO.md`, al estilo de T1a y T1b.
   - Borrar este fichero y corregir las referencias a él: `grep -rn "PLAN_R1" .` no debe encontrar nada.
6. **Commits, en este orden:**
   - (a) `assets` y el `?v=13` en todas las páginas, con la nota de `CONTINUAR.md`;
   - (b) R1a con `index.html` y `ESTADO.md`;
   - (c) R1b con `index.html`, `ESTADO.md`, el arreglo de M0 y la limpieza del punto 5.

   Push al final. Mensajes en el estilo del repositorio, por ejemplo: `Tema R1a (J36): los labs de robot learning en 2026, …`.
7. **Parada 🛑:** informe a Pablo con lo que pide `PROMPT_CONTINUAR.md` §4 (temas y hojas, cifras no verificadas, decisiones, desviaciones), más la lista de fotos usadas y los casos que se quedaron fuera por falta de fuentes.

### 4.4 Consumo

- Dos subagentes, uno por tema. Ningún subagente lanza otros.
- La investigación de cada subagente es parte de su encargo: no hay agentes de investigación aparte.
- La revisión la hace el agente principal, no un tercer agente.
- Si un subagente falla o se queda corto, el agente principal termina ese tema en lugar de relanzarlo desde cero (`SendMessage` para continuar con su contexto).

## 5. Fotos e imágenes

El repositorio es público, así que cada imagen incrustada se redistribuye. Reglas:

| Fuente | ¿Se incrusta? | Condición |
|---|---|---|
| Wikimedia Commons | Sí | Licencia leída en la página del fichero: CC0, dominio público, CC BY o CC BY-SA. Con CC BY-NC, solo si se anota que el curso no es comercial. Con ND, sin modificar (ni recortar). |
| Obras del Gobierno de EE. UU. (DoD/DVIDS, NASA, FAA) | Sí | Si la página dice dominio público y no hay aviso de copyright de terceros. |
| Figuras de papers | Sí | Solo si el paper tiene licencia CC BY, CC BY-SA o CC0, comprobado en la página de arXiv o de la revista. Se cita el paper y la figura. |
| Capturas propias (SITL, Gazebo, gráficas del curso) | Sí | — |
| Notas de prensa, webs de empresa, fotogramas de YouTube, X o LinkedIn | **No** | Se enlazan: «vídeo oficial: <enlace>», con fecha de consulta. |

- **Formato:** `img/<slug>.webp` o `.jpg`, como mucho 1600 px de ancho y 300 KB por imagen y 6 MB por tema. Convertir con Pillow (`uv run --with pillow`). Comprobar con `file` que lo descargado es una imagen.
- **Pie:** qué se ve y por qué importa, más el crédito en `.xsmall`: «Foto: autor, licencia (enlace), vía fuente».
- **`img/CREDITOS.md`:** una fila por imagen con fichero, qué muestra, autor, licencia con enlace, URL de la página de origen, fecha de consulta y si se ha modificado (recorte o compresión).
- **Las fotos no sustituyen a los esquemas.** Cada sistema importante lleva su diagrama SVG de arquitectura dibujado por el curso. La foto acompaña.
- Una foto sin licencia clara no entra, aunque sea la mejor. Se busca otra o se enlaza.

## 6. Laboratorios y detalles

### 6.1 Lab R1a: auditar los datos y las afirmaciones de un lab (unas 3 h, CPU)

1. Leer **solo los metadatos** (sin descargar vídeo) de entre 4 y 6 datasets públicos del Hugging Face Hub en formato LeRobot o equivalente (DROID, subconjuntos de Open X-Embodiment, AgiBot World u otros): episodios, duración, frecuencia de control, dimensión de acción y cuerpos. Tabla y figura de escalas calculadas.
2. Estadística de evaluación: con el script del tema, el intervalo de Wilson y la potencia para tres resultados publicados con su n, y el número de ensayos para distinguir dos tasas.
3. Aplicar la lista de lectura de demos a dos vídeos oficiales de 2025–2026 y redactar el veredicto.
4. Comprobación numérica: al menos un resultado de las hojas reproducido por el script del lab.

### 6.2 Lab R1b: un sistema real en números (unas 3 h, CPU)

1. Ficha de arquitectura de un sistema real elegido, con fuentes: sensores, cómputo, enlace, autonomía y cifras.
2. Script: envolvente de velocidad segura frente a alcance de percepción y latencia; alcance y energía de un dron de reparto con especificaciones publicadas; deriva sin GNSS.
3. Fiabilidad: horas necesarias para acotar una tasa de fallo con la regla del tres, comparadas con las horas publicadas de una flota.
4. Registros públicos de PX4 (Flight Review) con pérdida o degradación de GNSS: cómo reacciona el EKF, con las herramientas de T1a y T1b.

### 6.3 Detalles

- **IDs y carpetas:** `R1a_labs-robot-learning-2026` y `R1b_drones-autonomos-reales`. Kicker: «Fase 5 · Jornada 36» y «Jornada 37».
- **Hilos obligatorios:** latencia y frecuencia (`.hz`), observabilidad parcial, sim-to-real y cómputo limitado, en los dos.
- **Autoevaluación, lecturas guiadas, glosario y referencias** como en cualquier otro tema. Al menos una pregunta de cuenta en cada test.
- **Fechas:** todo lo que dependa del momento (financiación, estado regulatorio, productos) va con la fecha de la fuente. Se escribe «a noviembre de 2026» solo si se ha comprobado en esa fecha.

## 7. Regla para los temas siguientes: «Así lo hacen en la realidad»

Desde C1a, **cada tema lleva al menos dos hojas sobre un sistema real** (empresa, lab o despliegue) relacionado con su contenido. Cada una con:

- cifras verificadas;
- su foto si hay licencia (§5) y su esquema;
- lo que ese caso enseña del tema.

Ejemplos: C1b con el cómputo de Skydio y de DJI; C2b con la arquitectura de Zipline; C5 con la inferencia de un VLA en un robot de producto; C7 con un informe de accidente; B7b con un despliegue sim-to-real comercial. Los temas pueden remitir a R1a y R1b en lugar de repetirlos. Anotado en `ESTADO.md` (registro de revisiones) y en `CONTINUAR.md` §4.

## 8. Criterios de aceptación

- [ ] `temas/R1a_…/index.html` y `temas/R1b_…/index.html` existen, con 50–65 hojas, portada, objetivos, partes, tesis, resumen, test, lecturas, laboratorio, referencias y glosario.
- [ ] Cada cuenta de §3 está derivada en su hoja (notación, tabla «paso / por qué» y ejemplo del script en `figuras/`).
- [ ] Al menos 8 fotos con licencia libre por tema, todas en `img/CREDITOS.md` y con su crédito en el pie. Ninguna sin licencia comprobada.
- [ ] Al menos cinco sistemas reales con su diagrama de arquitectura en R1b, y al menos cinco organizaciones con sus cifras de datos, cómputo o despliegue en R1a.
- [ ] `checkOverflow()` vacío a 1440×900 en claro y oscuro, sin desborde a 375 px, sin errores de MathJax ni de consola, todas las imágenes cargadas y `verificar_glosario.py` sin fallos, en R1a, R1b y M0.
- [ ] `curso.css?v=13` en todas las páginas.
- [ ] `index.html`, `ESTADO.md`, `TEMARIO.md` y `CONTINUAR.md` actualizados, y este fichero borrado sin referencias colgando. Tres commits subidos.
- [ ] Informe de parada a Pablo.

## 9. Prompt

Se pega en Claude Code, abierto en la carpeta del repositorio.

```text
Crea los temas R1a y R1b del curso de robótica inteligente de este repositorio («Así lo hacen en la realidad»), siguiendo 00_plan/PLAN_R1.md.

1. Haz git pull. Lee, en este orden: CLAUDE.md, CONTINUAR.md, 00_plan/PLAN_R1.md (entero), 00_plan/ESTADO.md, 00_plan/TEMARIO.md (las filas y el contenido obligatorio de R1a y R1b) y 00_plan/diagnostico.md. Abre temas/M0_mapa-del-campo/index.html, temas/M1_herramientas-matematicas/index.html y temas/T1b_taller-simular-la-realidad/index.html como modelos.

2. Preparación (PLAN_R1.md §4.1), hecha por ti antes de lanzar nada: marca R1a y R1b como ⏳ en ESTADO.md; añade el componente de foto (figure.foto y .foto.duo) a assets/curso.css, sube curso.css a ?v=13 en todas las páginas y documéntalo en CONTINUAR.md §5; crea las dos carpetas con img/, figuras/, lab/ y img/CREDITOS.md.

3. Lanza dos subagentes en el mismo mensaje, uno para R1a y otro para R1b, cada uno con la orden de encargo de PLAN_R1.md §4.2 rellenada (ruta del repositorio, carpeta, puerto 8765 o 8766, ruta absoluta del scratchpad para los dossiers). Ningún subagente lanza otros ni toca git, assets/, index.html ni 00_plan/. Mientras trabajan, no hagas tú su trabajo.

4. Cuando llegue cada informe, revisa ese tema como revisor independiente (PLAN_R1.md §4.3, punto 1): la licencia de todas las fotos en su página de origen, una muestra de cifras y referencias contra su fuente, todas las cuentas rehechas contra el script, los límites del apartado de Ucrania y las comprobaciones del navegador de CONTINUAR.md §6 repetidas. Corrige solo lo que compruebes. Si un subagente se queda corto, continúalo con SendMessage o termina tú el tema; no lo relances desde cero.

5. Integra (PLAN_R1.md §4.3, puntos 2 a 4): coherencia entre los dos temas, la figura del plan de M0 (hoja 3) con las jornadas nuevas, index.html y ESTADO.md. Comprueba los criterios de aceptación de PLAN_R1.md §8.

6. Haz la limpieza de PLAN_R1.md §4.3, punto 5 (las reglas que siguen valiendo pasan a CONTINUAR.md y TEMARIO.md, y PLAN_R1.md se borra), y los tres commits del punto 6, en ese orden, y push.

7. Para (🛑 de R1b) y dime: qué temas has hecho, con su número de hojas; qué fotos has usado y con qué licencia; qué cifras o referencias no pudiste verificar y qué casos se quedaron fuera por falta de fuentes; qué decisiones de diseño tomaste; cualquier desviación respecto a PLAN_R1.md y por qué. Después espera mi respuesta. Lo que pida se anota en el registro de revisiones de ESTADO.md y se aplica también a los temas siguientes.

Entre el inicio y la parada no me preguntes nada salvo que estés bloqueado de verdad: las decisiones de profundidad y diseño son tuyas, guiadas por PLAN_R1.md y CONTINUAR.md. Nunca inventes referencias, autores ni cifras; las que solo publica una empresa van como afirmaciones suyas. Ninguna foto entra sin licencia libre comprobada: el repositorio es público.
```
