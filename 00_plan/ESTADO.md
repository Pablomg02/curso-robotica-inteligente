# Estado del curso

Registro de lo que está hecho, lo que falta y dónde parar para revisión. **Se actualiza al terminar cada tema**, en el mismo commit que el tema.

Leyenda: ✅ hecho · ⏳ en curso · ⬜ pendiente · 🛑 punto de parada para revisión de Pablo (tras terminar ese tema)

## Fase 0 · Arranque

| ID | Tema | Estado | Fecha | Notas |
|---|---|---|---|---|
| M0 | Mapa del campo | ✅ | 2026-11-06 | 55 hojas + lab 0. Plantilla de referencia. |
| M1 | Herramientas matemáticas | ✅ | 2026-11-06 | 44 hojas + lab. Modelo para temas con muchas derivaciones. |

## Fase 1 · El robot y su control

| ID | Tema | Estado | Fecha | Notas |
|---|---|---|---|---|
| A1 | Dinámica y modelado de multirrotor y ala fija | ✅ 🛑 | 2026-11-06 | 49 hojas + lab. Primera parada: revisar el formato en un tema técnico. Figuras y cifras calculadas en `figuras/figuras_a1.py`. Revisado tras la parada con verificación independiente: Cheeseman y Bennett, ec. (2) del informe original; Euler (ecuaciones de 1750, publicadas en 1752; *Theoria motus*, 1765); Rankine (1865) y R. E. Froude (1889); Lanchester (1908) y Bryan (1911); leyes de escala de Kumar y Michael en la versión de ISRR 2011 (empuje y par, cuenta nuestra); numeración de motores de PX4 y Betaflight; masa y columnas de NeuroBEM (`vbat` en décimas de voltio aunque el Readme diga voltios); figura de mérito corregida a 0,5–0,6 con medidas de Deters y Selig (2008). Sin acceso al texto completo de Mahony et al. (2012): solo se cita como tutorial general y la lectura esencial pasa a Martin y Salaün (2010, acceso abierto). Beard y McLain no publican trim ni autovalores del Aerosonde: los de la presentación son cálculo propio. Motor: arreglo de `assets/` (css v=7, js v=3); corregidas dos fórmulas cortadas de M1 (hojas 30 y 34). Corregido con A2: inercia compuesta del x500 (cuerpo + rotores) y RAPTOR citado en su versión de *Science Robotics* (2026). |
| A2 | Control clásico y jerarquía de lazos | ✅ | 2026-11-06 | 41 hojas + lab. Cifras y figuras en `figuras/figuras_a2.py`; parámetros y leyes de PX4 leídos en su código (main 95a0c23): lazo de *rate* a 800 Hz en multirrotor (`rc.mc_defaults`), filtros, anti-windup, error de cuaternión, normalización del asignador. Frecuencias de Betaflight, ArduPilot y Crazyflie leídas en su código. Sin acceso al original: la ley L1 de Park et al. (contrastada en tres fuentes secundarias, se dice en la hoja), el factor de separación de Beard y McLain (cifra de su borrador de 2.ª edición, se dice en la hoja), el texto de Brescianini y D'Andrea (2020) (cifras de la tesis y del informe de 2013) y los artículos de Lambregts (1983). Supuestos marcados: retardo de muestreo y cómputo de 1,5 periodos; motor como par efectivo de primer orden. Investigación hecha con agentes en paralelo; la revisión independiente encontró, entre otras cosas, que la inercia del x500 debía incluir los rotores (0,0239 en vez de 0,0217), corregido en A2 y en A1 (tabla de parámetros, ley de escala: ahora 125 rad/s² y 159 ms), y que el asignador de PX4 sin *airmode* baja el empuje antes de recortar el par. |
| A3a | Control óptimo: LQR, NMPC, MPCC, RTI | ✅ | 2026-11-06 | 41 hojas + lab. Cifras y figuras en `figuras/figuras_a3a.py`: LQR del x500, convergencia de Riccati, MPC con límite de inclinación frente a LQR saturado (resuelto con un ADMM propio), coste de Riccati frente a condensado y RTI frente a SQP convergido (Gauss-Newton con disparo simple en el cuadricóptero plano). Sin acceso al original: los axiomas de Mayne et al. (2000) (se cita el libro de Rawlings, Mayne y Diehl), los artículos de Richalet (1978), Bock y Plitt (1984) y Lam et al. (2010) (datos bibliográficos y resúmenes); dos preprints (Wikner et al., 2026; Hirst et al., 2025) marcados como tales. Investigación con los dos dossiers ya hechos; revisión independiente acotada. |
| A3b | MPC por muestreo, MPC con aprendizaje, MPC frente a RL | ✅ | 2026-11-06 | 38 hojas + lab. Cifras y figuras en `figuras/figuras_a3b.py`: aleatorio, CEM y MPPI en un escenario 2D con obstáculos (20 semillas) y tamaño efectivo de muestra frente a la temperatura. Sin acceso al texto completo: la revisión de Hewing et al. (2020) (solo su resumen y taxonomía); el ICRA 2016 de MPPI (se citan las cifras del T-RO 2018). Preprints marcados: MPPI-Generic y Dream to Fly (aceptado en ICRA 2026 según arXiv). Investigación con el dossier ya hecho; sin revisión independiente (tema sin derivaciones pesadas), revisado con capturas. |
| A4a | Filtrado bayesiano: de Bayes a Kalman y partículas | ✅ | 2026-11-06 | 35 hojas + lab. Cifras y figuras en `figuras/figuras_a4a.py`: filtro de Bayes en rejilla, NEES de un Kalman bien y mal ajustado, EKF/UKF/PF en un modelo bimodal, ESKF de actitud con y sin magnetómetro (inconsistencia en la guiñada), umbrales χ² y filtro complementario como Kalman estacionario. EKF2 de PX4 leído en su documentación y su código. Referencias verificadas en Crossref, OpenAlex y NTRS; sin acceso al texto completo de varias (se citan por su resumen). Sin agentes de investigación; revisión independiente acotada aplicada (dualidad con el LQR, resultado de la guiñada comprobado con 6 semillas, tolerancias del laboratorio). |
| A4b | Percepción para estimación: VIO, SLAM, eventos, filtros aprendidos | ✅ | 2026-11-06 | 34 hojas + lab. Cifras y figuras en `figuras/figuras_a4b.py`: efecto y estimación en línea del desfase cámara-IMU, error de reproyección equivalente e incertidumbre estéreo con los intrínsecos y la base de EuRoC (leídos en la configuración de OpenVINS). Un agente de investigación acotado (32 llamadas). Cifras de preprint marcadas (PL-EVIO). Sin revisión independiente (tema con pocas derivaciones nuevas): revisado con capturas. |
| A5 | Planificación y navegación | ✅ | 2026-11-07 | 39 hojas + lab. Cifras y figuras en `figuras/figuras_a5.py`: A*, RRT y RRT* en un mapa 2D (con recorte por visibilidad), minimum snap por KKT en tiempo normalizado (ley de escala ×128, reparto de tiempos optimizado; el óptimo sale continuo hasta la derivada 6 sin imponerlo), optimización local sobre un campo de distancias, distancia de parada y caminos de Dubins (los seis tipos, con error de cierre nulo) y radio sobre el suelo con viento. Parámetros de PX4 (suavizado con *jerk* limitado, prevención de colisiones) leídos en su código (main 95a0c23). Un agente de investigación acotado (33 llamadas); detalles de Richter et al., Foehn et al., EGO-Planner, Loquercio et al. y Zhang et al. comprobados en sus PDF. Sin verificar y por eso sin cifras: el enjambre de Zhou et al. (2022), de pago. Cifras de Zhang et al. (2025) tomadas del preprint, marcado. El nombre MINCO sale del README de GCOPTER, no del artículo. De paso, corregidas clases SVG inexistentes (`fill-muted`, `fill-faint`) en A2, A3b y A4a. Sin revisión independiente (derivaciones cortas y comprobadas numéricamente en el script), revisado con capturas. |
| H1 | Hito 1 y examen de fase | ✅ 🛑 | 2026-11-07 | Fin de fase 1. 20 hojas. Enunciado de integración del proyecto hilo (x500 a frecuencias reales, ESKF con medida retrasada, planificador con sensor de alcance limitado, tres seguidores intercambiables), interfaces, 10 criterios de aceptación con umbrales del curso (marcados como orientativos), errores típicos y plan de trabajo. Examen de 14 preguntas (8 de opción múltiple, 4 de cuentas y 2 abiertas, sobre 16) con tabla de qué repasar según los fallos. Las cifras salen de los temas A1–A5; los límites χ² del NEES, calculados con scipy. |

## Fase 2 · Simulación

| ID | Tema | Estado | Fecha | Notas |
|---|---|---|---|---|
| B1 | Panorama de simuladores | ✅ | 2026-11-07 | 41 hojas + lab. Cifras y figuras en `figuras/figuras_b1.py`: error de integración frente a coste en el x500 en bucle cerrado (Euler explícito, semiimplícito y RK4; la saturación de los motores baja RK4 de orden 4 a orden 2, comprobado quitándola), el mismo x500 y el mismo controlador con tres convenciones de planta leídas en el código (empuje instantáneo de MuJoCo Menagerie, `MulticopterMotorModel` de Gazebo, SIH de PX4) y cifras de rendimiento publicadas. Leído en el código (noviembre de 2026): paso del mundo de PX4 en Gazebo (4 ms), motor y arrastre del x500, DART como motor por defecto de gz-sim, bucle y modelo de SIH, `timestep` e integradores de MuJoCo, Menagerie, Isaac Lab 3.0, MuJoCo 4.0, Newton, estado de mantenimiento de los repositorios. Un agente de investigación acotado (41 llamadas): cifras de Flightmare, gym-pybullet-drones, OmniDrones, Aerial Gym, Isaac Gym, MuJoCo Playground, FlightGoggles y AirSim leídas en sus versiones de arXiv (no comparadas con las finales). Marcados: la cifra de 43 M FPS de Genesis (afirmación del equipo) y la réplica de Tao (blog, autor de un simulador competidor); preprints (VisFly, Isaac Lab, MuJoCo Playground). Sin verificar y por eso sin cifras: el contenido de CrazySim y el texto completo de Collins et al. (solo su resumen). Sin revisión independiente (pocas derivaciones, comprobadas en el script), revisado con capturas. |
| B2 | Física fiel | ✅ | 2026-11-07 | 40 hojas + lab. Cifras y figuras en `figuras/figuras_b2.py`: empuje a Ω fija con una BEM idealizada (subida, avance con disco horizontal e inclinado; coincide con el ajuste de Faessler et al. a menos del 1 % entre 2 y 5 m/s y se separa por encima), motor eléctrico con y sin frenado activo (τ = 11 ms; la asimetría de Gazebo sale del modelo), descarga de la batería en estacionario (margen de empuje de 1,63 a 1,20), turbulencia de Dryden validada por su espectro y su efecto según el arrastre, identificabilidad por mínimos cuadrados y extrapolación de caja gris frente a caja negra. Parámetros supuestos (pala, motor eléctrico, batería de 5 Ah) marcados en el script y en las hojas. Leído en el código: `THR_MDL_FAC`, estimación de batería y `MC_BAT_SCALE_EN` de PX4, `WindEffects` de Gazebo. Un agente de investigación acotado (39 llamadas): NeuroBEM, Torrente, Saviolo, KNODE-MPC, Neural-Fly, Neural Lander, Chen y Rincón-Mora y Bauersfeld y Scaramuzza leídos en texto completo. Solo el resumen: Svacha et al., Gill y D'Andrea, Blackbird, Pan et al. y AI-IO. Dryden: fórmulas de MIL-F-8785C tal como las reproduce MathWorks (normas originales no leídas, se dice en la hoja). Preprint marcado: Sym2Real. Sin revisión independiente (derivaciones comprobadas numéricamente en el script), revisado con capturas. |
| B3a | Modelos de sensores | ✅ | 2026-11-07 | 36 hojas + lab. Cifras y figuras en `figuras/figuras_b3a.py`: desviación de Allan de un giróscopo simulado de 12 h con los valores de TUM VI frente al del x500 de Gazebo (solo ruido blanco), deriva de la navegación inercial pura término a término (fórmulas comprobadas por Monte Carlo), obturador rodante y desenfoque, eventos de un borde, rayos sobre obstáculos finos y modelos de error GNSS. Leído en el código: ruido de IMU, barómetro, magnetómetro y GNSS de PX4 y Gazebo (el ruido «realista» de SIH está desactivado; el GNSS simulado tiene 2,5 cm de error y declara 0,9 m), filtros de giróscopo, zona muerta del barómetro y compensación del magnetómetro. Valores de Allan de TUM VI de su página y del PDF del artículo (BMI160, obturador global). Un agente de investigación acotado (49 llamadas). Pendientes de Allan leídas en la tesis de Hou (El-Sheimy et al. no se leyó: solo se cita para profundizar). Hojas de datos como afirmaciones del fabricante (BMI088, ICM-42688-P leída en una copia de terceros, u-blox). Sin verificar y por eso sin cifras: ADIS16448 (web del fabricante inaccesible), precisión de LiDAR comerciales, constante de tiempo del error GNSS (el Gauss-Markov de ejemplo está marcado como supuesto). Preprints o venue declarada por los autores marcados: EQS, SimULi. Sin revisión independiente, revisado con capturas. |
| B3b | Render y real2sim | ⬜ | | |
| B4 | Tiempo, asincronía y latencia en simulación | ⬜ | | |
| B6 | Simulación masiva y diferenciable | ⬜ 🛑 | | Fin de fase 2 |

## Fase 3 · Aprender a controlar

| ID | Tema | Estado | Fecha | Notas |
|---|---|---|---|---|
| A6a | Deep RL para control | ⬜ | | |
| A6b | RL en robots reales | ⬜ | | |
| A7a | Teoría de POMDPs | ⬜ | | |
| A7b | Deep RL bajo observabilidad parcial | ⬜ | | |
| A8a | Model-based RL y world models latentes | ⬜ | | |
| A8b | World models fundacionales | ⬜ 🛑 | | Mitad de fase 3: núcleo de la tesis |
| A9a | Adaptación online y meta-learning | ⬜ | | |
| A9b | Continual learning | ⬜ | | |
| A10 | Imitation learning y políticas generativas | ⬜ | | |
| A15a | MARL | ⬜ | | |
| A15b | Enjambres y multi-robot real | ⬜ | | |
| H3 | Hito 3 y examen de fase | ⬜ 🛑 | | Fin de fase 3 |

## Fase 4 · Razonar

| ID | Tema | Estado | Fecha | Notas |
|---|---|---|---|---|
| A11 | LLMs y VLMs para razonar y planificar | ⬜ | | |
| A12a | VLAs: fundamentos | ⬜ | | |
| A12b | VLAs en 2025–2026 y VLAs para drones | ⬜ | | |
| A13 | Arquitecturas jerárquicas y fusión de modelos | ⬜ | | |
| A14 | Seguridad y garantías | ⬜ 🛑 | | Fin de fase 4 |

## Fase 5 · Desplegar

| ID | Tema | Estado | Fecha | Notas |
|---|---|---|---|---|
| C1 | Cómputo embarcado | ⬜ | | |
| C2 | Arquitectura del sistema | ⬜ | | |
| C3 | Comunicaciones | ⬜ | | |
| C8 | ROS 2 a fondo | ⬜ | | |
| C4 | Tiempo real en robots | ⬜ | | |
| C5 | Inferencia en el borde | ⬜ | | |
| C6 | Sensores, actuadores y calibración | ⬜ | | |
| C7 | Operación y experimentación real + hito 4 | ⬜ 🛑 | | Fin de fase 5 |

## Fase 6 · Cerrar el círculo

| ID | Tema | Estado | Fecha | Notas |
|---|---|---|---|---|
| B5 | SIL, HIL y realidad mixta | ⬜ | | |
| B7a | Sim-to-real: taxonomía y técnicas | ⬜ | | |
| B7b | Sim-to-real en la práctica | ⬜ | | |
| B8 | Diseñar tu propio simulador | ⬜ | | |
| D1 | Datasets, benchmarks y evaluación | ⬜ | | |
| D2 | Investigar en robótica hoy | ⬜ | | |
| Final | Examen final | ⬜ 🛑 | | Fin del curso |

## Registro de revisiones de Pablo

Anota aquí cada revisión: fecha, qué temas revisó y qué pidió cambiar. Lo pedido se aplica también a los temas siguientes.

| Fecha | Temas revisados | Cambios pedidos | Aplicado |
|---|---|---|---|
| 2026-11-06 | Temario y hardware | Laboratorios independientes del hardware; nunca volar un dron; Pixhawk solo como HIL opcional | Sí |
| 2026-11-06 | A1 | Corregir lo que quedó dudoso en A1. No parar tras cada tema: solo al acabar la fase 1 (H1) y la fase 2 (B6). Paralelizar el trabajo con subagentes siempre que no reste calidad. | Sí: fuentes pendientes de A1 verificadas con agentes; investigación de cada tema en paralelo y revisión independiente antes del commit (`CONTINUAR.md` §3) |
| 2026-11-06 | — | El ritmo de consumo de IA es alto: usar los subagentes con más cuidado. | Sí: como mucho un agente de investigación por tema, con encargo acotado; consultas de código y documentación hechas directamente; revisión independiente solo en temas con muchas derivaciones (`CONTINUAR.md` §3.1) |
| 2026-11-07 | Fase 1 (A1–A5, H1) | «Se ve bien». Seguir hasta el final sin parar, revisando la calidad a medida que se avanza: completo, riguroso y de la máxima calidad. | Sí: las paradas 🛑 pasan a ser revisiones propias de fase (relectura de las hojas, comprobación de cifras y referencias cruzadas, anotadas en este fichero) sin esperar respuesta; se mantiene la política de agentes del 2026-11-06 |
