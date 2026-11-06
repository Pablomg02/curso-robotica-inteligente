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
| A4a | Filtrado bayesiano: de Bayes a Kalman y partículas | ⏳ | | |
| A4b | Percepción para estimación: VIO, SLAM, eventos, filtros aprendidos | ⬜ | | |
| A5 | Planificación y navegación | ⬜ | | |
| H1 | Hito 1 y examen de fase | ⬜ 🛑 | | Fin de fase 1 |

## Fase 2 · Simulación

| ID | Tema | Estado | Fecha | Notas |
|---|---|---|---|---|
| B1 | Panorama de simuladores | ⬜ | | |
| B2 | Física fiel | ⬜ | | |
| B3a | Modelos de sensores | ⬜ | | |
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
