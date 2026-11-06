# Temario definitivo y plan de jornadas

> Versión 1 · 2026-11-06. Sustituye al temario preliminar de `ANALISIS.md` §6. Construido con el diagnóstico de `diagnostico.md` y los ajustes de votación explicados allí.

## Cifras

- **45 presentaciones** en 7 fases (algunos temas "A fondo" se parten en dos: a y b).
- **48 jornadas de 8 h** (unas 10 semanas a tiempo completo): una por presentación, más 3 dedicadas a hitos y al examen final.
- Cada jornada tipo: unas 3–4 h de presentación con derivaciones, 2–3 h de laboratorio, 1 h de lectura guiada y un test de autoevaluación al final.
- Labs en **Python + JAX**, independientes del hardware concreto (ver "Política de hardware de los laboratorios").

## Cómo encaja el orden

Primero se construye el robot (dinámica, control, estimación), después su simulación, luego se le enseña a aprender (el núcleo de la tesis) y a razonar (foundation models). A continuación se lleva al hardware real y al final se cierra el círculo: SIL/HIL, sim-to-real y el diseño del simulador propio, cuando ya se sabe todo lo que tiene que modelar.

## Proyecto hilo: un dron de extremo a extremo en JAX

Es el miniproyecto integrador y crece con los laboratorios. Al final del curso es la primera versión del simulador y del pipeline de la tesis.

| Hito | Al acabar | Qué existe |
|---|---|---|
| H1 | Fase 1 | Cuadricóptero en JAX con control en cascada a frecuencias reales, MPC, ESKF y planificador |
| H2 | Fase 2 | Capas de realismo: motores, aerodinámica, viento, IMU con varianza de Allan, cámara, latencias y multi-frecuencia; miles de entornos en paralelo |
| H3 | Fase 3 | Políticas con belief, world model y adaptación entrenadas en ese simulador, también en equipo |
| H4 | Fase 5 | Política desplegada con ROS 2 y PX4 SITL, latencias medidas |
| H5 | Fase 6 | Política en una Pixhawk real en hardware-in-the-loop, evaluación contra datos de vuelo públicos y documento de diseño del simulador de la tesis |

## Política de hardware de los laboratorios

- **El curso no depende del hardware de nadie.** Cada laboratorio funciona con un PC con GPU NVIDIA de consumo (8 GB bastan para casi todo) o en la nube, y casi siempre también en CPU, más despacio. Cuando algo necesita más memoria, se indica y se da una alternativa (modelo más pequeño, cuantización o un servicio en la nube).
- **En ningún laboratorio se vuela un dron.** Los datos reales salen de datasets y logs de vuelo públicos.
- **Hardware opcional:** una controladora Pixhawk se usa como hardware-in-the-loop (el firmware real corre en la placa y el simulador hace de mundo). Un ordenador de a bordo tipo Jetson es opcional: los laboratorios de despliegue miden en el PC y comparan con cifras publicadas.

---

## Fase 0 · Arranque (2 jornadas)

| J | ID | Presentación | Profundidad | Lab |
|---|---|---|---|---|
| 1 | M0 | Mapa del campo: la pila de autonomía, los paradigmas y su evolución | A fondo | Entorno de trabajo: uv, JAX en GPU, estructura del proyecto hilo |
| 2 | M1 | Herramientas matemáticas: grupos de Lie, inferencia variacional, teoría de la información | Importante | SO(3)/SE(3) en JAX y un ELBO de juguete |

## Fase 1 · El robot y su control (8 jornadas)

| J | ID | Presentación | Profundidad | Lab |
|---|---|---|---|---|
| 3 | A1 | Dinámica y modelado de multirrotor y ala fija | Repaso con derivaciones | Dinámica del cuadricóptero en JAX (RK4, vmap) |
| 4 | A2 | Control clásico y jerarquía de lazos | Importante | PID en cascada a frecuencias reales; separación de escalas y retardos |
| 5 | A3a | Control óptimo: LQR, NMPC, MPCC, real-time iteration | A fondo | NMPC con acados para seguimiento |
| 6 | A3b | MPC por muestreo (MPPI, CEM), MPC con aprendizaje, MPC frente a RL | A fondo | MPPI vectorizado en JAX |
| 7 | A4a | Filtrado bayesiano: de Bayes a KF, EKF, ESKF, UKF y filtro de partículas | A fondo | ESKF de IMU y posición; comparación con PF |
| 8 | A4b | Percepción para estimación: VIO, SLAM, cámaras de eventos, filtros aprendidos | A fondo | Estimación visual-inercial y efecto del desfase temporal |
| 9 | A5 | Planificación y navegación | Importante | Minimum snap y planificador local con obstáculos |
| 10 | H1 | Hito 1 y test de fase | — | Integración del hito H1 |

## Fase 2 · Simulación: modelar la realidad (6 jornadas)

| J | ID | Presentación | Profundidad | Lab |
|---|---|---|---|---|
| 11 | B1 | Panorama de simuladores | A fondo | El mismo dron en dos simuladores; comparar |
| 12 | B2 | Física fiel: rotores, motor, ESC, batería, viento, identificación, modelos híbridos | Importante | Dinámica de motor, drag y viento; identificación desde un dataset de vuelo público |
| 13 | B3a | Modelos de sensores: IMU, cámaras, eventos, LiDAR, GNSS | A fondo | Varianza de Allan de datos de IMU públicos y modelo de IMU |
| 14 | B3b | Render y real2sim: motores gráficos, NeRF, Gaussian Splatting | A fondo | Observaciones del dron desde una escena 3DGS |
| 15 | B4 | Tiempo, asincronía y latencia en simulación | A fondo | Multi-frecuencia, buffers de retardo y jitter en el simulador JAX |
| 16 | B6 | Simulación masiva y diferenciable | A fondo | Miles de entornos en paralelo y gradientes a través del simulador |

## Fase 3 · Aprender a controlar (12 jornadas, núcleo de la tesis)

| J | ID | Presentación | Profundidad | Lab |
|---|---|---|---|---|
| 17 | A6a | Deep RL para control: PPO, SAC, GAE, entrenamiento masivo, recompensas | A fondo | PPO en JAX: hover y seguimiento |
| 18 | A6b | RL en robots reales: teacher-student, asymmetric actor-critic, espacios de acción | A fondo | Teacher privilegiado y student con observaciones ruidosas |
| 19 | A7a | Teoría de POMDPs: belief MDP, estados de información, planificación | A fondo | POMDP de juguete: belief exacto frente a POMCP |
| 20 | A7b | Deep RL bajo observabilidad parcial: memoria, belief aprendido, percepción activa | A fondo | Recurrente, belief explícito y belief aprendido en búsqueda con el dron |
| 21 | A8a | Model-based RL y world models latentes: de PILCO a Dreamer y TD-MPC2 | A fondo | RSSM mínimo en JAX |
| 22 | A8b | World models fundacionales: JEPA, modelos de vídeo, world models como simuladores | A fondo | Planificación latente tipo TD-MPC2 en el dron |
| 23 | A9a | Adaptación online y meta-learning: control adaptativo, RMA, Neural-Fly, meta-RL | A fondo | Adaptación a masa y viento desconocidos |
| 24 | A9b | Continual learning: olvido, pérdida de plasticidad, aprendizaje durante el despliegue | A fondo | Secuencia de tareas; medir olvido y plasticidad |
| 25 | A10 | Imitation learning y políticas generativas: BC, DAgger, ACT, diffusion, flow matching | Importante | Destilar un experto MPC en una política de flow matching |
| 26 | A15a | MARL: Dec-POMDP, CTDE, factorización de valor, MAPPO, comunicación | A fondo | MAPPO con varios drones en búsqueda |
| 27 | A15b | Enjambres y multi-robot real: ad hoc teamwork, adversarios, planificación descentralizada | A fondo | Compañeros desconocidos en la tarea de búsqueda |
| 28 | H3 | Hito 3 y test de fase | — | Integración del hito H3 |

## Fase 4 · Razonar: foundation models y arquitecturas (5 jornadas)

| J | ID | Presentación | Profundidad | Lab |
|---|---|---|---|---|
| 29 | A11 | LLMs y VLMs para razonar y planificar | A fondo | VLM local como planificador de misión |
| 30 | A12a | VLAs: fundamentos, arquitecturas, tokenización de acciones, datos | A fondo | Inferencia y latencia de un VLA pequeño |
| 31 | A12b | VLAs en 2025–2026, cross-embodiment y VLAs para drones | A fondo | Ajuste fino ligero de un VLA con datos del simulador |
| 32 | A13 | Arquitecturas jerárquicas y fusión de modelos | A fondo | System 2 lento + System 1 rápido con inferencia asíncrona |
| 33 | A14 | Seguridad y garantías | Importante | Filtro de seguridad CBF sobre la política aprendida |

## Fase 5 · Desplegar: hardware y sistema (8 jornadas)

| J | ID | Presentación | Profundidad | Lab |
|---|---|---|---|---|
| 34 | C1 | Cómputo embarcado | A fondo | Perfilado de latencia y memoria de un modelo; comparación con cifras publicadas de plataformas embarcadas |
| 35 | C2 | Arquitectura del sistema | A fondo | Arquitectura del dron de la tesis con presupuesto de latencias |
| 36 | C3 | Comunicaciones: buses, MAVLink, DDS, radio, sincronización | Importante | MAVLink contra PX4 SITL; medir latencias |
| 37 | C8 | ROS 2 a fondo | A fondo | Nodos, QoS, tf2, rosbag2/MCAP y puente con PX4 por uXRCE-DDS |
| 38 | C4 | Tiempo real en robots | A fondo | Latencia y jitter con kernel normal frente a PREEMPT_RT; executors |
| 39 | C5 | Inferencia en el borde | Importante | Exportar a ONNX, cuantizar y medir latencia (TensorRT si hay GPU NVIDIA) |
| 40 | C6 | Sensores, actuadores y calibración | A fondo | Calibración cámara-IMU con un dataset público; modelo de motor-hélice a partir de datos de banco publicados |
| 41 | C7 + H4 | Operación y experimentación real, e hito 4 | Repaso | Análisis de un ulog e integración del hito H4 |

## Fase 6 · Cerrar el círculo (7 jornadas)

| J | ID | Presentación | Profundidad | Lab |
|---|---|---|---|---|
| 42 | B5 | SIL, HIL y realidad mixta | A fondo | PX4 SITL y HITL con la controladora real |
| 43 | B7a | Sim-to-real: taxonomía de gaps y técnicas | A fondo | Ablación: qué capa de realismo importa más |
| 44 | B7b | Sim-to-real en la práctica: casos y protocolo | A fondo | Política en una Pixhawk en HIL y evaluación contra datos de vuelo públicos |
| 45 | B8 | Diseñar tu propio simulador | A fondo | Documento de diseño del simulador de la tesis |
| 46 | D1 | Datasets, benchmarks y evaluación | A fondo | Protocolo de evaluación de la tesis |
| 47 | D2 | Investigar en robótica hoy | A fondo | Mapa de lecturas y plan del primer año |
| 48 | Final | Examen final | — | Los 7 criterios de `ANALISIS.md` §3 |

---

## Ruta crítica

Si en algún momento hay que priorizar, estas 16 presentaciones son las que más pesan para la tesis: M0, M1, A2, A4a, A6b, A7a, A7b, A8a, A9a, A13, B4, B6, B7a, B8, C2 y C8.

## Pista paralela recomendada: C++

No forma parte del curso, pero PX4, ROS 2 y casi todo el código de tiempo real están en C++. Conviene ir practicándolo en paralelo, en torno a 30 minutos al día, para llegar a la Fase 5 leyendo C++ con soltura. Los recursos concretos se proponen en M0.

## Entregables por tema

Para cada ID:

- `temas/<ID>_<slug>/index.html`: la presentación.
- `temas/<ID>_<slug>/lab/`: el laboratorio (enunciado y código base).
- El test de autoevaluación va dentro de la presentación, con las respuestas ocultas hasta pulsar.
- Las lecturas guiadas van en la propia presentación: qué leer, qué buscar y qué preguntas responder.
