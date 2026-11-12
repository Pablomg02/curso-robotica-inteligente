# Temario definitivo y plan de jornadas

> Versión 1 · 2026-11-06. Sustituye al temario preliminar de `ANALISIS.md` §6. Construido con el diagnóstico de `diagnostico.md` y los ajustes de votación explicados allí.

## Cifras

- **52 presentaciones** en 7 fases (algunos temas "A fondo" se parten en dos: a y b; el taller práctico T1 y los casos reales R1 también).
- **55 jornadas de 8 h** (unas 11 semanas a tiempo completo): una por presentación, más 3 dedicadas a hitos y al examen final.
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

## Fase 5 · Desplegar: hardware y sistema (15 jornadas)

> Ampliada el 2026-11-10 a petición de Pablo: la arquitectura de un dron real (Pixhawk, PX4, MAVLink) tiene que quedar explicada a fondo, y también el cómputo (CPU, GPU, NPU), los protocolos y todo lo demás que hace falta para desplegar. Por eso C1, C2 y C3 se parten en dos. El contenido mínimo de cada tema está en «Contenido obligatorio del bloque C», más abajo. Los temas no se quedan en esa lista: es lo mínimo que tienen que cubrir.
>
> Ampliada otra vez el 2026-11-10 con el **taller práctico T1** (dos jornadas al empezar la fase), también a petición de Pablo: la pila de vuelo real (PX4, Gazebo, SITL, QGroundControl, MAVLink, MAVSDK, ROS 2) montada y medida en su PC, y el estado del arte en simular la realidad. Los temas C que vienen después se apoyan en lo que el taller deja montado y medido, y no lo repiten.
>
> Ampliada por tercera vez el 2026-11-12 con **R1a y R1b, «Así lo hacen en la realidad»** (J36–37), a petición de Pablo: faltaba la actualidad de la industria y de los labs (DeepMind, Physical Intelligence y compañía) con casos, fotos y figuras reales. Plan completo en [`PLAN_R1.md`](PLAN_R1.md). Desde C1a, cada tema lleva además al menos dos hojas sobre un sistema real (`PLAN_R1.md` §7).

| J | ID | Presentación | Profundidad | Lab |
|---|---|---|---|---|
| 34 | T1a | Taller I: la pila de vuelo, simulada de punta a punta | Práctico, a fondo | Contenedor con PX4 v1.17 SITL, Gazebo, ROS 2 Jazzy y MAVSDK 4; vuelos, registros y fallos medidos |
| 35 | T1b | Taller II: simular la realidad en 2026 | Práctico, a fondo | Registros públicos de un X500 V2 frente al SITL; Gazebo más real; cámara de splats con PX4 SITL; la red de PX4; SRCC |
| 36 | R1a | Así lo hacen en la realidad I: los labs de robot learning en 2026 | A fondo | Auditar los datos y las afirmaciones de un lab: metadatos de datasets públicos, Wilson y potencia de resultados publicados, lectura de demos |
| 37 | R1b | Así lo hacen en la realidad II: drones autónomos en el mundo real | A fondo | Un sistema real en números: ficha de arquitectura, velocidad segura, alcance, deriva sin GNSS, fiabilidad y registros públicos con GNSS degradado |
| 38 | C1a | Cómputo I: de la MCU a la GPU | A fondo | Roofline de tu GPU y de tu CPU con un microbenchmark; una MLP y una CNN pequeñas situadas en él |
| 39 | C1b | Cómputo II: NPUs, aceleradores y plataformas embarcadas | A fondo | Perfilado de latencia, memoria y energía estimada de un modelo; comparación con cifras publicadas de Jetson, NPUs y MCUs |
| 40 | C2a | El autopiloto por dentro: Pixhawk, PX4 y ArduPilot | A fondo | Registros públicos de Pixhawk reales frente al SITL de T1a: drivers, frecuencias, estados del *commander* y *failsafes*; ArduPilot SITL al lado |
| 41 | C2b | Arquitectura del sistema completo | A fondo | Arquitectura del dron de la tesis con presupuesto de latencias, cómputo, energía y peso |
| 42 | C3a | Buses y protocolos a bordo | Importante | Decodificar tramas (DShot, CRSF, CAN) a partir de capturas públicas o generadas; ancho de banda y latencia calculados |
| 43 | C3b | MAVLink, DDS, radio y sincronización | A fondo | MAVLink byte a byte (trama, CRC, firma) con pymavlink; enlace con pérdidas y retardo emulados sobre el SITL de T1a; desfase de relojes |
| 44 | C8 | ROS 2 a fondo | A fondo | Executors, tf2, launch, rosbag2/MCAP y ros2_control sobre el puente con PX4 de T1a; rendimiento y copia cero |
| 45 | C4 | Tiempo real en robots | A fondo | Latencia y jitter con kernel normal frente a PREEMPT_RT; executors |
| 46 | C5 | Inferencia en el borde | A fondo | Exportar a ONNX, cuantizar y medir latencia (TensorRT si hay GPU NVIDIA) |
| 47 | C6 | Sensores, actuadores, potencia y calibración | A fondo | Calibración cámara-IMU con un dataset público; modelo de motor-hélice a partir de datos de banco publicados |
| 48 | C7 + H4 | Operación y experimentación real, e hito 4 | Repaso | Análisis de un ulog e integración del hito H4 |

### Contenido obligatorio del bloque C

Cada punto se explica desde cero (Pablo es débil en la pila del robot real, `diagnostico.md`), con dibujos de la arquitectura, cifras verificadas en documentación oficial, hojas de datos o código, y su efecto sobre latencia, observabilidad, sim-to-real y cómputo limitado.

- **T1a · Taller I: la pila de vuelo, simulada.** Todo medido en el PC del curso y con cifras de ejecuciones reales.
  - El mapa: procesos, puertos (14550, 14540, 8888), SITL frente a SIH y HITL, versiones de noviembre de 2026, instalación en contenedor, el arranque (`rcS`, airframe, `px4-rc.gzsim`) cronometrado.
  - PX4 por dentro en SITL: uORB y *work queues* con sus frecuencias reales, la fase que cuesta ir a 250 Hz y no a 800, la consola, el lockstep con el reloj de Gazebo y el factor de velocidad (tope y varios drones).
  - Gazebo por dentro: ECS, pasos de la simulación, el SDF del x500, gz-transport y el puente; FLU/FRD, ENU/NED y la trampa del magnetómetro, derivados; mundos y cámaras y lo que cuestan.
  - QGroundControl, MAVLink (flujos y ancho de banda), MAVSDK 4 (sin gRPC), offboard (círculo con y sin prealimentación derivado; el EKF adelantado por el retardo de GNSS supuesto), escalón, misión, pymavlink.
  - ROS 2 y uXRCE-DDS: cliente y agente (la trampa del agente 3.x), nombres con versión, QoS, frecuencias y edades medidas, *timesync* (NTP derivado) y tiempo simulado, offboard desde ROS 2 y su ACK, modos externos.
  - Registros y fallos: el ulog, lo que no se registra, EKF frente a la verdad de Gazebo, inyección de fallos (qué funciona de verdad con Gazebo), pérdida de GNSS y de un motor medidas, SITL como banco de pruebas automático; lo que el SITL no simula y la comparación *sim-to-sim* con el simulador JAX.
- **T1b · Taller II: simular la realidad en 2026.**
  - La brecha medida con registros públicos de un X500 V2 con la misma versión de PX4: dispersión de sensores, espectro del giróscopo con la vibración de los motores, muestreo y pliegue, empuje de estacionario; la PSD del ruido blanco y el filtro del integrador de PX4, derivados.
  - Cerrar la brecha en Gazebo: ruido con sesgo y deriva en el SDF, IMU a 1 kHz y su coste, el viento del mundo `windy` (derivado y medido), qué identificar con datos reales.
  - Cámara neuronal: splats gaussianos conectados a PX4 SITL, su coste y su latencia, sus límites y el estado del arte en drones (SOUS VIDE, GRaD-Nav, FalconGym 2.0, VizFlyt…).
  - Isaac Sim 6.2, Isaac Lab 3.0 (Newton, sin Kit), Pegasus y su acoplamiento con PX4, el mapa de simuladores con PX4 y teselas geoespaciales; qué simulador para cada paso de la tesis.
  - Modelos generativos del mundo (Cosmos, Genie) y su uso en drones como fábricas de datos y representaciones.
  - La red neuronal dentro de PX4 v1.17 (`mc_nn_control`): arquitectura, la cadena de la salida al motor derivada, vuelo en SITL y cómputo en la placa.
  - Medir la fidelidad: métricas por eje, SRCC y el intervalo de Wilson derivados, y la escalera de simuladores de la tesis.
- **R1a · Los labs de robot learning en 2026** y **R1b · Drones autónomos en el mundo real.** El contenido mínimo, las cuentas obligatorias, la política de fotos y los laboratorios están en [`PLAN_R1.md`](PLAN_R1.md) §3, §5 y §6. Los temas C que vienen después retoman los casos de R1b (C1b el cómputo, C2b la arquitectura, C7 los accidentes) sin repetirlos.
- **Qué cambia para los temas C y B5 tras el taller.** C2a explica PX4 en la placa (NuttX, drivers, hardware Pixhawk, ArduPilot) partiendo del SITL que T1a ya recorrió; C3b baja MAVLink al byte, la radio y la sincronización, sin repetir los flujos medidos en T1a; C8 profundiza en ROS 2 (executors, tf2, ros2_control, copia cero) sobre el puente que T1a dejó montado; B5 se centra en HITL y realidad mixta; B7a y B7b usan la escalera de simuladores y el SRCC de T1b.
- **C1a · De la MCU a la GPU.**
  - Qué es un procesador por dentro: ciclo de instrucción, *pipeline*, caché y jerarquía de memoria, por qué la memoria manda más que los FLOPS.
  - Microcontroladores de controladora de vuelo: familia ARM Cortex-M (los STM32 de las Pixhawk), FPU, DSP, SRAM y *flash*, periféricos (temporizadores, DMA), sin MMU ni sistema operativo de propósito general.
  - CPUs de aplicación: ARM Cortex-A frente a x86, multinúcleo, SIMD (NEON, AVX), consumo.
  - La GPU por dentro: SIMT, *warps*, SMs, ocupación, jerarquía de memoria (registros, compartida, L2, DRAM), *tensor cores*, precisión (FP32, FP16, BF16, INT8, FP8) y el modelo de programación CUDA (kernels, bloques, hilos, *streams*).
  - El modelo *roofline*: intensidad aritmética, techo de memoria y de cómputo, con números de una GPU de consumo, de un Jetson y de un móvil.
  - Memoria unificada (Jetson, Apple, móviles) frente a GPU discreta con PCIe; qué cuesta copiar.
  - Por qué una red de control de un dron (*batch* 1, pocos parámetros) no se parece a entrenar: latencia frente a rendimiento.
- **C1b · NPUs, aceleradores y plataformas.**
  - Qué es una NPU: matrices sistólicas, *dataflow*, memoria en el chip, cuantización obligada; TPU como caso histórico.
  - Aceleradores concretos: DLA y PVA de Jetson, Qualcomm Hexagon, Hailo, Google Coral Edge TPU, NPUs de SoC (Rockchip RK3588 y otros), Intel y AMD; qué operadores soportan y qué pasa con los que no.
  - FPGA (cuándo merecen la pena) y neuromórfico (Loihi y similares, con cámaras de eventos), con su estado real.
  - TinyML: redes en la propia MCU (como la política de 2084 parámetros de M0).
  - TOPS anunciados frente a rendimiento medido; TOPS/W; por qué las cifras de fabricante no se comparan entre sí.
  - SWaP (tamaño, peso y potencia): cuánto pesa y consume cada opción y cuánto vuelo cuesta en un dron pequeño.
  - Plataformas reales: Jetson Orin, Qualcomm (ModalAI VOXL), Raspberry Pi con acelerador, Skydio y los ordenadores de los drones de competición e investigación (Swift, Agilicious).
- **C2a · El autopiloto por dentro.**
  - Qué hace un autopiloto y qué no: estimación, control, modos, seguridad; la frontera con el ordenador de a bordo.
  - Hardware Pixhawk: los estándares abiertos (FMU, *baseboard*, conectores), procesador principal y procesador de E/S, sensores redundantes, aislamiento de vibraciones y calefacción de la IMU, alimentación y *power module*, salidas de motor; una placa real desmontada en un dibujo.
  - PX4 por dentro: NuttX, módulos y *work queues*, uORB (publicación y suscripción, `timestamp_sample`), cómo fluye un dato desde el driver de la IMU hasta el motor, y a qué frecuencia va cada etapa.
  - *Commander*: estados de armado, *preflight checks*, modos de vuelo y la máquina de estados, *failsafes* (RC, datos, batería, geocerca).
  - *Navigator*, misiones, control de posición, actitud y *rate* (enlazando con A2), asignación de control y salida a los ESC.
  - Parámetros, *airframes*, `rcS` y el arranque; *logger* y ulog.
  - ArduPilot comparado: HAL, *scheduler*, ChibiOS, EKF3, modos; cuándo se usa cada uno.
  - Otros firmwares y plataformas: Betaflight (carreras), Crazyflie, controladoras propias de laboratorio.
- **C2b · Arquitectura del sistema completo.**
  - Las piezas: autopiloto, ordenador de a bordo (*companion*), sensores de percepción, radio, estación de tierra (QGroundControl), nube.
  - Quién hace qué y a qué frecuencia; las interfaces entre capas (offboard: posición, velocidad, CTBR, motores directos) y lo que cuesta cada una.
  - Presupuesto de latencias de extremo a extremo, de cómputo, de energía y de peso.
  - Redundancia, modos de fallo y degradación; qué pasa si se cae el *companion*.
  - Arquitecturas publicadas de sistemas reales (Agilicious, Swift, Skydio, MIT, ETH, drones de enjambre) y cómo encaja la red aprendida.
  - Arquitectura del dron de la tesis.
- **C3a · Buses y protocolos a bordo.**
  - Capa física y trama de cada bus, con su ancho de banda, latencia y robustez calculados: UART, SPI, I2C, CAN y CAN FD (DroneCAN y Cyphal), USB, Ethernet (en las Pixhawk recientes) y MIPI CSI para cámaras.
  - Mando a los motores: PWM, OneShot, Multishot, DShot y DShot bidireccional (telemetría de RPM), y ESCs por CAN.
  - Radiocontrol: SBUS, CRSF y ExpressLRS.
  - GNSS (UBX, NMEA, RTCM para RTK) e interrupciones y DMA en la MCU.
- **C3b · MAVLink, DDS, radio y sincronización.**
  - MAVLink v2: la trama byte a byte, CRC, firma, dialectos y microservicios (*heartbeat*, comandos, parámetros, misiones, offboard, ficheros); pymavlink, MAVSDK y MAVROS.
  - DDS y RTPS, uXRCE-DDS (agente y cliente), Zenoh; QoS y lo que pasa en una red con pérdidas.
  - Radio: telemetría (SiK, enlaces digitales), Wi-Fi, malla, 4G/5G; alcance, latencia, pérdidas y presupuesto de enlace.
  - Sincronización temporal: NTP, PTP, el *timesync* de MAVLink y de uXRCE-DDS, marcas de tiempo por hardware (enlaza con B4).
  - Seguridad de las comunicaciones (cifrado, firma, suplantación), en breve.
- **C8 · ROS 2 a fondo.** Nodos, temas, servicios y acciones; executors; QoS; tf2; *launch*; rosbag2 y MCAP; `px4_msgs` y el puente con PX4; ros2_control; rendimiento y copia cero.
- **C4 · Tiempo real.** RTOS (NuttX, FreeRTOS, Zephyr, ChibiOS), planificación por prioridades, inversión de prioridad, WCET, PREEMPT_RT, aislamiento de núcleos, executors de ROS 2 en tiempo real.
- **C5 · Inferencia en el borde.** Cuantización (PTQ, QAT, INT8, INT4), poda y destilación, ONNX y ONNX Runtime, TensorRT, compiladores (TVM y similares), runtimes de móvil y MCU (LiteRT, ExecuTorch), kernels fusionados, presupuestos de latencia y cuándo mandar a la nube.
- **C6 · Sensores, actuadores, potencia y calibración.** IMU, cámaras (global y rodante, exposición, sincronización por disparo), eventos, LiDAR, flujo óptico, GNSS y RTK, barómetro, magnetómetro; motores BLDC, ESC (conmutación trapezoidal frente a FOC), hélices, baterías LiPo y Li-ion, distribución de potencia; calibración intrínseca, extrínseca y temporal (Kalibr), de IMU y de magnetómetro.
- **C7 · Operación y experimentación real.** Pipeline de pruebas, logging (ulog, MCAP), análisis (Flight Review, PlotJuggler), seguridad en vuelo y normativa europea (EASA, breve).

## Fase 6 · Cerrar el círculo (7 jornadas)

| J | ID | Presentación | Profundidad | Lab |
|---|---|---|---|---|
| 49 | B5 | SIL, HIL y realidad mixta | A fondo | PX4 HITL con la controladora real (opcional) y SIH en la placa, frente al SITL de T1a |
| 50 | B7a | Sim-to-real: taxonomía de gaps y técnicas | A fondo | Ablación: qué capa de realismo importa más |
| 51 | B7b | Sim-to-real en la práctica: casos y protocolo | A fondo | Política en una Pixhawk en HIL y evaluación contra datos de vuelo públicos |
| 52 | B8 | Diseñar tu propio simulador | A fondo | Documento de diseño del simulador de la tesis |
| 53 | D1 | Datasets, benchmarks y evaluación | A fondo | Protocolo de evaluación de la tesis |
| 54 | D2 | Investigar en robótica hoy | A fondo | Mapa de lecturas y plan del primer año |
| 55 | Final | Examen final | — | Los 7 criterios de `ANALISIS.md` §3 |

---

## Ruta crítica

Si en algún momento hay que priorizar, estas 18 presentaciones son las que más pesan para la tesis: M0, M1, A2, A4a, A6b, A7a, A7b, A8a, A9a, A13, B4, B6, B7a, B8, T1a, C2a, C2b y C8.

## Pista paralela recomendada: C++

No forma parte del curso, pero PX4, ROS 2 y casi todo el código de tiempo real están en C++. Conviene ir practicándolo en paralelo, en torno a 30 minutos al día, para llegar a la Fase 5 leyendo C++ con soltura. Los recursos concretos se proponen en M0.

## Entregables por tema

Para cada ID:

- `temas/<ID>_<slug>/index.html`: la presentación.
- `temas/<ID>_<slug>/lab/`: el laboratorio (enunciado y código base).
- El test de autoevaluación va dentro de la presentación, con las respuestas ocultas hasta pulsar.
- Las lecturas guiadas van en la propia presentación: qué leer, qué buscar y qué preguntas responder.
