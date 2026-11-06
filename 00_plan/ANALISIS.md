# Curso intensivo de robótica inteligente: análisis de la petición

> Documento vivo. Creado el 2026-11-06. Recoge qué se pidió, cómo lo interpreto, qué decisiones quedan abiertas y cómo se va a construir el curso. Se actualiza cuando lleguen las respuestas del diagnóstico y la votación de temas (ver `diagnostico.md`, pendiente).

---

## 1. La petición, resumida con fidelidad

Pablo quiere un **curso intensivo para sí mismo**, de varias jornadas completas, para **ponerse al día en robótica inteligente** antes de arrancar su doctorado. Aunque el doctorado es sobre drones, el curso debe cubrir la robótica en general, porque el conocimiento se transfiere bien entre plataformas.

Al terminar debe tener **una visión clara del estado del arte actual**, organizada en tres bloques que pidió explícitamente como "ultra sólidos":

1. **Control de robots con IA.** Cómo se controlan hoy los robots sabiendo que hay latencia y que el modelo embarcado no puede ser enorme, pero aun así se necesita inteligencia. Incluye PID, MPC, LLMs, VLAs, world models, fusión de modelos y lo que haga falta.
2. **Simulación.** En qué entornos se simula y cómo se reproduce la realidad: hardware in the loop, asincronía, modelado fiel, etc. Le interesa sobre todo para **construir su propio simulador** y para saber **qué herramientas y capas se le pueden estar olvidando** al simular sus drones y robots reales. Sirve tanto para **entrenamiento** como para **despliegue**.
3. **Hardware de despliegue y arquitectura de sistema.** Qué hardware se usa, cómo se conectan las partes (control, razonamiento, tareas rápidas y lentas), cómo se comunican y si hay tiempo real.

Objetivo final: tener todo lo necesario para empezar un doctorado en **IA para drones pensado para sim-to-real**, no solo para simulación.

### Requisitos de formato

- **Teoría como presentaciones HTML**, una por tema, que funcionen bien y sean claras.
- Sin animaciones exageradas, pero con **esquemas y dibujos** cuando ayuden.
- **Investigación intensa antes de cada tema**: estado del arte, cómo se hacía antes, cómo se hace ahora y qué se espera en el futuro.
- **Referencias** en las diapositivas (papers, webs, repos) para profundizar y para anclar el contenido a la realidad.
- No es una compilación sin sentido del estado del arte: es **un curso completo**, con hilo didáctico.
- Se puede dividir en tantos temas como haga falta.

### Proceso acordado

1. Análisis de la petición, guardado en disco (este documento).
2. Cuestionario estratégico de diagnóstico: qué sabe y qué le falta.
3. Votación de temas para alinear el temario con sus intereses.
4. A partir de ahí, temario definitivo y producción tema a tema.

---

## 2. Contexto: el doctorado

Según su web (pablomagarinos.es, consultada el 2026-11-06; los datos son autodeclarados y pueden estar desactualizados, así que conviene confirmarlos):

- **Tesis:** *Belief-state and world-model-based learning for autonomous aerial systems in partially observable environments*. Programa de Doctorado en Tecnología Aeroespacial, Universidade de Vigo. Inicio: octubre de 2026.
- **Objetivo declarado:** diseñar, desarrollar y analizar métodos de aprendizaje basados en *belief states* y *world models* para que sistemas aéreos autónomos actúen y se adapten online con información incompleta, en entornos y condiciones desconocidos, junto a compañeros y adversarios desconocidos.
- **Líneas:** RL bajo observabilidad parcial, world models, belief states, meta-learning, continual learning, tareas de horizonte largo y sistemas multiagente. Los drones son el ancla de una investigación en *embodied AI*.
- **Antecedentes relevantes:** Grado en Ingeniería Aeroespacial (UVigo), Máster en Matemática Industrial (UC3M) con TFM en *MARL para búsqueda y rescate con drones*, investigador en ATRG (visión embarcada, estimación de pose, auto-entrenamiento con etiquetado débil), líder de UVigo Aerotech (cinco drones de ala fija de competición), simulador de pista en Formula Student, publicación sobre PPO para optimización aerodinámica en tiempo real. En el curso académico 2026–27 imparte *Sistemas de Tiempo Real* en la EEAE. Fundador de XISTRA (aprendizaje y adaptación en tiempo real para robots).

### Lo que esto implica para el curso

- **El núcleo de la tesis** (POMDPs, belief states, world models, adaptación online, meta- y continual learning, multiagente) no estaba en la lista explícita de la petición, pero **tiene que estar en el curso**, y con la máxima profundidad.
- La **base fuerte** parece ser el aprendizaje automático (RL, visión) y la ingeniería aeroespacial. La web no menciona ROS 2, PX4/ArduPilot, MPC, hardware embebido de vuelo ni despliegues reales en multirrotor. **Hipótesis a confirmar con el diagnóstico:** el mayor salto está en la pila robótica real (estimación, control de bajo nivel, middleware, hardware, tiempo real aplicado) y en el sim-to-real.
- Da clase de *Sistemas de Tiempo Real*, así que la teoría de RTOS y planificación probablemente se puede comprimir y orientar a su aplicación en robots (PX4/NuttX, PREEMPT_RT, executors de ROS 2).

---

## 3. Qué tiene que saber hacer al terminar

Criterios de "curso superado". Sirven para diseñar los temas y para la autoevaluación final:

1. **Dibujar de memoria la pila completa de un dron autónomo moderno**: sensores, estimación, control de bajo nivel, planificación, percepción, razonamiento, cómputo, buses y frecuencias. Con valores típicos de latencia y de frecuencia en cada capa, y sabiendo justificar por qué.
2. **Elegir y justificar una arquitectura de control** para un problema dado: clásico, óptimo, aprendido o jerárquico (System 1 / System 2). Saber dónde encaja cada familia (PID, MPC, RL, IL, VLA, world model) y dónde no.
3. **Situar cualquier paper de 2025–2026** en el mapa del campo: qué problema ataca, qué hace distinto, qué limitaciones tiene, y si su resultado sobreviviría en hardware real.
4. **Diseñar un pipeline de simulación** para entrenar y validar un dron: qué simulador o simuladores, qué capas de fidelidad (física, sensores, tiempo, latencias, actuadores), cómo se valida contra datos reales y qué se suele olvidar.
5. **Especificar el hardware de despliegue y las comunicaciones**: qué corre en el MCU, en el ordenador de a bordo y fuera de él; con qué protocolos y garantías temporales.
6. **Plantear una estrategia sim-to-real** completa para su tesis: fuentes de *gap*, mitigaciones, protocolo experimental y métricas.
7. **Conocer el ecosistema**: grupos, empresas, conferencias, benchmarks y herramientas de referencia, y saber cómo seguirle el ritmo al campo.

---

## 4. Principios de diseño

- **Investigación primero.** Cada tema se construye tras una búsqueda a fondo: surveys recientes, papers seminales, documentación oficial, repos y resultados en hardware real. Se priorizan las fuentes primarias. Las afirmaciones sin respaldo se marcan como tal.
- **Pasado → presente → futuro** en cada tema: cómo se hacía, por qué cambió, qué se hace hoy (2025–2026) y qué viene. Con fechas.
- **Anclaje a la realidad.** Números concretos (Hz, ms, W, g, TOPS, parámetros), sistemas reales que funcionan y resultados en hardware, no solo en simulación. Separar demo de producto y hype de evidencia.
- **Transferible pero con ancla en drones.** Cada concepto se explica en general (manipuladores, patas, vehículos) y luego se aterriza en multirrotores y ala fija.
- **Hilos conductores** que reaparecen en todos los temas:
  - *Latencia y frecuencia*: quién corre a qué velocidad y por qué.
  - *Incertidumbre y observabilidad parcial*: el núcleo de la tesis.
  - *Sim-to-real*: qué se rompe al pasar a hardware.
  - *Cómputo limitado (SWaP)*: tamaño, peso y potencia frente a inteligencia.
- **Ejemplo conductor.** Un mismo caso que recorre todo el curso: un cuadricóptero con cámara e IMU que explora y busca objetivos en un entorno desconocido y parcialmente observable, en solitario o en equipo. Cada tema añade una capa a ese sistema.
- **Conexión con la tesis** al final de cada tema: qué significa para un doctorado en belief states y world models para drones, y qué preguntas abiertas hay.

---

## 5. Estructura estándar de cada presentación

1. Portada: tema, bloque, duración estimada y prerrequisitos.
2. Objetivos de aprendizaje (3–5, verificables).
3. Por qué importa: el problema real que resuelve.
4. Historia breve: cómo se llegó aquí, con hitos y fechas.
5. Fundamentos: la teoría necesaria, con ecuaciones clave y esquemas.
6. Estado del arte 2025–2026: métodos, sistemas y resultados reales.
7. Cómo se hace en drones: casos concretos, números y arquitectura.
8. Limitaciones y problemas abiertos.
9. Hacia dónde va el campo.
10. Conexión con la tesis.
11. Resumen y preguntas de autoevaluación.
12. Lecturas: esenciales (pocas, para leer sí o sí) y de profundización.
13. Referencias completas con enlaces (DOI, arXiv, web oficial, repo).

---

## 6. Temario preliminar (para votar)

> **Superado:** el temario definitivo está en `TEMARIO.md`. Esta sección se conserva como registro de la propuesta que se votó.

Los identificadores coinciden con los de la votación. El número de presentaciones final dependerá de los votos: los temas marcados "a fondo" pueden partirse en dos, y los marcados "repaso" pueden fusionarse.

### Bloque 0 · Marco

- **M0 · Mapa del campo.** De la robótica clásica al aprendizaje y a los foundation models. Actores (grupos, empresas), conferencias y revistas, y cómo seguir el estado del arte.

### Bloque A · Control e inteligencia (bloque ultra sólido)

- **A1 · Dinámica y modelado.** Cuerpo rígido en SE(3), cuaterniones, modelo de rotor y motor, aerodinámica, planitud diferencial. Nociones de robots con patas y manipuladores.
- **A2 · Control clásico y jerarquía de lazos.** PID en cascada (rate, actitud, posición), INDI, control geométrico, asignación de control. Cómo lo hacen PX4 y Betaflight.
- **A3 · Control óptimo y MPC.** LQR, NMPC, MPCC (tiempo mínimo), MPPI, *real-time iteration*, acados. MPC con aprendizaje.
- **A4 · Estimación y percepción para control.** EKF/ESKF, VIO, SLAM, cámaras de eventos, sincronización y timestamps.
- **A5 · Planificación y navegación.** Minimum snap, búsqueda y muestreo, planificadores locales, exploración.
- **A6 · Deep RL para control robótico.** PPO/SAC, entrenamiento masivo en paralelo, *teacher-student* con información privilegiada, carreras de drones y locomoción.
- **A7 · RL bajo observabilidad parcial.** POMDPs, belief states, memoria (RNN, SSM, transformers), percepción activa. *Núcleo de la tesis.*
- **A8 · World models y RL basado en modelo.** Dreamer, TD-MPC, MuZero, JEPA, world models de vídeo como simuladores y para planificar. *Núcleo de la tesis.*
- **A9 · Adaptación online, meta-learning y continual learning.** Adaptación rápida a dinámicas desconocidas, meta-RL, olvido catastrófico. *Núcleo de la tesis.*
- **A10 · Imitation learning y políticas generativas.** BC, DAgger, *action chunking*, diffusion policy, flow matching, recogida de datos.
- **A11 · LLMs y VLMs para razonar y planificar.** Planificadores con LLM, generación de código como política, agentes y *grounding*.
- **A12 · VLAs y robot foundation models.** Familias principales, tokenización de acciones, *cross-embodiment* y VLAs para drones.
- **A13 · Arquitecturas jerárquicas y fusión de modelos.** System 1 / System 2, multi-frecuencia, inferencia asíncrona, políticas residuales, destilación.
- **A14 · Seguridad y garantías.** Control barrier functions, alcanzabilidad, safe RL, *shields*, detección de fallos y de fuera de distribución, incertidumbre.
- **A15 · Multiagente y enjambres.** MARL (CTDE, MAPPO), comunicación, coordinación, compañeros desconocidos (*ad hoc teamwork*) y adversarios. *Núcleo de la tesis.*

### Bloque B · Simulación (bloque ultra sólido)

- **B1 · Panorama de simuladores.** Gazebo, Isaac Sim/Lab, MuJoCo/MJX, Genesis, Newton, Flightmare, Aerial Gym, OmniDrones, gym-pybullet-drones, Pegasus y la herencia de AirSim. Qué usar para qué.
- **B2 · Física fiel.** Integradores, contacto, aerodinámica de rotores, motor, ESC y batería, viento y turbulencia, identificación de sistemas, modelos híbridos física + red.
- **B3 · Sensores y render.** Ruido de IMU, modelos de cámara (rolling shutter, desenfoque, exposición), cámaras de eventos, LiDAR, fotorrealismo, Gaussian Splatting y *real2sim*.
- **B4 · Tiempo, asincronía y latencia en simulación.** Lockstep, reloj simulado, multi-frecuencia, jitter, retardos de comunicación, determinismo.
- **B5 · SIL, HIL y realidad mixta.** SITL/HITL de PX4, bancos de pruebas, captura de movimiento, gemelos digitales.
- **B6 · Simulación para entrenamiento masivo.** Paralelismo en GPU, vectorización, simulación diferenciable, generación procedural, currículo, *domain randomization*.
- **B7 · Sim-to-real.** Taxonomía de *gaps*, *domain randomization*, identificación, *real2sim2real*, adaptación y protocolos de evaluación.
- **B8 · Diseñar tu propio simulador.** Arquitectura, formatos (USD, URDF, MJCF, SDF), APIs (Gymnasium), puentes con ROS 2 y PX4, validación contra datos reales.

### Bloque C · Hardware y despliegue (bloque ultra sólido)

- **C1 · Cómputo embarcado.** MCUs de controladora de vuelo, Jetson, Qualcomm, Raspberry Pi con aceleradores, FPGA. Compromisos de tamaño, peso y potencia.
- **C2 · Arquitectura del sistema.** Autopiloto, ordenador de a bordo y procesamiento externo: quién hace qué y a qué frecuencia.
- **C3 · Comunicaciones.** Buses (UART, SPI, I2C, CAN/DroneCAN, DShot), MAVLink, uXRCE-DDS, ROS 2/DDS/Zenoh y QoS, radio y 5G, sincronización temporal (PTP).
- **C4 · Tiempo real en robots.** RTOS (NuttX, FreeRTOS, Zephyr), PREEMPT_RT, planificación, WCET, executors de ROS 2, ros2_control.
- **C5 · Inferencia en el borde.** Cuantización, TensorRT/ONNX, destilación, presupuestos de latencia, a bordo frente a edge o nube.
- **C6 · Sensores, actuadores y calibración.** IMU, cámaras, eventos, LiDAR, ESC y motores, telemetría de RPM, calibración (Kalibr).
- **C7 · Operación y experimentación real.** Pipeline de pruebas, logging (ulog, MCAP), análisis, seguridad en vuelo y normativa (breve).

### Bloque D · Investigación

- **D1 · Datasets, benchmarks y evaluación.** Métricas, reproducibilidad y cómo evaluar sim-to-real con honestidad.
- **D2 · Investigar en robótica hoy.** Grupos, conferencias, problemas abiertos, tendencias y posicionamiento de la tesis.

---

## 7. Metodología de investigación por tema

Para cada tema, antes de escribir:

1. **Búsqueda amplia:** surveys de los últimos 2–3 años, papers seminales, trabajos de 2025–2026, documentación oficial y repositorios.
2. **Verificación:** comprobar fechas, autores, venue y enlaces de cada referencia. Nada de referencias inventadas. Si algo no se puede verificar, se dice.
3. **Números reales:** frecuencias, latencias, tamaños de modelo, cómputo y resultados en hardware, con su fuente.
4. **Visión crítica:** qué funciona de verdad en robots reales, qué es demo y qué sigue abierto.
5. **Guion didáctico:** ordenar el material para enseñar, no para enumerar.

---

## 8. Formato técnico de las presentaciones (propuesta)

- Un HTML por tema en `temas/<ID>_<slug>/index.html`, con estilos y motor de diapositivas compartidos en `assets/`.
- Navegación con teclado (flechas, inicio y fin), índice, número de diapositiva y modo impresión o PDF.
- Diagramas en SVG dibujados a mano: arquitecturas, bucles de control y líneas temporales de latencia.
- Ecuaciones con KaTeX guardado en local, para que funcione sin conexión.
- Referencias numeradas en cada diapositiva y una diapositiva final con todos los enlaces.
- Tema claro y oscuro y legible en pantalla grande.

---

## 9. Decisiones abiertas (las resuelve el cuestionario)

- Duración total y horas por jornada.
- Profundidad matemática: intuición, ecuaciones clave o derivaciones.
- Prácticas: solo teoría, fragmentos de código, laboratorios por tema o un miniproyecto integrador.
- Densidad de las diapositivas: autoexplicativas para estudio individual, ligeras con notas o mixtas.
- Idioma: español con términos técnicos en inglés (propuesta por defecto).
- Plataforma y sensores previstos para la tesis, y recursos disponibles (GPU, drones, captura de movimiento).
- Objetivo del simulador propio: entrenamiento masivo, validación de alta fidelidad o ambos.

---

## 10. Próximos pasos

1. Recoger el diagnóstico y la votación y guardarlos en `00_plan/diagnostico.md`.
2. Cerrar el temario definitivo, con orden, duración por tema y plan de jornadas.
3. Construir la plantilla de presentación (`assets/`) y validarla con el tema M0.
4. Producir los temas en orden, cada uno tras su investigación.
