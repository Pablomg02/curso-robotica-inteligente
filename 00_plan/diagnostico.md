# Diagnóstico y votación (2026-11-06)

Respuestas de Pablo al cuestionario inicial e interpretación para diseñar el curso. Pablo pidió explícitamente que la última palabra sobre importancia y profundidad la tenga el diseño del curso, y que sea intenso, exigente y en profundidad.

## 1. Autoevaluación (0 = nada · 1 = concepto · 2 = implementado · 3 = usado en serio · 4 = experto)

| Área | Nivel | Área | Nivel |
|---|---|---|---|
| Dinámica 6DoF de multirrotor | 3 | Imitation learning y políticas generativas | 1 |
| Control clásico (PID) | 2 | Transformers, LLMs, VLMs | 2 |
| Control moderno y óptimo | 1 | VLAs y foundation models | 1 |
| MPC y optimización numérica | 1 | Simuladores de robótica | 1 |
| Estimación de estado | 1 | ROS 2 | 0 |
| Planificación | 1 | Autopilotos (PX4, ArduPilot, Betaflight) | 1 |
| Deep RL | 4 | Hardware embebido | 1 |
| Observabilidad parcial (POMDP) | 3 | Tiempo real aplicado | 0 |
| World models / MBRL | 2 | Inferencia en el borde | 0 |
| MARL | 3 | | |

## 2. Contexto

- **Lenguajes con soltura:** Python y JAX. Sin C ni C++.
- **Matemáticas cómodas:** álgebra lineal, optimización, probabilidad bayesiana, sistemas dinámicos, teoría de control. No marcó grupos de Lie ni teoría de la información.
- **Vuelo real:** RC/FPV manual y multirrotor con autopiloto, además del ala fija de UVigo Aerotech.
- **Plataformas de la tesis:** multirrotor, ala fija, microdron y varios drones.
- **Sensores previstos:** todos (mono, estéreo/profundidad, eventos, LiDAR, IMU+GNSS, comunicación entre agentes).
- **Recursos:** GPU de 16 GB y Pixhawk. El resto lo compraría si hiciera falta, pero el curso no debe depender de su hardware (ver §6).
- **TFM:** framework MARL en Python en 2D, que no le convenció. Ahora construye sus propios entornos en JAX con ayuda de IA.
- **Simulador propio:** para entrenamiento y validación. Motivo: eliminar fricción posterior, sin problema en reutilizar piezas existentes.
- **Formato:** sin límite de jornadas, 8 h o más al día, derivaciones completas, diapositivas mixtas (texto y esquema a la par), español con términos en inglés.
- **Extras:** laboratorio por tema, miniproyecto integrador, tests de autoevaluación, lecturas guiadas de papers.

## 3. Sondeo: qué revelan las respuestas

**Frecuencias de los lazos.** Situó la actitud como el lazo más rápido (unos 50 Hz) y no distinguió el lazo de *rate* (velocidad angular). En los autopilotos reales el orden es el contrario: el *rate* es el más rápido (de cientos de Hz a varios kHz), después la actitud y la posición, y por último el planificador visual. La cifra de 50 Hz para la actitud está un orden de magnitud por debajo. Los valores exactos se verifican en M0 y A2. La intuición de que la visión y la planificación son lentas es correcta. La idea de poner una red rápida en el lazo de actitud es defendible (hay trabajos que lo hacen), pero falta el criterio de ancho de banda y de separación de escalas temporales. **Consecuencia:** A2 sube de "Repaso" a "Importante" y la jerarquía de frecuencias se convierte en un hilo de todo el curso.

**Belief state frente a memoria recurrente.** Buena intuición: la memoria recurrente es implícita y el belief puede entrenarse con una señal predictiva. Faltan dos piezas:
1. La definición formal: el belief es la distribución a posteriori sobre el estado oculto dada la historia, p(s_t | h_t), y es un estadístico suficiente para el control óptimo.
2. La clasificación de Dreamer: su RSSM es un filtro aproximado aprendido. Sus latentes (determinista más estocástico, con posterior q(z_t | h_t, o_t)) se entrenan con un objetivo predictivo y de reconstrucción (ELBO), así que funcionan como un belief en espacio latente. Es exactamente la "señal de la realidad" que él describe. Decir que "Dreamer no usa belief" no es correcto; lo correcto es que usa un belief latente aprendido, no una distribución explícita sobre el estado verdadero.

**Consecuencia:** A7 y A8 arrancan desde la teoría formal (filtro de Bayes, belief MDP, estados de información aproximados) antes de entrar en arquitecturas profundas. A4 (estimación) sube a "A fondo", porque el filtro de Kalman es el belief exacto del caso lineal gaussiano: es la cara clásica de su tesis.

**Diagnóstico de sim-to-real.** Respuesta muy buena: latencia y asincronía, ruido de IMU y GNSS, memoria para absorber el retardo, predicción del estado en el instante de actuación (compensación de retardo) y error en parámetros físicos. Faltan, para completar el cuadro:
- Dinámica de actuadores: constante de tiempo del motor, respuesta del ESC, curva de empuje y caída de tensión de la batería.
- Diferencias en el controlador de bajo nivel si la política da órdenes de *rate*.
- Filtrado del giróscopo en la controladora real.
- Políticas de alta frecuencia tipo *bang-bang* que explotan el simulador (regularización de suavidad).
- Escalado y normalización de acciones.
- Convenciones de ejes (NED/ENU, FRD/FLU).
- Efectos aerodinámicos no modelados.

Estos puntos se cubren en B2, B4 y B7.

## 4. Lectura global

- **Fuerte:** RL (incluido MARL y observabilidad parcial en la práctica), dinámica de vuelo, base matemática y ML en general. Tiene experiencia real de vuelo.
- **Medio:** world models, LLMs, control clásico (con lagunas en la jerarquía real de lazos).
- **Débil o nulo:** toda la pila robótica real (ROS 2, tiempo real aplicado, inferencia en el borde, embebido, autopilotos), control óptimo y MPC, estimación, planificación, imitation learning, VLAs y simuladores de robótica.
- **Lagunas transversales:** C/C++ (imprescindible para PX4, ROS 2 y tiempo real), grupos de Lie (necesarios para control geométrico, VIO y estimación en la variedad) y teoría de la información (ELBO, KL, información mutua; necesaria para world models y percepción activa).

Aunque la web indica que imparte *Sistemas de Tiempo Real*, marcó 0 en tiempo real aplicado: C4 se diseña desde los fundamentos.

## 5. Ajustes sobre la votación

| Tema | Voto | Decisión | Motivo |
|---|---|---|---|
| A2 Control clásico | Repaso | **Importante** | El sondeo muestra lagunas en la jerarquía de lazos, que es la base de A13, C2 y C4 |
| A3 MPC | Importante | **A fondo** | MPPI y CEM son la forma de planificar con un world model (TD-MPC2, PETS), y MPC frente a RL es un debate central en drones |
| A4 Estimación | Importante | **A fondo** | El filtro de Bayes es la versión clásica del belief state |
| B6 Simulación masiva | Importante | **A fondo** | Ya construye entornos en JAX: es justo lo que necesita para su simulador |
| Nuevo: M1 Herramientas matemáticas | — | **Importante** | Grupos de Lie, inferencia variacional y teoría de la información, con derivaciones |
| Nuevo: C8 ROS 2 a fondo | — | **A fondo** | Nivel 0 en el estándar de facto de la robótica |

El resto de votos se mantiene. C/C++ no entra como tema del curso: se recomienda como pista paralela (ver `TEMARIO.md`).

## 6. Hardware (decisión del 2026-11-06)

Pablo tiene una GPU de 16 GB y una Pixhawk, y comprará lo que haga falta, pero pidió que **el curso no se base en su hardware** y que **los laboratorios no usen drones**. Decisión:

- Laboratorios independientes del hardware: PC con GPU NVIDIA de consumo o nube, y CPU como alternativa.
- Ningún vuelo real. Los datos reales salen de datasets y logs públicos.
- La Pixhawk se puede usar como hardware-in-the-loop. La Jetson es opcional.

Ver la "Política de hardware de los laboratorios" en `TEMARIO.md`.
