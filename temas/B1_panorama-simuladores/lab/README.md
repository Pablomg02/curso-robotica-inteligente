# Laboratorio B1 · El mismo dron en dos simuladores

**Duración:** unas 3,5 h (el paso 5 es opcional).
**Requisitos:** tu simulador JAX del x500 (A1) con la cascada de A2, y MuJoCo para Python (`uv add mujoco`), que corre en CPU. Para la parte GPU del paso 4, `mujoco-mjx` o MuJoCo Warp (opcional). Para el paso 5, PX4 en Linux o en su contenedor oficial (opcional, sin GPU ni Gazebo).

## Objetivo

Montar el mismo x500 en tu simulador y en MuJoCo, volarlo con el mismo controlador y medir qué cambia y por qué. Al final tendrás un segundo simulador con el que validar tus políticas sin reentrenarlas (sim-to-sim), y comprobarás con tus números lo que calcula la presentación:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Error máximo de posición con paso de 1 ms (círculo de 4 m a 5 m/s, 4 s): Euler / RK4 | 2,7 mm / 0,2 µm | 3 |
| Pendiente del error entre 1 y 4 ms: Euler; RK4 con saturación; RK4 sin saturación | 1,00; 2,09; 4,06 | 3 |
| Escalón de alabeo de 20°: sobreoscilación y establecimiento (empuje instantáneo; Gazebo; SIH) | 0,2 % y 320 ms; 0 % y 332 ms; 4,0 % y 576 ms | 2 |
| Vuelo recto a 5 m/s: inclinación y retraso del PD (mismas tres plantas) | 0° y 0; 3,5° y 15 cm; 13,9° y 61 cm | 2 |
| Mando de estacionario en SIH con sus valores por defecto (1 kg, 4 × 5 N) | \(u = mg/(4T_{\max})\) ≈ 0,49 | 5 |

El script de las figuras, en numpy, está en `../figuras/figuras_b1.py`. Contiene el modelo del x500, el controlador y las tres convenciones de planta (empuje instantáneo como en MuJoCo Menagerie, motor de Gazebo y SIH de PX4). Úsalo como referencia para comprobar cada paso. Las cuentas a mano de las hojas (el oscilador con Euler explícito y semiimplícito, RK4 en \(\dot x = -x\), la saturación de juguete, el motor discretizado, el lazo de *rate* con retardo, el contacto por penalización y el *lockstep*) están en `../figuras/ejemplos_b1.py`, con su salida en `../figuras/salida_ejemplos.txt`.

## 1. El x500 en MuJoCo (45 min)

En `src/proyecto_hilo/simulacion/x500.xml`, un MJCF mínimo con lo que ya usas en tu simulador:

- un cuerpo libre (`freejoint`) de 2,064 kg con `diaginertia="0.02389 0.02389 0.044"`;
- cuatro `site` en (±0,174; ±0,174; 0), con los rotores 0 y 1 girando en sentido antihorario y el 2 y el 3 en sentido horario, como en `PX4-gazebo-models`;
- cuatro actuadores de tipo `general` sobre esos `site`, con `gear="0 0 1 0 0 ±0.016"` (empuje en z y par de reacción en guiñada, \(k_m\) = 0,016 m), `ctrlrange` en newtons entre \(k_T\cdot150^2\) y \(k_T\cdot1000^2\) (0,192 y 8,549 N) y la dinámica del motor como filtro de primer orden: `dyntype="filter" dynprm="0.0125"`;
- `<option integrator="RK4" timestep="0.0005"/>`.

Esqueleto de un actuador:

```xml
<general site="rotor0" gear="0 0 1 0 0 -0.016" dyntype="filter" dynprm="0.0125"
         gainprm="1" ctrlrange="0.192 8.549" ctrllimited="true"/>
```

**Comprobaciones:** `model.body_subtreemass` da 2,064 kg. Con los cuatro mandos a \(mg/4\) = 5,062 N y la activación inicial del filtro también a ese valor (`data.act[:] = 5.062`; si no, arranca en cero y el dron cae), se queda quieto (aceleración vertical por debajo de \(10^{-9}\) m/s²). Con un mando asimétrico, el signo del par coincide con el de tu simulador. MuJoCo ordena los cuaterniones como el proyecto hilo (w, x, y, z), con z hacia arriba y la gravedad por defecto a −9,81 m/s².

**Piensa:** el filtro de MuJoCo actúa sobre el empuje y el de Gazebo sobre \(\Omega\), con \(T = k_T\Omega^2\). Linealiza alrededor del estacionario: ¿son equivalentes para perturbaciones pequeñas? ¿Y para un escalón grande? El filtro de MuJoCo tampoco distingue entre subida y bajada (12,5 ms y 25 ms en Gazebo): ¿cómo lo modelarías (`dyntype="user"` o un mando corregido desde Python)?

## 2. Mismo controlador, dos simuladores (60 min)

En `experimentos/b1_dos_simuladores.py`:

1. El controlador geométrico de `figuras_b1.py` (ganancias 4; 3,5; 8 y 25) a 250 Hz, con retención de orden cero: en MuJoCo, 8 llamadas a `mj_step` por periodo de control.
2. Los tres ensayos de la presentación en los dos simuladores: escalón de alabeo de 20°, escalón lateral de 2 m y vuelo recto a 5 m/s.
3. El arrastre en MuJoCo: antes de cada `mj_step`, escribe la fuerza en `data.xfrc_applied` del cuerpo, con el modelo de rotor de Gazebo (\(-\sum|\Omega_i|\,c_d\,\mathbf v_\perp\)) o el lineal de SIH (\(-K_{dv}\mathbf v\)).

**Comprobaciones:** con la misma planta en los dos simuladores, las trayectorias deben coincidir dentro del error de integración (mucho menos de 1 mm en 5 s). Si no coinciden, el fallo casi siempre es de convenio: signo del par de guiñada, orden del cuaternión, ejes del cuerpo o momento en el que se aplica el mando. Después, cambia solo el modelo del motor y del arrastre y reproduce la tabla de la presentación (inclinación de 0°, 3,5° y 13,9° a 5 m/s; retrasos de 0, 15 y 61 cm).

## 3. Integradores y paso (40 min)

1. En MuJoCo, el ensayo del círculo de la figura (radio 4 m, 5 m/s, 4 s) con `integrator` = `Euler`, `RK4` e `implicitfast`, y pasos de 0,25 a 10 ms. Referencia: RK4 con 0,05 ms.
2. Lo mismo en tu simulador JAX, con Euler explícito, Euler semiimplícito y RK4.
3. Para cada caso, mide el error máximo de posición y el tiempo de CPU por segundo simulado.

**Comprobaciones:** pendiente 1 para Euler y 4 para RK4 si quitas la saturación de los motores; con saturación, RK4 baja a un orden cercano a 2. La hoja «Por qué una saturación baja RK4 de orden 4 a orden 2» explica el mecanismo y su ejemplo de juguete (ejemplo 7 del script) te sirve de test rápido antes del x500. Prueba también `dyntype="filterexact"`: ¿cambia el error con pasos grandes?

**Piensa:** ¿por qué Euler explícito y semiimplícito dan casi lo mismo aquí? Construye un caso en el que no sea así (un péndulo sin amortiguar, 60 s) y mide la deriva de energía con los dos. Compara con la hoja «Por qué el explícito gana energía y el semiimplícito no»: con \(z = h\omega\) pequeño, el explícito debería multiplicar la energía por \(1 + z^2\) en cada paso.

## 4. Rendimiento (40 min)

Mide pasos de física por segundo frente al número de entornos \(N\) = 1, 16, 256, 4096 y 65 536:

- tu simulador JAX con `jax.vmap` y `jax.jit`, en CPU y, si tienes, en GPU (descarta la primera llamada, que compila);
- MuJoCo en CPU: un bucle en Python con `mj_step` y el módulo `mujoco.rollout`, que reparte varias simulaciones entre hilos si le pasas varias `MjData`;
- opcional: MJX o MuJoCo Warp en GPU.

Representa los pasos por segundo frente a \(N\) en escala logarítmica. Anota en la tabla todo lo que pide la lista de la hoja «El caso Genesis»: hardware, qué es un paso (física o control), qué hace el dron, precisión numérica y si incluyes la compilación.

**Piensa:** ¿cuántos pasos por segundo necesitas para entrenar una política con PPO en una noche (A6a)? ¿Cuál de tus opciones los da?

## 5. PX4 SIH en SITL (opcional, 40 min)

1. Compila y arranca `make px4_sitl_sih sihsim_quadx` (o usa el paquete o contenedor precompilado de la documentación de PX4). No necesita Gazebo ni GPU.
2. En la consola de PX4, `param show SIH_*`: anota masa, inercias, \(T_{\max}\), \(\tau\) y arrastres. Compáralos con los que leíste en `sih_params.yaml` en la presentación.
3. Despega, mantén el estacionario y haz un escalón de posición (desde QGroundControl o con MAVSDK-Python en modo offboard). Descarga el ulog y analízalo con `pyulog`: mando de los motores en estacionario, respuesta del alabeo y del escalón.

**Comprobaciones:** el mando de estacionario ronda \(mg/(4T_{\max}) = 9{,}81/20 \approx 0{,}49\). Compara la respuesta del escalón con la de tu simulador ajustado a los parámetros de SIH. ¿A qué frecuencia publica SIH la IMU? Búscalo en el log.

## Entregables

- `simulacion/x500.xml` y `simulacion/mujoco_x500.py`, con tests: masa y estacionario correctos; un mando de alabeo da un par del signo esperado; con la misma planta, MuJoCo y tu simulador coinciden dentro de una tolerancia que justifiques.
- `experimentos/b1_dos_simuladores.py`, que genere: las tres respuestas en los dos simuladores, error frente a coste para los integradores y pasos por segundo frente a \(N\).
- Una tabla con tus valores frente a los de la presentación y una línea por cada diferencia que hayas explicado.

## Preguntas para cerrar

1. ¿Qué diferencias entre tus dos simuladores son de modelo y cuáles de implementación? ¿Qué experimento las separa?
2. Una política entrenada con empuje instantáneo pasa a SIH, con motor de 50 ms. ¿Qué esperas que ocurra? ¿Cómo lo evitarías sin cambiar de simulador de entrenamiento?
3. Para tu tesis, ¿qué pasa a ser el «segundo simulador» de referencia y por qué? Ten en cuenta mantenimiento, velocidad y si puede cargar tu modelo de dron sin traducciones a mano.
