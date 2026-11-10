# Laboratorio T1b · Mide tu brecha y monta tu cámara neuronal

**Duración:** unas 4 h.
**Requisitos:** el contenedor de T1a (`../../T1a_taller-pila-de-vuelo/lab/Containerfile`). Para la cámara neuronal (paso 3), una GPU NVIDIA con CUDA en el anfitrión y un entorno de Python con `torch` 2.4 (CUDA 12.4), `gsplat` 1.5.3 (rueda precompilada), `numpy` y `mavsdk` 4. Sin GPU se hace todo salvo el paso 3. Isaac Lab 3.0 (paso 6) es opcional. Nada vuela de verdad: los datos reales son registros públicos de Flight Review.
**Material:** los scripts de la presentación están en `../figuras/` (`comparar_real_sitl.py`, `extraer_neural.py`, `extraer_viento.py`, `ejemplos_t1b.py`) y `../figuras/experimentos/` (`camara_neuronal.py`, `imu_1khz.py`, `crear_mundo_1khz.sh`, `camaras.py`, `neural_vuelo.py`, `viento.py`, `viento_run.sh`). Úsalos de referencia; escribe los tuyos en el proyecto hilo (`src/proyecto_hilo/sim2real/`) y guarda las salidas en `experimentos/t1b_*`.

## Objetivo

Repetir con tus scripts las mediciones de la presentación, ampliarlas a más datos y llevarlas a tu propia política:

- la brecha real frente a SITL en sensores, vibración y empuje, con más de un registro;
- el efecto en tu política de A6b de acercar Gazebo al dron real (ruido identificado, IMU a 1 kHz, viento);
- una cámara de splats que sigue al dron de PX4 SITL, con su coste y su latencia;
- la red de `mc_nn_control` volando en SITL y cómo responde a su parámetro de empuje;
- el SRCC entre tu simulador JAX y Gazebo con varias variantes de tu política.

## 1. La brecha con más datos (50 min)

1. Descarga el índice público de Flight Review (`https://logs.px4.io/dbinfo`, unos 33 MB comprimido) y filtra con Python: `sys_autostart_id` = 4019 (X500 V2), hardware real (no `PX4_SITL`), versión de PX4 ≥ 1.16, sin errores (`num_logged_errors` = 0 y `error_labels` vacío) y entre 90 y 600 s. Elige **cinco** registros con algún tramo de estacionario en modo Position.
2. Descárgalos con `https://logs.px4.io/download?log=<id>` y pásalos por tu versión de `comparar_real_sitl.py`: ventana estacionaria automática, desviaciones típicas de `sensor_combined`, error de actitud, `hover_thrust_estimate`.
3. Si alguno trae `sensor_gyro_fifo` (perfil de registro con el bit 8), calcula su PSD con Welch y las frecuencias de giro y de paso de pala con `esc_status`.
4. Compara con el SITL de T1a (el mismo script sobre tu ulog).

**Comprueba:** el giróscopo y el acelerómetro reales se dispersan bastante más que en el SITL (en el registro de la presentación, 8 y entre 19 y 121 veces más); el HTE real queda lejos de 0,73; la PSD del SITL entre 20 y 50 Hz está cerca de \(S_0 = 2\sigma^2/f_s\) = 6,1·10⁻⁹ (rad/s)²/Hz una vez dividida por \(\cos^2(\pi f/f_s)\). Anota el rango de los cocientes entre los cinco registros: esa dispersión entre drones es lo que tendrás que aleatorizar.

## 2. Gazebo más parecido al dron real (60 min)

1. **Ruido identificado.** Con un tramo en reposo de un registro (el dron en el suelo, armado o no), calcula la varianza de Allan del giróscopo y del acelerómetro (B3a) y saca el ruido blanco y la inestabilidad del sesgo. Escríbelos en una copia del `x500_base` (`stddev`, `bias_mean`, `bias_stddev`, `dynamic_bias_stddev`, `dynamic_bias_correlation_time`). Comprueba con un registro del SITL que la varianza de Allan simulada se parece a la real.
2. **IMU a 1 kHz.** Ejecuta `crear_mundo_1khz.sh` sobre tu árbol de PX4 y arranca con `PX4_SIM_MODEL=gz_x500_1k PX4_GZ_WORLD=default_1khz`. Mide con `work_queue status` la frecuencia del lazo de *rate* y con `/world/default_1khz/stats` el tope del factor de tiempo real. Al cambiar \(f_s\), conserva la densidad de ruido: \(\sigma = \sqrt{S_0 f_s/2}\).
3. **Viento.** Repite `viento.py` en los mundos `default` y `windy` y calcula la inclinación media con `extraer_viento.py`. Cambia el viento del mundo a (0, 8, 0) m/s y predice la inclinación antes de volar con \(\tan\theta = 4 c_d \Omega |v| / (mg)\).
4. **Tu política.** Vuela en cada variante (base, ruido identificado, 1 kHz, viento) la política de A6b o el controlador de A2 por offboard, con la misma tarea y la verdad de Gazebo como referencia.

**Comprueba:** el lazo de *rate* a unos 1000 Hz y el tope de velocidad unas 3 veces menor que con 250 Hz; con (5, 2, 0) m/s, una inclinación de unos 3,8°; tu predicción con (0, 8, 0) a menos de 0,3° de lo medido. Escribe qué variante empeora más tu política y por qué.

## 3. La cámara neuronal (60 min, con GPU)

1. Descarga los modelos preentrenados de 3DGS (enlace en el README de `graphdeco-inria/gaussian-splatting`; licencia de investigación no comercial: no publiques imágenes) y quédate con una escena, por ejemplo `garden/point_cloud/iteration_30000/point_cloud.ply`.
2. Mide el coste del render con 25, 50 y 100 % de las gaussianas a 320×240, 640×480 y 1280×960 (modo `--sin-vuelo` de `camara_neuronal.py`). Ajusta una recta del tiempo frente al número de gaussianas.
3. Con PX4 SITL en marcha en el contenedor (red del anfitrión), lanza el bucle en vivo: poses de MAVSDK a 50 Hz, imágenes a 30 Hz, el x500 haciendo un círculo en offboard. Mide la edad de la pose al terminar cada imagen.
4. Sube las poses a 100 Hz (`set_rate_position_velocity_ned`) y repite. Después, **predice la pose** al instante de la imagen con la velocidad de MAVSDK (extrapolación lineal) y mide el error de posición de la cámara frente a la verdad de Gazebo con y sin predicción.

**Comprueba:** unos 4 ms por millón de gaussianas, casi sin depender de la resolución; edad mediana ≈ media espera + render (unos 31 ms a 50 Hz y unos 26 ms a 100 Hz con un render de 21 ms); la predicción reduce el error de la pose de la cámara en un círculo a 1,57 m/s.

## 4. La red de PX4 (40 min)

1. Compila `make px4_sitl_neural` en el contenedor, arranca con Gazebo y despega en modo normal. Pon `MC_NN_MANL_CTRL` a 0 y pasa a `commander mode ext1`. Publica consignas en `/fmu/in/trajectory_setpoint` desde ROS 2 (escalones de 2 m, como `neural_vuelo.py`).
2. Mide el error estacionario en altura en dos alturas.
3. Cambia `MC_NN_THRST_COEF` (por ejemplo 1,0 y 1,4; la documentación dice que bajarlo da más empuje) y repite. Calcula con la inversa de `RescaleActions()` qué salida de la red corresponde al mando medio en cada caso.

**Comprueba:** la red vuela el x500 y queda unos centímetros por debajo de la consigna de altura (7–16 cm en la presentación). Explica con tus números si el error estacionario cambia como esperarías al cambiar el coeficiente. El contador de rendimiento del módulo da 0 µs en lockstep: no lo uses como tiempo de cálculo.

## 5. SRCC entre tu simulador y Gazebo (40 min)

1. Entrena o toma cinco variantes de tu política de A6b (semillas distintas, o con y sin aleatorización de ruido).
2. Evalúa cada una en tu simulador JAX y en Gazebo con PX4 (offboard, sin pantalla, factor de velocidad el que aguante tu máquina) con la misma tarea y al menos 20 episodios por variante.
3. Calcula el SRCC (Pearson) y el Spearman entre las dos columnas, y el intervalo de Wilson de cada tasa de éxito.

**Comprueba:** tus intervalos con 20 episodios serán anchos (con un 90 % observado, de 0,70 a 0,97); di si con ellos puedes ordenar las variantes. Escribe cuántos episodios harían falta para separar las dos mejores.

## 6. Opcional: Isaac Lab 3.0 sin Kit (tiempo libre)

Instala Isaac Lab 3.0 con el motor Newton (sin Isaac Sim) siguiendo su documentación y entrena una tarea con un multirrotor si existe en tu versión, o porta el x500 de tu simulador. Mide pasos por segundo frente a tu simulador JAX en la misma GPU. Ojo con el orden de los cuaterniones: Isaac Lab 3.0 usa \((x, y, z, w)\) y PX4 \((w, x, y, z)\).

## Entregables

- `sim2real/brecha.py` y `experimentos/t1b_brecha.md`: tabla con los cinco registros reales y el SITL, y el rango de cada cociente.
- `sim2real/gazebo_real/`: el modelo del x500 con ruido identificado y la tabla del efecto de cada variante en tu política.
- `sim2real/camara_splats.py`: el bucle de la cámara con predicción de pose, y sus cifras de coste y latencia.
- `experimentos/t1b_srcc.md`: el SRCC, el Spearman y los intervalos de Wilson del paso 5.

## Preguntas de cierre

1. Los cocientes entre el dron real y el SITL cambian de un registro a otro. ¿Qué parte de esa variación es del dron (hélices, montaje de la IMU) y qué parte del vuelo (viento, piloto)? ¿Cómo lo separarías con registros públicos?
2. Si tu *belief* fusiona la imagen de la cámara de splats con la pose de PX4, ¿qué pasa si ignoras los 31 ms de diferencia? Relaciónalo con el NEES de A4a.
3. Para la escalera de simuladores de la parte 7, ¿qué peldaño te parece más barato con un SRCC alto frente al dron real para tu tarea de búsqueda, y qué experimento lo decidiría?
