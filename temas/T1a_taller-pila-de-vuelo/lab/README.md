# Laboratorio T1a · Tu pila de vuelo simulada, medida

**Duración:** unas 4 h.
**Requisitos:** un PC con Linux (8 núcleos y 16 GB bastan), Podman o Docker, y unos 10 GB de disco. La GPU solo hace falta para las cámaras (paso 2.5, opcional); sin ella funciona todo lo demás. QGroundControl 5.1 (AppImage) en tu escritorio. Nada vuela de verdad.
**Material:** `Containerfile` de esta carpeta. Los scripts que generaron las cifras de la presentación están en `../figuras/experimentos/`: úsalos como referencia cuando te atasques, pero escribe los tuyos.

## Objetivo

Montar la pila de vuelo real (PX4 v1.17.0 SITL, Gazebo Harmonic, ROS 2 Jazzy, Micro XRCE-DDS Agent v2.4.3, MAVSDK 4) y medir con tus propios scripts lo que la presentación midió:

- el lazo de *rate* a 250 Hz en SITL, frente a 800 Hz en una Pixhawk;
- el error de 1,57 m (modelo) o unos 1,8 m (medido) en un círculo sin velocidad de prealimentación, y la mejora con ella;
- el adelanto de la posición del EKF2, de \(v \times 0{,}11\) m, por el retardo de GNSS supuesto;
- el ACK de unos 10 ms de un comando enviado desde ROS 2;
- el tope de velocidad de la simulación en tu máquina.

Todo va al proyecto hilo, en `src/proyecto_hilo/sitl/` (scripts) y `experimentos/t1a_*` (salidas).

## 0. El contenedor (30 min, casi todo de espera)

1. Construye la imagen. En el PC del curso (28 hilos) tardó 8 min y ocupa 7,5 GB; en una máquina modesta, cuenta con 20–30 min:
   ```bash
   podman build -t taller-px4 -f Containerfile .
   ```
2. Arráncala con la red del anfitrión, para que QGroundControl vea el puerto 14550, y con una carpeta compartida:
   ```bash
   mkdir -p ~/taller && podman run -it --name px4 --network host -v ~/taller:/trabajo taller-px4 bash
   ```
   Si vas a usar cámaras (paso 2.5), añade `--device /dev/dri/renderDXXX --device /dev/dri/cardX` **solo de la GPU que tenga controlador dentro del contenedor**. Con Mesa, eso es la integrada (Intel o AMD). Si pasas también una NVIDIA sin su controlador, EGL se cuelga.
3. Abre más terminales en el mismo contenedor con `podman exec -it px4 bash`.

**Comprueba:** `gz sim --versions` da 8.x, `ls /px4/build/px4_sitl_default/bin/px4` existe y `python3 -c "import mavsdk; print(mavsdk.__file__)"` funciona.

## 1. Arrancar y cronometrar (20 min)

1. Lanza la simulación sin pantalla:
   ```bash
   cd /px4 && HEADLESS=1 make px4_sitl gz_x500
   ```
   Abre QGroundControl en tu escritorio: debe conectarse solo.
2. Repite el arranque marcando el tiempo de cada línea (mira `run_sitl.sh` en `figuras/experimentos/`) y apunta cuándo pasa cada hito: Gazebo lanzado, mundo listo, `gz_bridge`, MAVLink, `rcS` terminado y «Ready for takeoff».

**Comprueba:** el mundo tarda unos 5 s en estar listo y `rcS` termina en unos 10 s. Si tu máquina es más lenta, que sea más, pero el orden de los hitos debe ser el mismo.

## 2. PX4 y Gazebo por dentro (50 min)

1. En la consola `pxh>` (o con `bin/px4-uorb top -1` desde otra terminal), anota la frecuencia de `sensor_gyro`, `vehicle_angular_velocity`, `vehicle_attitude`, `vehicle_local_position`, `actuator_motors` y `sensor_gps`.
2. `work_queue status`: ¿en qué cola va `mc_rate_control` y a qué frecuencia? ¿Y `ekf2`?
3. `param show IMU_GYRO_RATEMAX` e `IMU_INTEG_RATE`. Explica por escrito por qué el lazo de *rate* no va a 800 Hz aunque el parámetro lo diga. Pista: `grep update_rate Tools/simulation/gz/models/x500_base/model.sdf`.
4. **Ejes.** Con el dron en el suelo:
   ```bash
   gz topic -e -t /world/default/model/x500_0/link/base_link/sensor/imu_sensor/imu -n 1
   ```
   frente a `listener sensor_combined` en `pxh>`. Comprueba el signo de \(z\) del acelerómetro en los dos y relaciónalo con \(\mathrm{diag}(1,-1,-1)\). Después, despega, gira el dron en guiñada a mano (QGroundControl, joystick virtual, modo Position) y comprueba el signo de la velocidad angular en \(z\).
5. **Opcional, cámaras.** Arranca `gz_x500_mono_cam` sin pantalla y mide la frecuencia de la imagen con `gz topic -f -t <tema> -d 10`. Pásala a ROS 2 con `ros_gz_bridge` y mira la imagen en `rqt_image_view`.

**Comprueba:** 250 Hz en la cadena IMU → *rate* → motores; 125 Hz en la posición local; 30 Hz en el GNSS. La cuenta de la hoja 18 de la presentación: \(540^\circ \cdot 6{,}3/250\) = 13,6° frente a 4,3° a 800 Hz.

## 3. QGroundControl (20 min)

1. Planifica una misión de cuatro puntos, guárdala como `.plan` y ábrela con un editor: identifica las coordenadas, la altura y el tipo de comando de cada punto.
2. Vuélala. Mientras tanto, en *Analyze → MAVLink Inspector*, apunta la frecuencia de `ATTITUDE`, `GLOBAL_POSITION_INT` y `GPS_RAW_INT`.
3. Abre la consola MAVLink de QGroundControl y ejecuta `mavlink status`.

## 4. MAVSDK 4 (60 min)

Escribe `sitl/mavsdk_taller.py` con la API **síncrona** de MAVSDK 4 (`from mavsdk import Mavsdk, Configuration, ComponentType`; los plugins se construyen: `Action(drone)`). Si copias un ejemplo de internet con `System()` y `await`, es de la 3.x: no funcionará.

1. **Conexión y salud:** `add_any_connection("udpin://0.0.0.0:14540")`, `first_autopilot`, espera a `health_all_ok`. Mide los tiempos.
2. **Despegue a 5 m.** Reintenta `arm()` si lo deniega. Espera a la altura con `subscribe_position_velocity_ned`.
3. **Círculo en offboard:** radio 5 m, periodo 20 s, consignas a 20 Hz, dos vueltas. Primero solo con `set_position_ned` y después con `set_position_velocity_ned` (velocidad del círculo como prealimentación). Antes de `offboard.start()` manda al menos 1 s de consignas.
4. **Escalón:** 5 m al norte, sin velocidad, durante 8 s.
5. **Misión:** cuadrado de 20 m a 5 m/s. Espera a una `home` válida (no NaN) y deja un segundo, o espera a que la misión sea válida, antes de `start_mission()`.
6. Aterriza y copia el ulog (`build/px4_sitl_default/rootfs/log/…`) a tu carpeta de experimentos.

**Comprueba** (leyendo el ulog con `pyulog`, posición verdadera `vehicle_local_position_groundtruth` frente a `trajectory_setpoint`):

- error medio del círculo sin prealimentación entre 1,5 y 2,1 m (el modelo da 1,57 m; se midieron 1,80 m), y con ella por debajo de 0,6 m;
- escalón: subida del 10 al 90 % en 1,2–1,8 s y sobreoscilación menor del 5 %;
- misión: entre 25 y 32 s (el modelo con *jerk* limitado da 25,7 s).

## 5. ROS 2 (50 min)

1. Arranca el agente en otra terminal: `MicroXRCEAgent udp4 -p 8888`. Comprueba con `ros2 topic list` que aparecen `/fmu/out/…` y `/fmu/in/…`. ¿Cuántos temas llevan el sufijo `_v1`?
2. Escribe un nodo `rclpy` que se suscriba a `/fmu/out/vehicle_odometry` **con la QoS por defecto**. Comprueba que no recibe nada. Cámbiala a `BEST_EFFORT` + `TRANSIENT_LOCAL` y comprueba que sí.
3. Mide durante 10 s la frecuencia de `sensor_combined`, `vehicle_odometry` y `vehicle_local_position_v1`, y la edad de la odometría: reloj de ROS en microsegundos menos `timestamp_sample`. Explica los 100 y 50 Hz con `dds_topics.yaml`.
4. Offboard desde ROS 2: `OffboardControlMode` + `TrajectorySetpoint` a 20 Hz y `VehicleCommand` 176 (1, 6) y 400 (1). Sin QGroundControl ni MAVSDK abiertos no armará: compruébalo, lee el mensaje de PX4 y arréglalo con `param set NAV_DLL_ACT 0` (solo en simulación).
5. Mide 20 veces la latencia del ACK de un cambio de modo (176 con 1, 3): desde que publicas hasta que llega `vehicle_command_ack`.
6. Graba 30 s de `/fmu/out/*` en MCAP: `ros2 bag record -s mcap -a`. Anota el tamaño.

**Comprueba:** unos 100 Hz en `sensor_combined` y la odometría, unos 50 Hz en la posición local, una edad de la odometría de pocos milisegundos y un ACK con una mediana de 10–15 ms y un máximo por debajo de 30 ms.

## 6. Registros y fallos (40 min)

1. Con el ulog del paso 4, calcula el error horizontal de `vehicle_local_position` frente a la verdad de Gazebo **por tramos de velocidad** y ajusta \(\mathbf e = \mathbf v\,\tau\). Compara \(\tau\) con `EKF2_GPS_DELAY`.
2. Pon `EKF2_GPS_DELAY` a 0 en un vuelo nuevo y repite: ¿desaparece el adelanto?
3. **GNSS:** en estacionario a 10 m, prueba `failure gps off` (con `SYS_FAILURE_EN` = 1) y comprueba en el registro que no pasa nada. Después, `param set EKF2_GPS_CTRL 0`. Mide cuándo salta el *failsafe* y qué modo elige PX4.
4. **Motor:** `failure motor off -i 1` con y sin `CA_FAILURE_MODE` = 1. Mide el tiempo hasta 90° de inclinación con `vehicle_attitude_groundtruth`.

**Comprueba:** \(\tau\) a menos de 10 ms de 110 ms; con `EKF2_GPS_DELAY` = 0, el error en movimiento baja a unos pocos centímetros. *Failsafe* «blind land» a los 3–6 s de quitar el GNSS; vuelco por debajo de 1 s con el motor apagado.

## 7. Escala (20 min)

1. Con `HEADLESS=1`, mide el factor de tiempo real alcanzado con `PX4_SIM_SPEED_FACTOR` de 1, 4, 16 y 100. Léelo con `gz topic -e -t /world/default/stats -n 1` (tiempo simulado frente al de pared) en dos instantes separados 10 s.
2. Lanza 4 drones en el mismo mundo (el primero sin `PX4_GZ_STANDALONE`; los demás con él y con `PX4_GZ_MODEL_POSE` distintos; `-i 1..4`) y repite la medida del tope.

**Comprueba:** lo pedido se cumple hasta un factor que depende de tu CPU, y el producto «drones × tope» se mantiene aproximadamente constante.

## Entregables

- `sitl/mavsdk_taller.py`, `sitl/ros2_taller.py` y un `sitl/analiza_ulog.py` que reproduzcan las comprobaciones de los pasos 4–6 con `pytest` (con tolerancias: dos ejecuciones no dan lo mismo bit a bit).
- `experimentos/t1a_resumen.md`: una tabla con tus números junto a los de la presentación y una línea por cada diferencia que encuentres.
- **Sim-to-sim:** las pruebas de estacionario, escalón y círculo en tu simulador JAX con tu controlador de A2, y una tabla que las compare con Gazebo y PX4. Explica cada diferencia: ¿modelo, controlador o estimador?

## Preguntas de cierre

1. ¿Por qué el SITL en *lockstep* no puede decirte nada del *jitter* de tu política en el ordenador de a bordo? ¿Qué parte de la cadena tendrías que medir en hardware (C4, C5)?
2. Has visto que el EKF2 va \(v \times 0{,}11\) m por delante en SITL. Si evaluaras una política con *belief* comparando su estimación con la del EKF2 en lugar de con la verdad, ¿qué conclusión errónea podrías sacar?
3. Para el hito H4, ¿desplegarías tu política como nodo offboard o como modo externo con `px4-ros2-interface-lib`? Argumenta con los *failsafes* y con la experiencia del operador.
