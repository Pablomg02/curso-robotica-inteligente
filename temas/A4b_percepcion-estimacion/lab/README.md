# Laboratorio A4b · Estimación visual-inercial y el efecto del desfase temporal

**Duración:** unas 3,5 h (el paso 4 es opcional).
**Requisitos:** el ESKF de 15 estados del laboratorio A4a y una secuencia de EuRoC descargada (A4a, paso 0). Los pasos 1 y 3 solo necesitan Python; el paso 2 compila OpenVINS (Linux, o un contenedor en otro sistema). Todo en CPU.

## Objetivo

Medir con tus propios números lo que la presentación calcula sobre el desfase temporal cámara-IMU, ver su efecto en un VIO real y practicar la geometría de la cámara:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Error RMS con 30 ms de desfase: ignorándolo, estimándolo, conocido | 8,8 cm; 1,3 cm; 1,2 cm | 1 |
| Desfase estimado (arranca en 0 ms) | 33 ms al final; dentro del 10 % a los 2 s | 1 |
| Error de reproyección equivalente \(f\omega t_d\) con \(f\) = 458,65 px | 9,2 px a 2 rad/s y 10 ms | 3 |
| Incertidumbre de profundidad estéreo de EuRoC a 10 m (\(\sigma_d\) = 0,5 px) | unos 99 cm | 3 |

El script de las figuras está en `../figuras/figuras_a4b.py`.

## 1. El desfase en tu ESKF (60 min)

1. **Simulación primero.** Reproduce la sección 2 del script con tu ESKF de A4a reducido a posición y velocidad: IMU (acelerómetro) a 200 Hz y una «posición visual» a 20 Hz tomada 30 ms antes de su marca de tiempo. Compara tres filtros: ignorando el desfase, conociéndolo y estimándolo como estado con \(\mathbf p(t - t_d) \approx \mathbf p(t) - \mathbf v(t)\,t_d\).
2. **Con EuRoC.** Usa tu ESKF de 15 estados con la IMU real. Construye la medida «visual» con la posición de `state_groundtruth_estimate0`, con 2 cm de ruido, a 20 Hz y retrasada 0, 10, 30 y 60 ms. Repite los tres filtros.

**Comprobaciones:** en simulación, las cifras de la tabla con ±20 %. Con EuRoC, representa el error RMS de posición frente al desfase para los tres filtros. ¿En qué tramos de la secuencia se nota más? Relaciónalo con la velocidad.

3. Repite la estimación con un tramo de vuelo casi estacionario. ¿Converge el desfase? Explícalo con el jacobiano.

## 2. Un VIO real: OpenVINS en EuRoC (75 min)

1. Instala OpenVINS siguiendo la guía de [docs.openvins.com](https://docs.openvins.com/) y ejecútalo en la misma secuencia con la configuración `config/euroc_mav` del repositorio.
2. Evalúa la trayectoria frente a la referencia con [evo](https://github.com/MichaelGrupp/evo) (`evo_ape`, alineando con SE(3); di qué alineación usas).
3. Desplaza las marcas de tiempo de las imágenes 10 y 30 ms (reescribe la columna de tiempo de `cam0/data.csv` y `cam1/data.csv`, o el equivalente en el formato que uses) y repite con `calib_cam_timeoffset: true` y `false` en `estimator_config.yaml`.

**Mide:** ATE de posición en los cinco casos y el desfase que estima OpenVINS cuando la calibración está activada. ¿Se parece el comportamiento al de tu filtro del paso 1?

## 3. Geometría de la cámara (30 min)

1. Lee los intrínsecos y los extrínsecos de las cámaras de EuRoC (en el `sensor.yaml` de cada cámara, o en `kalibr_imucam_chain.yaml` de la configuración de OpenVINS). Calcula la base estéreo.
2. Implementa la proyección \(\pi\) y su jacobiano en JAX; comprueba el jacobiano con `jax.jacfwd`.
3. Tabla de \(f\omega t_d\) y de \(\sigma_Z = Z^2\sigma_d/(fb)\) de la presentación.
4. Triangula unos cuantos puntos con dos imágenes estéreo de la secuencia (detección y emparejamiento con OpenCV) y compara la dispersión de su profundidad con \(\sigma_Z\).

## 4. Odometría inercial aprendida mínima (opcional, 45 min)

Una versión de juguete de TLIO o de Cioffi et al.:

1. Ventanas de 1 s de IMU (giróscopo y acelerómetro rotados a un marco sin guiñada con la actitud de la referencia) y, como etiqueta, el desplazamiento de la referencia en esa ventana.
2. Una red pequeña (MLP o convolución 1D) en JAX que predice el desplazamiento; entrénala con varias secuencias de EuRoC y evalúala en otra.
3. Funde sus predicciones en tu ESKF como medidas de desplazamiento entre clones de la pose.

¿Mejora a la integración pura de la IMU? ¿Y fuera de las secuencias de entrenamiento?

## Entregables

- `estimacion/desfase.py` (filtro con el desfase como estado) y `percepcion/camara.py` (proyección, jacobiano y triangulación), con tests: el jacobiano coincide con `jax.jacfwd`; con desfase nulo, los tres filtros del paso 1 dan lo mismo.
- `experimentos/a4b_desfase.py`, que genere: error frente a desfase en simulación y con EuRoC, y la tabla de ATE de OpenVINS.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. Tu dron de tesis llevará una cámara y una IMU en relojes distintos. ¿Qué harías para que el desfase no supere 1 ms, y qué harías si no puedes garantizarlo?
2. En el paso 2, ¿qué parte del error de OpenVINS se debe al desfase y qué parte a otras cosas? ¿Cómo lo separarías?
3. ¿Qué entradas le darías a tu world model para que aprenda a compensar un desfase variable?
