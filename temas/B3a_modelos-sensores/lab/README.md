# Laboratorio B3a · Varianza de Allan de una IMU real y un modelo de IMU validado

**Duración:** unas 3,5 h (el paso 5 es opcional).
**Requisitos:** numpy o JAX, y la secuencia *imu-static* de TUM VI (Calibration Sequences, en la página del dataset: <https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset>). Para el paso 4, el ESKF de A4a; para el paso 5, una secuencia de EuRoC (A4a). Todo en CPU.

## Objetivo

Caracterizar con datos reales el ruido de una IMU MEMS, construir un modelo de IMU para el simulador del proyecto hilo y validarlo con la misma herramienta con la que se caracterizó el sensor. Comprueba con tus números lo que calcula la presentación:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Valores de Allan publicados por TUM VI (BMI160), acelerómetro | \(\sigma_a\) = 0,0014 m/s²/√Hz; \(\sigma_{b_a}\) = 0,000086 m/s³/√Hz | 1 |
| Valores de Allan publicados por TUM VI, giróscopo | \(\sigma_g\) = 0,000080 rad/s/√Hz; \(\sigma_{b_g}\) = 0,0000022 rad/s²/√Hz | 1 |
| Giróscopo simulado de 12 h: \(N\) leído en 1 s; \(K\) leído en 3 s | 7,99·10⁻⁵ (verdadero 8,0·10⁻⁵); 1,93·10⁻⁶ (verdadero 2,2·10⁻⁶) | 2 |
| Deriva de posición sin corrección con la IMU de EuRoC (un eje) | 25 cm a 10 s; 29 m a 60 s | 3 |
| Cizalla del obturador rodante: \(f\) = 458,65 px, 5 rad/s, 30 ms | 69 px | 5 |

El script de las figuras está en `../figuras/figuras_b3a.py`.

## 1. Allan de una IMU real (60 min)

1. Descarga *imu-static* de TUM VI (lee en la página el formato: cada fila es tiempo, giróscopo, acelerómetro y temperatura). Guárdala en su propia carpeta de datos, fuera del código.
2. Comprueba la frecuencia real de muestreo con las marcas de tiempo (huecos, duplicados) y la temperatura a lo largo de la grabación.
3. Implementa la desviación de Allan solapada con la suma acumulada (ángulo o velocidad integrados), para unos 50 valores de \(\tau\) espaciados logarítmicamente entre 0,005 s y una décima parte de la duración. Con 111 h a 200 Hz son unos 80 millones de muestras por eje: trabaja en `float64` y por ejes.
4. Ajusta las rectas −1/2 y +1/2 y lee \(N\) en \(\tau\) = 1 s y \(K\) en \(\tau\) = 3 s; el mínimo da la inestabilidad del sesgo \(B\) = mínimo/0,664. Calcula la barra de error de cada punto con el número de ventanas independientes.

**Comprobaciones:** tus valores deben estar cerca de los publicados por TUM VI (tabla). Representa los seis ejes: ¿se parecen entre sí? ¿Hay algún tramo que no siga ninguna pendiente de la tabla (por ejemplo, por la temperatura)?

**Piensa:** repite con solo 1 h y con solo 10 min de datos. ¿Cuál de los tres parámetros deja de poder leerse antes?

## 2. Un modelo de IMU para el simulador (40 min)

En `src/proyecto_hilo/sensores/imu.py`, una función pura de JAX `imu_paso(clave, estado_imu, omega, f, dt) -> (medida, estado_imu)` con:

- ruido blanco \(\sigma/\sqrt{\Delta t}\), paseo del sesgo \(\sigma_b\sqrt{\Delta t}\) y sesgo inicial aleatorio;
- opcional: inestabilidad del sesgo como proceso de Gauss-Markov con su tiempo de correlación, factor de escala y desalineamiento fijos por episodio;
- la medida del acelerómetro es la fuerza específica en ejes cuerpo, \(R^\top(\dot{\mathbf v} + g\mathbf e_3)\), como viste con NeuroBEM en A1.

**Comprobaciones:** genera 12 h simuladas con los parámetros de TUM VI (con `jax.lax.scan` o vectorizado por bloques) y calcula su desviación de Allan con tu función del paso 1. Debe solaparse con la real en la zona de ruido blanco y del paseo del sesgo. Donde no se solape, explica qué falta en el modelo.

## 3. Navegación inercial pura (30 min)

1. Integra la IMU simulada en reposo (actitud, velocidad y posición) durante 60 s, 500 veces, con los parámetros de EuRoC y con los de TUM VI.
2. Desactiva por turnos cada término (blanco del acelerómetro, sesgo del acelerómetro, blanco del giróscopo, sesgo del giróscopo).

**Comprobaciones:** la desviación típica de la posición en un eje horizontal frente al tiempo sigue las fórmulas de la presentación (\(t^{3/2}\), \(t^{5/2}\), \(t^{5/2}\), \(t^{7/2}\)); con EuRoC, unos 25 cm a 10 s y 29 m a 60 s.

## 4. Vibración, filtro y su efecto en el ESKF (40 min)

1. Añade a la IMU simulada vibración: senoides a la frecuencia de giro de cada rotor y a su segundo y tercer armónico, con amplitud proporcional al empuje (elige una amplitud que dé una aceleración RMS de vibración de unos pocos m/s², y dilo).
2. Simula el muestreo a alta frecuencia y el filtro de paso bajo del giróscopo de PX4 (A2), y decima a la frecuencia de tu estimador.
3. Pasa las medidas a tu ESKF de A4a (con posición a 10 Hz) con tres configuraciones del filtro: densidades de Allan, densidades ×2 y ×10 (como TUM VI para Kalibr), y la vibración activada o no.

**Mide:** NEES de posición y de actitud en cada caso. ¿Con qué inflado vuelve a ser consistente el filtro cuando hay vibración? Compáralo con lo que hicieron TUM VI, OKVIS y ROVIO.

## 5. GNSS y cámara (opcional, 40 min)

1. **GNSS:** un Gauss-Markov de primer orden (por ejemplo, 1 m y 60 s; dilo si eliges otros), con 150 ms de retardo, pérdidas de 10 s y un `eph` coherente. Fúndelo en tu ESKF y compara con el ruido blanco de 1 m y con el modelo de PX4 (2,5 cm). ¿Qué modelo da un NEES consistente con el `eph` declarado?
2. **Obturador rodante:** con imágenes de EuRoC y la velocidad angular de su referencia, deforma cada imagen desplazando cada fila \(f\,\omega\,t_{ro}\,v/H\) (elige \(t_{ro}\) = 30 ms). Detecta y empareja características (OpenCV) en las imágenes originales y en las deformadas. ¿Cuánto aumenta el error de reproyección de la geometría epipolar?

## Entregables

- `sensores/allan.py` y `sensores/imu.py`, con tests: la Allan del ruido blanco puro tiene pendiente −1/2 (±0,05) y vale \(N\) en 1 s; la del paseo aleatorio puro vale \(K\sqrt{\tau/3}\); con todos los ruidos a cero, la IMU devuelve la verdad.
- `experimentos/b3a_imu.py`, que genere: Allan real frente a simulada, deriva por términos y NEES del ESKF con y sin vibración.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Qué parámetros de IMU pondrías en el simulador con el que entrenas una política, y cuáles en el estimador que lleva el dron? ¿Deben ser los mismos?
2. El sesgo de la IMU es estado oculto. ¿Lo estimaría una política recurrente entrenada con tu IMU simulada? ¿Cómo lo comprobarías?
3. Si en tu tesis usas un world model que recibe la IMU cruda, ¿qué frecuencia y qué filtrado le das? Piensa en la vibración, el aliasing y el coste de cómputo.
