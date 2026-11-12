# Laboratorio R1b · Un sistema real en números

**Duración:** unas 3 h.
**Requisitos:** cualquier ordenador con Python 3.10 o posterior. Solo CPU. Para las figuras, `matplotlib`; para los registros de vuelo, `pyulog` (`uv run --with matplotlib --with pyulog`). No hace falta GPU ni hardware: nada vuela. Los datos reales son fichas de fabricantes, documentos de la FAA y de EASA, artículos con licencia abierta y registros públicos de PX4 Flight Review.
**Material:** las cuentas y figuras de la presentación están en `../figuras/figuras_r1b.py` (su salida, en `../figuras/salida_ejemplos.txt`; cada constante lleva la clave de su referencia). Úsalo de referencia y copia lo que necesites (`wilson`, `fspl_db`). Escribe tus scripts en el proyecto hilo y guarda las salidas en `experimentos/r1b_*`. Los registros descargados son datos externos: cada uno en su propia carpeta, leídos con `pyulog` desde un script que esté en otra.

## Objetivo

Hacer con tus manos las cuatro lecturas de la jornada sobre un sistema real:

- su arquitectura, con cada dato en su fuente y su estado;
- las cuentas que dicen a qué velocidad puede ir, cuánto puede volar y cuánto deriva sin GNSS;
- qué fiabilidad se puede afirmar con las horas que publica;
- y cómo reacciona un estimador real (el EKF2 de PX4) cuando el GNSS se degrada.

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Mavic 4 Pro: distancia de parada a 18 m/s con τ = 0,1 s | 25,4 m | 2 |
| Mavic 4 Pro: velocidad máxima para parar en 24 m con τ = 0,1 s | 17,5 m/s | 2 |
| FlyCart 30 con 30 kg: potencia ideal de estacionario y eficiencia global | 7454 W; 0,56 | 2 |
| Zipline P1 a 60 mph con L/D = 10 y η = 0,6 | 994 W; 10,3 Wh/km | 2 |
| Deriva inercial pura en 60 s con los sesgos de B3a | 55,9 m | 2 |
| Horas sin fallos para descartar 10⁻⁷ /h al 95 % | 29 957 323 h | 3 |
| J/S de 1 W a 40 km frente a GPS L1 C/A; distancia que iguala 30,1 dB | 30,1 dB; 39,8 km | 2 |

## 1. La ficha de arquitectura (45 min)

1. Elige un sistema que **no** esté desmontado en la presentación: un dron de reparto, de inspección o de seguridad pública con documentación pública. Empieza por lo que publica un regulador, que es más fiable que una web comercial: las evaluaciones ambientales de la FAA (`https://www.faa.gov/uas/advanced_operations/nepa_and_drones`) traen masa, carga, velocidad, perfil de vuelo y procedimientos; los criterios de aeronavegabilidad del Federal Register, masa y velocidad máximas.
2. Rellena las siete cajas de la plantilla de la parte 2: sensores, estimación, autonomía (y qué hace sin enlace), control, cómputo, enlace y operación.
3. Cada dato con su fuente, su fecha y su estado: **C** (leído en la fuente primaria), **E** (solo lo dice la empresa) o **P** (dos medios fiables e independientes, o una agencia leída en su texto íntegro). Lo que no puedas comprobar va en gris, como «no publicado».

**Comprueba:** al menos diez datos, y ninguno sin fuente. Escribe en tres líneas qué cajas quedan en gris y qué tendría que publicar la empresa para que su sistema fuera auditable. Si alguna cifra de la empresa contradice la del regulador (como el crucero del P2 de Zipline: 70 mph en su ficha, 47 mph en la evaluación de la FAA), anota las dos.

## 2. Las cuentas del sistema (50 min)

1. **Envolvente de velocidad segura.** Programa \(v_{\max}(R, \tau) = a\left(-\tau + \sqrt{\tau^2 + 2R/a}\right)\) con \(a = g\tan\theta_{\max}\). Dibuja \(v_{\max}\) frente a \(R\) (de 5 a 120 m) para \(\tau\) = 0,05; 0,1; 0,3 y 0,5 s. Marca la velocidad y el alcance publicados de tu sistema (si es un multirrotor con evitación) y los del Mavic 4 Pro y el Skydio X10.
2. **Energía y alcance.** Si es un multirrotor: con la teoría de cantidad de movimiento (A1), \(P_i = T^{3/2}/\sqrt{2\rho A}\), y con la batería y el tiempo de estacionario publicados, saca su eficiencia global \(\eta_g\). Si es un ala o un híbrido: \(P = Wv/((L/D)\eta)\), con un barrido de \(L/D\) entre 6 y 15 y \(\eta\) entre 0,5 y 0,7, y compara el alcance que sale con el publicado. Calcula los Wh por km y por kg y sitúalo en la figura de la parte 7.
3. **Deriva sin GNSS.** A la velocidad de crucero de tu sistema, dibuja el error de posición frente a la distancia para: inercial pura (\(\tfrac12 b_a t^2 + \tfrac16 g\,b_g t^3\) con los sesgos de B3a), odometría con \(\delta\) = 0,6 % y 7,3 % (los extremos de SPRIN-D) y un error acotado de unos 10 m (la corrección con mapa). ¿A qué distancia deja de servir cada una si tu sistema necesita 20 m de precisión para entregar?
4. **Interferencia.** Reproduce con Friis la J/S de un interferidor de 1 W frente a GPS L1 C/A (−128,5 dBm) a 1, 10, 40 y 100 km, y la tolerancia \(10\log_{10}R_c - (C/N_0)_{\text{umbral}}\). Haz lo mismo para el enlace de mando de tu sistema si publica su potencia, o con 33 dBm de PIRE si no.

**Comprueba:** los siete valores de la tabla de arriba con tu código (con el Mavic 4 Pro, el FlyCart 30 y el P1). En tu sistema, di si su velocidad máxima publicada cabe en la envolvente con alguna τ razonable y qué τ implicaría.

## 3. Fiabilidad: horas necesarias frente a horas publicadas (30 min)

1. Programa \(\lambda_{95} = -\ln\alpha / T\) y la tabla inversa: horas sin fallos para descartar 10⁻⁴, 10⁻⁵, 10⁻⁶, 10⁻⁷ y 10⁻⁸ por hora con el 95 %.
2. Convierte a horas lo que publican las flotas: las 150 M de millas de Zipline (supón 47 y 60 mph, y di qué cambia), las 908 h de pruebas del MK30 y las 18 000 h del V-BAT. Para cada una, ¿qué tasa descartaría si no hubiera habido ningún fallo?
3. Con fallos, la regla del tres no vale. Para \(N\) fallos en \(T\) horas, la cota superior unilateral al 95 % de Poisson es \(\chi^2_{0{,}95}(2N+2)/(2T)\); con \(N = 0\) vuelve a dar \(\ln 20/T\). La función `poisson_exacto` de `../../R1a_labs-robot-learning-2026/figuras/ejemplos_r1a.py` da el intervalo bilateral: su extremo superior con `conf=0.90`, dividido por \(T\), es esta misma cota. Calcula cuántas horas haría falta volar con 1, 3 y 10 fallos para afirmar 10⁻⁶ /h.
4. Para el V-BAT, Reuters da más de 50 aeronaves perdidas de unas 200 en 18 meses, pero no las horas de esa flota. Escribe qué dato falta para convertir eso en una tasa por hora y qué rango saldría con supuestos razonables (dilos).

**Comprueba:** \(\ln 20 = 2{,}996\); 29 957 323 h para 10⁻⁷; Zipline entre 0,9·10⁻⁶ y 1,2·10⁻⁶ /h según la velocidad. Explica en dos líneas por qué EASA no pide demostrar 10⁻⁷ con horas.

## 4. El EKF de PX4 cuando se degrada el GNSS (55 min)

1. Descarga el índice público de Flight Review (`https://logs.px4.io/dbinfo`, que redirige a un JSON comprimido de unos 33 MB) en su propia carpeta de datos. Filtra como en T1b: hardware real (descarta `PX4_SITL`), un cuadricóptero (`sys_autostart_id`; T1b usó 4019, el X500 V2), PX4 1.15 o posterior, entre 120 y 1800 s. Descarga **una veintena** de candidatos, uno a uno y sin prisa, con `https://logs.px4.io/download?log=<id>`.
2. Busca en cada registro, solo en vuelo (`vehicle_land_detected.landed` falso), un tramo con el GNSS degradado: en `sensor_gps` (o `vehicle_gps_position`), `fix_type` por debajo de 3, una caída de `satellites_used`, `eph` que se dispara, o `jamming_state` = 2 (aviso) o 3 (crítico) en receptores que lo informan. Si no encuentras ninguno, amplía la búsqueda o repite el paso 3 del GNSS del laboratorio de T1a en el SITL (allí `failure gps off` no hacía nada y el corte se provoca con `param set EKF2_GPS_CTRL 0`) y dilo.
3. En el tramo elegido, dibuja en una misma escala de tiempo:
   - del receptor: `fix_type`, `satellites_used`, `eph`, `jamming_indicator`;
   - del EKF2: en `estimator_status_flags`, `cs_gps` (hasta 1.15) o `cs_gnss_pos` (1.16 y 1.17), y `cs_inertial_dead_reckoning`; en `estimator_status`, los bits de `gps_check_fail_flags` y `pos_test_ratio`;
   - del estimado: en `vehicle_local_position`, `eph`, `xy_valid` y `dead_reckoning`;
   - de la reacción: en `failsafe_flags`, `global_position_invalid` y `local_position_invalid`; en `vehicle_status`, `nav_state` (por ejemplo POSCTL = 2, AUTO_RTL = 5, DESCEND = 12, AUTO_LAND = 18).
4. Mide tres retardos: desde que el receptor empeora hasta que el EKF deja de fusionar GNSS, hasta que salta el *failsafe* y hasta que cambia el modo. Mide también cuánto crece el `eph` del estimado durante la navegación a estima y compáralo con la curva inercial del paso 2.

**Comprueba:** una figura con las cuatro filas y una tabla con los tres retardos en milisegundos, con el identificador del registro y su versión de PX4. Escribe qué parte de lo que ves es un *belief* (la covarianza que crece) y qué parte es una regla (el umbral que dispara el *failsafe*), y qué haría distinto tu política con *belief* de la tesis.

## Entregables

- `experimentos/r1b_ficha.md`: la ficha de arquitectura del paso 1, con su tabla de fuentes y estados.
- `src/proyecto_hilo/sistemas_reales/cuentas.py` y `experimentos/r1b_cuentas.md`: la envolvente, la energía, la deriva y la J/S, con las comprobaciones de la tabla de arriba.
- `experimentos/r1b_fiabilidad.md`: la tabla de horas necesarias frente a horas publicadas y el caso con fallos.
- `src/proyecto_hilo/sistemas_reales/ekf_gnss.py` y `experimentos/r1b_ekf_gnss.md`: la figura del paso 4, la tabla de retardos y los identificadores de los registros.
