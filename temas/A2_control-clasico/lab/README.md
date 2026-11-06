# Laboratorio A2 · La cascada de PX4 en el proyecto hilo, a frecuencias reales

**Duración:** unas 3,5 h (los pasos 6 y 7 son opcionales si vas justo).
**Requisitos:** el modelo de 17 estados del laboratorio A1 (`dinamica/cuadricoptero.py`) con los parámetros del x500. CPU suficiente; la GPU solo acelera el paso 3.

## Objetivo

Añadir al proyecto hilo un simulador **multifrecuencia** y la cascada de control de un multirrotor tal como la implementa PX4: lazo de velocidad angular con su PID, sus filtros y su anti-windup, lazo de actitud con error de cuaternión y prioridad a la inclinación, y lazos de velocidad y posición. Después, medir lo que la presentación calcula:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Cruce y margen de fase del lazo de *rate* del x500 con las ganancias de PX4 a 250 Hz (cuenta lineal) | 6,3 Hz y 71° | 3 |
| Retardo extra que aguanta ese lazo | unos 31 ms | 4 |
| Escalón de 1 rad/s: subida al 90 % y sobrepaso | 76 ms y 5 %; con +20 ms, 47 % | 2 y 4 |
| Cascada con interno de 2.º orden: sobrepaso con \(r = \omega_i/k\) = 2 y 4 | 23 % y 0 % | 5 |
| Perturbación de 0,3 N·m solo con el lazo de *rate*: deriva de ángulo a 1 s | PID 11,7°, INDI 0,4° | 6 |
| Guiado L1 con \(L_1\) = 80 m a 25 m/s desde 30 m: tiempo al 5 % | 6,6 s | 7 |

El script que generó las cifras está en `../figuras/figuras_a2.py`. Úsalo para comparar al final de cada paso, no antes.

## 0. Preparación (10 min)

- Convenio del proyecto hilo (A1): mundo ENU, cuerpo FLU, cuaterniones de Hamilton `[w, x, y, z]`. PX4 trabaja en NED/FRD: convierte en la frontera y deja una nota en cada función que copies de su código.
- `float64` para los tests (`jax.config.update("jax_enable_x64", True)`).
- Lee, en el código de PX4 (rama `main`), estos tres ficheros, que vas a imitar:
  - `src/lib/rate_control/rate_control.cpp` (PID de *rate* y anti-windup),
  - `src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp` (error de cuaternión y prioridad a la inclinación),
  - `src/lib/control_allocation/control_allocation/ControlAllocationPseudoInverse.cpp` (normalización del asignador).

## 1. Simulador multifrecuencia (40 min)

En `src/proyecto_hilo/sim/multifrecuencia.py`. La física avanza a un paso base fijo (\(\Delta t\) = 0,5 ms con RK4) y cada controlador se ejecuta cuando le toca, con retenedor de orden cero entre ejecuciones:

```python
class Reloj(NamedTuple):
    k: jnp.ndarray          # contador de pasos base (entero)

def toca(reloj, periodo_pasos):
    return (reloj.k % periodo_pasos) == 0

def paso(carry, _):
    x, ctrl, mando, reloj, cola = carry
    # 1) el controlador lee la medida si le toca; si no, conserva su estado y su mando
    nuevo_ctrl, nuevo_mando = controlador(ctrl, medir(x))
    ctrl = jax.tree.map(lambda a, b: jnp.where(toca(reloj, N_RATE), a, b), nuevo_ctrl, ctrl)
    mando_calc = jnp.where(toca(reloj, N_RATE), nuevo_mando, mando)
    # 2) cola de retardo: el mando calculado sale D pasos base después
    cola = jnp.roll(cola, -1, axis=0).at[-1].set(mando_calc)   # longitud D + 1 para D pasos de retardo
    x = rk4_paso(f, x, cola[0], reloj.k * DT, DT)
    return (x, ctrl, mando_calc, Reloj(reloj.k + 1), cola), x
```

`jax.lax.scan` sobre `paso`, con la cola como array de tamaño fijo: para un retardo de \(D\) pasos base hace falta longitud \(D + 1\). Los «1,5 periodos» de la presentación son el retenedor (medio periodo) más una cola de un periodo del lazo (`N_RATE` pasos). Escribe un controlador por frecuencia (*rate*, actitud, velocidad/posición) y encadénalos en el mismo `paso`.

**Tests** (`tests/test_multifrecuencia.py`):
1. Con un controlador que devuelve el número de veces que se ha ejecutado, a 250 Hz y paso base de 0,5 ms, tras 1 s ha corrido exactamente 250 veces.
2. Con retardo de \(D\) pasos, un escalón en el mando aparece en la planta exactamente \(D\) pasos después.

## 2. El lazo de *rate* de PX4 (45 min)

En `src/proyecto_hilo/control/rate_pid.py`, la ley de `RateControl::update` por eje:

\[ \tau_{\text{sp}} = P\,e + I_{\text{acum}} - D\,\dot\omega_f + FF\,\omega_{\text{sp}},\qquad e = \omega_{\text{sp}} - \omega_f \]

con las ganancias ya multiplicadas por `MC_*RATE_K`, y:

- \(\omega_f\): giróscopo con un Butterworth de 2.º orden a 40 Hz (`IMU_GYRO_CUTOFF`), discretizado como `LowPassFilter2p` de PX4 (bilineal). La clase `Biquad` de `figuras_a2.py` te sirve de referencia.
- \(\dot\omega_f\): diferencia del giróscopo sin ese filtro, pasada por un filtro de 1.er orden a 20 Hz (`IMU_DGYRO_CUTOFF`), con \(\alpha = \Delta t/(\tau + \Delta t)\) y \(\tau = 1/(2\pi f_c)\) como el `AlphaFilter` de PX4. Ojo: la descripción del parámetro dice Butterworth de 2.º orden, pero el código usa este filtro de 1.er orden.
- Integrador como en `updateIntegral`: no acumula error que empuje hacia una saturación señalada por el asignador, factor \(\max(0, 1 - (e/400^\circ\!/\mathrm{s})^2)\), límite \(\pm\)0,3 y nada en el suelo.

Valores por defecto de PX4 (`mc_rate_control_params.yaml`): alabeo y cabeceo P 0,15, I 0,2, D 0,003, FF 0, K 1; guiñada P 0,2, I 0,1, D 0.

**Del par normalizado a los motores.** Calcula la pseudoinversa de la matriz de asignación de A1 y normalízala como PX4: las columnas de alabeo y cabeceo se dividen por \(\sqrt{\|\mathbf m\|^2/(n/2)}\), con \(n\) los motores que contribuyen; la de guiñada, por su máximo; la de empuje, por la media de sus valores absolutos. Comprueba que, en el x500, un par de alabeo normalizado de 1 sube dos motores \(1/\sqrt2\) y baja los otros dos lo mismo. La salida de motor \(u\in[0,1]\) se convierte en consigna \(\Omega = 150 + 850u\) rad/s (`SIM_GZ_EC_MIN/MAX` del airframe `4001_gz_x500`), que entra en el motor de primer orden de A1.

**Tests:**
1. Estacionario: con empuje colectivo constante y consigna de *rate* nula, el dron no gira (\(|\omega| < 10^{-6}\) rad/s en 2 s).
2. Escalón de 1 rad/s en alabeo con el lazo a 250 Hz y 1,5 periodos de retardo de muestreo y cómputo. Con la inercia compuesta \(J_x = 0{,}0239\) y un motor simétrico de 12,5 ms: subida al 90 % en 76 ± 6 ms y sobrepaso del 5 ± 2 %. Con el motor asimétrico de A1 (12,5 ms al subir, 25 ms al bajar): unos 65 ms. Explica por qué la asimetría acelera la subida.
3. Anti-windup: forzando la bandera de saturación positiva, el integrador no crece con error positivo y sí decrece con error negativo.

## 3. Respuesta en frecuencia medida (30 min)

Mide el lazo abierto sin abrirlo: inyecta una senoide \(d\) a la entrada de la planta (\(u = c + d\)), registra la salida del controlador \(c\) y la entrada total \(u\), y estima \(L(j\omega) = -C(j\omega)/U(j\omega)\) con la componente de Fourier a la frecuencia inyectada. Barre de 1 a 40 Hz (con `vmap` sobre las frecuencias va en segundos).

**Comprobación:** la cuenta lineal da 6,3 Hz y 71°. Con la planta no lineal de cuatro motores, el script mide 6,7 Hz y 69° si el motor es simétrico, y 5,9 Hz y 64° con el asimétrico. Tus valores deben quedar a ±10 % y ±6° de los del script con el mismo motor. Si no salen, revisa el retardo de muestreo y la normalización del asignador antes que las ganancias.

## 4. Retardo y estabilidad (20 min)

Añade de 0 a 30 ms de retardo extra entre el controlador y los motores (cola del paso 1). Para cada valor mide el sobrepaso del escalón del paso 2.

**Comprobaciones:** con +20 ms, sobrepaso de unos 47 % y oscilación de unos 8 Hz. Busca el retardo con el que la oscilación deja de amortiguarse y compáralo con el margen de retardo calculado, \(\varphi_m/\omega_c \approx 31\) ms.

## 5. Actitud y posición (45 min)

En `control/actitud.py`, la ley de `AttitudeControl::update`:

1. Actitud reducida: giro mínimo de \(\mathbf z = R\mathbf e_3\) a \(\mathbf z_d\) aplicado a la actitud actual.
2. Resto de guiñada escalado por `MC_YAW_WEIGHT` = 0,4.
3. \(\mathbf q_e = \mathbf q^{-1}\otimes\mathbf q_d\) en el hemisferio con \(q_{e,w}\ge0\) y \(\boldsymbol\omega_{\text{sp}} = K\circ 2\mathbf q_{e,v}\), con \(K\) = (4; 4; 2,8) (`MC_ROLL_P`, `MC_PITCH_P`, `MC_YAW_P`), recortada a 220 °/s en alabeo y cabeceo y 200 °/s en guiñada.

En la rama `main`, PX4 no aplica esta ley directamente a la consigna: la filtra antes con un modelo de referencia de 2.º orden (`MC_REF_W_N` = 50 rad/s, prealimentación `MC_REF_FF` = 0). Empieza sin él y añádelo después; anota cuál usas en cada medida.

En `control/posicion.py`, un P de posición que da consigna de velocidad y un PID de velocidad que da aceleración; de la aceleración, el empuje colectivo y la actitud deseada (el eje \(\mathbf z_d\) es la dirección de \(\mathbf a + g\mathbf e_3\), como en la planitud de A1), con un límite de inclinación. Usa las ganancias por defecto de PX4 que encontrarás en `src/modules/mc_pos_control/multicopter_position_control_gain_params.yaml` y los límites de `multicopter_position_control_limits_params.yaml`, y anota las que uses.

**Tests y medidas:**
1. **Hemisferio:** con \(\mathbf q\) y \(-\mathbf q\) la consigna de *rate* es la misma.
2. **Separación de escalas:** con el lazo de *rate* cerrado, mide su ancho de banda efectivo \(\omega_i\) y barre `MC_ROLL_P` de 2 a 30 con escalones pequeños (5° o menos; con \(k\) = 22, un escalón de 10° ya pide 220 °/s y satura). Representa el sobrepaso frente a \(r = \omega_i/k\). La tabla de la presentación supone un interno de 2.º orden con \(\zeta\) = 0,7, así que la comparación es cualitativa: busca a partir de qué \(r\) desaparece el sobrepaso.
3. **Vuelo a un punto:** del estacionario a un punto a 5 m. Registra las cuatro señales de la cascada y comprueba visualmente que cada lazo va varias veces más rápido que el de fuera.

## 6. INDI (30 min, opcional)

Sustituye el PID de *rate* por INDI:

\[ \mathbf u = \mathbf u_{0,f} + G^{-1}\big(K_\omega(\boldsymbol\omega_{\text{sp}} - \boldsymbol\omega_f) - \dot{\boldsymbol\omega}_f\big) \]

donde \(\mathbf u_{0,f}\) es el par que producen las velocidades de motor **medidas** (el simulador las tiene; en el dron real llegan por DShot bidireccional) pasado por el mismo filtro de 20 Hz que \(\dot{\boldsymbol\omega}\). Usa \(K_\omega\) = 45 rad/s, el mismo ancho de banda que el PID.

**Comprobación:** con un par de perturbación de 0,3 N·m en alabeo desde \(t\) = 0,05 s y solo el lazo de *rate*, el ángulo acumulado a 1 s es de unos 11,7° con el PID y de unos 0,4° con INDI. Después, desincroniza los filtros (filtra \(\dot\omega\) a 20 Hz y \(\mathbf u_0\) a 60 Hz) y describe qué pasa.

## 7. Guiado L1 (20 min, opcional)

Con un modelo cinemático (\(\dot x = V\cos\psi\), \(\dot y = V\sin\psi\), \(\dot\psi = a/V\)) a 25 m/s, implementa \(a = 2V^2\sin\eta/L_1\) para seguir una recta desde 30 m de separación.

**Comprobaciones:** con \(L_1\) = 80 m, llega al 5 % de la separación inicial en unos 6,6 s y pide un alabeo máximo de unos 31°. Repite con el Aerosonde de A1 y un lazo de alabeo como el de la presentación (\(k_{p_\phi}\) = 3,0, \(k_{d_\phi}\) = 0,041 s) en lugar del modelo cinemático: ¿cuánto empeora?

## Entregables

- `sim/multifrecuencia.py`, `control/rate_pid.py`, `control/actitud.py`, `control/posicion.py` (y `control/indi.py` si haces el paso 6), con `uv run pytest` en verde.
- `experimentos/a2_lazos.py`, que genera: Bode medido frente a calculado (paso 3), sobrepaso frente a retardo (paso 4) y sobrepaso frente a \(r\) (paso 5).
- Una tabla con tus valores frente a los de la presentación y una línea explicando cada diferencia de más del 10 %.

## Preguntas para cerrar

1. En tu simulador, el lazo de *rate* va a 250 Hz como en el SITL de Gazebo. Una placa real con PX4 lo corre a 800 Hz por defecto en multirrotores, y una con Betaflight a 4–8 kHz. ¿Qué cambia para una política que da CTBR a 100 Hz entrenada en tu simulador? ¿Y para una que da mandos de motor?
2. Con la cola de retardo del paso 1, ¿cuál es el retardo máximo que puede tener una política que sustituya al lazo de actitud del x500 sin bajar de 45° de margen?
3. INDI necesita las velocidades de motor medidas. Si tu dron no tiene telemetría de RPM, ¿qué harías? Piensa en la diferencia entre medir el estado del actuador y estimarlo con un modelo, y en lo que eso implica para el belief.
