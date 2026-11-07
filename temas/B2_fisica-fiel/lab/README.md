# Laboratorio B2 · Motor, arrastre y viento; identificación con datos de vuelo reales

**Duración:** unas 3,5 h (el paso 5 es opcional).
**Requisitos:** tu simulador JAX del x500 (A1, A2, B1) y los CSV de NeuroBEM que descargaste en A1 (paso 5 de su laboratorio). Todo en CPU; la GPU solo ayuda si entrenas el residuo del paso 5 con muchos datos.

## Objetivo

Añadir al simulador del proyecto hilo las capas de realismo de la presentación, cada una con un interruptor, e identificar con datos reales los parámetros que se pueden identificar. Al final debes poder decir, para cada capa, cuánto cambia el vuelo y con qué confianza conoces sus parámetros. Comprueba con tus números lo que calcula la presentación:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Constante de tiempo del motor equivalente del x500 en estacionario | 11,0 ms; bajada sin frenado al 63 % en 53 ms, y 98 ms de 810 a 565 rad/s | 1 |
| Autonomía en estacionario (4S, 5 Ah, OCV lineal de PX4, 5 mΩ por celda) y margen de empuje | 17,5 min; de 1,63 a 1,20 veces el peso | 1 |
| Empuje a \(\Omega\) fija con la BEM: subida a 2 m/s; avance a 4 m/s con disco horizontal e inclinado | 0,881; 1,068 y 1,018 | 2 |
| Dryden a 10 m, turbulencia ligera | \(\sigma_u\) = 1,46 m/s, \(L_u\) = 67 m, \(\sigma_w\) = 0,77 m/s, \(L_w\) = 10 m | 3 |
| Desviación típica en estacionario con esa turbulencia y viento medio de 5 m/s | 4,0 cm (arrastre de Gazebo) y 15,9 cm (SIH) | 3 |
| Desviación relativa de \(\hat k_h\) con datos hasta 1, 2 y 4 m/s (datos sintéticos) | 33 %, 7,5 % y 1,9 % | 4 |

El script de las figuras está en `../figuras/figuras_b2.py`. Los parámetros marcados como SUPUESTOS en él (pala, motor eléctrico, batería) son valores típicos, no medidas de un fabricante: úsalos para reproducir las cifras y cámbialos por los de tu dron cuando los tengas.

## 1. Motor, ESC y batería (50 min)

En `src/proyecto_hilo/dinamica/actuadores.py`:

1. El modelo eléctrico del motor, \(J_r\dot\Omega = (k_t/R)(dV - k_e\Omega) - k_Q\Omega^2\), con un indicador de frenado activo (si no lo hay, la corriente no puede ser negativa). Parámetros del motor equivalente: \(R\) = 0,12 Ω, \(J_r\) = 2,5·10⁻⁵ kg m², \(k_e = k_t\) elegida para que, con 16,8 V y mando 1, el motor llegue a 1000 rad/s con \(k_Q = k_m k_T\).
2. Una versión reducida, de primer orden asimétrico (\(\tau_{\uparrow}\), \(\tau_{\downarrow}\)), para cuando no quieras el modelo eléctrico.
3. La batería: OCV lineal por celda entre 3,6 y 4,2 V, \(R_0\) = 5 mΩ por celda y, opcional, una rama RC. El ESC convierte el mando en tensión media \(dV_{\text{bus}}\).

**Comprobaciones:** linealiza con `jax.jacfwd` y compara el autovalor con \(\tau = J_r/(k_tk_e/R + 2k_Q\Omega_0)\) = 11,0 ms. Escalones de mando 0,55 ↔ 0,80: tiempos al 63 % de 11,0 ms (subida), 11,1 ms (bajada con frenado) y 53 ms (bajada sin frenado). Vuelo en estacionario hasta 3,6 V por celda: 17,5 min y margen de empuje de 1,63 a 1,20.

**Piensa:** ajusta un primer orden asimétrico a las respuestas del modelo eléctrico. ¿Qué \(\tau_\downarrow\) sale sin frenado? ¿Se parece al de Gazebo (25 ms)? ¿Depende del tamaño del escalón?

## 2. Rotor con elemento de pala (40 min)

En `dinamica/rotor_bem.py`, la BEM de la presentación: \(C_T(\mu, \lambda)\) con torsión lineal y el flujo de Glauert resuelto por punto fijo, vectorizado con `jax.vmap` sobre los cuatro rotores y sobre los entornos.

1. Calibra \(\theta_0\) para que, a 770 rad/s y sin viento, cada rotor dé 5,06 N. Comprueba la figura de mérito (0,63 con \(C_{d0}\) = 0,02).
2. Representa \(T/T_h\) a \(\Omega\) fija frente a la velocidad: subida pura, avance con el disco horizontal y avance con el disco inclinado a \(\tan\alpha = dv/g\). Superpón \(1 + k_hv^2/g\) con \(k_h\) = 0,009 m⁻¹.
3. Mete la BEM en el simulador, con la velocidad relativa al aire de cada rotor en sus ejes (incluye \(\boldsymbol\omega\times\mathbf r_i\)).

**Comprobaciones:** 0,881 a 2 m/s de subida; 1,068 (horizontal) y 1,018 (inclinado) a 4 m/s; a 10 m/s, 1,311 y 0,871. Con el punto fijo, comprueba que el residuo \(|\lambda - \lambda_c - C_T/(2\sqrt{\mu^2+\lambda^2})|\) baja de 10⁻¹⁰.

**Piensa:** ¿cuánto cuesta la BEM por paso frente a \(k_T\Omega^2\) en tu simulador vectorizado? ¿Cuántos pasos por segundo pierdes con 4096 entornos (B1, paso 4)?

## 3. Viento y turbulencia de Dryden (40 min)

En `dinamica/viento.py`:

1. Un viento medio (constante o con una deriva lenta) más turbulencia de Dryden a baja altura: las escalas y varianzas de MIL-F-8785C de la presentación, la componente longitudinal y la lateral con la recursión exacta de primer orden, y la vertical con el filtro de segundo orden.
2. Valida el generador con una realización de 600 s: desviación típica y espectro (periodograma promediado por bandas) frente a los analíticos.
3. Estacionario del x500 con el PD de B1 y viento medio de 5 m/s más la ráfaga, con los tres arrastres de la presentación.

**Comprobaciones:** \(\sigma_u\) = 1,46 m/s y \(L_u\) = 67 m a 10 m de altura; el espectro simulado a menos de unas 0,1 décadas del analítico; desviaciones típicas de posición del orden de 4 cm (Gazebo) y 16 cm (SIH). Con varias semillas, ¿cuánto varían esas cifras?

## 4. Identificación con NeuroBEM (50 min)

Con los CSV de NeuroBEM (recuerda lo que viste en A1: `acc` es fuerza específica en ejes cuerpo, `vel` velocidad en ejes cuerpo y `vbat` en décimas de voltio):

1. **Sustentación de traslación:** ajusta \(a_z - (k_T/m)\sum_i\Omega_i^2 = k_h\,(v_x^2 + v_y^2) + b\) por mínimos cuadrados. Da \(\hat k_h\) con su desviación típica (\(\sigma^2(\Phi^\top\Phi)^{-1}\), con \(\sigma\) estimada de los residuos).
2. **Identificabilidad:** repite con subconjuntos limitados a \(|v| <\) 1, 2, 4 y 8 m/s. ¿Se reproduce la tendencia de la figura sintética?
3. **Tensión de batería:** añade \(V_{\text{bat}}\) como regresor (o identifica \(k_T\) en tramos de tensión distinta). ¿Cambia \(k_T\) con la tensión cuando ya usas la velocidad medida de los rotores? ¿Por qué no debería?
4. **Arrastre con inclinación:** compara el ajuste lineal \(a_{xy} = -d\,v_{xy}\) con uno que añada \(c\,v|v|\). ¿Hasta qué velocidad basta el lineal?

**Comprobaciones:** escribe en una tabla cada parámetro con su intervalo y el rango de velocidades de los datos con los que lo identificaste. Si un parámetro tiene una desviación relativa mayor del 30 %, no lo uses como valor fijo: aleatorízalo en ese rango.

## 5. Un residuo aprendido y su extrapolación (opcional, 40 min)

1. Entrena en JAX (Flax u Optax, o una red a mano) un MLP pequeño que prediga el residuo de fuerza de NeuroBEM respecto a tu modelo físico identificado (\(k_T\Omega^2\) más arrastre), con entradas \((\mathbf v^B, \boldsymbol\omega, \Omega_{1..4})\).
2. Entrena solo con muestras por debajo de 5 m/s y evalúa por encima. Haz lo mismo con un modelo puramente aprendido (sin física) y con un polinomio.
3. Representa el RMSE frente a la velocidad para los tres.

**Comprobaciones:** reproduce cualitativamente lo de la tabla II de NeuroBEM: el residuo sobre la física extrapola mejor que el polinomio. No esperes sus cifras exactas: ellos usan BEM, una red temporal y todo el dataset.

## Entregables

- `dinamica/actuadores.py`, `rotor_bem.py` y `viento.py`, con tests: el motor eléctrico linealizado da la \(\tau\) analítica; la batería conserva la carga (\(\int I\,dt\) = capacidad gastada); el generador de Dryden da la varianza teórica en una realización larga; con viento nulo y \(\mu = \lambda_c = 0\), la BEM da el empuje de calibración.
- `experimentos/b2_capas.py`, que genere: respuestas del motor, descarga de la batería, \(T/T_h\) frente a la velocidad, espectro de Dryden y la tabla de identificación con intervalos.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Qué capas de realismo cambian de verdad el comportamiento de tu controlador de A2 o de tu MPC de A3? Ordénalas por efecto y por coste de cómputo.
2. Si tu política solo pudiera observar la velocidad de los rotores y la IMU, ¿podría inferir el viento? Razónalo con el modelo de arrastre y con el de la BEM.
3. Con lo que identificaste en el paso 4, ¿qué rangos de aleatorización usarías para entrenar una política robusta (B7)? ¿Cuáles salen de los datos y cuáles son supuestos?
