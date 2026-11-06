# Laboratorio A1 · El cuadricóptero del proyecto hilo, verificado contra la física y contra datos reales

**Duración estimada:** 3,5 h · **Tema:** A1 · **Hardware:** cualquier PC; todo funciona en CPU y la GPU solo acelera el paso 3. **Datos:** NeuroBEM (público; basta con descargar dos o tres CSV, ver el paso 5).

## Objetivo

Construir el módulo de dinámica del proyecto hilo (cuadricóptero de 17 estados y avión de ala fija) con tests contra soluciones conocidas, y verificar numéricamente estos resultados de la presentación:

| Resultado de la presentación | Valor esperado |
|---|---|
| Velocidad de motor del x500 en estacionario | ≈ 770 rad/s |
| Tasa de crecimiento del giro alrededor del eje intermedio (inercia de NeuroBEM, 20 rad/s) | λ ≈ 5,65 s⁻¹ |
| Velocidad a la que satura el empuje en la lemniscata de 5 m (límite 50,5 m/s²) | ≈ 15 m/s |
| Error de actitud por ignorar un arrastre de 0,465 s⁻¹ a 4 m/s | ≈ 10,8° |
| Periodo de la fugoide del Aerosonde a 25 m/s (modelo completo y Lanchester) | ≈ 13,1 s y 11,3 s |
| Parámetros identificables con IMU y RPM | solo cocientes (p. ej. k_T/m) |

El script que generó las figuras de la presentación está en `../figuras/figuras_a1.py`. Intenta no mirarlo hasta terminar cada paso: úsalo para comparar.

## 0. Convenios y preparación (10 min)

Convenio interno del proyecto hilo, fijado en A1: **mundo ENU (z hacia arriba), cuerpo FLU, cuaterniones de Hamilton en orden w, x, y, z** (como `jaxlie`). Todo lo que venga de PX4 (NED/FRD) se convierte en la frontera.

```bash
uv add remotezip   # solo para el paso 5
```

Reutiliza `dinamica/so3.py` del laboratorio de M1 y `sim/integradores.py` (RK4) del laboratorio 0. Trabaja en `float64` para los tests (`jax.config.update("jax_enable_x64", True)`) y en `float32` para las medidas de rendimiento.

## 1. Modelo de 17 estados (60 min)

En `src/proyecto_hilo/dinamica/cuadricoptero.py`:

- Un `NamedTuple` de parámetros (es un pytree, así que `vmap` y `jit` lo aceptan): `m`, `J` (3,), `kT`, `kq_kt` (k_Q/k_T, en m), `rotores` (4, 3) en FLU, `sentidos` (4,) con +1 si el rotor gira en sentido antihorario visto desde arriba, `tau_sub`, `tau_baj`, `omega_max`, `D` (3,).
- `x500()`: parámetros del modelo de Gazebo de PX4 (masa 2,0 kg más 4 rotores de 0,0161 kg; J = diag(0,02167; 0,02167; 0,04); k_T = 8,54858·10⁻⁶; momentConstant = 0,016; τ = 12,5 ms al subir y 25 ms al bajar; Ω_max = 1000 rad/s). Los rotores, con la numeración de PX4 (`CA_ROTOR0..3`), están en FRD en (0,174; 0,174), (−0,174; −0,174), (0,174; −0,174) y (−0,174; 0,174), y los dos primeros giran en sentido antihorario. **Conviértelos a FLU** (cambia el signo de y).
- La matriz de asignación en FLU. Cada rotor empuja \(T_i\mathbf e_3\) desde \(\mathbf r_i\); el par es \(\mathbf r_i\times T_i\mathbf e_3 = (y_iT_i,\ -x_iT_i,\ 0)\), y la reacción de un rotor antihorario empuja al cuerpo en sentido horario, que con z hacia arriba es negativo:

```python
def matriz_asignacion(p):
    r = p.rotores                         # (4, 3) en FLU
    kappa = -p.sentidos * p.kq_kt         # par de guiñada por newton de empuje
    return jnp.stack([jnp.ones(4), r[:, 1], -r[:, 0], kappa])   # filas: T, tau_x, tau_y, tau_z
```

- La dinámica `f(x, u, p)` con `x = [p(3), v(3), q(4), w(3), Om(4)]` y `u` = consignas de velocidad de motor, recortadas a `[0, omega_max]`:

\[ \dot{\mathbf v} = -g\mathbf e_3 + \tfrac{T}{m}R\mathbf e_3 - RDR^\top\mathbf v,\quad \dot{\mathbf q} = \tfrac12\mathbf q\otimes(0,\boldsymbol\omega),\quad \dot{\boldsymbol\omega} = J^{-1}(\boldsymbol\tau - \boldsymbol\omega\times J\boldsymbol\omega),\quad \dot\Omega_i = \frac{\Omega^{\text{cmd}}_i - \Omega_i}{\tau_i} \]

  con `tau_i = jnp.where(Om_cmd > Om, tau_sub, tau_baj)`.
- Un paso RK4 que renormaliza el cuaternión al final (o, mejor, que integra la actitud con `Exp` de M1).

Tests en `tests/test_cuadricoptero.py`:

1. **Estacionario exacto:** con \(\Omega_i = \sqrt{mg/(4k_T)}\) (comprueba que sale ≈ 770 rad/s) y el dron nivelado, la posición se mueve menos de 1 µm en 5 s.
2. **Caída libre:** con los motores parados, \(z(t) = z_0 - \tfrac12 g t^2\).
3. **Signos de la asignación:** pide un par de alabeo positivo (en FLU, ala derecha abajo): los dos rotores de la derecha (y < 0) deben bajar y los de la izquierda subir. Comprueba también \(M M^{-1} = I\).
4. **Motor:** ante un escalón, \(\Omega\) alcanza el 63,2 % del salto en \(\tau_\text{sub}\) al subir y en \(\tau_\text{baj}\) al bajar.

## 2. Euler sin par y el eje intermedio (20 min)

Con la inercia del dron de NeuroBEM, \(J = \mathrm{diag}(0{,}0025;\ 0{,}0021;\ 0{,}0043)\ \mathrm{kg\,m^2}\), integra solo \(J\dot{\boldsymbol\omega} = -\boldsymbol\omega\times J\boldsymbol\omega\) desde \(\boldsymbol\omega_0 = (20;\ 0{,}02;\ 0{,}02)\) rad/s durante 4 s con Δt = 0,1 ms.

1. Test: energía y \(\|J\boldsymbol\omega\|\) se conservan con error relativo < 10⁻⁹.
2. Ajusta una recta a \(\log|\omega_y(t)|\) entre 0,2 y 0,8 s: la pendiente debe estar a menos de un 2 % de \(\lambda = 5{,}65\ \mathrm{s^{-1}}\).
3. Repite girando alrededor de z (eje mayor) y de y (eje menor): ¿crece la perturbación?
4. Reproduce la figura de la presentación.

## 3. Miles de drones distintos con `vmap` (30 min)

Muestrea 4096 juegos de parámetros con el espíritu de la aleatorización de RAPTOR (lee su apéndice S3 para los detalles exactos; aquí se simplifica):

- tamaño \(s\sim U(0,1)\) llevado a masa con su cubo, \(m = (m_\min^{1/3} + s\,(m_\max^{1/3} - m_\min^{1/3}))^3\), con \(m\in[0{,}03;\ 3]\) kg;
- empuje/peso \(\sim U(1{,}5;\ 5)\); brazo \(l = m^{1/3}/7{,}9\) (relación masa-tamaño del Crazyflie) con un ±10 % de ruido;
- aceleración angular máxima de alabeo \(\alpha_\max\sim U(40;\ 1200)\ \mathrm{rad/s^2}\), de donde \(J_{xx} = J_{yy} = 2T_\max\,(l/\sqrt2)/\alpha_\max\), y \(J_{zz} = 1{,}832\,(J_{xx}+J_{yy})/2\);
- \(k_Q/k_T\sim U(0{,}005;\ 0{,}05)\) m, \(\tau_\text{sub}\sim U(0{,}03;\ 0{,}1)\) s y \(\tau_\text{baj}\sim U(0{,}03;\ 0{,}3)\) s.

Pasos:

1. Simula los 4096 en estacionario, cada uno a su \(\Omega_h\), durante 2 s con Δt = 1 ms (`jax.vmap` sobre los parámetros, `lax.scan` en el tiempo). Test: ninguno se mueve más de 1 mm.
2. Mide pasos por segundo en CPU y GPU (recuerda calentar `jit` y usar `block_until_ready`) y compáralo con la tabla del laboratorio 0. ¿Cuánto cuesta un cuadricóptero frente a un péndulo?
3. **Paso de integración y motor:** con \(\tau = 30\) ms, integra un escalón de motor con Δt = 1, 5, 10 y 20 ms y compara con la solución exacta \(1 - e^{-t/\tau}\). ¿A partir de qué Δt el error supera el 1 %? Comprueba la regla de la presentación (Δt ≲ τ/5) y luego integra el motor de forma exacta: \(\Omega_{k+1} = \Omega^\text{cmd} + (\Omega_k - \Omega^\text{cmd})\,e^{-\Delta t/\tau}\).

## 4. Planitud (40 min)

En `src/proyecto_hilo/dinamica/planitud.py`:

1. Lemniscata de Gerono horizontal, \(x = A\cos\theta\), \(y = A\sin\theta\cos\theta\), \(\theta = Wt\), A = 5 m, rumbo fijo. Escribe a mano (o con `jax.jacfwd` encadenado) posición, velocidad, aceleración, jerk y snap.
2. Implementa el mapa plano de la presentación: \(T\), \(R\) y \(\boldsymbol\omega\) a partir de aceleración, jerk y rumbo.
3. **Test de consistencia:** integra solo la cinemática (\(\dot{\mathbf p} = \mathbf v\), \(\dot{\mathbf v} = -g\mathbf e_3 + \tfrac Tm R\mathbf e_3\), \(\dot R = R[\boldsymbol\omega]_\times\)) alimentándola con \(T(t)\) y \(\boldsymbol\omega(t)\) del mapa plano, desde el estado inicial de la referencia. Tras una vuelta a 5 m/s de velocidad máxima, el error de posición debe ser menor de 1 cm con Δt = 1 ms.
4. Barre la velocidad máxima de 1 a 20 m/s y reproduce la figura de la presentación: empuje máximo por unidad de masa frente al límite de 50,5 m/s² (debe saturar hacia 15 m/s) y, con arrastre isótropo \(d = 0{,}465\ \mathrm{s^{-1}}\), el error de actitud si se ignora el arrastre (≈ 10,8° a 4 m/s).
5. Añade el snap: \(\dot{\boldsymbol\omega}\), los pares y, con \(M^{-1}\) del x500, los empujes de cada motor a 5 m/s. ¿Alguno se sale de [0; 8,55] N?

## 5. Datos reales: NeuroBEM (50 min)

El dataset de NeuroBEM (Bauersfeld et al., RSS 2021) tiene 1 h 15 min de vuelos de un dron de carreras con captura de movimiento, velocidades de motor y tensión de batería. El zip de datos procesados ocupa 627 MB, pero el servidor admite descargas parciales, así que basta con extraer algunos CSV:

```python
from remotezip import RemoteZip
URL = "https://download.ifi.uzh.ch/rpg/NeuroBEM/processed_data.zip"
with RemoteZip(URL) as z:
    nombres = [n for n in z.namelist() if n.endswith(".csv")]
    for n in nombres:
        if "2021-02-03-17-37-27" in n or "2021-02-05-14-19-34" in n:   # círculo a 3 m/s y oscilación vertical
            z.extract(n, "data/neurobem")
```

Lee antes el [Readme](https://rpg.ifi.uzh.ch/neuro_bem/Readme.html) y `Flights.txt` (qué maniobra es cada vuelo). Los datos no se suben al repositorio (`data/` está en `.gitignore`).

1. **Qué miden las columnas.** El Readme habla de «acceleration body» y «velocity body». Compruébalo tú: deriva la posición, rótala a ejes cuerpo con el cuaternión y compara. En nuestras pruebas, `vel` es la velocidad en ejes cuerpo y `acc` se comporta como **fuerza específica en ejes cuerpo** (lo que mediría un acelerómetro ideal: \(R^\top(\dot{\mathbf v} + g\mathbf e_3)\)), no como aceleración cinemática. Mundo con z hacia arriba y cuerpo FLU. Ojo también con `vbat`: el Readme dice que va en voltios, pero en los archivos que probamos va de 149 a 165, que solo cuadra en décimas de voltio (14,9–16,5 V, propio de una batería 4S; la configuración de la batería es interpretación nuestra).
2. **Empuje.** Ajusta por mínimos cuadrados \(a_z = (k_T/m)\sum_i\Omega_i^2 + b\) con varios vuelos. Con la masa del Readme (0,772 kg; el artículo dice 752 g, anota la discrepancia) obtén \(k_T\). ¿Es pequeño el término independiente \(b\)? ¿Mejora el ajuste si añades la tensión de la batería o la velocidad en el plano?
3. **Arrastre de rotor.** Ajusta \(a_x = -d_x v_x^B\) y \(a_y = -d_y v_y^B\). Compara con los 0,39–0,54 s⁻¹ de Faessler et al. (otro dron de la misma clase). Representa el residuo frente a la velocidad: ¿sigue siendo lineal hasta las velocidades más altas del dataset?
4. **Identificabilidad.** Escribe la predicción del acelerómetro como función de \((m, k_T)\) y calcula con `jax.jacfwd` la jacobiana de todas las medidas de un vuelo respecto a esos dos parámetros. Comprueba que \(J^\top J\) (la información de Fisher con ruido gaussiano) tiene rango 1. ¿En qué dirección del plano \((m, k_T)\) está su vector nulo?
5. **Numeración.** En el Readme, los motores van detrás-derecha, delante-derecha, detrás-izquierda y delante-izquierda (orden de Betaflight). Escribe la permutación a la numeración de PX4 y úsala para comprobar con los datos el signo del par de guiñada: \(\dot\omega_z\) frente a \(\sum_i s_i\Omega_i^2\).

## 6. Ala fija: el Aerosonde (40 min)

En `src/proyecto_hilo/dinamica/ala_fija.py`, implementa el modelo de 9 estados \((u, v, w, \phi, \theta, \psi, p, q, r)\) de Beard y McLain con los coeficientes de [`aerosonde_parameters.py`](https://github.com/byu-magicc/mavsim_public/blob/main/mavsim_python/parameters/aerosonde_parameters.py): sustentación con mezcla de pérdida, resistencia \(C_D = C_{D_p} + (C_{L_0} + C_{L_\alpha}\alpha)^2/(\pi e\,AR)\), fuerzas y momentos laterales, y el modelo de hélice y motor del mismo fichero.

1. **Equilibrio** a \(V_a = 25\) m/s en vuelo recto y nivelado: resuelve con Newton (jacobiana con `jax.jacfwd`) las seis ecuaciones \(\dot u = \dot v = \dot w = \dot p = \dot q = \dot r = 0\) en \((\alpha, \delta_e, \delta_t, \delta_a, \delta_r, \phi)\). Con nuestra implementación sale \(\alpha \approx 2{,}85°\), \(\delta_e \approx -7{,}1°\), \(\delta_t \approx 0{,}77\). Si no te sale, revisa los signos de \(f_x\), \(f_z\) y el modelo de resistencia.
2. **Linealiza** con `jax.jacfwd`, separa los bloques longitudinal \((u, w, q, \theta)\) y lateral \((v, p, r, \phi)\) y calcula los autovalores. Compara con la tabla de la presentación (periodo corto 0,64 s, fugoide 13,1 s, convergencia en alabeo 45 ms, balanceo holandés 1,35 s, espiral inestable).
3. Repite a 20 y 35 m/s. ¿Escala el periodo de la fugoide con \(V\), como predice Lanchester? ¿Y el periodo corto?
4. Integra el modelo **no lineal** desde el equilibrio con \(\Delta u_0 = +2\) m/s y compara con la respuesta lineal de la presentación.

## Entregables

- `dinamica/cuadricoptero.py`, `dinamica/planitud.py` y `dinamica/ala_fija.py` con `uv run pytest` en verde.
- Un cuaderno o script con: la figura del eje intermedio, la tabla de pasos por segundo con 4096 drones, la figura de planitud, los ajustes de \(k_T/m\) y \(D\) con NeuroBEM y la tabla de modos del Aerosonde.
- Una nota breve (`experimentos/a1_identificabilidad.md`) con el rango de la información de Fisher del paso 5.4 y su interpretación.

## Preguntas para cerrar

1. Tu simulador del paso 3 tiene drones con \(\tau_\text{baj}\) de hasta 300 ms. ¿Qué consecuencia tiene para un controlador de rate que manda frenar un motor? ¿Por qué RAPTOR aleatoriza más la bajada que la subida?
2. Con los datos de NeuroBEM, ¿qué variables tendría que estimar un belief sobre la dinámica de ese dron, y cuáles bastaría con fijarlas?
3. Si tu política del tema A6 da comandos de motor directos, ¿qué partes de este laboratorio tendrían que estar en el simulador con el que la entrenas, y cuáles podrías dejar fuera?
