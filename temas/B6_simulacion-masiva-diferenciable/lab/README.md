# Laboratorio B6 · Miles de entornos en paralelo y gradientes a través del simulador

**Duración:** unas 4 h.
**Requisitos:** tu simulador JAX del x500 con las capas de B2–B4. Una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]"`, o la variante de CUDA que corresponda a tu driver) o una GPU en la nube; todo funciona también en CPU (`JAX_PLATFORMS=cpu uv run --with jax`), más despacio a partir de unos cientos de entornos.

## Objetivo

Convertir tu simulador en uno masivo (miles de entornos en un solo proceso) y diferenciable, medir su rendimiento en tu máquina, y comprobar cuándo un gradiente a través del simulador ayuda y cuándo engaña. Comprueba con tus números lo que calcula la presentación (medido con `figuras/figuras_b6.py` en un i7-14700K y una RTX 4060 Ti; tus cifras de rendimiento serán otras):

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Pasos de entorno por segundo, 1 entorno en Python con numpy | unos 3600–3900 | 1 |
| Pico en CPU (28 hilos) | unos 5,5 millones (256 y 65 536 entornos) | 1 |
| Pico en GPU | unos 72 millones con 65 536 entornos; baja con 262 144 y con un millón | 1 |
| Coste de un gradiente de primer orden frente a solo simular (H = 1000) | ×4,3 en CPU, ×3,4 en GPU | 3 |
| Orden cero con 32 simulaciones | ×48 en CPU, ×2,5 en GPU | 3 |
| Muro duro: gradiente verdadero y de primer orden a 3,4 m | −59,8 frente a 2,00 | 4 |
| Muro suavizado (sigmoide de 1 cm): desviación por muestra del primer orden a 3,4 m | 337 | 4 |

El script de las figuras está en `../figuras/figuras_b6.py` (unos minutos; imprime el dispositivo usado). Las cuentas a mano de la presentación (estimador de orden cero, BPTT de tres pasos, el muro en 1-D, la varianza de la sigmoide) están en `../figuras/ejemplos_b6.py`, que corre en CPU en segundos.

## 1. Tu simulador, vectorizado (50 min)

1. Escribe el paso de tu entorno para **un** dron, con estado de forma fija y sin ramas de Python que dependan del estado (usa `jnp.where`/`lax.select`, y `lax.cond` solo con predicados escalares, como en B4).
2. Reinicio automático: cuando un entorno termina, su estado se sustituye por uno inicial con `jnp.where`, sin salir de `jit`.
3. Vectoriza con `vmap` y despliega con `lax.scan`. Mide pasos de entorno por segundo para 1, 16, 256, 4096, 16 384 y 65 536 entornos en CPU y, si puedes, en GPU. Compara con tu bucle en Python y con las cifras de la presentación.
4. Repite con la cámara de B3b o un sensor caro activado a 30 Hz. ¿Cuánto cae el rendimiento?

**Piensa:** ¿por qué la GPU es más lenta que la CPU con un solo entorno? ¿Por qué empeora más allá de cierto número de entornos?

## 2. Lo que se rompe a escala (40 min)

1. **Aleatoriedad:** una clave por entorno (`jax.random.split`), sin reutilizar claves entre pasos. Test: con la misma semilla, dos ejecuciones dan el mismo resultado en el mismo dispositivo.
2. **NaN:** inyecta un NaN en un entorno y comprueba que tu reinicio lo aísla y que las estadísticas del lote (media del retorno) no se contaminan.
3. **Reinicios sincronizados:** empieza todos los entornos a la vez con episodios de 5 s y mira la distribución del tiempo dentro del episodio en cada lote. Desincronízalos (duración inicial aleatoria) y compara.
4. **Aleatorización por entorno:** masa, arrastre, constante del motor (B2), ruido de la IMU (B3a) y retardos (B4) sorteados por entorno al reiniciar.
5. **Precisión:** despliega 10 minutos simulados en float32 y en float64 y compara la norma del cuaternión y la deriva de la energía.

## 3. Gradientes de primer orden y de orden cero (45 min)

Con el dron plano de `figuras_b6.py` (o con el tuyo):

1. Calcula el gradiente del coste medio respecto a las ganancias con `jax.grad` a través de `lax.scan` y con un estimador de orden cero antitético (16 perturbaciones, 32 simulaciones).
2. Reproduce la tabla de varianzas para horizontes de 10 a 3000 pasos y los dos juegos de ganancias.
3. Mide el tiempo de cada estimador en tu CPU y en tu GPU.
4. Optimiza las ganancias con los dos estimadores (Adam, mismo presupuesto de tiempo de reloj) y compara las curvas de coste frente al tiempo.

**Piensa:** ¿qué estimador elegirías en tu portátil y cuál en una GPU? ¿Por qué cambia la respuesta?

## 4. El muro (30 min)

1. Reproduce el experimento del muro: coste esperado frente a la altura de consigna, gradiente verdadero (diferencias finitas sobre muchas muestras), de primer orden y de orden cero.
2. Haz un descenso por gradiente desde 4,6 m con cada estimador. ¿Dónde acaba cada uno?
3. Suaviza el muro con una sigmoide de anchura 1 cm, 10 cm y 50 cm. ¿Cómo cambian el sesgo y la varianza del primer orden? ¿Y el mínimo del problema suavizado respecto al real?

## 5. Horizonte y explosión de gradientes (25 min)

1. Con las ganancias poco amortiguadas, mide la norma del gradiente de una muestra frente al horizonte y su cola (percentil 99).
2. Haz las ganancias inestables (amortiguamiento negativo en x) con la saturación de inclinación activa: la trayectoria queda acotada, pero ¿y el gradiente?
3. Trunca la retropropagación a ventanas de 50, 100 y 300 pasos (gradiente a través de cada ventana, sin pasar por las fronteras). ¿Qué ganas en varianza y qué pierdes en sesgo? En A6a verás cómo SHAC tapa ese sesgo con un crítico.

## 6. Identificación por gradiente (30 min)

1. Genera una trayectoria con tu simulador completo (B2) con un arrastre y una constante de motor «verdaderos».
2. Desde valores iniciales erróneos, recupera esos dos parámetros minimizando el error de la trayectoria simulada frente a la «medida» con `jax.grad` a través de la simulación (horizonte corto, multiple shooting si diverge).
3. Repite con datos reales del dataset de B2 si tu modelo lo permite. ¿Es mejor que los mínimos cuadrados de B2? ¿Cuándo lo sería?

## Entregables

- `sim/vectorizado.py` (paso, reinicio y despliegue vectorizados) con tests de determinismo, aislamiento de NaN y forma fija.
- `experimentos/b6_rendimiento.py` (rendimiento frente a entornos en CPU y GPU), `experimentos/b6_gradientes.py` (varianzas, tiempos, muro, horizonte) y `experimentos/b6_identificacion.py`.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. Para entrenar tu política de tesis, ¿cuántos pasos de entorno por segundo necesitas para que un entrenamiento de \(10^9\) pasos quepa en una noche? ¿Te llega con tu GPU?
2. Tu tarea de tesis tiene contactos (aterrizar, chocar) y observaciones parciales. ¿Dónde usarías gradientes a través del simulador y dónde no?
3. Dreamer aprende el actor retropropagando a través de un modelo aprendido. ¿Qué de lo que has visto aquí (horizonte, discontinuidades, varianza) le afecta igual y qué no?
