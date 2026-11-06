# Laboratorio 0 · Banco de trabajo del proyecto hilo

**Duración estimada:** 3 h · **Tema:** M0 · **Hito:** arranque del proyecto hilo

## Objetivo

Dejar listo el repositorio en el que crecerá el proyecto hilo durante todo el curso, con JAX en GPU, tests desde el primer día y dos medidas de referencia:

1. Cuántos pasos de simulación por segundo da tu GPU con miles de entornos en paralelo.
2. Cuánto retardo aguanta un lazo de control antes de volverse inestable, comparado con la teoría de M0 (\(\tau_{\max} = \mathrm{PM}/\omega_c\)).

## 1. Proyecto y entorno (30 min)

```bash
uv init --package proyecto-hilo
cd proyecto-hilo
uv add "jax[cuda12]" numpy matplotlib
uv add --dev pytest ruff
```

Si tu driver admite CUDA 13 y la documentación de JAX de tu versión lo recomienda, usa el extra `cuda13`. Comprueba que JAX ve la GPU:

```bash
uv run python -c "import jax; print(jax.devices())"
```

Debe aparecer un dispositivo `CudaDevice`. Si solo sale `CpuDevice`, revisa la versión del driver (`nvidia-smi`) antes de seguir.

## 2. Estructura (15 min)

```
proyecto-hilo/
├── src/proyecto_hilo/
│   ├── dinamica/      # A1: modelos de cuerpo rígido, rotores…
│   ├── control/       # A2, A3: PID, MPC…
│   ├── estimacion/    # A4: filtros
│   ├── sim/           # integradores, entornos vectorizados, latencias
│   └── utils/
├── experimentos/      # scripts que generan figuras y tablas
└── tests/
```

Regla para todo el curso: **cada módulo nuevo entra con al menos un test** que compare contra una solución conocida (analítica, conservación de energía, simetría…).

## 3. Integrador RK4 genérico (45 min)

Escribe en `sim/integradores.py` un paso de Runge-Kutta 4 que sirva para cualquier dinámica `f(x, u, t) -> dx/dt` sobre arrays de JAX:

```python
import jax.numpy as jnp

def rk4_paso(f, x, u, t, dt):
    k1 = f(x, u, t)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, u, t + 0.5 * dt)
    k4 = f(x + dt * k3, u, t + dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
```

Pruébalo con un péndulo simple, \(\ddot\theta = -(g/L)\sin\theta\):

- **Test de exactitud:** con \(\theta_0 = 0{,}01\) rad, compara con la solución de ángulo pequeño \(\theta(t)=\theta_0\cos(\sqrt{g/L}\,t)\) durante 10 s.
- **Test de orden:** integra con \(\Delta t\) y con \(\Delta t/2\) contra una referencia muy fina. El cociente de errores debe estar cerca de \(2^4 = 16\). Si sale cerca de 2 o de 4, hay un error en el integrador.

Usa `jax.lax.scan` para el bucle temporal: es la forma de que todo quede compilado con `jit`.

## 4. Vectorización y rendimiento (45 min)

Con `jax.vmap` sobre el estado inicial, simula N péndulos en paralelo durante 1000 pasos y mide pasos por segundo (N × 1000 / tiempo):

| N entornos | Pasos/s (GPU) | Pasos/s (CPU) |
|---|---|---|
| 1 | | |
| 1 024 | | |
| 65 536 | | |

Detalles que importan:

- Llama una vez a la función antes de medir: la primera llamada incluye la compilación.
- Usa `.block_until_ready()` antes de parar el cronómetro: JAX es asíncrono.
- Usa `float32`, que es lo que usarás para entrenar.

Guarda la tabla en `experimentos/rendimiento.md`. Es tu referencia para comparar con Isaac Lab, MJX o Aerial Gym en B1 y B6.

## 5. Estabilidad frente a retardo (45 min)

El experimento que conecta con la teoría de M0. Planta integradora controlada por un proporcional con retardo de \(d\) pasos:

\[ x_{k+1} = x_k + \Delta t\, u_k, \qquad u_k = -K\, x_{k-d} \]

En tiempo continuo es el lazo \(L(s) = K e^{-s\tau}/s\) con \(\tau = d\,\Delta t\). Su cruce está en \(\omega_c = K\) y su margen de fase sin retardo es de 90°. Por tanto, la teoría predice inestabilidad cuando

\[ K\tau \ge \frac{\pi}{2} \quad\Longleftrightarrow\quad d^* \approx \frac{\pi}{2K\Delta t}. \]

Con \(\Delta t = 1\) ms y \(K = 10\ \mathrm{s^{-1}}\), la predicción es \(d^* \approx 157\) pasos (157 ms).

1. Implementa el retardo con un buffer circular dentro de `lax.scan` (te servirá en B4).
2. Barre \(d\) de 0 a 250 pasos usando `vmap` sobre \(d\) (o sobre una máscara) y mide si \(|x|\) crece o decrece al final.
3. Grafica el umbral experimental frente a la predicción para \(K \in \{2, 5, 10, 20\}\).
4. Explica en dos frases por qué el umbral discreto se separa un poco de la fórmula continua cuando \(K\Delta t\) deja de ser pequeño.

## Entregables

- Repositorio con `uv run pytest` en verde (exactitud y orden de RK4, y un test del buffer de retardo).
- `experimentos/rendimiento.md` con la tabla de pasos por segundo.
- `experimentos/retardo.png`: umbral de estabilidad experimental frente al teórico.

## Preguntas para cerrar

1. ¿Cuántos pasos por segundo necesitarías para reproducir las 10⁸ interacciones de Swift en 50 minutos? ¿Las alcanzas con un péndulo? ¿Y con un cuadricóptero con cámara?
2. Si un lazo de actitud real cruza a 10 Hz, ¿cuántos milisegundos de retardo admite con 45° de margen? ¿Qué te dice eso sobre poner ahí un modelo que tarde 50 ms?
