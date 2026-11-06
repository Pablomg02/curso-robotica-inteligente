# Laboratorio M1 · SO(3) desde cero, un ELBO comprobable y una decisión informada

**Duración estimada:** 3 h · **Tema:** M1 · **Hardware:** cualquier PC; todo funciona en CPU y la GPU solo acelera los barridos.

## Objetivo

Construir el módulo de rotaciones del proyecto hilo con tests y verificar numéricamente tres resultados de la presentación: la integración en el grupo, la cota ELBO ≤ log p(x) y la diferencia entre KL directa e inversa. Termina con una decisión de percepción activa basada en ganancia de información.

```bash
uv add jaxlie   # solo para comparar en los tests
```

## 1. SO(3) en JAX (45 min)

En `src/proyecto_hilo/dinamica/so3.py`, implementa sin usar `jaxlie`:

- `hat(w)` y `vee(W)`.
- `exp(phi)`: Rodrigues, con desarrollo de Taylor cuando `θ < 1e-4` (usa `jnp.where` con valores seguros en ambas ramas para que el gradiente no dé `NaN`).
- `log(R)`: con el caso `θ → 0` (Taylor) y el caso `θ → π` (eje a partir de `R + I`).
- `oplus(R, tau) = R @ exp(tau)` y `ominus(R2, R1) = log(R1.T @ R2)`.
- `jr(phi)`: jacobiano derecho.

Tests mínimos en `tests/test_so3.py`:

1. `exp(log(R)) ≈ R` para 1000 rotaciones aleatorias (genera `R` con `exp` de vectores de norma hasta π).
2. `log(exp(phi)) ≈ phi` para `‖phi‖ < π`.
3. `exp(phi + dphi) ≈ exp(phi) @ exp(jr(phi) @ dphi)` para `dphi` pequeño (error de segundo orden).
4. Coincidencia con `jaxlie.SO3.exp(phi).as_matrix()`.
5. Gradientes finitos de `exp` y `log` en `θ = 0`, con `jax.grad`.

## 2. Tres formas de integrar la actitud (40 min)

Simula un **movimiento cónico** (el eje de giro precesa) durante 20 s con `Δt = 1 ms` y velocidad angular en cuerpo conocida. Integra:

1. **Ángulos de Euler** con \(\dot{\boldsymbol\eta} = E(\boldsymbol\eta)\boldsymbol\omega\).
2. **Primer orden**: \(R_{k+1} = R_k + R_k[\boldsymbol\omega]_\times\Delta t\), sin reortonormalizar.
3. **Exponencial**: \(R_{k+1} = R_k\,\mathrm{Exp}(\boldsymbol\omega\Delta t)\).

Grafica el error de actitud frente a una referencia fina (con `ominus`, en grados) y \(\|R^\top R - I\|_F\). Después lleva el cabeceo a 90° y observa qué le pasa a la integración con Euler.

## 3. El plátano de incertidumbre (20 min)

Reproduce la figura de la presentación: robot en el origen, avance de 5 m (σ = 0,1 m) y rumbo con σ = 0,35 rad.

- Nube de 5000 muestras.
- Elipse de 2σ obtenida linealizando en coordenadas cartesianas.
- La misma incertidumbre representada en el tangente de SE(2): muestrea \(\boldsymbol\varepsilon\sim\mathcal N(0,\Sigma)\) y aplica `Exp`. ¿Qué representación ajusta mejor la nube?

## 4. Un ELBO que se puede comprobar (40 min)

Usa un modelo lineal gaussiano, el único caso en que todo es exacto:

\[ z\sim\mathcal N(0, I_2),\qquad x = Wz + b + \epsilon,\ \ \epsilon\sim\mathcal N(0, \sigma^2 I_5) \]

Aquí \(\log p(x) = \log\mathcal N(x;\,b,\,WW^\top + \sigma^2 I)\) y el posterior \(p(z\mid x)\) también es gaussiano y conocido.

1. Genera 2000 datos con \(W\), \(b\) y \(\sigma\) fijos.
2. Define un codificador amortizado lineal \(q_\phi(z\mid x) = \mathcal N(Ax + c,\ \mathrm{diag}(e^{s}))\) y estima el ELBO con reparametrización (8 muestras por dato).
3. Optimiza \(\phi\) con Adam, con el modelo fijo.
4. Grafica a lo largo del entrenamiento el ELBO medio y el \(\log p(x)\) exacto medio. Comprueba que el ELBO nunca lo supera y que la brecha baja.
5. Explica por qué la brecha no llega exactamente a cero: pista, el posterior verdadero tiene covarianza llena y tu \(q\) es diagonal.

## 5. KL directa frente a inversa (25 min)

Con \(p = \tfrac12\mathcal N(-2, 0{,}6^2) + \tfrac12\mathcal N(2, 0{,}6^2)\) y \(q = \mathcal N(\mu, \sigma^2)\):

- **KL directa** \(\mathrm{KL}(p\|q)\): minimiza \(-\mathbb E_p[\log q]\) con muestras de \(p\). Comprueba que converge a \(\mu \approx 0\), \(\sigma \approx 2{,}09\).
- **KL inversa** \(\mathrm{KL}(q\|p)\): minimiza con reparametrización. Arranca desde \(\mu_0 = 0{,}5\) y desde \(\mu_0 = -0{,}5\). ¿A qué modo va cada una?

## 6. Ganancia de información para elegir un punto de vista (30 min)

Un objetivo está en una posición 2D desconocida con belief \(\mathcal N(\bar{\mathbf x}, P)\), \(P = \mathrm{diag}(9, 1)\) m². Un dron puede observarlo desde 8 puntos situados en un círculo de 10 m alrededor de \(\bar{\mathbf x}\), con un sensor de distancia y rumbo (ruido de 0,5 m y 2°).

1. Linealiza la medida en \(\bar{\mathbf x}\) para cada punto de vista y calcula \(\mathrm{IG} = \tfrac12\log\det(I + R^{-1}HPH^\top)\).
2. Elige el mejor punto de vista y explica geométricamente por qué es ese.
3. Simula 2000 casos: compara el error final tras una medida desde el mejor punto con el de un punto al azar.

## Entregables

- `so3.py` con los cinco tests en verde.
- Un cuaderno o script con cuatro figuras: error de integración, plátano, ELBO frente a log p(x), y KL directa frente a inversa.
- Una tabla con la IG de los 8 puntos de vista y el resultado de la simulación.

## Preguntas para cerrar

1. Tu ESKF del tema A4a guardará la actitud como cuaternión. ¿De qué tamaño será su covarianza de actitud y por qué?
2. Si tu belief sobre la posición del objetivo fuera bimodal, ¿qué pasaría con la regla de la ganancia de información linealizada? ¿Qué harías en su lugar?
