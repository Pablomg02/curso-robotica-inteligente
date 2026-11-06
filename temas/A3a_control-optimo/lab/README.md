# Laboratorio A3a · LQR, MPC y NMPC con acados para el x500

**Duración:** unas 3,5 h (el paso 5 es opcional).
**Requisitos:** el modelo del x500 del laboratorio A1 (con la inercia compuesta, \(J = \operatorname{diag}(0{,}0239; 0{,}0239; 0{,}044)\)) y la cascada de A2. Todo corre en CPU.

## Objetivo

Construir en el proyecto hilo los controladores óptimos del x500 y comprobar con tus propios números lo que la presentación calcula:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Polos del LQR del x500 (Bryson, 100 Hz) | 2,6 rad/s (horizontal), 6,3 (inclinación), 19,7 (rate) | 1 |
| Horizonte con el que la ganancia queda al 10 % de la LQR infinita (eje x, sin coste terminal) | \(N\) = 96 (0,96 s) | 1 |
| Salto de 1 m con límite de 20°: inclinación máxima | MPC 20,0°, LQR saturado 53,2° | 2 |
| Salto de 2 m con LQR saturado | diverge en el modelo lineal | 2 |
| Coste del QP con \(n\) = 12, \(m\) = 4, \(N\) = 20, condensando en cada muestra, frente a Riccati | unas 5 veces | 2 |
| RTI frente a SQP convergido (cuadricóptero plano, lemniscata) | mismo error RMS (3,80 cm), diferencia máxima de 0,03 cm, unas 7 veces menos cálculo | 4 |

El script de las figuras está en `../figuras/figuras_a3a.py`. Úsalo para comparar al final de cada paso.

## 0. Instalación (30 min, se puede hacer antes)

- **Python:** `uv add numpy scipy osqp casadi` en el proyecto hilo.
- **acados** (v0.6.0 o posterior), compilado desde el código fuente según [docs.acados.org/installation](https://docs.acados.org/installation/index.html):

```bash
git clone https://github.com/acados/acados.git && cd acados
git submodule update --recursive --init
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON .. && make install -j4
pip install -e <acados_root>/interfaces/acados_template
export ACADOS_SOURCE_DIR=<acados_root>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<acados_root>/lib
```

  La primera vez que generes un *solver*, acados descarga el binario `t_renderer`. En Windows, usa WSL. Si no consigues instalarlo, haz el paso 3 con la alternativa en JAX.

## 1. LQR del x500 (40 min)

En `src/proyecto_hilo/control/lqr.py`:

1. Linealiza tu modelo de A1 en estacionario con `jax.jacfwd`, perturbando la actitud en el espacio tangente, \(\mathbf q = \mathbf q_0\otimes\operatorname{Exp}(\delta\boldsymbol\theta)\) (M1), para quedarte con 12 estados: posición, velocidad, \(\delta\boldsymbol\theta\) y velocidad angular; entradas: empuje colectivo y tres pares. Si derivas directamente respecto al cuaternión (13 estados), la parte vectorial vale \(\theta/2\) (y sale \(\dot v_x = 2g\,q_y\)) y aparece un modo no controlable, la norma del cuaternión, con el que la DARE no tiene solución estabilizante. Compara con las matrices del script (sección 1): \(\dot v_x = g\theta\), \(\dot v_y = -g\phi\), \(\dot v_z = \Delta T/m\) en ENU/FLU.
2. Discretiza con retenedor de orden cero a 100 Hz (`jax.scipy.linalg.expm` de la matriz aumentada).
3. Resuelve la DARE iterando Riccati hasta converger, y compárala con `scipy.linalg.solve_discrete_are`.

**Tests:**
- \(A - BK\) tiene todos sus autovalores dentro del círculo unidad.
- Con los pesos de Bryson del script (0,5 m; 1 m/s; 0,3 rad, 0,5 en guiñada; 2 rad/s; mandos de 13,9 N y 1; 1; 0,1 N·m), los polos continuos equivalentes, \(\ln\lambda/\Delta t\), se agrupan en 2,1–2,6, 6,3–6,4 y 19,7 rad/s.
- Recursión de horizonte finito con coste terminal nulo en el subsistema del eje x: la ganancia del primer paso queda al 10 % de la infinita con \(N\) = 96.

**Pregunta:** cambia el peso de la posición horizontal por 10. ¿Cómo se mueven los tres grupos de polos? ¿Se mantiene la separación de escalas?

## 2. MPC lineal con restricciones (45 min)

En `control/mpc_lineal.py`, con el subsistema del eje x (\(x, v_x, \theta, q\)) a 50 Hz:

1. Construye la forma condensada \(X = \Phi x_0 + \Gamma U\) y el QP de la presentación, con \(N\) = 50, coste terminal de la DARE, \(|\tau_y| \le 2{,}43\) N·m y \(|\theta| \le 20°\).
2. Resuélvelo con OSQP en cada paso, con arranque en caliente (`warm_start`), y compara con el ADMM sencillo del script.
3. Simula el salto de 1 m con el MPC y con el LQR saturado (mismos pesos: \(Q = \operatorname{diag}(10; 1; 1; 0{,}01)\), \(R = 0{,}05\)).

**Comprobaciones:** inclinación máxima de 20,0° ± 0,2° con el MPC (la tolerancia por defecto de OSQP admite pequeñas violaciones) y de 53,2° con el LQR saturado; con 2 m, el LQR saturado diverge. Mide el tiempo medio por QP de OSQP.

4. Escribe tú dos *solvers* sin restricciones del mismo problema: uno condensado (construir \(H\) y factorizarlo con Cholesky en cada llamada) y otro por la recursión de Riccati. Comprueba que dan el mismo \(u_0\) y mide su tiempo para \(N\) = 20, 80 y 160. Para este subsistema (\(n\) = 4, \(m\) = 1), el recuento da cocientes de ×3, ×48 y ×192 (sección 4 del script). No uses OSQP para esta medida: factoriza el problema disperso una vez y reutiliza la factorización, así que no muestra el escalado cúbico frente a lineal.

## 3. NMPC del x500 con acados (60 min)

1. Escribe el modelo en CasADi (`AcadosModel`): estado \((p, v, q, \omega)\) de 13 componentes y entradas = empujes de los cuatro motores (deja la dinámica de motor fuera del NMPC; la tendrá la planta).
2. OCP: \(N\) = 20 nodos de 0,05 s (1 s), coste `LINEAR_LS` sobre posición, velocidad, actitud (parte vectorial del error de cuaternión) y empujes respecto al estacionario; restricciones \(0 \le T_i \le 8{,}55\) N.
3. Opciones: `nlp_solver_type='SQP_RTI'`, `hessian_approx='GAUSS_NEWTON'`, `integrator_type='ERK'`, `qp_solver='PARTIAL_CONDENSING_HPIPM'`.
4. Lazo cerrado a 100 Hz contra **tu** simulador de A1 (motores incluidos), siguiendo una lemniscata de 3 m × 1 m a unos 4 m/s. Normaliza el cuaternión de la planta antes de pasárselo al *solver*.

**Mide:** error RMS de posición, tiempo por llamada (media y máximo) y empuje mínimo pedido.

**Alternativa sin acados:** usa el Gauss-Newton con disparo simple de `figuras_a3a.py` (sección 5), pásalo a JAX con `jax.jit` y aplícalo al modelo plano.

## 4. RTI, SQP y latencia (30 min)

1. Repite el paso 3 con `nlp_solver_type='SQP'` (hasta 20 iteraciones). Compara error y tiempo con el RTI.
2. Separa las fases del RTI con `rti_phase` (1 = preparación, 2 = realimentación) y mide cada una: ¿cuánto dura la realimentación frente a la preparación?
3. Añade 10 y 30 ms de retardo entre el *solver* y la planta. Después compénsalo prediciendo con el modelo el estado en el instante en que el mando llegará al dron.

**Comprobación:** con el cuadricóptero plano del script, el RTI y el SQP convergido dan el mismo error RMS (3,80 cm, con 0,03 cm de diferencia máxima) y el SQP necesita unas 7 iteraciones por paso. En el x500 completo, la razón de tiempos será distinta: anota la tuya.

## 5. MPCC en el plano (opcional, 30 min)

Con el modelo plano (\(y, z\)), añade la variable de progreso \(\vartheta\) y su velocidad \(v\) como estado y mando virtual, y el coste de contorno y retraso de la presentación sobre un circuito circular. Barre el peso del progreso \(\rho\) y representa el tiempo por vuelta frente al error de contorno máximo.

## Entregables

- `control/lqr.py`, `control/mpc_lineal.py` y `control/nmpc.py`, con tests en verde.
- `experimentos/a3a_mpc.py`, que genere: polos del LQR, salto con y sin restricción, tiempo de QP frente a \(N\), y error y tiempo de RTI frente a SQP con 0, 10 y 30 ms de retardo.
- Una tabla con tus valores frente a los de la presentación y una línea por cada diferencia de más del 10 %.

## Preguntas para cerrar

1. Tu NMPC necesita el estado completo a 100 Hz. ¿De dónde sale en un dron real, con qué retardo, y cómo cambia eso el diseño (A4)?
2. ¿Qué lazo de la cascada de A2 dejarías debajo del NMPC, y por qué no lo sustituirías por el propio NMPC?
3. Si sustituyes el modelo de CasADi por un modelo aprendido (A3b, A8), ¿qué propiedades de este tema pierdes: estabilidad, factibilidad, coste computacional, convergencia del RTI?
