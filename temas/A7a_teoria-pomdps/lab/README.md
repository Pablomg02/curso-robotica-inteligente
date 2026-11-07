# Laboratorio A7a · Belief exacto frente a POMCP, y tu entorno de búsqueda

**Duración:** unas 4 h.
**Requisitos:** Python con numpy; solo CPU. Opcional: JAX para vectorizar las simulaciones (B6) o [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) para comparar con resolvedores establecidos. **Cuidado con la memoria:** no evalúes en una rejilla fina todos los vectores α enumerados (crecen como \(|\mathcal A|\,|\Gamma|^{|\mathcal Z|}\)); usa la envolvente exacta o poda por programación lineal, y lanza las pruebas con un límite (`ulimit -v 4000000`).

## Objetivo

Construir y resolver POMDP pequeños de forma exacta, comprobar cuándo fallan las aproximaciones y dejar listo el entorno de búsqueda con un dron que usarán A7b, A8a y A15a. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a7a.py`, salida en `figuras/salida.txt`; los ejemplos hechos a mano de las hojas, como la actualización del belief paso a paso, el backup de vectores α o las cuentas de QMDP, están en `figuras/ejemplos_a7a.py`, salida en `figuras/salida_ejemplos.txt`):

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Tigre (γ = 0,95): vectores α en la envolvente | 3, 5 y 9 en los horizontes 1–3; 9 en el límite | 1 |
| Umbral de la política del tigre | escuchar mientras 0,040 < b < 0,960: oír dos veces más un lado | 1 |
| Valor en b = ½ | escuchar 19,37; abrir −26,60 | 1 |
| Pasillo con cartel | exacta 5,19; ir y adivinar −4,44; QMDP −1,91 (no elige nunca) | 2 |
| POMCP en el tigre, 10 pasos, 1024 simulaciones | despliegue aleatorio −40,4; despliegue «escuchar» 2,32; exacta 6,16 | 3 |
| Búsqueda (6 × 6, P_D = 0,6), pasos medios hasta detectar | barrido 37,9; máximo del belief 34,0; ganancia por distancia 32,4; POMCP con 500 simulaciones 34,4 | 4 |
| POMCP con la heurística como despliegue | 47,8 (100 simulaciones) y 39,6 (300) | 4 |

## 1. El tigre, exacto (60 min)

1. Escribe el modelo de Kaelbling et al. (§5.1): \(T\), \(O\), \(R\) como arrays. Usa \(\gamma = 0{,}95\).
2. Actualización del belief. **Test:** tras «izquierda, izquierda, derecha», \(b = 0{,}85\).
3. Iteración de valor exacta con vectores α: para cada acción, todas las combinaciones de un vector por observación; poda con la envolvente superior. Con dos estados la envolvente es la de un conjunto de rectas en \([0, 1]\): hazla exacta (envolvente convexa de rectas), no evaluando en una rejilla.
4. Representa \(V_1\), \(V_2\), \(V_3\) y el límite; cuenta los vectores. Calcula los umbrales de la política y comprueba la regla «escucha hasta oír dos veces más un lado».
5. Comprueba con números la desigualdad de Jensen de la presentación: en \(p = 0{,}5\), compara \(\sum_z P(z)\,V(b^z)\) con \(V(0{,}5)\).

## 2. QMDP y el cartel (30 min)

1. Reproduce el pasillo de la sección 3: el valor exacto (5,19), el de adivinar (−4,44) y el bloqueo de QMDP en la bifurcación.
2. Haz el cartel ruidoso (dice el lado correcto con probabilidad 0,8). El belief ya no toma tres valores: discretízalo o resuelve con vectores α (2 estados ocultos × 8 posiciones). ¿Sigue mereciendo la pena ir al cartel? ¿Y si hay que volver a leerlo?
3. Inventa otro problema en el que QMDP falle y otro en el que acierte.

## 3. POMCP (60 min)

1. Implementa POMCP: árbol de historias, UCB, despliegue, belief de partículas con rechazo y reinvigoración.
2. En el tigre, representa el retorno frente al número de simulaciones con barras de error y la línea de la política exacta.
3. Cambia el despliegue aleatorio por uno que solo escuche, o por uno que use el umbral: ¿cuántas simulaciones ahorras?
4. Reutiliza el subárbol tras cada paso en lugar de empezar de cero.

## 4. Búsqueda con el dron (60 min)

En `src/proyecto_hilo/pomdp/busqueda.py`:

1. La rejilla de la sección 5: objetivo quieto, prior con una zona caliente, detección con \(P_D = 0{,}6\), sin falsas alarmas. Interfaz de entorno vectorizable (como B6): estado, observación, belief exacto.
2. Políticas: barrido, ir al máximo del belief, ganancia por distancia y POMCP. Mismos objetivos para todas; media, mediana y fracción sin encontrar.
3. Añade falsas alarmas (\(P_{FA} = 0{,}05\)) y una acción «declarar encontrado» con recompensa +20 si acierta y −20 si no. ¿Cómo cambia el comportamiento?
4. Guarda el entorno con tests: lo usarás en A7b con políticas recurrentes y belief aprendido.

## 5. Observaciones continuas (30 min, opcional)

1. Cambia el detector: si el objetivo está a menos de 1,5 casillas, devuelve su posición con ruido gaussiano (σ = 0,5 casillas) con probabilidad \(P_D\).
2. Ejecuta POMCP tal cual y mide cuántas visitas reciben los nodos de observación. Implementa el ensanchamiento progresivo en las observaciones y las partículas con pesos de POMCPOW, y compara.

## Entregables

- `pomdp/tigre.py` (modelo, belief, vectores α, tests), `pomdp/pomcp.py`, `pomdp/busqueda.py` con tests.
- `experimentos/a7a.py`, que genere la figura del valor del tigre, la de POMCP y la tabla de búsqueda.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. En tu tesis, ¿qué parte del estado es observable directamente, cuál se estima con un filtro (A4a) y cuál solo se descubre buscando?
2. ¿Qué presupuesto de cómputo por paso tendría POMCP a bordo de tu dron, y cuántas simulaciones supone eso con tu simulador?
3. ¿Qué te dice la cota del AIS sobre cómo entrenar el belief aprendido de A7b?
