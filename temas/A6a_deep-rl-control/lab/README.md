# Laboratorio A6a · PPO en JAX: estacionario y seguimiento

**Duración:** unas 4 h.
**Requisitos:** tu simulador vectorizado de B6 y `optax`. Una GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`) o en la nube; en CPU funciona con menos entornos (por ejemplo, 256) y bastante más tiempo.

## Objetivo

Escribir PPO desde cero sobre tu simulador del x500, entrenar una política de estacionario y otra de seguimiento, y comprobar con varias semillas qué decisiones de recompensa e implementación importan. Comprueba con tus números lo que calcula la presentación (una ejecución de `figuras/figuras_a6a.py` en una RTX 4060 Ti; en GPU, los resultados cambian entre ejecuciones aunque la semilla sea la misma):

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Entrenamiento de 78,6 M pasos (4096 entornos × 32 pasos, 600 actualizaciones) | unos 10 s en GPU | 2 |
| Estacionario desde 200 arranques aleatorios, recompensa acotada | 0 choques; 2–5 cm de error según la semilla | 2 |
| Recompensa ingenua \(1 - c - 10\cdot\)choque, 5 semillas | ninguna vuela sin choques con error < 20 cm | 3 |
| 16 384 entornos × 8 pasos frente a 4096 × 32 (mismas muestras) | 0,69 frente a 0,78 de recompensa media | 5 |
| Gradiente de la política con GAE en la cadena de 19 estados | sesgo −17 % con λ = 0; sin sesgo apreciable con λ ≥ 0,95 | 1 |
| Rejilla con +1 por estar junto a la meta | nunca llega | 3 |

## 1. PPO desde cero (60 min)

En `src/proyecto_hilo/rl/ppo.py`, con la estructura de `figuras_a6a.py`:

1. Despliegue con `lax.scan` sobre tus entornos vectorizados: observación, acción gaussiana, \(\log\pi\), valor, recompensa, finales por choque y por tiempo.
2. GAE con `lax.scan(..., reverse=True)`. **Test:** en la cadena de 19 estados de la sección 2 del script, reproduce el sesgo y la varianza del gradiente frente a \(\lambda\).
3. Arranque del valor en los cortes por tiempo (\(r \mathrel{+}= \gamma V(s_\text{final})\)); **test:** un entorno que solo termina por tiempo debe dar el mismo objetivo del valor con y sin el corte.
4. Pérdida recortada, valor, minilotes barajados, épocas, recorte de la norma del gradiente. Registra KL, fracción recortada y σ.

## 2. Estacionario (40 min)

1. Entrena con la recompensa acotada \(1 - \tanh c - 10\cdot\)choque, 5 semillas.
2. Evalúa cada política desde 200 arranques aleatorios durante 10 s: choques, error a 1 s y error en régimen.
3. Presenta media y dispersión entre semillas, no la mejor.
4. Ejecuta el script dos veces en la GPU con la misma semilla y compara. Después, dentro de un mismo proceso. ¿Qué cambia?

## 3. Recompensas (40 min)

1. Entrena con la recompensa ingenua \(1 - c - 10\cdot\)choque (5 semillas) y cuenta cuántas aprenden a estrellarse. Comprueba la derivación de la presentación: ¿cuál es la recompensa media por paso al principio del entrenamiento?
2. Añade una constante de supervivencia, como Eschmann et al., hasta que deje de compensar terminar.
3. Prueba un término de potencial \(\gamma\Phi(s') - \Phi(s)\) con \(\Phi = -\|e_p\|\): ¿cambia la política final? ¿Y la velocidad de aprendizaje?
4. Reproduce la rejilla de la sección 3 del script y crea tú una bonificación que se pueda explotar.

## 4. Seguimiento de trayectorias (40 min)

1. Cambia el entorno para que la consigna se mueva: círculos y lemniscatas con radio y periodo aleatorios, y añade a la observación la velocidad (y, si quieres, la aceleración) de la referencia.
2. Entrena y evalúa en el círculo de 1 m de radio y 6 s de periodo de la presentación (la política de estacionario da unos 65 cm de error) y en trayectorias que no vio.
3. Compara con el seguidor por planitud y el LQR de A3a en las mismas trayectorias.

## 5. Ablaciones (40 min)

Con el estacionario y 3 semillas por variante:

1. \(\lambda\) ∈ {0, 0,9, 0,95, 0,99} y entornos × pasos ∈ {256 × 512, 1024 × 128, 4096 × 32, 16 384 × 8}, a igual número de muestras: recompensa frente a pasos y frente a tiempo de reloj.
2. Ganancia de la última capa de la política: 0,01 frente a \(\sqrt 2\).
3. Normalización de observaciones con media y varianza móviles frente a la escala fija.
4. Cuaternión frente a matriz de rotación en la observación.

## 6. Con las capas de realismo (20 min)

Activa en el entrenamiento la aleatorización por entorno de masa y constante de motor (B2), el ruido de la IMU (B3a) y un retardo de 20–40 ms en la observación (B4). Evalúa la política de los pasos anteriores (entrenada sin ellas) y la nueva en el simulador con todas las capas. Es el punto de partida de A6b.

## Entregables

- `rl/ppo.py` con tests (GAE en la cadena, corte por tiempo, forma de los lotes), y `rl/entornos.py` con estacionario y seguimiento.
- `experimentos/a6a_ppo.py`, que genere: curvas de aprendizaje con bandas entre semillas, tabla de evaluación, tabla de recompensas y ablaciones.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. Tu tarea de tesis tiene episodios que terminan cuando el dron encuentra el objetivo o se estrella. ¿Qué forma debe tener la recompensa para que ni terminar pronto ni alargar el episodio compense?
2. ¿Qué parte de la observación de este laboratorio no tendrías en vuelo real, y cómo la sustituirías?
3. Con lo medido en el paso 5, ¿cuántos entornos usarías en tu GPU para la tarea de tesis, que necesita memoria de varios segundos?
