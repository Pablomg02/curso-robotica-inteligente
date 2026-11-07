# Laboratorio A9a · Masa y viento desconocidos en el x500

**Duración:** unas 4 h.
**Requisitos:** el x500 de A8a (B2) y JAX con optax. Basta la CPU; con GPU NVIDIA de consumo (`uv run --with "jax[cuda13]" --with optax`), todo el script tarda unos minutos. Lanza los entrenamientos con `python -u` y un límite de memoria (`ulimit -v`). No cierres un `jax.jit` sobre arrays grandes: pásalos como argumentos.

## Objetivo

Comparar formas de adaptarse a una carga que se suelta en vuelo y a un viento que entra de golpe, desde un PD sin adaptación hasta un estimador aprendido desde el historial. Distinguir seguir bien de identificar bien. Comprueba con tus números lo que calcula la presentación (`figuras/figuras_a9a.py`, salida en `figuras/salida_gpu.txt`, determinista en la misma GPU):

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Flotar, RMS tras soltar la carga (5–8 s): PD / PID / Kalman f / oráculo | 43,8 / 37,5 / 3,4 / 1,4 cm | 2 |
| Pico tras soltar la carga: Kalman f / Kalman (m, f) | 5,7 / 8,8 cm | 2 |
| Error de masa a los 4,9 s con Kalman (m, f): flotar / ocho (suponer la nominal: 25,6 %) | 30,1 / 9,6 % | 3 |
| Ídem con varianza pequeña para f_z, flotando | 1,2 % | 3 |
| Error de la fuerza con 40 puntos: vector constante / constante + base | 1,36 / 1,06 N² | 4 |
| Ocho, RMS 1–5 s: Kalman f / base aprendida / oráculo | 9,2 / 9,0 / 8,0 cm | 4 |
| GRU flotando: error de masa; RMS 2–5 s | 0,5 %; 2,2 cm | 5 |
| GRU con corriente descendente: error de masa; RMS 2–5 s | 5,7 %; 2,2 cm | 5 |

## 1. Escenarios (30 min)

1. Añade al x500 una fuerza aerodinámica \(f = k_l v_r + k_q |v_r| v_r\) con viento \(v_r = v_w - v\) (nuestro supuesto: \(k_l = 0{,}15\), \(k_q = 0{,}03\)). Ráfagas con un proceso de Ornstein-Uhlenbeck.
2. Flotar: carga de +10–40 % que se suelta a los 5 s; viento de 4–10 m/s que entra a los 2 s. Ocho: 2 m en 10 s con oscilación vertical de 1 m.
3. Mide la fuerza específica con un acelerómetro con ruido y el empuje con el modelo del motor.

## 2. Filtros adaptativos (60 min)

1. Escribe \(y = T R e_3 = m s - f\) y estima \((m, f)\) con un filtro de Kalman con paseo aleatorio. Filtra igual la medida y el regresor.
2. Variantes: solo \(f\) con la masa nominal; \((m, f)\); \((m, f)\) regularizado hacia la masa nominal; \((m, f)\) con varianza pequeña para \(f_z\).
3. **Tests:** (a) con masa y viento conocidos y sin ruido, el filtro debe quedarse quieto en los valores verdaderos; (b) con el regresor constante, la covarianza en la dirección \((1, s)\) debe crecer.

## 3. Identificabilidad (45 min)

1. Representa la masa estimada al flotar y en el ocho. Relaciónalo con la excitación persistente.
2. Sube el ruido del acelerómetro a 0,2 m/s² y observa la deriva de la masa hacia abajo. Explícala como un sesgo de errores en las variables: el ruido está en el regresor, y la fuerza, que se adapta deprisa, se lleva la banda baja.
3. Ajusta la regularización (`R_REG`) para que la masa no derive al flotar y siga identificándose en el ocho. ¿Existe un valor que haga las dos cosas?

## 4. Base meta-aprendida (45 min)

1. Genera episodios con vientos distintos y meta-aprende \(\phi(v)\): regresión de cresta con 40 puntos de cada episodio (bucle interno) y error en el resto (bucle externo).
2. Úsala en un filtro sobre \(f = c + \phi(v)\,a\). Compara con el Kalman \(f\) al flotar y en el ocho.
3. Busca el régimen en el que la base ayuda: un ocho más rápido, más viento o un filtro más lento.

## 5. Estimador por historial (40 min)

1. Entrena una GRU que estime masa y fuerza desde (fuerza específica, empuje, velocidad), con los valores verdaderos del simulador, como el módulo de adaptación de RMA.
2. Pruébala fuera de distribución: más carga y más viento, y una corriente vertical. ¿Qué regularidad de los datos estaba usando?
3. Sondea su estado oculto (A7b): ¿qué codifica?

## Entregables

- `adaptacion/kalman.py` y `adaptacion/gru.py` con tests; `experimentos/a9a.py`, que genere las tablas de bucle cerrado e identificabilidad.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. ¿Qué parámetros de tu dron de tesis estimarías en vuelo, y cuáles fijarías?
2. ¿Qué maniobra harías para identificar la masa sin comprometer una búsqueda? ¿Cuánto vale esa información?
3. Si tu estimador aprendido se entrena en simulación, ¿qué regularidades del simulador podría estar aprendiendo como prior?
