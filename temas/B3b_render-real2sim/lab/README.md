# Laboratorio B3b · Observaciones del dron desde una escena 3DGS

**Duración:** unas 3,5 h (el paso 5 es opcional).
**Requisitos:** una GPU NVIDIA con CUDA para `gsplat` (8 GB bastan para renderizar una escena preentrenada a resolución reducida; para entrenar, mejor 12 GB o más) o una GPU en la nube. Sin GPU: el paso 2 tiene una alternativa en CPU con tu propio rasterizador en JAX, a baja resolución y con una parte de las gaussianas. Tu simulador del x500 y los modelos de sensor de B3a.

## Objetivo

Usar una escena real reconstruida con Gaussian Splatting como «mundo visual» de tu simulador: renderizar lo que vería la cámara del dron a lo largo de una trayectoria, añadirle los defectos de sensor de B3a y medir dónde deja de ser fiable la escena. Comprueba con tus números lo que calcula la presentación:

| Resultado de la presentación | Valor | Paso |
|---|---|---|
| Render volumétrico de un rayo (cristal y pared): color con 16, 32 y 64 muestras uniformes | 0,4763; 0,4288; 0,4283 (referencia 0,4283) | 1 |
| Profundidad esperada en ese rayo | 3,57 m: un punto en el aire, entre el cristal y la pared | 1 |
| Parámetros por gaussiana y memoria de un millón | 59 floats; 236 MB | 2 |
| Ángulo a la vista de captura más cercana (captura a 1,5 m, dron a 5 m de altura y 5 m de radio) | 32,9° | 4 |

El script de las figuras está en `../figuras/figuras_b3b.py`.

## 1. Render volumétrico a mano (30 min)

En `src/proyecto_hilo/render/volumen.py`, en JAX:

1. La cuadratura del render volumétrico: \(\alpha_i = 1 - e^{-\sigma_i\delta_i}\), \(T_i = \prod_{j<i}(1-\alpha_j)\), \(C = \sum_i T_i\alpha_i c_i\), y la profundidad esperada \(\sum_i T_i\alpha_i t_i/\sum_i T_i\alpha_i\).
2. Reproduce el rayo de la presentación (cristal gaussiano en 2 m, pared en 4 m) y la tabla de error frente al número de muestras, uniformes y estratificadas.
3. Calcula, además de la profundidad esperada, la mediana de la profundidad (donde la opacidad acumulada cruza 0,5).

**Comprobaciones:** las cifras de la tabla. **Piensa:** ¿qué profundidad usarías para detectar colisiones a partir de un campo de radiancia: la esperada o la mediana? ¿Por qué?

## 2. Una escena 3DGS real (50 min)

1. Instala `gsplat` (licencia Apache-2.0) y descarga **una** escena preentrenada del artículo original de 3DGS (por ejemplo, `bicycle` o `garden`, de Mip-NeRF 360) con su script `examples/download_3dgs_paper_scenes.py`, que extrae solo esa escena del archivo de 14,66 GB con peticiones por rangos (unos 700 MB). Guárdala en su propia carpeta de datos. Ojo: esos modelos preentrenados se distribuyen con la licencia del código original de 3DGS, que solo permite uso de investigación no comercial.
2. Carga el `point_cloud.ply`, cuenta las gaussianas y calcula la memoria de sus parámetros. Comprueba que cada una tiene 59 números (posición, escala, rotación, opacidad y armónicos esféricos de grado 3).
3. Renderiza la escena con `gsplat.rasterization` desde algunas de las cámaras de captura (las poses de COLMAP vienen con el dataset Mip-NeRF 360; descarga solo la escena con `remotezip`, como hiciste con NeuroBEM) y mide el tiempo por imagen a 1/4 de resolución y a resolución completa.

**Sin GPU:** escribe un rasterizador mínimo en JAX (proyección \(\Sigma' = JW\Sigma W^\top J^\top\), orden por profundidad y composición alfa por píxel) y úsalo con las 50 000 gaussianas más opacas a 160 × 120 píxeles. Será lento, pero te enseña exactamente lo que hace `gsplat`.

**Comprobaciones:** las imágenes desde las cámaras de captura deben parecerse a las fotos originales; calcula su PSNR frente a ellas.

## 3. La cámara del dron en la escena (50 min)

1. Alinea la escena con tu mundo: COLMAP no conoce la escala ni la gravedad. Estima la vertical con la dirección media de las cámaras (o con el plano del suelo, ajustando un plano a las gaussianas más bajas) y fija la escala con una medida conocida de la escena (dila).
2. Vuela el x500 de tu simulador por una trayectoria dentro de la escena (por ejemplo, un círculo alrededor del objeto central) y renderiza la cámara frontal a 30 Hz con la pose del simulador más los extrínsecos cámara-cuerpo.
3. Añade los efectos de B3a: desenfoque por movimiento (promediando varios renders dentro del tiempo de exposición), obturador rodante (render por bandas de filas con la pose de cada instante) y ruido de disparo y de lectura.

**Mide:** tiempo de render por imagen con y sin efectos. ¿Cuántos pasos por segundo con imagen te caben en tu GPU? Compáralo con las cifras de B1.

## 4. ¿Dónde deja de ser fiable la escena? (40 min)

1. Para un conjunto de vistas de prueba, a distintas alturas y distancias, calcula el ángulo a la cámara de captura más cercana (como en la presentación).
2. Usa como «verdad» las fotos de captura que hayas dejado fuera del entrenamiento si entrenas tú la escena (paso 5) o, con la escena preentrenada, mide al menos las métricas sin referencia: fracción de píxeles vacíos o con opacidad acumulada baja, y gaussianas grandes y alargadas en pantalla.
3. Representa esas métricas frente al ángulo y a la altura.

**Comprobaciones:** con una captura a pie, la calidad cae cuando el dron sube por encima de la altura de captura (a 5 m, unos 33° de la vista más cercana en el ejemplo de la presentación).

## 5. Entrenar tu propia escena (opcional, 40 min de GPU)

1. Con `examples/simple_trainer.py` de `gsplat`, entrena la escena `garden` a 1/4 de resolución, dejando fuera un 10 % de las imágenes.
2. Calcula PSNR, SSIM y LPIPS en las vistas de prueba y su relación con el ángulo a la vista de entrenamiento más cercana.
3. Prueba la opción de 3DGUT (`with_ut`), que admite cámaras con distorsión y obturador rodante, y compara el render con tu aproximación por bandas del paso 3.

## Entregables

- `render/volumen.py` y `render/splats.py` (envoltorio de `gsplat` o rasterizador propio), con tests: la cuadratura con muchas muestras converge al color analítico del rayo; una gaussiana sola proyectada con \(JW\Sigma W^\top J^\top\) coincide con el muestreo Monte Carlo de puntos 3D proyectados.
- `experimentos/b3b_escena.py`, que genere: el vídeo de la cámara del dron con y sin efectos de sensor, el tiempo por imagen y la calidad frente al ángulo a la captura.
- Una tabla con tus valores frente a los de la presentación.

## Preguntas para cerrar

1. Si tu política de tesis aprende de imágenes renderizadas con 3DGS, ¿qué parte de su error en vuelo esperas que venga del render y qué parte de la dinámica? ¿Cómo lo separarías?
2. ¿Cómo capturarías una escena para que un dron pueda volar en ella a 5–10 m de altura sin salirse de las vistas fiables?
3. Una escena 3DGS es estática y con la iluminación del día de la captura. ¿Qué aleatorizarías encima para que la política no aprenda esa escena concreta?
