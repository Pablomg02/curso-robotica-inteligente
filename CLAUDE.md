# Curso intensivo de robótica inteligente

Curso autodidacta de Pablo Magariños para empezar al día su doctorado (*belief states* y *world models* para drones en entornos parcialmente observables, Universidade de Vigo, desde octubre de 2026). Cubre control con IA, simulación, y hardware y despliegue, todo orientado a sim-to-real.

**Antes de trabajar, lee [`CONTINUAR.md`](CONTINUAR.md)** (manual completo) y [`00_plan/ESTADO.md`](00_plan/ESTADO.md) (qué está hecho y dónde parar). El temario está en `00_plan/TEMARIO.md`; el nivel y las preferencias de Pablo, en `00_plan/diagnostico.md`.

## Reglas

- Español, con los términos técnicos habituales en inglés. Trato de tú a Pablo.
- Cada tema es una presentación HTML en `temas/<ID>_<slug>/index.html` hecha con `assets/curso.css` y `assets/curso.js`, más su laboratorio en `lab/README.md`. M0 y M1 son el modelo.
- Investigar a fondo antes de escribir: revisiones recientes, papers seminales, trabajo de 2025–2026, documentación oficial y cifras de sistemas reales. Estructura pasado → presente → futuro.
- Cada referencia y cada cifra se verifica en su fuente. Nunca inventar referencias, autores, páginas ni números; si algo no se puede confirmar, se omite o se marca. Las cifras que solo publica una empresa se presentan como afirmaciones suyas.
- Derivaciones completas (plegables), figuras SVG y datos de figuras calculados con un script.
- Matemáticas desplegadas, sin densidad: notación de cada símbolo, de dónde sale cada fórmula, derivación paso a paso con el porqué de cada línea y un ejemplo numérico del script; mejor más hojas que hojas comprimidas (`CONTINUAR.md` §4, M1 como modelo).
- Hilos en todos los temas: latencia y frecuencia, observabilidad parcial, sim-to-real y cómputo limitado. Cerrar cada tema con su conexión con la tesis.
- Laboratorios independientes del hardware de Pablo: PC con GPU NVIDIA de consumo o nube, con CPU como alternativa. Nunca se vuela un dron; los datos reales salen de datasets públicos. Una Pixhawk solo como hardware-in-the-loop opcional.
- Ningún tema se da por terminado sin comprobar en el navegador que `curso.checkOverflow()` devuelve `[]` a 1440×900 y que nada desborda en horizontal a ancho de móvil (`CONTINUAR.md` §6). Al cambiar `assets/`, subir el `?v=` en todas las páginas.
- Al terminar cada tema: actualizar `index.html` y `00_plan/ESTADO.md`, commit y push. Parar en los temas marcados con 🛑 en `ESTADO.md` para que Pablo revise.
- Las decisiones de profundidad y diseño las toma el curso (Pablo lo pidió así), guiadas por `TEMARIO.md`. Lo que Pablo pida en una revisión se anota en `ESTADO.md` y se aplica a los temas siguientes.
