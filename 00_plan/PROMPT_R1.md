# Prompt para crear R1a y R1b con Claude Code

Pega el bloque de abajo en Claude Code, abierto en la carpeta del repositorio. Claude preparará el terreno, escribirá los dos temas con un subagente cada uno en paralelo, los revisará, hará los commits y se detendrá para que los revises. El plan completo está en [`PLAN_R1.md`](PLAN_R1.md).

---

```text
Crea los temas R1a y R1b del curso de robótica inteligente de este repositorio («Así lo hacen en la realidad»), siguiendo 00_plan/PLAN_R1.md.

1. Haz git pull. Lee, en este orden: CLAUDE.md, CONTINUAR.md, 00_plan/PLAN_R1.md (entero), 00_plan/ESTADO.md, 00_plan/TEMARIO.md (las filas y el contenido obligatorio de R1a y R1b) y 00_plan/diagnostico.md. Abre temas/M0_mapa-del-campo/index.html, temas/M1_herramientas-matematicas/index.html y temas/T1b_taller-simular-la-realidad/index.html como modelos.

2. Preparación (PLAN_R1.md §4.1), hecha por ti antes de lanzar nada: marca R1a y R1b como ⏳ en ESTADO.md; añade el componente de foto (figure.foto y .foto.duo) a assets/curso.css, sube curso.css a ?v=13 en todas las páginas y documéntalo en CONTINUAR.md §5; crea las dos carpetas con img/, figuras/, lab/ y img/CREDITOS.md.

3. Lanza dos subagentes en el mismo mensaje, uno para R1a y otro para R1b, cada uno con la orden de encargo de PLAN_R1.md §4.2 rellenada (ruta del repositorio, carpeta, puerto 8765 o 8766, ruta absoluta del scratchpad para los dossiers). Ningún subagente lanza otros ni toca git, assets/, index.html ni 00_plan/. Mientras trabajan, no hagas tú su trabajo.

4. Cuando llegue cada informe, revisa ese tema como revisor independiente (PLAN_R1.md §4.3, punto 1): la licencia de todas las fotos en su página de origen, una muestra de cifras y referencias contra su fuente, todas las cuentas rehechas contra el script, los límites del apartado de Ucrania y las comprobaciones del navegador de CONTINUAR.md §6 repetidas. Corrige solo lo que compruebes. Si un subagente se queda corto, continúalo con SendMessage o termina tú el tema; no lo relances desde cero.

5. Integra (PLAN_R1.md §4.3, puntos 2 a 4): coherencia entre los dos temas, la figura del plan de M0 (hoja 3) con las jornadas nuevas, index.html y ESTADO.md. Comprueba los criterios de aceptación de PLAN_R1.md §8.

6. Haz los tres commits de PLAN_R1.md §4.3, punto 5, en ese orden, y push.

7. Para (🛑 de R1b) y dime: qué temas has hecho, con su número de hojas; qué fotos has usado y con qué licencia; qué cifras o referencias no pudiste verificar y qué casos se quedaron fuera por falta de fuentes; qué decisiones de diseño tomaste; cualquier desviación respecto a PLAN_R1.md y por qué. Después espera mi respuesta. Lo que pida se anota en el registro de revisiones de ESTADO.md y se aplica también a los temas siguientes.

Entre el inicio y la parada no me preguntes nada salvo que estés bloqueado de verdad: las decisiones de profundidad y diseño son tuyas, guiadas por PLAN_R1.md y CONTINUAR.md. Nunca inventes referencias, autores ni cifras; las que solo publica una empresa van como afirmaciones suyas. Ninguna foto entra sin licencia libre comprobada: el repositorio es público.
```
