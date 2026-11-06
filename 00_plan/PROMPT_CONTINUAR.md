# Prompt para continuar el curso con Claude Code

Pega el bloque de abajo en Claude Code, abierto en la carpeta del repositorio. Claude producirá los temas en orden y se detendrá en cada punto 🛑 de `ESTADO.md` para que revises. Cuando lo hayas revisado, responde "sigue" (o di qué cambiar) y continuará hasta la siguiente parada.

---

```text
Continúa el curso de robótica inteligente de este repositorio.

1. Haz git pull. Lee, en este orden: CLAUDE.md, CONTINUAR.md, 00_plan/ESTADO.md, 00_plan/TEMARIO.md, 00_plan/diagnostico.md y 00_plan/ANALISIS.md. Abre temas/M0_mapa-del-campo/index.html y temas/M1_herramientas-matematicas/index.html como modelo de estilo, estructura y calidad.

2. Produce los temas pendientes en el orden de ESTADO.md, uno a uno, siguiendo el flujo de CONTINUAR.md §3 sin saltarte pasos: investigación a fondo en la web antes de escribir (revisiones recientes, papers seminales, trabajo de 2025–2026, documentación oficial y cifras de sistemas reales), verificación de cada referencia y cada cifra en su fuente, presentación con derivaciones completas, figuras calculadas con un script, test, lecturas guiadas, laboratorio y comprobación en el navegador (CONTINUAR.md §6).

3. Al terminar cada tema: actualiza index.html y 00_plan/ESTADO.md, haz commit y push.

4. Para en cada tema marcado con 🛑 en ESTADO.md, después de hacer su commit y push. En la parada, dime:
   - qué temas has hecho desde la última parada, con su número de hojas;
   - qué cifras o referencias no pudiste verificar del todo;
   - qué decisiones de diseño tomaste que debería conocer;
   - cualquier desviación respecto a TEMARIO.md y por qué.
   Después espera mi respuesta. Si digo "sigue", continúa hasta la siguiente parada. Si pido cambios, anótalos en el registro de revisiones de ESTADO.md, aplícalos a lo ya hecho cuando proceda y tenlos en cuenta en todos los temas siguientes.

5. Entre paradas no me preguntes nada salvo que estés bloqueado de verdad: las decisiones de profundidad y diseño son tuyas, guiadas por TEMARIO.md y CONTINUAR.md. Los laboratorios no pueden depender de mi hardware ni usar drones (CONTINUAR.md §7).

6. Si el contexto se hace largo, apóyate en ESTADO.md y en los commits para no perder el hilo: cada tema terminado debe quedar completo y subido antes de empezar el siguiente.
```
