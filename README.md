# Curso intensivo de robótica inteligente

Curso autodidacta para empezar al día un doctorado en IA para drones pensado para sim-to-real: cómo se controlan hoy los robots con IA (de PID y MPC a VLAs y world models), cómo se simulan para entrenar y validar, y con qué hardware y comunicaciones se despliegan.

- **45 presentaciones HTML** en 48 jornadas, repartidas en 7 fases.
- Cada tema lleva **laboratorio**, **test de autoevaluación** y **lecturas guiadas** con referencias verificadas.
- Un **proyecto hilo** en JAX (simulador y pipeline de dron) crece con los laboratorios.

## Ver el curso

Abre [`index.html`](index.html) en el navegador: enlaza todos los temas disponibles. Las fuentes y las fórmulas se cargan de internet.

Si un navegador se niega a cargar los estilos desde archivos locales, sirve la carpeta:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

y abre `http://localhost:8765`.

Dentro de una presentación: flechas o espacio para avanzar, `L` para el modo lectura, `I` para el índice y `T` para cambiar entre tema claro y oscuro.

## Estado

Ver [`00_plan/ESTADO.md`](00_plan/ESTADO.md). Hechos: **M0** (mapa del campo), **M1** (herramientas matemáticas), **A1** (dinámica y modelado), **A2** (control clásico y jerarquía de lazos), **A3a** (control óptimo), **A3b** (MPC por muestreo y aprendizaje), **A4a** (filtrado bayesiano), **A4b** (percepción para estimación), **A5** (planificación y navegación) y el hito **H1** con el examen de la fase 1.

## Estructura

```
00_plan/        análisis de la petición, diagnóstico, temario, estado y prompt de continuación
assets/         motor de presentaciones compartido (CSS y JS)
temas/<ID>_*/   una presentación por tema (index.html), su laboratorio (lab/README.md) y, desde A1, el script de figuras (figuras/)
index.html      índice del curso
CONTINUAR.md    manual para producir los temas que faltan
CLAUDE.md       reglas para Claude Code
```

## Continuar el trabajo

1. Clona el repositorio y abre Claude Code en su carpeta.
2. Pega el prompt de [`00_plan/PROMPT_CONTINUAR.md`](00_plan/PROMPT_CONTINUAR.md).
3. Claude produce los temas en orden y se detiene en cada punto de revisión (🛑 en `ESTADO.md`). Revisa, responde "sigue" o pide cambios.

El manual completo del flujo de trabajo, los componentes del motor y las comprobaciones de calidad está en [`CONTINUAR.md`](CONTINUAR.md).
