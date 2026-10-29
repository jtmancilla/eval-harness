# Flujo de trabajo — MAYA fase 1

El flujo es completamente local y no requiere credenciales ni conectividad con MAYA.

```text
[1. Instalar] → [2. Generar] → [3. Revisar] → [4. Validar]
```

## 1. Instalar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Generar

```bash
python -m src.maya.generator --config configs/maya_matrix.yaml
```

La configuración define la semilla, el total de casos, los niveles de saturación y las rutas de salida. La ejecución inicial produce 480 casos y ocho ejemplos versionados.

## 3. Revisar

1. Abrir `data/maya/phase1/coverage.json` y comprobar:
   - `total_cases: 480`.
   - `scenario_count: 30`.
   - `uncovered_operation_ids: []`.
2. Revisar preguntas representativas en `data/fixtures/maya_phase1_sample.json`.
3. Abrir `data/maya/phase1/maya_matrix.csv` y validar las expectativas por caso.
4. Usar `case_id` para enlazar `cases.jsonl` con `expectations.jsonl`.

No entregar `expectations.jsonl` al sistema evaluado: contiene el ground truth.

## 4. Validar

```bash
pytest tests/ -v
mypy src/ tests/
ruff check .
```

La prueba extremo a extremo genera una matriz reducida en un directorio temporal y comprueba JSONL, CSV, cobertura, ejemplos, contratos, reproducibilidad, DAGs y bloqueos de seguridad.

## Registro de resultados

El CSV contiene columnas para documentar, sin modificar las expectativas originales:

- Respuesta y ruta observadas.
- Acciones, validaciones y escrituras observadas.
- Correctitud de ruta, selección, argumentos, orden, validaciones, persistencia y respuesta.
- Veredicto final y notas de revisión.

Una observación debe enlazarse siempre mediante `case_id`. La evaluación de lenguaje natural es semántica; los hechos requeridos no imponen una redacción literal.

## Fase 2

La ejecución contra API y el estrés masivo están fuera de este flujo. Una fase posterior deberá añadir un adaptador independiente que consuma únicamente los casos de entrada, registre las observaciones y nunca ejecute las herramientas declarativas del catálogo sintético.
