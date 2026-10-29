# MAYA Evaluation Harness — Fase 1

Generador determinista de una matriz sintética para evaluar el comportamiento conversacional y transaccional de **MAYA, el asistente de Banorte**.

La fase 1 produce casos de entrada y sus resultados esperados. No llama a MAYA ni a proveedores externos, no necesita credenciales y no utiliza datos personales reales.

## Qué evalúa

Cada caso permite contrastar posteriormente:

- **Enrutamiento:** RAG para información bancaria, IBM Assistant para operaciones o rechazo fuera de alcance.
- **Selección de acciones:** operación y herramienta correctas, con argumentos sintéticos esperados.
- **Orden de ejecución:** dependencias explícitas para flujos que deben ejecutarse uno a uno.
- **Validaciones:** autenticación, titularidad, fondos, dispositivo, token y requisitos de cada operación.
- **Persistencia:** escrituras esperadas en bases y logs, además de escrituras prohibidas cuando una operación debe bloquearse.
- **Respuesta final:** hechos obligatorios, contenido prohibido, aclaraciones y escalamiento humano.

El CSV incluye columnas vacías para registrar las respuestas observadas y evaluar cada dimensión por separado.

## Cobertura inicial

La configuración genera **480 casos: 30 escenarios × 16 personalidades**.

- Las **17 operaciones bancarias** descritas para MAYA.
- Consultas informativas RAG, plática genérica y preguntas fuera de alcance.
- Solicitudes ambiguas que requieren aclaración.
- Cuatro secuencias multiacción con dependencias.
- Inyección de instrucciones, omisión de autenticación y suplantación de canal.
- Diez estilos narrativos en español de México.
- Regiones, edades, canales, experiencia digital y urgencias diversas.
- Catálogos de 18, 36 y 72 herramientas, incluidos señuelos no ejecutables.

Los identificadores como `SYN-NOMINA-001` o `SYN-CUSTOMER-*` son exclusivamente sintéticos.

## Estructura

```text
eval-harness/
├── configs/maya_matrix.yaml
├── data/
│   ├── fixtures/maya_phase1_sample.json
│   └── maya/phase1/                 # salidas generadas, ignoradas por Git
├── src/
│   └── maya/
│       ├── contracts.py             # casos, expectativas y observaciones
│       ├── personas.py              # perfiles sintéticos
│       ├── narratives.py            # estilos de preguntas
│       ├── tools.py                 # herramientas y señuelos declarativos
│       ├── scenarios.py             # escenarios y DAGs de ejecución
│       └── generator.py             # CLI y exportación
├── tests/test_maya_matrix.py
├── AGENTS.md
├── WORKFLOW.md
└── pyproject.toml
```

## Instalación

Requiere Python 3.11 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Las únicas dependencias de ejecución son **Pydantic** y **PyYAML**.

## Generar la matriz

```bash
python -m src.maya.generator --config configs/maya_matrix.yaml
```

La semilla configurada hace que la generación sea reproducible. Para una muestra reducida:

```bash
python -m src.maya.generator \
  --config configs/maya_matrix.yaml \
  --total-cases 30 \
  --output-dir data/maya/sample \
  --sample-output data/fixtures/maya_sample_30.json
```

### Archivos generados

| Archivo | Propósito |
| :--- | :--- |
| `data/maya/phase1/cases.jsonl` | Entradas para un futuro adaptador, sin respuestas esperadas. |
| `data/maya/phase1/expectations.jsonl` | Ground truth separado por `case_id`. |
| `data/maya/phase1/maya_matrix.csv` | Matriz de 40 columnas para revisión y registro de observaciones. |
| `data/maya/phase1/coverage.json` | Distribuciones y cobertura de las 17 operaciones. |
| `data/fixtures/maya_phase1_sample.json` | Ocho ejemplos representativos versionados. |

## Validación

```bash
pytest tests/ -v
mypy src/ tests/
ruff check .
```

Consulta [`WORKFLOW.md`](WORKFLOW.md) para el flujo operativo y [`AGENTS.md`](AGENTS.md) para las reglas de diseño.

## Fuera de alcance de fase 1

No se incluyen llamadas HTTP, adaptadores de IBM Assistant o Google Model Armor, credenciales, concurrencia, pruebas de carga ni ejecución real de operaciones bancarias. Esos componentes pertenecen a una fase posterior y deben consumir `cases.jsonl` sin exponer `expectations.jsonl` al sistema evaluado.
