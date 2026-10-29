# AGENTS.md — Gobierno del arnés MAYA

**Repositorio:** `eval-harness`
**Alcance actual:** fase 1, generación local de la matriz sintética de evaluación de MAYA
**Entorno:** Python 3.11+ | Pydantic V2 | PyYAML

## 1. Propósito

El repositorio genera casos sintéticos para evaluar el enrutamiento, la ejecución secuencial, las validaciones, la persistencia y la respuesta final de MAYA. La arquitectura funcional considera tres salidas:

- `RAG` para consultas informativas bancarias.
- `IBM_ASSISTANT` para consultas patrimoniales y operaciones.
- `OUT_OF_SCOPE` para plática o temas no bancarios.

La fase 1 no ejecuta MAYA ni operaciones bancarias. No contiene clientes HTTP, adaptadores propietarios, concurrencia ni pruebas de carga.

## 2. Invariantes

1. **Datos exclusivamente sintéticos.** No incorporar datos personales, cuentas, credenciales, tokens o movimientos reales. Los identificadores sintéticos usan el prefijo `SYN-`.
2. **Cero red y cero secretos.** La generación y las pruebas no requieren conectividad, archivos `.env` ni claves de proveedores.
3. **Contratos estrictos.** Los modelos Pydantic usan `extra="forbid"` y `frozen=True`. Todo módulo Python inicia con `from __future__ import annotations` y evita `typing.Any` salvo justificación explícita.
4. **Separación del ground truth.** `cases.jsonl` no contiene rutas, acciones ni respuestas esperadas. `expectations.jsonl` se mantiene separado y nunca se entrega al sistema evaluado.
5. **Ejecución uno a uno.** Las acciones compuestas se representan como un DAG acíclico mediante `depends_on`; las acciones se almacenan en orden topológico.
6. **Bloqueo ante validación fallida.** Una validación fallida impide ejecutar la acción y sus dependientes. Las operaciones rechazadas declaran escrituras prohibidas.
7. **Herramientas declarativas.** El catálogo y sus señuelos no son ejecutables ni están acoplados a OpenAI, IBM o Google.
8. **Respuesta semántica.** Las expectativas de lenguaje natural describen hechos obligatorios y contenido prohibido, no una redacción literal.
9. **Generación reproducible.** Usar RNG local sembrado e IDs estables; no introducir reloj, UUID aleatorios ni servicios generativos en el compilador.

## 3. Cobertura de referencia

La configuración canónica produce 480 casos: 30 escenarios por 16 personalidades. Debe conservar:

- Las 17 operaciones bancarias.
- Consultas RAG, fuera de alcance y aclaraciones.
- Flujos multiacción y escenarios adversariales.
- Diez estilos narrativos y diversidad regional, etaria y de urgencia.
- Niveles de saturación configurables.
- Columnas independientes para ruta, selección, argumentos, orden, validaciones, persistencia, respuesta y veredicto.

Al modificar escenarios o contratos, actualizar las pruebas y el reporte de cobertura correspondiente.

## 4. Estructura canónica

```text
eval-harness/
├── configs/maya_matrix.yaml
├── data/
│   ├── fixtures/maya_phase1_sample.json
│   └── maya/phase1/                 # salidas generadas, ignoradas
├── src/
│   ├── __init__.py
│   └── maya/
│       ├── __init__.py
│       ├── contracts.py
│       ├── personas.py
│       ├── narratives.py
│       ├── tools.py
│       ├── scenarios.py
│       └── generator.py
├── tests/
│   ├── __init__.py
│   └── test_maya_matrix.py
├── .gitignore
├── AGENTS.md
├── README.md
├── WORKFLOW.md
└── pyproject.toml
```

No reintroducir módulos de benchmarks ajenos en esta rama. Una fase 2 debe implementarse como un adaptador separado que consuma casos y produzca observaciones, sin acoplar el generador a una API.

## 5. Operación

Generación canónica:

```bash
python -m src.maya.generator --config configs/maya_matrix.yaml
```

Artefactos:

- `cases.jsonl`: entradas aisladas.
- `expectations.jsonl`: resultados esperados.
- `maya_matrix.csv`: revisión y enriquecimiento con observaciones.
- `coverage.json`: cobertura y distribuciones.
- `maya_phase1_sample.json`: ejemplos versionados.

## 6. Calidad y Git

Antes de entregar cambios:

```bash
pytest tests/ -v
mypy src/ tests/
ruff check .
```

También ejecutar el generador real y comprobar 480 casos, 30 escenarios, 17 operaciones cubiertas y 480 filas lógicas en el CSV.

No registrar credenciales ni salidas generadas. Mantener `.env`, entornos virtuales, cachés, JSONL y `data/maya/phase1/` fuera de Git. No mezclar cambios de otras líneas de evaluación en esta rama.
