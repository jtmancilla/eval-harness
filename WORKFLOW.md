# WORKFLOW.md — Guía de Operación del Arnés de Evaluación

Procedimiento estándar de 5 pasos para compilar, validar con dry-run, despachar, monitorear y auditar lotes en OpenAI Batch API.

```
[1. Generar Lotes] 
       ↓
[2. Validación Pre-vuelo (--dry-run)] 
       ↓
[3. Despacho Batch API] 
       ↓
[4. Monitoreo y Descarga] 
       ↓
[5. Auditoría y Visualizaciones]
```

---

## Paso 1: Generación de Lotes Particionados
Compila los archivos JSONL asegurando un archivo independiente por modelo y respetando $N \in \{10, 50, 128\}$:
```bash
python src/eval/batch_generator.py
```
*Archivos generados en `data/batches/`:*
* `eval_batch_gpt-5.6-luna.jsonl` (300 solicitudes, /v1/chat/completions)
* `eval_batch_gpt-5.6-sol.jsonl` (300 solicitudes, /v1/chat/completions)
* `eval_batch_gpt-5.6-terra.jsonl` (300 solicitudes, /v1/chat/completions)
* `eval_batch_gpt-6-astra.jsonl` (300 solicitudes, /v1/responses)

---

## Paso 2: Validación Pre-vuelo (--dry-run)
Valida la sintaxis de las solicitudes, el tope de herramientas ($N \le 128$) y la gobernanza presupuestal (< $300 USD) antes de incurrir en costos en la cola batch:
```bash
# Inspección previa y cálculo de presupuesto:
python src/eval/batch_dispatcher.py --dry-run data/batches/eval_batch_gpt-6-astra.jsonl
```

---

## Paso 3: Despacho a OpenAI Batch API
Envía cada archivo particionado de forma secuencial y registra los `Batch ID` emitidos:
```bash
# Envío formal a la cola Batch:
for file in data/batches/eval_batch_*.jsonl; do
    echo "--- Despachando $file ---"
    python src/eval/batch_dispatcher.py --submit "$file"
done
```

---

## Paso 4: Monitoreo de Estatus y Descarga de Resultados
Consulta el progreso de los lotes encolados hasta que alcancen el estado `completed`:
```bash
python src/eval/batch_dispatcher.py --status <BATCH_ID>
```

Descarga las respuestas directamente a `results/` utilizando los flags del despachador:
```bash
# Ejemplo para la tríada gpt-5.6 / gpt-6-astra:
python src/eval/batch_dispatcher.py --download <BATCH_ID_LUNA> --output results/output_luna.jsonl
python src/eval/batch_dispatcher.py --download <BATCH_ID_SOL> --output results/output_sol.jsonl
python src/eval/batch_dispatcher.py --download <BATCH_ID_TERRA> --output results/output_terra.jsonl
python src/eval/batch_dispatcher.py --download <BATCH_ID_ASTRA> --output results/output_astra.jsonl
```

Verifica la integridad de las líneas por archivo (300 por archivo):
```bash
wc -l results/output_*.jsonl
```

---

## Paso 5: Auditoría Cuantitativa y Generación de Gráficas
Ejecuta el auditor sobre los archivos descargados para calcular **PGDR** (Pre-Gate Defect Rate), confirmar el **0.0%** de brechas en el sistema (**System Breach**) y generar las figuras analíticas:

```bash
# 1. Auditoría cuantitativa (genera el resumen canónico):
python src/eval/trace_auditor.py results/output_*.jsonl --output results/benchmark_summary.json

# 2. Auditoría de costos y tokens facturados:
python src/eval/cost_auditor.py

# 3. Generación de figuras analíticas de alta resolución:
python src/eval/generate_report.py --input results/benchmark_summary.json --out-dir results/figures/
```

