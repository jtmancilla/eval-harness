# WORKFLOW.md — Guía de Operación del Arnés de Evaluación

Procedimiento estándar de 6 pasos para compilar, validar, despachar, monitorear y auditar lotes en OpenAI Batch API.

```
[1. Generar Lotes] 
       ↓
[2. Smoke Test (HTTP 200)] 
       ↓
[3. Despacho Batch API] 
       ↓
[4. Monitoreo de Estatus] 
       ↓
[5. Descarga Vía CLI] 
       ↓
[6. Auditoría Cuantitativa]
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

## Paso 2: Smoke Test Pre-vuelo (Obligatorio)
Valida una solicitud real con $N=128$ herramientas contra la API en vivo para certificar respuesta HTTP 200 antes de incurrir en costos o bloqueos en la cola batch:
```bash
python tests/smoke_astra_responses.py
# O su alias canónico:
python tests/smoke_batch_payload.py
```

---

## Paso 3: Despacho a OpenAI Batch API
Inspecciona con `--dry-run` o envía cada archivo particionado de forma secuencial y registra los `Batch ID` emitidos:
```bash
# Inspección previa (--dry-run):
python src/eval/batch_dispatcher.py --dry-run data/batches/eval_batch_gpt-6-astra.jsonl

# Envío formal a la cola Batch:
for file in data/batches/eval_batch_*.jsonl; do
    echo "--- Despachando $file ---"
    python src/eval/batch_dispatcher.py --submit "$file"
done
```

---

## Paso 4: Monitoreo de Estatus
Consulta el progreso de los lotes encolados hasta que alcancen el estado `completed`:
```bash
python src/eval/batch_dispatcher.py --status <BATCH_ID>
```

---

## Paso 5: Descarga Formal de Resultados vía CLI
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

## Paso 6: Auditoría Cuantitativa y Extracción de Métricas
Ejecuta el auditor sobre los archivos descargados para calcular **PGDR** (Pre-Gate Defect Rate), confirmar el **0.0%** de brechas en el sistema (**System Breach**) y evaluar **SCR**, **SCAR**, **CDS** y **CTO**:

```bash
# Opción A: Paso directo de múltiples archivos o comodín glob
python src/eval/trace_auditor.py results/output_*.jsonl --output results/benchmark_summary.json

# Opción B: Concatenando en un archivo consolidado
cat results/output_*.jsonl > results/consolidated_traces.jsonl
python src/eval/trace_auditor.py results/consolidated_traces.jsonl --output results/benchmark_summary.json
```
