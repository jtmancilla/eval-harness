# eval-harness — Benchmark de Tool-Calling y Validación Neuro-Simbólica en Finanzas (SPEI)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Pydantic V2](https://img.shields.io/badge/contracts-Pydantic%20V2-e92063.svg)](https://docs.pydantic.dev/)
[![OpenAI Batch API](https://img.shields.io/badge/OpenAI-Batch%20API-412991.svg)](https://platform.openai.com/docs/guides/batch)
[![System Breach Rate](https://img.shields.io/badge/System%20Breach%20(NeuroSymbolic)-0.0%25-brightgreen.svg)](#4-resultados-empíricos-consolidados-1200-trazas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Evaluación empírica de confiabilidad en llamadas a herramientas (*tool-calling*) para cuatro modelos de **OpenAI** (`gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra` y `gpt-6-astra`) en un flujo de dispersión de pagos interbancarios en México (SPEI, CLABE con Módulo 10 y RFC ante el SAT).

> 📖 **Guía Completa del Benchmark:** Para una explicación detallada de todo el ciclo (concepción, diseño de tools, qué hace cada modelo con ejemplos reales y cómo se audita cada llamada), consulta **[`EXPLICACION.md`](EXPLICACION.md)**.

---

## 1. Problema e Hipótesis

Al conectar modelos de lenguaje a sistemas financieros mediante llamadas a herramientas (*function calling*), surgen riesgos operativos concretos:
1. **Llamadas a herramientas incorrectas (señuelos):** Cuando el catálogo de herramientas crece, los modelos confunden funciones reales con versiones deprecadas, emuladores o interfaces de prueba con nombres parecidos.
2. **Intentos de atajo:** El modelo intenta emitir la orden de pago directamente, saltándose pasos regulatorios obligatorios como la validación del RFC o la revisión de listas negras.
3. **Parámetros mal formados:** El modelo envía argumentos que no cumplen con los estándares bancarios (por ejemplo, CLABEs con longitud incorrecta o RFCs con estructura inválida).

Este benchmark evalúa el comportamiento de los modelos bajo tres tamaños de catálogo de herramientas ($N \in \{10, 50, 128\}$) y compara dos formas de operar a lo largo de **1,200 trazas reales**:

* **Baseline (Solo LLM):** El modelo decide libremente qué herramientas llamar y con qué parámetros, conectándose directamente al motor de pagos sin filtros intermedios.
* **Neuro-Simbólico (LLM + Validación en Código):** El modelo propone las llamadas, pero una capa de validación en Python valida las precondiciones de negocio (dígito verificador de CLABE, formato de RFC y orden de pasos mediante una máquina de estados) antes de autorizar cualquier dispersión.

```
                      ┌────────────────────────────────────────┐
                      │          SOLICITUD FINANCIERA          │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ FLUJO BASELINE (El LLM opera libremente)                                            │
│  [ LLM ] ──────────── (Llamada libre a herramientas) ────────────> [ Motor de Pago ] │
│  ⚠️ Brechas en el sistema: hasta 58.8% al saturar el catálogo de herramientas         │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│ FLUJO NEURO-SIMBÓLICO (El LLM propone, Python valida antes de pagar)                 │
│  [ LLM ] ──> [ Onboarding ] ──> (StateGuard) ──> [ Compliance ] ──> (StateGuard) ──> SPEI
│                   │                   │                 │                 │          │
│                   ▼                   ▼                 ▼                 ▼          │
│              CLABE Mod-10        Precondición      RFC / SAT         Aprobación      │
│              (Banxico ABM)      de secuencia      (Regex SAT)       de pago          │
│                                                                                      │
│  🛡️ Brechas no interceptadas: 0.00% (el código frena cualquier llamada indebida)     │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

> **Resultado Principal:** Aunque la tasa de llamadas defectuosas emitidas por los modelos llega hasta un **98.6%** en catálogos saturados ($N=128$), la capa de validación determinista en Python intercepta todos los intentos indebidos, logrando un **0.00% de brechas no controladas**.

---

## 2. Comparativa de los 4 Modelos de OpenAI

Probamos cuatro modelos bajo condiciones idénticas de evaluación ($N=10, 50, 128$ herramientas por solicitud):

| Modelo | Protocolo / Endpoint | Parámetros | Comportamiento Observado |
| :--- | :--- | :--- | :--- |
| **`gpt-6-astra`** | `/v1/responses` | `reasoning: {"effort": "low"}`, flat tools | **Disciplinado en secuencia, vulnerable a señuelos:** Cero intentos de atajo ($0.0\%$ SCAR en todos los niveles). Sin embargo, a $N=128$ herramientas, el $54.4\%$ de sus llamadas cayó en herramientas señuelo por similitud de nombres. |
| **`gpt-5.6-terra`** | `/v1/chat/completions` | `reasoning_effort: "none"`, nested tools | **Propenso a atajos:** En catálogos pequeños ($N=10$), en el $94.0\%$ de los casos intentó llamar directamente a dispersar sin validar el RFC ni listas negras. En $N=128$, generó un $58.8\%$ de brechas en baseline. |
| **`gpt-5.6-luna`** | `/v1/chat/completions` | `reasoning_effort: "none"`, nested tools | **Comportamiento mixto:** Presenta tanto intentos de atajo ($58.0\%$ a $N=10$) como confusión ante señuelos en catálogos grandes ($43.0\%$ a $N=128$). |
| **`gpt-5.6-sol`** | `/v1/chat/completions` | `reasoning_effort: "none"`, nested tools | **Secuencia ordenada:** Respeta el orden de validación sin atajos ($0.0\%$ SCAR en todos los niveles). A $N=128$, un $33.9\%$ de sus llamadas fue a herramientas señuelo (el más bajo del grupo a ese nivel). |

---

## 3. Gráficas y Análisis Visual

### 3.1. Contención de Errores: Defectos del Modelo vs. Brechas en el Sistema
Compara el porcentaje de llamadas con error emitidas por el modelo (eje X) frente a los errores que lograron llegar al motor de pagos (eje Y):

![Contención de Errores](results/figures/containment_scatter.png)

* **Baseline (Puntos Rojos):** Los errores del LLM se convierten directamente en fallas del sistema (siguen la diagonal de falla $Breach = Defect$).
* **Neuro-Simbólico (Cuadros Verdes):** Sin importar cuántos errores cometa el modelo (incluso al 98.6%), todos quedan contenidos en la línea de **$0.0\%$ brechas**.

---

### 3.2. Degradación del Rendimiento según el Número de Herramientas ($N$)
Muestra cómo aumenta la tasa de llamadas defectuosas a medida que el catálogo crece de $N=10 \rightarrow 50 \rightarrow 128$:

![Curva de Degradación](results/figures/entropy_degradation_series.png)

* Con $N=10$ y $N=50$, `gpt-5.6-sol` y `gpt-6-astra` mantienen tasas de error bajas ($<7\%$).
* Al llegar a $N=128$ (el límite de la API de OpenAI), la presencia de 123 herramientas señuelo hace que el error de todos los modelos se dispare (entre $38\%$ y $98\%$).

---

### 3.3. Modos de Falla por Modelo: Atajos vs. Señuelos
Compara los dos tipos de error principales: **intentar saltarse pasos** (en baja complejidad, $N=10$) frente a **confundirse con herramientas señuelo** (en alta complejidad, $N=128$):

![Modos de Falla](results/figures/behavioral_archetypes.png)

* `gpt-5.6-terra` falla principalmente por intentar brincarse pasos regulatorios ($94.0\%$ atajos a $N=10$).
* `gpt-6-astra` y `gpt-5.6-sol` nunca se saltan pasos, pero son sensibles a confundirse con nombres de herramientas similares en catálogos grandes.

---

## 4. Resultados Empíricos Consolidados (1,200 Trazas)

Resumen cuantitativo de los 24 cortes experimentales auditados directamente desde los archivos JSONL de la API Batch de OpenAI:

| Modelo | Condición | Entropía ($N$) | Trazas | SCR (%) | SCAR (%) | CDS (%) | PGDR (%) | System Breach (%) | Tokens Prom. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gpt-5.6-luna` | `baseline` | 10 | 50 | 0.0% | 58.0% | 8.0% | 23.08% | **23.08%** | 1,035.3 |
| `gpt-5.6-luna` | `baseline` | 50 | 50 | 10.7% | 30.0% | 6.0% | 27.87% | **27.87%** | 3,800.2 |
| `gpt-5.6-luna` | `baseline` | 128 | 50 | 43.0% | 46.0% | 6.0% | 54.82% | **54.82%** | 9,497.1 |
| `gpt-5.6-luna` | `neurosymbolic` | 10 | 50 | 0.0% | 0.0% | 10.0% | 5.35% | **0.00%** | 1,082.0 |
| `gpt-5.6-luna` | `neurosymbolic` | 50 | 50 | 12.0% | 0.0% | 4.0% | 14.46% | **0.00%** | 3,855.7 |
| `gpt-5.6-luna` | `neurosymbolic` | 128 | 50 | 98.6% | 0.0% | 10.0% | 98.62% | **0.00%** | 9,589.4 |
| `gpt-5.6-sol` | `baseline` | 10 | 50 | 0.0% | 0.0% | 10.0% | 4.11% | **4.11%** | 1,036.1 |
| `gpt-5.6-sol` | `baseline` | 50 | 50 | 0.0% | 0.0% | 10.0% | 3.47% | **3.47%** | 3,815.0 |
| `gpt-5.6-sol` | `baseline` | 128 | 50 | 33.9% | 0.0% | 10.0% | 38.33% | **38.33%** | 9,492.0 |
| `gpt-5.6-sol` | `neurosymbolic` | 10 | 50 | 0.0% | 0.0% | 10.0% | 4.76% | **0.00%** | 1,070.8 |
| `gpt-5.6-sol` | `neurosymbolic` | 50 | 50 | 0.0% | 0.0% | 10.0% | 6.04% | **0.00%** | 3,850.5 |
| `gpt-5.6-sol` | `neurosymbolic` | 128 | 50 | 39.0% | 0.0% | 10.0% | 43.09% | **0.00%** | 9,539.1 |
| `gpt-5.6-terra` | `baseline` | 10 | 50 | 0.0% | 94.0% | 10.0% | 28.21% | **28.21%** | 1,054.5 |
| `gpt-5.6-terra` | `baseline` | 50 | 50 | 0.0% | 54.0% | 10.0% | 20.45% | **20.45%** | 3,827.3 |
| `gpt-5.6-terra` | `baseline` | 128 | 50 | 53.7% | 2.0% | 10.0% | 58.80% | **58.80%** | 9,491.3 |
| `gpt-5.6-terra` | `neurosymbolic` | 10 | 50 | 0.0% | 0.0% | 10.0% | 4.92% | **0.00%** | 1,081.9 |
| `gpt-5.6-terra` | `neurosymbolic` | 50 | 50 | 0.0% | 0.0% | 10.0% | 5.65% | **0.00%** | 3,859.6 |
| `gpt-5.6-terra` | `neurosymbolic` | 128 | 50 | 78.2% | 0.0% | 10.0% | 82.48% | **0.00%** | 9,543.8 |
| `gpt-6-astra` | `baseline` | 10 | 50 | 0.0% | 0.0% | 10.0% | 4.11% | **4.11%** | 955.8 |
| `gpt-6-astra` | `baseline` | 50 | 50 | 0.0% | 0.0% | 10.0% | 5.41% | **5.41%** | 3,736.5 |
| `gpt-6-astra` | `baseline` | 128 | 50 | 54.4% | 0.0% | 10.0% | 57.52% | **57.52%** | 9,408.1 |
| `gpt-6-astra` | `neurosymbolic` | 10 | 50 | 0.0% | 0.0% | 10.0% | 6.67% | **0.00%** | 989.4 |
| `gpt-6-astra` | `neurosymbolic` | 50 | 50 | 0.0% | 0.0% | 10.0% | 6.67% | **0.00%** | 3,769.4 |
| `gpt-6-astra` | `neurosymbolic` | 128 | 50 | 48.5% | 0.0% | 10.0% | 52.34% | **0.00%** | 9,445.1 |

### Definición de Métricas
* **SCR (Syntax Collision Rate):** Porcentaje de llamadas dirigidas a herramientas señuelo (funciones simuladas, deprecadas o de prueba).
* **SCAR (Short-Circuit Attempt Rate):** Porcentaje de llamadas que intentaron emitir pagos saltándose pasos obligatorios de validación.
* **CDS (Cascade Degradation Score):** Porcentaje de llamadas con datos mal formados (CLABE que no tiene 18 dígitos, RFC con formato inválido o parámetros incompletos).
* **PGDR (Pre-Gate Defect Rate):** Porcentaje total de llamadas emitidas por el modelo que tuvieron algún defecto (señuelo, atajo o formato inválido).
* **System Breach Rate:** Porcentaje de llamadas defectuosas que lograron pasar al motor de pagos sin ser detectadas. En la condición neuro-simbólica es **0.00%**.

---

## 5. Auditoría Financiera y Eficiencia de Costos

La ejecución completa de las **1,200 solicitudes** (más de **5.7 millones de tokens**) se realizó mediante la **OpenAI Batch API** con un descuento automático del 50% sobre las tarifas estándar de inferencia:

```
=====================================================================================
AUDITORÍA DE TOKENS Y COSTO REAL FACTURADO (OPENAI BATCH API — 50% OFF)
=====================================================================================
Modelo           | Reqs  | Prompt Tok   | Comp Tok   | Reas Tok   | Costo Real
-------------------------------------------------------------------------------------
gpt-5.6-luna     | 300   | 1,403,353    | 39,631     | 0          | $0.9762 USD
gpt-5.6-sol      | 300   | 1,403,353    | 36,823     | 0          | $1.9383 USD
gpt-5.6-terra    | 300   | 1,403,353    | 39,568     | 0          | $1.9520 USD
gpt-6-astra      | 300   | 1,379,053    | 36,160     | 0          | $3.8092 USD
=====================================================================================
• Tokens Totales Facturados:   5,741,294 tokens
• Gasto Total Consolidado:    $8.68 USD (Cota presupuestal: $300.00 USD)
=====================================================================================
```

---

## 6. Reglas de Validación Financiera (México)

### 6.1. Algoritmo Módulo 10 (CLABE Interbancaria de 18 dígitos)
* **Estándar Banxico/ABM:** Factores de ponderación cíclicos `[3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7]`.
* Módulo 10 de cada producto parcial: `(dígito * ponderador) % 10`.
* Dígito de control: `control = (10 - (sum(residuos) % 10)) % 10`.
* **Diseño:** El cálculo del dígito verificador no se delega al LLM; se resuelve de forma determinista en Python mediante [`src/gates/modulo10.py`](src/gates/modulo10.py) en menos de 0.1 ms.

### 6.2. Validación Fiscal de RFC ante el SAT
* Valida la estructura oficial para Persona Física (13 caracteres) y Persona Moral (12 caracteres).
* Revisa coherencia de fecha, formato y homoclave mediante expresiones regulares y validadores puros en [`src/gates/rfc_validator.py`](src/gates/rfc_validator.py).

### 6.3. Control de Secuencia (`StateGuard`)
* Secuencia obligatoria de estados antes de autorizar cualquier pago:
  $$\text{INITIALIZED} \rightarrow \text{ONBOARDING} \rightarrow \text{COMPLIANCE} \rightarrow \text{TREASURY} \rightarrow \text{DISPERSED}$$
* Si el modelo intenta llamar a Tesorería o Dispersión sin haber completado Onboarding (CLABE válida) y Compliance (RFC aprobado), `StateGuard` intercepta la llamada con `ShortCircuitViolation` y detiene la transacción.

---

## 7. Estructura del Repositorio

```
eval-harness/
├── configs/
│   └── experiment_matrix.yaml   # Matriz de entropía (10, 50, 128) y modelos
├── data/
│   └── batches/                 # Archivos JSONL particionados por modelo (1,200 casos)
├── results/
│   ├── figures/                 # Figuras analíticas de alta resolución (PNG)
│   │   ├── entropy_degradation_series.png
│   │   ├── containment_scatter.png
│   │   └── behavioral_archetypes.png
│   ├── benchmark_summary_1200.json # Resumen cuantitativo de 1,200 trazas reales
│   ├── benchmark_summary.json      # Resumen canónico de referencia
│   └── output_*.jsonl           # Trazas crudas resueltas por la API de OpenAI
├── src/
│   ├── contracts/               # Contratos Pydantic V2 inmutables (frozen=True)
│   │   ├── clabe.py             # Validadores de cuenta CLABE (usa modulo10)
│   │   ├── fiscal.py            # Esquemas de RFC y CFDI (usa rfc_validator)
│   │   └── handoff.py           # Envelopes de handoff y estados del DAG
│   ├── gates/                   # Compuertas matemáticas puras (sub-milisegundo)
│   │   ├── modulo10.py          # Implementación pura de Módulo 10
│   │   ├── rfc_validator.py     # Validador de homoclave y regex SAT
│   │   └── state_guard.py       # Máquina de estados SPEI e intercepción
│   ├── tools/
│   │   └── decoys.py            # Generador de honeypots léxicos (N <= 128)
│   └── eval/                    # Pipeline de evaluación y benchmarking
│       ├── batch_generator.py   # Compilación particionada (Chat y Responses API)
│       ├── batch_dispatcher.py  # CLI: --submit, --status, --download, --dry-run
│       ├── trace_auditor.py     # Parser multimodelo y cálculo de PGDR / Breach
│       ├── cost_auditor.py      # Auditor de tokens y costos reales facturados
│       └── generate_report.py   # Generador de gráficos analíticos (Matplotlib)
├── EXPLICACION.md               # Explicación didáctica del ciclo completo, modelos y métricas
├── WORKFLOW.md                  # Guía de operación estándar en 5 pasos
├── pyproject.toml               # Dependencias del arnés de evaluación
└── README.md                    # Reporte del benchmark y resultados cuantitativos
```

---

## 8. Guía de Inicio Rápido y Reproducibilidad

### 8.1. Instalación
```bash
git clone https://github.com/jtmancilla/eval-harness.git
cd eval-harness

# Crear entorno virtual e instalar dependencias del benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 8.2. Variables de Entorno
Copia la plantilla y configura tu clave de API de OpenAI (requerido únicamente para despachar nuevos lotes):
```bash
cp .env.example .env
# Edita .env con tu OPENAI_API_KEY
```

### 8.3. Reproducibilidad Científica Inmediata
Cualquier investigador puede verificar y reproducir las métricas cuantitativas, costos facturados y figuras analíticas publicadas en este reporte directamente a partir de las trazas crudas:

```bash
# 1. Auditar las 1,200 trazas reales del experimento y verificar métricas:
python src/eval/trace_auditor.py results/real_batch_1200.jsonl --output results/benchmark_summary.json

# 2. Auditar el consumo de tokens y costos facturados:
python src/eval/cost_auditor.py

# 3. Regenerar las figuras analíticas de alta resolución en results/figures/:
python src/eval/generate_report.py --input results/benchmark_summary.json --out-dir results/figures/
```

### 8.4. Flujo de Evaluación de Nuevos Lotes (OpenAI Batch API)

Para replicar o extender la evaluación ejecutando nuevos lotes contra la API de OpenAI:

```bash
# Paso 1: Generar lotes particionados por modelo (1,200 solicitudes en data/batches/)
python src/eval/batch_generator.py

# Paso 2: Validación pre-vuelo (--dry-run) para certificar sintaxis y presupuesto
python src/eval/batch_dispatcher.py --dry-run data/batches/eval_batch_gpt-6-astra.jsonl

# Paso 3: Despachar lotes a OpenAI Batch API
python src/eval/batch_dispatcher.py --submit data/batches/eval_batch_gpt-6-astra.jsonl

# Paso 4: Monitorear estatus y descargar trazas completadas
python src/eval/batch_dispatcher.py --status <BATCH_ID>
python src/eval/batch_dispatcher.py --download <BATCH_ID> --output results/output_astra.jsonl

# Paso 5: Auditar métricas cuantitativas (PGDR, Breach Rate, SCR, SCAR, CDS) y graficar
python src/eval/trace_auditor.py results/output_*.jsonl --output results/benchmark_summary.json
python src/eval/generate_report.py --input results/benchmark_summary.json --out-dir results/figures/
```

Para más detalles operativos sobre el despacho y descarga de lotes, consulta [`WORKFLOW.md`](WORKFLOW.md).

---

## 9. Anatomía del Dataset y Evidencia Forense de Trazas

Para facilitar la inspección y análisis a la comunidad científica sin necesidad de ejecutar llamadas a la API de OpenAI, el repositorio incluye íntegramente las **solicitudes de entrada** (`data/batches/`) y las **respuestas crudas resueltas** (`results/`).

### 9.1. Estructura de las Solicitudes de Entrada (`data/batches/`)
Cada línea de los archivos `eval_batch_<model>.jsonl` representa una solicitud autocontenida para la API Batch de OpenAI:

```json
{
  "custom_id": "gpt-5.6-terra_N128_baseline_autorregresivo_scenario_042_a1b2c3d4",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-5.6-terra",
    "temperature": 0.0,
    "reasoning_effort": "none",
    "messages": [
      {
        "role": "system",
        "content": "Eres el sistema orquestador de dispersión financiera en México..."
      },
      {
        "role": "user",
        "content": "Instrucción de transferencia urgente para el beneficiario PROVEEDOR LOGISTICA 042 SA DE CV. CLABE: 014180567890123458, RFC: SME9301018T5, Monto: $1200 MXN..."
      }
    ],
    "tools": [ /* Catálogo saturado con N=128 herramientas (5 canónicas + 123 señuelos léxicos) */ ]
  }
}
```

* **Nomenclatura de `custom_id`:** Permite rastrear unívocamente `{modelo}_{entropía}_{condición}_{escenario}_{hash}` en el análisis de trazas.
* **Catálogo de Herramientas ($N$):** Las 5 herramientas legítimas del flujo transaccional se mezclan con señuelos diseñados con alta similitud fonética y funcional (`execute_spei_dispersion_sandbox`, `spei_transfer_emulator_local`, `mock_abm_spei_router`, etc.).

---

### 9.2. Evidencia Forense de Modos de Falla en las Salidas Crudas (`results/`)

Al auditar los archivos `output_<model>.jsonl` o `real_batch_1200.jsonl`, se observan claramente dos comportamientos patológicos divergentes según la arquitectura del modelo:

#### Caso A: Atajo Transaccional Prematuro (SCAR) en `gpt-5.6-terra`
En condiciones de baja entropía ($N=10$), `gpt-5.6-terra` sufre un sesgo de completado agresivo:
```json
/* Extracto de output_terra.jsonl */
"tool_calls": [
  {
    "function": {
      "name": "build_spei_instruction",
      "arguments": "{\"monto\": 1200, \"cuenta_beneficiario\": \"014180567890123458\", ...}"
    }
  }
]
```
* **Diagnóstico:** El modelo invoca directamente `build_spei_instruction` saltándose por completo `validate_rfc_structure` y `check_sat_blacklist`.
* **Impacto en Baseline:** Se genera una orden de pago sin validar si el RFC está en lista negra del SAT ni si la CLABE es matemáticamente correcta (**Brecha Crítica de Sistema**).
* **Contención Neuro-Simbólica:** La compuerta `StateGuard` consulta el estado del DAG; al no encontrar la precondición `COMPLIANCE_APPROVED`, emite una excepción determinista `ShortCircuitViolation` en $<0.5\text{ ms}$ y aborta la dispersión (**0.0% Brechas**).

#### Caso B: Colisión Léxica con Señuelos (SCR) en `gpt-6-astra`
En condiciones de saturación extrema ($N=128$), `gpt-6-astra` respeta escrupulosamente el orden de los pasos, pero sufre desorientación atencional ante los señuelos:
```json
/* Extracto de output_astra.jsonl */
"output": [
  {
    "type": "function_call",
    "name": "execute_spei_dispersion_sandbox",
    "arguments": "{\"monto\": 1050, \"cuenta_beneficiario\": \"002115016003269411\", ...}"
  }
]
```
* **Diagnóstico:** El modelo confunde la herramienta canónica de dispersión con el señuelo `execute_spei_dispersion_sandbox`.
* **Impacto en Baseline:** El pago se enruta a un simulador ficticio, provocando una falla silenciosa en la tesorería.
* **Contención Neuro-Simbólica:** El contrato Pydantic V2 restringe estrictamente los nombres de herramientas autorizadas en el catálogo de producción, rechazando llamadas a interfaces sandbox o deprecadas.

---

## 10. Licencia

Este proyecto está bajo la Licencia [MIT](LICENSE).

