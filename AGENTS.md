# AGENTS.md — System Manifest & Multi-Agent Governance Framework
**Repository:** `eval-harness` (`git@github.com:jtmancilla/eval-harness.git`)  
**Environment:** Python 3.12 | Pydantic V2 | OpenAI Batch API | macOS  
**Core Domain:** Mexican Regulated Financial Systems (SPEI, CLABE Módulo 10, RFC/CFDI)

---

## 1. Misión del Proyecto y Diseño Experimental

`eval-harness` es un arnés de evaluación empírica de misión crítica diseñado para contrastar dos paradigmas de ejecución en flujos financieros regulados en México:
1. **Baseline (Tool-Calling Autorregresivo Abierto):** Orquestación estándar de llamadas a herramientas donde el modelo LLM decide libremente la secuencia, precedencia y argumentos sin compuertas intermedias.
2. **Neuro-Simbólico (Arquitectura con Compuertas Deterministas):** Supervisión estricta mediante contratos Pydantic V2 inmutables, compuertas simbólicas puras (`src/gates/`) y una máquina de estados finitos acíclica (`StateGuard`) que valida precondiciones antes de cualquier llamada a motor o dispersor.

### 1.1. Hipótesis Central
Bajo catálogos densos de herramientas ($N \in \{10, 50, 128\}$) saturados de colisión léxica y señuelos (*honeypots*), los modelos autorregresivos sufren sobreconfianza catastrófica (*Catastrophic Tool Over-reliance*), intentos de atajo (*short-circuiting*) y degradación de esquemas. La arquitectura neuro-simbólica garantiza una tasa de brecha de sistema (*System Breach Rate*) de **0.0%** sin importar la tasa intrínseca de defectos del modelo (*Pre-Gate Defect Rate*).

```
[ Usuario / Webhook ]
         │
         ▼
┌──────────────────┐       HandoffEnvelope       ┌──────────────────┐
│  Capa Neuronal   │ ──────────────────────────> │  Capa Simbólica  │
│ (Modelos LLM)    │                             │ (Gates / Python) │
│ - Inferencia     │ <────────────────────────── │ - Determinismo   │
│ - Parsing texto  │      Validación / Rechazo   │ - Pydantic V2    │
│ - Intenciones    │                             │ - StateGuard DAG │
└──────────────────┘                             └──────────────────┘
                                                          │
                                                [ Dispersión SPEI ]
```

### 1.2. Invariantes Innegociables
1. **Axioma de Aritmética Cero en LLMs:** Ningún modelo de lenguaje debe computar ponderaciones, algoritmos de módulo, sumas de verificación ni validaciones de expresiones regulares en prompts o chains-of-thought. Cualquier cálculo de integridad se delega exclusivamente a `src/gates/`.
2. **Inmutabilidad de Contratos:** Toda comunicación inter-agente viaja dentro de instancias de `HandoffEnvelope[T]` con `model_config = ConfigDict(extra='forbid', frozen=True)`. No se toleran payloads en texto libre, diccionarios genéricos ni llaves dinámicas no declaradas.
3. **Cero Bypass Transaccional (*Short-Circuit Prevention*):** Ninguna instrucción puede invocar el tool-calling de Tesorería ni la compuerta de dispersión SPEI si la traza no contiene la firma de estado `COMPLIANCE_APPROVED` registrada en el grafo.
4. **Fallo Temprano Local:** Las violaciones sintácticas, colisiones de herramientas o discrepancias de esquema deben abortar la ejecución en código local determinista sin generar reintentos deliberativos costosos en el LLM.

---

## 2. Invariantes del Dominio Financiero (México)

Cualquier módulo que opere sobre `src/contracts/` o `src/gates/` debe preservar estrictamente estas especificaciones:

### 2.1. CLABE Interbancaria (18 dígitos)
* **Estructura Oficial Banxico/ABM:** 3 dígitos de código de banco + 3 dígitos de plaza/sucursal + 11 dígitos de cuenta + 1 dígito de control.
* **Algoritmo Módulo 10 Ponderado:**
  * Factores de ponderación cíclicos: `[3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7]`.
  * Multiplicación elemento a elemento, extrayendo el residuo módulo 10 de cada producto parcial: `(dígito * ponderador) % 10`.
  * Suma total de los residuos módulo 10: `suma = sum(residuos)`.
  * Dígito verificador calculado como: `control = (10 - (suma % 10)) % 10`.
* **Regla Inquebrantable:** Ninguna cuenta con CLABE que no satisfaga `src/gates/modulo10.py` puede avanzar en el flujo transaccional.

### 2.2. RFC (Registro Federal de Contribuyentes)
* **Persona Física:** 13 caracteres alfanuméricos (`4 letras + 6 dígitos AAMMDD + 3 homoclave`).
* **Persona Moral:** 12 caracteres alfanuméricos (`3 letras + 6 dígitos AAMMDD + 3 homoclave`).
* **Validación SAT:** Expresión regular canónica SAT y coherencia básica de calendario en `src/gates/rfc_validator.py`.

### 2.3. Máquina de Estados Finitos (`StateGuard`)
El plano de control mantiene la traza en un DAG inmutable con transiciones formalmente autorizadas:

| Estado Actual | Evento Disparador | Compuerta Simbólica | Estado Siguiente |
| :--- | :--- | :--- | :--- |
| `INITIALIZED` | Recepción de solicitud | Integridad estructural del mensaje | `ONBOARDING_PENDING` |
| `ONBOARDING_PENDING` | Validación de cuenta | `validate_clabe(clabe) == True` | `ONBOARDING_COMPLETED` |
| `ONBOARDING_COMPLETED`| Despacho a Compliance | Verificación de firma y hash anterior | `COMPLIANCE_PENDING` |
| `COMPLIANCE_PENDING` | Aprobación fiscal | `validate_rfc(rfc) == True` | `COMPLIANCE_APPROVED` |
| `COMPLIANCE_PENDING` | Rechazo por riesgo | Registro de causal en auditoría | `COMPLIANCE_REJECTED` |
| `COMPLIANCE_APPROVED` | Despacho a Tesorería | Verificación estricta de no-bypass | `TREASURY_PENDING` |
| `TREASURY_PENDING` | Despacho de pago | Verificación de fondos y clave de rastreo | `DISPERSED` |

Cualquier salto directo no registrado (e.g., llamar directamente a Tesorería sin Compliance) levanta `ShortCircuitViolation` y aborta el proceso de inmediato.

---

## 3. Restricciones e Invariantes de OpenAI Batch API

A partir de auditorías empíricas en producción y restricciones de la API de OpenAI, se aplican las siguientes reglas duras en `src/eval/`:

1. **Tope de Herramientas ($N \le 128$):**
   * OpenAI impone un límite absoluto de **128 herramientas** en el parámetro `tools`.
   * La matriz de entropía queda formalizada en: **$N \in \{10, 50, 128\}$**. Queda prohibido generar catálogos con $N > 128$.

2. **Homogeneidad de Modelo por Archivo Batch:**
   * OpenAI Batch API rechaza lotes que contengan modelos heterogéneos.
   * Cada archivo `.jsonl` debe contener única y exclusivamente peticiones para un solo modelo.
   * Nomenclatura obligatoria: `data/batches/eval_batch_<model>.jsonl` (300 casos por archivo, 1,200 en total).

3. **Bifurcación Dinámica de Protocolo según Modelo:**
   * **Familia `gpt-5.6` (`luna`, `sol`, `terra`):**
     * Endpoint: `url: "/v1/chat/completions"`.
     * Parámetro obligatorio con herramientas: `"reasoning_effort": "none"`.
     * Formato de tools: Esquema anidado estándar (`{"type": "function", "function": {...}}`).
   * **Modelos de Razonamiento Puro (`gpt-6-astra`):**
     * Endpoint: `url: "/v1/responses"`.
     * No soporta `reasoning_effort: "none"`. Requiere `"reasoning": {"effort": "low"}`.
     * Formato de tools: Esquema plano sin anidar (`{"type": "function", "name": ..., "description": ..., "parameters": ...}`).

4. **Pre-flight Smoke Test Obligatorio:**
   * Antes de despachar un lote asíncrono, debe validarse la conectividad y compatibilidad mediante una llamada en vivo con $N=128$ (`tests/smoke_astra_responses.py`) certificando respuesta HTTP 200.

5. **Gobernanza Presupuestal:**
   * Cota máxima de **$300.00 USD** por corrida completa de evaluación, verificada automáticamente en `src/eval/batch_dispatcher.py`.

---

## 4. Métricas de Evaluación y Fórmulas Matemáticas

El analizador `src/eval/trace_auditor.py` procesa los archivos `.jsonl` de salida y reporta el comportamiento en dos capas críticas:

### 4.1. Pre-Gate Defect Rate (PGDR) — Calidad Intrínseca del LLM
Mide el porcentaje de intenciones brutas emitidas por el modelo que requirieron intercepción por anomalías sintácticas o normativas:

$$\text{PGDR} = \frac{\text{Defectos Pre-Gate}}{\text{Total de intenciones emitidas}} = \frac{\text{SCR} + \text{SCAR} + \text{CDS}}{\text{Total de llamadas / intenciones}}$$

* **SCR (Syntax Collision Rate):** Porcentaje de tool-calls dirigidas a herramientas señuelo (*honeypots* léxicos o de prefijo).
* **SCAR (Short-Circuit Attempt Rate):** Porcentaje de trazas que intentan transferir fondos o saltar a Tesorería sin autorización previa de Compliance.
* **CDS (Cascade Degradation Score):** Porcentaje de fallos por corrupción de contratos Pydantic V2 (CLABE con dígito alterado, truncamiento de ceros a la izquierda, RFC malformado).

### 4.2. Post-Gate System Breach Rate — Resiliencia Arquitectural
Mide las violaciones que superaron las defensas y alcanzaron el motor transaccional o la base de datos:

$$\text{System Breach Rate} = \frac{\text{Transacciones indebidas no interceptadas}}{\text{Total de intenciones emitidas}}$$

* En la condición **Neuro-Simbólica (`neurosymbolic_handoff`)**, la compuerta determinista garantiza estrictamente **0.0%**.
* En la condición **Baseline (`baseline_autorregresivo`)**, las violaciones se materializan directamente como brechas en el sistema (alcanzando hasta 58.8% en condiciones de alta entropía).

### 4.3. Compute Token Overhead (CTO)
Comparativa del gasto de tokens entre la resolución deliberativa frente a la intercepción simbólica inmediata:
$$\text{CTO Delta} = \text{Tokens}_{\text{neurosymbolic}} - \text{Tokens}_{\text{baseline}}$$

---

## 5. Estructura Canónica del Repositorio

```
eval-harness/
├── configs/
│   └── experiment_matrix.yaml   # Matriz de entropía (10, 50, 128) y modelos
├── data/
│   ├── batches/                 # JSONL de entrada particionados por modelo (Git-ignored)
│   └── fixtures/                # Muestras mínimas para pruebas unitarias
├── results/                     # JSONL descargados y resúmenes JSON (Git-ignored)
│   ├── figures/                 # Figuras analíticas de alta resolución (PNG)
│   ├── benchmark_summary_1200.json # Resumen cuantitativo de 1,200 trazas reales
│   └── benchmark_summary.json   # Resumen canónico de referencia
├── src/
│   ├── contracts/               # Contratos Pydantic V2 inmutables (frozen=True)
│   │   ├── clabe.py             # Tipos y validadores de cuenta CLABE
│   │   ├── fiscal.py            # Tipos y esquemas de RFC y CFDI
│   │   └── handoff.py           # Envelopes de handoff y estados del DAG
│   ├── gates/                   # Compuertas puras y deterministas (sub-milisegundo)
│   │   ├── modulo10.py          # Implementación pura de algoritmo Módulo 10
│   │   ├── rfc_validator.py     # Validador de homoclave y regex SAT
│   │   └── state_guard.py       # Máquina de estados SPEI e intercepción
│   ├── tools/
│   │   └── decoys.py            # Generador de honeypots léxicos (N <= 128)
│   └── eval/                    # Pipeline de evaluación y benchmarking
│       ├── batch_generator.py   # Compilación particionada (Chat y Responses API)
│       ├── batch_dispatcher.py  # CLI: --submit, --status, --download, --dry-run
│       ├── trace_auditor.py     # Parser multimodelo, métricas PGDR y System Breach
│       ├── cost_auditor.py      # Auditor de tokens y costos reales facturados
│       └── generate_report.py   # Generador de gráficos analíticos (Matplotlib)
├── tests/
│   ├── unit/                    # Pruebas unitarias de compuertas, contratos y eval
│   ├── smoke_astra_responses.py # Smoke test en vivo con gpt-6-astra y N=128
│   └── smoke_batch_payload.py   # Alias canónico de ejecución de smoke test
├── .env                         # Credenciales (OPENAI_API_KEY) — PROHIBIDO EN GIT
├── .env.example                 # Plantilla de variables de entorno requeridas
├── .gitignore                   # Blindaje estricto de secretos, caches y datasets
├── AGENTS.md                    # Manifiesto y marco de gobernanza
├── WORKFLOW.md                  # Guía de operación estándar en 6 pasos
├── README.md                    # Presentación del benchmark y comparativa de modelos
└── pyproject.toml               # Configuración de dependencias, ruff y pytest
```

---

## 6. Reglas de Conducta para Agentes Autónomos (Antigravity Code)

1. **Cero Hardcoding (`NO HARCODES`):**
   * Toda nueva funcionalidad debe integrarse en los módulos canónicos con argumentos formales de CLI (`argparse`) o tipado dinámico.
   * Prohibido insertar strings fijos de endpoints o condicionales ad-hoc no declarados.
2. **Tipado Estricto:**
   * Todo módulo de Python debe iniciar con `from __future__ import annotations`.
   * Prohibido el uso de `typing.Any` sin justificación explícita. Debe pasar `mypy` sin errores.
3. **Pureza Simbólica de las Compuertas:**
   * El paquete `src/gates/` no debe importar librerías de modelos, ni dependencias asíncronas pesadas, ni clientes HTTP. Debe permanecer compuesto exclusivamente por funciones puras con tiempo de respuesta sub-milisegundo.
4. **Seguridad y Git Hygiene:**
   * Jamás registrar claves (`OPENAI_API_KEY`) fuera de `.env`.
   * Verificar que `.gitignore` excluya `.DS_Store`, `.venv/`, `data/batches/*.jsonl` y `results/*.jsonl`.
   * `git status` debe mantenerse limpio y libre de archivos temporales.
5. **Calidad de Software:**
   * Todo código producido debe verificarse con `pytest tests/unit/ -v`, `mypy src/ tests/unit/` y `ruff check .` con cero advertencias.