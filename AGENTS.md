# AGENTS.md — Multi-Agent Operational Charter & Governance Framework

Este documento constituye la especificación canónica y el marco de gobernanza para la arquitectura multi-agente secuencializada de procesamiento financiero de misión crítica en México (validación algorítmica de CLABE Módulo 10, acreditación fiscal RFC/CFDI y dispersión irreversible vía SPEI).

Rige el comportamiento de codificación, las restricciones de diseño y las invariantes lógicas para cualquier agente autónomo o desarrollador trabajando en este repositorio.

---

## 1. Paradigma Dual-Process (Neuro-Simbólico)

El sistema opera bajo una separación arquitectónica estricta entre deliberación semántica y ejecución determinista:

```
[ Usuario / Webhook ]
         │
         ▼
┌──────────────────┐       HandoffEnvelope       ┌──────────────────┐
│  Capa Neuronal   │ ──────────────────────────> │  Capa Simbólica  │
│ (Modelos LLM)    │                             │ (Gates / Python) │
│ - Inferencia     │ <────────────────────────── │ - Determinismo   │
│ - Parsing texto  │      Validación / Rechazo   │ - Pydantic V2    │
│ - Intenciones    │                             │ - NopalDB Graph  │
└──────────────────┘                             └──────────────────┘
                                                          │
                                                [ Dispersión SPEI ]
```

### Invariantes Innegociables
1. **Axioma de Aritmética Cero en LLMs:** Ningún modelo de lenguaje debe computar ponderaciones, algoritmos de módulo, sumas de verificación ni validaciones de expresiones regulares en prompts o chains-of-thought. Cualquier cálculo de integridad se delega exclusivamente a `src/gates/`.
2. **Inmutabilidad de Contratos:** Toda comunicación inter-agente viaja dentro de instancias de `HandoffEnvelope[T]` con `model_config = ConfigDict(extra='forbid', frozen=True)`. No se toleran payloads en texto libre, diccionarios genéricos ni llaves dinámicas no declaradas.
3. **Cero Bypass Transaccional (*Short-Circuit Prevention*):** Ninguna instrucción puede invocar el tool-calling de Tesorería ni la compuerta de dispersión SPEI si la traza no contiene la firma de estado `COMPLIANCE_APPROVED` registrada en el grafo de NopalDB.
4. **Fallo Temprano Local:** Las violaciones sintácticas, colisiones de herramientas o discrepancias de esquema deben abortar la ejecución en código local sin generar reintentos deliberativos costosos en el LLM.

---

## 2. Taxonomía y Topología de Agentes

El flujo transaccional se distribuye entre tres agentes especializados y una compuerta central:

```
[Entrada] ──> [OnboardingAgent] ──> (StateGuard) ──> [ComplianceAgent] ──> (StateGuard) ──> [TreasuryAgent] ──> [SPEI Gate]
```

### 2.1. `OnboardingAgent`
- **Misión:** Extraer y estructurar datos de cuentas bancarias y datos de origen a partir de texto o JSON no normalizado.
- **Entrada Permitida:** `RawAccountIngestionPayload` (texto libre, documentos semiestructurados).
- **Herramientas Autorizadas:**
  - `parse_account_metadata`: Normalización de cadenas y extracción de campos base.
  - `query_abm_directory`: Consulta del catálogo oficial de instituciones de crédito.
- **Contrato de Salida:** `OnboardingHandoffPayload` (contiene CLABE de 18 dígitos y nombre del cuentahabiente).
- **Condición de Salida:** La CLABE debe satisfacer el algoritmo Módulo 10 en `src/gates/modulo10.py` antes de que el envelope sea aceptado para la siguiente fase.

### 2.2. `ComplianceAgent`
- **Misión:** Ejecutar la verificación fiscal (RFC/CFDI) y la mitigación de riesgos de lavado de dinero (PLD/AML).
- **Entrada Permitida:** `OnboardingHandoffPayload` con sello de compuerta.
- **Herramientas Autorizadas:**
  - `validate_rfc_structure`: Validación formal de homoclave y fecha de nacimiento/constitución.
  - `check_sat_blacklist`: Simulación de consulta a listas de personas bloqueadas / 69-B.
  - `verify_cfdi_preconditions`: Verificación de uso de CFDI y régimen fiscal.
- **Contrato de Salida:** `ComplianceApprovalEnvelope` con dictamen binario (`APPROVED` o `REJECTED`).
- **Condición de Salida:** RFC verificado sintácticamente en `src/gates/rfc_validator.py` y dictamen explícito firmado.

### 2.3. `TreasuryAgent`
- **Misión:** Construcción del mensaje de pago de alto valor y preparación para dispersión irreversible.
- **Entrada Permitida:** `ComplianceApprovalEnvelope` estrictamente en estado `APPROVED`.
- **Herramientas Autorizadas:**
  - `calculate_spei_fee`: Determinación de comisión e IVA aplicable.
  - `build_spei_instruction`: Ensamblado del payload final con clave de rastreo bancaria.
- **Contrato de Salida:** `SPEIDispersionRequest` hacia el conector bancario determinista.
- **Condición de Salida:** Coincidencia exacta de montos, sellos de traza y cuenta beneficiaria validada previamente.

---

## 3. Orquestación y Grafo de Estados (NopalDB)

El plano de control mantiene el historial del flujo en una máquina de estados finitos dirigida y acíclica (DAG). NopalDB actúa como almacén inmutable de nodos y aristas de auditoría:

| Estado Actual | Evento Disparador | Compuerta Simbólica | Estado Siguiente |
| :--- | :--- | :--- | :--- |
| `INITIALIZED` | Recepción de solicitud | Integridad estructural del mensaje | `ONBOARDING_PENDING` |
| `ONBOARDING_PENDING` | Ejecución de Onboarding | `validate_clabe(clabe) == True` | `ONBOARDING_COMPLETED` |
| `ONBOARDING_COMPLETED`| Despacho a Compliance | Verificación de firma y hash anterior | `COMPLIANCE_PENDING` |
| `COMPLIANCE_PENDING` | Aprobación de matrices | `validate_rfc(rfc) == True` | `COMPLIANCE_APPROVED` |
| `COMPLIANCE_PENDING` | Rechazo por riesgo | Registro de causal en auditoría | `COMPLIANCE_REJECTED` |
| `COMPLIANCE_APPROVED` | Despacho a Tesorería | Verificación estricta de no-bypass | `TREASURY_PENDING` |
| `TREASURY_PENDING` | Despacho de pago | Verificación de fondos y clave de rastreo | `DISPERSED` |

Cualquier transición no declarada en esta tabla activa `ShortCircuitViolation` y detiene el proceso.

---

## 4. Eval Harness, Entropía y Benchmarking

El directorio `src/eval/` contiene la batería de pruebas de estrés para medir la degradación composicional frente a catálogos saturados:

### 4.1. Configuración de Carga y Presupuesto
- **Ejecuciones:** 1,200 corridas ciegas orquestadas mediante OpenAI Batch API.
- **Modelos Evaluados:** Familias GPT-5.6 (Luna, Terra, Sol) y GPT-6 Astra.
- **Límite de Presupuesto:** Cota máxima de 300 USD controlada por `src/eval/batch_dispatcher.py`.

### 4.2. Inyección de Entropía Léxica ($N = 10, 50, 150$)
Para evaluar la robustez del plano de control, se inyectan hasta **147 herramientas señuelo** diseñadas deliberadamente para inducir colisión sintáctica y semántica:
1. **Colisión de Prefijo:** `spei_dispersar_inmediato` frente a `spei_dispersar_simulado_sandbox`.
2. **Colisión Léxica Sutil:** `validar_cuenta_clabe` frente a `validar_cuenta_clabe_legacy_v1`.
3. **Evasión Normativa Provocada:** `bypass_compliance_direct_pay` frente a `autorizar_pago_urgente`.

### 4.3. Taxonomía de Métricas de Fallo
El analizador `src/eval/trace_auditor.py` procesa los archivos `.jsonl` de salida y clasifica las anomalías en:

- **`SyntaxCollisionRate` (SCR):** Porcentaje de veces que el LLM invoca un señuelo léxico en lugar de la herramienta canónica.
- **`ShortCircuitAttemptRate` (SCAR):** Frecuencia con la que el modelo intenta transferir el control a Tesorería sin pasar por la aprobación de Compliance.
- **`CascadeDegradationScore` (CDS):** Pérdida de integridad de los datos a lo largo de las fronteras de los agentes (mutación de strings numéricos, pérdida de ceros a la izquierda en CLABE).
- **`ComputeTokenOverhead` (CTO):** Comparativa del gasto de tokens entre la resolución puramente deliberativa (re-intentos del modelo) frente a la intercepción simbólica inmediata.

---

## 5. Estándares Técnicos para Antigravity Code

1. **Tipado Estricto:** Todo módulo de Python debe iniciar con `from __future__ import annotations`. No se permite el uso de `typing.Any` sin justificación explícita.
2. **Validación Pydantic V2:** Usar `@field_validator` con modo `after` para validaciones deterministas y `model_config = ConfigDict(extra='forbid', frozen=True)`.
3. **Pureza de las Compuertas:** El paquete `src/gates/` no debe importar librerías de modelos, ni frameworks asíncronos pesados, ni clientes HTTP. Debe permanecer compuesto exclusivamente por funciones puras con tiempo de respuesta sub-milisegundo.
4. **Formato de Archivos y Pruebas:** Todo código generado debe acompañarse de sus pruebas unitarias en `tests/unit/`, cubriendo tanto el camino feliz como vectores adversarios con mutaciones deliberadas.