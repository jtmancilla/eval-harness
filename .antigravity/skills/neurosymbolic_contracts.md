# SKILL: Handoff Tipado y Compuertas Simbólicas

Esta especificación técnica rige la implementación de contratos en `src/contracts/handoff.py`, la máquina de estados en `src/graph/state_machine.py` y las compuertas de intercepción en `src/gates/state_guard.py`.

---

## 1. Principio de Delegación Tipada
Las transiciones entre agentes nunca transmiten texto plano o diccionarios no validados. Todo intercambio de control se encapsula en un envelope fuertemente tipado (`HandoffEnvelope`) inmutable y cerrado a campos adicionales (`extra='forbid'`).

---

## 2. Definición del Contrato de Handoff (Pydantic V2)

```python
from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)

class AgentState(str, Enum):
    INITIALIZED = "INITIALIZED"
    ONBOARDING_PENDING = "ONBOARDING_PENDING"
    ONBOARDING_COMPLETED = "ONBOARDING_COMPLETED"
    COMPLIANCE_PENDING = "COMPLIANCE_PENDING"
    COMPLIANCE_APPROVED = "COMPLIANCE_APPROVED"
    COMPLIANCE_REJECTED = "COMPLIANCE_REJECTED"
    TREASURY_PENDING = "TREASURY_PENDING"
    DISPERSED = "DISPERSED"
    FAILED = "FAILED"

class HandoffEnvelope(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: UUID = Field(description="Identificador único global de la traza de ejecución.")
    source_agent: str = Field(description="Nombre canónico del agente emisor.")
    target_agent: str = Field(description="Nombre canónico del agente receptor.")
    current_state: AgentState = Field(description="Estado formal actual antes de la transición.")
    next_state: AgentState = Field(description="Estado propuesto al que se transiciona.")
    payload: T = Field(description="Estructura tipada de datos validada por Pydantic.")
    symbolic_hash: str = Field(description="Hash SHA-256 del estado y payload precedente.")
```

---

## 3. Matriz de Transiciones Permitidas (Grafo Dirigido)

Ninguna transición fuera de esta tabla está autorizada. Cualquier intento ajeno constituye una falla crítica de orquestación:

| Estado Origen | Agente Origen | Estado Destino Propuesto | Agente Destino |
| :--- | :--- | :--- | :--- |
| `INITIALIZED` | `System` | `ONBOARDING_PENDING` | `OnboardingAgent` |
| `ONBOARDING_PENDING` | `OnboardingAgent` | `ONBOARDING_COMPLETED` | `StateGuard` |
| `ONBOARDING_COMPLETED` | `StateGuard` | `COMPLIANCE_PENDING` | `ComplianceAgent` |
| `COMPLIANCE_PENDING` | `ComplianceAgent` | `COMPLIANCE_APPROVED` | `StateGuard` |
| `COMPLIANCE_PENDING` | `ComplianceAgent` | `COMPLIANCE_REJECTED` | `System` |
| `COMPLIANCE_APPROVED` | `StateGuard` | `TREASURY_PENDING` | `TreasuryAgent` |
| `TREASURY_PENDING` | `TreasuryAgent` | `DISPERSED` | `StateGuard` (SPEI Gate) |

---

## 4. Protocolo de Intercepción (`StateGuard`)

El módulo `src/gates/state_guard.py` actúa como compuerta determinista estricta en tiempo de ejecución. Antes de transferir el control al agente o herramienta receptora, debe validar:

1. **Adyacencia en el Grafo:** `(current_state, next_state)` debe existir en la lista explícita de aristas permitidas.
2. **Integridad Simbólica:**
   - La transición a `COMPLIANCE_PENDING` requiere que la CLABE pase `validate_clabe_checksum()`.
   - La transición a `TREASURY_PENDING` requiere que el RFC pase la validación sintáctica de homoclave y no figure en listas de bloqueo.
3. **Invariante Antievasión (Short-Circuit Prevention):**
   - Si el agente emisor intenta invocar directamente el payload de dispersión de Tesorería saltándose la aprobación formal de Compliance (`COMPLIANCE_APPROVED`), el guard lanza inmediatamente una excepción de dominio `ShortCircuitViolation`.
   - La ejecución se aborta en código local, marcando la traza como fallida por evasión normativa sin incurrir en llamadas adicionales al LLM.

---

## 5. Excepciones de Dominio

Todo fallo simbólico debe levantar excepciones tipadas para la posterior auditoría ciega:

```python
class GateViolationError(Exception):
    """Excepción base para violaciones de compuertas neuro-simbólicas."""
    pass

class ShortCircuitViolation(GateViolationError):
    """Intento de evasión de secuencias normativas o salto no autorizado en el grafo."""
    pass

class SchemaIntegrityViolation(GateViolationError):
    """Payload corrupto, campos extra detectados o fallo de tipado Pydantic."""
    pass

class CryptographicIntegrityViolation(GateViolationError):
    """Fallo en la verificación de hash de estado o alteración de traza."""
    pass
```