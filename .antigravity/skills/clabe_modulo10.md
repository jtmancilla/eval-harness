# SKILL: Algoritmo de Verificación CLABE Módulo 10 (Norma Oficial ABM / Banxico)

Esta especificación técnica rige la implementación determinista de la compuerta de validación en `src/gates/modulo10.py` y el validador de campo en `src/contracts/clabe.py`.

---

## 1. Fundamento Normativo y Estructura
La Clave Bancaria Estandarizada (CLABE) en México está regida por la Asociación de Bancos de México (ABM) y el Banco de México (Banxico). Consiste en una cadena de texto numérico de exactamente **18 caracteres**:

```text
 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
[ B A N C O ] [ P L A Z A ] [      N Ú M E R O   D E   C U E N T A      ] [ D V ]
```

- **Dígitos 01 al 03 (Banco):** Código de la institución financiera asignado por ABM.
- **Dígitos 04 al 06 (Plaza):** Código de la sucursal o ubicación geográfica.
- **Dígitos 07 al 17 (Cuenta):** Número de cuenta del beneficiario (rellenado con ceros a la izquierda si tiene menor longitud).
- **Dígito 18 (Dígito de Control / Verificador - DV):** Resultado del algoritmo Módulo 10 ponderado aplicado a los primeros 17 dígitos.

---

## 2. Especificación Matemática del Algoritmo

Para una CLABE representada como una secuencia de dígitos $D = [d_0, d_1, \dots, d_{16}, d_{17}]$ donde cada $d_i \in \{0, \dots, 9\}$:

1. **Secuencia de Ponderaciones ($W$):**
   Se utiliza el vector cíclico de factores de peso $[3, 7, 1]$ repetido para las primeras 17 posiciones:
   $$W = [3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7]$$

2. **Cálculo de Productos Modulares:**
   Para cada posición $i \in \{0, 1, \dots, 16\}$, se multiplica el dígito por su factor de peso correspondiente y se obtiene el residuo en base 10:
   $$p_i = (d_i \times w_i) \pmod{10}$$

3. **Suma Total de Residuos:**
   Se calcula la suma escalar de todos los residuos intermedios:
   $$S = \sum_{i=0}^{16} p_i$$

4. **Extracción del Dígito Verificador ($DV_{calculado}$):**
   Se obtiene el complemento a 10 del residuo modular de la suma total:
   $$DV_{calculado} = (10 - (S \pmod{10})) \pmod{10}$$
   *(Nota crítica: El segundo $\pmod{10}$ garantiza que si $(S \pmod{10}) = 0$, el resultado final sea $0$ y no $10$).*

5. **Condición de Validez:**
   La CLABE es íntegra y válida si y solo si:
   $$d_{17} = DV_{calculado}$$

---

## 3. Excepciones de Dominio

La compuerta debe lanzar excepciones explícitas para distinguir fallos estructurales de fallos de integridad:

```python
class ClabeValidationError(ValueError):
    """Excepción base para errores de validación de CLABE."""
    pass

class ClabeLengthError(ClabeValidationError):
    """Longitud distinta a 18 caracteres."""
    pass

class ClabeNonNumericError(ClabeValidationError):
    """Presencia de caracteres alfabéticos o especiales."""
    pass

class ClabeChecksumMismatchError(ClabeValidationError):
    """El dígito verificador provisto no coincide con el cálculo del Módulo 10."""
    pass
```

---

## 4. Implementación Canónica Determinista (`src/gates/modulo10.py`)

Esta función es pura, $O(1)$ en tiempo y memoria, y opera de forma completamente desacoplada de cualquier LLM o framework web:

```python
from __future__ import annotations

# Factores de ponderación fijos según estándar ABM
CLABE_WEIGHTS: tuple[int, ...] = (3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7)

def compute_clabe_control_digit(clabe_17: str) -> int:
    """Calcula el dígito verificador para los primeros 17 dígitos de una CLABE.

    Raises:
        ClabeLengthError: Si la cadena no contiene exactamente 17 caracteres.
        ClabeNonNumericError: Si la cadena contiene caracteres no numéricos.
    """
    if len(clabe_17) != 17:
        raise ClabeLengthError(f"Se requieren exactamente 17 dígitos, recibidos {len(clabe_17)}")
    if not clabe_17.isdigit():
        raise ClabeNonNumericError("La secuencia base debe contener únicamente dígitos numéricos")

    total_sum = sum(
        (int(digit) * weight) % 10
        for digit, weight in zip(clabe_17, CLABE_WEIGHTS, strict=True)
    )
    return (10 - (total_sum % 10)) % 10


def validate_clabe(clabe: str) -> bool:
    """Valida la integridad estructural y matemática de una CLABE completa de 18 dígitos.

    Returns:
        True si la CLABE cumple con la longitud, formato y dígito de control.

    Raises:
        ClabeLengthError: Longitud inválida.
        ClabeNonNumericError: Caracteres inválidos.
        ClabeChecksumMismatchError: Fallo de dígito verificador.
    """
    if len(clabe) != 18:
        raise ClabeLengthError(f"La CLABE debe tener 18 dígitos exactos, recibidos {len(clabe)}")
    if not clabe.isdigit():
        raise ClabeNonNumericError("La CLABE debe contener exclusivamente dígitos numéricos")

    expected_control = int(clabe[17])
    calculated_control = compute_clabe_control_digit(clabe[:17])

    if expected_control != calculated_control:
        raise ClabeChecksumMismatchError(
            f"Dígito verificador inválido: provisto={expected_control}, esperado={calculated_control}"
        )

    return True
```

---

## 5. Integración con Contratos Pydantic V2 (`src/contracts/clabe.py`)

Para garantizar que ningún agente pueda propagar una CLABE inválida en el payload, el modelo de datos debe compilar la validación en tiempo de parseo:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.gates.modulo10 import validate_clabe

class ClabeAccount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clabe: str = Field(
        ...,
        min_length=18,
        max_length=18,
        description="Cuenta CLABE bancaria mexicana de 18 dígitos con Módulo 10 válido."
    )

    @field_validator("clabe", mode="after")
    @classmethod
    def enforce_modulo10_integrity(cls, value: str) -> str:
        # validate_clabe levantará la excepción correspondiente si falla
        validate_clabe(value)
        return value
```

---

## 6. Vectores de Prueba Oficiales (Unit Tests Fixtures)

Utilizar estos vectores en `tests/unit/test_modulo10.py` para asegurar determinismo:

| CLABE Completa | Dígito Verificador | Estatus Esperado | Razón de Falla (si aplica) |
| :--- | :--- | :--- | :--- |
| `002115016003269412` | `2` | **VÁLIDA** | Suma modular correcta (Banco: 002 Banamex). |
| `032180000118359719` | `9` | **VÁLIDA** | Suma modular correcta (Banco: 032 IXE). |
| `012180004467389025` | `5` | **VÁLIDA** | Suma modular correcta (Banco: 012 BBVA). |
| `002115016003269418` | `8` | **INVÁLIDA** | `ClabeChecksumMismatchError` (esperaba `2`). |
| `01218000446738902`  | N/A | **INVÁLIDA** | `ClabeLengthError` (17 dígitos). |
| `0121800044673890259` | N/A | **INVÁLIDA** | `ClabeLengthError` (19 dígitos). |
| `01218000446738902X` | `X` | **INVÁLIDA** | `ClabeNonNumericError` (carácter no numérico). |