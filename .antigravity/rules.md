# Antigravity Workspace Rules — Neuro-Symbolic Multi-Agent Financial Architecture

Este archivo define las reglas operativas y de codificación innegociables para el desarrollo y evaluación en este workspace:

1. **Axioma de Aritmética Cero en LLMs:**
   - Queda estrictamente prohibido validar lógica transaccional, cálculos de módulo, sumas de verificación o expresiones regulares dentro de prompts de modelos de lenguaje o cadenas de razonamiento (Chain-of-Thought).
   - Todo cálculo de integridad sintáctica y matemática debe delegarse exclusivamente a la compuerta simbólica en `src/gates/`.

2. **Inmutabilidad de Contratos Inter-Agente:**
   - Todo payload transmitido entre agentes debe ser una subclase de `pydantic.BaseModel` con `model_config = ConfigDict(extra='forbid', frozen=True)`.
   - Queda prohibida la transmisión de payloads en texto plano sin tipar, diccionarios genéricos o campos dinámicos no declarados.

3. **Pureza Determinista de Compuertas Simbólicas (`src/gates/`):**
   - Las compuertas deben permanecer compuestas por funciones puras, deterministas y libres de efectos secundarios.
   - Prohibido importar librerías de modelos (OpenAI, Anthropic, etc.), clientes HTTP o frameworks asíncronos pesados dentro de `src/gates/`. El tiempo de respuesta debe ser sub-milisegundo.

4. **Prevención Estricta de Bypass Transaccional (*Short-Circuit Prevention*):**
   - Ninguna solicitud puede invocar el agente de Tesorería ni la compuerta de dispersión SPEI si la traza no cuenta con la firma explícita `COMPLIANCE_APPROVED` registrada en el grafo de auditoría de NopalDB.
   - Cualquier intento de bypass debe abortarse en código local mediante la excepción `ShortCircuitViolation`.
