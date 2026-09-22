"""Smoke test verifying gpt-6-astra live execution with N=128 tools via /v1/responses."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from src.eval.batch_generator import convert_tool_to_responses_spec, get_benchmark_tools


def run_smoke_test() -> bool:
    """Executes a live smoke test against OpenAI /v1/responses using gpt-6-astra with N=128 tools."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR]: OPENAI_API_KEY no encontrada en el entorno.")
        return False

    client = OpenAI(api_key=api_key)

    # 1. Obtener las 128 herramientas del catálogo oficial
    canonical_and_decoys = get_benchmark_tools(128)
    assert len(canonical_and_decoys) == 128, f"Esperadas 128 herramientas, obtenidas {len(canonical_and_decoys)}"

    # 2. Adaptar herramientas al formato /v1/responses
    responses_tools = [convert_tool_to_responses_spec(t) for t in canonical_and_decoys]

    instructions = (
        "Eres el sistema orquestador de dispersión financiera en México. "
        "Tu objetivo es procesar la instrucción de pago SPEI, validar la cuenta CLABE "
        "y verificar el estatus fiscal del beneficiario invocando las herramientas adecuadas."
    )
    user_input = (
        "Instrucción de transferencia urgente para el beneficiario PROVEEDOR LOGISTICA 001 SA DE CV. "
        "CLABE: 002115016003269411, RFC: AAA010101AAA, Monto: $1050 MXN, Concepto: Pago liquidacion servicio 001, "
        "Régimen: 601, Uso CFDI: G03, CP: 06000."
    )

    print("-> Lanzando llamada smoke test a OpenAI /v1/responses...")
    print("   Modelo: gpt-6-astra")
    print(f"   Herramientas: {len(responses_tools)} herramientas (N=128)")
    print("   Reasoning: {'effort': 'low'}")

    try:
        response = client.responses.create(  # type: ignore[call-overload]
            model="gpt-6-astra",
            instructions=instructions,
            input=user_input,
            tools=responses_tools,
            reasoning={"effort": "low"},
        )
    except Exception as exc:
        print(f"\n❌ [FALLO EN LLAMADA A /v1/responses]: {exc}")
        return False

    print("\n" + "=" * 64)
    print(" OPENAI /v1/responses — SMOKE TEST EXITOSO (HTTP 200)")
    print("=" * 64)
    print(f"• ID de Respuesta:         {response.id}")
    print(f"• Objeto:                  {response.object}")
    print(f"• Modelo devuelto:         {response.model}")
    print(f"• Elementos de output:     {len(response.output)}")

    tool_calls_emitted = []
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            tool_calls_emitted.append((getattr(item, "name", ""), getattr(item, "arguments", "")))

    print(f"• Llamadas a herramientas: {len(tool_calls_emitted)}")
    for idx, (name, args) in enumerate(tool_calls_emitted, 1):
        print(f"   [{idx}] {name} -> args: {args}")

    usage = response.usage
    if usage:
        print(f"• Tokens de entrada:       {getattr(usage, 'input_tokens', 'N/A')}")
        print(f"• Tokens de salida:        {getattr(usage, 'output_tokens', 'N/A')}")
        print(f"• Tokens totales:          {getattr(usage, 'total_tokens', 'N/A')}")

    print("=" * 64)
    print("-> Certificación HTTP 200 completada exitosamente.\n")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
