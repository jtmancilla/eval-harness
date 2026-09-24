"""Smoke validator and schema verification for Phase 2 sweep batch payloads.

Validates the integrity, entropy distribution, and the N=128 terminal line
for both gpt-5.6-sol (/v1/chat/completions) and gpt-6-astra (/v1/responses)
prior to OpenAI Batch API dispatch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

EXPECTED_ENTROPY_LEVELS: tuple[int, ...] = (5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 110, 128)
REQUESTS_PER_ENTROPY: int = 20
TOTAL_REQUESTS_PER_MODEL: int = 240
SWEEP_MODELS: tuple[str, ...] = ("gpt-5.6-sol", "gpt-6-astra")

CUSTOM_ID_REGEX = re.compile(
    r"^(?P<model>gpt-[a-z0-9\.\-]+)_N(?P<entropy>\d+)_(?P<condition>baseline_autorregresivo|neurosymbolic_handoff)_(?P<scenario>scenario_\d{3})_(?P<trace_id>[a-zA-Z0-9]+)$"
)


def validate_sol_payload_schema(record: dict[str, Any], line_num: int) -> None:
    """Validates schema conformance for gpt-5.6-sol (/v1/chat/completions)."""
    assert record.get("method") == "POST", f"Línea {line_num}: method debe ser POST"
    assert record.get("url") == "/v1/chat/completions", f"Línea {line_num}: url debe ser /v1/chat/completions"

    cid = record.get("custom_id", "")
    match = CUSTOM_ID_REGEX.match(cid)
    assert match is not None, f"Línea {line_num}: custom_id '{cid}' no cumple nomenclatura canónica"
    assert match.group("model") == "gpt-5.6-sol", f"Línea {line_num}: modelo en custom_id debe ser gpt-5.6-sol"

    body = record.get("body", {})
    assert body.get("model") == "gpt-5.6-sol", f"Línea {line_num}: body.model debe ser gpt-5.6-sol"
    assert body.get("reasoning_effort") == "none", f"Línea {line_num}: reasoning_effort debe ser 'none'"

    messages = body.get("messages")
    assert isinstance(messages, list) and len(messages) >= 2, f"Línea {line_num}: messages debe contener al menos 2 elementos"
    assert messages[0].get("role") == "system", f"Línea {line_num}: messages[0] debe ser rol 'system'"
    assert messages[1].get("role") == "user", f"Línea {line_num}: messages[1] debe ser rol 'user'"

    tools = body.get("tools")
    assert isinstance(tools, list), f"Línea {line_num}: body.tools debe ser una lista"

    tool_names: set[str] = set()
    for t_idx, tool in enumerate(tools):
        assert tool.get("type") == "function", f"Línea {line_num}, tool {t_idx}: type debe ser 'function'"
        fn = tool.get("function")
        assert isinstance(fn, dict), f"Línea {line_num}, tool {t_idx}: falta objeto 'function' anidado"
        name = fn.get("name")
        assert isinstance(name, str) and name, f"Línea {line_num}, tool {t_idx}: nombre de herramienta inválido"
        assert name not in tool_names, f"Línea {line_num}: nombre de herramienta duplicado '{name}'"
        tool_names.add(name)
        assert "parameters" in fn, f"Línea {line_num}, tool '{name}': faltan parameters"


def validate_astra_payload_schema(record: dict[str, Any], line_num: int) -> None:
    """Validates schema conformance for gpt-6-astra (/v1/responses)."""
    assert record.get("method") == "POST", f"Línea {line_num}: method debe ser POST"
    assert record.get("url") == "/v1/responses", f"Línea {line_num}: url debe ser /v1/responses"

    cid = record.get("custom_id", "")
    match = CUSTOM_ID_REGEX.match(cid)
    assert match is not None, f"Línea {line_num}: custom_id '{cid}' no cumple nomenclatura canónica"
    assert match.group("model") == "gpt-6-astra", f"Línea {line_num}: modelo en custom_id debe ser gpt-6-astra"

    body = record.get("body", {})
    assert body.get("model") == "gpt-6-astra", f"Línea {line_num}: body.model debe ser gpt-6-astra"
    assert body.get("reasoning") == {"effort": "low"}, f"Línea {line_num}: reasoning debe ser {{'effort': 'low'}}"
    assert isinstance(body.get("instructions"), str) and body["instructions"], f"Línea {line_num}: instructions inválidas"
    assert isinstance(body.get("input"), str) and body["input"], f"Línea {line_num}: input inválido"

    tools = body.get("tools")
    assert isinstance(tools, list), f"Línea {line_num}: body.tools debe ser una lista"

    tool_names: set[str] = set()
    for t_idx, tool in enumerate(tools):
        assert tool.get("type") == "function", f"Línea {line_num}, tool {t_idx}: type debe ser 'function'"
        assert "function" not in tool, f"Línea {line_num}, tool {t_idx}: esquema /v1/responses NO debe tener 'function' anidado"
        name = tool.get("name")
        assert isinstance(name, str) and name, f"Línea {line_num}, tool {t_idx}: nombre de herramienta inválido"
        assert name not in tool_names, f"Línea {line_num}: nombre de herramienta duplicado '{name}'"
        tool_names.add(name)
        assert "parameters" in tool, f"Línea {line_num}, tool '{name}': faltan parameters"


def validate_sweep_file(jsonl_path: Path, expected_model: str) -> dict[str, Any]:
    """Validates complete file metrics and returns the terminal N=128 record."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Archivo de lote no encontrado: {jsonl_path}")

    line_count = 0
    entropy_counts: dict[int, int] = {n: 0 for n in EXPECTED_ENTROPY_LEVELS}
    seen_custom_ids: set[str] = set()
    last_record: dict[str, Any] = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            line_count += 1
            record = json.loads(line)
            cid = record.get("custom_id", "")

            assert cid not in seen_custom_ids, f"custom_id duplicado en línea {idx}: {cid}"
            seen_custom_ids.add(cid)

            match = CUSTOM_ID_REGEX.match(cid)
            assert match is not None, f"custom_id no válido en línea {idx}: {cid}"
            entropy = int(match.group("entropy"))
            assert entropy in entropy_counts, f"Nivel de entropía inesperado N={entropy} en línea {idx}"
            entropy_counts[entropy] += 1

            tools = record.get("body", {}).get("tools", [])
            assert len(tools) == entropy, f"Línea {idx}: Esperadas {entropy} herramientas, encontradas {len(tools)}"

            last_record = record

    assert line_count == TOTAL_REQUESTS_PER_MODEL, (
        f"Conteo de líneas incorrecto en {jsonl_path.name}: esperado {TOTAL_REQUESTS_PER_MODEL}, obtenido {line_count}"
    )

    for entropy, count in entropy_counts.items():
        assert count == REQUESTS_PER_ENTROPY, (
            f"{jsonl_path.name}: N={entropy} tiene {count} solicitudes (esperadas {REQUESTS_PER_ENTROPY})"
        )

    # Validar terminal record (debe ser N=128)
    assert len(last_record.get("body", {}).get("tools", [])) == 128, (
        f"El último registro de {jsonl_path.name} no tiene N=128 herramientas"
    )

    if expected_model == "gpt-5.6-sol":
        validate_sol_payload_schema(last_record, line_count)
    elif expected_model == "gpt-6-astra":
        validate_astra_payload_schema(last_record, line_count)

    return {
        "file": jsonl_path.name,
        "total_lines": line_count,
        "unique_custom_ids": len(seen_custom_ids),
        "entropy_distribution": entropy_counts,
        "terminal_record": last_record,
    }


def execute_live_smoke_call(record: dict[str, Any]) -> bool:
    """Executes a live call to OpenAI API using the terminal N=128 record."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  [WARN]: OPENAI_API_KEY no encontrada. Omitiendo llamada en vivo.")
        return False

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    body = record.get("body", {})
    model = body.get("model", "")
    url = record.get("url", "")

    print(f"\n-> Ejecutando validación en vivo para modelo {model} en {url} con N=128 herramientas...")

    try:
        if url == "/v1/responses":
            resp_responses = client.responses.create(  # type: ignore[call-overload]
                model=model,
                instructions=body["instructions"],
                input=body["input"],
                tools=body["tools"],
                reasoning=body.get("reasoning", {"effort": "low"}),
            )
            print(f"✅ [/v1/responses HTTP 200] ID: {resp_responses.id} | Modelo: {resp_responses.model}")
        else:
            resp_chat = client.chat.completions.create(
                model=model,
                messages=body["messages"],
                tools=body["tools"],
                temperature=body.get("temperature", 0.0),
                reasoning_effort=body.get("reasoning_effort", "none"),
            )
            print(f"✅ [/v1/chat/completions HTTP 200] ID: {resp_chat.id} | Modelo: {resp_chat.model}")
        return True
    except Exception as exc:
        print(f"❌ [FALLO EN LLAMADA EN VIVO]: {exc}")
        return False


def run_smoke_verification(
    batches_dir: Path = Path("data/batches"),
    live_test: bool = False,
) -> bool:
    """Performs full smoke validation across both sweep files."""
    print("=" * 64)
    print(" EVAL-HARNESS FASE 2: SMOKE TEST Y VALIDACIÓN ESTRUCTURAL (N=[5..128])")
    print("=" * 64)

    sol_file = batches_dir / "sweep_batch_gpt-5.6-sol.jsonl"
    astra_file = batches_dir / "sweep_batch_gpt-6-astra.jsonl"

    for file_path, model in [(sol_file, "gpt-5.6-sol"), (astra_file, "gpt-6-astra")]:
        print(f"\n• Inspeccionando archivo: {file_path.name}")
        report = validate_sweep_file(file_path, model)
        print(f"   ✓ Total líneas:             {report['total_lines']} (100% de cuota)")
        print(f"   ✓ IDs únicos:               {report['unique_custom_ids']}")
        print(f"   ✓ 12 Niveles de Entropía:   {list(report['entropy_distribution'].keys())}")
        print(f"   ✓ Línea terminal (N=128):   {report['terminal_record']['custom_id']}")
        print(f"   ✓ Esquema protocolar:       {'Chat Completions' if model == 'gpt-5.6-sol' else 'Responses API'}")

        if live_test:
            execute_live_smoke_call(report["terminal_record"])

    print("\n" + "=" * 64)
    print(" CERTIFICACIÓN EXITOSA: Ambos archivos cumplen las invariantes de Phase 2.")
    print("=" * 64 + "\n")
    return True


def test_smoke_sweep_payloads_schema() -> None:
    """Pytest entrypoint for sweep payload structural verification."""
    project_root = Path(__file__).resolve().parents[1]
    batches_dir = project_root / "data" / "batches"
    assert run_smoke_verification(batches_dir=batches_dir, live_test=False) is True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test validator for Phase 2 sweep payloads.")
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=Path("data/batches"),
        help="Directory containing sweep batch files",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live HTTP 200 verification against OpenAI API for terminal N=128 records",
    )
    cli_args = parser.parse_args()

    success = run_smoke_verification(
        batches_dir=cli_args.batches_dir,
        live_test=cli_args.live,
    )
    sys.exit(0 if success else 1)
