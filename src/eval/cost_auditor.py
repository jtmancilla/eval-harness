"""Auditor de consumo de tokens y costos reales (Batch API 50% discount).

Soporta indistintamente /v1/chat/completions y /v1/responses.
"""
from __future__ import annotations

import json
from typing import Any

# Tarifas estándar estimadas (por 1M de tokens)
# Batch API aplica automáticamente el 50% de descuento sobre el precio de lista
RATES: dict[str, tuple[float, float]] = {
    # Modelo: (Input $/1M, Output $/1M)
    "gpt-5.6-luna": (1.25, 5.00),    # Batch 50%: $0.625 / $2.50
    "gpt-5.6-sol":  (2.50, 10.00),   # Batch 50%: $1.250 / $5.00
    "gpt-5.6-terra": (2.50, 10.00),  # Batch 50%: $1.250 / $5.00
    "gpt-6-astra":  (5.00, 20.00),   # Batch 50%: $2.500 / $10.00
}


def audit_real_costs(jsonl_path: str = "results/real_batch_1200.jsonl") -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, int]] = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            body = data.get("response", {}).get("body", {})
            model = body.get("model", "unknown")

            # Normalizar nombre del modelo
            base_model = next((m for m in RATES if m in model), model)

            usage = body.get("usage", {})

            # Soporte agnóstico para chat.completions y responses API
            prompt_tok = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
            completion_tok = usage.get("completion_tokens") or usage.get("output_tokens", 0)

            # Extracción de tokens de razonamiento
            comp_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
            reasoning_tok = comp_details.get("reasoning_tokens", 0)

            if base_model not in totals:
                totals[base_model] = {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                }

            totals[base_model]["requests"] += 1
            totals[base_model]["prompt_tokens"] += prompt_tok
            totals[base_model]["completion_tokens"] += completion_tok
            totals[base_model]["reasoning_tokens"] += reasoning_tok

    print("\n" + "=" * 85)
    print("AUDITORÍA DE TOKENS Y COSTO REAL FACTURADO (OPENAI BATCH API — 50% OFF)")
    print("=" * 85)
    print(f"{'Modelo':<16} | {'Reqs':<5} | {'Prompt Tok':<12} | {'Comp Tok':<10} | {'Reas Tok':<10} | {'Costo Real'}")
    print("-" * 85)

    total_cost_usd = 0.0
    total_tokens_all = 0
    results: dict[str, dict[str, Any]] = {}

    for model_name in ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"]:
        if model_name not in totals:
            continue
        m_data = totals[model_name]
        in_tok = m_data["prompt_tokens"]
        out_tok = m_data["completion_tokens"]
        total_tokens_all += (in_tok + out_tok)

        # Precios con 50% de descuento aplicado
        in_rate, out_rate = RATES.get(model_name, (2.5, 10.0))
        batch_in_rate = (in_rate * 0.5) / 1_000_000
        batch_out_rate = (out_rate * 0.5) / 1_000_000

        cost = (in_tok * batch_in_rate) + (out_tok * batch_out_rate)
        total_cost_usd += cost
        results[model_name] = {
            **m_data,
            "cost_usd": round(cost, 4),
        }

        print(f"{model_name:<16} | {m_data['requests']:<5} | {in_tok:<12,d} | {out_tok:<10,d} | {m_data['reasoning_tokens']:<10,d} | ${cost:.4f} USD")

    print("=" * 85)
    print(f"• Tokens Totales Facturados:   {total_tokens_all:,d}")
    print(f"• Gasto Total Consolidado:    ${total_cost_usd:.2f} USD")
    print("=" * 85 + "\n")

    return results


if __name__ == "__main__":
    audit_real_costs()
