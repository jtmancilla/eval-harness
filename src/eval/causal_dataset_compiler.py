"""Causal Inference Synthetic Dataset Compiler (Phases 2-4).

Multi-phase async pipeline that transforms deterministic CausalDAGSpec objects
into humanized financial narratives and assembles them as OpenAI Batch API
payloads for gpt-6-astra evaluation.

Phase 2: Semantic mapping of abstract nodes → financial variables (AsyncOpenAI → gpt-5.6-sol)
Phase 3: Narrative humanization with spurious correlations (AsyncOpenAI → gpt-5.6-sol)
Phase 4: Batch API record assembly (/v1/responses for gpt-6-astra)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random as _stdlib_random
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any  # Used for OpenAI client compat and JSON deserialization
from uuid import uuid4

import yaml  # type: ignore[import-untyped]

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:  # type: ignore[misc]
        return False


try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = Any  # type: ignore[misc,assignment]

from tenacity import retry, stop_after_attempt, wait_exponential

from src.eval.causal_dag_factory import CausalDAGFactory, CausalDAGSpec

# --------------------------------------------------------------------------- #
# Data Contracts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SemanticMapping:
    """Mapping from abstract node names to financial variable names."""

    dag_id: str
    mappings: dict[str, str]


@dataclass(frozen=True)
class CausalBatchRecord:
    """Complete record ready for Batch API dispatch with ground truth."""

    custom_id: str
    dag_spec: CausalDAGSpec
    semantic_mapping: SemanticMapping
    narrative: str
    effort: str
    batch_payload: dict[str, Any]


# --------------------------------------------------------------------------- #
# Spurious Correlation Pool (domain-specific noise for narrative injection)
# --------------------------------------------------------------------------- #

SPURIOUS_PAIRS: list[tuple[str, str]] = [
    (
        "el precio internacional del petróleo Brent",
        "el índice de morosidad en tarjetas de crédito",
    ),
    (
        "las ventas de automóviles importados",
        "el volumen de transferencias SPEI interbancarias",
    ),
    (
        "la depreciación del peso frente al dólar",
        "las aperturas de cuentas de ahorro digitales",
    ),
    (
        "el consumo de energía eléctrica industrial",
        "los rendimientos de CETES a 28 días",
    ),
    (
        "la tasa de desempleo juvenil",
        "el número de sucursales bancarias abiertas",
    ),
    (
        "las exportaciones de aguacate",
        "el monto de créditos hipotecarios otorgados",
    ),
    (
        "el tráfico aéreo nacional de pasajeros",
        "la emisión de bonos corporativos en BMV",
    ),
    (
        "las remesas familiares recibidas",
        "el spread entre TIIE y tasa de fondeo",
    ),
    (
        "la producción de acero nacional",
        "los pagos de nómina vía SPEI",
    ),
    (
        "el índice de confianza del consumidor",
        "las provisiones para riesgos crediticios",
    ),
]

# --------------------------------------------------------------------------- #
# Prompt Templates
# --------------------------------------------------------------------------- #

PHASE2_SYSTEM_PROMPT = (
    "Eres un Economista Jefe del sector bancario mexicano con 20 años de experiencia "
    "en análisis de riesgos sistémicos, política monetaria y regulación financiera "
    "(Banxico, CNBV, SHCP).\n\n"
    "Recibirás la topología de un grafo causal con nodos abstractos y relaciones "
    "direccionales. Tu tarea es asignar a cada nodo un nombre de variable financiera "
    "o macroeconómica realista del contexto mexicano que respete lógicamente la "
    "dirección causal de cada arista.\n\n"
    "Reglas estrictas:\n"
    "1. Cada nombre debe ser una variable real y específica del sector financiero/bancario mexicano.\n"
    "2. Las direcciones causales deben tener sentido económico.\n"
    "3. No alteres la estructura del grafo ni inventes aristas nuevas.\n"
    "4. Los nombres deben ser concisos (máximo 5 palabras por variable).\n"
    "5. Responde ÚNICAMENTE en formato JSON válido, sin texto adicional."
)

PHASE3_SYSTEM_PROMPT = (
    "Eres simultáneamente un analista financiero senior redactando un informe de "
    "riesgos para el comité directivo de un banco mexicano sistémicamente importante, "
    "y un periodista económico de El Financiero escribiendo una nota de análisis.\n\n"
    "Tu objetivo es redactar un texto narrativo de 400-500 palabras que describa "
    "implícitamente las relaciones causales entre las variables financieras.\n\n"
    "Restricciones Críticas de Calidad:\n"
    "1. DEBES inyectar al menos 2 correlaciones espurias narrativas: menciona fenómenos "
    "que ocurren simultáneamente pero NO se causan mutuamente.\n"
    "2. Usa lenguaje deliberadamente ambiguo: alterna entre verbos causales (provoca, "
    "genera, causa) y verbos correlacionales (se asocia con, coincide con, va de la mano).\n"
    "3. Incluye datos numéricos ficticios pero plausibles (porcentajes, puntos base, "
    "montos en MXN o USD).\n"
    "4. Usa jerga corporativa bancaria mexicana (CAR, ICAP, cartera vencida, provisiones, "
    "spread, TIIE, CETES).\n"
    "5. NO reveles la estructura matemática subyacente.\n"
    "6. Incluye al menos una referencia a reguladores mexicanos (Banxico, CNBV, SHCP, SAT).\n"
    "7. El texto debe parecer un fragmento real de un informe trimestral o nota periodística."
)

EVALUATOR_INSTRUCTIONS = (
    "Eres un experto en inferencia causal y economía financiera con formación doctoral. "
    "Se te presentará un texto narrativo sobre relaciones entre variables financieras "
    "del mercado mexicano.\n\n"
    "Tu tarea es analizar el texto con rigor y responder tres preguntas en niveles "
    "crecientes de sofisticación causal, basándote en el razonamiento causal formal "
    "(Pearl, Rubin), NO en meras asociaciones narrativas.\n\n"
    "IMPORTANTE: Distingue cuidadosamente entre:\n"
    "- Correlación observacional (lo que co-ocurre en los datos)\n"
    "- Causalidad real (lo que causa qué, basado en mecanismos plausibles)\n"
    "- Correlación espuria (co-ocurrencias sin mecanismo causal)\n\n"
    "Los valores numéricos están expresados en desviaciones estándar respecto a "
    "sus medias históricas.\n\n"
    "Responde ÚNICAMENTE en formato JSON con esta estructura exacta:\n"
    "{\n"
    '  "L1": {"direction": "positive|negative|none", "magnitude": <float -1 a 1>, '
    '"reasoning": "<razonamiento>"},\n'
    '  "L2": {"direction": "positive|negative|none", "causal_effect": <float>, '
    '"reasoning": "<razonamiento>"},\n'
    '  "L3": {"counterfactual_value": <float>, "reasoning": "<razonamiento>"}\n'
    "}"
)


# --------------------------------------------------------------------------- #
# JSON Extraction Utility
# --------------------------------------------------------------------------- #

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()

    # 1. Try direct parse
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from markdown code block
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1).strip())  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # 3. Try finding first { ... } block
    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


def _deterministic_dag_seed(base_seed: int, dag_id: str) -> int:
    """Produce a deterministic integer seed from base_seed + dag_id (hash-stable)."""
    raw = hashlib.sha256(f"{base_seed}:{dag_id}".encode()).hexdigest()[:8]
    return int(raw, 16)


# --------------------------------------------------------------------------- #
# Compiler
# --------------------------------------------------------------------------- #


class CausalDatasetCompiler:
    """Multi-phase async pipeline for causal benchmark dataset generation.

    Phase 2: Semantic mapping of abstract DAG nodes → financial variables.
    Phase 3: Narrative humanization with spurious correlation injection.
    Phase 4: Batch API record assembly for gpt-6-astra evaluation.
    """

    DEFAULT_GENERATOR_MODEL = "gpt-6-astra"
    DEFAULT_EVALUATOR_MODEL = "gpt-6-astra"
    DEFAULT_NUM_DAGS = 50
    DEFAULT_SEED = 42
    BUDGET_CEILING_USD: Decimal = Decimal("300.00")

    def __init__(
        self,
        generator_model: str = DEFAULT_GENERATOR_MODEL,
        generator_reasoning_effort: str = "low",
        evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
        reasoning_efforts: tuple[str, ...] = ("low", "medium", "high"),
        num_dags: int = DEFAULT_NUM_DAGS,
        seed: int = DEFAULT_SEED,
        templates: list[str] | None = None,
        output_prefix: str = "causal_batch",
        budget_ceiling_usd: Decimal = Decimal("300.00"),
        concurrency: int = 10,
    ) -> None:
        load_dotenv()
        self.generator_model = generator_model
        self.generator_reasoning_effort = generator_reasoning_effort
        self.evaluator_model = evaluator_model
        self.reasoning_efforts = reasoning_efforts
        self.num_dags = num_dags
        self.seed = seed
        self.templates = templates
        self.output_prefix = output_prefix
        self.budget_ceiling_usd = budget_ceiling_usd
        self._semaphore = asyncio.Semaphore(concurrency)

        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            self._client: AsyncOpenAI | None = AsyncOpenAI(api_key=api_key)
        else:
            self._client = None

    async def _call_llm(self, instructions: str, user_input: str) -> str:
        """Call LLM adaptively based on model family (responses vs chat completions)."""
        if self._client is None:
            return ""

        async with self._semaphore:
            if "astra" in self.generator_model:
                response = await self._client.responses.create(  # type: ignore[call-overload]
                    model=self.generator_model,
                    instructions=instructions,
                    input=user_input,
                    reasoning={"effort": self.generator_reasoning_effort},
                )
                return response.output_text or ""
            else:
                response = await self._client.chat.completions.create(
                    model=self.generator_model,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": user_input},
                    ],
                )
                return response.choices[0].message.content or ""

    # ------------------------------------------------------------------- #
    # Phase 2: Semantic Mapping
    # ------------------------------------------------------------------- #

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _phase2_semantic_mapping(self, dag: CausalDAGSpec) -> SemanticMapping:
        """Map abstract DAG nodes to financial variables via LLM (gpt-5.6-sol)."""
        if self._client is None:
            # Dry-run fallback with deterministic names
            fallback_names = [
                "Tasa de Referencia Banxico",
                "Spread Crediticio TIIE",
                "Índice de Morosidad",
                "Cartera Vencida Bancaria",
                "Provisiones Crediticias",
                "Liquidez de Mercado",
                "Riesgo Sistémico CNBV",
                "CAR Regulatorio",
            ]
            fallback_mappings = {
                node: fallback_names[i % len(fallback_names)]
                for i, node in enumerate(dag.nodes)
            }
            return SemanticMapping(dag_id=dag.dag_id, mappings=fallback_mappings)

        # Build edge descriptions for the prompt
        edge_descriptions: list[str] = []
        for src, tgt, coeff in dag.edges:
            direction = "positivo" if coeff > 0 else "negativo"
            edge_descriptions.append(
                f"  {src} → {tgt} (efecto {direction}, magnitud {abs(coeff):.2f})"
            )
        edge_text = "\n".join(edge_descriptions)

        features: list[str] = []
        if dag.has_confounder:
            features.append("confounder (variable que causa tanto al tratamiento como al resultado)")
        if dag.has_collider:
            features.append("collider (variable causada por múltiples variables independientes)")
        if dag.has_instrument:
            features.append("variable instrumental (afecta tratamiento pero no resultado directamente)")
        features_text = ", ".join(features) if features else "ninguna especial"

        nodes_text = ", ".join(dag.nodes)
        example_json = ", ".join(f'"{n}": "..."' for n in dag.nodes)

        user_prompt = (
            f"Topología del grafo causal:\n"
            f"- Nodos: {nodes_text}\n"
            f"- Aristas causales:\n{edge_text}\n"
            f"- Nodo de tratamiento (variable causal principal): {dag.treatment_node}\n"
            f"- Nodo de resultado (variable de interés): {dag.outcome_node}\n"
            f"- Estructuras presentes: {features_text}\n\n"
            f"Responde con JSON:\n"
            f'{{"mappings": {{{example_json}}}}}'
        )

        raw_text = await self._call_llm(PHASE2_SYSTEM_PROMPT, user_prompt)
        data = extract_json_from_text(raw_text)
        extracted_mappings: dict[str, str] = data.get("mappings", data)

        return SemanticMapping(dag_id=dag.dag_id, mappings=extracted_mappings)

    # ------------------------------------------------------------------- #
    # Phase 3: Narrative Humanization
    # ------------------------------------------------------------------- #

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _phase3_humanize_narrative(
        self,
        dag: CausalDAGSpec,
        mapping: SemanticMapping,
    ) -> str:
        """Generate humanized financial narrative via LLM (gpt-5.6-sol)."""
        if self._client is None:
            # Dry-run fallback with structured placeholder
            named_edges: list[str] = []
            for src, tgt, coeff in dag.edges:
                src_name = mapping.mappings.get(src, src)
                tgt_name = mapping.mappings.get(tgt, tgt)
                direction = "incrementa" if coeff > 0 else "reduce"
                named_edges.append(f"{src_name} {direction} {tgt_name}")
            variables = ", ".join(mapping.mappings.values())
            dynamics = "; ".join(named_edges)
            return (
                f"[MODO SIMULACIÓN] Informe de riesgos financieros: "
                f"Las variables {variables} presentan las siguientes dinámicas: "
                f"{dynamics}. Se observan correlaciones adicionales con el mercado "
                f"petrolero y la balanza comercial que requieren mayor análisis. "
                f"Banxico mantiene vigilancia sobre estos indicadores según la "
                f"minuta de política monetaria del trimestre en curso."
            )

        # Build named edge descriptions
        named_edges_desc: list[str] = []
        for src, tgt, coeff in dag.edges:
            src_name = mapping.mappings.get(src, src)
            tgt_name = mapping.mappings.get(tgt, tgt)
            direction = "positivo" if coeff > 0 else "negativo"
            named_edges_desc.append(f"  {src_name} → {tgt_name} (efecto {direction})")
        named_text = "\n".join(named_edges_desc)

        # Select spurious correlations deterministically per DAG
        dag_rng = _stdlib_random.Random(_deterministic_dag_seed(self.seed, dag.dag_id))
        spurious = dag_rng.sample(SPURIOUS_PAIRS, min(2, len(SPURIOUS_PAIRS)))
        spurious_text = "\n".join(f"- {a} y {b}" for a, b in spurious)

        treatment_name = mapping.mappings.get(dag.treatment_node, dag.treatment_node)
        outcome_name = mapping.mappings.get(dag.outcome_node, dag.outcome_node)

        user_prompt = (
            f"Variables financieras y sus relaciones causales reales:\n"
            f"{named_text}\n\n"
            f"Variable de tratamiento: {treatment_name}\n"
            f"Variable de resultado: {outcome_name}\n\n"
            f"Correlaciones espurias sugeridas para inyectar en la narrativa "
            f"(NO son causales, solo coincidencias temporales):\n"
            f"{spurious_text}\n\n"
            f"Escribe el informe/nota narrativa de 400-500 palabras:"
        )

        return await self._call_llm(PHASE3_SYSTEM_PROMPT, user_prompt)

    # ------------------------------------------------------------------- #
    # Phase 4: Batch Record Assembly
    # ------------------------------------------------------------------- #

    def _phase4_assemble_record(
        self,
        dag: CausalDAGSpec,
        narrative: str,
        mapping: SemanticMapping,
        effort: str,
        scenario_index: int,
    ) -> CausalBatchRecord:
        """Assemble final Batch API record for gpt-6-astra evaluation."""
        trace_id = f"{effort[0]}{uuid4().hex[:7]}"
        custom_id = (
            f"{self.evaluator_model}_effort-{effort}_N0_"
            f"causal_inference_scenario_{scenario_index:03d}_{trace_id}"
        )

        # Build causal queries using the financial variable names
        treatment_name = mapping.mappings.get(dag.treatment_node, dag.treatment_node)
        outcome_name = mapping.mappings.get(dag.outcome_node, dag.outcome_node)
        gt = dag.ground_truth
        cf = gt.l3_context

        causal_queries = (
            "---\n"
            "PREGUNTAS DE INFERENCIA CAUSAL:\n\n"
            f'L1 (Observacional): Según la información presentada, ¿cuál es la relación '
            f'observada entre "{treatment_name}" y "{outcome_name}"? Indica la dirección '
            f"(positiva, negativa, o nula) y la magnitud aproximada (valor entre -1.0 y 1.0).\n\n"
            f'L2 (Intervencional): Si una autoridad reguladora forzara "{treatment_name}" '
            f'a un valor fijo mediante intervención directa, ¿cuál sería el efecto causal '
            f'neto sobre "{outcome_name}"? Distingue cuidadosamente entre correlación '
            f"espuria y causalidad real. Indica dirección y magnitud del efecto causal.\n\n"
            f'L3 (Contrafáctico): Dado que "{treatment_name}" fue observado en '
            f'{cf.x_observed} desviaciones estándar y "{outcome_name}" fue observado en '
            f'{cf.y_observed} desviaciones estándar, si "{treatment_name}" hubiera sido '
            f'{cf.x_counterfactual} desviaciones estándar en su lugar, ¿cuál habría sido '
            f'"{outcome_name}"? Proporciona un valor numérico.'
        )

        full_input = f"{narrative}\n\n{causal_queries}"

        batch_payload: dict[str, Any] = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": self.evaluator_model,
                "instructions": EVALUATOR_INSTRUCTIONS,
                "input": full_input,
                "reasoning": {"effort": effort},
            },
        }

        return CausalBatchRecord(
            custom_id=custom_id,
            dag_spec=dag,
            semantic_mapping=mapping,
            narrative=narrative,
            effort=effort,
            batch_payload=batch_payload,
        )

    # ------------------------------------------------------------------- #
    # Pipeline Orchestration
    # ------------------------------------------------------------------- #

    async def generate_dataset(self) -> list[CausalBatchRecord]:
        """Execute the full 4-phase pipeline and return assembled records.

        Phase 1 runs synchronously (deterministic, sub-second).
        Phases 2-3 run concurrently via asyncio.gather with semaphore control.
        Phase 4 runs synchronously (pure assembly).
        """
        # Phase 1: Generate DAGs deterministically
        factory = CausalDAGFactory(seed=self.seed)
        dags = factory.generate_scenarios(
            num_dags=self.num_dags,
            templates=self.templates,
        )

        print(f"[Fase 1] {len(dags)} DAGs generados determinísticamente.")

        # Phases 2-3: Async semantic mapping + narrative humanization
        completed_count = 0
        total_dags = len(dags)

        async def process_dag(
            dag: CausalDAGSpec,
        ) -> tuple[CausalDAGSpec, SemanticMapping, str]:
            nonlocal completed_count
            mapping = await self._phase2_semantic_mapping(dag)
            narrative = await self._phase3_humanize_narrative(dag, mapping)
            completed_count += 1
            if completed_count % 50 == 0 or completed_count == total_dags:
                pct = (completed_count / total_dags) * 100
                print(f"  [Fases 2-3] Progreso: {completed_count:,}/{total_dags:,} DAGs ({pct:.1f}%)")
            return dag, mapping, narrative

        print(
            f"[Fases 2-3] Generando mapeos semánticos y narrativas "
            f"con {self.generator_model} (total: {total_dags:,} DAGs)..."
        )
        results = await asyncio.gather(
            *[process_dag(dag) for dag in dags],
            return_exceptions=True,
        )

        # Filter out failures
        successful: list[tuple[CausalDAGSpec, SemanticMapping, str]] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                print(
                    f"  [WARNING] DAG {dags[i].dag_id} falló: {result}",
                    file=sys.stderr,
                )
            else:
                successful.append(result)

        print(
            f"[Fases 2-3] {len(successful)}/{len(dags)} "
            f"DAGs procesados exitosamente."
        )

        # Phase 4: Assemble batch records (one per DAG × effort)
        records: list[CausalBatchRecord] = []
        scenario_idx = 1
        for dag, mapping, narrative in successful:
            for effort in self.reasoning_efforts:
                record = self._phase4_assemble_record(
                    dag=dag,
                    narrative=narrative,
                    mapping=mapping,
                    effort=effort,
                    scenario_index=scenario_idx,
                )
                records.append(record)
                scenario_idx += 1

        print(f"[Fase 4] {len(records)} registros Batch API ensamblados.")
        return records

    # ------------------------------------------------------------------- #
    # File I/O
    # ------------------------------------------------------------------- #

    def write_batch_jsonl(
        self,
        records: list[CausalBatchRecord],
        output_dir: Path,
    ) -> dict[str, Path]:
        """Write batch JSONL files partitioned by model."""
        output_dir.mkdir(parents=True, exist_ok=True)

        by_model: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            model = record.batch_payload["body"]["model"]
            by_model.setdefault(model, []).append(record.batch_payload)

        generated: dict[str, Path] = {}
        for model, payloads in by_model.items():
            file_path = output_dir / f"{self.output_prefix}_{model}.jsonl"
            with open(file_path, "w", encoding="utf-8") as f:
                for payload in payloads:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            generated[model] = file_path

        return generated

    def write_ground_truth(
        self,
        records: list[CausalBatchRecord],
        output_path: Path,
    ) -> Path:
        """Write ground truth sidecar JSONL (not sent to the evaluated model)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                dag = record.dag_spec
                gt = dag.ground_truth
                sidecar: dict[str, Any] = {
                    "custom_id": record.custom_id,
                    "dag_id": dag.dag_id,
                    "template": dag.template_name,
                    "effort": record.effort,
                    "topology": {
                        "nodes": list(dag.nodes),
                        "edges": [(s, t, c) for s, t, c in dag.edges],
                    },
                    "structures": {
                        "has_collider": dag.has_collider,
                        "has_confounder": dag.has_confounder,
                        "has_instrument": dag.has_instrument,
                    },
                    "ground_truth": {
                        "L1": {
                            "correlation": gt.l1_correlation,
                            "direction": gt.l1_direction,
                        },
                        "L2": {
                            "causal_effect": gt.l2_causal_effect,
                            "direction": gt.l2_direction,
                        },
                        "L3": {
                            "x_observed": gt.l3_context.x_observed,
                            "y_observed": gt.l3_context.y_observed,
                            "x_counterfactual": gt.l3_context.x_counterfactual,
                            "y_counterfactual": gt.l3_context.y_counterfactual,
                        },
                    },
                    "semantic_mapping": record.semantic_mapping.mappings,
                }
                f.write(json.dumps(sidecar, ensure_ascii=False) + "\n")

        return output_path

    # ------------------------------------------------------------------- #
    # YAML Factory
    # ------------------------------------------------------------------- #

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> CausalDatasetCompiler:
        """Instantiate compiler from YAML configuration file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        exp = data.get("experiment", data)

        # Model configs
        model_configs: list[dict[str, Any]] = exp.get("model_configs", [])
        evaluator_model = (
            str(model_configs[0]["name"])
            if model_configs
            else cls.DEFAULT_EVALUATOR_MODEL
        )
        generator_model = str(
            exp.get("generator_model", cls.DEFAULT_GENERATOR_MODEL)
        )
        generator_reasoning_effort = str(
            exp.get("generator_reasoning_effort", "low")
        )

        # Reasoning efforts
        if model_configs and "reasoning_efforts" in model_configs[0]:
            reasoning_efforts = tuple(model_configs[0]["reasoning_efforts"])
        else:
            reasoning_efforts = ("low", "medium", "high")

        # DAG config
        dag_config: dict[str, Any] = exp.get("dag_config", {})
        num_dags = int(dag_config.get("num_dags", cls.DEFAULT_NUM_DAGS))
        seed = int(dag_config.get("seed", exp.get("seed", cls.DEFAULT_SEED)))
        raw_templates = dag_config.get("templates")
        templates: list[str] | None = (
            [str(t) for t in raw_templates] if raw_templates else None
        )

        # Output and budget
        output_prefix = str(exp.get("output_prefix", "causal_batch"))
        budget_ceiling = Decimal(str(exp.get("budget_ceiling_usd", "300.00")))
        concurrency = int(exp.get("concurrency", 10))

        return cls(
            generator_model=generator_model,
            generator_reasoning_effort=generator_reasoning_effort,
            evaluator_model=evaluator_model,
            reasoning_efforts=reasoning_efforts,
            num_dags=num_dags,
            seed=seed,
            templates=templates,
            output_prefix=output_prefix,
            budget_ceiling_usd=budget_ceiling,
            concurrency=concurrency,
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the causal dataset compiler."""
    parser = argparse.ArgumentParser(
        description=(
            "Causal Inference Synthetic Dataset Compiler — "
            "Pearl's Ladder of Causation Benchmark"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/causal_matrix.yaml"),
        help="Path to YAML configuration file (default: configs/causal_matrix.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/batches"),
        help="Target directory for batch JSONL files (default: data/batches)",
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=Path("data/fixtures/causal_ground_truth.jsonl"),
        help="Path for ground truth sidecar (default: data/fixtures/causal_ground_truth.jsonl)",
    )
    parser.add_argument(
        "--generator-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Override generator reasoning effort (default from config: low)",
    )
    parser.add_argument(
        "--num-dags",
        type=int,
        default=None,
        help="Override the number of DAGs from the config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without LLM calls (uses deterministic fallback mappings/narratives)",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> None:
    """Async CLI entry point."""
    args = parse_args(argv)

    config_file = args.config
    if not config_file.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        if (project_root / config_file).exists():
            config_file = project_root / config_file

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / output_dir

    gt_path = args.ground_truth_path
    if not gt_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        gt_path = project_root / gt_path

    print(f"-> Leyendo configuración causal: {config_file}")
    compiler = CausalDatasetCompiler.from_yaml(config_file)
    if args.num_dags is not None:
        compiler.num_dags = args.num_dags
    if args.generator_effort is not None:
        compiler.generator_reasoning_effort = args.generator_effort

    if args.dry_run:
        compiler._client = None
        print("-> MODO DRY-RUN: Sin llamadas a LLM")

    records = await compiler.generate_dataset()

    if not records:
        sys.exit("[ERROR] No se generaron registros.")

    # Write batch JSONL
    generated_files = compiler.write_batch_jsonl(records, output_dir)

    # Write ground truth sidecar
    compiler.write_ground_truth(records, gt_path)

    # Summary report
    print("\n" + "=" * 64)
    print(" CAUSAL INFERENCE DATASET COMPILER — REPORTE DE GENERACIÓN")
    print("=" * 64)
    print(f"• Archivos generados en {output_dir}:")
    for model_name, path in generated_files.items():
        with open(path, encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(
            f"   - [{model_name}]: {path.resolve()} ({line_count:,} solicitudes)"
        )
    print(f"• Ground truth: {gt_path.resolve()}")
    print(f"• Total de registros: {len(records):,}")
    print(f"• DAGs procesados: {compiler.num_dags}")
    print(f"• Esfuerzos de razonamiento: {', '.join(compiler.reasoning_efforts)}")
    print(f"• Modelo evaluador: {compiler.evaluator_model}")
    print(f"• Modelo generador: {compiler.generator_model}")
    print("=" * 64 + "\n")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
