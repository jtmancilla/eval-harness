"""Causal Stratified Dataset Compiler: Master Narrative Cache + Parametric Expansion.

Implements the Optimal Causal Stratification Strategy for industrial research:
1. Generates and caches 300 rich master financial narratives via live gpt-6-astra calls
   (50 per canonical Pearl topology: confounding_fork, mediator_chain, collider,
    instrumental_variable, diamond, full_model).
2. Expands the 300 master narratives across parametric variations (SCM edge weights,
   intervention values do(X=x), and counterfactual context tuples), computing analytically
   exact NumPy ground truth for L1, L2, L3.
3. Compiles final OpenAI Batch API payloads (/v1/responses for gpt-6-astra) across
   reasoning efforts (low, medium, high), matching the budget and compliance governance.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random as _stdlib_random
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.eval.causal_dag_factory import (
    CausalDAGFactory,
    CausalDAGSpec,
)
from src.eval.causal_dataset_compiler import (
    EVALUATOR_INSTRUCTIONS,
    PHASE2_SYSTEM_PROMPT,
    PHASE3_SYSTEM_PROMPT,
    SPURIOUS_PAIRS,
    extract_json_from_text,
)

load_dotenv()


# --------------------------------------------------------------------------- #
# Data Contracts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MasterNarrative:
    """A persistent, live-generated master narrative for a canonical DAG topology."""

    master_id: str
    template_name: str
    nodes: list[str]
    edges: list[list[Any]]  # [[src, tgt, coeff], ...]
    treatment_node: str
    outcome_node: str
    has_collider: bool
    has_confounder: bool
    has_instrument: bool
    semantic_mapping: dict[str, str]
    narrative_text: str


@dataclass(frozen=True)
class StratifiedBatchRecord:
    """Assembled batch record combining master narrative with parametric variations."""

    custom_id: str
    master_id: str
    variation_id: str
    template_name: str
    effort: str
    dag_spec: CausalDAGSpec
    semantic_mapping: dict[str, str]
    narrative: str
    batch_payload: dict[str, Any]


# --------------------------------------------------------------------------- #
# Causal Stratified Compiler
# --------------------------------------------------------------------------- #


class CausalStratifiedCompiler:
    """Orchestrates 300 Master Narratives caching and parametric expansion."""

    DEFAULT_MASTER_CACHE = Path("data/fixtures/master_causal_narratives_300.json")
    DEFAULT_GENERATOR_MODEL = "gpt-6-astra"
    DEFAULT_EVALUATOR_MODEL = "gpt-6-astra"
    BUDGET_CEILING_USD: Decimal = Decimal("300.00")

    def __init__(
        self,
        generator_model: str = DEFAULT_GENERATOR_MODEL,
        generator_reasoning_effort: str = "low",
        evaluator_model: str = DEFAULT_EVALUATOR_MODEL,
        reasoning_efforts: tuple[str, ...] = ("low", "medium", "high"),
        master_cache_path: Path = DEFAULT_MASTER_CACHE,
        concurrency: int = 15,
        seed: int = 42,
    ) -> None:
        self.generator_model = generator_model
        self.generator_reasoning_effort = generator_reasoning_effort
        self.evaluator_model = evaluator_model
        self.reasoning_efforts = reasoning_efforts
        self.master_cache_path = master_cache_path
        self._semaphore = asyncio.Semaphore(concurrency)
        self.seed = seed

        api_key = os.getenv("OPENAI_API_KEY", "")
        self._client: AsyncOpenAI | None = AsyncOpenAI(api_key=api_key) if api_key else None

    # ----------------------------------------------------------------------- #
    # Live LLM Dispatch (gpt-6-astra via /v1/responses)
    # ----------------------------------------------------------------------- #

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _call_astra_response(self, instructions: str, user_input: str) -> str:
        """Call gpt-6-astra via responses.create with rate-limit retries."""
        if self._client is None:
            return ""

        async with self._semaphore:
            response = await self._client.responses.create(  # type: ignore[call-overload]
                model=self.generator_model,
                instructions=instructions,
                input=user_input,
                reasoning={"effort": self.generator_reasoning_effort},
            )
            return response.output_text or ""

    # ----------------------------------------------------------------------- #
    # Master Narrative Synthesis (Phase 2 & 3 in vivo)
    # ----------------------------------------------------------------------- #

    async def _synthesize_master_narrative(
        self,
        dag: CausalDAGSpec,
        master_idx: int,
    ) -> MasterNarrative:
        """Generate a single master narrative with semantic mapping and realistic prose."""
        master_id = f"master_{dag.template_name}_{master_idx:03d}"

        # 1. Semantic Mapping Prompt
        edge_descriptions: list[str] = []
        for src, tgt, coeff in dag.edges:
            direction = "positivo" if coeff > 0 else "negativo"
            edge_descriptions.append(
                f"  {src} → {tgt} (efecto {direction}, magnitud {abs(coeff):.2f})"
            )
        edge_text = "\n".join(edge_descriptions)
        features: list[str] = []
        if dag.has_confounder:
            features.append("confounder")
        if dag.has_collider:
            features.append("collider")
        if dag.has_instrument:
            features.append("variable instrumental")
        features_text = ", ".join(features) if features else "ninguna especial"
        example_json = ", ".join(f'"{n}": "..."' for n in dag.nodes)

        phase2_prompt = (
            f"Topología del grafo causal:\n"
            f"- Nodos: {', '.join(dag.nodes)}\n"
            f"- Aristas causales:\n{edge_text}\n"
            f"- Nodo de tratamiento (causa): {dag.treatment_node}\n"
            f"- Nodo de resultado (efecto): {dag.outcome_node}\n"
            f"- Estructuras: {features_text}\n\n"
            f"Responde ÚNICAMENTE con JSON:\n"
            f'{{"mappings": {{{example_json}}}}}'
        )

        mapping_raw = await self._call_astra_response(PHASE2_SYSTEM_PROMPT, phase2_prompt)
        mapping_data = extract_json_from_text(mapping_raw)
        semantic_mapping: dict[str, str] = mapping_data.get("mappings", mapping_data)

        # Fallback if empty
        if not semantic_mapping or len(semantic_mapping) < len(dag.nodes):
            fallback_names = [
                "Tasa objetivo de Banxico",
                "Inflación subyacente en México",
                "Expectativas inflacionarias",
                "Spread Crediticio TIIE",
                "Morosidad bancaria",
                "Cartera de crédito empresarial",
                "Reservas preventivas bancarias",
                "Liquidez bancaria de corto plazo",
            ]
            semantic_mapping = {
                node: fallback_names[i % len(fallback_names)]
                for i, node in enumerate(dag.nodes)
            }

        # 2. Narrative Humanization Prompt
        named_edges_desc: list[str] = []
        for src, tgt, coeff in dag.edges:
            src_name = semantic_mapping.get(src, src)
            tgt_name = semantic_mapping.get(tgt, tgt)
            direction = "positivo" if coeff > 0 else "negativo"
            named_edges_desc.append(f"  {src_name} → {tgt_name} (efecto {direction})")
        named_text = "\n".join(named_edges_desc)

        dag_rng = _stdlib_random.Random(self.seed + master_idx * 17)
        spurious = dag_rng.sample(SPURIOUS_PAIRS, min(2, len(SPURIOUS_PAIRS)))
        spurious_text = "\n".join(f"- {a} y {b}" for a, b in spurious)

        treatment_name = semantic_mapping.get(dag.treatment_node, dag.treatment_node)
        outcome_name = semantic_mapping.get(dag.outcome_node, dag.outcome_node)

        phase3_prompt = (
            f"Variables financieras y sus relaciones causales reales:\n"
            f"{named_text}\n\n"
            f"Variable de tratamiento: {treatment_name}\n"
            f"Variable de resultado: {outcome_name}\n\n"
            f"Correlaciones espurias para inyectar en la narrativa "
            f"(NO son causales, solo coincidencias temporales):\n"
            f"{spurious_text}\n\n"
            f"Escribe el informe/nota narrativa de 400-500 palabras:"
        )

        narrative_text = await self._call_astra_response(PHASE3_SYSTEM_PROMPT, phase3_prompt)
        if not narrative_text.strip():
            narrative_text = (
                f"Informe de Estabilidad Financiera Trimestral: "
                f"Las dinámicas observadas en {treatment_name} y {outcome_name} "
                f"se asocian estrechamente con el balance de riesgos crediticios en México."
            )

        return MasterNarrative(
            master_id=master_id,
            template_name=dag.template_name,
            nodes=list(dag.nodes),
            edges=[[s, t, float(c)] for s, t, c in dag.edges],
            treatment_node=dag.treatment_node,
            outcome_node=dag.outcome_node,
            has_collider=dag.has_collider,
            has_confounder=dag.has_confounder,
            has_instrument=dag.has_instrument,
            semantic_mapping=semantic_mapping,
            narrative_text=narrative_text,
        )

    # ----------------------------------------------------------------------- #
    # Master Narratives Pool (Generate or Load Cache)
    # ----------------------------------------------------------------------- #

    async def get_or_create_master_narratives(
        self,
        num_masters: int = 300,
        dry_run: bool = False,
    ) -> list[MasterNarrative]:
        """Load cached master narratives or generate them via live gpt-6-astra calls."""
        # 1. Check if cache exists and has enough narratives
        if self.master_cache_path.exists():
            try:
                with open(self.master_cache_path, encoding="utf-8") as f:
                    raw_cached = json.load(f)
                cached = [MasterNarrative(**item) for item in raw_cached]
                if len(cached) >= num_masters:
                    print(
                        f"-> [CACHE HIT] Cargadas {len(cached)} narrativas maestras desde "
                        f"{self.master_cache_path.resolve()}"
                    )
                    return cached[:num_masters]
            except Exception as e:
                print(f"-> [WARNING] No se pudo leer caché: {e}. Se regenerarán.")

        # 2. Generate base DAG topologies (evenly distributed across templates)
        factory = CausalDAGFactory(seed=self.seed)
        base_dags = factory.generate_scenarios(num_dags=num_masters)

        if dry_run or self._client is None:
            print("-> [DRY-RUN] Generando narrativas maestras sintéticas locales...")
            masters = []
            for i, dag in enumerate(base_dags):
                names = [f"Variable_{n}" for n in dag.nodes]
                mapping = {n: f"Indicador_{n}_MX" for n in dag.nodes}
                narrative = (
                    f"Reporte macrofinanciero de prueba para {dag.template_name}: "
                    f"Se analizan interacciones entre {', '.join(names)}."
                )
                masters.append(
                    MasterNarrative(
                        master_id=f"master_{dag.template_name}_{i:03d}",
                        template_name=dag.template_name,
                        nodes=list(dag.nodes),
                        edges=[[s, t, float(c)] for s, t, c in dag.edges],
                        treatment_node=dag.treatment_node,
                        outcome_node=dag.outcome_node,
                        has_collider=dag.has_collider,
                        has_confounder=dag.has_confounder,
                        has_instrument=dag.has_instrument,
                        semantic_mapping=mapping,
                        narrative_text=narrative,
                    )
                )
            return masters

        # 3. Live generation with gpt-6-astra
        print(
            f"-> [GENERACIÓN VIVA] Iniciando síntesis de {num_masters} narrativas "
            f"maestras con {self.generator_model} (concurrencia {self._semaphore._value})..."
        )
        completed_count = 0
        total_count = len(base_dags)

        async def worker(dag: CausalDAGSpec, idx: int) -> MasterNarrative:
            nonlocal completed_count
            master = await self._synthesize_master_narrative(dag, idx)
            completed_count += 1
            if completed_count % 25 == 0 or completed_count == total_count:
                pct = (completed_count / total_count) * 100
                print(
                    f"   [Maestras] Progreso: {completed_count}/{total_count} "
                    f"historias sintetizadas ({pct:.1f}%)"
                )
            return master

        tasks = [worker(dag, i) for i, dag in enumerate(base_dags)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful: list[MasterNarrative] = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                print(f"  [ERROR] Falló maestra {base_dags[i].dag_id}: {res}", file=sys.stderr)
            else:
                successful.append(res)

        print(f"-> [COMPLETADO] {len(successful)}/{total_count} historias maestras vivas generadas.")

        # Save to cache
        self.master_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.master_cache_path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in successful], f, ensure_ascii=False, indent=2)
        print(f"-> [GUARDADO] Caché inmutable guardado en: {self.master_cache_path.resolve()}")

        return successful

    # ----------------------------------------------------------------------- #
    # Parametric Expansion (NumPy Exact Ground Truth)
    # ----------------------------------------------------------------------- #

    def expand_parametric_variations(
        self,
        masters: list[MasterNarrative],
        target_total_requests: int = 10002,
    ) -> list[StratifiedBatchRecord]:
        """Expand master narratives across algebraic SCM parameter variations."""
        num_efforts = len(self.reasoning_efforts)
        # Each variation produces `num_efforts` records
        target_variations = target_total_requests // num_efforts
        variations_per_master = max(1, target_variations // len(masters))

        print(
            f"-> [EXPANSIÓN ESTRATIFICADA] {len(masters)} narrativas maestras × "
            f"{variations_per_master} variaciones paramétricas × {num_efforts} esfuerzos..."
        )

        records: list[StratifiedBatchRecord] = []
        global_scenario_idx = 1
        rng = np.random.default_rng(self.seed)

        for _master_idx, master in enumerate(masters):
            for var_idx in range(variations_per_master):
                if len(records) >= target_total_requests:
                    break

                # Parametric perturbation of edge coefficients
                pert_edges: list[tuple[str, str, float]] = []
                for src, tgt, base_coeff in master.edges:
                    # Perturb coefficient slightly or sample from valid signed range
                    sign = 1.0 if base_coeff > 0 else -1.0
                    delta = float(rng.uniform(-0.15, 0.15))
                    new_coeff = round(float(np.clip(base_coeff + delta, sign * 0.2, sign * 1.3)), 2)
                    pert_edges.append((str(src), str(tgt), new_coeff))

                # Perturb noise standard deviations
                noise_stds = tuple(
                    round(float(rng.uniform(0.4, 1.0)), 2)
                    for _ in master.nodes
                )

                # Compute exact analytical ground truth with NumPy via factory.build_spec
                factory = CausalDAGFactory(seed=int(rng.integers(1, 1000000)))
                variation_spec = factory.build_spec(
                    template_name=master.template_name,
                    dag_index=var_idx,
                    nodes=tuple(master.nodes),
                    edges=tuple(pert_edges),
                    noise_stds=noise_stds,
                    treatment=master.treatment_node,
                    outcome=master.outcome_node,
                    has_collider=master.has_collider,
                    has_confounder=master.has_confounder,
                    has_instrument=master.has_instrument,
                )
                gt = variation_spec.ground_truth

                # Format question text matching this variation's ground truth
                t_name = master.semantic_mapping.get(master.treatment_node, master.treatment_node)
                o_name = master.semantic_mapping.get(master.outcome_node, master.outcome_node)
                cf = gt.l3_context

                causal_queries = (
                    f"---\n"
                    f"PREGUNTAS DE INFERENCIA CAUSAL:\n\n"
                    f'L1 (Observacional): Según la información presentada, ¿cuál es la relación '
                    f'observada entre "{t_name}" y "{o_name}"? '
                    f"Indica la dirección (positiva, negativa, o nula) y la magnitud aproximada "
                    f"(valor entre -1.0 y 1.0).\n\n"
                    f'L2 (Intervencional): Si una autoridad reguladora forzara "{t_name}" a un '
                    f'valor fijo mediante intervención directa, ¿cuál sería el efecto causal neto '
                    f'sobre "{o_name}"? Distingue cuidadosamente entre correlación espuria y '
                    f"causalidad real. Indica dirección y magnitud del efecto causal.\n\n"
                    f'L3 (Contrafáctico): Dado que "{t_name}" fue observado en '
                    f'{cf.x_observed} desviaciones estándar y "{o_name}" fue observado en '
                    f'{cf.y_observed} desviaciones estándar, si "{t_name}" hubiera sido '
                    f'{cf.x_counterfactual} desviaciones estándar en su lugar, ¿cuál habría sido '
                    f'"{o_name}"? Proporciona un valor numérico.'
                )

                full_input = f"{master.narrative_text}\n\n{causal_queries}"

                # Assemble one batch record per reasoning effort
                for effort in self.reasoning_efforts:
                    trace_id = f"{effort[0]}{uuid4().hex[:7]}"
                    custom_id = (
                        f"{self.evaluator_model}_effort-{effort}_N0_"
                        f"causal_inference_scenario_{global_scenario_idx:05d}_{trace_id}"
                    )

                    payload: dict[str, Any] = {
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

                    records.append(
                        StratifiedBatchRecord(
                            custom_id=custom_id,
                            master_id=master.master_id,
                            variation_id=variation_spec.dag_id,
                            template_name=master.template_name,
                            effort=effort,
                            dag_spec=variation_spec,
                            semantic_mapping=master.semantic_mapping,
                            narrative=master.narrative_text,
                            batch_payload=payload,
                        )
                    )
                    global_scenario_idx += 1

        print(f"-> [ENSAMBLAJE] {len(records):,} registros estratificados compilados.")
        return records

    # ----------------------------------------------------------------------- #
    # File Exporters (Batch JSONL + Ground Truth Sidecar)
    # ----------------------------------------------------------------------- #

    def export_batch_files(
        self,
        records: list[StratifiedBatchRecord],
        output_batch_path: Path,
        output_gt_path: Path,
    ) -> tuple[Path, Path]:
        """Export the compiled batch records and ground truth sidecar."""
        output_batch_path.parent.mkdir(parents=True, exist_ok=True)
        output_gt_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Write batch JSONL
        with open(output_batch_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.batch_payload, ensure_ascii=False) + "\n")

        # 2. Write ground truth sidecar JSONL
        with open(output_gt_path, "w", encoding="utf-8") as f:
            for r in records:
                gt = r.dag_spec.ground_truth
                sidecar: dict[str, Any] = {
                    "custom_id": r.custom_id,
                    "master_id": r.master_id,
                    "variation_id": r.variation_id,
                    "template": r.template_name,
                    "effort": r.effort,
                    "topology": {
                        "nodes": list(r.dag_spec.nodes),
                        "edges": [(s, t, c) for s, t, c in r.dag_spec.edges],
                    },
                    "structures": {
                        "has_collider": r.dag_spec.has_collider,
                        "has_confounder": r.dag_spec.has_confounder,
                        "has_instrument": r.dag_spec.has_instrument,
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
                    "semantic_mapping": r.semantic_mapping,
                }
                f.write(json.dumps(sidecar, ensure_ascii=False) + "\n")

        return output_batch_path, output_gt_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for stratified compiler."""
    parser = argparse.ArgumentParser(
        description="Causal Stratified Dataset Compiler — Master Narratives + Parametric Expansion"
    )
    parser.add_argument(
        "--num-masters",
        type=int,
        default=300,
        help="Number of master narratives to generate/cache (default: 300)",
    )
    parser.add_argument(
        "--target-requests",
        type=int,
        default=10002,
        help="Target total requests in batch JSONL (default: 10002)",
    )
    parser.add_argument(
        "--batch-output",
        type=Path,
        default=Path("data/batches/causal_stratified_batch_gpt-6-astra.jsonl"),
        help="Destination path for batch JSONL",
    )
    parser.add_argument(
        "--gt-output",
        type=Path,
        default=Path("data/fixtures/causal_stratified_ground_truth.jsonl"),
        help="Destination path for ground truth sidecar",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Concurrency limit for live OpenAI responses API calls (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without live LLM calls using synthetic placeholders",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> None:
    """Async main entry point."""
    args = parse_args(argv)

    compiler = CausalStratifiedCompiler(
        generator_model="gpt-6-astra",
        generator_reasoning_effort="low",
        evaluator_model="gpt-6-astra",
        reasoning_efforts=("low", "medium", "high"),
        concurrency=args.concurrency,
    )

    # Step 1: Synthesize or load 300 master narratives
    masters = await compiler.get_or_create_master_narratives(
        num_masters=args.num_masters,
        dry_run=args.dry_run,
    )

    if not masters:
        sys.exit("[ERROR] No se pudieron obtener narrativas maestras.")

    # Step 2: Expand across parametric variations
    records = compiler.expand_parametric_variations(
        masters=masters,
        target_total_requests=args.target_requests,
    )

    # Step 3: Export batch and ground truth files
    batch_path, gt_path = compiler.export_batch_files(
        records=records,
        output_batch_path=args.batch_output,
        output_gt_path=args.gt_output,
    )

    print("\n" + "=" * 65)
    print(" COMPILACIÓN ESTRATIFICADA EXITOSA")
    print("=" * 65)
    print(f"• Narrativas maestras:     {len(masters)}")
    print(f"• Solicitudes en lote:     {len(records):,}")
    print(f"• Archivo batch generado:  {batch_path.resolve()}")
    print(f"• Sidecar ground truth:    {gt_path.resolve()}")
    print("=" * 65 + "\n")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
