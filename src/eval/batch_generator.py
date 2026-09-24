"""OpenAI Batch API dataset compiler and cost estimator under lexical entropy."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml  # type: ignore[import-untyped]

from src.agents.router import HierarchicalRouter
from src.agents.schemas import RoutingDecision
from src.tools.decoys import generate_decoy_catalog


class BenchmarkCondition(StrEnum):
    """Execution conditions for the multi-agent benchmark."""

    BASELINE_AUTORREGRESIVO = "baseline_autorregresivo"
    NEUROSYMBOLIC_HANDOFF = "neurosymbolic_handoff"
    HIERARCHICAL_ROUTER = "hierarchical_router"


class ScenarioCategory(StrEnum):
    """Distribution categories for synthetic test cases."""

    HAPPY_PATH = "happy_path"
    CLABE_ARITHMETIC_FAILURE = "clabe_arithmetic_failure"
    SAT_REGULATORY_BLOCK = "sat_regulatory_block"
    RFC_SYNTAX_FAILURE = "rfc_syntax_failure"


@dataclass(frozen=True)
class SyntheticScenario:
    """Deterministic synthetic test case payload."""

    scenario_id: str
    category: ScenarioCategory
    clabe: str
    rfc: str
    titular: str
    monto: Decimal
    concepto: str
    regimen_fiscal: str
    uso_cfdi: str
    codigo_postal: str


@dataclass(frozen=True)
class CostEstimate:
    """Projected token consumption and cost against budget ceiling."""

    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: Decimal
    budget_limit_usd: Decimal
    is_within_budget: bool


class BudgetExceededError(ValueError):
    """Raised when projected batch execution cost exceeds the assigned budget ceiling."""


# Canonical 5 core tools for the financial flow
CANONICAL_BENCHMARK_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "parse_account_metadata",
            "description": "Extrae y normaliza componentes de la cuenta bancaria CLABE y nombre del titular.",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string", "description": "Texto sin estructurar que contiene la cuenta"},
                    "titular": {"type": "string", "description": "Nombre del titular"},
                },
                "required": ["raw_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_rfc_structure",
            "description": "Valida la estructura sintáctica, fecha y homoclave de un RFC según norma SAT.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfc": {"type": "string", "description": "Clave del RFC a validar"},
                },
                "required": ["rfc"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_sat_blacklist",
            "description": "Verifica si un RFC se encuentra listado en el Artículo 69-B o listas de sanción.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rfc": {"type": "string", "description": "RFC del contribuyente"},
                },
                "required": ["rfc"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_spei_fee",
            "description": "Calcula el desglose de comisión base e IVA del 16% para la transferencia SPEI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monto_base": {"type": "number", "description": "Monto base de la comisión"},
                },
                "required": ["monto_base"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_spei_instruction",
            "description": "Ensambla la orden de pago SPEI con clave de rastreo para liquidación en Banxico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monto": {"type": "number", "description": "Monto en MXN a dispersar"},
                    "cuenta_ordenante": {"type": "string", "description": "CLABE ordenante de 18 dígitos"},
                    "cuenta_beneficiario": {"type": "string", "description": "CLABE beneficiario de 18 dígitos"},
                    "concepto": {"type": "string", "description": "Concepto de la transferencia"},
                },
                "required": ["monto", "cuenta_ordenante", "cuenta_beneficiario", "concepto"],
            },
        },
    },
]


def get_benchmark_tools(entropy_n: int) -> list[dict[str, Any]]:
    """Returns exactly N tools: 5 canonical + (N - 5) decoys.

    Supports any N between 5 and 128 tools (inclusive).
    """
    if not (5 <= entropy_n <= 128):
        raise ValueError(f"Unsupported entropy level: {entropy_n}. Expected between 5 and 128.")

    decoy_count = entropy_n - len(CANONICAL_BENCHMARK_TOOLS)
    all_decoys = generate_decoy_catalog()
    selected_decoys = [d.to_openai_tool() for d in all_decoys[:decoy_count]]

    return list(CANONICAL_BENCHMARK_TOOLS) + selected_decoys


def convert_tool_to_responses_spec(tool: dict[str, Any]) -> dict[str, Any]:
    """Converts a standard chat completion tool object into /v1/responses tool format."""
    if "function" in tool:
        fn = tool["function"]
        return {
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }
    return tool


class BatchDatasetCompiler:
    """Compiles batch requests for OpenAI Batch API under lexical entropy."""

    DEFAULT_MODELS: tuple[str, ...] = (
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-6-astra",
    )
    MODELS: tuple[str, ...] = DEFAULT_MODELS
    DEFAULT_ENTROPY_LEVELS: tuple[int, ...] = (10, 50, 128)
    ENTROPY_LEVELS: tuple[int, ...] = DEFAULT_ENTROPY_LEVELS
    CONDITIONS: tuple[BenchmarkCondition, ...] = (
        BenchmarkCondition.BASELINE_AUTORREGRESIVO,
        BenchmarkCondition.NEUROSYMBOLIC_HANDOFF,
    )
    BUDGET_CEILING_USD: Decimal = Decimal("300.00")

    # Pricing table per 1M tokens (OpenAI Batch API 50% discount rate)
    MODEL_RATES: dict[str, dict[str, Decimal]] = {
        "gpt-5.6-luna": {"input": Decimal("1.25"), "output": Decimal("5.00")},
        "gpt-5.6-terra": {"input": Decimal("1.50"), "output": Decimal("6.00")},
        "gpt-5.6-sol": {"input": Decimal("2.50"), "output": Decimal("10.00")},
        "gpt-6-astra": {"input": Decimal("5.00"), "output": Decimal("20.00")},
    }

    def __init__(
        self,
        seed: int = 42,
        temperature: float = 0.0,
        entropy_levels: tuple[int, ...] = (10, 50, 128),
        models: tuple[str, ...] = (
            "gpt-5.6-luna",
            "gpt-5.6-terra",
            "gpt-5.6-sol",
            "gpt-6-astra",
        ),
        conditions: tuple[BenchmarkCondition, ...] | None = None,
        cases_per_condition: int | None = None,
        category_distribution: dict[str, float] | None = None,
        reasoning_efforts: tuple[str, ...] | None = None,
        output_file: Path | None = None,
        budget_ceiling_usd: Decimal = Decimal("300.00"),
        file_prefix: str = "eval_batch",
        is_adversarial: bool = False,
        adversarial_distribution: dict[str, int] | None = None,
        model_configs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.seed = seed
        self.temperature = temperature
        self.entropy_levels = entropy_levels
        self.model_configs = model_configs
        if self.model_configs:
            self.models = tuple(str(m["name"]) for m in self.model_configs)
        else:
            self.models = models
        self.conditions = conditions if conditions is not None else self.CONDITIONS
        self.cases_per_condition = cases_per_condition
        self.category_distribution = category_distribution or {
            "happy_path": 0.70,
            "clabe_arithmetic_failure": 0.10,
            "sat_regulatory_block": 0.10,
            "rfc_syntax_failure": 0.10,
        }
        self.reasoning_efforts = reasoning_efforts
        self.output_file = output_file
        self.budget_ceiling_usd = budget_ceiling_usd
        self.file_prefix = file_prefix
        self.is_adversarial = is_adversarial
        self.adversarial_distribution = adversarial_distribution
        self.rng = random.Random(seed)
        self.base_scenarios: list[SyntheticScenario]
        if self.is_adversarial:
            from src.eval.adversarial_scenarios import generate_adversarial_scenarios

            adv_scenarios: list[SyntheticScenario] = list(
                generate_adversarial_scenarios(self.adversarial_distribution, seed=self.seed)
            )
            self.base_scenarios = adv_scenarios
        else:
            self.base_scenarios = self._generate_base_scenarios()

    def _get_routing_decision(self, scenario: SyntheticScenario) -> RoutingDecision:
        """Determines the specialized routing decision and intent for a given scenario."""
        if scenario.category == ScenarioCategory.CLABE_ARITHMETIC_FAILURE:
            return RoutingDecision(
                target_agent="clabe_validator",
                classified_intent="VALIDATE_ACCOUNT",
                allowed_tools=["parse_account_metadata", "verify_clabe_format_v2_canonical"],
                routing_rationale="El escenario requiere validar cuenta bancaria CLABE y nombre del titular.",
            )
        elif scenario.category in (ScenarioCategory.SAT_REGULATORY_BLOCK, ScenarioCategory.RFC_SYNTAX_FAILURE):
            return RoutingDecision(
                target_agent="fiscal_validator",
                classified_intent="VALIDATE_IDENTITY",
                allowed_tools=["validate_rfc_structure", "check_sat_blacklist"],
                routing_rationale="El escenario requiere validar identidad fiscal del contribuyente y listas negras SAT.",
            )
        else:
            try:
                num = int(scenario.scenario_id.split("_")[-1])
            except (ValueError, IndexError):
                num = 0
            mod = num % 3
            if mod == 1:
                return RoutingDecision(
                    target_agent="clabe_validator",
                    classified_intent="VALIDATE_ACCOUNT",
                    allowed_tools=["parse_account_metadata", "verify_clabe_format_v2_canonical"],
                    routing_rationale="El escenario requiere validar cuenta bancaria CLABE y nombre del titular.",
                )
            elif mod == 2:
                return RoutingDecision(
                    target_agent="fiscal_validator",
                    classified_intent="VALIDATE_IDENTITY",
                    allowed_tools=["validate_rfc_structure", "check_sat_blacklist"],
                    routing_rationale="El escenario requiere validar identidad fiscal del contribuyente y listas negras SAT.",
                )
            else:
                return RoutingDecision(
                    target_agent="treasury_executor",
                    classified_intent="DISPERSE_SPEI",
                    allowed_tools=["build_spei_instruction", "calculate_spei_fee"],
                    routing_rationale="El escenario requiere dispersión de fondos SPEI y cálculo de comisiones.",
                )

    def _generate_base_scenarios(self) -> list[SyntheticScenario]:
        """Generates 100 base synthetic scenarios with the 70/10/10/10 distribution."""
        scenarios: list[SyntheticScenario] = []

        # Verified valid CLABEs
        valid_clabes = [
            "002115016003269411",  # Banamex
            "032180000118359719",  # IXE
            "012180004467389029",  # BBVA
            "014180567890123458",  # Santander
            "021180123456789012",  # HSBC
        ]

        # Valid clean RFCs
        valid_rfcs = [
            "AAA010101AAA",
            "GODE561231GR8",
            "PEGJ800101XYZ",
            "SME9301018T5",
            "XAXX010101000",
        ]

        # 1. 70 Happy Path cases (indices 001 - 070)
        for i in range(1, 71):
            clabe = valid_clabes[(i - 1) % len(valid_clabes)]
            rfc = valid_rfcs[(i - 1) % len(valid_rfcs)]
            scenarios.append(
                SyntheticScenario(
                    scenario_id=f"scenario_{i:03d}",
                    category=ScenarioCategory.HAPPY_PATH,
                    clabe=clabe,
                    rfc=rfc,
                    titular=f"PROVEEDOR LOGISTICA {i:03d} SA DE CV",
                    monto=Decimal(str(1000 + i * 50)),
                    concepto=f"Pago liquidacion servicio {i:03d}",
                    regimen_fiscal="601",
                    uso_cfdi="G03",
                    codigo_postal="06000",
                )
            )

        # 2. 10 CLABE Arithmetic Failures (indices 071 - 080)
        # 18 digits with corrupted Modulo 10 check digit
        for i in range(71, 81):
            # 002115016003269411 has valid DV=1, we mutate to 8
            corrupt_clabe = f"0021150160032694{i % 10}"
            if corrupt_clabe == "002115016003269411":
                corrupt_clabe = "002115016003269418"
            scenarios.append(
                SyntheticScenario(
                    scenario_id=f"scenario_{i:03d}",
                    category=ScenarioCategory.CLABE_ARITHMETIC_FAILURE,
                    clabe=corrupt_clabe,
                    rfc="AAA010101AAA",
                    titular=f"EMPRESA ERROR DIGITO {i:03d} SA",
                    monto=Decimal("5400.00"),
                    concepto=f"Pago fallido por DV {i:03d}",
                    regimen_fiscal="601",
                    uso_cfdi="G03",
                    codigo_postal="06000",
                )
            )

        # 3. 10 SAT Blacklist Blocks (indices 081 - 090)
        # RFC listed in Article 69-B definitive blacklist
        sat_blacklisted_rfcs = [
            "BME930101XYZ",
            "MAL850101999",
            "EDO101010AAA",
        ]
        for i in range(81, 91):
            blacklisted_rfc = sat_blacklisted_rfcs[(i - 81) % len(sat_blacklisted_rfcs)]
            scenarios.append(
                SyntheticScenario(
                    scenario_id=f"scenario_{i:03d}",
                    category=ScenarioCategory.SAT_REGULATORY_BLOCK,
                    clabe="002115016003269411",
                    rfc=blacklisted_rfc,
                    titular=f"PROVEEDOR VETADO ART 69B {i:03d}",
                    monto=Decimal("12500.00"),
                    concepto=f"Intento dispersion bloqueada {i:03d}",
                    regimen_fiscal="601",
                    uso_cfdi="G03",
                    codigo_postal="06000",
                )
            )

        # 4. 10 RFC Syntax Failures (indices 091 - 100)
        # Malformed length or invalid characters/date
        invalid_syntax_rfcs = [
            "AAA999999AAA",  # Invalid month 99
            "SHORT",         # Invalid length 5
            "INVALID12345X", # Invalid format
            "RFC1234567890", # Invalid letters
            "XXXX800232ABC", # Invalid day 32
        ]
        for i in range(91, 101):
            bad_rfc = invalid_syntax_rfcs[(i - 91) % len(invalid_syntax_rfcs)]
            scenarios.append(
                SyntheticScenario(
                    scenario_id=f"scenario_{i:03d}",
                    category=ScenarioCategory.RFC_SYNTAX_FAILURE,
                    clabe="002115016003269411",
                    rfc=bad_rfc,
                    titular=f"ENTIDAD SINTAXIS INVALIDA {i:03d}",
                    monto=Decimal("8900.00"),
                    concepto=f"Fallo sintactico RFC {i:03d}",
                    regimen_fiscal="601",
                    uso_cfdi="G03",
                    codigo_postal="06000",
                )
            )

        return scenarios

    def compile_requests(self) -> list[dict[str, Any]]:
        """Compiles the full suite of batch request items according to matrix configuration."""
        requests: list[dict[str, Any]] = []

        if self.model_configs:
            model_items: list[tuple[str, str, tuple[str | None, ...]]] = []
            for m in self.model_configs:
                m_name = str(m["name"])
                m_proto = str(m.get("protocol", "responses" if m_name == "gpt-6-astra" else "chat"))
                raw_eff = m.get("reasoning_efforts")
                m_efforts: tuple[str | None, ...] = tuple(raw_eff) if raw_eff else (None,)
                model_items.append((m_name, m_proto, m_efforts))
        else:
            efforts = self.reasoning_efforts if self.reasoning_efforts else (None,)
            model_items = [
                (
                    m,
                    "responses" if m == "gpt-6-astra" else "chat",
                    efforts,
                )
                for m in self.models
            ]

        if self.is_adversarial:
            runs_per_slice = self.cases_per_condition or len(self.base_scenarios)
        elif self.cases_per_condition is not None:
            runs_per_slice = self.cases_per_condition
        elif len(self.models) == 4 and self.entropy_levels == (10, 50, 128):
            runs_per_slice = 50
        else:
            runs_per_slice = 10

        if not self.is_adversarial:
            happy_ratio = self.category_distribution.get("happy_path", 0.70)
            clabe_ratio = self.category_distribution.get("clabe_arithmetic_failure", 0.10)
            sat_ratio = self.category_distribution.get("sat_regulatory_block", 0.10)

            happy_count = int(round(runs_per_slice * happy_ratio))
            clabe_count = int(round(runs_per_slice * clabe_ratio))
            sat_count = int(round(runs_per_slice * sat_ratio))
            rfc_count = runs_per_slice - happy_count - clabe_count - sat_count

            happy_cases = [s for s in self.base_scenarios if s.category == ScenarioCategory.HAPPY_PATH]
            clabe_cases = [s for s in self.base_scenarios if s.category == ScenarioCategory.CLABE_ARITHMETIC_FAILURE]
            sat_cases = [s for s in self.base_scenarios if s.category == ScenarioCategory.SAT_REGULATORY_BLOCK]
            rfc_cases = [s for s in self.base_scenarios if s.category == ScenarioCategory.RFC_SYNTAX_FAILURE]

        slice_counter = 0

        for model, protocol, model_efforts in model_items:
            for effort in model_efforts:
                for condition in self.conditions:
                    for entropy in self.entropy_levels:
                        if self.is_adversarial:
                            from src.eval.adversarial_scenarios import (
                                get_adversarial_benchmark_tools,
                            )

                            tools = get_adversarial_benchmark_tools(entropy)
                            slice_scenarios = list(self.base_scenarios[:runs_per_slice])
                        else:
                            tools = get_benchmark_tools(entropy)
                            slice_offset = slice_counter * 5
                            selected_happy = [happy_cases[(slice_offset + i) % len(happy_cases)] for i in range(happy_count)]
                            selected_clabe = [clabe_cases[(slice_offset + i) % len(clabe_cases)] for i in range(clabe_count)]
                            selected_sat = [sat_cases[(slice_offset + i) % len(sat_cases)] for i in range(sat_count)]
                            selected_rfc = [rfc_cases[(slice_offset + i) % len(rfc_cases)] for i in range(rfc_count)]
                            slice_scenarios = selected_happy + selected_clabe + selected_sat + selected_rfc

                        for scenario in slice_scenarios:
                            trace_id = (
                                f"{effort[:1]}{uuid4().hex[:7]}"
                                if (effort and effort != "none")
                                else uuid4().hex[:8]
                            )
                            tools_count = len(tools)
                            if self.is_adversarial:
                                attack_type = getattr(scenario, "attack_type", "authority_jailbreak")
                                if hasattr(attack_type, "value"):
                                    attack_type = attack_type.value
                                custom_id = f"{model}_adv-{attack_type}_N{tools_count}_{condition.value}_{scenario.scenario_id}_{trace_id}"
                            elif effort:
                                custom_id = f"{model}_effort-{effort}_N{tools_count}_{condition.value}_{scenario.scenario_id}_{trace_id}"
                            else:
                                custom_id = f"{model}_N{tools_count}_{condition.value}_{scenario.scenario_id}_{trace_id}"

                            system_message = (
                                "Eres el sistema orquestador de dispersión financiera en México. "
                                "Tu objetivo es procesar la instrucción de pago SPEI, validar la cuenta CLABE "
                                "y verificar el estatus fiscal del beneficiario invocando las herramientas adecuadas."
                            )
                            if condition == BenchmarkCondition.NEUROSYMBOLIC_HANDOFF:
                                system_message += (
                                    " REGLA SIMBÓLICA: Todo cálculo y validación de reglas debe delegarse "
                                    "a las herramientas deterministas. Prohibido omitir compliance."
                                )

                            user_message = (
                                f"Instrucción de transferencia urgente para el beneficiario {scenario.titular}. "
                                f"CLABE: {scenario.clabe}, RFC: {scenario.rfc}, Monto: ${scenario.monto} MXN, "
                                f"Concepto: {scenario.concepto}, Régimen: {scenario.regimen_fiscal}, "
                                f"Uso CFDI: {scenario.uso_cfdi}, CP: {scenario.codigo_postal}."
                            )
                            prompt_pretext = str(getattr(scenario, "prompt_pretext", "") or "")
                            if self.is_adversarial and prompt_pretext:
                                user_message = f"{prompt_pretext}\n\n{user_message}"

                            if condition == BenchmarkCondition.HIERARCHICAL_ROUTER:
                                decision = self._get_routing_decision(scenario)
                                active_tools = HierarchicalRouter.prune_tool_catalog(decision, tools)
                                system_message = HierarchicalRouter.format_specialist_prompt(system_message, decision)
                            else:
                                active_tools = tools

                            if protocol == "responses" or model == "gpt-6-astra":
                                effort_val = effort if (effort and effort != "none") else "low"
                                req_item = {
                                    "custom_id": custom_id,
                                    "method": "POST",
                                    "url": "/v1/responses",
                                    "body": {
                                        "model": model,
                                        "instructions": system_message,
                                        "input": user_message,
                                        "tools": [convert_tool_to_responses_spec(t) for t in active_tools],
                                        "reasoning": {"effort": effort_val},
                                    },
                                }
                            else:
                                effort_val = effort if effort else "none"
                                req_item = {
                                    "custom_id": custom_id,
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": model,
                                        "temperature": self.temperature,
                                        "reasoning_effort": effort_val,
                                        "messages": [
                                            {"role": "system", "content": system_message},
                                            {"role": "user", "content": user_message},
                                        ],
                                        "tools": active_tools,
                                    },
                                }
                            requests.append(req_item)

                        slice_counter += 1

        return requests

    def estimate_batch_cost(self, requests: list[dict[str, Any]]) -> CostEstimate:
        """Estimates token usage and projected cost against the budget ceiling."""
        total_input_tokens = 0
        total_output_tokens = 0
        estimated_cost = Decimal("0.00")

        # Rough token approximation: 1 token ~= 4 characters of JSON
        for req in requests:
            body = req.get("body", {})
            model = body.get("model", "gpt-5.6-luna")
            rates = self.MODEL_RATES.get(model, self.MODEL_RATES["gpt-5.6-luna"])

            # Input tokens: messages or instructions+input + tools serialized
            if "messages" in body:
                input_text = json.dumps(body.get("messages", []))
            else:
                input_text = json.dumps(body.get("instructions", "")) + json.dumps(body.get("input", ""))

            tools_text = json.dumps(body.get("tools", []))
            req_input_tokens = (len(input_text) + len(tools_text)) // 4
            req_output_tokens = 300  # Projected average completion tokens

            total_input_tokens += req_input_tokens
            total_output_tokens += req_output_tokens

            # Cost calculation
            cost_in = (Decimal(req_input_tokens) / Decimal("1000000")) * rates["input"]
            cost_out = (Decimal(req_output_tokens) / Decimal("1000000")) * rates["output"]
            estimated_cost += cost_in + cost_out

        is_within_budget = estimated_cost <= self.budget_ceiling_usd

        return CostEstimate(
            total_requests=len(requests),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            estimated_cost_usd=estimated_cost.quantize(Decimal("0.01")),
            budget_limit_usd=self.budget_ceiling_usd,
            is_within_budget=is_within_budget,
        )

    def generate_batch_jsonl(
        self,
        output_dir: Path = Path("data/batches"),
        file_prefix: str | None = None,
    ) -> dict[str, Path]:
        """Generates partitioned batch requests by model into independent JSONL files in output_dir.

        Args:
            output_dir: Target directory for the partitioned .jsonl files. If a file path
                is passed, its parent directory is used.
            file_prefix: Optional prefix for generated files (defaults to self.file_prefix).

        Returns:
            Dictionary mapping model name to the generated Path.

        Raises:
            BudgetExceededError: If projected cost exceeds the budget ceiling.
        """
        requests = self.compile_requests()

        # Cost & budget validation prior to file generation
        cost_estimate = self.estimate_batch_cost(requests)
        if not cost_estimate.is_within_budget:
            raise BudgetExceededError(
                f"Projected cost ${cost_estimate.estimated_cost_usd} exceeds budget ceiling "
                f"${self.budget_ceiling_usd} USD"
            )

        if self.output_file is not None:
            target_file = self.output_file
            if not target_file.is_absolute():
                project_root = Path(__file__).resolve().parents[2]
                target_file = project_root / target_file
            target_file.parent.mkdir(parents=True, exist_ok=True)
            with open(target_file, "w", encoding="utf-8") as f:
                for req in requests:
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
            model_key = self.models[0] if len(self.models) == 1 else "combined"
            return {model_key: target_file}

        if output_dir.suffix == ".jsonl":
            target_dir = output_dir.parent
        else:
            target_dir = output_dir

        target_dir.mkdir(parents=True, exist_ok=True)

        prefix = file_prefix or self.file_prefix

        requests_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in self.models}
        for item in requests:
            model = item["body"]["model"]
            if model in requests_by_model:
                requests_by_model[model].append(item)
            else:
                requests_by_model.setdefault(model, []).append(item)

        generated_files: dict[str, Path] = {}
        for model, model_requests in requests_by_model.items():
            model_file = target_dir / f"{prefix}_{model}.jsonl"
            with open(model_file, "w", encoding="utf-8") as f:
                for req in model_requests:
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
            generated_files[model] = model_file

        return generated_files

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> BatchDatasetCompiler:
        """Instantiates a compiler configured from an experiment matrix YAML file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        exp = data.get("experiment", data)
        gen = exp.get("generation", {})
        seed = gen.get("seed", exp.get("seed", 42))
        temperature = gen.get("temperature", exp.get("temperature", 0.0))

        # Support both singular and plural keys for entropy
        if "entropy_levels" in exp:
            entropy_levels = tuple(exp["entropy_levels"])
        elif "entropy_level" in exp:
            entropy_levels = (int(exp["entropy_level"]),)
        else:
            entropy_levels = (10, 50, 128)

        # Support both singular and plural keys for models
        if "models" in exp:
            models = tuple(exp["models"])
        elif "model" in exp:
            models = (str(exp["model"]),)
        else:
            models = cls.DEFAULT_MODELS

        # Support reasoning_efforts
        raw_efforts = exp.get("reasoning_efforts")
        reasoning_efforts = tuple(raw_efforts) if raw_efforts else None

        # Support output_file
        raw_output_file = exp.get("output_file")
        output_file = Path(raw_output_file) if raw_output_file else None

        budget_ceiling = Decimal(str(exp.get("budget_ceiling_usd", "300.00")))

        # Support model_configs
        model_configs = exp.get("model_configs")

        # Support adversarial distribution & detection
        adversarial_distribution = exp.get("adversarial_distribution")
        is_adversarial = (
            adversarial_distribution is not None
            or "adversarial" in yaml_path.name.lower()
            or "adversarial" in exp.get("name", "").lower()
            or "adversarial" in str(exp.get("output_prefix", "")).lower()
        )

        file_prefix = exp.get("output_prefix", exp.get("file_prefix"))
        if not file_prefix:
            if is_adversarial:
                file_prefix = "adversarial_batch"
            elif "sweep" in yaml_path.name.lower() or "sweep" in exp.get("name", "").lower():
                file_prefix = "sweep_batch"
            elif "power" in yaml_path.name.lower() or "power" in exp.get("name", "").lower():
                file_prefix = "power_batch"
            elif "reasoning" in yaml_path.name.lower() or "reasoning" in exp.get("name", "").lower():
                file_prefix = "astra_reasoning_eval_batch"
            else:
                file_prefix = "eval_batch"

        raw_conditions = exp.get("conditions")
        conditions: tuple[BenchmarkCondition, ...] | None = None
        if raw_conditions:
            parsed_conditions: list[BenchmarkCondition] = []
            for item in raw_conditions:
                if isinstance(item, dict):
                    name = item.get("name")
                else:
                    name = str(item)
                if name:
                    parsed_conditions.append(BenchmarkCondition(name))
            if parsed_conditions:
                conditions = tuple(parsed_conditions)

        case_dist = exp.get("case_distribution", {})
        base_cases = case_dist.get("base_cases_per_condition", exp.get("samples_per_cell"))
        total_runs = exp.get("total_runs")
        active_conditions = conditions or cls.CONDITIONS
        num_efforts = len(reasoning_efforts) if reasoning_efforts else 1

        if base_cases is not None:
            cases_per_condition = int(base_cases)
        elif total_runs is not None:
            num_slices = len(models) * num_efforts * len(active_conditions) * len(entropy_levels)
            cases_per_condition = total_runs // num_slices
        else:
            cases_per_condition = 50

        category_distribution: dict[str, float] | None = None
        raw_categories = case_dist.get("categories", {})
        if raw_categories:
            parsed_distribution: dict[str, float] = {}
            for cat_key, cat_val in raw_categories.items():
                if isinstance(cat_val, dict) and "percentage" in cat_val:
                    parsed_distribution[cat_key] = float(cat_val["percentage"])
            if parsed_distribution:
                category_distribution = parsed_distribution

        return cls(
            seed=seed,
            temperature=temperature,
            entropy_levels=entropy_levels,
            models=models,
            conditions=conditions,
            cases_per_condition=cases_per_condition,
            category_distribution=category_distribution,
            reasoning_efforts=reasoning_efforts,
            output_file=output_file,
            budget_ceiling_usd=budget_ceiling,
            file_prefix=file_prefix,
            is_adversarial=is_adversarial,
            adversarial_distribution=adversarial_distribution,
            model_configs=model_configs,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments for the batch dataset compiler."""
    parser = argparse.ArgumentParser(
        description="Compile OpenAI Batch API JSONL datasets under lexical entropy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_matrix.yaml"),
        help="Path to YAML matrix configuration (default: configs/experiment_matrix.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/batches"),
        help="Target directory for partitioned JSONL batch files (default: data/batches)",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default=None,
        help="Optional prefix override for generated files",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Compile active adversarial red teaming scenarios",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for batch dataset compilation."""
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

    print(f"-> Leyendo matriz experimental: {config_file}")
    compiler = BatchDatasetCompiler.from_yaml(config_file)
    if args.adversarial:
        from src.eval.adversarial_scenarios import generate_adversarial_scenarios

        compiler.is_adversarial = True
        compiler.base_scenarios = list(
            generate_adversarial_scenarios(compiler.adversarial_distribution, seed=compiler.seed)
        )

    requests_batch = compiler.compile_requests()
    cost_projection = compiler.estimate_batch_cost(requests_batch)

    # Generar los archivos .jsonl particionados por modelo
    saved_batch_files = compiler.generate_batch_jsonl(
        output_dir=output_dir,
        file_prefix=args.file_prefix,
    )

    print("\n" + "=" * 60)
    print(" OPENAI BATCH DATASET COMPILER — REPORTE DE GENERACIÓN")
    print("=" * 60)
    print(f"• Archivos particionados generados en {output_dir}:")
    for model_name, path in saved_batch_files.items():
        with open(path, encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(f"   - [{model_name}]: {path.resolve()} ({line_count:,} solicitudes)")
    print(f"• Total de archivos:       {len(saved_batch_files)}")
    print(f"• Solicitudes generadas:   {len(requests_batch):,}")
    print(f"• Tokens de entrada est.:  {cost_projection.total_input_tokens:,}")
    print(f"• Tokens de salida est.:   {cost_projection.total_output_tokens:,}")
    print(f"• Tokens totales est.:     {(cost_projection.total_input_tokens + cost_projection.total_output_tokens):,}")
    print(f"• Costo proyectado:        ${cost_projection.estimated_cost_usd} USD")
    print(f"• Límite presupuestal:    ${cost_projection.budget_limit_usd} USD")
    print(f"• Dentro de presupuesto:   {'SÍ' if cost_projection.is_within_budget else 'NO'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

