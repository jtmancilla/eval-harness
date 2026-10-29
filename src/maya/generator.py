"""Deterministic compiler for the MAYA phase-one evaluation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

from src.maya.contracts import (
    MayaCase,
    MayaExpectedOutcome,
    MayaModel,
    MayaSyntheticContext,
)
from src.maya.narratives import NARRATIVE_STYLES, render_user_text
from src.maya.personas import PERSONAS
from src.maya.scenarios import SCENARIOS
from src.maya.tools import CANONICAL_TOOLS_BY_NAME, build_tool_catalog

SCHEMA_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVED_COLUMNS = (
    "observed_response",
    "observed_route",
    "observed_actions",
    "observed_validations",
    "observed_writes",
    "route_correct",
    "action_selection_correct",
    "arguments_correct",
    "execution_order_correct",
    "validations_correct",
    "persistence_and_logs_correct",
    "response_correct",
    "final_verdict",
    "reviewer_notes",
)

MATRIX_COLUMNS = (
    "case_id",
    "schema_version",
    "scenario_id",
    "kind",
    "complexity",
    "persona_id",
    "region",
    "state",
    "city",
    "age",
    "segment",
    "channel",
    "digital_literacy",
    "urgency",
    "style_id",
    "saturation_level",
    "user_text",
    "expected_route",
    "expected_disposition",
    "expected_actions",
    "expected_order",
    "expected_validations",
    "expected_writes",
    "forbidden_writes",
    "response_required_facts",
    "response_forbidden_content",
    *OBSERVED_COLUMNS,
)


@dataclass(frozen=True)
class MayaGeneratorConfig:
    name: str = "maya_agentic_evaluation_phase1"
    version: str = SCHEMA_VERSION
    seed: int = 42
    total_cases: int = 480
    saturation_levels: tuple[int, ...] = (18, 36, 72)
    output_dir: Path = Path("data/maya/phase1")
    sample_output: Path = Path("data/fixtures/maya_phase1_sample.json")

    def __post_init__(self) -> None:
        if self.total_cases <= 0:
            raise ValueError("total_cases must be positive")
        if not self.saturation_levels:
            raise ValueError("At least one saturation level is required")
        if len(set(self.saturation_levels)) != len(self.saturation_levels):
            raise ValueError("Saturation levels must be unique")

    @classmethod
    def from_yaml(cls, path: Path) -> MayaGeneratorConfig:
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open(encoding="utf-8") as stream:
            loaded: object = yaml.safe_load(stream)
        if not isinstance(loaded, dict):
            raise ValueError("MAYA configuration must be a YAML object")
        root = cast(dict[str, object], loaded)
        section_value = root.get("maya_matrix", root)
        if not isinstance(section_value, dict):
            raise ValueError("maya_matrix must be a YAML object")
        section = cast(dict[str, object], section_value)
        levels_value = section.get("saturation_levels", [18, 36, 72])
        if not isinstance(levels_value, list):
            raise ValueError("saturation_levels must be a list")
        levels = tuple(int(str(value)) for value in cast(list[object], levels_value))
        return cls(
            name=str(section.get("name", cls.name)),
            version=str(section.get("version", cls.version)),
            seed=int(str(section.get("seed", cls.seed))),
            total_cases=int(str(section.get("total_cases", cls.total_cases))),
            saturation_levels=levels,
            output_dir=Path(str(section.get("output_dir", cls.output_dir))),
            sample_output=Path(str(section.get("sample_output", cls.sample_output))),
        )


@dataclass(frozen=True)
class MayaMatrixBundle:
    cases: tuple[MayaCase, ...]
    expectations: tuple[MayaExpectedOutcome, ...]


class MayaMatrixGenerator:
    def __init__(self, config: MayaGeneratorConfig) -> None:
        self.config = config

    def generate(self) -> MayaMatrixBundle:
        rng = random.Random(self.config.seed)
        cells = [
            (scenario_index, (scenario_index + persona_offset) % len(PERSONAS))
            for persona_offset in range(len(PERSONAS))
            for scenario_index in range(len(SCENARIOS))
        ]
        selected_cells = [cells[index % len(cells)] for index in range(self.config.total_cases)]
        rng.shuffle(selected_cells)
        occurrences: Counter[tuple[int, int]] = Counter()
        cases: list[MayaCase] = []
        expectations: list[MayaExpectedOutcome] = []

        for scenario_index, persona_index in selected_cells:
            scenario = SCENARIOS[scenario_index]
            persona = PERSONAS[persona_index]
            cell = (scenario_index, persona_index)
            occurrences[cell] += 1
            case_id = (
                f"maya-{scenario.scenario_id}-{persona.persona_id}-"
                f"{occurrences[cell]:02d}"
            )
            style = NARRATIVE_STYLES[
                (len(cases) + scenario_index * 3 + persona_index) % len(NARRATIVE_STYLES)
            ]
            saturation = self.config.saturation_levels[
                (len(cases) + scenario_index + persona_index) % len(self.config.saturation_levels)
            ]
            utterance = rng.choice(scenario.utterances)
            context = MayaSyntheticContext(
                customer_id=f"SYN-CUSTOMER-{persona.persona_id.upper()}",
                product_ids=(
                    "SYN-NOMINA-001",
                    "SYN-AHORRO-002",
                    "SYN-CREDIT-003",
                    "SYN-DEVICE-001",
                ),
                authenticated=scenario.authenticated,
                channel_profile_complete=scenario.channel_profile_complete,
                token_active=scenario.token_active,
                registered_device=scenario.registered_device,
                sufficient_funds=scenario.sufficient_funds,
                eligible_purchase=scenario.eligible_purchase,
            )
            case = MayaCase(
                case_id=case_id,
                schema_version=self.config.version,
                scenario_id=scenario.scenario_id,
                kind=scenario.kind,
                complexity=scenario.complexity,
                persona=persona,
                style=style,
                user_text=render_user_text(utterance, style, persona, rng),
                context=context,
                available_tools=build_tool_catalog(saturation, rng),
                knowledge_snippets=scenario.knowledge_snippets,
                saturation_level=saturation,
            )
            expected = MayaExpectedOutcome(
                case_id=case_id,
                expected_route=scenario.expected_route,
                disposition=scenario.disposition,
                actions=scenario.actions,
                validations=scenario.validations,
                writes=scenario.writes,
                forbidden_writes=scenario.forbidden_writes,
                response=scenario.response,
            )
            self._validate_case_pair(case, expected)
            cases.append(case)
            expectations.append(expected)

        bundle = MayaMatrixBundle(cases=tuple(cases), expectations=tuple(expectations))
        self.validate_bundle(bundle)
        return bundle

    @staticmethod
    def _validate_case_pair(case: MayaCase, expected: MayaExpectedOutcome) -> None:
        if case.case_id != expected.case_id:
            raise ValueError("Case and expectation IDs must match")
        available_canonical = {
            tool.name for tool in case.available_tools if tool.canonical
        }
        for action in expected.actions:
            if action.tool_name not in available_canonical:
                raise ValueError(f"Action tool is not available: {action.tool_name}")
            tool = CANONICAL_TOOLS_BY_NAME[action.tool_name]
            if action.operation_id != tool.operation_id:
                raise ValueError(f"Operation ID mismatch for {action.tool_name}")
            declared_parameters = {parameter.name for parameter in tool.parameters}
            arguments = {argument.name for argument in action.arguments}
            if not arguments.issubset(declared_parameters):
                raise ValueError(f"Undeclared arguments for {action.tool_name}: {arguments}")
            if action.should_execute and not set(tool.required).issubset(arguments):
                raise ValueError(f"Missing required arguments for {action.tool_name}")

    def validate_bundle(self, bundle: MayaMatrixBundle) -> None:
        if len(bundle.cases) != len(bundle.expectations):
            raise ValueError("Cases and expectations must have identical lengths")
        case_ids = [case.case_id for case in bundle.cases]
        expected_ids = [expected.case_id for expected in bundle.expectations]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Case IDs must be unique")
        if set(case_ids) != set(expected_ids):
            raise ValueError("Every case must have exactly one expectation")

    def coverage(self, bundle: MayaMatrixBundle) -> dict[str, object]:
        self.validate_bundle(bundle)
        expected_by_id = {expected.case_id: expected for expected in bundle.expectations}
        operation_counts: Counter[str] = Counter()
        for expected in bundle.expectations:
            for action in expected.actions:
                if action.operation_id is not None:
                    operation_counts[f"{action.operation_id:02d}"] += 1

        def age_band(age: int) -> str:
            if age <= 24:
                return "18-24"
            if age <= 39:
                return "25-39"
            if age <= 59:
                return "40-59"
            return "60+"

        return {
            "name": self.config.name,
            "schema_version": self.config.version,
            "seed": self.config.seed,
            "total_cases": len(bundle.cases),
            "scenario_count": len({case.scenario_id for case in bundle.cases}),
            "persona_count": len({case.persona.persona_id for case in bundle.cases}),
            "by_scenario": _string_counts(case.scenario_id for case in bundle.cases),
            "by_persona": _string_counts(case.persona.persona_id for case in bundle.cases),
            "by_region": _string_counts(case.persona.region for case in bundle.cases),
            "by_age_band": _string_counts(age_band(case.persona.age) for case in bundle.cases),
            "by_style": _string_counts(case.style.style_id for case in bundle.cases),
            "by_urgency": _string_counts(case.persona.urgency.value for case in bundle.cases),
            "by_complexity": _string_counts(case.complexity.value for case in bundle.cases),
            "by_route": _string_counts(
                expected_by_id[case.case_id].expected_route.value for case in bundle.cases
            ),
            "by_saturation": _string_counts(str(case.saturation_level) for case in bundle.cases),
            "operation_coverage": dict(sorted(operation_counts.items())),
            "uncovered_operation_ids": [
                f"{operation_id:02d}"
                for operation_id in range(1, 18)
                if operation_counts[f"{operation_id:02d}"] == 0
            ],
        }

    def export(
        self,
        bundle: MayaMatrixBundle,
        output_dir: Path | None = None,
        sample_output: Path | None = None,
    ) -> dict[str, Path]:
        self.validate_bundle(bundle)
        target_dir = _resolve_path(output_dir or self.config.output_dir)
        sample_path = _resolve_path(sample_output or self.config.sample_output)
        target_dir.mkdir(parents=True, exist_ok=True)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        paths = {
            "cases": target_dir / "cases.jsonl",
            "expectations": target_dir / "expectations.jsonl",
            "matrix": target_dir / "maya_matrix.csv",
            "coverage": target_dir / "coverage.json",
            "sample": sample_path,
        }
        _write_jsonl(paths["cases"], bundle.cases)
        _write_jsonl(paths["expectations"], bundle.expectations)
        self._write_matrix(paths["matrix"], bundle)
        coverage = self.coverage(bundle)
        paths["coverage"].write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_sample(paths["sample"], bundle)
        return paths

    @staticmethod
    def _write_matrix(path: Path, bundle: MayaMatrixBundle) -> None:
        expected_by_id = {expected.case_id: expected for expected in bundle.expectations}
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(MATRIX_COLUMNS))
            writer.writeheader()
            for case in bundle.cases:
                expected = expected_by_id[case.case_id]
                actions = [
                    {
                        "action_id": action.action_id,
                        "operation_id": action.operation_id,
                        "tool_name": action.tool_name,
                        "arguments": {
                            argument.name: argument.value for argument in action.arguments
                        },
                        "depends_on": list(action.depends_on),
                        "should_execute": action.should_execute,
                    }
                    for action in expected.actions
                ]
                row: dict[str, object] = {
                    "case_id": case.case_id,
                    "schema_version": case.schema_version,
                    "scenario_id": case.scenario_id,
                    "kind": case.kind.value,
                    "complexity": case.complexity.value,
                    "persona_id": case.persona.persona_id,
                    "region": case.persona.region,
                    "state": case.persona.state,
                    "city": case.persona.city,
                    "age": case.persona.age,
                    "segment": case.persona.segment.value,
                    "channel": case.persona.channel.value,
                    "digital_literacy": case.persona.digital_literacy.value,
                    "urgency": case.persona.urgency.value,
                    "style_id": case.style.style_id,
                    "saturation_level": case.saturation_level,
                    "user_text": case.user_text,
                    "expected_route": expected.expected_route.value,
                    "expected_disposition": expected.disposition.value,
                    "expected_actions": _json(actions),
                    "expected_order": _json([action.action_id for action in expected.actions]),
                    "expected_validations": _json(
                        [validation.model_dump(mode="json") for validation in expected.validations]
                    ),
                    "expected_writes": _json(
                        [write.model_dump(mode="json") for write in expected.writes]
                    ),
                    "forbidden_writes": _json(
                        [write.model_dump(mode="json") for write in expected.forbidden_writes]
                    ),
                    "response_required_facts": _json(expected.response.required_facts),
                    "response_forbidden_content": _json(expected.response.forbidden_content),
                }
                row.update({column: "" for column in OBSERVED_COLUMNS})
                writer.writerow(row)

    @staticmethod
    def _write_sample(path: Path, bundle: MayaMatrixBundle) -> None:
        expected_by_id = {expected.case_id: expected for expected in bundle.expectations}
        preferred_scenarios = (
            "op01_transfer",
            "rag_cardless_withdrawal",
            "out_of_scope_python",
            "clarify_transfer",
            "adv_unauthenticated_transfer",
            "multi_token_then_transfer",
            "multi_transactions_then_dispute",
            "op16_token_activation",
        )
        selected: list[MayaCase] = []
        used: set[str] = set()
        for scenario_id in preferred_scenarios:
            case = next((case for case in bundle.cases if case.scenario_id == scenario_id), None)
            if case is not None:
                selected.append(case)
                used.add(case.case_id)
        for case in bundle.cases:
            if len(selected) >= len(preferred_scenarios):
                break
            if case.case_id not in used:
                selected.append(case)
                used.add(case.case_id)
        examples = [
            {
                "case": case.model_dump(mode="json"),
                "expectation": expected_by_id[case.case_id].model_dump(mode="json"),
            }
            for case in selected
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "description": "Ejemplos sintéticos representativos; no contienen datos reales.",
            "examples": examples,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _string_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_jsonl(path: Path, records: tuple[MayaModel, ...]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(_json(record.model_dump(mode="json")) + "\n")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the MAYA phase-one evaluation matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/maya_matrix.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-output", type=Path, default=None)
    parser.add_argument("--total-cases", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = MayaGeneratorConfig.from_yaml(_resolve_path(args.config))
    config = replace(
        config,
        total_cases=args.total_cases if args.total_cases is not None else config.total_cases,
        output_dir=args.output_dir if args.output_dir is not None else config.output_dir,
        sample_output=(
            args.sample_output if args.sample_output is not None else config.sample_output
        ),
    )
    generator = MayaMatrixGenerator(config)
    bundle = generator.generate()
    paths = generator.export(bundle)
    coverage = generator.coverage(bundle)
    print(f"Matriz: {config.name}")
    print(f"Casos generados: {len(bundle.cases)}")
    print(f"Escenarios cubiertos: {coverage['scenario_count']}")
    print(f"Operaciones sin cobertura: {coverage['uncovered_operation_ids']}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
