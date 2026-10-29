"""Unit and end-to-end tests for the MAYA phase-one evaluation matrix."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from src.maya.contracts import (
    MayaComplexity,
    MayaDisposition,
    MayaExpectedAction,
    MayaExpectedOutcome,
    MayaResponseExpectation,
    MayaRoute,
    MayaScenarioKind,
    MayaValidationExpectation,
)
from src.maya.generator import (
    OBSERVED_COLUMNS,
    MayaGeneratorConfig,
    MayaMatrixGenerator,
    main,
)
from src.maya.narratives import NARRATIVE_STYLES
from src.maya.personas import PERSONAS
from src.maya.scenarios import SCENARIOS, SCENARIOS_BY_ID
from src.maya.tools import (
    BANKING_OPERATION_TOOLS,
    CANONICAL_TOOLS,
    build_tool_catalog,
)


def _outcome(
    actions: tuple[MayaExpectedAction, ...],
    validations: tuple[MayaValidationExpectation, ...] = (),
) -> MayaExpectedOutcome:
    return MayaExpectedOutcome(
        case_id="maya-test",
        expected_route=MayaRoute.IBM_ASSISTANT,
        disposition=MayaDisposition.COMPLETED,
        actions=actions,
        validations=validations,
        response=MayaResponseExpectation(),
    )


def test_catalogs_cover_personas_scenarios_and_seventeen_operations() -> None:
    assert len(PERSONAS) == 16
    assert len(SCENARIOS) == 30
    assert len(NARRATIVE_STYLES) == 10
    assert len(BANKING_OPERATION_TOOLS) == 17
    assert len(CANONICAL_TOOLS) == 18
    assert {tool.operation_id for tool in BANKING_OPERATION_TOOLS} == set(range(1, 18))
    assert len({persona.persona_id for persona in PERSONAS}) == len(PERSONAS)
    assert len({scenario.scenario_id for scenario in SCENARIOS}) == len(SCENARIOS)


def test_default_generation_has_complete_stratified_coverage() -> None:
    generator = MayaMatrixGenerator(MayaGeneratorConfig())
    bundle = generator.generate()
    coverage = generator.coverage(bundle)

    assert len(bundle.cases) == len(bundle.expectations) == 480
    assert coverage["uncovered_operation_ids"] == []
    assert set(cast("dict[str, int]", coverage["operation_coverage"])) == {
        f"{operation_id:02d}" for operation_id in range(1, 18)
    }
    scenario_counts = Counter(case.scenario_id for case in bundle.cases)
    persona_counts = Counter(case.persona.persona_id for case in bundle.cases)
    assert set(scenario_counts.values()) == {16}
    assert set(persona_counts.values()) == {30}
    assert {case.style.style_id for case in bundle.cases} == {
        style.style_id for style in NARRATIVE_STYLES
    }
    assert {case.saturation_level for case in bundle.cases} == {18, 36, 72}


def test_generation_is_reproducible_and_seed_controls_variation() -> None:
    config = MayaGeneratorConfig(total_cases=60, saturation_levels=(18, 36), seed=7)
    first = MayaMatrixGenerator(config).generate()
    second = MayaMatrixGenerator(config).generate()
    different = MayaMatrixGenerator(
        MayaGeneratorConfig(total_cases=60, saturation_levels=(18, 36), seed=8)
    ).generate()

    assert [case.model_dump_json() for case in first.cases] == [
        case.model_dump_json() for case in second.cases
    ]
    assert [case.model_dump_json() for case in first.cases] != [
        case.model_dump_json() for case in different.cases
    ]
    assert len({case.case_id for case in first.cases}) == 60


def test_cases_do_not_leak_expected_outcomes() -> None:
    bundle = MayaMatrixGenerator(MayaGeneratorConfig(total_cases=30)).generate()
    expectations_by_id = {expected.case_id: expected for expected in bundle.expectations}

    for case in bundle.cases:
        payload = case.model_dump()
        assert "expected_route" not in payload
        assert "actions" not in payload
        assert "validations" not in payload
        assert "writes" not in payload
        assert "response" not in payload
        assert expectations_by_id[case.case_id].case_id == case.case_id


def test_tool_saturation_is_exact_unique_and_provider_neutral() -> None:
    canonical_names = {tool.name for tool in CANONICAL_TOOLS}
    for size in (18, 36, 72):
        catalog = build_tool_catalog(size, random.Random(42))
        assert len(catalog) == size
        assert len({tool.name for tool in catalog}) == size
        assert canonical_names.issubset({tool.name for tool in catalog})
        assert all(
            tool.target_canonical_tool in canonical_names
            for tool in catalog
            if not tool.canonical
        )

    with pytest.raises(ValueError, match="Catalog size"):
        build_tool_catalog(17, random.Random(42))


def test_action_graph_rejects_cycles() -> None:
    first = MayaExpectedAction(action_id="a", tool_name="tool_a", depends_on=("b",))
    second = MayaExpectedAction(action_id="b", tool_name="tool_b", depends_on=("a",))

    with pytest.raises(ValueError, match="acyclic"):
        _outcome((first, second))


def test_action_graph_requires_topological_order() -> None:
    dependent = MayaExpectedAction(action_id="a", tool_name="tool_a", depends_on=("b",))
    prerequisite = MayaExpectedAction(action_id="b", tool_name="tool_b")

    with pytest.raises(ValueError, match="topological"):
        _outcome((dependent, prerequisite))


def test_failed_validation_blocks_dependants() -> None:
    failed = MayaExpectedAction(
        action_id="a", tool_name="tool_a", should_execute=False
    )
    dependant = MayaExpectedAction(
        action_id="b", tool_name="tool_b", depends_on=("a",)
    )
    failed_validation = MayaValidationExpectation(
        validation_id="a_session",
        action_id="a",
        description="Authentication failed",
        passes=False,
    )

    with pytest.raises(ValueError, match="block the action and its dependants"):
        _outcome((failed, dependant), (failed_validation,))


def test_ambiguous_and_unauthenticated_transfers_have_no_debit() -> None:
    ambiguous = SCENARIOS_BY_ID["clarify_transfer"]
    assert ambiguous.disposition == MayaDisposition.NEEDS_CLARIFICATION
    assert ambiguous.response.must_request_clarification is True
    assert ambiguous.actions[0].should_execute is False
    assert ambiguous.validations[0].passes is False
    assert ambiguous.forbidden_writes[0].record_type == "unvalidated_funds_transfer"

    unauthenticated = SCENARIOS_BY_ID["adv_unauthenticated_transfer"]
    assert unauthenticated.authenticated is False
    assert unauthenticated.actions[0].should_execute is False
    assert unauthenticated.forbidden_writes[0].record_type == "unauthenticated_funds_transfer"
    assert "contraseña" in unauthenticated.response.required_facts[1]


def test_multi_action_scenarios_encode_safe_one_to_one_sequences() -> None:
    compound = [
        scenario for scenario in SCENARIOS if scenario.kind == MayaScenarioKind.MULTI_ACTION
    ]
    assert len(compound) == 4
    for scenario in compound:
        assert scenario.complexity == MayaComplexity.MULTI_ACTION
        assert len(scenario.actions) == 2
        first, second = scenario.actions
        assert first.depends_on == ()
        assert second.depends_on == (first.action_id,)

    token_transfer = SCENARIOS_BY_ID["multi_token_then_transfer"]
    assert [action.tool_name for action in token_transfer.actions] == [
        "activate_mobile_token",
        "transfer_funds",
    ]
    assert token_transfer.token_active is False


def test_response_contract_is_strict() -> None:
    with pytest.raises(ValidationError):
        MayaResponseExpectation.model_validate(
            {"language": "es-MX", "undeclared_field": "not allowed"}
        )


def test_cli_exports_jsonl_csv_coverage_and_sample(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"
    sample_path = tmp_path / "sample.json"
    main(
        [
            "--config",
            "configs/maya_matrix.yaml",
            "--total-cases",
            "30",
            "--output-dir",
            str(output_dir),
            "--sample-output",
            str(sample_path),
        ]
    )

    case_lines = (output_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    expected_lines = (
        (output_dir / "expectations.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(case_lines) == len(expected_lines) == 30
    first_case = json.loads(case_lines[0])
    first_expected = json.loads(expected_lines[0])
    assert first_case["case_id"] == first_expected["case_id"]
    assert "expected_route" not in first_case
    assert first_expected["expected_route"]

    with (output_dir / "maya_matrix.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert len(rows) == 30
    assert reader.fieldnames is not None
    assert set(OBSERVED_COLUMNS).issubset(reader.fieldnames)
    assert all(row["final_verdict"] == "" for row in rows)
    assert all(row["observed_response"] == "" for row in rows)
    assert any("¿" in row["user_text"] for row in rows)

    coverage = json.loads((output_dir / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["total_cases"] == 30
    assert coverage["scenario_count"] == 30
    assert coverage["uncovered_operation_ids"] == []
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert len(sample["examples"]) == 8
    assert all("case" in example and "expectation" in example for example in sample["examples"])
