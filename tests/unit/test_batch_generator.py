"""Unit tests for the OpenAI Batch API dataset compiler, JSONL generator, and cost estimator."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.eval.batch_generator import (
    BatchDatasetCompiler,
    ScenarioCategory,
    get_benchmark_tools,
)

CUSTOM_ID_REGEX = re.compile(
    r"^(gpt-[a-z0-9\.\-]+)_N(10|50|150)_(baseline_autorregresivo|neurosymbolic_handoff)_(scenario_\d{3})_([a-f0-9]+)$"
)


def test_base_scenarios_distribution() -> None:
    """Verifies that base scenarios strictly satisfy the 70/10/10/10 distribution."""
    compiler = BatchDatasetCompiler(seed=42)
    scenarios = compiler.base_scenarios

    assert len(scenarios) == 100

    happy_count = sum(1 for s in scenarios if s.category == ScenarioCategory.HAPPY_PATH)
    clabe_fail_count = sum(1 for s in scenarios if s.category == ScenarioCategory.CLABE_ARITHMETIC_FAILURE)
    sat_block_count = sum(1 for s in scenarios if s.category == ScenarioCategory.SAT_REGULATORY_BLOCK)
    rfc_fail_count = sum(1 for s in scenarios if s.category == ScenarioCategory.RFC_SYNTAX_FAILURE)

    assert happy_count == 70
    assert clabe_fail_count == 10
    assert sat_block_count == 10
    assert rfc_fail_count == 10


def test_get_benchmark_tools_composition() -> None:
    """Verifies exact tool composition for N=10 (5+5), N=50 (5+45), N=150 (5+145)."""
    tools_10 = get_benchmark_tools(10)
    tools_50 = get_benchmark_tools(50)
    tools_150 = get_benchmark_tools(150)

    assert len(tools_10) == 10
    assert len(tools_50) == 50
    assert len(tools_150) == 150

    # Ensure all tool names within each set are unique
    for tool_set in (tools_10, tools_50, tools_150):
        names = [t["function"]["name"] for t in tool_set]
        assert len(names) == len(set(names))


def test_compiler_generates_exact_1200_requests() -> None:
    """Verifies compilation produces exactly 1,200 valid request objects."""
    compiler = BatchDatasetCompiler(seed=42)
    requests = compiler.compile_requests()

    assert len(requests) == 1200


def test_custom_ids_uniqueness_and_typed_structure() -> None:
    """Verifies all 1,200 custom_ids are strictly unique and match formal naming convention."""
    compiler = BatchDatasetCompiler(seed=42)
    requests = compiler.compile_requests()

    custom_ids = [req["custom_id"] for req in requests]
    assert len(custom_ids) == len(set(custom_ids)) == 1200

    for cid in custom_ids:
        match = CUSTOM_ID_REGEX.match(cid)
        assert match is not None, f"custom_id '{cid}' does not match expected nomenclature"


def test_tool_counts_match_entropy_level_in_each_request() -> None:
    """Verifies that the number of tools in body.tools exactly matches N (10, 50, 150)."""
    compiler = BatchDatasetCompiler(seed=42)
    requests = compiler.compile_requests()

    for req in requests:
        cid = req["custom_id"]
        tools = req["body"]["tools"]

        if "_N10_" in cid:
            assert len(tools) == 10
        elif "_N50_" in cid:
            assert len(tools) == 50
        elif "_N150_" in cid:
            assert len(tools) == 150
        else:
            raise ValueError(f"Unknown entropy level marker in {cid}")


def test_batch_jsonl_generation_and_parseable_lines(tmp_path: Path) -> None:
    """Verifies file generation of 1,200 JSON-parseable lines in OpenAI Batch API format."""
    output_file = tmp_path / "test_batch_1200.jsonl"
    compiler = BatchDatasetCompiler(seed=42)

    generated_path = compiler.generate_batch_jsonl(output_file)
    assert generated_path.exists()

    line_count = 0
    with open(generated_path, encoding="utf-8") as f:
        for line in f:
            line_count += 1
            record = json.loads(line)
            assert record["method"] == "POST"
            assert record["url"] == "/v1/chat/completions"
            assert "custom_id" in record
            body = record["body"]
            assert body["model"] in compiler.MODELS
            assert isinstance(body["messages"], list)
            assert len(body["messages"]) >= 2
            assert isinstance(body["tools"], list)

    assert line_count == 1200


def test_cost_estimator_under_budget_ceiling() -> None:
    """Verifies that token and cost projections remain safely under the $300 USD ceiling."""
    compiler = BatchDatasetCompiler(seed=42)
    requests = compiler.compile_requests()
    cost_estimate = compiler.estimate_batch_cost(requests)

    assert cost_estimate.total_requests == 1200
    assert cost_estimate.total_input_tokens > 0
    assert cost_estimate.total_output_tokens > 0
    assert cost_estimate.is_within_budget is True
    assert cost_estimate.estimated_cost_usd < cost_estimate.budget_limit_usd
