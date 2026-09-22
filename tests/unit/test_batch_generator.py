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
    r"^(gpt-[a-z0-9\.\-]+)_N(10|50|128)_(baseline_autorregresivo|neurosymbolic_handoff)_(scenario_\d{3})_([a-f0-9]+)$"
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
    """Verifies exact tool composition for N=10 (5+5), N=50 (5+45), N=128 (5+123)."""
    tools_10 = get_benchmark_tools(10)
    tools_50 = get_benchmark_tools(50)
    tools_128 = get_benchmark_tools(128)

    assert len(tools_10) == 10
    assert len(tools_50) == 50
    assert len(tools_128) == 128

    # Ensure all tool names within each set are unique
    for tool_set in (tools_10, tools_50, tools_128):
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
    """Verifies that the number of tools in body.tools exactly matches N (10, 50, 128)."""
    compiler = BatchDatasetCompiler(seed=42)
    requests = compiler.compile_requests()

    for req in requests:
        cid = req["custom_id"]
        tools = req["body"]["tools"]

        if "_N10_" in cid:
            assert len(tools) == 10
        elif "_N50_" in cid:
            assert len(tools) == 50
        elif "_N128_" in cid:
            assert len(tools) == 128
        else:
            raise ValueError(f"Unknown entropy level marker in {cid}")


def test_batch_jsonl_generation_and_parseable_lines(tmp_path: Path) -> None:
    """Verifies generation of partitioned JSONL files for each model under OpenAI Batch API format."""
    compiler = BatchDatasetCompiler(seed=42)

    generated_files = compiler.generate_batch_jsonl(tmp_path)
    assert len(generated_files) == 4
    assert set(generated_files.keys()) == set(compiler.MODELS)

    total_lines = 0
    for model, path in generated_files.items():
        assert path.exists()
        assert path.name == f"eval_batch_{model}.jsonl"

        expected_reasoning = "low" if model == "gpt-6-astra" else "none"
        n10_count = 0
        n50_count = 0
        n128_count = 0
        line_count = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line_count += 1
                record = json.loads(line)
                assert record["method"] == "POST"
                cid = record["custom_id"]
                if "_N10_" in cid:
                    n10_count += 1
                elif "_N50_" in cid:
                    n50_count += 1
                elif "_N128_" in cid:
                    n128_count += 1

                body = record["body"]
                assert body["model"] == model

                if model == "gpt-6-astra":
                    assert record["url"] == "/v1/responses"
                    assert body["reasoning"] == {"effort": "low"}
                    assert "instructions" in body
                    assert "input" in body
                    assert isinstance(body["tools"], list)
                    assert "name" in body["tools"][0]
                    assert "description" in body["tools"][0]
                    assert "parameters" in body["tools"][0]
                else:
                    assert record["url"] == "/v1/chat/completions"
                    assert body["reasoning_effort"] == expected_reasoning
                    assert isinstance(body["messages"], list)
                    assert len(body["messages"]) >= 2
                    assert isinstance(body["tools"], list)
                    assert "function" in body["tools"][0]

        assert line_count == 300
        assert n10_count == 100
        assert n50_count == 100
        assert n128_count == 100
        total_lines += line_count

    assert total_lines == 1200


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
