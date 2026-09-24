"""Unit tests for Hierarchical Multi-Agent Router and Handoff Schemas (Phase 2B)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.router import CANONICAL_TOOL_REGISTRY, HierarchicalRouter
from src.agents.schemas import AgentDomain, RouterTriageDecision, RoutingDecision
from src.eval.batch_generator import get_benchmark_tools
from src.tools.decoys import generate_decoy_catalog


def test_router_triage_decision_immutability() -> None:
    """Verifies that RouterTriageDecision enforces Pydantic V2 frozen immutability."""
    decision = RouterTriageDecision(
        selected_domain=AgentDomain.COMPLIANCE,
        confidence_score=0.98,
        pruned_tool_names=("validate_rfc", "check_blacklist_69b"),
        rationale="Tax compliance verification required.",
    )
    assert decision.selected_domain == AgentDomain.COMPLIANCE
    assert decision.confidence_score == 0.98
    assert len(decision.pruned_tool_names) == 2

    with pytest.raises(ValidationError):
        # Attempting mutation must fail due to frozen=True
        decision.selected_domain = AgentDomain.TREASURY  # type: ignore[misc]


def test_routing_decision_rejects_more_than_three_tools() -> None:
    """Validates that RoutingDecision raises ValidationError if allowed_tools > 3."""
    with pytest.raises(ValidationError):
        RoutingDecision(
            target_agent="treasury_executor",
            classified_intent="DISPERSE_SPEI",
            allowed_tools=["tool_1", "tool_2", "tool_3", "tool_4"],
            routing_rationale="Excessive tool allocation exceeding upper bound of 3.",
        )


def test_routing_decision_rejects_empty_tools() -> None:
    """Validates that RoutingDecision raises ValidationError if allowed_tools is empty."""
    with pytest.raises(ValidationError):
        RoutingDecision(
            target_agent="treasury_executor",
            classified_intent="DISPERSE_SPEI",
            allowed_tools=[],
            routing_rationale="Empty tool allocation.",
        )


def test_routing_decision_valid_bounds() -> None:
    """Validates that RoutingDecision accepts valid 1, 2, or 3 tools."""
    for count in (1, 2, 3):
        decision = RoutingDecision(
            target_agent="clabe_validator",
            classified_intent="VALIDATE_ACCOUNT",
            allowed_tools=[f"canonical_tool_{i}" for i in range(count)],
            routing_rationale=f"Valid allocation of {count} tools.",
        )
        assert len(decision.allowed_tools) == count


def test_prune_tool_catalog_reduces_128_to_max_3_canonical() -> None:
    """Validates that prune_tool_catalog reduces a 128-tool catalog to <= 3 canonical tools."""
    catalog_128 = get_benchmark_tools(128)
    assert len(catalog_128) == 128

    decision = RoutingDecision(
        target_agent="clabe_validator",
        classified_intent="VALIDATE_ACCOUNT",
        allowed_tools=["parse_account_metadata", "verify_clabe_format_v2_canonical"],
        routing_rationale="Validation of Mexican bank account format and ownership.",
    )

    pruned = HierarchicalRouter.prune_tool_catalog(decision, catalog_128)
    assert len(pruned) <= 3
    assert len(pruned) > 0

    for tool in pruned:
        name = HierarchicalRouter._extract_tool_name(tool)
        assert name in CANONICAL_TOOL_REGISTRY["clabe_validator"]
        assert name in decision.allowed_tools


def test_prune_tool_catalog_blocks_honeypot_decoys() -> None:
    """Validates that honeypot decoy tools are strictly blocked by CANONICAL_TOOL_REGISTRY."""
    catalog_128 = get_benchmark_tools(128)
    decoys = generate_decoy_catalog()
    honeypot = decoys[0].name

    # Attempting to allow a honeypot decoy alongside a legitimate canonical tool
    decision = RoutingDecision(
        target_agent="treasury_executor",
        classified_intent="DISPERSE_SPEI",
        allowed_tools=["build_spei_instruction", honeypot],
        routing_rationale="Dispersal instruction with adversarial honeypot injection.",
    )

    pruned = HierarchicalRouter.prune_tool_catalog(decision, catalog_128)
    pruned_names = [HierarchicalRouter._extract_tool_name(t) for t in pruned]

    assert honeypot not in pruned_names
    assert "build_spei_instruction" in pruned_names
    assert len(pruned) <= 3


def test_prune_tool_catalog_all_honeypots_blocked() -> None:
    """Validates that if only honeypot decoys are requested, output is an empty catalog."""
    catalog_128 = get_benchmark_tools(128)
    decoys = generate_decoy_catalog()
    honeypot1 = decoys[0].name
    honeypot2 = decoys[1].name

    decision = RoutingDecision(
        target_agent="fiscal_validator",
        classified_intent="VALIDATE_IDENTITY",
        allowed_tools=[honeypot1, honeypot2],
        routing_rationale="Adversarial decision targeting compliance with decoys.",
    )

    pruned = HierarchicalRouter.prune_tool_catalog(decision, catalog_128)
    assert pruned == []


def test_prune_tool_catalog_flat_responses_format() -> None:
    """Validates pruning against flat OpenAI Responses schema (used for reasoning models)."""
    flat_tools = [
        {"type": "function", "name": "build_spei_instruction", "parameters": {}},
        {"type": "function", "name": "calculate_spei_fee", "parameters": {}},
        {"type": "function", "name": "adversarial_honeypot_bypass", "parameters": {}},
    ]
    decision = RoutingDecision(
        target_agent="treasury_executor",
        classified_intent="DISPERSE_SPEI",
        allowed_tools=["build_spei_instruction", "calculate_spei_fee", "adversarial_honeypot_bypass"],
        routing_rationale="SPEI payment fee calculation and instruction assembly.",
    )
    pruned = HierarchicalRouter.prune_tool_catalog(decision, flat_tools)
    names = [HierarchicalRouter._extract_tool_name(t) for t in pruned]

    assert len(pruned) == 2
    assert "build_spei_instruction" in names
    assert "calculate_spei_fee" in names
    assert "adversarial_honeypot_bypass" not in names


def test_format_specialist_prompt() -> None:
    """Validates system prompt formatting with role, intent, and rationale injection."""
    decision = RoutingDecision(
        target_agent="fiscal_validator",
        classified_intent="VALIDATE_IDENTITY",
        allowed_tools=["validate_rfc_structure", "check_sat_blacklist"],
        routing_rationale="RFC verification against SAT blacklist article 69-B.",
    )
    base_prompt = "Process batch item payload for RFC: XAXX010101000."
    formatted = HierarchicalRouter.format_specialist_prompt(base_prompt, decision)

    assert "fiscal_validator" in formatted
    assert "VALIDATE_IDENTITY" in formatted
    assert "RFC verification against SAT blacklist" in formatted
    assert "validate_rfc_structure" in formatted
    assert base_prompt in formatted


def test_trace_auditor_custom_id_regex_hierarchical_router() -> None:
    """Validates that CUSTOM_ID_REGEX in trace_auditor parses hierarchical_router condition."""
    from src.eval.trace_auditor import CUSTOM_ID_REGEX

    cid = "gpt-5.6-sol_N128_hierarchical_router_scenario_006_4ad3979e28f5"
    match = CUSTOM_ID_REGEX.match(cid)
    assert match is not None
    assert match.group("model") == "gpt-5.6-sol"
    assert match.group("entropy") == "128"
    assert match.group("condition") == "hierarchical_router"
    assert match.group("scenario") == "scenario_006"
    assert match.group("trace_id") == "4ad3979e28f5"


def test_batch_compiler_hierarchical_router_from_yaml() -> None:
    """Validates that BatchDatasetCompiler properly compiles hierarchical_router from yaml."""
    from pathlib import Path

    from src.eval.batch_generator import BatchDatasetCompiler
    from src.eval.trace_auditor import CUSTOM_ID_REGEX

    yaml_path = Path("configs/router_matrix.yaml")
    compiler = BatchDatasetCompiler.from_yaml(yaml_path)
    requests = compiler.compile_requests()

    assert len(requests) == 120

    baseline_reqs = [r for r in requests if "baseline_autorregresivo" in r["custom_id"]]
    router_reqs = [r for r in requests if "hierarchical_router" in r["custom_id"]]

    assert len(baseline_reqs) == 60
    assert len(router_reqs) == 60

    # Baseline has full 128 tools
    for req in baseline_reqs:
        assert len(req["body"]["tools"]) == 128
        assert CUSTOM_ID_REGEX.match(req["custom_id"]) is not None

    # Hierarchical router has pruned tools (<= 3)
    for req in router_reqs:
        assert len(req["body"]["tools"]) <= 3
        match = CUSTOM_ID_REGEX.match(req["custom_id"])
        assert match is not None
        assert match.group("condition") == "hierarchical_router"


def test_trace_auditor_regex_with_effort_tokens() -> None:
    """Validates that CUSTOM_ID_REGEX parses custom_ids with effort tokens."""
    from src.eval.trace_auditor import CUSTOM_ID_REGEX

    for effort in ("low", "medium", "high"):
        cid = f"gpt-6-astra_effort-{effort}_N128_hierarchical_router_scenario_001_abc12345"
        match = CUSTOM_ID_REGEX.match(cid)
        assert match is not None
        assert match.group("model") == "gpt-6-astra"
        assert match.group("effort") == effort
        assert match.group("entropy") == "128"
        assert match.group("condition") == "hierarchical_router"
        assert match.group("scenario") == "scenario_001"
        assert match.group("trace_id") == "abc12345"


def test_batch_compiler_astra_reasoning_escalation() -> None:
    """Validates that BatchDatasetCompiler compiles reasoning escalation matrix properly."""
    from pathlib import Path

    from src.eval.batch_generator import BatchDatasetCompiler
    from src.eval.trace_auditor import CUSTOM_ID_REGEX

    yaml_path = Path("configs/astra_reasoning_matrix.yaml")
    compiler = BatchDatasetCompiler.from_yaml(yaml_path)
    requests = compiler.compile_requests()

    assert len(requests) == 900
    for req in requests:
        cid = req["custom_id"]
        match = CUSTOM_ID_REGEX.match(cid)
        assert match is not None
        assert match.group("model") == "gpt-6-astra"
        assert match.group("effort") in ("low", "medium", "high")
        assert match.group("entropy") == "128"
        assert req["url"] == "/v1/responses"
        assert req["body"]["reasoning"]["effort"] == match.group("effort")


