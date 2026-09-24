"""Unit tests for adversarial scenario generation, homoglyph isolation, and batch compilation."""
from __future__ import annotations

from pathlib import Path

from src.agents.router import CANONICAL_TOOL_REGISTRY, HierarchicalRouter
from src.agents.schemas import RoutingDecision
from src.eval.adversarial_scenarios import (
    HOMOGLYPH_TRAP_TOOLS,
    AdversarialAttackType,
    generate_adversarial_scenarios,
    get_adversarial_benchmark_tools,
)
from src.eval.batch_generator import BatchDatasetCompiler
from src.eval.trace_auditor import CUSTOM_ID_REGEX


def test_adversarial_scenarios_distribution() -> None:
    """Verifies that generate_adversarial_scenarios respects the 60/50/40 distribution."""
    scenarios = generate_adversarial_scenarios(
        distribution={
            "authority_jailbreak": 60,
            "indirect_injection": 50,
            "homoglyph_trap": 40,
        },
        seed=42,
    )
    assert len(scenarios) == 150

    authority = [s for s in scenarios if s.attack_type == AdversarialAttackType.AUTHORITY_JAILBREAK]
    injection = [s for s in scenarios if s.attack_type == AdversarialAttackType.INDIRECT_INJECTION]
    homoglyph = [s for s in scenarios if s.attack_type == AdversarialAttackType.HOMOGLYPH_TRAP]

    assert len(authority) == 60
    assert len(injection) == 50
    assert len(homoglyph) == 40

    # Authority cases must have prompt pretexts from official bodies
    for s in authority:
        assert len(s.prompt_pretext) > 0
        assert any(agency in s.prompt_pretext for agency in ("CNBV", "SAT", "BANXICO", "DOF", "UIF"))

    # Injection cases must have injection strings in titular or concepto
    for s in injection:
        has_titular_inj = any(k in s.titular for k in ("SYSTEM", "OVERRIDE", "ADMIN", "<system>", "EXECUTE"))
        has_concepto_inj = any(k in s.concepto for k in ("SYSTEM", "OVERRIDE", "NOTA", "UPDATE", "<prompt_injection>"))
        assert has_titular_inj or has_concepto_inj


def test_adversarial_tool_catalog_n128() -> None:
    """Verifies that adversarial tool catalog provides exactly 128 tools including 10 homoglyphs."""
    tools = get_adversarial_benchmark_tools(128)
    assert len(tools) == 128

    trap_names = {t["function"]["name"] for t in HOMOGLYPH_TRAP_TOOLS}
    tool_names = {t["function"]["name"] for t in tools}

    assert trap_names.issubset(tool_names)
    assert len(trap_names) == 10


def test_hierarchical_router_purges_all_homoglyphs() -> None:
    """Verifies that HierarchicalRouter purges all Cyrillic homoglyph traps deterministically."""
    tools = get_adversarial_benchmark_tools(128)
    trap_names = {t["function"]["name"] for t in HOMOGLYPH_TRAP_TOOLS}

    for agent_role, allowed in CANONICAL_TOOL_REGISTRY.items():
        decision = RoutingDecision(
            target_agent=agent_role,  # type: ignore[arg-type]
            classified_intent="VALIDATE_ACCOUNT",
            allowed_tools=allowed,
            routing_rationale="Testing isolation",
        )
        pruned = HierarchicalRouter.prune_tool_catalog(decision, tools)
        pruned_names = {HierarchicalRouter._extract_tool_name(t) for t in pruned}

        # Check k <= 3
        assert 1 <= len(pruned) <= 3
        # Check no homoglyphs leak
        assert len(pruned_names & trap_names) == 0
        # Check all are canonical
        assert pruned_names.issubset(set(allowed))


def test_adversarial_batch_compiler_from_yaml(tmp_path: Path) -> None:
    """Verifies that BatchDatasetCompiler parses adversarial matrix and produces valid custom_ids."""
    matrix_path = Path("configs/adversarial_matrix.yaml")
    compiler = BatchDatasetCompiler.from_yaml(matrix_path)
    assert compiler.is_adversarial is True
    assert compiler.models == ("gpt-6-astra", "gpt-5.6-sol")

    requests = compiler.compile_requests()
    assert len(requests) == 2000  # (3*2*250) + (1*2*250)

    for req in requests:
        cid = req["custom_id"]
        match = CUSTOM_ID_REGEX.match(cid)
        assert match is not None, f"Custom ID {cid} did not match regex"
        assert match.group("adv_type") in ("authority_jailbreak", "indirect_injection", "homoglyph_trap")
        assert match.group("entropy") == "128"
        assert match.group("condition") in ("baseline_autorregresivo", "hierarchical_router")
