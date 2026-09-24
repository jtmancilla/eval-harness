"""Unit tests for Hierarchical Multi-Agent Router and Handoff Schemas (Phase 2B)."""
from __future__ import annotations

import pytest

from src.agents.router import HierarchicalRouter
from src.agents.schemas import AgentDomain, RouterTriageDecision


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

    with pytest.raises(Exception):
        # Attempting mutation must fail due to frozen=True
        decision.selected_domain = AgentDomain.TREASURY  # type: ignore[misc]


def test_hierarchical_router_initialization() -> None:
    """Verifies basic initialization of HierarchicalRouter scaffolding."""
    router = HierarchicalRouter(full_catalog=[])
    assert router._full_catalog == []
