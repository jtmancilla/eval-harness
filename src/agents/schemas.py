"""Pydantic V2 contracts and handoff schemas for Hierarchical Multi-Agent Routing (Phase 2B).

Defines strongly-typed payloads, router classification intent, pruned tool catalogs,
and immutable handoff transitions between triage and domain-specialist agents.
"""
from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentDomain(StrEnum):
    """Specialized domain sub-catalogs for hierarchical routing."""

    ONBOARDING = "ONBOARDING"
    COMPLIANCE = "COMPLIANCE"
    TREASURY = "TREASURY"
    REJECTION = "REJECTION"


class RouterTriageDecision(BaseModel):
    """Immutable routing and catalog pruning decision produced by the hierarchical router."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the router triage decision.",
    )
    selected_domain: AgentDomain = Field(
        description="Target specialized domain selected for tool delegation.",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Router confidence score for the selected domain classification.",
    )
    pruned_tool_names: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Subset of canonical and decoy tools exposed to the target specialist agent.",
    )
    rationale: str = Field(
        default="",
        description="Formal justification or chain-of-routing summary for audit compliance.",
    )
