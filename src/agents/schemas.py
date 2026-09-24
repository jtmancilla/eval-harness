"""Pydantic V2 contracts and handoff schemas for Hierarchical Multi-Agent Routing (Phase 2B).

Defines strongly-typed payloads, router classification intent, pruned tool catalogs,
and immutable handoff transitions between triage and domain-specialist agents.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentDomain(StrEnum):
    """Specialized domain sub-catalogs for hierarchical routing."""

    ONBOARDING = "ONBOARDING"
    COMPLIANCE = "COMPLIANCE"
    TREASURY = "TREASURY"
    REJECTION = "REJECTION"


class RoutingDecision(BaseModel):
    """Immutable routing decision output by HierarchicalRouter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_agent: Literal["fiscal_validator", "clabe_validator", "treasury_executor"] = Field(
        description="Designated specialist agent receiving the delegation.",
    )
    classified_intent: Literal["VALIDATE_ACCOUNT", "VALIDATE_IDENTITY", "DISPERSE_SPEI"] = Field(
        description="Classified business intent extracted from user transaction prompt.",
    )
    allowed_tools: list[str] = Field(
        min_length=1,
        max_length=3,
        description="Whitelisted tool identifiers strictly scoped for the target specialist agent.",
    )
    routing_rationale: str = Field(
        description="Technical and compliance rationale justifying the triage routing decision.",
    )


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


__all__ = [
    "AgentDomain",
    "RoutingDecision",
    "RouterTriageDecision",
]
