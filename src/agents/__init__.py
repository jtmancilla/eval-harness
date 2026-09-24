"""Hierarchical Multi-Agent Router package for regulated financial systems (Phase 2B)."""
from __future__ import annotations

from src.agents.router import CANONICAL_TOOL_REGISTRY, HierarchicalRouter
from src.agents.schemas import (
    AgentDomain,
    RouterTriageDecision,
    RoutingDecision,
)

__all__ = [
    "CANONICAL_TOOL_REGISTRY",
    "HierarchicalRouter",
    "AgentDomain",
    "RouterTriageDecision",
    "RoutingDecision",
]
