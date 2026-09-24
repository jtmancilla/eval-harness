"""Hierarchical Multi-Agent Router and Catalog Pruning Engine (Phase 2B).

Implements deterministic lexical triage, domain intent classification, and tool catalog
pruning to neutralize tool-overreliance and decoy collisions under high entropy (N=128).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.agents.schemas import RoutingDecision

# Canonical authorized tool registry per specialist agent role
CANONICAL_TOOL_REGISTRY: dict[str, list[str]] = {
    "clabe_validator": [
        "parse_account_metadata",
        "verify_clabe_format_v2_canonical",
    ],
    "fiscal_validator": [
        "validate_rfc_structure",
        "check_sat_blacklist",
    ],
    "treasury_executor": [
        "build_spei_instruction",
        "calculate_spei_fee",
    ],
}


class HierarchicalRouter:
    """Orchestrates hierarchical triage and domain-specific tool catalog pruning.

    Enforces deterministic scoping of tools before dispatching requests to specialized
    downstream agents (clabe_validator, fiscal_validator, treasury_executor).
    """

    def __init__(self, full_catalog: Sequence[dict[str, Any]] | None = None) -> None:
        """Initializes the hierarchical router with an optional full tool catalog."""
        self._full_catalog: list[dict[str, Any]] = list(full_catalog) if full_catalog else []

    @staticmethod
    def _extract_tool_name(tool: dict[str, Any]) -> str:
        """Extracts tool name from OpenAI Chat or Responses tool dictionary."""
        if "function" in tool and isinstance(tool["function"], dict):
            return str(tool["function"].get("name", ""))
        return str(tool.get("name", ""))

    @classmethod
    def prune_tool_catalog(
        cls,
        decision: RoutingDecision,
        full_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filtra el array de herramientas dejando únicamente aquellas que coincidan con
        `decision.allowed_tools` Y que pertenezcan a la lista blanca de `CANONICAL_TOOL_REGISTRY`
        para el agente asignado.
        """
        registry_allowed = set(CANONICAL_TOOL_REGISTRY.get(decision.target_agent, []))
        decision_allowed = set(decision.allowed_tools)
        permitted_names = registry_allowed & decision_allowed

        return [
            tool
            for tool in full_tools
            if cls._extract_tool_name(tool) in permitted_names
        ]

    @staticmethod
    def format_specialist_prompt(base_prompt: str, decision: RoutingDecision) -> str:
        """Estructura el system prompt especializado inyectando el rol asignado y la intención."""
        specialist_header = (
            f"[SPECIALIST_AGENT_ROLE: {decision.target_agent}]\n"
            f"Classified Intent: {decision.classified_intent}\n"
            f"Routing Rationale: {decision.routing_rationale}\n"
            f"Allowed Tools: {', '.join(decision.allowed_tools)}\n"
            f"[/SPECIALIST_AGENT_ROLE]\n\n"
        )
        return f"{specialist_header}{base_prompt}"


__all__ = [
    "CANONICAL_TOOL_REGISTRY",
    "HierarchicalRouter",
]
