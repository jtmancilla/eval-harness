"""Unit tests for the 147 deterministic decoy tools catalog and entropy injector."""
from __future__ import annotations

import re

from src.tools.decoys import (
    CANONICAL_TOOLS,
    CollisionType,
    generate_decoy_catalog,
    get_active_tools_for_entropy,
)


def test_catalog_exact_total_count() -> None:
    """Verifies that the catalog yields exactly 147 deterministic decoy tools."""
    catalog = generate_decoy_catalog()
    assert len(catalog) == 147


def test_catalog_names_are_unique() -> None:
    """Verifies that all 147 decoy tools have unique canonical names."""
    catalog = generate_decoy_catalog()
    names = [decoy.name for decoy in catalog]
    assert len(names) == len(set(names)) == 147


def test_catalog_three_explicit_families_distribution() -> None:
    """Verifies the exact distribution across the three mandated families: 50, 50, 47."""
    catalog = generate_decoy_catalog()

    family_env = [d for d in catalog if d.collision_type == CollisionType.ENVIRONMENT_LEGACY]
    family_syn = [d for d in catalog if d.collision_type == CollisionType.OPERATIONAL_SYNONYM]
    family_bypass = [d for d in catalog if d.collision_type == CollisionType.REGULATORY_BYPASS]

    assert len(family_env) == 50, f"Expected 50 ENVIRONMENT_LEGACY, got {len(family_env)}"
    assert len(family_syn) == 50, f"Expected 50 OPERATIONAL_SYNONYM, got {len(family_syn)}"
    assert len(family_bypass) == 47, f"Expected 47 REGULATORY_BYPASS, got {len(family_bypass)}"


def test_no_sequential_numeric_filler_suffixes() -> None:
    """Ensures strictly NO sequential filler suffixes (_01, _02, _001, etc.) exist."""
    catalog = generate_decoy_catalog()
    filler_suffix_pattern = re.compile(r"_\d+$")

    offending_tools: list[str] = []
    for decoy in catalog:
        if filler_suffix_pattern.search(decoy.name):
            offending_tools.append(decoy.name)

    assert not offending_tools, f"Found tools with numeric filler suffixes: {offending_tools}"


def test_decoy_json_schemas_validity() -> None:
    """Ensures all decoy tools define valid JSON schema objects for OpenAI function calling."""
    catalog = generate_decoy_catalog()

    for decoy in catalog:
        tool_spec = decoy.to_openai_tool()
        assert tool_spec["type"] == "function"
        fn = tool_spec["function"]
        assert fn["name"] == decoy.name
        assert len(fn["description"]) > 10

        params = fn["parameters"]
        assert params.get("type") == "object"
        assert isinstance(params.get("properties"), dict)
        assert len(params["properties"]) >= 1
        assert isinstance(params.get("required"), list)


def test_decoy_canonical_targets_validity() -> None:
    """Ensures every decoy points to a recognized canonical pipeline tool."""
    catalog = generate_decoy_catalog()
    for decoy in catalog:
        assert decoy.target_canonical_tool in CANONICAL_TOOLS


def test_entropy_injection_scaling() -> None:
    """Verifies that active tool catalogs scale according to entropy levels (N=10, 50, 150)."""
    canonical_mock = [{"type": "function", "function": {"name": "canonical_spei"}}]

    tools_10 = get_active_tools_for_entropy(10, canonical_mock)
    assert len(tools_10) == 1 + 10

    tools_50 = get_active_tools_for_entropy(50, canonical_mock)
    assert len(tools_50) == 1 + 50

    tools_150 = get_active_tools_for_entropy(150, canonical_mock)
    # Capped at 147 decoys + 1 canonical
    assert len(tools_150) == 1 + 147
