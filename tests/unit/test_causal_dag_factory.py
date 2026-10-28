"""Unit tests for the Causal DAG Factory (Phase 1).

Validates topological correctness, analytical ground truth computation, and the
key mathematical properties of each template topology:
- Confounding fork: L2 = 0 (correlation without causation)
- Collider: L1 ≈ 0 (marginal independence)
- Mediator chain: L2 = product of path coefficients
- Diamond: L2 = sum of path products
- Instrumental variable: L2 = direct X→Y coefficient only
- Full model: L2 = mediator path product only
- Counterfactual identity: y_cf = y_obs + ACE * Δx (linear SCM)
"""
from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
import pytest

from src.eval.causal_dag_factory import (
    CausalDAGFactory,
    classify_direction,
)


class TestClassifyDirection:
    """Tests for the direction classification utility."""

    def test_positive(self) -> None:
        assert classify_direction(0.5, 0.1) == "positive"

    def test_negative(self) -> None:
        assert classify_direction(-0.3, 0.1) == "negative"

    def test_none_at_threshold(self) -> None:
        assert classify_direction(0.05, 0.1) == "none"

    def test_none_zero(self) -> None:
        assert classify_direction(0.0, 0.05) == "none"


class TestCausalDAGFactory:
    """Tests for DAG generation and ground truth computation."""

    def test_generate_correct_count(self) -> None:
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=12)
        assert len(scenarios) == 12

    def test_all_dags_are_acyclic(self) -> None:
        """Every generated DAG must pass networkx acyclicity validation."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=30)
        for dag in scenarios:
            G = nx.DiGraph()
            G.add_nodes_from(dag.nodes)
            for s, t, _ in dag.edges:
                G.add_edge(s, t)
            assert nx.is_directed_acyclic_graph(G), f"{dag.dag_id} is not a DAG"

    def test_template_balance(self) -> None:
        """Templates cycle evenly across generated DAGs."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=12)
        templates = [s.template_name for s in scenarios]
        assert templates.count("confounding_fork") == 2
        assert templates.count("mediator_chain") == 2
        assert templates.count("collider") == 2
        assert templates.count("instrumental_variable") == 2
        assert templates.count("diamond") == 2
        assert templates.count("full_model") == 2

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical results."""
        factory1 = CausalDAGFactory(seed=123)
        factory2 = CausalDAGFactory(seed=123)
        s1 = factory1.generate_scenarios(num_dags=6)
        s2 = factory2.generate_scenarios(num_dags=6)
        for a, b in zip(s1, s2, strict=True):
            assert a.dag_id == b.dag_id
            assert a.ground_truth.l1_correlation == b.ground_truth.l1_correlation
            assert a.ground_truth.l2_causal_effect == b.ground_truth.l2_causal_effect
            assert a.ground_truth.l3_context.y_counterfactual == b.ground_truth.l3_context.y_counterfactual

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different coefficients."""
        factory1 = CausalDAGFactory(seed=42)
        factory2 = CausalDAGFactory(seed=999)
        s1 = factory1.generate_scenarios(num_dags=6)
        s2 = factory2.generate_scenarios(num_dags=6)
        # At least some edges should have different coefficients
        diffs = sum(
            1
            for a, b in zip(s1, s2, strict=True)
            if a.ground_truth.l2_causal_effect != b.ground_truth.l2_causal_effect
        )
        assert diffs > 0

    def test_ground_truth_fields_complete(self) -> None:
        """All ground truth fields are populated and well-typed."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=12)
        for dag in scenarios:
            gt = dag.ground_truth
            assert gt.l1_direction in ("positive", "negative", "none")
            assert gt.l2_direction in ("positive", "negative", "none")
            assert isinstance(gt.l1_correlation, float)
            assert isinstance(gt.l2_causal_effect, float)
            assert isinstance(gt.l3_context.x_observed, float)
            assert isinstance(gt.l3_context.y_observed, float)
            assert isinstance(gt.l3_context.x_counterfactual, float)
            assert isinstance(gt.l3_context.y_counterfactual, float)

    def test_dag_spec_immutable(self) -> None:
        """CausalDAGSpec is frozen and cannot be mutated."""
        factory = CausalDAGFactory(seed=42)
        dag = factory.generate_scenarios(num_dags=1)[0]
        with pytest.raises(AttributeError):
            dag.dag_id = "mutated"  # type: ignore[misc]

    # ------------------------------------------------------------------- #
    # Template-Specific Ground Truth Validation
    # ------------------------------------------------------------------- #

    def test_confounding_fork_l2_is_zero(self) -> None:
        """In a confounding fork (Z→X, Z→Y), the causal effect X→Y is exactly 0."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=6, templates=["confounding_fork"])
        for dag in scenarios:
            assert abs(dag.ground_truth.l2_causal_effect) < 1e-10
            assert dag.ground_truth.l2_direction == "none"
            # But L1 should be nonzero (spurious correlation through Z)
            assert abs(dag.ground_truth.l1_correlation) > 0.1

    def test_collider_marginal_independence(self) -> None:
        """In a collider (X→C←Y), X and Y are marginally independent."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=6, templates=["collider"])
        for dag in scenarios:
            # Marginal independence: correlation is exactly 0
            assert abs(dag.ground_truth.l1_correlation) < 1e-10
            assert dag.ground_truth.l1_direction == "none"
            # No causal effect either
            assert abs(dag.ground_truth.l2_causal_effect) < 1e-10
            assert dag.ground_truth.l2_direction == "none"

    def test_mediator_chain_l2_equals_product(self) -> None:
        """In X→M→Y, ACE = coeff(X→M) * coeff(M→Y)."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=6, templates=["mediator_chain"])
        for dag in scenarios:
            xm = next(c for s, t, c in dag.edges if s == "X" and t == "M")
            my = next(c for s, t, c in dag.edges if s == "M" and t == "Y")
            expected_ace = xm * my
            assert abs(dag.ground_truth.l2_causal_effect - expected_ace) < 1e-6

    def test_diamond_l2_is_sum_of_paths(self) -> None:
        """In diamond X→M1→Y + X→M2→Y, ACE = a*c + b*d."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=6, templates=["diamond"])
        for dag in scenarios:
            xm1 = next(c for s, t, c in dag.edges if s == "X" and t == "M1")
            xm2 = next(c for s, t, c in dag.edges if s == "X" and t == "M2")
            m1y = next(c for s, t, c in dag.edges if s == "M1" and t == "Y")
            m2y = next(c for s, t, c in dag.edges if s == "M2" and t == "Y")
            expected_ace = xm1 * m1y + xm2 * m2y
            assert abs(dag.ground_truth.l2_causal_effect - expected_ace) < 1e-6

    def test_instrumental_variable_l2_is_direct_effect(self) -> None:
        """In IV model (I→X→Y, U→X, U→Y), ACE = coeff(X→Y) only."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(
            num_dags=6, templates=["instrumental_variable"]
        )
        for dag in scenarios:
            xy_coeff = next(c for s, t, c in dag.edges if s == "X" and t == "Y")
            assert abs(dag.ground_truth.l2_causal_effect - xy_coeff) < 1e-6

    def test_full_model_l2_is_mediator_path(self) -> None:
        """In full model (I→X→M→Y + confounders + collider), ACE = X→M * M→Y."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=6, templates=["full_model"])
        for dag in scenarios:
            xm = next(c for s, t, c in dag.edges if s == "X" and t == "M")
            my = next(c for s, t, c in dag.edges if s == "M" and t == "Y")
            expected_ace = xm * my
            assert abs(dag.ground_truth.l2_causal_effect - expected_ace) < 1e-6

    # ------------------------------------------------------------------- #
    # Counterfactual Identity (holds for ALL linear SCMs)
    # ------------------------------------------------------------------- #

    def test_counterfactual_identity(self) -> None:
        """For linear SCMs: y_cf = y_obs + ACE * (x' - x_obs).

        This is the fundamental theorem of linear counterfactuals.
        """
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=30)
        for dag in scenarios:
            gt = dag.ground_truth
            cf = gt.l3_context
            delta_x = cf.x_counterfactual - cf.x_observed
            expected_y_cf = cf.y_observed + gt.l2_causal_effect * delta_x
            assert abs(cf.y_counterfactual - expected_y_cf) < 1e-6, (
                f"{dag.dag_id}: counterfactual identity violated. "
                f"Expected {expected_y_cf:.6f}, got {cf.y_counterfactual:.6f}"
            )

    def test_counterfactual_delta_x_is_one(self) -> None:
        """The counterfactual perturbation is always exactly +1.0."""
        factory = CausalDAGFactory(seed=42)
        scenarios = factory.generate_scenarios(num_dags=12)
        for dag in scenarios:
            cf = dag.ground_truth.l3_context
            delta = cf.x_counterfactual - cf.x_observed
            assert abs(delta - 1.0) < 1e-6

    # ------------------------------------------------------------------- #
    # Structural Features
    # ------------------------------------------------------------------- #

    def test_structural_features_confounding_fork(self) -> None:
        factory = CausalDAGFactory(seed=42)
        dag = factory.generate_scenarios(num_dags=1, templates=["confounding_fork"])[0]
        assert dag.has_confounder is True
        assert dag.has_collider is False
        assert dag.has_instrument is False

    def test_structural_features_collider(self) -> None:
        factory = CausalDAGFactory(seed=42)
        dag = factory.generate_scenarios(num_dags=1, templates=["collider"])[0]
        assert dag.has_collider is True
        assert dag.has_confounder is False
        assert dag.has_instrument is False

    def test_structural_features_instrumental_variable(self) -> None:
        factory = CausalDAGFactory(seed=42)
        dag = factory.generate_scenarios(
            num_dags=1, templates=["instrumental_variable"]
        )[0]
        assert dag.has_instrument is True
        assert dag.has_confounder is True
        assert dag.has_collider is False

    def test_structural_features_full_model(self) -> None:
        factory = CausalDAGFactory(seed=42)
        dag = factory.generate_scenarios(num_dags=1, templates=["full_model"])[0]
        assert dag.has_collider is True
        assert dag.has_confounder is True
        assert dag.has_instrument is True

    def test_unknown_template_raises(self) -> None:
        factory = CausalDAGFactory(seed=42)
        with pytest.raises(ValueError, match="Unknown template"):
            factory.generate_scenarios(num_dags=1, templates=["nonexistent"])
