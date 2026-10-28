"""Deterministic Causal DAG Factory for Pearl's Ladder of Causation Benchmark.

Generates structurally diverse Directed Acyclic Graphs (DAGs) with linear Structural
Causal Models (SCMs), computing analytically exact ground truth for three levels of
causal reasoning:
  - L1 (Observational): Pearson correlation under the full generative model.
  - L2 (Interventional): Average Causal Effect via do-calculus truncation.
  - L3 (Counterfactual): Abduction-Action-Prediction under linear SCM identities.

This module is purely deterministic (no LLM, no network IO) and follows the
purity constraints of src/gates/ (sub-millisecond, pure functions).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx  # type: ignore[import-untyped]
import numpy as np

# --------------------------------------------------------------------------- #
# Data Contracts (frozen, immutable)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CounterfactualContext:
    """Observed and counterfactual values for L3 evaluation."""

    x_observed: float
    y_observed: float
    x_counterfactual: float
    y_counterfactual: float  # Ground truth answer


@dataclass(frozen=True)
class CausalGroundTruth:
    """Analytically computed ground truth for all three rungs of Pearl's Ladder."""

    l1_correlation: float  # Pearson r(X, Y) under observation
    l1_direction: str  # "positive" | "negative" | "none"
    l2_causal_effect: float  # ACE = E[Y | do(X = x+1)] - E[Y | do(X = x)]
    l2_direction: str  # "positive" | "negative" | "none"
    l3_context: CounterfactualContext


@dataclass(frozen=True)
class CausalDAGSpec:
    """Complete specification of a causal DAG with ground truth metadata."""

    dag_id: str
    template_name: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, float], ...]  # (source, target, coefficient)
    noise_stds: tuple[float, ...]  # One per node, same order as nodes
    treatment_node: str
    outcome_node: str
    has_collider: bool
    has_confounder: bool
    has_instrument: bool
    ground_truth: CausalGroundTruth


# --------------------------------------------------------------------------- #
# Direction Classification
# --------------------------------------------------------------------------- #

_CORR_THRESHOLD = 0.10
_EFFECT_THRESHOLD = 0.05


def classify_direction(value: float, threshold: float) -> str:
    """Classify a numerical value as positive, negative, or none."""
    if abs(value) < threshold:
        return "none"
    return "positive" if value > 0 else "negative"


# --------------------------------------------------------------------------- #
# Analytical Ground Truth Computation (numpy)
# --------------------------------------------------------------------------- #


def build_coefficient_matrix(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str, float]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Build coefficient matrix B where B[i,j] = coeff of j in equation for i.

    Convention: X_i = sum_j B[i,j] * X_j + eps_i
    """
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    B = np.zeros((n, n))
    for source, target, coeff in edges:
        B[node_idx[target], node_idx[source]] = coeff
    return B, node_idx


def compute_covariance_matrix(
    B: np.ndarray,
    noise_vars: np.ndarray,
) -> np.ndarray:
    """Compute the covariance matrix of the observed variables.

    For linear SCM: X = (I - B)^{-1} eps
    Sigma = (I-B)^{-1} Sigma_eps ((I-B)^{-1})^T
    """
    n = B.shape[0]
    inv = np.linalg.inv(np.eye(n) - B)
    return inv @ np.diag(noise_vars) @ inv.T


def compute_observational_correlation(
    B: np.ndarray,
    noise_vars: np.ndarray,
    source_idx: int,
    target_idx: int,
) -> float:
    """Compute Pearson correlation r(X, Y) under the observational distribution."""
    sigma = compute_covariance_matrix(B, noise_vars)
    cov = sigma[source_idx, target_idx]
    std_x = np.sqrt(sigma[source_idx, source_idx])
    std_y = np.sqrt(sigma[target_idx, target_idx])
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
    return float(cov / (std_x * std_y))


def compute_causal_effect(
    B: np.ndarray,
    source_idx: int,
    target_idx: int,
) -> float:
    """Compute Average Causal Effect via do-calculus: mutilate incoming edges to treatment.

    E[Y | do(X=x)] = ACE * x + const
    Returns the ACE coefficient.
    """
    n = B.shape[0]
    B_mut = B.copy()
    B_mut[source_idx, :] = 0.0  # Remove all parents of treatment
    inv = np.linalg.inv(np.eye(n) - B_mut)
    return float(inv[target_idx, source_idx])


def compute_counterfactual(
    B: np.ndarray,
    noise_vars: np.ndarray,
    source_idx: int,
    target_idx: int,
    rng: np.random.Generator,
) -> CounterfactualContext:
    """Compute counterfactual Y_{X=x'} using linear SCM identity.

    For linear SCMs the counterfactual simplifies to:
        Y_cf = Y_obs + ACE * (x' - x_obs)
    """
    n = B.shape[0]
    noise_stds = np.sqrt(noise_vars)
    eps = rng.normal(0.0, noise_stds, size=n)

    # Compute observed values: X = (I - B)^{-1} @ eps
    inv = np.linalg.inv(np.eye(n) - B)
    x_vec = inv @ eps

    x_obs = round(float(x_vec[source_idx]), 4)
    y_obs = round(float(x_vec[target_idx]), 4)

    # Counterfactual: x' = x_obs + 1.0 (unit perturbation)
    x_prime = round(x_obs + 1.0, 4)

    # Linear SCM identity: y_cf = y_obs + ACE * delta_x
    ace = compute_causal_effect(B, source_idx, target_idx)
    y_cf = round(y_obs + ace * (x_prime - x_obs), 4)

    return CounterfactualContext(
        x_observed=x_obs,
        y_observed=y_obs,
        x_counterfactual=x_prime,
        y_counterfactual=y_cf,
    )


# --------------------------------------------------------------------------- #
# DAG Factory
# --------------------------------------------------------------------------- #


class CausalDAGFactory:
    """Generates structurally diverse causal DAGs with analytical ground truth.

    Supports 6 template topologies covering all canonical causal structures:
    confounders, colliders, mediators, instrumental variables, diamond, and
    full composite models.  Coefficients are drawn from a seeded RNG to ensure
    diversity across DAGs while preserving reproducibility.
    """

    TEMPLATES: tuple[str, ...] = (
        "confounding_fork",
        "mediator_chain",
        "collider",
        "instrumental_variable",
        "diamond",
        "full_model",
    )

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------- #
    # Coefficient Generators
    # ------------------------------------------------------------------- #

    def _positive_coeff(self) -> float:
        """Draw a positive coefficient in [0.3, 0.9]."""
        return round(float(self._rng.uniform(0.3, 0.9)), 2)

    def _signed_coeff(self) -> float:
        """Draw a coefficient with random sign in [-0.9, -0.3] ∪ [0.3, 0.9]."""
        mag = self._rng.uniform(0.3, 0.9)
        sign = self._rng.choice([-1.0, 1.0])
        return round(float(mag * sign), 2)

    # ------------------------------------------------------------------- #
    # Ground Truth Builder
    # ------------------------------------------------------------------- #

    def build_spec(
        self,
        template_name: str,
        dag_index: int,
        nodes: tuple[str, ...],
        edges: tuple[tuple[str, str, float], ...],
        noise_stds: tuple[float, ...],
        treatment: str,
        outcome: str,
        has_collider: bool,
        has_confounder: bool,
        has_instrument: bool,
    ) -> CausalDAGSpec:
        """Compute ground truth and assemble the full DAGSpec."""
        B, node_idx = build_coefficient_matrix(nodes, edges)
        noise_vars = np.array([s**2 for s in noise_stds])
        src_idx = node_idx[treatment]
        tgt_idx = node_idx[outcome]

        # L1: Observational correlation
        l1_corr = compute_observational_correlation(B, noise_vars, src_idx, tgt_idx)

        # L2: Average Causal Effect
        l2_ace = compute_causal_effect(B, src_idx, tgt_idx)

        # L3: Counterfactual (deterministic sub-RNG per DAG)
        cf_rng = np.random.default_rng(self._rng.integers(0, 2**31))
        l3_ctx = compute_counterfactual(B, noise_vars, src_idx, tgt_idx, cf_rng)

        # Validate acyclicity with networkx
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for src, tgt, _ in edges:
            G.add_edge(src, tgt)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError(
                f"Generated graph for {template_name}_{dag_index:03d} is not a DAG"
            )

        ground_truth = CausalGroundTruth(
            l1_correlation=round(l1_corr, 4),
            l1_direction=classify_direction(l1_corr, _CORR_THRESHOLD),
            l2_causal_effect=round(l2_ace, 4),
            l2_direction=classify_direction(l2_ace, _EFFECT_THRESHOLD),
            l3_context=l3_ctx,
        )

        return CausalDAGSpec(
            dag_id=f"{template_name}_{dag_index:03d}",
            template_name=template_name,
            nodes=nodes,
            edges=edges,
            noise_stds=noise_stds,
            treatment_node=treatment,
            outcome_node=outcome,
            has_collider=has_collider,
            has_confounder=has_confounder,
            has_instrument=has_instrument,
            ground_truth=ground_truth,
        )

    _build_spec = build_spec

    # ------------------------------------------------------------------- #
    # Template Topologies
    # ------------------------------------------------------------------- #

    def _confounding_fork(self, dag_index: int) -> CausalDAGSpec:
        """Z → X, Z → Y: Correlation without causation.

        Ground truth: L1 ≠ 0 (spurious correlation through Z), L2 = 0.
        This is the classic confounding trap — the model must distinguish
        correlation from causation.
        """
        a = self._positive_coeff()
        b = self._positive_coeff()

        nodes = ("Z", "X", "Y", "W")
        edges = (
            ("Z", "X", a),  # Confounder → Treatment
            ("Z", "Y", b),  # Confounder → Outcome
            ("X", "W", self._signed_coeff()),  # Downstream noise
        )
        noise_stds = (1.0, 1.0, 1.0, 1.0)

        return self._build_spec(
            template_name="confounding_fork",
            dag_index=dag_index,
            nodes=nodes,
            edges=edges,
            noise_stds=noise_stds,
            treatment="X",
            outcome="Y",
            has_collider=False,
            has_confounder=True,
            has_instrument=False,
        )

    def _mediator_chain(self, dag_index: int) -> CausalDAGSpec:
        """X → M → Y: Indirect causal effect through mediator.

        Ground truth: L1 ≠ 0, L2 = coeff(X→M) * coeff(M→Y).
        """
        a = self._positive_coeff()
        b = self._signed_coeff()
        c = self._signed_coeff()

        nodes = ("X", "M", "Y", "C")
        edges = (
            ("X", "M", a),  # Treatment → Mediator
            ("M", "Y", b),  # Mediator → Outcome
            ("C", "Y", c),  # Independent cause of Y (adds realism)
        )
        noise_stds = (1.0, 1.0, 1.0, 1.0)

        return self._build_spec(
            template_name="mediator_chain",
            dag_index=dag_index,
            nodes=nodes,
            edges=edges,
            noise_stds=noise_stds,
            treatment="X",
            outcome="Y",
            has_collider=False,
            has_confounder=False,
            has_instrument=False,
        )

    def _collider(self, dag_index: int) -> CausalDAGSpec:
        """X → C ← Y: Marginal independence, conditional dependence.

        Ground truth: L1 ≈ 0 (marginally independent), L2 = 0.
        The key trap: conditioning on C creates spurious X-Y dependence
        (Berkson's bias).  The model should report no relationship.
        """
        a = self._signed_coeff()
        b = self._signed_coeff()

        nodes = ("X", "Y", "C", "D")
        edges = (
            ("X", "C", a),  # Treatment → Collider
            ("Y", "C", b),  # Outcome → Collider
            ("C", "D", self._signed_coeff()),  # Downstream
        )
        noise_stds = (1.0, 1.0, 1.0, 1.0)

        return self._build_spec(
            template_name="collider",
            dag_index=dag_index,
            nodes=nodes,
            edges=edges,
            noise_stds=noise_stds,
            treatment="X",
            outcome="Y",
            has_collider=True,
            has_confounder=False,
            has_instrument=False,
        )

    def _instrumental_variable(self, dag_index: int) -> CausalDAGSpec:
        """I → X → Y with U → X, U → Y: Confounded but with instrument.

        Ground truth: L1 ≠ 0 (confounded), L2 = direct X→Y coefficient.
        The confounding through U inflates the naive correlation, but the
        do-calculus correctly isolates the true causal effect.
        """
        a = self._positive_coeff()  # I → X
        b = self._signed_coeff()  # X → Y (true causal)
        c = self._positive_coeff()  # U → X (confounder)
        d = self._signed_coeff()  # U → Y (confounder)

        nodes = ("I", "X", "Y", "U", "W")
        edges = (
            ("I", "X", a),  # Instrument → Treatment
            ("X", "Y", b),  # Treatment → Outcome (true effect)
            ("U", "X", c),  # Confounder → Treatment
            ("U", "Y", d),  # Confounder → Outcome
            ("Y", "W", self._signed_coeff()),  # Downstream
        )
        noise_stds = (1.0, 1.0, 1.0, 1.0, 1.0)

        return self._build_spec(
            template_name="instrumental_variable",
            dag_index=dag_index,
            nodes=nodes,
            edges=edges,
            noise_stds=noise_stds,
            treatment="X",
            outcome="Y",
            has_collider=False,
            has_confounder=True,
            has_instrument=True,
        )

    def _diamond(self, dag_index: int) -> CausalDAGSpec:
        """X → M1, X → M2, M1 → Y, M2 → Y: Multiple causal pathways.

        Ground truth: L2 = coeff(X→M1)*coeff(M1→Y) + coeff(X→M2)*coeff(M2→Y).
        Tests whether the model can aggregate effects across parallel mediators.
        """
        a = self._positive_coeff()  # X → M1
        b = self._signed_coeff()  # X → M2
        c = self._positive_coeff()  # M1 → Y
        d = self._signed_coeff()  # M2 → Y

        nodes = ("X", "M1", "M2", "Y")
        edges = (
            ("X", "M1", a),
            ("X", "M2", b),
            ("M1", "Y", c),
            ("M2", "Y", d),
        )
        noise_stds = (1.0, 1.0, 1.0, 1.0)

        return self._build_spec(
            template_name="diamond",
            dag_index=dag_index,
            nodes=nodes,
            edges=edges,
            noise_stds=noise_stds,
            treatment="X",
            outcome="Y",
            has_collider=False,
            has_confounder=False,
            has_instrument=False,
        )

    def _full_model(self, dag_index: int) -> CausalDAGSpec:
        """Composite: instrument + confounder + mediator + collider (6 nodes).

        I → X → M → Y, U → X, U → Y, X → C ← Y
        Ground truth: L2 = coeff(X→M) * coeff(M→Y) (through mediator only).
        This is the hardest template — combines all canonical structures.
        """
        nodes = ("I", "X", "M", "Y", "U", "C")
        edges = (
            ("I", "X", self._positive_coeff()),  # Instrument → Treatment
            ("X", "M", self._positive_coeff()),  # Treatment → Mediator
            ("M", "Y", self._signed_coeff()),  # Mediator → Outcome
            ("U", "X", self._positive_coeff()),  # Confounder → Treatment
            ("U", "Y", self._signed_coeff()),  # Confounder → Outcome
            ("X", "C", self._signed_coeff()),  # Treatment → Collider
            ("Y", "C", self._signed_coeff()),  # Outcome → Collider
        )
        noise_stds = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        return self._build_spec(
            template_name="full_model",
            dag_index=dag_index,
            nodes=nodes,
            edges=edges,
            noise_stds=noise_stds,
            treatment="X",
            outcome="Y",
            has_collider=True,
            has_confounder=True,
            has_instrument=True,
        )

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def generate_scenarios(
        self,
        num_dags: int = 50,
        templates: Sequence[str] | None = None,
    ) -> list[CausalDAGSpec]:
        """Generate a diverse set of causal DAGs with analytically exact ground truth.

        Cycles through templates to ensure balanced topological representation.
        Each DAG receives unique coefficients drawn from the seeded RNG.

        Args:
            num_dags: Total number of DAGs to generate.
            templates: Subset of template names to use (defaults to all 6).

        Returns:
            List of CausalDAGSpec with complete ground truth for L1, L2, L3.
        """
        template_dispatch = {
            "confounding_fork": self._confounding_fork,
            "mediator_chain": self._mediator_chain,
            "collider": self._collider,
            "instrumental_variable": self._instrumental_variable,
            "diamond": self._diamond,
            "full_model": self._full_model,
        }

        template_names = list(templates) if templates else list(self.TEMPLATES)
        for name in template_names:
            if name not in template_dispatch:
                raise ValueError(
                    f"Unknown template: {name}. "
                    f"Available: {list(template_dispatch.keys())}"
                )
        template_fns = [template_dispatch[t] for t in template_names]

        scenarios: list[CausalDAGSpec] = []
        for i in range(num_dags):
            fn = template_fns[i % len(template_fns)]
            scenarios.append(fn(dag_index=i))

        return scenarios
