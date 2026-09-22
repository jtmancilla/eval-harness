"""Deterministic symbolic gate intercepting inter-agent transitions to enforce invariants.

Guarantees graph adjacency, contract immutability, and zero regulatory bypass (Short-Circuit Prevention).
"""
from __future__ import annotations

from typing import Any

from src.contracts.handoff import AgentState, HandoffEnvelope


class GateViolationError(Exception):
    """Base exception for neuro-symbolic gate violations."""


class ShortCircuitViolation(GateViolationError):
    """Raised when an execution trace attempts an unauthorized leap or bypasses compliance."""


class SchemaIntegrityViolation(GateViolationError):
    """Raised when payload attributes violate Pydantic strict schemas or contain forbidden fields."""


class CryptographicIntegrityViolation(GateViolationError):
    """Raised when cryptographic state hashes or trace signatures fail verification."""


# Canonical valid DAG edge transitions: (current_state, next_state)
ALLOWED_TRANSITIONS: frozenset[tuple[AgentState, AgentState]] = frozenset(
    {
        (AgentState.INITIALIZED, AgentState.ONBOARDING_PENDING),
        (AgentState.ONBOARDING_PENDING, AgentState.ONBOARDING_COMPLETED),
        (AgentState.ONBOARDING_COMPLETED, AgentState.COMPLIANCE_PENDING),
        (AgentState.COMPLIANCE_PENDING, AgentState.COMPLIANCE_APPROVED),
        (AgentState.COMPLIANCE_PENDING, AgentState.COMPLIANCE_REJECTED),
        (AgentState.COMPLIANCE_APPROVED, AgentState.TREASURY_PENDING),
        (AgentState.TREASURY_PENDING, AgentState.DISPERSED),
    }
)


class StateGuard:
    """Deterministic runtime interceptor enforcing regulatory and structural invariants."""

    @classmethod
    def verify_transition(
        cls,
        envelope: HandoffEnvelope[Any],
        verified_history: list[AgentState] | None = None,
    ) -> bool:
        """Validates that the requested state transition adheres strictly to the DAG.

        Args:
            envelope: The typed handoff envelope carrying proposed transition.
            verified_history: Sequential list of historically attested states for this trace.

        Returns:
            True if transition satisfies all invariant rules.

        Raises:
            ShortCircuitViolation: If transition edge is invalid, jumps ahead,
                                  or reaches Treasury without COMPLIANCE_APPROVED.
            CryptographicIntegrityViolation: If envelope hash is missing or blank.
        """
        # 1. Cryptographic presence check
        if not envelope.symbolic_hash or len(envelope.symbolic_hash) != 64:
            raise CryptographicIntegrityViolation("Invalid or missing SHA-256 symbolic hash")

        # 2. Graph Adjacency Verification
        transition = (envelope.current_state, envelope.next_state)
        if transition not in ALLOWED_TRANSITIONS:
            raise ShortCircuitViolation(
                f"Unauthorized transition attempt in DAG: {envelope.current_state.value} -> {envelope.next_state.value}"
            )

        # 3. Strict Short-Circuit Invariant: Treasury Dispatch Guard
        if envelope.next_state in (AgentState.TREASURY_PENDING, AgentState.DISPERSED):
            if envelope.current_state != AgentState.COMPLIANCE_APPROVED and envelope.current_state != AgentState.TREASURY_PENDING:
                raise ShortCircuitViolation(
                    f"Short-Circuit detected: Cannot enter '{envelope.next_state.value}' from '{envelope.current_state.value}'"
                )

            # Check trace history to ensure COMPLIANCE_APPROVED was genuinely reached
            if verified_history is not None:
                if AgentState.COMPLIANCE_APPROVED not in verified_history:
                    raise ShortCircuitViolation(
                        "Regulatory Bypass detected: Cannot proceed to Treasury without explicit COMPLIANCE_APPROVED in trace history"
                    )

        return True
