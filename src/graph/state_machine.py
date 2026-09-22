"""Deterministic directed acyclic graph (DAG) state machine for workflow orchestration."""
from __future__ import annotations

from typing import Any
from uuid import UUID

from src.contracts.handoff import AgentState, HandoffEnvelope
from src.gates.state_guard import StateGuard


class FinancialWorkflowStateMachine:
    """Manages sequential state transitions according to canonical governance rules."""

    def __init__(self, trace_id: UUID) -> None:
        self.trace_id = trace_id
        self._current_state: AgentState = AgentState.INITIALIZED
        self._history: list[AgentState] = [AgentState.INITIALIZED]
        self._envelopes: list[HandoffEnvelope[Any]] = []

    @property
    def current_state(self) -> AgentState:
        """Current operational state."""
        return self._current_state

    @property
    def history(self) -> list[AgentState]:
        """Immutable copy of transition history."""
        return list(self._history)

    def transition(self, envelope: HandoffEnvelope[Any]) -> AgentState:
        """Executes a transition through the StateGuard gate.

        Args:
            envelope: Typed handoff envelope proposing the next state.

        Returns:
            The newly acknowledged current state.

        Raises:
            ShortCircuitViolation: If transition is forbidden or bypasses compliance.
            CryptographicIntegrityViolation: If envelope integrity verification fails.
        """
        if envelope.trace_id != self.trace_id:
            raise ValueError(f"Trace ID mismatch: expected {self.trace_id}, got {envelope.trace_id}")

        if envelope.current_state != self._current_state:
            raise ValueError(
                f"State desynchronization: machine is in {self._current_state.value}, envelope claims {envelope.current_state.value}"
            )

        # Intercept via symbolic gate
        StateGuard.verify_transition(envelope, verified_history=self._history)

        # Commit transition
        self._current_state = envelope.next_state
        self._history.append(envelope.next_state)
        self._envelopes.append(envelope)

        return self._current_state
