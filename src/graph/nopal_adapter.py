"""Mock/stub adapter for NopalDB graph database trace persistence and state audit."""
from __future__ import annotations

from typing import Any
from uuid import UUID

from src.contracts.handoff import AgentState


class NopalDBNode:
    """Represents an immutable node in the NopalDB audit graph."""

    def __init__(self, trace_id: UUID, state: AgentState, metadata: dict[str, Any]) -> None:
        self.trace_id = trace_id
        self.state = state
        self.metadata = metadata


class NopalDBEdge:
    """Represents a validated transition edge between states in NopalDB."""

    def __init__(
        self,
        trace_id: UUID,
        from_state: AgentState,
        to_state: AgentState,
        symbolic_hash: str,
    ) -> None:
        self.trace_id = trace_id
        self.from_state = from_state
        self.to_state = to_state
        self.symbolic_hash = symbolic_hash


class NopalDBAdapter:
    """In-memory stub representing connection to NopalDB graph database.

    Maintains immutable node and edge logs per trace_id for forensic audit.
    """

    def __init__(self, connection_uri: str = "memory://nopaldb/local") -> None:
        self.connection_uri = connection_uri
        self._nodes: dict[UUID, list[NopalDBNode]] = {}
        self._edges: dict[UUID, list[NopalDBEdge]] = {}

    def record_node(
        self,
        trace_id: UUID,
        state: AgentState,
        metadata: dict[str, Any] | None = None,
    ) -> NopalDBNode:
        """Persists a new state node in the audit graph."""
        node = NopalDBNode(trace_id=trace_id, state=state, metadata=metadata or {})
        self._nodes.setdefault(trace_id, []).append(node)
        return node

    def record_edge(
        self,
        trace_id: UUID,
        from_state: AgentState,
        to_state: AgentState,
        symbolic_hash: str,
    ) -> NopalDBEdge:
        """Records a verified transition edge sealed with its symbolic SHA-256 hash."""
        edge = NopalDBEdge(
            trace_id=trace_id,
            from_state=from_state,
            to_state=to_state,
            symbolic_hash=symbolic_hash,
        )
        self._edges.setdefault(trace_id, []).append(edge)
        return edge

    def has_compliance_approval(self, trace_id: UUID) -> bool:
        """Verifies if the trace has formally traversed COMPLIANCE_APPROVED."""
        nodes = self._nodes.get(trace_id, [])
        return any(node.state == AgentState.COMPLIANCE_APPROVED for node in nodes)

    def get_trace_history(self, trace_id: UUID) -> list[AgentState]:
        """Retrieves ordered sequence of traversed states for the given trace."""
        return [node.state for node in self._nodes.get(trace_id, [])]
