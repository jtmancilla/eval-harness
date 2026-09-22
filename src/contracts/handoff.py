"""Handoff contracts and state enums for inter-agent delegation."""
from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)


class AgentState(StrEnum):
    """Canonical states for the regulated fintech multi-agent lifecycle."""

    INITIALIZED = "INITIALIZED"
    ONBOARDING_PENDING = "ONBOARDING_PENDING"
    ONBOARDING_COMPLETED = "ONBOARDING_COMPLETED"
    COMPLIANCE_PENDING = "COMPLIANCE_PENDING"
    COMPLIANCE_APPROVED = "COMPLIANCE_APPROVED"
    COMPLIANCE_REJECTED = "COMPLIANCE_REJECTED"
    TREASURY_PENDING = "TREASURY_PENDING"
    DISPERSED = "DISPERSED"
    FAILED = "FAILED"


class HandoffEnvelope(BaseModel, Generic[T]):
    """Immutable envelope encapsulating state transitions between autonomous agents.

    Enforces strict Pydantic V2 validation with extra='forbid' and frozen=True.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: UUID = Field(
        default_factory=uuid4,
        description="Global unique identifier for the execution trace."
    )
    source_agent: str = Field(description="Canonical identifier of dispatching agent.")
    target_agent: str = Field(description="Canonical identifier of receiving agent or gate.")
    current_state: AgentState = Field(description="Formal state prior to transition.")
    next_state: AgentState = Field(description="Proposed target state after transition.")
    payload: T = Field(description="Strongly-typed Pydantic model payload.")
    symbolic_hash: str = Field(
        description="SHA-256 cryptographic hash sealing state and payload."
    )

    @classmethod
    def create(
        cls,
        source_agent: str,
        target_agent: str,
        current_state: AgentState,
        next_state: AgentState,
        payload: T,
        trace_id: UUID | None = None,
        previous_hash: str = "",
    ) -> HandoffEnvelope[T]:
        """Constructs an envelope and computes its deterministic SHA-256 hash."""
        resolved_trace_id = trace_id or uuid4()
        payload_json = payload.model_dump_json()
        raw_token = f"{resolved_trace_id}:{source_agent}:{target_agent}:{current_state.value}:{next_state.value}:{payload_json}:{previous_hash}"
        symbolic_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

        return cls(
            trace_id=resolved_trace_id,
            source_agent=source_agent,
            target_agent=target_agent,
            current_state=current_state,
            next_state=next_state,
            payload=payload,
            symbolic_hash=symbolic_hash,
        )
