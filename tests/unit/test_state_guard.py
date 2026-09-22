"""Unit tests for StateGuard symbolic gate and short-circuit prevention."""
from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel

from src.contracts.handoff import AgentState, HandoffEnvelope
from src.gates.state_guard import (
    CryptographicIntegrityViolation,
    ShortCircuitViolation,
    StateGuard,
)


class DummyPayload(BaseModel):
    sample: str


def test_valid_step_transition() -> None:
    """Standard sequential transition must be allowed by StateGuard."""
    envelope = HandoffEnvelope.create(
        source_agent="System",
        target_agent="OnboardingAgent",
        current_state=AgentState.INITIALIZED,
        next_state=AgentState.ONBOARDING_PENDING,
        payload=DummyPayload(sample="test"),
    )
    assert StateGuard.verify_transition(envelope) is True


def test_short_circuit_direct_jump_forbidden() -> None:
    """Direct jump from ONBOARDING_PENDING to TREASURY_PENDING must trigger ShortCircuitViolation."""
    envelope = HandoffEnvelope.create(
        source_agent="AdversarialAgent",
        target_agent="TreasuryAgent",
        current_state=AgentState.ONBOARDING_PENDING,
        next_state=AgentState.TREASURY_PENDING,
        payload=DummyPayload(sample="bypass_attempt"),
    )
    with pytest.raises(ShortCircuitViolation, match="Unauthorized transition attempt in DAG"):
        StateGuard.verify_transition(envelope)


def test_compliance_bypass_history_detection() -> None:
    """Transition to TREASURY_PENDING without COMPLIANCE_APPROVED in verified history must fail."""
    envelope = HandoffEnvelope.create(
        source_agent="StateGuard",
        target_agent="TreasuryAgent",
        current_state=AgentState.COMPLIANCE_APPROVED,
        next_state=AgentState.TREASURY_PENDING,
        payload=DummyPayload(sample="treasury_dispatch"),
    )
    # History lacks COMPLIANCE_APPROVED
    tampered_history = [AgentState.INITIALIZED, AgentState.ONBOARDING_PENDING]
    with pytest.raises(ShortCircuitViolation, match="Regulatory Bypass detected"):
        StateGuard.verify_transition(envelope, verified_history=tampered_history)


def test_missing_hash_signature() -> None:
    """Envelope missing a valid SHA-256 hash must fail cryptographic verification."""
    envelope = HandoffEnvelope(
        trace_id=uuid4(),
        source_agent="System",
        target_agent="OnboardingAgent",
        current_state=AgentState.INITIALIZED,
        next_state=AgentState.ONBOARDING_PENDING,
        payload=DummyPayload(sample="test"),
        symbolic_hash="invalid_short_hash",
    )
    with pytest.raises(CryptographicIntegrityViolation):
        StateGuard.verify_transition(envelope)
