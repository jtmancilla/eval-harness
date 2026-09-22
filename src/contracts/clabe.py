"""Typed contracts for Mexican bank account CLABE ingestion and onboarding."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.gates.modulo10 import validate_clabe


class ClabeAccount(BaseModel):
    """Immutable model representing an 18-digit Mexican bank CLABE.

    Integrates deterministic Modulo 10 verification at instantiation time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    clabe: str = Field(
        ...,
        min_length=18,
        max_length=18,
        description="18-digit Mexican CLABE string satisfying official ABM Modulo 10 check."
    )

    @field_validator("clabe", mode="after")
    @classmethod
    def enforce_modulo10_integrity(cls, value: str) -> str:
        """Validates the CLABE control digit deterministically via the symbolic gate."""
        validate_clabe(value)
        return value

    @property
    def bank_code(self) -> str:
        """Extracts 3-digit ABM bank code."""
        return self.clabe[:3]

    @property
    def branch_code(self) -> str:
        """Extracts 3-digit regional branch/plaza code."""
        return self.clabe[3:6]

    @property
    def account_number(self) -> str:
        """Extracts 11-digit customer account number."""
        return self.clabe[6:17]

    @property
    def control_digit(self) -> str:
        """Extracts single control digit."""
        return self.clabe[17]


class RawAccountIngestionPayload(BaseModel):
    """Raw, semi-structured ingestion payload received by OnboardingAgent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_text: str = Field(description="Unprocessed text or document snippet from client/webhook.")
    source_channel: str = Field(default="api", description="Channel of origin (e.g., webhook, portal).")
    declared_bank: str | None = Field(default=None, description="Optional bank name hint.")


class OnboardingHandoffPayload(BaseModel):
    """Normalized payload dispatched by OnboardingAgent to Compliance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    account: ClabeAccount = Field(description="Validated 18-digit CLABE account.")
    account_holder_name: str = Field(
        min_length=2,
        max_length=120,
        description="Normalized legal name of the account holder."
    )
    institution_name: str = Field(description="Resolved credit institution name from ABM catalog.")
