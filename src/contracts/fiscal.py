"""Typed contracts for tax accreditation, RFC validation, and CFDI metadata."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.gates.rfc_validator import is_persona_moral, validate_rfc


class TaxpayerType(StrEnum):
    """SAT taxpayer classification."""

    PERSONA_FISICA = "PERSONA_FISICA"
    PERSONA_MORAL = "PERSONA_MORAL"


class ComplianceVerdict(StrEnum):
    """Binary decision emitted by ComplianceAgent."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class RFCData(BaseModel):
    """Immutable model representing a validated Mexican RFC."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rfc: str = Field(
        ...,
        min_length=12,
        max_length=13,
        description="SAT RFC string for Persona Física (13) or Moral (12)."
    )
    taxpayer_type: TaxpayerType = Field(
        description="Taxpayer classification inferred or verified."
    )

    @field_validator("rfc", mode="after")
    @classmethod
    def enforce_rfc_syntax(cls, value: str) -> str:
        """Validates RFC format and homoclave deterministically via the symbolic gate."""
        normalized = value.strip().upper()
        validate_rfc(normalized)
        return normalized

    @classmethod
    def from_rfc(cls, rfc_str: str) -> RFCData:
        """Convenience constructor inferring taxpayer type from length and pattern."""
        normalized = rfc_str.strip().upper()
        validate_rfc(normalized)
        taxpayer_type = (
            TaxpayerType.PERSONA_MORAL if is_persona_moral(normalized) else TaxpayerType.PERSONA_FISICA
        )
        return cls(rfc=normalized, taxpayer_type=taxpayer_type)


class CFDIMetadata(BaseModel):
    """Fiscal metadata required for CFDI 4.0 invoicing and tax compliance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    regimen_fiscal: str = Field(
        pattern=r"^\d{3}$",
        description="3-digit SAT tax regime code (e.g., 601, 605, 626)."
    )
    uso_cfdi: str = Field(
        pattern=r"^[A-Z0-9]{3,4}$",
        description="CFDI usage catalog key (e.g., G03, CP01, S01)."
    )
    codigo_postal: str = Field(
        min_length=5,
        max_length=5,
        pattern=r"^\d{5}$",
        description="5-digit fiscal postal code of receiver."
    )


class ComplianceApprovalPayload(BaseModel):
    """Dictamen payload emitted by ComplianceAgent and sealed by StateGuard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: ComplianceVerdict = Field(description="Binary compliance determination.")
    rfc_data: RFCData = Field(description="Validated RFC details.")
    cfdi_metadata: CFDIMetadata = Field(description="Tax invoicing preconditions.")
    aml_risk_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Anti-Money Laundering calculated risk score (0.0=lowest, 1.0=critical)."
    )
    sat_blacklist_clear: bool = Field(
        description="True if not present in SAT Article 69-B or designated blocked persons list."
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Formal justification if verdict is REJECTED."
    )
