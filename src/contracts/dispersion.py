"""Typed contracts for Mexican SPEI high-value interbank payment dispersion."""
from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from src.contracts.clabe import ClabeAccount
from src.contracts.fiscal import RFCData


class SPEIFeeCalculation(BaseModel):
    """Breakdown of transaction fees and applicable value-added tax (IVA)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_fee: Decimal = Field(
        ...,
        ge=Decimal("0.00"),
        description="Base clearing fee in MXN."
    )
    iva_rate: Decimal = Field(
        default=Decimal("0.16"),
        description="Value Added Tax statutory rate (16% in Mexico)."
    )
    iva_amount: Decimal = Field(
        ...,
        ge=Decimal("0.00"),
        description="Computed IVA charge in MXN."
    )
    total_fee: Decimal = Field(
        ...,
        ge=Decimal("0.00"),
        description="Total fee payable (base + IVA) in MXN."
    )

    @classmethod
    def compute(cls, base_fee: Decimal, iva_rate: Decimal = Decimal("0.16")) -> SPEIFeeCalculation:
        """Deterministically computes total fee and IVA amounts."""
        iva_amount = (base_fee * iva_rate).quantize(Decimal("0.01"))
        total_fee = (base_fee + iva_amount).quantize(Decimal("0.01"))
        return cls(
            base_fee=base_fee,
            iva_rate=iva_rate,
            iva_amount=iva_amount,
            total_fee=total_fee,
        )


class SPEIDispersionRequest(BaseModel):
    """Immutable payload defining an irreversible SPEI interbank transfer instruction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    monto: Decimal = Field(
        ...,
        gt=Decimal("0.00"),
        description="Principal transfer amount in Mexican Pesos (MXN)."
    )
    concepto_pago: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description="SPEI payment concept description (maximum 40 alphanumeric characters)."
    )
    cuenta_ordenante: ClabeAccount = Field(
        description="Ordering entity 18-digit CLABE account."
    )
    cuenta_beneficiario: ClabeAccount = Field(
        description="Beneficiary 18-digit CLABE account."
    )
    rfc_beneficiario: RFCData = Field(
        description="Beneficiary RFC accredited by compliance."
    )
    clave_rastreo: str = Field(
        ...,
        min_length=10,
        max_length=30,
        description="Official interbank tracking key (clave de rastreo Banxico)."
    )
    referencia_numerica: int = Field(
        default=1,
        ge=1,
        le=9999999,
        description="7-digit numeric reference field."
    )
    fee_calculation: SPEIFeeCalculation | None = Field(
        default=None,
        description="Optional breakdown of SPEI processing commission."
    )


class SPEIDispersionReceipt(BaseModel):
    """Final immutable execution receipt emitted upon successful SPEI clearing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clave_rastreo: str = Field(description="Confirmed Banxico tracking key.")
    estatus_operacion: str = Field(default="LIQUIDADA", description="Settlement status.")
    timestamp_liquidacion: str = Field(description="ISO-8601 UTC clearing timestamp.")
    sello_digital_banxico: str = Field(description="Digital signature/hash acknowledging receipt.")
