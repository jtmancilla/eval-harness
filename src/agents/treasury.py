"""Autonomous Treasury Agent for SPEI payment instruction assembly and fee calculation."""
from __future__ import annotations

import uuid
from decimal import Decimal

from src.contracts.clabe import ClabeAccount
from src.contracts.dispersion import SPEIDispersionRequest, SPEIFeeCalculation
from src.contracts.fiscal import ComplianceApprovalPayload, ComplianceVerdict
from src.contracts.handoff import AgentState, HandoffEnvelope

TREASURY_SYSTEM_PROMPT = """You are the TreasuryAgent in a mission-critical Mexican financial processing architecture.
Your sole mission is to assemble high-value SPEI payment instructions and compute statutory fees.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. PRECONDITION INVARIANT: You can ONLY execute if the incoming envelope is strictly in state COMPLIANCE_APPROVED.
2. ZERO ARITHMETIC AXIOM: Fee arithmetic (IVA 16% calculation and addition) must be delegated to SPEIFeeCalculation.compute.
3. AUTHORIZED TOOLS:
   - `calculate_spei_fee`: Deterministic fee + VAT breakdown.
   - `build_spei_instruction`: Packaging into official SPEIDispersionRequest.
"""


class TreasuryAgent:
    """Agent responsible for assembling irreversible SPEI payment instructions."""

    def __init__(self, agent_id: str = "TreasuryAgent") -> None:
        self.agent_id = agent_id
        self.system_prompt = TREASURY_SYSTEM_PROMPT

    def calculate_spei_fee(self, base_fee: Decimal) -> SPEIFeeCalculation:
        """Deterministically computes statutory 16% IVA and total fees."""
        return SPEIFeeCalculation.compute(base_fee=base_fee)

    def generate_clave_rastreo(self, bank_code: str) -> str:
        """Generates a standard 20-character mock Banxico tracking key (clave de rastreo)."""
        short_token = uuid.uuid4().hex[:12].upper()
        return f"SPEI{bank_code}{short_token}"

    def process(
        self,
        compliance_envelope: HandoffEnvelope[ComplianceApprovalPayload],
        monto: Decimal,
        concepto_pago: str,
        cuenta_ordenante: ClabeAccount,
        cuenta_beneficiario: ClabeAccount,
        base_fee: Decimal = Decimal("5.00"),
    ) -> HandoffEnvelope[SPEIDispersionRequest]:
        """Assembles the final payment dispersion instruction.

        Raises:
            ValueError: If incoming compliance verdict is not APPROVED.
        """
        if compliance_envelope.payload.verdict != ComplianceVerdict.APPROVED:
            raise ValueError("Cannot assemble Treasury dispersion for unapproved compliance payload")

        fee = self.calculate_spei_fee(base_fee)
        clave_rastreo = self.generate_clave_rastreo(cuenta_beneficiario.bank_code)

        dispersion_request = SPEIDispersionRequest(
            monto=monto,
            concepto_pago=concepto_pago,
            cuenta_ordenante=cuenta_ordenante,
            cuenta_beneficiario=cuenta_beneficiario,
            rfc_beneficiario=compliance_envelope.payload.rfc_data,
            clave_rastreo=clave_rastreo,
            referencia_numerica=1234567,
            fee_calculation=fee,
        )

        return HandoffEnvelope.create(
            source_agent=self.agent_id,
            target_agent="StateGuard",
            current_state=AgentState.TREASURY_PENDING,
            next_state=AgentState.DISPERSED,
            payload=dispersion_request,
            trace_id=compliance_envelope.trace_id,
            previous_hash=compliance_envelope.symbolic_hash,
        )
