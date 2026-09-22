"""Autonomous Compliance Agent for fiscal accreditation, SAT blacklist screening, and AML mitigation."""
from __future__ import annotations

from src.contracts.clabe import OnboardingHandoffPayload
from src.contracts.fiscal import (
    CFDIMetadata,
    ComplianceApprovalPayload,
    ComplianceVerdict,
    RFCData,
)
from src.contracts.handoff import AgentState, HandoffEnvelope

COMPLIANCE_SYSTEM_PROMPT = """You are the ComplianceAgent in a mission-critical Mexican financial processing architecture.
Your sole mission is to execute rigorous fiscal accreditation (RFC/CFDI) and Anti-Money Laundering (AML/PLD) screening.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. ZERO ARITHMETIC AXIOM: RFC validation and date checking are deterministic and must be performed by the symbolic gate.
2. AUTHORIZED TOOLS:
   - `validate_rfc_structure`: Syntactic homoclave and length verification.
   - `check_sat_blacklist`: Simulation of queries against SAT Article 69-B (EFOS/EDOS) and blocked persons lists.
   - `verify_cfdi_preconditions`: Matching tax regime against CFDI usage catalogs.
3. OUTPUT INTEGRITY: Emit an explicit binary verdict (APPROVED or REJECTED). You are strictly forbidden from initiating payments or calculating fees.
"""

# Mock list of RFCs flagged under SAT 69-B or sanctions list
SAT_69B_MOCK_BLACKLIST: frozenset[str] = frozenset(
    {
        "BME930101XYZ",
        "MAL850101999",
        "EDO101010AAA",
    }
)


class ComplianceAgent:
    """Agent performing AML screening, SAT blacklists lookup, and RFC accreditation."""

    def __init__(self, agent_id: str = "ComplianceAgent") -> None:
        self.agent_id = agent_id
        self.system_prompt = COMPLIANCE_SYSTEM_PROMPT

    def validate_rfc_structure(self, rfc_str: str) -> RFCData:
        """Validates RFC via the underlying deterministic gate."""
        return RFCData.from_rfc(rfc_str)

    def check_sat_blacklist(self, rfc: str) -> bool:
        """Simulates checking SAT 69-B / blocked persons lists. Returns True if clear."""
        return rfc.strip().upper() not in SAT_69B_MOCK_BLACKLIST

    def verify_cfdi_preconditions(self, regimen_fiscal: str, uso_cfdi: str, cp: str) -> CFDIMetadata:
        """Constructs and validates CFDI metadata."""
        return CFDIMetadata(
            regimen_fiscal=regimen_fiscal,
            uso_cfdi=uso_cfdi,
            codigo_postal=cp,
        )

    def process(
        self,
        onboarding_envelope: HandoffEnvelope[OnboardingHandoffPayload],
        rfc_str: str,
        regimen_fiscal: str,
        uso_cfdi: str,
        codigo_postal: str,
    ) -> HandoffEnvelope[ComplianceApprovalPayload]:
        """Evaluates compliance status and yields a sealed ComplianceApprovalPayload envelope."""
        rfc_data = self.validate_rfc_structure(rfc_str)
        is_clear = self.check_sat_blacklist(rfc_data.rfc)
        cfdi_meta = self.verify_cfdi_preconditions(regimen_fiscal, uso_cfdi, codigo_postal)

        if not is_clear:
            verdict = ComplianceVerdict.REJECTED
            rejection_reason = "RFC detected in SAT 69-B definitive blacklist (EFOS/EDOS)"
            next_state = AgentState.COMPLIANCE_REJECTED
            target_agent = "System"
            risk_score = 0.99
        else:
            verdict = ComplianceVerdict.APPROVED
            rejection_reason = None
            next_state = AgentState.COMPLIANCE_APPROVED
            target_agent = "StateGuard"
            risk_score = 0.05

        payload = ComplianceApprovalPayload(
            verdict=verdict,
            rfc_data=rfc_data,
            cfdi_metadata=cfdi_meta,
            aml_risk_score=risk_score,
            sat_blacklist_clear=is_clear,
            rejection_reason=rejection_reason,
        )

        return HandoffEnvelope.create(
            source_agent=self.agent_id,
            target_agent=target_agent,
            current_state=AgentState.COMPLIANCE_PENDING,
            next_state=next_state,
            payload=payload,
            trace_id=onboarding_envelope.trace_id,
            previous_hash=onboarding_envelope.symbolic_hash,
        )
