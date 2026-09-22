"""Autonomous Onboarding Agent for raw account ingestion and structured normalization."""
from __future__ import annotations

import re
from uuid import UUID

from src.contracts.clabe import ClabeAccount, OnboardingHandoffPayload, RawAccountIngestionPayload
from src.contracts.handoff import AgentState, HandoffEnvelope

ONBOARDING_SYSTEM_PROMPT = """You are the OnboardingAgent in a mission-critical Mexican financial processing architecture.
Your sole mission is to parse, normalize, and extract bank account metadata from unstructured text or payloads.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. ZERO ARITHMETIC AXIOM: NEVER attempt to calculate or verify checksums, Modulo 10, or mathematical weights in prompts or thought chains.
2. TOOL BOUNDARIES: You are strictly authorized to use ONLY:
   - `parse_account_metadata`: Clean text, extract strings, normalize names.
   - `query_abm_directory`: Resolve 3-digit bank codes against official ABM directory.
3. OUTPUT INTEGRITY: Output must strictly conform to OnboardingHandoffPayload. You do NOT perform compliance checks or fee calculations.
"""

# Mock ABM directory mapping 3-digit bank codes to official entity names
ABM_DIRECTORY: dict[str, str] = {
    "002": "Banco Nacional de México, S.A. (Banamex)",
    "012": "BBVA México, S.A.",
    "014": "Banco Santander México, S.A.",
    "021": "HSBC México, S.A.",
    "030": "Banco del Bajío, S.A.",
    "032": "IXE Banco, S.A.",
    "036": "Banco Inbursa, S.A.",
    "044": "Scotiabank Inverlat, S.A.",
    "058": "Banco Regional de Monterrey, S.A. (Banregio)",
    "072": "Banco Mercantil del Norte, S.A. (Banorte)",
}


class OnboardingAgent:
    """Agent responsible for account extraction, name normalization, and institution lookup."""

    def __init__(self, agent_id: str = "OnboardingAgent") -> None:
        self.agent_id = agent_id
        self.system_prompt = ONBOARDING_SYSTEM_PROMPT

    def parse_account_metadata(self, raw_text: str) -> dict[str, str]:
        """Extracts candidate CLABE digits and account holder names via deterministic regex."""
        clabe_match = re.search(r"\b(\d{18})\b", raw_text)
        clabe = clabe_match.group(1) if clabe_match else ""

        # Basic heuristic for extracting legal/account holder name
        name_match = re.search(r"(?:a\s+favor\s+de|titular|beneficiario|nombre):\s*([A-Za-zÁÉÍÓÚáéíóúÑñ\s]+)", raw_text, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else "TITULAR NO ESPECIFICADO"

        return {"clabe": clabe, "holder_name": name.upper()}

    def query_abm_directory(self, bank_code: str) -> str:
        """Looks up the official institution name from the 3-digit ABM bank code."""
        return ABM_DIRECTORY.get(bank_code, f"Institución Bancaria Desconocida ({bank_code})")

    def process(
        self,
        payload: RawAccountIngestionPayload,
        trace_id: UUID,
    ) -> HandoffEnvelope[OnboardingHandoffPayload]:
        """Processes raw text ingestion and packages into an OnboardingHandoffPayload envelope.

        Raises:
            ClabeValidationError: If extracted CLABE is invalid according to Modulo 10.
        """
        metadata = self.parse_account_metadata(payload.raw_text)
        account = ClabeAccount(clabe=metadata["clabe"])
        bank_name = self.query_abm_directory(account.bank_code)

        handoff_payload = OnboardingHandoffPayload(
            account=account,
            account_holder_name=metadata["holder_name"],
            institution_name=bank_name,
        )

        return HandoffEnvelope.create(
            source_agent=self.agent_id,
            target_agent="StateGuard",
            current_state=AgentState.ONBOARDING_PENDING,
            next_state=AgentState.ONBOARDING_COMPLETED,
            payload=handoff_payload,
            trace_id=trace_id,
        )
