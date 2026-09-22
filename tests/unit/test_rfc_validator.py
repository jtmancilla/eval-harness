"""Unit tests for deterministic RFC validation gate and contracts."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts.fiscal import RFCData, TaxpayerType
from src.gates.rfc_validator import (
    RFCFormatError,
    RFCLengthError,
    is_persona_fisica,
    is_persona_moral,
    validate_rfc,
)


def test_valid_persona_moral() -> None:
    """Verifies valid 12-character Persona Moral RFC."""
    rfc = "AAA010101AAA"
    assert validate_rfc(rfc) is True
    assert is_persona_moral(rfc) is True
    assert is_persona_fisica(rfc) is False

    data = RFCData.from_rfc(rfc)
    assert data.taxpayer_type == TaxpayerType.PERSONA_MORAL


def test_valid_persona_fisica() -> None:
    """Verifies valid 13-character Persona Física RFC."""
    rfc = "GODE561231GR8"
    assert validate_rfc(rfc) is True
    assert is_persona_fisica(rfc) is True
    assert is_persona_moral(rfc) is False

    data = RFCData.from_rfc(rfc)
    assert data.taxpayer_type == TaxpayerType.PERSONA_FISICA


def test_invalid_rfc_length() -> None:
    """RFCs with length != 12 and != 13 must raise RFCLengthError."""
    with pytest.raises(RFCLengthError):
        validate_rfc("SHORT")

    with pytest.raises(RFCLengthError):
        validate_rfc("WAYTOOLONGRFC12345")


def test_invalid_rfc_syntax_or_date() -> None:
    """RFCs with malformed dates (e.g., month 99) must raise RFCFormatError."""
    with pytest.raises(RFCFormatError):
        validate_rfc("AAA999999AAA")  # Invalid month 99


def test_rfc_data_pydantic_contract_validation() -> None:
    """RFCData model must enforce valid RFC via Pydantic validator."""
    with pytest.raises(ValidationError):
        RFCData(rfc="INVALID_RFC", taxpayer_type=TaxpayerType.PERSONA_MORAL)
