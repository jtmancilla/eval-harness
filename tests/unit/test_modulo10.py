"""Unit tests for deterministic CLABE Modulo 10 verification gate and contract."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts.clabe import ClabeAccount
from src.gates.modulo10 import (
    ClabeChecksumMismatchError,
    ClabeLengthError,
    ClabeNonNumericError,
    compute_clabe_control_digit,
    validate_clabe,
)


@pytest.mark.parametrize(
    ("clabe_18", "expected_control"),
    [
        ("002115016003269411", 1),  # Banamex (Official Banamex example)
        ("032180000118359719", 9),  # IXE (Standard ABM documentation example)
        ("012180004467389029", 9),  # BBVA
    ],
)
def test_valid_clabes_pass(clabe_18: str, expected_control: int) -> None:
    """Verifies that official valid Mexican CLABEs compute expected control digits."""
    assert compute_clabe_control_digit(clabe_18[:17]) == expected_control
    assert validate_clabe(clabe_18) is True

    # Pydantic contract instantiation test
    account = ClabeAccount(clabe=clabe_18)
    assert account.clabe == clabe_18
    assert account.control_digit == str(expected_control)


def test_invalid_checksum_raises_error() -> None:
    """Tampered control digit must raise ClabeChecksumMismatchError."""
    with pytest.raises(ClabeChecksumMismatchError, match="Invalid CLABE control digit"):
        validate_clabe("002115016003269418")  # Expected 2, given 8


@pytest.mark.parametrize("invalid_len", ["01218000446738902", "0121800044673890259"])
def test_invalid_length_raises_error(invalid_len: str) -> None:
    """CLABEs with != 18 digits must raise ClabeLengthError."""
    with pytest.raises(ClabeLengthError):
        validate_clabe(invalid_len)


def test_non_numeric_characters_raise_error() -> None:
    """CLABEs containing alphabetical or special characters must raise ClabeNonNumericError."""
    with pytest.raises(ClabeNonNumericError):
        validate_clabe("01218000446738902X")


def test_clabe_account_contract_rejection() -> None:
    """ClabeAccount Pydantic model must reject malformed CLABEs."""
    with pytest.raises(ValidationError):
        ClabeAccount(clabe="002115016003269418")
