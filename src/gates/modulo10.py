"""Deterministic symbolic gate for CLABE Modulo 10 check digit verification.

Conforms to official Mexican banking standards (ABM / Banco de México).
Calculates check digit using the cyclic weight sequence (3, 7, 1) across the first 17 digits.
"""
from __future__ import annotations

# Official ABM / Banco de México cyclic weighting vector
CLABE_WEIGHTS: tuple[int, ...] = (3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7, 1, 3, 7)


class ClabeValidationError(ValueError):
    """Base exception for all CLABE validation failures."""


class ClabeLengthError(ClabeValidationError):
    """Raised when CLABE string length differs from the required specification."""


class ClabeNonNumericError(ClabeValidationError):
    """Raised when CLABE contains non-numeric characters."""


class ClabeChecksumMismatchError(ClabeValidationError):
    """Raised when control digit does not match Modulo 10 computed check digit."""


def compute_clabe_control_digit(clabe_17: str) -> int:
    """Computes the control digit for the first 17 characters of a Mexican CLABE.

    Args:
        clabe_17: 17-digit numeric string representing bank code, branch, and account number.

    Returns:
        The computed control digit (0-9).

    Raises:
        ClabeLengthError: If clabe_17 length is not exactly 17 digits.
        ClabeNonNumericError: If clabe_17 contains any non-digit character.
    """
    if len(clabe_17) != 17:
        raise ClabeLengthError(
            f"Expected exactly 17 digits for base calculation, received {len(clabe_17)}"
        )
    if not clabe_17.isdigit():
        raise ClabeNonNumericError("Base CLABE sequence must contain only numeric digits")

    total_sum = sum(
        (int(digit) * weight) % 10
        for digit, weight in zip(clabe_17, CLABE_WEIGHTS, strict=True)
    )
    return (10 - (total_sum % 10)) % 10


def validate_clabe(clabe: str) -> bool:
    """Validates the structural and mathematical integrity of an 18-digit CLABE.

    Args:
        clabe: Complete 18-digit CLABE string.

    Returns:
        True if the CLABE is structurally sound and passes Modulo 10 verification.

    Raises:
        ClabeLengthError: If length != 18.
        ClabeNonNumericError: If non-digit characters are found.
        ClabeChecksumMismatchError: If the 18th digit does not match the computed control digit.
    """
    if len(clabe) != 18:
        raise ClabeLengthError(
            f"CLABE must be exactly 18 digits long, received {len(clabe)}"
        )
    if not clabe.isdigit():
        raise ClabeNonNumericError("CLABE must contain exclusively numeric digits")

    expected_control = int(clabe[17])
    calculated_control = compute_clabe_control_digit(clabe[:17])

    if expected_control != calculated_control:
        raise ClabeChecksumMismatchError(
            f"Invalid CLABE control digit: provided={expected_control}, expected={calculated_control}"
        )

    return True
