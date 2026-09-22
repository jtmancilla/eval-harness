"""Deterministic symbolic gate for Mexican RFC (Registro Federal de Contribuyentes) validation.

Verifies structural rules, date consistency, and homoclave formatting for both
Persona Física (13 characters) and Persona Moral (12 characters).
"""
from __future__ import annotations

import re

# Strict regular expression patterns conforming to SAT specifications
RFC_FISICA_PATTERN = re.compile(
    r"^[A-Z&Ñ]{4}"              # 4 letters (initials)
    r"\d{2}(?:0[1-9]|1[0-2])"   # Year (2 digits) + Month (01-12)
    r"(?:0[1-9]|[12]\d|3[01])"  # Day (01-31)
    r"[A-Z0-9]{3}$"             # Homoclave (3 alphanumeric characters)
)

RFC_MORAL_PATTERN = re.compile(
    r"^[A-Z&Ñ]{3}"              # 3 letters (company abbreviation)
    r"\d{2}(?:0[1-9]|1[0-2])"   # Year (2 digits) + Month (01-12)
    r"(?:0[1-9]|[12]\d|3[01])"  # Day (01-31)
    r"[A-Z0-9]{3}$"             # Homoclave (3 alphanumeric characters)
)


class RFCValidationError(ValueError):
    """Base exception for RFC validation failures."""


class RFCLengthError(RFCValidationError):
    """Raised when RFC string length is neither 12 nor 13 characters."""


class RFCFormatError(RFCValidationError):
    """Raised when RFC does not match formal SAT syntactic or date rules."""


def is_persona_moral(rfc: str) -> bool:
    """Checks whether the RFC belongs structurally to an incorporated entity (12 characters)."""
    return len(rfc) == 12 and bool(RFC_MORAL_PATTERN.match(rfc.upper()))


def is_persona_fisica(rfc: str) -> bool:
    """Checks whether the RFC belongs structurally to an individual (13 characters)."""
    return len(rfc) == 13 and bool(RFC_FISICA_PATTERN.match(rfc.upper()))


def validate_rfc(rfc: str) -> bool:
    """Validates structural syntax and homoclave formatting of a Mexican RFC.

    Args:
        rfc: Clean RFC string (12 or 13 uppercase characters).

    Returns:
        True if the RFC complies with SAT syntactic specifications.

    Raises:
        RFCLengthError: If length is not 12 or 13.
        RFCFormatError: If the RFC violates character, date, or homoclave patterns.
    """
    normalized_rfc = rfc.strip().upper()
    length = len(normalized_rfc)

    if length not in (12, 13):
        raise RFCLengthError(
            f"RFC must be 12 (Moral) or 13 (Física) characters, received {length}"
        )

    if length == 12:
        if not RFC_MORAL_PATTERN.match(normalized_rfc):
            raise RFCFormatError(f"Invalid RFC Persona Moral syntax or date: '{normalized_rfc}'")
    else:
        if not RFC_FISICA_PATTERN.match(normalized_rfc):
            raise RFCFormatError(f"Invalid RFC Persona Física syntax or date: '{normalized_rfc}'")

    return True
