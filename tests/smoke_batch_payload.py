"""Smoke test entrypoint executing N=128 live check against OpenAI API."""
from __future__ import annotations

import sys

from tests.smoke_astra_responses import run_smoke_test

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
