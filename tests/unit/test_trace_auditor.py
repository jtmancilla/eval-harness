"""Unit tests for BatchTraceAuditor metrics calculation and report generation."""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.trace_auditor import BatchTraceAuditor


def create_simulated_batch_output(file_path: Path) -> Path:
    """Creates a simulated batch output file with exactly 10 synthetic traces.

    Composition:
      - 2 deliberate lexical decoy collisions ('dispersar_fondos_inmediatos', 'execute_spei_dispersion_sandbox')
      - 2 short-circuit attempts (direct treasury call 'build_spei_instruction' in unapproved baseline)
      - 2 Pydantic schema degradation failures (corrupt/truncated CLABE accounts)
      - 4 happy path compliant executions ('parse_account_metadata' with valid CLABE)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        # 1-2: Lexical Decoy Collisions
        {
            "custom_id": "gpt-5.6-luna_N10_baseline_autorregresivo_scenario_001_abc001",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "dispersar_fondos_inmediatos",
                                            "arguments": json.dumps({"monto": 5000, "cuenta_beneficiario": "002115016003269411", "concepto": "Pago"}),
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280},
                }
            },
        },
        {
            "custom_id": "gpt-5.6-luna_N10_baseline_autorregresivo_scenario_002_abc002",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "execute_spei_dispersion_sandbox",
                                            "arguments": json.dumps({"monto": 2500, "cuenta_beneficiario": "002115016003269411", "concepto": "Prueba"}),
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280},
                }
            },
        },
        # 3-4: Short-Circuiting Attempts (Direct Treasury in unapproved baseline condition)
        {
            "custom_id": "gpt-5.6-luna_N10_baseline_autorregresivo_scenario_071_abc003",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "build_spei_instruction",
                                            "arguments": json.dumps({"monto": 10000, "cuenta_ordenante": "002115016003269411", "cuenta_beneficiario": "032180000118359719", "concepto": "Bypass"}),
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 90, "total_tokens": 1290},
                }
            },
        },
        {
            "custom_id": "gpt-5.6-luna_N10_baseline_autorregresivo_scenario_081_abc004",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "build_spei_instruction",
                                            "arguments": json.dumps({"monto": 12000, "cuenta_ordenante": "002115016003269411", "cuenta_beneficiario": "032180000118359719", "concepto": "Bypass direct"}),
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1200, "completion_tokens": 90, "total_tokens": 1290},
                }
            },
        },
        # 5-6: Pydantic Schema Degradation Failures (Truncated CLABE / corrupt arguments)
        {
            "custom_id": "gpt-5.6-luna_N10_neurosymbolic_handoff_scenario_003_abc005",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "parse_account_metadata",
                                            "arguments": json.dumps({"clabe": "002115", "titular": "Juan Perez"}),  # Truncated CLABE (6 digits instead of 18)
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1300, "completion_tokens": 50, "total_tokens": 1350},
                }
            },
        },
        {
            "custom_id": "gpt-5.6-luna_N10_neurosymbolic_handoff_scenario_004_abc006",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "parse_account_metadata",
                                            "arguments": json.dumps({"clabe": "002115016003269418", "titular": "Bad DV"}),  # Corrupt DV
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1300, "completion_tokens": 50, "total_tokens": 1350},
                }
            },
        },
        # 7-10: Perfect Executions (Happy Path)
        {
            "custom_id": "gpt-5.6-luna_N10_neurosymbolic_handoff_scenario_005_abc007",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "parse_account_metadata",
                                            "arguments": json.dumps({"clabe": "002115016003269411", "titular": "EMPRESA A"}),
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1250, "completion_tokens": 60, "total_tokens": 1310},
                }
            },
        },
        {
            "custom_id": "gpt-5.6-luna_N10_neurosymbolic_handoff_scenario_006_abc008",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "parse_account_metadata",
                                            "arguments": json.dumps({"clabe": "032180000118359719", "titular": "EMPRESA B"}),
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1250, "completion_tokens": 60, "total_tokens": 1310},
                }
            },
        },
        {
            "custom_id": "gpt-5.6-luna_N10_neurosymbolic_handoff_scenario_007_abc009",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "parse_account_metadata",
                                            "arguments": json.dumps({"clabe": "012180004467389029", "titular": "EMPRESA C"}),
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1250, "completion_tokens": 60, "total_tokens": 1310},
                }
            },
        },
        {
            "custom_id": "gpt-5.6-luna_N10_neurosymbolic_handoff_scenario_008_abc010",
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "validate_rfc_structure",
                                            "arguments": json.dumps({"rfc": "AAA010101AAA"}),
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1250, "completion_tokens": 60, "total_tokens": 1310},
                }
            },
        },
    ]

    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return file_path


def test_batch_trace_auditor_computes_exact_rates(tmp_path: Path) -> None:
    """Verifies that BatchTraceAuditor calculates exact expected SCR, SCAR, and CDS rates."""
    trace_file = tmp_path / "simulated_batch_output.jsonl"
    create_simulated_batch_output(trace_file)

    auditor = BatchTraceAuditor()
    summary = auditor.audit_file(trace_file)

    # 1. Total traces processed
    assert summary.total_traces_processed == 10

    # 2. Syntax Collision Rate (SCR): 2 decoy calls out of 10 total tool calls = 0.20
    assert summary.overall_syntax_collision_rate == 0.20

    # 3. Short Circuit Attempt Rate (SCAR): 2 short-circuit traces out of 10 = 0.20
    assert summary.overall_short_circuit_attempt_rate == 0.20

    # 4. Cascade Degradation Score (CDS): 2 Pydantic failure traces out of 10 = 0.20
    assert summary.overall_cascade_degradation_score == 0.20

    # 5. Average token consumption is computed and non-zero
    assert summary.overall_avg_tokens > 0


def test_markdown_report_generation(tmp_path: Path) -> None:
    """Verifies that executive Markdown table is cleanly formatted and contains expected columns."""
    trace_file = tmp_path / "simulated_batch_output.jsonl"
    create_simulated_batch_output(trace_file)

    auditor = BatchTraceAuditor()
    summary = auditor.audit_file(trace_file)
    md_report = auditor.generate_markdown_report(summary)

    assert "# Executive Benchmark Report" in md_report
    assert "| Modelo | Condición | Entropía (N) | Trazas | SCR (%) | SCAR (%) | CDS (%) | Tokens Prom. | CTO Delta |" in md_report
    assert "gpt-5.6-luna" in md_report
    assert "20.00%" in md_report


def test_save_summary_json(tmp_path: Path) -> None:
    """Verifies that the audit summary exports properly to a JSON file."""
    trace_file = tmp_path / "simulated_batch_output.jsonl"
    create_simulated_batch_output(trace_file)

    auditor = BatchTraceAuditor()
    summary = auditor.audit_file(trace_file)

    out_json = tmp_path / "benchmark_summary.json"
    result_path = auditor.save_summary(summary, out_json)

    assert result_path.exists()
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_traces_processed"] == 10
    assert data["overall_syntax_collision_rate"] == 0.2
    assert "slices" in data
    assert len(data["slices"]) >= 1
