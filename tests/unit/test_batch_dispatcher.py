"""Unit tests for OpenAIBatchDispatcher budget validation and dry-run execution."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.eval.batch_dispatcher import OpenAIBatchDispatcher, download_batch_output


def test_validate_budget_limit() -> None:
    """Verifies that token cost calculations accurately enforce the 300 USD ceiling."""
    dispatcher = OpenAIBatchDispatcher(dry_run=True)

    # ~100,000 tokens should comfortably be within budget
    assert dispatcher.validate_budget_limit(estimated_tokens=100000, max_usd=300.0) is True

    # 1,000,000,000 tokens should far exceed $300 USD
    assert dispatcher.validate_budget_limit(estimated_tokens=1000000000, max_usd=300.0) is False


def test_dry_run_batch_submission(tmp_path: Path) -> None:
    """Verifies that dry-run mode simulates batch submission without network calls."""
    dummy_jsonl = tmp_path / "dummy_requests.jsonl"
    dummy_jsonl.write_text('{"custom_id": "test_001"}\n', encoding="utf-8")

    dispatcher = OpenAIBatchDispatcher(dry_run=True)
    result = dispatcher.submit_batch(dummy_jsonl)

    assert result.is_dry_run is True
    assert result.batch_id.startswith("batch_dryrun_")
    assert result.status == "validating"


def test_dry_run_check_status() -> None:
    """Verifies that check_status returns completed mock metadata in dry-run mode."""
    dispatcher = OpenAIBatchDispatcher(dry_run=True)
    status_info = dispatcher.check_status("batch_dryrun_12345")

    assert status_info["status"] == "completed"
    assert status_info["request_counts"]["total"] == 1200
    assert status_info["is_dry_run"] is True


def test_dry_run_download_results(tmp_path: Path) -> None:
    """Verifies that download_results writes simulated outputs in dry-run mode."""
    dispatcher = OpenAIBatchDispatcher(dry_run=True)
    out_file = tmp_path / "downloaded_results.jsonl"

    res_path = dispatcher.download_results("batch_dryrun_12345", out_file)
    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")
    assert "dry_run" in content


def test_submit_batch_missing_file_raises_error() -> None:
    """Missing jsonl path must raise FileNotFoundError."""
    dispatcher = OpenAIBatchDispatcher(dry_run=True)
    with pytest.raises(FileNotFoundError):
        dispatcher.submit_batch(Path("/non/existent/path.jsonl"))


def test_inspect_and_validate_file_success(tmp_path: Path) -> None:
    """Verifies that line-by-line inspection correctly validates formal OpenAI Batch structure."""
    valid_jsonl = tmp_path / "valid_batch.jsonl"
    line_data = {
        "custom_id": "test_request_001",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [],
        },
    }
    valid_jsonl.write_text(json.dumps(line_data) + "\n", encoding="utf-8")

    dispatcher = OpenAIBatchDispatcher(dry_run=True)
    summary = dispatcher.inspect_and_validate_file(valid_jsonl, max_usd=300.0)

    assert summary["valid"] is True
    assert summary["total_requests"] == 1
    assert summary["unique_custom_ids"] == 1
    assert summary["is_within_budget"] is True


def test_inspect_and_validate_file_duplicate_custom_id(tmp_path: Path) -> None:
    """Duplicate custom_id must be detected and rejected immediately."""
    dup_jsonl = tmp_path / "dup_batch.jsonl"
    line1 = {
        "custom_id": "duplicate_id",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "1"}], "tools": []},
    }
    line2 = {
        "custom_id": "duplicate_id",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "2"}], "tools": []},
    }
    dup_jsonl.write_text(json.dumps(line1) + "\n" + json.dumps(line2) + "\n", encoding="utf-8")

    dispatcher = OpenAIBatchDispatcher(dry_run=True)
    with pytest.raises(ValueError, match="duplicado detectado"):
        dispatcher.inspect_and_validate_file(dup_jsonl)


def test_cli_dry_run_invocation(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verifies main() CLI execution with --dry-run argument."""
    from src.eval.batch_dispatcher import main

    test_file = tmp_path / "cli_test.jsonl"
    test_req = {
        "custom_id": "cli_req_001",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "test"}], "tools": []},
    }
    test_file.write_text(json.dumps(test_req) + "\n", encoding="utf-8")

    main(["--dry-run", str(test_file)])
    captured = capsys.readouterr()
    assert "OPENAI BATCH API — REPORTE DE INSPECCIÓN (--dry-run)" in captured.out
    assert "Solicitudes válidas:" in captured.out


def test_download_batch_output_not_completed(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that download_batch_output prints a warning and returns None when not completed."""
    mock_client = MagicMock()
    mock_batch = MagicMock()
    mock_batch.status = "in_progress"
    mock_batch.request_counts = MagicMock(total=300, completed=150, failed=0)
    mock_client.batches.retrieve.return_value = mock_batch

    result = download_batch_output(mock_client, "batch_test_123")
    assert result is None

    captured = capsys.readouterr()
    assert "[ADVERTENCIA]: El lote 'batch_test_123' no está en estado 'completed'." in captured.out
    assert "in_progress" in captured.out


def test_download_batch_output_completed_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies that download_batch_output saves file and prints line count when completed."""
    mock_client = MagicMock()
    mock_batch = MagicMock()
    mock_batch.status = "completed"
    mock_batch.output_file_id = "file_out_123"
    mock_client.batches.retrieve.return_value = mock_batch

    mock_file_content = MagicMock()
    mock_file_content.text = '{"line": 1}\n{"line": 2}\n\n{"line": 3}\n'
    mock_client.files.content.return_value = mock_file_content

    dest_file = tmp_path / "subfolder" / "batch_out.jsonl"
    result = download_batch_output(mock_client, "batch_test_456", output_path=str(dest_file))

    assert result == dest_file
    assert dest_file.exists()
    assert dest_file.read_text(encoding="utf-8") == mock_file_content.text

    captured = capsys.readouterr()
    assert "OPENAI BATCH API — DESCARGA EXITOSA" in captured.out
    assert "3" in captured.out  # 3 non-empty lines

