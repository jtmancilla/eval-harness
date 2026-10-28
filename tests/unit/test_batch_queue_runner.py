"""Unit tests for sequential batch queue runner."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.batch_queue_runner import interleave_file_lists, run_sequential_queue


def test_interleave_file_lists() -> None:
    l1 = [Path("a1.jsonl"), Path("a2.jsonl"), Path("a3.jsonl")]
    l2 = [Path("b1.jsonl"), Path("b2.jsonl")]

    inter = interleave_file_lists(l1, l2)
    assert [p.name for p in inter] == [
        "a1.jsonl",
        "b1.jsonl",
        "a2.jsonl",
        "b2.jsonl",
        "a3.jsonl",
    ]


def test_run_sequential_queue_dry_run(tmp_path: Path) -> None:
    batch1 = tmp_path / "part01.jsonl"
    batch2 = tmp_path / "part02.jsonl"

    rec1 = {
        "custom_id": "c1",
        "method": "POST",
        "url": "/v1/responses",
        "body": {"model": "gpt-6-astra", "input": "Hello"},
    }
    rec2 = {
        "custom_id": "c2",
        "method": "POST",
        "url": "/v1/responses",
        "body": {"model": "gpt-6-astra", "input": "World"},
    }

    batch1.write_text(json.dumps(rec1) + "\n", encoding="utf-8")
    batch2.write_text(json.dumps(rec2) + "\n", encoding="utf-8")

    out_dir = tmp_path / "results"
    records = run_sequential_queue(
        batch_files=[batch1, batch2],
        results_dir=out_dir,
        poll_interval_seconds=1,
        dry_run=True,
    )

    assert len(records) == 2
    assert records[0]["status"] == "completed"
    assert records[1]["status"] == "completed"

    # Running again should see them as already completed
    records_resumed = run_sequential_queue(
        batch_files=[batch1, batch2],
        results_dir=out_dir,
        poll_interval_seconds=1,
        dry_run=True,
    )
    assert len(records_resumed) == 2
    assert records_resumed[0]["status"] == "already_completed"
    assert records_resumed[1]["status"] == "already_completed"
