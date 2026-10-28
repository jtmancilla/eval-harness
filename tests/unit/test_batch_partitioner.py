"""Unit tests for batch partitioner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.batch_partitioner import (
    calculate_tokens_for_record,
    partition_batch,
)


def test_calculate_tokens_for_record() -> None:
    record = {
        "custom_id": "test_01",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-6-astra",
            "instructions": "Hello " * 10,
            "input": "World " * 10,
        },
    }
    tok = calculate_tokens_for_record(record)
    assert tok > 0


def test_partition_batch_creates_chunks(tmp_path: Path) -> None:
    batch_file = tmp_path / "test_batch.jsonl"
    gt_file = tmp_path / "test_gt.jsonl"

    records = []
    gt_records = []
    for i in range(10):
        rec = {
            "custom_id": f"rec_{i}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-6-astra",
                "instructions": "Instruction text",
                "input": f"Sample input data {i}",
            },
        }
        records.append(json.dumps(rec))
        gt_records.append(json.dumps({"custom_id": f"rec_{i}", "val": i}))

    batch_file.write_text("\n".join(records) + "\n", encoding="utf-8")
    gt_file.write_text("\n".join(gt_records) + "\n", encoding="utf-8")

    # Partition with chunk_size = 4
    parts = partition_batch(
        batch_file=batch_file,
        ground_truth_file=gt_file,
        output_dir=tmp_path,
        chunk_size=4,
        max_tokens_per_chunk=10000,
    )

    assert len(parts) == 3
    assert parts[0]["requests_count"] == 4
    assert parts[1]["requests_count"] == 4
    assert parts[2]["requests_count"] == 2

    # Check files exist
    assert Path(str(parts[0]["batch_file"])).is_file()
    assert Path(str(parts[0]["ground_truth_file"])).is_file()


def test_partition_batch_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        partition_batch("non_existent_file.jsonl")
