"""Unit tests for the Causal Stratified Dataset Compiler."""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.causal_stratified_compiler import (
    CausalStratifiedCompiler,
    MasterNarrative,
    StratifiedBatchRecord,
)


class TestCausalStratifiedCompiler:
    """Tests for master narratives caching and parametric expansion."""

    def test_dry_run_master_narratives_generation(self, tmp_path: Path) -> None:
        """Verifies dry-run generation produces well-structured MasterNarratives."""
        cache_file = tmp_path / "test_masters.json"
        compiler = CausalStratifiedCompiler(
            master_cache_path=cache_file,
            seed=42,
        )

        # Run async method via pytest asyncio
        import asyncio

        masters = asyncio.run(
            compiler.get_or_create_master_narratives(num_masters=12, dry_run=True)
        )

        assert len(masters) == 12
        templates = [m.template_name for m in masters]
        assert templates.count("confounding_fork") == 2
        assert templates.count("mediator_chain") == 2
        assert templates.count("collider") == 2

        for m in masters:
            assert isinstance(m, MasterNarrative)
            assert len(m.semantic_mapping) == len(m.nodes)
            assert "Reporte" in m.narrative_text

    def test_expand_parametric_variations_total_count(self, tmp_path: Path) -> None:
        """Verifies parametric expansion hits the exact target request count."""
        cache_file = tmp_path / "test_masters.json"
        compiler = CausalStratifiedCompiler(
            master_cache_path=cache_file,
            reasoning_efforts=("low", "medium", "high"),
            seed=42,
        )

        import asyncio

        masters = asyncio.run(
            compiler.get_or_create_master_narratives(num_masters=6, dry_run=True)
        )

        records = compiler.expand_parametric_variations(
            masters=masters,
            target_total_requests=36,
        )

        assert len(records) == 36
        efforts = [r.effort for r in records]
        assert efforts.count("low") == 12
        assert efforts.count("medium") == 12
        assert efforts.count("high") == 12

        # Check payload structure
        for r in records:
            assert isinstance(r, StratifiedBatchRecord)
            payload = r.batch_payload
            assert payload["method"] == "POST"
            assert payload["url"] == "/v1/responses"
            assert payload["body"]["model"] == "gpt-6-astra"
            assert payload["body"]["reasoning"]["effort"] in ("low", "medium", "high")
            assert "PREGUNTAS DE INFERENCIA CAUSAL" in payload["body"]["input"]

    def test_export_batch_files(self, tmp_path: Path) -> None:
        """Verifies batch and ground truth sidecar files are exported correctly."""
        compiler = CausalStratifiedCompiler(
            master_cache_path=tmp_path / "masters.json",
            seed=42,
        )

        import asyncio

        masters = asyncio.run(
            compiler.get_or_create_master_narratives(num_masters=6, dry_run=True)
        )
        records = compiler.expand_parametric_variations(masters, target_total_requests=18)

        batch_out = tmp_path / "batch.jsonl"
        gt_out = tmp_path / "gt.jsonl"

        compiler.export_batch_files(records, batch_out, gt_out)

        assert batch_out.exists()
        assert gt_out.exists()

        batch_lines = [json.loads(line) for line in batch_out.read_text(encoding="utf-8").strip().split("\n")]
        gt_lines = [json.loads(line) for line in gt_out.read_text(encoding="utf-8").strip().split("\n")]

        assert len(batch_lines) == 18
        assert len(gt_lines) == 18

        # Uniqueness of custom_ids
        custom_ids = [b["custom_id"] for b in batch_lines]
        assert len(set(custom_ids)) == 18

        # Check sidecar content
        gt_first = gt_lines[0]
        assert "L1" in gt_first["ground_truth"]
        assert "L2" in gt_first["ground_truth"]
        assert "L3" in gt_first["ground_truth"]
