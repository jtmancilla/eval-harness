"""Batch partitioner for OpenAI Batch API rate-limit management.

Splits large batch JSONL files and their corresponding ground truth sidecars
into safe chunks that respect OpenAI Tier 1/2 enqueued token limits.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("batch_partitioner")


def calculate_tokens_for_record(record: Mapping[str, object]) -> int:
    """Estimates the input token count of a batch request record."""
    body = record.get("body")
    if not isinstance(body, dict):
        return 0

    instructions = str(body.get("instructions", ""))
    input_text = str(body.get("input", ""))
    messages = body.get("messages")
    tools = body.get("tools")

    raw_text = instructions + input_text
    if isinstance(messages, list):
        raw_text += json.dumps(messages)
    if isinstance(tools, list):
        raw_text += json.dumps(tools)

    # 4 characters per token heuristic
    return max(1, len(raw_text) // 4)


def partition_batch(
    batch_file: Path | str,
    ground_truth_file: Path | str | None = None,
    output_dir: Path | str | None = None,
    chunk_size: int = 495,
    max_tokens_per_chunk: int = 750_000,
) -> list[dict[str, object]]:
    """Partitions a batch file (and optional ground truth) into smaller sub-batches.

    Args:
        batch_file: Path to the batch JSONL file.
        ground_truth_file: Optional path to matching ground truth JSONL file.
        output_dir: Destination directory for partitioned files.
        chunk_size: Number of requests per chunk.
        max_tokens_per_chunk: Maximum cumulative estimated tokens per chunk.

    Returns:
        List of metadata dicts describing each created partition.
    """
    batch_path = Path(batch_file)
    if not batch_path.is_file():
        raise FileNotFoundError(f"Archivo de lote no encontrado: {batch_path}")

    gt_lines: list[str] | None = None
    if ground_truth_file:
        gt_path = Path(ground_truth_file)
        if not gt_path.is_file():
            raise FileNotFoundError(f"Archivo de ground truth no encontrado: {gt_path}")
        with open(gt_path, encoding="utf-8") as fg:
            gt_lines = [line.strip() for line in fg if line.strip()]

    with open(batch_path, encoding="utf-8") as fb:
        batch_lines = [line.strip() for line in fb if line.strip()]

    total_records = len(batch_lines)
    if gt_lines is not None and len(gt_lines) != total_records:
        raise ValueError(
            f"Discrepancia en conteo de líneas: batch tiene {total_records}, ground truth tiene {len(gt_lines)}"
        )

    out_batch_dir = Path(output_dir) if output_dir else batch_path.parent
    out_batch_dir.mkdir(parents=True, exist_ok=True)

    partitions: list[dict[str, object]] = []
    current_batch_chunk: list[str] = []
    current_gt_chunk: list[str] = []
    current_tokens = 0
    part_idx = 1

    stem = batch_path.stem
    gt_stem = Path(ground_truth_file).stem if ground_truth_file else None
    gt_parent = Path(ground_truth_file).parent if ground_truth_file else None

    def flush_chunk() -> None:
        nonlocal part_idx, current_batch_chunk, current_gt_chunk, current_tokens
        if not current_batch_chunk:
            return

        part_num = f"{part_idx:02d}"
        part_batch_name = f"{stem}_part{part_num}.jsonl"
        part_batch_path = out_batch_dir / part_batch_name

        with open(part_batch_path, "w", encoding="utf-8") as fout:
            for item in current_batch_chunk:
                fout.write(item + "\n")

        part_gt_path: Path | None = None
        if gt_stem and gt_parent and current_gt_chunk:
            part_gt_name = f"{gt_stem}_part{part_num}.jsonl"
            part_gt_path = gt_parent / part_gt_name
            with open(part_gt_path, "w", encoding="utf-8") as fgout:
                for item in current_gt_chunk:
                    fgout.write(item + "\n")

        info: dict[str, object] = {
            "part_index": part_idx,
            "requests_count": len(current_batch_chunk),
            "estimated_tokens": current_tokens,
            "batch_file": str(part_batch_path),
            "ground_truth_file": str(part_gt_path) if part_gt_path else None,
        }
        partitions.append(info)
        logger.info(
            "Partición %s creada: %d peticiones (~%d tokens) -> %s",
            part_num,
            len(current_batch_chunk),
            current_tokens,
            part_batch_path.name,
        )

        part_idx += 1
        current_batch_chunk = []
        current_gt_chunk = []
        current_tokens = 0

    for idx, b_line in enumerate(batch_lines):
        rec = json.loads(b_line)
        tok = calculate_tokens_for_record(rec)

        # Check if adding this record would exceed chunk_size or token limit
        if (len(current_batch_chunk) >= chunk_size) or (
            current_batch_chunk and (current_tokens + tok > max_tokens_per_chunk)
        ):
            flush_chunk()

        current_batch_chunk.append(b_line)
        if gt_lines is not None:
            current_gt_chunk.append(gt_lines[idx])
        current_tokens += tok

    flush_chunk()
    return partitions


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Particiona archivos JSONL de OpenAI Batch para respetar límites de cola Tier 1/2."
    )
    parser.add_argument("batch_file", type=str, help="Ruta al archivo batch .jsonl a dividir")
    parser.add_argument("--gt", type=str, default=None, help="Ruta al archivo ground truth .jsonl correspondiente")
    parser.add_argument("--chunk-size", type=int, default=495, help="Peticiones por partición (default: 495)")
    parser.add_argument("--max-tokens", type=int, default=750_000, help="Tokens máximos por partición (default: 750,000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directorio destino para las particiones")

    args = parser.parse_args(argv)

    parts = partition_batch(
        batch_file=args.batch_file,
        ground_truth_file=args.gt,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        max_tokens_per_chunk=args.max_tokens,
    )

    print("\n" + "=" * 64)
    print(f" RESUMEN DE PARTICIONES CREADAS ({len(parts)} particiones)")
    print("=" * 64)
    for p in parts:
        print(f"• Parte {p['part_index']:02d}: {p['requests_count']} reqs | ~{p['estimated_tokens']:,} tokens | {Path(str(p['batch_file'])).name}")
    print("=" * 64)


if __name__ == "__main__":
    main()
