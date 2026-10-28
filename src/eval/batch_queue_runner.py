"""Sequential batch queue runner for OpenAI Tier 1 rate-limited workloads.

Submits partitioned batch files one by one, monitors each job until completion,
downloads the output, and automatically advances to the next partition.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.eval.batch_dispatcher import OpenAIBatchDispatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_queue_runner")


def interleave_file_lists(*file_lists: list[Path]) -> list[Path]:
    """Interleaves multiple lists of Paths in a round-robin sequence."""
    interleaved: list[Path] = []
    max_len = max((len(lst) for lst in file_lists), default=0)
    for i in range(max_len):
        for lst in file_lists:
            if i < len(lst):
                interleaved.append(lst[i])
    return interleaved


def run_sequential_queue(
    batch_files: list[Path],
    results_dir: Path | str = "results",
    poll_interval_seconds: int = 30,
    api_key: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Executes a list of batch files sequentially, respecting OpenAI queue limits.

    Args:
        batch_files: Sorted list of batch JSONL files to process.
        results_dir: Directory where downloaded outputs will be saved.
        poll_interval_seconds: Seconds between polling checks.
        api_key: OpenAI API key.
        dry_run: If True, simulates dispatch and completion.

    Returns:
        List of summary dicts for each completed partition.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dispatcher = OpenAIBatchDispatcher(api_key=api_key, dry_run=dry_run)
    run_records: list[dict[str, Any]] = []

    total_files = len(batch_files)
    logger.info("Iniciando ejecución secuencial de cola para %d archivo(s)", total_files)

    for idx, batch_file in enumerate(batch_files, start=1):
        output_file = out_dir / f"output_{batch_file.name}"

        # 1. Idempotency / Resume check
        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info(
                "[%d/%d] Salida previa encontrada para %s. Omitiendo...",
                idx,
                total_files,
                batch_file.name,
            )
            run_records.append({
                "batch_file": str(batch_file),
                "output_file": str(output_file),
                "status": "already_completed",
            })
            continue

        # 2. Pre-flight validation
        logger.info("[%d/%d] Validando archivo de lote: %s", idx, total_files, batch_file.name)
        summary = dispatcher.inspect_and_validate_file(batch_file, max_usd=300.0)
        if summary.get("exceeds_queue_limit") and not dry_run:
            logger.error(
                "El archivo %s excede el límite de cola Tier 1 (%d tokens). Abortando.",
                batch_file.name,
                summary.get("queue_limit_tokens", 900_000),
            )
            break

        # Check for active existing batch for this file in OpenAI to resume without re-submitting
        active_batch_id: str | None = None
        if not dry_run and dispatcher.client:
            try:
                recent_batches = dispatcher.client.batches.list(limit=10)
                for b in recent_batches.data:
                    if b.status in ("validating", "in_progress", "finalizing"):
                        try:
                            f_obj = dispatcher.client.files.retrieve(b.input_file_id)
                            if f_obj.filename == batch_file.name:
                                active_batch_id = b.id
                                logger.info(
                                    "Lote activo previo detectado en OpenAI: ID = %s (estado: %s). Re-enganchando...",
                                    b.id,
                                    b.status,
                                )
                                break
                        except Exception:
                            continue
            except Exception as err:
                logger.warning("No se pudo inspeccionar lotes activos previos: %s", err)

        if active_batch_id:
            batch_id = active_batch_id
            status = "in_progress"
        else:
            # 3. Submit batch
            logger.info("[%d/%d] Despachando lote a OpenAI Batch API...", idx, total_files)
            sub_res = dispatcher.submit_batch(batch_file)
            batch_id = sub_res.batch_id
            status = sub_res.status
            logger.info("Lote encolado exitosamente: ID = %s", batch_id)

        # 4. Polling loop until terminal state
        while status in ("validating", "in_progress", "finalizing"):
            if dry_run:
                status = "completed"
                break

            info = dispatcher.check_status(batch_id)
            status = str(info.get("status", "unknown"))
            req_counts = info.get("request_counts", {})
            completed_c = req_counts.get("completed", 0) if isinstance(req_counts, dict) else 0
            total_c = req_counts.get("total", 0) if isinstance(req_counts, dict) else 0

            if status not in ("validating", "in_progress", "finalizing"):
                break

            logger.info(
                "Estado del lote %s: '%s' (%s/%s completadas). Esperando %ds...",
                batch_id,
                status,
                completed_c,
                total_c,
                poll_interval_seconds,
            )
            time.sleep(poll_interval_seconds)

        if status != "completed":
            logger.error("El lote %s terminó en estado no exitoso: '%s'. Deteniendo cola.", batch_id, status)
            run_records.append({
                "batch_file": str(batch_file),
                "batch_id": batch_id,
                "status": status,
                "error": True,
            })
            break

        # 5. Download results
        logger.info("[%d/%d] Lote %s completado. Descargando resultados...", idx, total_files, batch_id)
        downloaded_path = dispatcher.download_results(batch_id, output_path=output_file)
        logger.info("Resultados guardados en: %s", downloaded_path)

        run_records.append({
            "batch_file": str(batch_file),
            "batch_id": batch_id,
            "output_file": str(downloaded_path),
            "status": "completed",
        })

    return run_records


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Orquestador secuencial para OpenAI Batch API en entornos Tier 1."
    )
    parser.add_argument(
        "patterns",
        nargs="+",
        type=str,
        help="Uno o más patrones glob de archivos a procesar (ej. 'data/batches/*part*.jsonl')",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Alterna los archivos de los distintos patrones uno a uno (round-robin)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directorio donde se guardarán los resultados descargados (default: results)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Intervalo de consulta en segundos (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula el despacho y descarga sin llamadas a OpenAI",
    )

    args = parser.parse_args(argv)

    if args.interleave and len(args.patterns) > 1:
        groups = [sorted(Path(p) for p in glob.glob(pat)) for pat in args.patterns]
        matched = interleave_file_lists(*groups)
    else:
        matched = []
        for pat in args.patterns:
            matched.extend(sorted(Path(p) for p in glob.glob(pat)))

    if not matched:
        sys.exit(f"No se encontraron archivos coincidentes con los patrones: {args.patterns}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("Error: OPENAI_API_KEY no encontrada en las variables de entorno.")

    run_sequential_queue(
        batch_files=matched,
        results_dir=Path(args.results_dir),
        poll_interval_seconds=args.poll_interval,
        api_key=api_key,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
