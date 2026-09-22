"""OpenAI Batch API dispatcher, budget governance client, and CLI tool."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:  # type: ignore[misc]
        return False


try:
    from openai import OpenAI
except ImportError:
    OpenAI = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class BatchSubmissionResult:
    """Metadata representing a submitted OpenAI batch job."""

    batch_id: str
    input_file_id: str
    status: str
    is_dry_run: bool


class OpenAIBatchDispatcher:
    """Dispatches, tracks, and downloads OpenAI Batch API jobs with budget enforcement."""

    DEFAULT_BUDGET_CEILING_USD: Decimal = Decimal("300.00")

    # Blended Batch API discount pricing per 1M tokens (50% off standard)
    MODEL_RATES: dict[str, dict[str, Decimal]] = {
        "gpt-5.6-luna": {"input": Decimal("1.25"), "output": Decimal("5.00")},
        "gpt-5.6-terra": {"input": Decimal("1.50"), "output": Decimal("6.00")},
        "gpt-5.6-sol": {"input": Decimal("2.50"), "output": Decimal("10.00")},
        "gpt-6-astra": {"input": Decimal("5.00"), "output": Decimal("20.00")},
    }
    DEFAULT_RATES: dict[str, Decimal] = {"input": Decimal("2.50"), "output": Decimal("10.00")}

    def __init__(
        self,
        api_key: str | None = None,
        dry_run: bool = False,
        client: Any | None = None,
    ) -> None:
        load_dotenv()
        self.dry_run = dry_run
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

        if not self.dry_run and client is None:
            if not self.api_key:
                self.dry_run = True
                self.client = None
            else:
                self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = client

    def validate_budget_limit(
        self,
        estimated_tokens: int,
        max_usd: float = 300.0,
    ) -> bool:
        """Verifies that the projected token volume stays within the assigned budget ceiling."""
        ceiling = Decimal(str(max_usd))

        tokens_dec = Decimal(str(estimated_tokens))
        input_tokens = tokens_dec * Decimal("0.85")
        output_tokens = tokens_dec * Decimal("0.15")

        cost_in = (input_tokens / Decimal("1000000")) * self.DEFAULT_RATES["input"]
        cost_out = (output_tokens / Decimal("1000000")) * self.DEFAULT_RATES["output"]
        total_projected_cost = (cost_in + cost_out).quantize(Decimal("0.01"))

        return total_projected_cost <= ceiling

    def inspect_and_validate_file(
        self,
        jsonl_path: Path,
        max_usd: float = 300.0,
    ) -> dict[str, Any]:
        """Line-by-line inspection validating OpenAI Batch API specs, uniqueness, and budget."""
        if not jsonl_path.exists():
            raise FileNotFoundError(f"El archivo batch no existe: {jsonl_path}")

        seen_custom_ids: set[str] = set()
        models_found: set[str] = set()
        urls_found: set[str] = set()
        total_requests = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = Decimal("0.00")

        with open(jsonl_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Línea {line_idx} no es un JSON válido: {exc}") from exc

                # 1. Validación de método y endpoint
                method = record.get("method")
                url = record.get("url")
                if method != "POST":
                    raise ValueError(f"Línea {line_idx}: método inválido '{method}', esperado 'POST'")
                if url not in ("/v1/chat/completions", "/v1/responses"):
                    raise ValueError(f"Línea {line_idx}: url inválida '{url}', esperado '/v1/chat/completions' o '/v1/responses'")
                urls_found.add(url)

                # 2. Validación de custom_id y unicidad
                custom_id = record.get("custom_id")
                if not custom_id or not isinstance(custom_id, str):
                    raise ValueError(f"Línea {line_idx}: 'custom_id' ausente o vacío")
                if custom_id in seen_custom_ids:
                    raise ValueError(f"Línea {line_idx}: 'custom_id' duplicado detectado: '{custom_id}'")
                seen_custom_ids.add(custom_id)

                # 3. Validación de cuerpo de solicitud
                body = record.get("body")
                if not isinstance(body, dict):
                    raise ValueError(f"Línea {line_idx}: 'body' ausente o no es un objeto")

                model = body.get("model", "")
                messages = body.get("messages")
                instructions = body.get("instructions")
                input_data = body.get("input")
                tools = body.get("tools")

                if not model:
                    raise ValueError(f"Línea {line_idx}: modelo no especificado en body")

                if url == "/v1/chat/completions":
                    if not isinstance(messages, list) or len(messages) == 0:
                        raise ValueError(f"Línea {line_idx}: 'messages' debe ser una lista no vacía para /v1/chat/completions")
                    input_text = json.dumps(messages)
                elif url == "/v1/responses":
                    if not instructions and not input_data and not messages:
                        raise ValueError(f"Línea {line_idx}: debe contener 'instructions' o 'input' para /v1/responses")
                    input_text = json.dumps(instructions or "") + json.dumps(input_data or messages or "")
                else:
                    input_text = ""

                if not isinstance(tools, list):
                    raise ValueError(f"Línea {line_idx}: 'tools' debe ser un array de herramientas")

                models_found.add(model)
                rates = self.MODEL_RATES.get(model, self.DEFAULT_RATES)

                # 4. Proyección de tokens y costos
                req_in_tokens = (len(input_text) + len(json.dumps(tools))) // 4
                req_out_tokens = 300

                total_input_tokens += req_in_tokens
                total_output_tokens += req_out_tokens

                cost_in = (Decimal(req_in_tokens) / Decimal("1000000")) * rates["input"]
                cost_out = (Decimal(req_out_tokens) / Decimal("1000000")) * rates["output"]
                total_cost += cost_in + cost_out
                total_requests += 1

        ceiling = Decimal(str(max_usd))
        is_within_budget = total_cost <= ceiling

        return {
            "valid": True,
            "total_requests": total_requests,
            "unique_custom_ids": len(seen_custom_ids),
            "models": sorted(models_found),
            "urls": sorted(urls_found),
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "estimated_total_tokens": total_input_tokens + total_output_tokens,
            "estimated_cost_usd": total_cost.quantize(Decimal("0.01")),
            "budget_limit_usd": ceiling,
            "is_within_budget": is_within_budget,
            "file_path": str(jsonl_path.resolve()),
        }

    def submit_batch(
        self,
        jsonl_path: Path,
        completion_window: Literal["24h"] = "24h",
        endpoint: str | None = None,
    ) -> BatchSubmissionResult:
        """Uploads the dataset and initiates an OpenAI Batch job.

        Infers the API endpoint dynamically from the first record of the JSONL if not provided.
        """
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Batch dataset not found: {jsonl_path}")

        # Infiere dinámicamente el endpoint a partir de la url del primer registro
        if endpoint is None:
            endpoint = "/v1/chat/completions"
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        first_rec = json.loads(line)
                        endpoint = first_rec.get("url", "/v1/chat/completions")
                        break

        if self.dry_run:
            fake_batch_id = f"batch_dryrun_{uuid4().hex[:12]}"
            fake_file_id = f"file_dryrun_{uuid4().hex[:12]}"
            return BatchSubmissionResult(
                batch_id=fake_batch_id,
                input_file_id=fake_file_id,
                status="validating",
                is_dry_run=True,
            )

        assert self.client is not None, "OpenAI client must be initialized for non-dry-run mode"

        with open(jsonl_path, "rb") as f:
            batch_file = self.client.files.create(file=f, purpose="batch")

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint=endpoint,  # type: ignore[arg-type]
            completion_window=completion_window,
        )

        return BatchSubmissionResult(
            batch_id=batch_job.id,
            input_file_id=batch_file.id,
            status=batch_job.status,
            is_dry_run=False,
        )

    def check_status(self, batch_id: str) -> dict[str, Any]:
        """Queries the current progress of a submitted batch job."""
        if self.dry_run or batch_id.startswith("batch_dryrun_"):
            return {
                "id": batch_id,
                "status": "completed",
                "output_file_id": f"file_dryrun_out_{uuid4().hex[:8]}",
                "error_file_id": None,
                "request_counts": {"total": 1200, "completed": 1200, "failed": 0},
                "is_dry_run": True,
            }

        assert self.client is not None, "OpenAI client must be initialized"
        batch_job = self.client.batches.retrieve(batch_id)

        counts = {}
        if hasattr(batch_job, "request_counts") and batch_job.request_counts:
            counts = {
                "total": getattr(batch_job.request_counts, "total", 0),
                "completed": getattr(batch_job.request_counts, "completed", 0),
                "failed": getattr(batch_job.request_counts, "failed", 0),
            }

        return {
            "id": batch_job.id,
            "status": batch_job.status,
            "output_file_id": getattr(batch_job, "output_file_id", None),
            "error_file_id": getattr(batch_job, "error_file_id", None),
            "request_counts": counts,
            "is_dry_run": False,
        }

    def download_results(
        self,
        batch_id: str,
        output_path: Path,
    ) -> Path:
        """Downloads the completion output file for a successfully processed batch."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.dry_run or batch_id.startswith("batch_dryrun_"):
            with open(output_path, "w", encoding="utf-8") as f:
                f.write('{"dry_run": true, "batch_id": "' + batch_id + '"}\n')
            return output_path

        status_info = self.check_status(batch_id)
        output_file_id = status_info.get("output_file_id")

        if not output_file_id:
            raise ValueError(
                f"Batch {batch_id} does not have an output file available yet (status: {status_info.get('status')})"
            )

        assert self.client is not None, "OpenAI client must be initialized"
        content = self.client.files.content(output_file_id).text

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path


def download_batch_output(
    client: OpenAI,
    batch_id: str,
    output_path: str | None = None,
) -> Path | None:
    """Retrieves and writes the output file of a completed OpenAI Batch job.

    Args:
        client: Authenticated OpenAI client instance.
        batch_id: Identifier of the batch job to download.
        output_path: Optional file path for the downloaded JSONL. Defaults to
            'results/{batch_id}_output.jsonl'.

    Returns:
        The Path to the saved file if completed, or None if the batch is not completed.
    """
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        progress = ""
        if hasattr(batch, "request_counts") and batch.request_counts:
            c = batch.request_counts
            progress = f" (Completadas: {getattr(c, 'completed', 0)}/{getattr(c, 'total', 0)}, Fallidas: {getattr(c, 'failed', 0)})"
        print(f"[ADVERTENCIA]: El lote '{batch_id}' no está en estado 'completed'. Estado actual: '{batch.status}'{progress}.")
        return None

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id or not isinstance(output_file_id, str):
        print(f"[ADVERTENCIA]: El lote '{batch_id}' está completado pero no tiene 'output_file_id' disponible.")
        return None

    target = Path(output_path) if output_path else Path(f"results/{batch_id}_output.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)

    content = client.files.content(output_file_id).text
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)

    non_empty_lines = sum(1 for line in content.splitlines() if line.strip())
    print("\n" + "=" * 64)
    print(" OPENAI BATCH API — DESCARGA EXITOSA")
    print("=" * 64)
    print(f"• Batch ID:                {batch_id}")
    print(f"• Archivo guardado:        {target.resolve()}")
    print(f"• Líneas no vacías:        {non_empty_lines:,}")
    print("=" * 64 + "\n")

    return target


def main(argv: list[str] | None = None) -> None:
    """CLI Entrypoint for OpenAI Batch API operations."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API Dispatcher & Budget Governance CLI (Regulated Fintech Mexico)"
    )
    parser.add_argument(
        "--dry-run",
        metavar="RUTA_JSONL",
        nargs="?",
        const=Path("eval_batch_1200.jsonl"),
        type=Path,
        help="Inspecciona y valida formalmente el archivo .jsonl sin llamar a la API",
    )
    parser.add_argument(
        "--submit",
        metavar="RUTA_JSONL",
        type=Path,
        help="Sube e inicia el lote en OpenAI Batch API (requiere OPENAI_API_KEY en entorno)",
    )
    parser.add_argument(
        "--status",
        metavar="BATCH_ID",
        type=str,
        help="Consulta el estatus actual de un lote en OpenAI Batch API",
    )
    parser.add_argument(
        "--download",
        metavar="BATCH_ID",
        type=str,
        help="Descarga el archivo de resultados para un lote completado en OpenAI Batch API",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="RUTA_DESTINO",
        type=str,
        default=None,
        help="Ruta destino opcional para el archivo descargado",
    )

    args = parser.parse_args(argv)

    if args.dry_run:
        target_path = args.dry_run
        print(f"\n-> Iniciando inspección estricta (--dry-run) en: {target_path}")

        dispatcher = OpenAIBatchDispatcher(dry_run=True)
        try:
            summary = dispatcher.inspect_and_validate_file(target_path, max_usd=300.0)
        except Exception as exc:
            sys.exit(f"\n[ERROR DE VALIDACIÓN]: {exc}")

        print("\n" + "=" * 64)
        print(" OPENAI BATCH API — REPORTE DE INSPECCIÓN (--dry-run)")
        print("=" * 64)
        print(f"• Archivo verificado:      {summary['file_path']}")
        endpoints_label = ", ".join(f"POST {u}" for u in summary.get("urls", [])) or "POST"
        print(f"• Solicitudes válidas:     {summary['total_requests']:,} ({endpoints_label})")
        print(f"• Unicidad de custom_id:   100% Únicos ({summary['unique_custom_ids']:,} identificadores)")
        print(f"• Modelos en lote:         {', '.join(summary['models'])}")
        print(f"• Tokens entrada est.:     {summary['estimated_input_tokens']:,}")
        print(f"• Tokens salida est.:      {summary['estimated_output_tokens']:,}")
        print(f"• Tokens totales est.:     {summary['estimated_total_tokens']:,}")
        print(f"• Costo proyectado:        ${summary['estimated_cost_usd']} USD")
        print(f"• Límite presupuestal:    ${summary['budget_limit_usd']} USD")
        print(f"• Dentro de presupuesto:   {'SÍ (Aprobado)' if summary['is_within_budget'] else 'NO (Excedido)'}")
        print("=" * 64)
        print("-> Estatus: Inspección completada exitosamente sin emitir llamadas a OpenAI.\n")
        return

    if args.submit:
        target_path = args.submit
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            sys.exit("\n[ERROR DE AUTENTICACIÓN]: La variable OPENAI_API_KEY no está configurada en el entorno.\n")

        print(f"\n-> Validando precondiciones antes del despacho formal: {target_path}")
        dispatcher = OpenAIBatchDispatcher(api_key=api_key, dry_run=False)

        try:
            summary = dispatcher.inspect_and_validate_file(target_path, max_usd=300.0)
        except Exception as exc:
            sys.exit(f"\n[ERROR EN PRE-VALIDACIÓN]: {exc}")

        if not summary["is_within_budget"]:
            sys.exit(
                f"\n[ABORTADO]: El costo proyectado (${summary['estimated_cost_usd']}) supera la cota de "
                f"${summary['budget_limit_usd']} USD.\n"
            )

        print("-> Subiendo archivo e iniciando job en OpenAI Batch API...")
        result = dispatcher.submit_batch(target_path)

        print("\n" + "=" * 64)
        print(" OPENAI BATCH API — DESPACHO EXITOSO")
        print("=" * 64)
        print(f"• Batch ID:                {result.batch_id}")
        print(f"• Input File ID:           {result.input_file_id}")
        print(f"• Estado inicial:          {result.status}")
        print("=" * 64 + "\n")
        return

    if args.status:
        batch_id = args.status
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            print(f"[MODO SIMULACIÓN]: Sin OPENAI_API_KEY. Consultando estado simulado para {batch_id}...")
            dispatcher = OpenAIBatchDispatcher(dry_run=True)
        else:
            dispatcher = OpenAIBatchDispatcher(api_key=api_key, dry_run=False)

        info = dispatcher.check_status(batch_id)

        print("\n" + "=" * 64)
        print(" OPENAI BATCH API — CONSULTA DE ESTATUS")
        print("=" * 64)
        print(f"• Batch ID:                {info['id']}")
        print(f"• Estado actual:           {info['status']}")
        print(f"• Output File ID:          {info.get('output_file_id') or 'N/A (En proceso)'}")
        print(f"• Error File ID:           {info.get('error_file_id') or 'Ninguno'}")
        counts = info.get("request_counts", {})
        if counts:
            print(f"• Solicitudes totales:     {counts.get('total', 0):,}")
            print(f"• Completadas:             {counts.get('completed', 0):,}")
            print(f"• Fallidas:                {counts.get('failed', 0):,}")
        print("=" * 64 + "\n")
        return

    if args.download:
        batch_id = args.download
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            sys.exit("\n[ERROR DE AUTENTICACIÓN]: La variable OPENAI_API_KEY no está configurada en el entorno.\n")

        client = OpenAI(api_key=api_key)
        download_batch_output(client, batch_id, output_path=args.output)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
