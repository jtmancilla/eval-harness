"""Declarative banking tools and non-executable lexical decoys for MAYA."""

from __future__ import annotations

import random

from src.maya.contracts import MayaToolParameter, MayaToolSpec


def _parameter(
    name: str,
    value_type: str,
    description: str,
    allowed_values: tuple[str, ...] = (),
) -> MayaToolParameter:
    return MayaToolParameter(
        name=name,
        value_type=value_type,
        description=description,
        allowed_values=allowed_values,
    )


def _tool(
    name: str,
    operation_id: int | None,
    description: str,
    parameters: tuple[MayaToolParameter, ...] = (),
    required: tuple[str, ...] = (),
) -> MayaToolSpec:
    return MayaToolSpec(
        name=name,
        operation_id=operation_id,
        description=description,
        parameters=parameters,
        required=required,
        canonical=True,
    )


BANKING_OPERATION_TOOLS: tuple[MayaToolSpec, ...] = (
    _tool(
        "transfer_funds",
        1,
        "Transfiere recursos entre cuentas propias, de terceros o por SPEI.",
        (
            _parameter("source_product_id", "string", "Producto sintético de cargo."),
            _parameter("destination_reference", "string", "Cuenta o referencia sintética destino."),
            _parameter("amount_mxn", "number", "Monto expresado en pesos mexicanos."),
        ),
        ("source_product_id", "destination_reference", "amount_mxn"),
    ),
    _tool(
        "pay_credit_card",
        2,
        "Aplica un pago total o parcial a una tarjeta de crédito.",
        (
            _parameter("source_product_id", "string", "Producto sintético de cargo."),
            _parameter("card_product_id", "string", "Tarjeta sintética que recibe el pago."),
            _parameter("amount_mxn", "number", "Monto expresado en pesos mexicanos."),
        ),
        ("source_product_id", "card_product_id", "amount_mxn"),
    ),
    _tool(
        "pay_service",
        3,
        "Paga un servicio mediante una referencia proporcionada por el cliente.",
        (
            _parameter("service_reference", "string", "Referencia sintética del servicio."),
            _parameter("amount_mxn", "number", "Importe del servicio."),
        ),
        ("service_reference", "amount_mxn"),
    ),
    _tool(
        "pay_taxes",
        4,
        "Paga una contribución mediante línea de captura.",
        (
            _parameter("capture_line", "string", "Línea de captura sintética."),
            _parameter("amount_mxn", "number", "Importe de la contribución."),
        ),
        ("capture_line", "amount_mxn"),
    ),
    _tool(
        "enroll_installments",
        5,
        "Difiere una compra elegible a meses con o sin intereses.",
        (
            _parameter("transaction_id", "string", "Compra sintética elegible."),
            _parameter("months", "integer", "Número de mensualidades."),
        ),
        ("transaction_id", "months"),
    ),
    _tool(
        "set_card_block",
        6,
        "Activa o desactiva el bloqueo temporal de una tarjeta.",
        (
            _parameter("card_product_id", "string", "Tarjeta sintética."),
            _parameter("blocked", "boolean", "Estado de bloqueo solicitado."),
        ),
        ("card_product_id", "blocked"),
    ),
    _tool(
        "report_card_lost_or_stolen",
        7,
        "Reporta definitivamente una tarjeta por robo o extravío.",
        (
            _parameter("card_product_id", "string", "Tarjeta sintética comprometida."),
            _parameter("reason", "string", "Causal del reporte.", ("ROBO", "EXTRAVIO")),
        ),
        ("card_product_id", "reason"),
    ),
    _tool(
        "contract_product_or_promotion",
        8,
        "Contrata un producto o activa una promoción disponible.",
        (
            _parameter("offer_id", "string", "Oferta sintética vigente."),
            _parameter("accepted", "boolean", "Aceptación expresa del cliente."),
        ),
        ("offer_id", "accepted"),
    ),
    _tool(
        "get_balance",
        9,
        "Consulta el saldo actual de un producto contratado.",
        (_parameter("product_id", "string", "Producto sintético a consultar."),),
        ("product_id",),
    ),
    _tool(
        "get_recent_transactions",
        10,
        "Consulta los movimientos recientes de un producto.",
        (
            _parameter("product_id", "string", "Producto sintético a consultar."),
            _parameter("limit", "integer", "Número máximo de movimientos."),
        ),
        ("product_id", "limit"),
    ),
    _tool(
        "get_clabe",
        11,
        "Recupera la CLABE asociada a una cuenta del titular autenticado.",
        (_parameter("product_id", "string", "Cuenta sintética a consultar."),),
        ("product_id",),
    ),
    _tool("list_products", 12, "Lista los productos activos del cliente autenticado."),
    _tool(
        "get_account_statement",
        13,
        "Emite un estado de cuenta del periodo solicitado.",
        (
            _parameter("product_id", "string", "Producto sintético."),
            _parameter("period", "string", "Periodo documental en formato AAAA-MM."),
        ),
        ("product_id", "period"),
    ),
    _tool(
        "create_dispute",
        14,
        "Registra una aclaración sobre un movimiento no reconocido.",
        (
            _parameter("transaction_id", "string", "Movimiento sintético aclarado."),
            _parameter("reason", "string", "Motivo de la aclaración."),
        ),
        ("transaction_id", "reason"),
    ),
    _tool(
        "get_dispute_status",
        15,
        "Consulta el estado de una aclaración mediante su folio.",
        (_parameter("dispute_folio", "string", "Folio sintético de aclaración."),),
        ("dispute_folio",),
    ),
    _tool(
        "activate_mobile_token",
        16,
        "Activa el token celular en un dispositivo previamente registrado.",
        (_parameter("device_id", "string", "Dispositivo sintético registrado."),),
        ("device_id",),
    ),
    _tool(
        "handoff_to_human",
        17,
        "Transfiere la conversación a una persona ejecutiva conservando el historial.",
        (_parameter("reason", "string", "Motivo del escalamiento."),),
        ("reason",),
    ),
)

RAG_TOOL = _tool(
    "search_bank_knowledge",
    None,
    "Busca información bancaria general en documentos aprobados.",
    (_parameter("query", "string", "Consulta informativa del usuario."),),
    ("query",),
)

CANONICAL_TOOLS: tuple[MayaToolSpec, ...] = (*BANKING_OPERATION_TOOLS, RAG_TOOL)
CANONICAL_TOOLS_BY_NAME = {tool.name: tool for tool in CANONICAL_TOOLS}
_DECOY_PREFIXES = (
    "legacy",
    "express",
    "instant",
    "external",
    "beta",
    "batch",
    "backoffice",
    "manual",
    "unverified",
    "bypass",
    "shadow",
    "draft",
)


def generate_decoy_tools() -> tuple[MayaToolSpec, ...]:
    decoys: list[MayaToolSpec] = []
    for prefix in _DECOY_PREFIXES:
        for target in CANONICAL_TOOLS:
            decoys.append(
                MayaToolSpec(
                    name=f"{prefix}_{target.name}",
                    operation_id=None,
                    description=(
                        f"Señuelo sintético no ejecutable de la familia {prefix}; "
                        f"no sustituye a {target.name}."
                    ),
                    parameters=target.parameters,
                    required=target.required,
                    canonical=False,
                    target_canonical_tool=target.name,
                )
            )
    return tuple(decoys)


def build_tool_catalog(total: int, rng: random.Random) -> tuple[MayaToolSpec, ...]:
    decoys = list(generate_decoy_tools())
    if total < len(CANONICAL_TOOLS) or total > len(CANONICAL_TOOLS) + len(decoys):
        raise ValueError(
            f"Catalog size must be between {len(CANONICAL_TOOLS)} and "
            f"{len(CANONICAL_TOOLS) + len(decoys)}"
        )
    rng.shuffle(decoys)
    catalog = [*CANONICAL_TOOLS, *decoys[: total - len(CANONICAL_TOOLS)]]
    rng.shuffle(catalog)
    return tuple(catalog)
