"""Mexican Spanish narrative variation without external generative services."""

from __future__ import annotations

import random

from src.maya.contracts import MayaNarrativeStyle, MayaPersona

NARRATIVE_STYLES: tuple[MayaNarrativeStyle, ...] = (
    MayaNarrativeStyle(
        style_id="s_directo",
        description="Solicitud explícita, completa y sin contexto adicional.",
        language_register="neutral",
        noise_level="low",
    ),
    MayaNarrativeStyle(
        style_id="s_conversacional",
        description="Saludo y lenguaje cotidiano antes de la solicitud.",
        language_register="informal",
        noise_level="low",
    ),
    MayaNarrativeStyle(
        style_id="s_formal",
        description="Tratamiento de usted y formulación cortés.",
        language_register="formal",
        noise_level="low",
    ),
    MayaNarrativeStyle(
        style_id="s_tipografico",
        description="Errores tipográficos leves que no alteran montos ni referencias.",
        language_register="informal",
        noise_level="medium",
    ),
    MayaNarrativeStyle(
        style_id="s_fragmentado",
        description="Mensaje dividido en frases cortas conservando la intención.",
        language_register="neutral",
        noise_level="medium",
    ),
    MayaNarrativeStyle(
        style_id="s_urgente",
        description="Contexto de urgencia sin autorizar omisiones de seguridad.",
        language_register="directo",
        noise_level="medium",
    ),
    MayaNarrativeStyle(
        style_id="s_saturado",
        description="Narrativa extensa con contexto irrelevante para la operación.",
        language_register="conversacional",
        noise_level="high",
    ),
    MayaNarrativeStyle(
        style_id="s_cambio_intencion",
        description="Una idea inicial descartada seguida de la intención vigente.",
        language_register="conversacional",
        noise_level="high",
    ),
    MayaNarrativeStyle(
        style_id="s_ruido_multitema",
        description="Comentarios adicionales que explícitamente no solicitan otra acción.",
        language_register="conversacional",
        noise_level="high",
    ),
    MayaNarrativeStyle(
        style_id="s_regional",
        description="Ubicación geográfica natural sin estereotipos ni modismos forzados.",
        language_register="neutral",
        noise_level="medium",
    ),
)

STYLES_BY_ID = {style.style_id: style for style in NARRATIVE_STYLES}

_CONVERSATIONAL_PREFIXES = (
    "Hola, ¿me echas la mano? ",
    "Buenas, ¿me ayudas con esto? ",
    "Hola. Quiero resolver algo: ",
)
_URGENT_PREFIXES = (
    "Es urgente y necesito confirmar que quede bien atendido: ",
    "Tengo poco tiempo, pero no omitas ninguna validación: ",
    "Necesito atenderlo hoy: ",
)
_TYPO_REPLACEMENTS = (
    ("transferencia", "tranferencia"),
    ("tarjeta", "tarjta"),
    ("consulta", "consutla"),
    ("bloquear", "bloqear"),
    ("Quiero", "Kiero"),
    ("¿Cómo", "¿Como"),
    ("estado de cuenta", "estado de qenta"),
)


def _apply_typos(text: str) -> str:
    for original, replacement in _TYPO_REPLACEMENTS:
        if original in text:
            return text.replace(original, replacement, 1)
    return f"Kiero hacer esta solicitud: {text}"


def render_user_text(
    utterance: str,
    style: MayaNarrativeStyle,
    persona: MayaPersona,
    rng: random.Random,
) -> str:
    if style.style_id == "s_directo":
        return utterance
    if style.style_id == "s_conversacional":
        return f"{rng.choice(_CONVERSATIONAL_PREFIXES)}{utterance}"
    if style.style_id == "s_formal":
        return f"Buenas tardes. Agradeceré su apoyo con la siguiente solicitud: {utterance}"
    if style.style_id == "s_tipografico":
        return _apply_typos(utterance)
    if style.style_id == "s_fragmentado":
        fragmented = utterance.replace(", ", ".\n").replace(" y luego ", ".\nDespués, ")
        return f"Se lo escribo por partes.\n{fragmented}"
    if style.style_id == "s_urgente":
        return f"{rng.choice(_URGENT_PREFIXES)}{utterance}"
    if style.style_id == "s_saturado":
        context = (
            "Le escribo después de un día bastante largo. Esta mañana actualicé mi teléfono, "
            "ordené unos papeles y estuve revisando correos pendientes. También tuve que salir "
            "a hacer unos pendientes personales y por eso hasta ahora puedo sentarme a escribir. "
            "Le comparto todo este contexto porque no quiero dejar nada importante fuera, aunque "
            "lo único que necesito resolver con el banco es lo siguiente. "
        )
        closing = (
            " Cuando termine, por favor indíqueme claramente el resultado de esta solicitud; "
            "el resto de mi relato es solamente contexto y no requiere ninguna gestión."
        )
        return f"{context}{utterance}{closing}"
    if style.style_id == "s_cambio_intencion":
        return (
            "Al principio pensé que necesitaba hablar con soporte, pero ya entendí lo que ocupo. "
            f"Descarte esa primera idea y atienda solamente esto: {utterance}"
        )
    if style.style_id == "s_ruido_multitema":
        return (
            "La aplicación se actualizó anoche y tuve que volver a entrar; no necesito ayuda con eso. "
            f"Tampoco hace falta explicarme la actualización. Mi única solicitud es: {utterance}"
        )
    if style.style_id == "s_regional":
        return f"Le escribo desde {persona.city}, {persona.state}. {utterance}"
    raise ValueError(f"Unsupported narrative style: {style.style_id}")
