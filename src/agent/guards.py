"""Garde-fous contre la prompt injection autour des exécutions d'outils.

- sanitize_text : neutralise ce qui, dans un texte non fiable (passage du
  corpus, sortie d'outil), pourrait se faire passer pour une instruction
  ou une balise de rôle du format de chat ;
- detect_injection : signale (sans bloquer) les formulations typiques
  d'injection, pour la trace et l'évaluation ;
- ungrounded_numbers : repère les nombres d'une réponse absents de toutes
  les sources (sorties d'outils, passages, question).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Caractères de contrôle, de largeur nulle et de contrôle bidirectionnel.
_INVISIBLE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u2064\ufeff]"
)

# Marqueurs de rôle ou de format de chat (ChatML, Llama, balises XML de rôle).
_ROLE_MARKERS = re.compile(
    r"<\|[^|>]{0,40}\|>"
    r"|\[/?(?:INST|SYS)\]"
    r"|<<\s*/?SYS\s*>>"
    r"|</?\s*(?:system|assistant|user|tool|tool_call|function|instructions?)\b[^>]{0,40}>"
    r"|^\s*(?:system|assistant|developer)\s*:",
    flags=re.IGNORECASE | re.MULTILINE,
)

NEUTRALIZED = "[balise neutralisée]"

_INJECTION_PATTERNS = [
    r"ignore[rz]?\s+(?:toutes?\s+)?(?:les\s+|tes\s+|vos\s+)?(?:instructions|consignes|règles)",
    r"oublie[rz]?\s+(?:toutes?\s+)?(?:les\s+|tes\s+|vos\s+)?(?:instructions|consignes|règles)",
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+instructions",
    r"disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above)",
    r"(?:révèle|affiche|donne|montre|répète)[a-z]*\s+(?:moi\s+)?(?:ton|tes|le|les|votre)\s+(?:prompt|instructions|consignes)",
    r"(?:prompt|message)\s+syst[eè]me",
    r"system\s+prompt",
    r"tu\s+es\s+(?:maintenant|désormais)",
    r"you\s+are\s+now",
    r"(?:mode|rôle)\s+(?:développeur|developer|admin|dan)\b",
    r"sans\s+(?:utiliser\s+)?(?:les\s+)?(?:documents|sources|outils)",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), flags=re.IGNORECASE)


def sanitize_text(text: str, max_chars: int | None = None) -> str:
    """Nettoie un texte non fiable avant de le renvoyer au LLM."""
    text = _INVISIBLE.sub("", text)
    text = _ROLE_MARKERS.sub(NEUTRALIZED, text)

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + " […]"

    return text


def detect_injection(text: str) -> bool:
    return _INJECTION.search(text) is not None


# ============================================================
# ANCRAGE DES NOMBRES
# ============================================================

# « 15 133 », « 4 467,5 », « 82,6 », « 26920.0 », « 1 234 567 ».
_NUMBER = re.compile(
    r"(?<![\w.,])\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?|(?<![\w.,])\d+(?:[.,]\d+)?"
)

# Les petits entiers (nombre de personnes, de pièces, de jours…) viennent
# le plus souvent de la question ou du texte : on ne les contrôle pas.
_MIN_CHECKED = 100


def extract_numbers(text: str) -> set[float]:
    numbers = set()

    for match in _NUMBER.findall(text):
        cleaned = re.sub(r"[ \u00a0\u202f]", "", match).replace(",", ".")
        try:
            numbers.add(float(cleaned))
        except ValueError:
            continue

    return numbers


def ungrounded_numbers(answer: str, sources: Iterable[str]) -> list[float]:
    """Nombres (≥ 100 ou décimaux) de la réponse absents de toutes les sources.

    Une valeur est considérée ancrée si une source contient la même valeur
    à l'arrondi près (4 467 dans la réponse, 4467.0 dans la sortie d'outil).
    """
    known = set()
    for source in sources:
        known |= extract_numbers(source)

    known_rounded = {round(value) for value in known} | {
        round(value, 1) for value in known
    }

    missing = []
    for value in sorted(extract_numbers(answer)):
        if value.is_integer() and value < _MIN_CHECKED:
            continue
        if (
            value in known
            or round(value) in known_rounded
            or round(value, 1) in known_rounded
        ):
            continue
        missing.append(value)

    return missing
