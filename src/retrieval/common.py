"""Fonctions partagées par tous les retrievers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# Chemins par défaut (définis une seule fois dans config.py).
from src.retrieval.config import (  # noqa: F401
    CHUNKS_PATH,
    GOLDEN_PATH,
    INDEX_DIR,
    ROOT_DIR,
)


def load_jsonl(path: Path | str) -> list[dict]:
    rows = []

    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_chunks(path: Path | str = CHUNKS_PATH) -> list[dict]:
    chunks = load_jsonl(path)

    if not chunks:
        raise ValueError(f"Aucun chunk trouvé dans {path}.")

    return chunks


def sha256_file(path: Path | str) -> str:
    """Empreinte SHA-256 d'un fichier : change dès qu'un octet change."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def rank_indices(
    scores: np.ndarray,
    top_k: int | None = None,
    allowed: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Classe les chunks par score décroissant.

    - tri stable : en cas d'égalité, l'ordre du fichier de chunks est
      conservé, donc le résultat est reproductible ;
    - allowed (optionnel) : masque booléen, un booléen par chunk, pour
      ne garder que certains chunks (par exemple une seule source) ;
    - top_k = None : on renvoie le classement complet.

    Renvoie une liste de couples (indice du chunk, score).
    """
    order = np.argsort(-scores, kind="stable")

    if allowed is not None:
        order = order[allowed[order]]

    if top_k is not None:
        order = order[:top_k]

    return [(int(index), float(scores[index])) for index in order]
