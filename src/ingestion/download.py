"""Téléchargement en streaming des sources de données ouvertes."""

from __future__ import annotations

import logging
import shutil
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

import certifi

logger = logging.getLogger(__name__)

USER_AGENT = "Projet-Generative-AI-HLM/0.1 (projet etudiant ENPC)"

# Le Python de python.org sur macOS n'utilise pas le trousseau système :
# on fournit explicitement le bundle de certificats de certifi.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def download(url: str, dest: Path, overwrite: bool = False) -> Path:
    """Télécharge `url` vers `dest` (écriture atomique via un fichier .part)."""
    if dest.exists() and not overwrite:
        logger.info("Déjà présent, téléchargement ignoré : %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")

    logger.info("Téléchargement : %s -> %s", url, dest)
    request = Request(url, headers={"User-Agent": USER_AGENT})

    with (
        urlopen(request, timeout=120, context=_SSL_CONTEXT) as response,
        partial.open("wb") as f,
    ):
        shutil.copyfileobj(response, f, length=1024 * 1024)

    partial.replace(dest)
    logger.info("Terminé : %s (%.1f Mo)", dest, dest.stat().st_size / 1e6)

    return dest
