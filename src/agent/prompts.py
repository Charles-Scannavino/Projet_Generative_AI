"""Chargement et rendu des prompts (prompts/*.md.j2).

Seules des variables FIABLES (configuration, liste des outils) sont rendues
dans les gabarits. La question de l'utilisateur et les sorties d'outils
passent uniquement dans des messages `user` / `tool` séparés.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import PROMPTS_DIR
from src.settings import Backend, get_settings


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def render(name: str, **trusted_vars: object) -> str:
    return _environment().get_template(name).render(**trusted_vars)


def prompt_version(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def system_prompt(backend: Backend, tool_names: list[str]) -> str:
    tools = get_settings().tools
    return render(
        "system.md.j2",
        backend=backend,
        tool_names=tool_names,
        plafonds_annee=tools.plafonds.annee,
        revenus_annee=tools.plafonds.annee_revenus,
        rpls_millesime=tools.rpls.millesime,
        dvf_annees=tools.dvf.annees,
    )
