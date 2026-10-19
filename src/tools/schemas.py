"""Types partagés des outils déterministes.

Les entrées des outils peuvent provenir d'un LLM : elles sont donc
validées strictement (champs inconnus refusés, pas de conversion
implicite, chaînes contraintes par motif et longueur, valeurs fermées).
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Lettres (accents et ligatures compris), chiffres pour les arrondissements
# (« Paris 15e »), espace, apostrophes et tiret. Pas de ponctuation de
# phrase, de retour à la ligne ni de balise : une consigne glissée dans ce
# champ est rejetée avant même la recherche dans le référentiel.
CommuneName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-zÀ-ÖØ-öø-ÿŒœÆæ0-9' ’\-]+$",
    ),
]

CodeInsee = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^(\d{5}|2[AB]\d{3})$"),
]

Departement = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^(\d{2}|2[AB]|97\d)$"),
]


class StrictInput(BaseModel):
    """Base des entrées d'outils : extra interdit, aucune coercition."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DataProvenance(BaseModel):
    source_name: str
    source_url: str
    data_year: int = Field(description="Millésime des données utilisées.")
    retrieved_on: date | None = Field(
        default=None,
        description="Date de constitution du fichier traité, si connue.",
    )


class CommuneInfo(BaseModel):
    """Commune résolue dans le COG."""

    code_insee: str = Field(
        description="Code le plus précis : arrondissement municipal si demandé.",
    )
    code_commune: str = Field(description="Code de la commune de rattachement.")
    libelle: str
    departement: str
    region: str


class ToolOutput(BaseModel):
    provenance: DataProvenance
    caveats: list[str] = Field(default_factory=list)
