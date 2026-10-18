"""`market_prices` : statistiques de prix de vente DVF par commune.

Données : data/processed/dvf_communes.csv (src.preprocessing.build_dvf_stats).
DVF décrit des VENTES, jamais des loyers.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.settings import get_settings
from src.tools.communes import resolve_commune
from src.tools.errors import DataNotCovered
from src.tools.schemas import (
    CommuneInfo,
    CommuneName,
    DataProvenance,
    Departement,
    StrictInput,
    ToolOutput,
)

TypeLocal = Literal["Appartement", "Maison", "Tous"]


class MarketPricesInput(StrictInput):
    commune: CommuneName = Field(description="Commune (nom ou code Insee).")
    departement: Departement | None = Field(
        default=None,
        description=(
            "Département (ex. \"93\"), uniquement s'il est donné par l'utilisateur ; "
            "sert à distinguer des communes homonymes."
        ),
    )
    type_local: TypeLocal = Field(
        default="Tous",
        description="Appartement, Maison, ou Tous (les deux confondus).",
    )
    annee: int | None = Field(
        default=None,
        description="Année des ventes ; toutes les années disponibles si absent.",
    )


class PriceStats(BaseModel):
    annee: int
    type_local: TypeLocal
    n_ventes: int
    statistiques_masquees: bool = Field(
        description="True si le nombre de ventes est sous le seuil de publication.",
    )
    prix_m2_median: float | None = None
    prix_m2_q1: float | None = None
    prix_m2_q3: float | None = None
    valeur_fonciere_mediane: float | None = None
    surface_mediane_m2: float | None = None


class MarketPricesResult(ToolOutput):
    commune: CommuneInfo
    annees_disponibles: list[int]
    statistiques: list[PriceStats]


@lru_cache(maxsize=1)
def _load_stats() -> pd.DataFrame:
    path = get_settings().tools.dvf.path
    return pd.read_csv(path, dtype={"code_insee": str})


def market_prices(request: MarketPricesInput) -> MarketPricesResult:
    """Prix médian au m² (et quartiles) des ventes de logements d'une commune.

    Raises:
        CommuneNotFound, AmbiguousCommune: commune non résolue.
        DataNotCovered: département hors DVF ou année non disponible.
    """
    settings = get_settings().tools.dvf
    commune, _ = resolve_commune(request.commune, request.departement)

    if commune.departement in settings.departements_non_couverts:
        raise DataNotCovered(
            "DVF ne couvre pas l'Alsace-Moselle (Bas-Rhin, Haut-Rhin, Moselle, "
            "régime du livre foncier) ni Mayotte : aucune statistique de prix "
            "n'est disponible pour cette commune."
        )

    if request.annee is not None and request.annee not in settings.annees:
        raise DataNotCovered(
            f"Années DVF disponibles : {', '.join(map(str, settings.annees))}."
        )

    annees = [request.annee] if request.annee is not None else list(settings.annees)

    stats = _load_stats()
    rows = stats[
        (stats["code_insee"] == commune.code_insee)
        & (stats["annee"].isin(annees))
        & (stats["type_local"] == request.type_local)
    ].sort_values("annee")

    results = []

    for row in rows.itertuples(index=False):
        masked = row.n_ventes < settings.min_ventes
        results.append(
            PriceStats(
                annee=row.annee,
                type_local=row.type_local,
                n_ventes=row.n_ventes,
                statistiques_masquees=masked,
                **(
                    {}
                    if masked
                    else {
                        "prix_m2_median": row.prix_m2_median,
                        "prix_m2_q1": row.prix_m2_q1,
                        "prix_m2_q3": row.prix_m2_q3,
                        "valeur_fonciere_mediane": row.valeur_fonciere_mediane,
                        "surface_mediane_m2": row.surface_mediane_m2,
                    }
                ),
            )
        )

    caveats = [
        (
            "Prix de VENTE issus des actes notariés (DVF) : ils ne renseignent pas "
            "sur les loyers, ni sur les loyers des logements sociaux."
        ),
        (
            "Seules les ventes portant sur un unique logement (maison ou appartement, "
            "dépendances incluses) sont retenues ; prix au m² = valeur foncière / "
            "surface réelle bâtie."
        ),
        (
            f"Sous {settings.min_ventes} ventes, les statistiques ne sont pas publiées "
            "(échantillon insuffisant)."
        ),
    ]

    if not results:
        caveats.append(
            "Aucune vente de logement retenue pour cette commune sur la période."
        )

    return MarketPricesResult(
        commune=commune,
        annees_disponibles=list(settings.annees),
        statistiques=results,
        provenance=DataProvenance(
            source_name=settings.source_name,
            source_url=settings.source_url,
            data_year=max(annees),
        ),
        caveats=caveats,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Démonstration de market_prices.")
    parser.add_argument(
        "json",
        help='Entrée JSON, ex. \'{"commune": "Lyon", "type_local": "Appartement"}\'',
    )
    args = parser.parse_args()

    result = market_prices(MarketPricesInput.model_validate_json(args.json))
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
