"""`social_housing_stats` : composition du parc social d'une commune (RPLS).

Données : data/processed/rpls_communes.csv (src.preprocessing.build_rpls_stats).
Photographie au 1er janvier : ni disponibilité, ni vacance, ni loyers.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, Field

from src.settings import get_settings
from src.tools.communes import resolve_commune
from src.tools.schemas import (
    CommuneInfo,
    CommuneName,
    DataProvenance,
    Departement,
    StrictInput,
    ToolOutput,
)


class SocialHousingInput(StrictInput):
    commune: CommuneName = Field(description="Commune (nom ou code Insee).")
    departement: Departement | None = Field(
        default=None,
        description=(
            "Département (ex. \"93\"), uniquement s'il est donné par l'utilisateur ; "
            "sert à distinguer des communes homonymes."
        ),
    )


class RepartitionFinancement(BaseModel):
    plai: int = Field(description="PLAI et assimilés (très social).")
    plus: int = Field(description="PLUS et assimilés (dont HLM/O, PLA ordinaire, HBM).")
    pls: int = Field(description="PLS et assimilés (intermédiaire).")
    pli: int = Field(
        description="PLI (intermédiaire, hors logement social au sens strict)."
    )
    autres: int = Field(description="Financements anciens ou spécifiques.")


class RepartitionTaille(BaseModel):
    une_piece: int
    deux_pieces: int
    trois_pieces: int
    quatre_pieces: int
    cinq_pieces_et_plus: int


class SocialHousingResult(ToolOutput):
    commune: CommuneInfo
    millesime: int = Field(description="Parc au 1er janvier de cette année.")
    nb_logements: int
    financement_initial: RepartitionFinancement | None
    nb_collectif: int | None
    nb_individuel: int | None
    nb_logements_etudiants: int | None
    taille: RepartitionTaille | None
    surface_moyenne_m2: float | None
    part_en_qpv_pct: float | None = Field(
        description="Part des logements situés en quartier prioritaire de la ville.",
    )
    annee_construction_mediane: int | None
    part_dpe_f_g_pct: float | None = Field(
        description="Part des logements classés F ou G, parmi ceux dont le DPE est renseigné.",
    )


@lru_cache(maxsize=1)
def _load_stats() -> pd.DataFrame:
    path = get_settings().tools.rpls.path
    return pd.read_csv(path, dtype={"code_insee": str}).set_index("code_insee")


def _pct(part: int, total: int) -> float | None:
    return round(100 * part / total, 1) if total else None


def social_housing_stats(request: SocialHousingInput) -> SocialHousingResult:
    """Nombre et caractéristiques des logements sociaux d'une commune.

    Raises:
        CommuneNotFound, AmbiguousCommune: commune non résolue.
    """
    settings = get_settings().tools.rpls
    commune, _ = resolve_commune(request.commune, request.departement)

    caveats = [
        (
            f"Photographie du parc au 1er janvier {settings.millesime} : le RPLS ne "
            "dit pas quels logements sont disponibles aujourd'hui."
        ),
        (
            "Le fichier public du RPLS ne contient ni la vacance, ni les loyers, ni "
            "l'occupation des logements."
        ),
        (
            "Le financement indiqué est le financement INITIAL du logement, regroupé "
            "par le projet (PLAI / PLUS / PLS / PLI / autres) ; il ne correspond pas "
            "forcément aux plafonds de ressources appliqués aujourd'hui."
        ),
        (
            "Le taux de logements sociaux au sens de la loi SRU ne peut pas être "
            "calculé avec ces seules données (il faut le nombre de résidences principales)."
        ),
    ]

    stats = _load_stats()

    provenance = DataProvenance(
        source_name=settings.source_name,
        source_url=settings.source_url,
        data_year=settings.millesime,
    )

    if commune.code_insee not in stats.index:
        caveats.append(
            "Aucun logement social de bailleur recensé dans le RPLS pour cette commune."
        )
        return SocialHousingResult(
            commune=commune,
            millesime=settings.millesime,
            nb_logements=0,
            financement_initial=None,
            nb_collectif=None,
            nb_individuel=None,
            nb_logements_etudiants=None,
            taille=None,
            surface_moyenne_m2=None,
            part_en_qpv_pct=None,
            annee_construction_mediane=None,
            part_dpe_f_g_pct=None,
            provenance=provenance,
            caveats=caveats,
        )

    row = stats.loc[commune.code_insee]
    total = int(row["nb_logements"])

    return SocialHousingResult(
        commune=commune,
        millesime=settings.millesime,
        nb_logements=total,
        financement_initial=RepartitionFinancement(
            plai=int(row["nb_plai"]),
            plus=int(row["nb_plus"]),
            pls=int(row["nb_pls"]),
            pli=int(row["nb_pli"]),
            autres=int(row["nb_autres_financements"]),
        ),
        nb_collectif=int(row["nb_collectif"]),
        nb_individuel=int(row["nb_individuel"]),
        nb_logements_etudiants=int(row["nb_etudiant"]),
        taille=RepartitionTaille(
            une_piece=int(row["nb_1_piece"]),
            deux_pieces=int(row["nb_2_pieces"]),
            trois_pieces=int(row["nb_3_pieces"]),
            quatre_pieces=int(row["nb_4_pieces"]),
            cinq_pieces_et_plus=int(row["nb_5_pieces_et_plus"]),
        ),
        surface_moyenne_m2=None
        if pd.isna(row["surface_moyenne_m2"])
        else float(row["surface_moyenne_m2"]),
        part_en_qpv_pct=_pct(int(row["nb_en_qpv"]), total),
        annee_construction_mediane=None
        if pd.isna(row["annee_construction_mediane"])
        else int(row["annee_construction_mediane"]),
        part_dpe_f_g_pct=_pct(int(row["nb_dpe_f_g"]), int(row["nb_dpe_renseigne"])),
        provenance=provenance,
        caveats=caveats,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Démonstration de social_housing_stats."
    )
    parser.add_argument("json", help='Entrée JSON, ex. \'{"commune": "Créteil"}\'')
    args = parser.parse_args()

    result = social_housing_stats(SocialHousingInput.model_validate_json(args.json))
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
