"""Construit le référentiel des communes à partir du COG Insee.

Entrée  : data/raw/cog/v_commune_<millésime>.csv (téléchargé si absent)
Sortie  : data/processed/communes_ref.csv (versionné)

Le référentiel garde les communes (COM), les arrondissements municipaux
de Paris, Lyon et Marseille (ARM), ainsi que les communes déléguées et
associées (COMD, COMA) comme alias de leur commune de rattachement.

    uv run python -m src.preprocessing.build_communes_ref
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import RAW_DIR
from src.ingestion.download import download
from src.settings import get_settings

logger = logging.getLogger(__name__)

KEPT_TYPES = {"COM", "ARM", "COMD", "COMA"}


def build() -> pd.DataFrame:
    settings = get_settings().tools.communes
    raw_path = RAW_DIR / "cog" / f"v_commune_{settings.millesime}.csv"
    download(settings.source_url, raw_path)

    cog = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    cog = cog[cog["TYPECOM"].isin(KEPT_TYPES)].copy()

    # Les arrondissements, communes déléguées et associées n'ont pas de
    # département/région renseignés : on les reprend de la commune parente.
    communes = cog[cog["TYPECOM"] == "COM"].set_index("COM")

    is_child = cog["TYPECOM"] != "COM"
    cog["code_commune"] = cog["COM"].where(~is_child, cog["COMPARENT"])

    parent = communes.reindex(cog["code_commune"])
    cog["departement"] = cog["DEP"].where(cog["DEP"] != "", parent["DEP"].to_numpy())
    cog["region"] = cog["REG"].where(cog["REG"] != "", parent["REG"].to_numpy())

    departements_path = RAW_DIR / "cog" / f"v_departement_{settings.millesime}.csv"
    download(settings.departements_url, departements_path)
    departements = pd.read_csv(departements_path, dtype=str, keep_default_na=False)
    cog["departement_libelle"] = cog["departement"].map(
        departements.set_index("DEP")["LIBELLE"]
    )

    ref = cog.rename(
        columns={
            "TYPECOM": "typecom",
            "COM": "code_insee",
            "LIBELLE": "libelle",
            "NCCENR": "libelle_sans_article",
        }
    )[
        [
            "typecom",
            "code_insee",
            "code_commune",
            "libelle",
            "libelle_sans_article",
            "departement",
            "departement_libelle",
            "region",
        ]
    ]

    missing = ref[
        (ref["departement"] == "")
        | (ref["code_commune"] == "")
        | ref["departement_libelle"].isna()
    ]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} lignes du COG sans département ou commune parente."
        )

    ref = ref.drop_duplicates().sort_values(["code_commune", "typecom", "code_insee"])
    ref.to_csv(settings.path, index=False)

    logger.info(
        "Référentiel écrit : %s (%s)",
        settings.path,
        ref["typecom"].value_counts().to_dict(),
    )
    return ref


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()
