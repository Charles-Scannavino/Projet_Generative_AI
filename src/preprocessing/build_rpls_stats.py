"""Agrège le RPLS (un logement par ligne) en statistiques par commune.

Entrée  : data/raw/rpls/rpls_<millésime>.csv (src.ingestion.rpls)
Sortie  : data/processed/rpls_communes.csv (versionné)

Le fichier public ne contient ni loyer, ni vacance, ni occupation : on
agrège donc uniquement la composition du parc (financement initial, type
de construction, taille, surface, QPV, ancienneté, DPE).

Regroupement des financements initiaux (choix du projet, d'après les
libellés SDES) :
    PLAI  : 10 PLA d'intégration / LLTS, 11 PLA-LM / PLA-TS / PLA insertion
    PLUS  : 12 PLA ordinaire, 13 PLUS / LLS, 50 HBM, 52 HLM/O
    PLS   : 14 PLS / PPLS / PCLS / PLA CFF
    PLI   : 16 PLI
    autres: 15, 17, 49, 51, 53, 54, 55, 99 (financements anciens ou spécifiques)

Les arrondissements de Paris, Lyon et Marseille sont agrégés à leur niveau
et au niveau de la commune.

    uv run python -m src.preprocessing.build_rpls_stats
"""

from __future__ import annotations

import logging

import duckdb

from src.settings import get_settings

logger = logging.getLogger(__name__)

FINANCEMENT_GROUPS = {
    "plai": ("10", "11"),
    "plus": ("12", "13", "50", "52"),
    "pls": ("14",),
    "pli": ("16",),
}


def build() -> None:
    settings = get_settings()
    rpls = settings.tools.rpls

    if not rpls.raw_path.exists():
        raise FileNotFoundError(
            f"{rpls.raw_path} absent : lancer d'abord `uv run python -m src.ingestion.rpls`."
        )

    con = duckdb.connect()
    con.execute(
        "CREATE TABLE ref AS SELECT * FROM read_csv(?, header = true, all_varchar = true)",
        [str(settings.tools.communes.path)],
    )
    con.execute(
        """
        CREATE TABLE code_map AS
        SELECT DISTINCT code_insee AS code_rpls, code_insee AS code_out
        FROM ref WHERE typecom IN ('COM', 'ARM')
        UNION
        SELECT DISTINCT code_insee, code_commune
        FROM ref WHERE typecom IN ('ARM', 'COMD', 'COMA')
        """
    )

    known = tuple(code for codes in FINANCEMENT_GROUPS.values() for code in codes)
    financement_counts = ",\n".join(
        f"count(*) FILTER (WHERE r.FINAN_CODE IN {codes}) AS nb_{name}"
        for name, codes in FINANCEMENT_GROUPS.items()
    )

    con.execute(
        """
        CREATE TABLE logements AS
        SELECT * FROM read_csv(?, header = true, all_varchar = true, delim = ';', quote = '"')
        """,
        [str(rpls.raw_path)],
    )

    total, unmapped = con.execute(
        """
        SELECT
            count(*),
            count(*) FILTER (WHERE DEPCOM NOT IN (SELECT code_rpls FROM code_map))
        FROM logements
        """
    ).fetchone()
    logger.info(
        "Logements lus : %s ; code commune absent du COG (écartés) : %s",
        f"{total:,}",
        f"{unmapped:,}",
    )

    stats = con.execute(
        f"""
        SELECT
            m.code_out AS code_insee,
            count(*) AS nb_logements,
            {financement_counts},
            count(*) FILTER (WHERE r.FINAN_CODE NOT IN {known} OR r.FINAN_CODE IS NULL) AS nb_autres_financements,
            count(*) FILTER (WHERE r.TYPECONST_CODE = 'C') AS nb_collectif,
            count(*) FILTER (WHERE r.TYPECONST_CODE = 'I') AS nb_individuel,
            count(*) FILTER (WHERE r.TYPECONST_CODE = 'E') AS nb_etudiant,
            count(*) FILTER (WHERE TRY_CAST(r.NBPIECE AS INTEGER) = 1) AS nb_1_piece,
            count(*) FILTER (WHERE TRY_CAST(r.NBPIECE AS INTEGER) = 2) AS nb_2_pieces,
            count(*) FILTER (WHERE TRY_CAST(r.NBPIECE AS INTEGER) = 3) AS nb_3_pieces,
            count(*) FILTER (WHERE TRY_CAST(r.NBPIECE AS INTEGER) = 4) AS nb_4_pieces,
            count(*) FILTER (WHERE TRY_CAST(r.NBPIECE AS INTEGER) >= 5) AS nb_5_pieces_et_plus,
            round(avg(TRY_CAST(r.SURFHAB AS DOUBLE)), 1) AS surface_moyenne_m2,
            count(*) FILTER (WHERE r.QPV_CODE = '1') AS nb_en_qpv,
            CAST(quantile_disc(TRY_CAST(r.CONSTRUCT AS INTEGER), 0.5) AS INTEGER) AS annee_construction_mediane,
            count(*) FILTER (WHERE r.DPEENERGIE IN ('A','B','C','D','E','F','G')) AS nb_dpe_renseigne,
            count(*) FILTER (WHERE r.DPEENERGIE IN ('F','G')) AS nb_dpe_f_g
        FROM logements r
        JOIN code_map m ON r.DEPCOM = m.code_rpls
        GROUP BY m.code_out
        ORDER BY code_insee
        """
    ).df()

    stats.to_csv(rpls.path, index=False)
    logger.info(
        "Écrit : %s (%s communes, %s logements)",
        rpls.path,
        f"{len(stats):,}",
        f"{total - unmapped:,}",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()
