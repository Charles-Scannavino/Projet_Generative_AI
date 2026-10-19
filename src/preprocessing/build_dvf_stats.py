"""Agrège DVF (France entière) en statistiques de prix par commune × année × type.

Entrée  : data/raw/dvf/dvf_<année>.csv.gz (src.ingestion.dvf)
Sortie  : data/processed/dvf_communes.csv (versionné)

Méthode (choix documentés, appliqués avec DuckDB directement sur le gzip) :
- seules les mutations `Vente` sont retenues ;
- une mutation peut couvrir plusieurs lots et répète sa `valeur_fonciere`
  sur chaque ligne : on ne garde que les mutations portant sur exactement
  UN logement (Maison ou Appartement), dépendances tolérées, sans local
  commercial, situées dans une seule commune ;
- prix au m² = valeur foncière / surface réelle bâtie, borné par
  configs/tools.yaml (dvf.prix_m2_min / prix_m2_max, dvf.surface_min_m2) ;
- les arrondissements de Paris, Lyon et Marseille sont agrégés à la fois
  à leur niveau et au niveau de la commune ; les codes de communes
  déléguées sont rattachés à leur commune nouvelle.

Le seuil minimal de ventes (dvf.min_ventes) est appliqué par l'outil, pas
ici : le fichier garde n_ventes pour permettre de changer le seuil sans
reconstruire.

    uv run python -m src.preprocessing.build_dvf_stats
"""

from __future__ import annotations

import logging

import duckdb

from src.settings import get_settings

logger = logging.getLogger(__name__)

LOGEMENTS = ("Maison", "Appartement")


def build() -> None:
    settings = get_settings()
    dvf = settings.tools.dvf
    con = duckdb.connect()

    con.execute(
        """
        CREATE TABLE ref AS
        SELECT * FROM read_csv(?, header = true, all_varchar = true)
        """,
        [str(settings.tools.communes.path)],
    )

    # Code DVF -> code(s) de sortie : la commune elle-même, plus la commune
    # parente pour un arrondissement ; une commune déléguée -> sa parente.
    con.execute(
        """
        CREATE TABLE code_map AS
        SELECT DISTINCT code_insee AS code_dvf, code_insee AS code_out
        FROM ref WHERE typecom IN ('COM', 'ARM')
        UNION
        SELECT DISTINCT code_insee, code_commune
        FROM ref WHERE typecom IN ('ARM', 'COMD', 'COMA')
        """
    )

    sales_queries = []

    for annee in dvf.annees:
        path = dvf.raw_dir / f"dvf_{annee}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} absent : lancer d'abord `uv run python -m src.ingestion.dvf`."
            )

        con.execute(
            f"""
            CREATE TABLE sales_{annee} AS
            WITH raw AS (
                SELECT
                    id_mutation,
                    TRY_CAST(valeur_fonciere AS DOUBLE) AS valeur,
                    code_commune,
                    type_local,
                    TRY_CAST(surface_reelle_bati AS DOUBLE) AS surface,
                    id_parcelle,
                    nombre_pieces_principales
                -- Format explicite : les guillemets n'apparaissent que tard dans
                -- le fichier et échappent à la détection automatique.
                FROM read_csv(?, header = true, all_varchar = true, delim = ',', quote = '"')
                WHERE nature_mutation = 'Vente'
            ),
            mutations AS (
                SELECT
                    id_mutation,
                    any_value(valeur) AS valeur,
                    count(DISTINCT code_commune) AS n_communes,
                    any_value(code_commune) AS code_commune,
                    count(DISTINCT CASE WHEN type_local IN {LOGEMENTS}
                        THEN concat_ws('|', id_parcelle, type_local, surface, nombre_pieces_principales)
                    END) AS n_logements,
                    count(*) FILTER (WHERE type_local LIKE 'Local%') AS n_locaux_activite,
                    any_value(type_local) FILTER (WHERE type_local IN {LOGEMENTS}) AS type_local,
                    max(surface) FILTER (WHERE type_local IN {LOGEMENTS}) AS surface
                FROM raw
                GROUP BY id_mutation
            )
            SELECT
                {annee} AS annee,
                code_commune,
                type_local,
                valeur,
                surface,
                valeur / surface AS prix_m2
            FROM mutations
            WHERE n_logements = 1
              AND n_locaux_activite = 0
              AND n_communes = 1
              AND valeur > 0
              AND surface >= ?
              AND valeur / surface BETWEEN ? AND ?
            """,
            [str(path), dvf.surface_min_m2, dvf.prix_m2_min, dvf.prix_m2_max],
        )

        n_sales = con.execute(f"SELECT count(*) FROM sales_{annee}").fetchone()[0]
        logger.info(
            "%s : %s ventes d'un logement unique retenues", annee, f"{n_sales:,}"
        )
        sales_queries.append(f"SELECT * FROM sales_{annee}")

    con.execute(f"CREATE TABLE sales AS {' UNION ALL '.join(sales_queries)}")

    unmapped = con.execute(
        """
        SELECT count(*), count(DISTINCT code_commune)
        FROM sales s ANTI JOIN code_map m ON s.code_commune = m.code_dvf
        """
    ).fetchone()
    logger.info(
        "Ventes dont le code commune est absent du COG (écartées) : %s (%s codes)",
        *unmapped,
    )

    stats = con.execute(
        """
        SELECT
            m.code_out AS code_insee,
            s.annee,
            coalesce(s.type_local, 'Tous') AS type_local,
            count(*) AS n_ventes,
            round(quantile_cont(s.prix_m2, 0.5)) AS prix_m2_median,
            round(quantile_cont(s.prix_m2, 0.25)) AS prix_m2_q1,
            round(quantile_cont(s.prix_m2, 0.75)) AS prix_m2_q3,
            round(quantile_cont(s.valeur, 0.5)) AS valeur_fonciere_mediane,
            round(quantile_cont(s.surface, 0.5), 1) AS surface_mediane_m2
        FROM sales s
        JOIN code_map m ON s.code_commune = m.code_dvf
        GROUP BY GROUPING SETS (
            (m.code_out, s.annee, s.type_local),
            (m.code_out, s.annee)
        )
        ORDER BY code_insee, annee, type_local
        """
    ).df()

    stats.to_csv(dvf.path, index=False)
    logger.info("Écrit : %s (%s lignes)", dvf.path, f"{len(stats):,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()
