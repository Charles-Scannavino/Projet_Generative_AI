"""Télécharge le RPLS (données détaillées au logement, SDES) en ne gardant que
les colonnes utiles, via l'API DiDo.

    uv run python -m src.ingestion.rpls
"""

from __future__ import annotations

import logging
from urllib.parse import urlencode

from src.ingestion.download import download
from src.settings import get_settings


def main() -> None:
    settings = get_settings().tools.rpls
    query = urlencode(
        {
            "millesime": settings.dido_millesime,
            "withColumnName": "true",
            "withColumnDescription": "false",
            "columns": ",".join(settings.columns),
        }
    )
    download(f"{settings.download_url}?{query}", settings.raw_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
