"""Télécharge les fichiers DVF géolocalisés (France entière) des années configurées.

uv run python -m src.ingestion.dvf
"""

from __future__ import annotations

import logging

from src.ingestion.download import download
from src.settings import get_settings


def main() -> None:
    settings = get_settings().tools.dvf

    for annee in settings.annees:
        download(
            settings.download_url_template.format(annee=annee),
            settings.raw_dir / f"dvf_{annee}.csv.gz",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
