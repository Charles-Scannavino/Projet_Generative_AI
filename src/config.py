"""Chemins fixes du projet.

Seules des constantes immuables vivent ici. Les paramètres réglables
(millésimes, seuils, modèles…) sont dans configs/*.yaml et sont lus via
src.settings.get_settings().
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT_DIR / "configs"
PROMPTS_DIR = ROOT_DIR / "prompts"

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
