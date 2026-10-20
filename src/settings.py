"""Réglages typés du projet, chargés depuis configs/*.yaml, .env et l'environnement.

Priorité (de la plus forte à la plus faible) :
    arguments explicites > variables d'environnement > .env > YAML

Les variables d'environnement utilisent le préfixe HLM_ et `__` pour
descendre dans l'arborescence : HLM_TOOLS__DVF__MIN_VENTES=10.

Un YAML invalide lève une ValidationError au premier appel de get_settings().
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from src.config import CONFIG_DIR, ROOT_DIR


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ============================================================
# OUTILS DÉTERMINISTES
# ============================================================


class CommunesSettings(_Section):
    path: Path
    millesime: int
    source_name: str
    source_url: str
    departements_url: str

    _abs_path = field_validator("path")(_resolve)


class CommuneRef(_Section):
    nom: str
    departement: str


class PlafondsSettings(_Section):
    annee: int
    annee_revenus: int
    path: Path
    source_name: str
    source_url: str
    pls_source_url: str
    communes_limitrophes: list[CommuneRef]
    departements_idf_hors_paris: list[str]
    departements_non_couverts: list[str]

    _abs_path = field_validator("path")(_resolve)


class RplsSettings(_Section):
    millesime: int
    path: Path
    raw_path: Path
    source_name: str
    source_url: str
    download_url: str
    dido_millesime: str
    columns: list[str]

    _abs_paths = field_validator("path", "raw_path")(_resolve)


class DvfSettings(_Section):
    annees: list[int]
    path: Path
    raw_dir: Path
    source_name: str
    source_url: str
    download_url_template: str
    min_ventes: int
    surface_min_m2: float
    prix_m2_min: float
    prix_m2_max: float
    departements_non_couverts: list[str]

    _abs_paths = field_validator("path", "raw_dir")(_resolve)


class ToolsSettings(_Section):
    communes: CommunesSettings
    plafonds: PlafondsSettings
    rpls: RplsSettings
    dvf: DvfSettings


# ============================================================
# AGENT
# ============================================================

Backend = Literal["openrouter", "ollama", "anthropic"]


class BackendSettings(_Section):
    model: str
    # OpenRouter uniquement : Ollama utilise OLLAMA_HOST / OLLAMA_NUM_CTX (.env).
    base_url: str | None = None
    tool_choice: Literal["required", "auto"] | None = None
    app_name: str | None = None
    # Anthropic uniquement : effort explicite (Opus 5.5 : thinking toujours actif,
    # effort par défaut `medium`) et plafond de tokens générés.
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    max_tokens: int | None = None


class Price(_Section):
    input: float
    output: float


class AgentSettings(_Section):
    backend: Backend
    temperature: float
    seed: int
    max_steps: int
    max_calls_per_tool: int
    max_chunk_chars: int
    search_k: int
    request_timeout_s: float
    backends: dict[Backend, BackendSettings]
    prices_usd_per_mtok: dict[str, Price]


# ============================================================
# RACINE
# ============================================================

YAML_FILES = [
    CONFIG_DIR / "tools.yaml",
    CONFIG_DIR / "agent.yaml",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HLM_",
        env_nested_delimiter="__",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        yaml_file=YAML_FILES,
        yaml_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    tools: ToolsSettings
    agent: AgentSettings

    # Secret : lu uniquement depuis l'environnement ou .env, jamais depuis le YAML.
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENROUTER_API_KEY",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Réglages du projet, chargés et validés une seule fois par processus."""
    return Settings()
