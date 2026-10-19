"""Résolution d'une commune (nom ou code Insee) dans le référentiel COG.

Le référentiel est une liste fermée : une chaîne qui ne s'y résout pas
n'atteint jamais les données des outils.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass

import pandas as pd

from src.settings import get_settings
from src.tools.errors import AmbiguousCommune, CommuneNotFound
from src.tools.schemas import CommuneInfo

_CODE_INSEE_RE = re.compile(r"^(\d{5}|2[AB]\d{3})$")

# Abréviations courantes, appliquées sur un nom déjà normalisé.
_ABBREVIATIONS = {
    "st": "saint",
    "ste": "sainte",
}


def normalize_name(name: str) -> str:
    """Forme canonique d'un nom de commune pour la recherche.

    « Saint-Ouen-sur-Seine », « st ouen sur seine » et « SAINT OUEN SUR
    SEINE » donnent la même clé.
    """
    text = unicodedata.normalize("NFKD", name)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("œ", "oe").replace("æ", "ae")
    text = re.sub(r"['’\-]", " ", text)
    words = text.split()

    return " ".join(_ABBREVIATIONS.get(word, word) for word in words)


def _mentions(text: str, candidates: set[str]) -> bool:
    """True si l'un des noms normalisés apparaît en mots entiers dans `text`.

    La ponctuation (« Saint-Denis, en Seine-Saint-Denis, … ») ne doit pas
    empêcher la correspondance : on ne garde que lettres et chiffres.
    """
    haystack = f" {re.sub(r'[^a-z0-9]+', ' ', normalize_name(text))} "
    return any(
        c and f" {re.sub(r'[^a-z0-9]+', ' ', c).strip()} " in haystack
        for c in candidates
    )


@dataclass(frozen=True)
class _Entry:
    info: CommuneInfo
    is_alias: bool  # commune déléguée ou associée


class CommuneResolver:
    def __init__(self, ref: pd.DataFrame) -> None:
        communes = ref[ref["typecom"] == "COM"].set_index("code_insee")

        self._by_code: dict[str, CommuneInfo] = {}
        self._by_name: dict[str, list[_Entry]] = {}
        self._departement_names: dict[str, str] = dict(
            zip(ref["departement"], ref["departement_libelle"], strict=True)
        )

        for row in ref.itertuples(index=False):
            parent = communes.loc[row.code_commune]
            is_alias = row.typecom in {"COMD", "COMA"}

            info = CommuneInfo(
                # Une commune déléguée n'a pas de données propres : on la
                # ramène à sa commune de rattachement.
                code_insee=row.code_commune if is_alias else row.code_insee,
                code_commune=row.code_commune,
                libelle=parent["libelle"] if is_alias else row.libelle,
                departement=row.departement,
                region=row.region,
            )

            if not is_alias:
                self._by_code[row.code_insee] = info

            entry = _Entry(info=info, is_alias=is_alias)
            keys = {
                normalize_name(row.libelle),
                normalize_name(row.libelle_sans_article),
            }

            # « Paris 15e Arrondissement » est aussi cherché comme « Paris 15e ».
            if row.typecom == "ARM":
                keys |= {key.removesuffix(" arrondissement") for key in keys}

            for key in keys:
                self._by_name.setdefault(key, []).append(entry)

    def is_ambiguous(self, commune: str) -> bool:
        """True si le nom désigne plusieurs communes en l'absence de département."""
        try:
            self.resolve(commune)
        except AmbiguousCommune:
            return True
        except CommuneNotFound:
            return False
        return False

    def departement_mentioned(self, departement: str, text: str) -> bool:
        """True si le code (« 93 ») ou le nom (« Seine-Saint-Denis ») figure dans `text`."""
        if re.search(rf"(?<![\dA-Za-z]){re.escape(departement)}(?![\dA-Za-z])", text):
            return True

        name = self._departement_names.get(departement)
        if name is None:
            return False

        normalized = normalize_name(name)
        # « La Réunion » doit aussi être reconnu dans « à Saint-Denis de Réunion ».
        without_article = re.sub(r"^(?:la|le|les|l) ", "", normalized)

        return _mentions(text, {normalized, without_article})

    def commune_mentioned(self, commune: str, text: str) -> bool:
        """True si la commune figure dans `text`, telle quelle ou sous son libellé officiel.

        « St Ouen » dans la question et « Saint-Ouen » proposé par le modèle
        correspondent ; « Paris » proposé pour une question qui ne cite aucune
        commune ne correspond pas.
        """
        if _CODE_INSEE_RE.match(commune):
            return commune in text

        candidates = {normalize_name(commune)}
        try:
            info, _ = self.resolve(commune)
            candidates.add(normalize_name(info.libelle))
        except (AmbiguousCommune, CommuneNotFound):
            pass
        candidates |= {c.removesuffix(" arrondissement") for c in candidates}

        return _mentions(text, candidates)

    def resolve(
        self,
        commune: str,
        departement: str | None = None,
    ) -> tuple[CommuneInfo, str | None]:
        """Renvoie la commune et, si elle a été trouvée via un alias, son ancien nom.

        Raises:
            CommuneNotFound: aucun nom ni code ne correspond.
            AmbiguousCommune: plusieurs communes distinctes correspondent.
        """
        if _CODE_INSEE_RE.match(commune):
            info = self._by_code.get(commune)
            if info is None:
                raise CommuneNotFound()
            return info, None

        entries = self._by_name.get(normalize_name(commune), [])

        if departement is not None:
            entries = [e for e in entries if e.info.departement == departement]

        # Une commune ou un arrondissement portant ce nom l'emporte sur une
        # commune déléguée homonyme (ex. Saint-Denis, COMD de 93066).
        direct = [e for e in entries if not e.is_alias]
        if direct:
            entries = direct

        distinct = {e.info.code_insee: e for e in entries}

        if not distinct:
            raise CommuneNotFound()

        if len(distinct) > 1:
            raise AmbiguousCommune(
                [
                    {
                        "code_insee": e.info.code_insee,
                        "libelle": e.info.libelle,
                        "departement": e.info.departement,
                    }
                    for e in sorted(distinct.values(), key=lambda e: e.info.code_insee)
                ]
            )

        entry = next(iter(distinct.values()))
        alias_of = commune.strip() if entry.is_alias else None

        return entry.info, alias_of


_resolver: CommuneResolver | None = None
_lock = threading.Lock()


def get_resolver() -> CommuneResolver:
    global _resolver

    with _lock:
        if _resolver is None:
            path = get_settings().tools.communes.path
            ref = pd.read_csv(path, dtype=str, keep_default_na=False)
            _resolver = CommuneResolver(ref)

        return _resolver


def resolve_commune(
    commune: str,
    departement: str | None = None,
) -> tuple[CommuneInfo, str | None]:
    return get_resolver().resolve(commune, departement)
