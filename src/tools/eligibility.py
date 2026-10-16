"""Outil `check_eligibility` : éligibilité à un logement social (calcul déterministe).

Le LLM n'effectue AUCUN calcul : il extrait la situation du ménage dans
`EligibilityInput`, puis ce module compare le revenu aux plafonds officiels.

Source des barèmes : arrêté du 29 juillet 1987 modifié par l'arrêté du
19 décembre 2025 (plafonds applicables au 1er janvier 2026), tableau ANIL
https://www.anil.org/aj-plafonds-ressources-2026-logement-social/
PLS = PLUS x 1,3 (arrondi à l'euro) : ce sont les montants affichés par la
fiche Service-Public F869 (« revenu maximum pour obtenir un logement social »).
Métropole uniquement : les plafonds outre-mer sont différents.
"""

from __future__ import annotations

import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Zone = Literal["paris_limitrophes", "idf_hors_paris", "autres_regions"]

# Montants annuels en euros : catégories 1 à 6, puis personne supplémentaire.
PLAFONDS: dict[int, dict[str, dict[Zone, list[int]]]] = {
    2026: {
        "plai": {
            "paris_limitrophes": [14811, 24140, 31643, 34637, 41203, 46369, 5166],
            "idf_hors_paris": [14811, 24140, 29018, 31860, 37719, 42444, 4727],
            "autres_regions": [12870, 18753, 22551, 25092, 29359, 33086, 3689],
        },
        "plus": {
            "paris_limitrophes": [26920, 40233, 52740, 62968, 74919, 84304, 9394],
            "idf_hors_paris": [26920, 40233, 48362, 57930, 68577, 77171, 8598],
            "autres_regions": [23403, 31254, 37584, 45374, 53376, 60156, 6710],
        },
    }
}
SOURCE_URL = "https://www.anil.org/aj-plafonds-ressources-2026-logement-social/"

# Communes limitrophes de Paris (liste de la fiche F869).
PARIS_LIMITROPHES = {
    "paris", "aubervilliers", "bagnolet", "boulogne-billancourt", "charenton-le-pont", "clichy",
    "fontenay-sous-bois", "gentilly", "issy-les-moulineaux", "ivry-sur-seine", "joinville-le-pont",
    "le kremlin-bicetre", "les lilas", "le pre-saint-gervais", "levallois-perret", "malakoff",
    "montreuil", "montrouge", "neuilly-sur-seine", "nogent-sur-marne", "pantin", "puteaux",
    "saint-cloud", "saint-denis", "saint-mande", "saint-maurice", "saint-ouen", "suresnes",
    "vanves", "vincennes",
}
IDF_DEPARTEMENTS = {"75", "77", "78", "91", "92", "93", "94", "95"}
OUTRE_MER_PREFIX = ("971", "972", "973", "974", "976", "97")
# Repli quand le département n'est pas fourni (à remplacer par la table commune -> département
# du fichier RPLS dès qu'il est chargé).
DEPARTEMENT_CONNU = {
    "creteil": "94", "vitry-sur-seine": "94", "argenteuil": "95", "versailles": "78",
    "nanterre": "92", "evry-courcouronnes": "91", "meaux": "77", "cergy": "95",
    "lyon": "69", "villeurbanne": "69", "marseille": "13", "toulouse": "31", "nice": "06",
    "nantes": "44", "strasbourg": "67", "montpellier": "34", "bordeaux": "33", "lille": "59",
    "rennes": "35", "reims": "51", "grenoble": "38", "dijon": "21", "angers": "49",
}


def normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    return " ".join(text.replace("’", "'").split())


class EligibilityInput(BaseModel):
    """Situation du ménage, extraite par le LLM puis VALIDÉE ici."""

    revenu_fiscal_reference: float = Field(ge=0, description="Total des RFR N-2 de toutes les personnes à loger, en euros.")
    nb_personnes: int = Field(ge=1, le=20, description="Nombre total de personnes à loger, demandeur inclus.")
    commune: str = Field(min_length=1, description="Commune du logement demandé.")
    departement: str | None = Field(default=None, description="Code département, ex. '94'.")
    en_couple: bool = False
    personnes_a_charge: int = Field(default=0, ge=0)
    age_demandeur: int | None = Field(default=None, ge=16, le=120)
    age_conjoint: int | None = Field(default=None, ge=16, le=120)
    handicap: bool = Field(default=False, description="Au moins une personne avec carte mobilité inclusion invalidité.")
    annee: int = Field(default=2026, description="Année de la demande (barème du 1er janvier).")

    @model_validator(mode="after")
    def coherent(self) -> "EligibilityInput":
        adults = 2 if self.en_couple else 1
        if self.personnes_a_charge > self.nb_personnes - 1:
            raise ValueError("personnes_a_charge incompatible avec nb_personnes")
        if self.en_couple and self.nb_personnes < adults:
            raise ValueError("un couple compte au moins 2 personnes")
        return self


class EligibilityResult(BaseModel):
    annee_bareme: int
    annee_revenus: int
    zone: Zone
    categorie: int
    jeune_menage: bool
    revenu: float
    plafonds: dict[str, int]
    eligible: dict[str, bool]
    verdict: str
    source: str = SOURCE_URL


class EligibilityError(ValueError):
    """Situation non couverte : l'agent doit demander une précision ou s'abstenir."""


def resolve_zone(commune: str, departement: str | None) -> Zone:
    name = normalize(commune)
    dep = (departement or DEPARTEMENT_CONNU.get(name, "")).strip().upper()
    if dep.startswith(OUTRE_MER_PREFIX) and len(dep) == 3:
        raise EligibilityError("Plafonds outre-mer non couverts par cet outil.")
    if name in PARIS_LIMITROPHES or name.startswith("paris "):
        return "paris_limitrophes"
    if not dep:
        raise EligibilityError(f"Département inconnu pour « {commune} » : préciser le département.")
    return "idf_hors_paris" if dep in IDF_DEPARTEMENTS else "autres_regions"


def household_category(data: EligibilityInput) -> tuple[int, bool]:
    """Catégorie de ménage (1 à 6, au-delà = 6 + personnes supplémentaires)."""
    ages = (data.age_demandeur, data.age_conjoint)
    jeune_menage = data.en_couple and None not in ages and sum(ages) <= 55
    n = data.nb_personnes
    if n == 1:
        category = 1
    elif not data.en_couple and data.personnes_a_charge >= 1 and n == 1 + data.personnes_a_charge:
        category = n + 1                      # personne seule avec personne(s) à charge
    elif jeune_menage and n == 2:
        category = 3                          # jeune ménage sans personne à charge
    else:
        category = n
    if data.handicap:
        category += 1                         # catégorie supérieure
    return category, jeune_menage


def ceiling(table: list[int], category: int) -> int:
    if category <= 6:
        return table[category - 1]
    return table[5] + (category - 6) * table[6]


def check_eligibility(data: EligibilityInput | dict) -> EligibilityResult:
    if isinstance(data, dict):
        data = EligibilityInput.model_validate(data)
    if data.annee not in PLAFONDS:
        raise EligibilityError(f"Barème {data.annee} non disponible (années connues : {sorted(PLAFONDS)}).")
    zone = resolve_zone(data.commune, data.departement)
    category, jeune_menage = household_category(data)
    bareme = PLAFONDS[data.annee]
    plus = ceiling(bareme["plus"][zone], category)
    plafonds = {
        "plai": ceiling(bareme["plai"][zone], category),
        "plus": plus,
        "pls": round(plus * 1.3),
    }
    eligible = {kind: data.revenu_fiscal_reference <= amount for kind, amount in plafonds.items()}
    if eligible["plai"]:
        verdict = "Éligible au PLAI (logement très social), donc aussi au PLUS et au PLS."
    elif eligible["plus"]:
        verdict = "Éligible au PLUS (logement social classique) et au PLS, pas au PLAI."
    elif eligible["pls"]:
        verdict = "Éligible uniquement au PLS (logement intermédiaire)."
    else:
        verdict = "Non éligible : revenus au-dessus du plafond PLS (sauf exceptions, ex. baisse de revenus d'au moins 10 %)."
    return EligibilityResult(
        annee_bareme=data.annee,
        annee_revenus=data.annee - 2,
        zone=zone,
        categorie=category,
        jeune_menage=jeune_menage,
        revenu=data.revenu_fiscal_reference,
        plafonds=plafonds,
        eligible=eligible,
        verdict=verdict,
    )


def flat_parameters(model: type[BaseModel]) -> dict:
    """Schéma JSON « à plat » : Ollama ne lit que `type`, `description` et `enum` par propriété,
    alors que Pydantic écrit les champs optionnels en `anyOf: [{type: X}, {type: null}]`."""
    schema = model.model_json_schema()
    properties = {}
    for name, prop in schema["properties"].items():
        kind = prop.get("type") or next(
            (option["type"] for option in prop.get("anyOf", []) if option.get("type") != "null"), "string"
        )
        properties[name] = {"type": kind, "description": prop.get("description", name.replace("_", " "))}
        if "enum" in prop:
            properties[name]["enum"] = prop["enum"]
    return {"type": "object", "properties": properties, "required": schema.get("required", [])}


# Description donnée au LLM (format « tools » d'Ollama / OpenAI).
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "check_eligibility",
        "description": (
            "Vérifie si un ménage peut obtenir un logement social (PLAI, PLUS, PLS) en comparant "
            "son revenu fiscal de référence aux plafonds officiels. N'utiliser que si le revenu, "
            "la commune et la composition du foyer sont connus ; sinon demander la précision."
        ),
        "parameters": flat_parameters(EligibilityInput),
    },
}
