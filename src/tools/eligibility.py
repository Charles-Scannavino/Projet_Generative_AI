"""`check_eligibility` : comparaison du revenu d'un ménage aux plafonds HLM.

Le LLM n'effectue AUCUN calcul : il fournit la situation du ménage, ce module
compare le revenu aux plafonds officiels.

Barème : arrêté du 29 juillet 1987 modifié par l'arrêté du 19 décembre 2025
(plafonds au 1er janvier 2026, PLAI / PLUS / PLS), tableau ANIL ; PLS = PLUS x 1,3
tel qu'affiché par la fiche Service-Public F869. Zones : Paris et communes
limitrophes / reste de l'Île-de-France / autres régions métropolitaines. Les DOM
relèvent d'un autre régime et ne sont pas couverts. La commune est résolue dans
le COG (src/tools/communes.py), jamais devinée à partir du seul nom.

Catégories de ménage (arrêté du 29 juillet 1987) :
    1. une personne seule ;
    2. deux personnes sans personne à charge, hors jeune ménage, ou une
       personne seule en situation de handicap ;
    3. trois personnes, ou une personne seule avec une personne à charge,
       ou un jeune ménage sans personne à charge, ou deux personnes dont au
       moins une en situation de handicap ;
    4 à 6. n personnes, ou une personne seule avec n - 2 personnes à
       charge, ou n - 1 personnes dont au moins une en situation de handicap.
Au-delà de 6, chaque personne supplémentaire ajoute un montant fixe.

Chaque définition place un ménage de n personnes en catégorie n ou n + 1 :
elles ne se cumulent pas (une personne seule avec un enfant handicapé est
en catégorie 3, comme « deux personnes dont une handicapée »). Lecture
littérale des définitions ; à confirmer sur la version consolidée Légifrance.

Deux formats d'entrée :
- `EligibilityInput` : schéma strict exposé au LLM par l'agent à outils ;
- `HouseholdSituation` : situation « telle que décrite » (en couple, âges,
  handicap), utilisée par l'agent à formulaire JSON, l'UI et les tests.
  Un `dict` passé à `check_eligibility` est lu dans ce format.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from typing import Annotated, Literal

import pandas as pd
from pydantic import Field, model_validator

from src.settings import get_settings
from src.tools.communes import resolve_commune
from src.tools.errors import DataNotCovered, ToolError
from src.tools.schemas import (
    CommuneInfo,
    CommuneName,
    DataProvenance,
    Departement,
    StrictInput,
    ToolOutput,
)

Zone = Literal["paris_limitrophes", "idf_hors_paris", "autres_regions"]
Financement = Literal["plai", "plus", "pls"]

FINANCEMENTS: tuple[Financement, ...] = ("plai", "plus", "pls")

ZONE_LABELS: dict[Zone, str] = {
    "paris_limitrophes": "Paris et communes limitrophes",
    "idf_hors_paris": "Île-de-France hors Paris et communes limitrophes",
    "autres_regions": "autres régions métropolitaines",
}

PARIS_CODE_COMMUNE = "75056"
MAX_CATEGORIE_TABLE = 6
JEUNE_MENAGE_AGE_MAX = 55

# Situation non couverte (commune inconnue ou ambiguë, outre-mer, année) :
# l'appelant doit demander une précision ou s'abstenir. Toutes les erreurs
# d'outil sont des ValueError.
EligibilityError = ToolError


# ============================================================
# ENTRÉES
# ============================================================


class EligibilityInput(StrictInput):
    revenu_fiscal_reference: Annotated[float, Field(ge=0, le=10_000_000)] = Field(
        description=(
            "Somme des revenus fiscaux de référence de toutes les personnes à "
            "loger, en euros, pour l'année N-2 (2024 pour une demande en 2026)."
        ),
    )
    nb_personnes: Annotated[int, Field(ge=1, le=20)] = Field(
        description="Nombre total de personnes à loger, demandeur compris.",
    )
    nb_personnes_a_charge: Annotated[int, Field(ge=0, le=19)] = Field(
        default=0,
        description="Nombre de personnes à charge parmi les personnes à loger.",
    )
    jeune_menage: bool = Field(
        default=False,
        description=(
            "Couple (marié, pacsé ou concubin) sans personne à charge dont la "
            "somme des âges est d'au plus 55 ans."
        ),
    )
    personne_handicapee: bool = Field(
        default=False,
        description=(
            "Au moins une personne à loger titulaire de la carte mobilité "
            "inclusion mention invalidité."
        ),
    )
    commune: CommuneName | None = Field(
        default=None,
        description="Commune du logement demandé (nom ou code Insee).",
    )
    departement: Departement | None = Field(
        default=None,
        description=(
            "Département (ex. \"93\"), uniquement s'il est donné par l'utilisateur ; "
            "sert à distinguer des communes homonymes."
        ),
    )
    zone: Zone | None = Field(
        default=None,
        description="Zone de plafonds, si la commune n'est pas connue.",
    )
    annee: int = Field(default=2026, description="Année de la demande.")

    @model_validator(mode="after")
    def _coherence(self) -> EligibilityInput:
        if (self.commune is None) == (self.zone is None):
            raise ValueError(
                "Fournir soit `commune`, soit `zone` (exactement un des deux)."
            )

        if self.nb_personnes_a_charge > self.nb_personnes - 1:
            raise ValueError(
                "`nb_personnes_a_charge` doit être inférieur à `nb_personnes`."
            )

        return self


class HouseholdSituation(StrictInput):
    """Situation du ménage telle que décrite par l'utilisateur.

    C'est le schéma exposé au LLM par l'agent à outils, et celui rempli par l'agent
    à formulaire JSON et l'UI : le modèle recopie des faits (en couple, âges,
    personnes à charge), Python en déduit la catégorie et le « jeune ménage ».
    Strict : champs inconnus refusés, aucune conversion implicite.
    """

    revenu_fiscal_reference: float = Field(
        ge=0,
        le=10_000_000,
        description="Total des RFR N-2 de toutes les personnes à loger, en euros.",
    )
    nb_personnes: int = Field(
        ge=1, le=20, description="Nombre total de personnes à loger, demandeur inclus."
    )
    commune: CommuneName = Field(
        description="Commune du logement demandé (nom ou code Insee)."
    )
    departement: Departement | None = Field(
        default=None,
        description=(
            "Département (ex. \"93\"), uniquement s'il est donné par l'utilisateur ; "
            "sert à distinguer des communes homonymes."
        ),
    )
    en_couple: bool = Field(
        default=False,
        description="Demandeur marié, pacsé ou en concubinage avec un codemandeur.",
    )
    personnes_a_charge: int = Field(
        default=0,
        ge=0,
        description="Enfants ou autres personnes à charge parmi les personnes à loger.",
    )
    age_demandeur: int | None = Field(
        default=None, ge=16, le=120, description="Âge du demandeur, s'il est indiqué."
    )
    age_conjoint: int | None = Field(
        default=None, ge=16, le=120, description="Âge du conjoint, s'il est indiqué."
    )
    handicap: bool = Field(
        default=False,
        description="Au moins une personne avec carte mobilité inclusion invalidité.",
    )
    annee: int = Field(
        default=2026, description="Année de la demande (barème du 1er janvier)."
    )

    @model_validator(mode="after")
    def _coherence(self) -> HouseholdSituation:
        if self.personnes_a_charge > self.nb_personnes - 1:
            raise ValueError("personnes_a_charge incompatible avec nb_personnes")
        if self.en_couple and self.nb_personnes < 2:
            raise ValueError("un couple compte au moins 2 personnes")
        return self

    def to_input(self) -> EligibilityInput:
        ages = (self.age_demandeur, self.age_conjoint)
        jeune_menage = (
            self.en_couple and None not in ages and sum(ages) <= JEUNE_MENAGE_AGE_MAX
        )
        # Dans un couple, deux adultes ne sont pas à charge : on ne peut pas
        # relever de « personne seule avec personnes à charge ».
        a_charge = (
            min(self.personnes_a_charge, self.nb_personnes - 2)
            if self.en_couple
            else self.personnes_a_charge
        )
        return EligibilityInput(
            revenu_fiscal_reference=float(self.revenu_fiscal_reference),
            nb_personnes=self.nb_personnes,
            nb_personnes_a_charge=a_charge,
            jeune_menage=jeune_menage,
            personne_handicapee=self.handicap,
            commune=self.commune,
            departement=self.departement or None,
            annee=self.annee,
        )


# ============================================================
# SORTIE
# ============================================================


class EligibilityResult(ToolOutput):
    categorie: int = Field(
        description="Catégorie de ménage (au-delà de 6 : 6 + personnes supplémentaires)."
    )
    justification_categorie: str
    jeune_menage: bool = Field(
        description="Le critère « jeune ménage » a été appliqué."
    )
    zone: Zone
    justification_zone: str
    commune: CommuneInfo | None
    revenu_fiscal_reference: float
    annee_bareme: int
    annee_revenus: int
    plafonds: dict[Financement, int] = Field(
        description="Plafond annuel par financement, en euros."
    )
    eligible: dict[Financement, bool]
    ecarts_eur: dict[Financement, float] = Field(
        description="Plafond moins revenu : positif = marge, négatif = dépassement.",
    )
    eligible_logement_social: bool = Field(
        description="Revenu sous au moins un plafond (le PLS étant le plus élevé).",
    )
    verdict: str


# ============================================================
# LOGIQUE (fonctions pures)
# ============================================================


def categorie_menage(
    nb_personnes: int,
    nb_personnes_a_charge: int = 0,
    jeune_menage: bool = False,
    personne_handicapee: bool = False,
) -> tuple[int, str]:
    """Catégorie de ménage au sens de l'arrêté du 29 juillet 1987, avec sa justification."""
    n = nb_personnes
    reasons = []

    if n >= 2 and nb_personnes_a_charge == n - 1:
        reasons.append(f"personne seule avec {n - 1} personne(s) à charge")

    if n == 2 and nb_personnes_a_charge == 0 and jeune_menage:
        reasons.append("jeune ménage sans personne à charge")

    if personne_handicapee:
        reasons.append(
            "au moins une personne titulaire de la carte mobilité inclusion invalidité"
        )

    if reasons:
        categorie = n + 1
        justification = (
            f"{n} personne(s) à loger, "
            + " ; ".join(reasons)
            + f" : catégorie {categorie}"
        )
    else:
        categorie = n
        justification = f"{n} personne(s) à loger : catégorie {categorie}"

    if jeune_menage and not (n == 2 and nb_personnes_a_charge == 0):
        justification += " (le critère « jeune ménage » ne s'applique qu'à un couple sans personne à charge)"

    return categorie, justification


@lru_cache(maxsize=1)
def _limitrophes_codes() -> frozenset[str]:
    codes = set()

    for ref in get_settings().tools.plafonds.communes_limitrophes:
        info, _ = resolve_commune(ref.nom, ref.departement)
        codes.add(info.code_commune)

    return frozenset(codes)


def zone_from_commune(info: CommuneInfo) -> tuple[Zone, str]:
    settings = get_settings().tools.plafonds

    if info.departement in settings.departements_non_couverts:
        raise DataNotCovered(
            "Les plafonds de l'arrêté du 29 juillet 1987 ne s'appliquent pas aux "
            "départements d'outre-mer (régime distinct). Consulter la fiche "
            "Service-Public F869 ou l'ADIL du département."
        )

    if info.code_commune == PARIS_CODE_COMMUNE:
        return "paris_limitrophes", f"{info.libelle} : Paris"

    if info.code_commune in _limitrophes_codes():
        return "paris_limitrophes", f"{info.libelle} : commune limitrophe de Paris"

    if info.departement in settings.departements_idf_hors_paris:
        return "idf_hors_paris", (
            f"{info.libelle} ({info.departement}) : Île-de-France, "
            "hors Paris et communes limitrophes"
        )

    return "autres_regions", (
        f"{info.libelle} ({info.departement}) : hors Île-de-France"
    )


@lru_cache(maxsize=1)
def _load_table() -> pd.DataFrame:
    settings = get_settings().tools.plafonds
    table = pd.read_csv(settings.path, dtype={"categorie": str})
    table = table[table["annee"] == settings.annee]

    expected = len(ZONE_LABELS) * len(FINANCEMENTS) * (MAX_CATEGORIE_TABLE + 1)
    if len(table) != expected:
        raise ValueError(
            f"Barème {settings.annee} incomplet : {len(table)} lignes, {expected} attendues."
        )

    return table.set_index(["zone", "financement", "categorie"])


def plafond(zone: Zone, financement: Financement, categorie: int) -> int:
    table = _load_table()["plafond_eur"]
    base = min(categorie, MAX_CATEGORIE_TABLE)
    value = int(table.loc[(zone, financement, str(base))])

    if categorie > MAX_CATEGORIE_TABLE:
        supplement = int(table.loc[(zone, financement, "supp")])
        value += (categorie - MAX_CATEGORIE_TABLE) * supplement

    return value


def verdict(eligible: dict[Financement, bool]) -> str:
    if eligible["plai"]:
        return "Éligible au PLAI (logement très social), donc aussi au PLUS et au PLS."
    if eligible["plus"]:
        return "Éligible au PLUS (logement social classique) et au PLS, pas au PLAI."
    if eligible["pls"]:
        return "Éligible uniquement au PLS (logement intermédiaire)."
    return (
        "Non éligible : revenus au-dessus du plafond PLS (sauf exceptions, "
        "ex. baisse de revenus d'au moins 10 %)."
    )


# ============================================================
# OUTIL
# ============================================================


def check_eligibility(
    request: EligibilityInput | HouseholdSituation | dict,
) -> EligibilityResult:
    """Compare le revenu fiscal de référence du ménage aux plafonds PLAI / PLUS / PLS.

    Un `dict` est lu comme une `HouseholdSituation`.

    Raises:
        pydantic.ValidationError: situation incohérente (ValueError).
        EligibilityError (CommuneNotFound, AmbiguousCommune, DataNotCovered) :
            commune non résolue, outre-mer ou année hors barème.
    """
    if isinstance(request, dict):
        request = HouseholdSituation.model_validate(request)
    if isinstance(request, HouseholdSituation):
        request = request.to_input()

    settings = get_settings().tools.plafonds

    if request.annee != settings.annee:
        raise DataNotCovered(
            f"Seul le barème {settings.annee} est disponible (revenus "
            f"{settings.annee_revenus})."
        )

    caveats = [
        (
            "Être sous un plafond rend éligible, mais ne garantit pas l'attribution "
            "d'un logement : la commission d'attribution examine chaque dossier."
        ),
        "Une demande de logement social peut être déposée quel que soit le revenu.",
        (
            f"Revenu à comparer : somme des revenus fiscaux de référence "
            f"{settings.annee_revenus} de toutes les personnes à loger (avis "
            f"d'imposition {settings.annee_revenus + 1}). Si les revenus ont baissé "
            "d'au moins 10 %, ceux de l'année N-1 ou des 12 derniers mois peuvent "
            "être retenus."
        ),
        "Plafonds PLS : Service-Public F869 (130 % du PLUS).",
    ]

    commune = None

    if request.commune is not None:
        commune, alias_of = resolve_commune(request.commune, request.departement)
        zone, justification_zone = zone_from_commune(commune)

        if alias_of is not None:
            caveats.append(
                f"« {alias_of} » est une commune déléguée ou associée de "
                f"{commune.libelle} : la zone retenue est celle de {commune.libelle}. "
                "À confirmer auprès du bailleur ou de l'ADIL."
            )
    else:
        zone = request.zone
        justification_zone = f"zone fournie : {ZONE_LABELS[zone]}"

    categorie, justification_categorie = categorie_menage(
        request.nb_personnes,
        request.nb_personnes_a_charge,
        request.jeune_menage,
        request.personne_handicapee,
    )

    revenu = request.revenu_fiscal_reference
    plafonds = {f: plafond(zone, f, categorie) for f in FINANCEMENTS}
    eligible = {f: revenu <= montant for f, montant in plafonds.items()}

    return EligibilityResult(
        categorie=categorie,
        justification_categorie=justification_categorie,
        jeune_menage=(
            request.jeune_menage
            and request.nb_personnes == 2
            and request.nb_personnes_a_charge == 0
        ),
        zone=zone,
        justification_zone=justification_zone,
        commune=commune,
        revenu_fiscal_reference=revenu,
        annee_bareme=settings.annee,
        annee_revenus=settings.annee_revenus,
        plafonds=plafonds,
        eligible=eligible,
        ecarts_eur={f: round(montant - revenu, 2) for f, montant in plafonds.items()},
        eligible_logement_social=any(eligible.values()),
        verdict=verdict(eligible),
        provenance=DataProvenance(
            source_name=settings.source_name,
            source_url=settings.source_url,
            data_year=settings.annee,
        ),
        caveats=caveats,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Démonstration de check_eligibility.")
    parser.add_argument(
        "json",
        help='Entrée JSON, ex. \'{"revenu_fiscal_reference": 30000, "nb_personnes": 3, "commune": "Créteil"}\'',
    )
    args = parser.parse_args()

    result = check_eligibility(EligibilityInput.model_validate_json(args.json))
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
