"""Tests des outils déterministes (lancer : uv run pytest -q tests/test_tools.py)."""

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
import pytest

from src.settings import get_settings
from src.tools.communes import normalize_name, resolve_commune
from src.tools.eligibility import (
    EligibilityError,
    EligibilityInput,
    HouseholdSituation,
    categorie_menage,
    check_eligibility,
    plafond,
)
from src.tools.errors import AmbiguousCommune, CommuneNotFound, DataNotCovered
from src.tools.market_prices import MarketPricesInput, market_prices
from src.tools.social_housing import SocialHousingInput, social_housing_stats

# ============================================================
# COMMUNES
# ============================================================


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Saint-Ouen-sur-Seine", "saint ouen sur seine"),
        ("st ouen sur seine", "saint ouen sur seine"),
        ("L’Haÿ-les-Roses", "l hay les roses"),
        ("ŒUILLY", "oeuilly"),
    ],
)
def test_normalize_name(name, expected):
    assert normalize_name(name) == expected


@pytest.mark.parametrize(
    ("commune", "departement", "code_insee"),
    [
        ("Créteil", None, "94028"),
        ("creteil", None, "94028"),
        ("Saint-Denis", "93", "93066"),
        ("Paris", None, "75056"),
        ("Paris 15e", None, "75115"),
        ("75115", None, "75115"),
        ("Abergement-Clémenciat", None, "01001"),
    ],
)
def test_resolution_commune(commune, departement, code_insee):
    info, _ = resolve_commune(commune, departement)
    assert info.code_insee == code_insee


def test_commune_homonyme_ambigue():
    with pytest.raises(AmbiguousCommune) as error:
        resolve_commune("Saint-Denis")

    departements = {c["departement"] for c in error.value.candidates}
    assert {"93", "974"} <= departements


def test_commune_deleguee_rattachee_a_sa_commune_nouvelle():
    info, alias_of = resolve_commune("Pierrefitte-sur-Seine")

    assert info.code_insee == "93066"
    assert alias_of == "Pierrefitte-sur-Seine"


@pytest.mark.parametrize("commune", ["Xyzville", "99999"])
def test_commune_introuvable(commune):
    with pytest.raises(CommuneNotFound):
        resolve_commune(commune)


# ============================================================
# ÉLIGIBILITÉ
# ============================================================


@pytest.mark.parametrize(
    ("args", "categorie"),
    [
        ((1, 0, False, False), 1),
        ((1, 0, False, True), 2),  # personne seule handicapée
        ((2, 0, False, False), 2),
        ((2, 0, True, False), 3),  # jeune ménage
        ((2, 1, False, False), 3),  # personne seule + 1 à charge
        ((2, 0, False, True), 3),  # 2 personnes dont 1 handicapée
        ((2, 1, False, True), 3),  # définitions non cumulatives
        ((3, 1, False, False), 3),  # couple + 1 enfant
        ((3, 2, False, False), 4),  # personne seule + 2 à charge
        ((4, 2, False, False), 4),
        ((5, 4, False, False), 6),  # personne seule + 4 à charge
        ((6, 4, False, True), 7),
        ((8, 6, False, False), 8),
    ],
)
def test_categorie_menage(args, categorie):
    assert categorie_menage(*args)[0] == categorie


def test_jeune_menage_ignore_si_personnes_a_charge():
    categorie, justification = categorie_menage(3, 1, True, False)

    assert categorie == 3
    assert "ne s'applique" in justification


def test_plafond_personne_supplementaire():
    # Autres régions, PLAI : 33 086 € (catégorie 6) + 2 × 3 689 €.
    assert plafond("autres_regions", "plai", 8) == 33_086 + 2 * 3_689


def test_pls_egal_130_pct_du_plus():
    table = pd.read_csv(get_settings().tools.plafonds.path, dtype={"categorie": str})
    pivot = table.pivot_table(
        index=["zone", "categorie"], columns="financement", values="plafond_eur"
    )

    for (zone, categorie), row in pivot.iterrows():
        expected = int(Decimal(str(row["plus"] * 1.3)).quantize(0, ROUND_HALF_UP))
        assert row["pls"] == expected, (zone, categorie)


def test_plafonds_pls_presents_dans_F869_du_corpus():
    """Les plafonds PLS saisis doivent figurer dans la fiche F869 du corpus RAG."""
    with open(
        get_settings().tools.plafonds.path.parents[0]
        / "service_public_hlm_documents.jsonl",
        encoding="utf-8",
    ) as f:
        f869 = next(
            json.loads(line)["text"]
            for line in f
            if json.loads(line)["document_id"] == "F869"
        )

    numbers_in_f869 = {
        int(re.sub(r"\s", "", match))
        for match in re.findall(r"\d{1,3}(?:[\s ]\d{3})+", f869)
    }

    table = pd.read_csv(get_settings().tools.plafonds.path)
    pls_rows = table.loc[table["financement"] == "pls", "plafond_eur"]
    pls = set(pls_rows)

    # 3 zones x (6 catégories + personne supplémentaire) ; garde-fou contre un
    # filtre vide (codes de financement mal orthographiés) qui ferait passer le test.
    assert len(pls_rows) == 21
    assert pls <= numbers_in_f869, sorted(pls - numbers_in_f869)


@pytest.mark.parametrize(
    ("commune", "departement", "zone"),
    [
        ("Paris", None, "paris_limitrophes"),
        ("Paris 15e", None, "paris_limitrophes"),
        ("Montreuil", "93", "paris_limitrophes"),
        ("Saint-Ouen-sur-Seine", None, "paris_limitrophes"),
        ("Créteil", None, "idf_hors_paris"),
        ("Lyon", None, "autres_regions"),
    ],
)
def test_zone(commune, departement, zone):
    result = check_eligibility(
        EligibilityInput(
            revenu_fiscal_reference=20_000.0,
            nb_personnes=1,
            commune=commune,
            departement=departement,
        )
    )
    assert result.zone == zone


def test_limite_exacte_eligible():
    # Créteil, 3 personnes : plafond PLUS = 48 362 €.
    result = check_eligibility(
        EligibilityInput(
            revenu_fiscal_reference=48_362.0, nb_personnes=3, commune="Créteil"
        )
    )
    assert result.plafonds["plus"] == 48_362
    assert result.eligible["plus"]
    assert result.ecarts_eur["plus"] == 0


def test_au_dessus_de_tous_les_plafonds():
    result = check_eligibility(
        EligibilityInput(
            revenu_fiscal_reference=200_000.0, nb_personnes=1, zone="autres_regions"
        )
    )

    assert not result.eligible_logement_social
    assert all(ecart < 0 for ecart in result.ecarts_eur.values())
    assert result.verdict.startswith("Non éligible")


def test_dom_non_couvert():
    with pytest.raises(DataNotCovered):
        check_eligibility(
            EligibilityInput(
                revenu_fiscal_reference=20_000.0,
                nb_personnes=1,
                commune="Fort-de-France",
            )
        )


def test_annee_hors_bareme():
    with pytest.raises(DataNotCovered):
        check_eligibility(
            EligibilityInput(
                revenu_fiscal_reference=20_000.0,
                nb_personnes=1,
                zone="autres_regions",
                annee=2025,
            )
        )


def test_commune_deleguee_signalee():
    result = check_eligibility(
        EligibilityInput(
            revenu_fiscal_reference=20_000.0,
            nb_personnes=1,
            commune="Pierrefitte-sur-Seine",
        )
    )

    assert result.zone == "paris_limitrophes"
    assert any("Pierrefitte-sur-Seine" in caveat for caveat in result.caveats)


def test_sortie_eligibilite_tracable():
    result = check_eligibility(
        EligibilityInput(
            revenu_fiscal_reference=20_000.0, nb_personnes=1, zone="autres_regions"
        )
    )

    assert result.annee_bareme == 2026
    assert result.annee_revenus == 2024
    assert result.provenance.source_url.startswith("https://")
    assert any("ne garantit pas" in caveat for caveat in result.caveats)


# Format « ménage » (agent à formulaire JSON, UI) et régressions trouvées en
# comparant avec l'implémentation précédente, qui classait la zone sur le seul nom.


@pytest.mark.parametrize(
    ("situation", "zone", "categorie"),
    [
        ({"commune": "Saint-Denis", "departement": "11"}, "autres_regions", 1),
        ({"commune": "Saint-Denis", "departement": "93"}, "paris_limitrophes", 1),
        ({"commune": "Montreuil-sur-Mer"}, "autres_regions", 1),
        ({"commune": "Quimper"}, "autres_regions", 1),
        ({"commune": "Saint-Ouen-sur-Seine"}, "paris_limitrophes", 1),
    ],
)
def test_zone_resolue_par_le_cog(situation, zone, categorie):
    result = check_eligibility(
        {"revenu_fiscal_reference": 20_000, "nb_personnes": 1, **situation}
    )
    assert (result.zone, result.categorie) == (zone, categorie)


@pytest.mark.parametrize("commune", ["Saint-Denis", "Montreuil"])
def test_homonyme_sans_departement_refuse(commune):
    with pytest.raises(EligibilityError) as error:
        check_eligibility(
            {"revenu_fiscal_reference": 20_000, "nb_personnes": 1, "commune": commune}
        )
    assert error.value.code == "ambiguous_commune"


def test_menage_jeune_couple_selon_les_ages():
    young = HouseholdSituation(
        revenu_fiscal_reference=1,
        nb_personnes=2,
        commune="Lyon",
        en_couple=True,
        age_demandeur=25,
        age_conjoint=27,
    ).to_input()
    older = young.model_copy(update={"jeune_menage": False})

    assert young.jeune_menage
    assert check_eligibility(young).categorie == 3
    assert check_eligibility(older).categorie == 2


def test_couple_jamais_classe_personne_seule_avec_charges():
    # Couple déclarant (à tort) 2 personnes à charge sur 3 : pas de catégorie 4.
    result = check_eligibility(
        {
            "revenu_fiscal_reference": 1,
            "nb_personnes": 3,
            "commune": "Lyon",
            "en_couple": True,
            "personnes_a_charge": 2,
        }
    )
    assert result.categorie == 3


def test_erreurs_d_outil_sont_des_value_error():
    assert issubclass(EligibilityError, ValueError)


# ============================================================
# DVF
# ============================================================


def test_prix_lyon():
    result = market_prices(
        MarketPricesInput(commune="Lyon", type_local="Appartement", annee=2025)
    )
    (stats,) = result.statistiques

    assert stats.n_ventes > 1000
    assert 2_000 < stats.prix_m2_median < 10_000
    assert stats.prix_m2_q1 <= stats.prix_m2_median <= stats.prix_m2_q3
    assert any("loyers" in caveat for caveat in result.caveats)


def test_paris_agrege_les_arrondissements():
    paris = market_prices(MarketPricesInput(commune="Paris", annee=2025)).statistiques[
        0
    ]
    paris_15 = market_prices(
        MarketPricesInput(commune="Paris 15e", annee=2025)
    ).statistiques[0]

    assert paris.n_ventes > paris_15.n_ventes > 0


def test_alsace_moselle_non_couverte():
    with pytest.raises(DataNotCovered):
        market_prices(MarketPricesInput(commune="Strasbourg"))


def test_annee_dvf_indisponible():
    with pytest.raises(DataNotCovered):
        market_prices(MarketPricesInput(commune="Lyon", annee=2019))


def test_petit_echantillon_masque():
    settings = get_settings().tools.dvf
    stats = pd.read_csv(settings.path, dtype={"code_insee": str})
    small = stats[
        (stats["n_ventes"] < settings.min_ventes) & (stats["type_local"] == "Tous")
    ].iloc[0]

    result = market_prices(
        MarketPricesInput(commune=small["code_insee"], annee=int(small["annee"]))
    )
    (row,) = result.statistiques

    assert row.statistiques_masquees
    assert row.prix_m2_median is None


# ============================================================
# RPLS
# ============================================================


def test_parc_social_creteil():
    result = social_housing_stats(SocialHousingInput(commune="Créteil"))
    f = result.financement_initial

    assert result.nb_logements > 10_000
    assert f.plai + f.plus + f.pls + f.pli + f.autres == result.nb_logements
    assert result.millesime == 2025
    assert any("disponibles aujourd'hui" in caveat for caveat in result.caveats)


def test_commune_sans_logement_social():
    stats = pd.read_csv(get_settings().tools.rpls.path, dtype={"code_insee": str})
    ref = pd.read_csv(get_settings().tools.communes.path, dtype=str)
    code = next(
        c
        for c in ref.loc[ref["typecom"] == "COM", "code_insee"]
        if c not in set(stats["code_insee"]) and not c.startswith("97")
    )

    result = social_housing_stats(SocialHousingInput(commune=code))

    assert result.nb_logements == 0
    assert result.financement_initial is None
