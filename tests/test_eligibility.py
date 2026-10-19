"""Tests de check_eligibility : mêmes cas que le golden set (q030-q034, q040)."""

import json
from pathlib import Path

import pytest

from src.tools.eligibility import EligibilityError, check_eligibility

ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "q030": dict(revenu_fiscal_reference=28000, nb_personnes=3, commune="Lyon", en_couple=True, personnes_a_charge=1),
    "q031": dict(revenu_fiscal_reference=30000, nb_personnes=1, commune="Paris"),
    "q032": dict(revenu_fiscal_reference=12000, nb_personnes=1, commune="Bordeaux"),
    "q033": dict(revenu_fiscal_reference=45000, nb_personnes=2, commune="Marseille", en_couple=True, age_demandeur=40, age_conjoint=38),
    "q034": dict(revenu_fiscal_reference=50000, nb_personnes=2, commune="Créteil", en_couple=True, age_demandeur=25, age_conjoint=27),
    "q040": dict(revenu_fiscal_reference=90000, nb_personnes=1, commune="Lyon"),
}


def golden_expected() -> dict[str, dict]:
    rows = [json.loads(line) for line in (ROOT / "eval" / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: row["expected"] for row in rows if "expected" in row}


@pytest.mark.parametrize("qid", sorted(CASES))
def test_matches_golden(qid):
    expected = golden_expected()[qid]
    result = check_eligibility(CASES[qid])
    assert result.zone == expected["zone"]
    assert result.categorie == expected["categorie"]
    assert result.eligible == {"plai": expected["plai"], "plus": expected["plus"], "pls": expected["pls"]}


def test_pls_matches_service_public_f869():
    # Montants PLS affichés par la fiche F869 (Paris, 1 personne / province, 2 personnes).
    assert check_eligibility(CASES["q031"]).plafonds["pls"] == 34996
    province_couple = dict(revenu_fiscal_reference=1, nb_personnes=2, commune="Lyon", en_couple=True, age_demandeur=40, age_conjoint=40)
    assert check_eligibility(province_couple).plafonds["pls"] == 40630


def test_single_parent_and_handicap_move_up_one_category():
    single_parent = dict(revenu_fiscal_reference=1, nb_personnes=2, commune="Lyon", personnes_a_charge=1)
    assert check_eligibility(single_parent).categorie == 3
    assert check_eligibility({**CASES["q032"], "handicap": True}).categorie == 2


def test_more_than_six_people():
    big = dict(revenu_fiscal_reference=1, nb_personnes=8, commune="Lyon", en_couple=True, personnes_a_charge=6)
    assert check_eligibility(big).plafonds["plus"] == 60156 + 2 * 6710


def test_unknown_department_and_year_are_refused():
    with pytest.raises(EligibilityError):
        check_eligibility(dict(revenu_fiscal_reference=1, nb_personnes=1, commune="Trifouillis"))
    with pytest.raises(EligibilityError):
        check_eligibility({**CASES["q032"], "annee": 2019})
    with pytest.raises(EligibilityError):
        check_eligibility({**CASES["q032"], "departement": "974"})


def test_invalid_household_is_rejected():
    with pytest.raises(ValueError):
        check_eligibility(dict(revenu_fiscal_reference=1, nb_personnes=1, commune="Lyon", personnes_a_charge=2))
