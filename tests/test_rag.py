"""Backend « rag » (module de Charles adapté au contrat commun), sans réseau ni LLM.

Lancer : uv run pytest -q tests/test_rag.py
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.agent import ask, rag
from src.agent.llm import LLMConfigError, LLMRefusal, LLMTurn
from src.retrieval.schemas import RetrievedChunk, SearchResponse

PASSAGES = [
    RetrievedChunk(
        rank=1,
        chunk_id="F21051::0",
        document_id="F21051",
        source="SERVICE_PUBLIC",
        title="Supplément de loyer de solidarité (surloyer)",
        url="https://www.service-public.gouv.fr/particuliers/vosdroits/F21051",
        text="Le surloyer est dû si les revenus dépassent d'au moins 20 % le plafond.",
        score=1.0,
    ),
    RetrievedChunk(
        rank=2,
        chunk_id="LEGIARTI000006824658::0",
        document_id="LEGIARTI000006824658",
        source="LEGI",
        title="Code de la construction et de l'habitation",
        article="L441-3",
        url="https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006824658",
        text="Les organismes d'habitations à loyer modéré exigent un supplément de loyer.",
        score=0.5,
    ),
]


def search_returning(results: list[RetrievedChunk]):
    def search(query: str, k: int = 5, **_: Any) -> SearchResponse:
        kept = results[:k]
        return SearchResponse(
            query=query,
            results=kept,
            document_ids=[r.document_id for r in kept],
            method="fake",
            models={},
            latency_ms=1.0,
        )

    return search


class FakeJSONClient:
    def __init__(
        self, outcome: dict | str | Exception, finish_reason: str = "stop"
    ) -> None:
        self.outcome = outcome
        self.finish_reason = finish_reason
        self.calls: list[tuple[list[dict], dict, str]] = []

    def complete_json(self, messages, schema, name):
        self.calls.append((messages, schema, name))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        content = (
            self.outcome if isinstance(self.outcome, str) else json.dumps(self.outcome)
        )
        return LLMTurn(
            content=content,
            tool_calls=[],
            tokens_in=500,
            tokens_out=50,
            cost_usd=0.002,
            finish_reason=self.finish_reason,
            model="fake-model",
        )


@pytest.fixture
def llm(monkeypatch):
    """Installe un faux client et renvoie une fonction pour scripter sa sortie."""
    built: dict[str, Any] = {}

    def install(outcome, *, results=PASSAGES, finish_reason="stop") -> FakeJSONClient:
        client = FakeJSONClient(outcome, finish_reason)

        def fake_build(provider=None, model=None):
            built.update(provider=provider, model=model)
            return client

        monkeypatch.setattr(
            rag, "get_search", lambda name="hybrid": search_returning(results)
        )
        monkeypatch.setattr(rag, "build_client", fake_build)
        return client

    install.built = built
    return install


def answer(text: str, cited: list[str], abstained: bool = False) -> dict:
    return {"answer": text, "citation_document_ids": cited, "abstained": abstained}


def test_reponse_citee_avec_article_legi(llm):
    client = llm(
        answer(
            "Le surloyer est dû au-delà de 20 % du plafond.",
            ["F21051", "LEGIARTI000006824658", "F21051"],
        )
    )

    response = ask("Quand paie-t-on un surloyer ?", backend="rag", provider="anthropic")

    assert response.stop_reason == "answered" and not response.abstained
    assert [(c.document_id, c.article) for c in response.citations] == [
        ("F21051", None),
        ("LEGIARTI000006824658", "L441-3"),
    ]
    assert response.warnings == []
    assert response.cost_usd == 0.002
    assert llm.built == {"provider": "anthropic", "model": None}

    ((messages, schema, name),) = client.calls
    assert messages[0]["content"] == rag.RAG_SYSTEM_PROMPT
    context = json.loads(messages[1]["content"])
    assert context["question"] == "Quand paie-t-on un surloyer ?"
    assert [p["article"] for p in context["passages"]] == [None, "L441-3"]
    assert schema == rag.GROUNDED_ANSWER_JSON_SCHEMA and name == "grounded_answer"


def test_fournisseur_par_defaut_ollama_sans_cout(llm):
    llm(answer("Le surloyer est dû au-delà de 20 % du plafond.", ["F21051"]))

    response = rag.answer_with_rag("Quand paie-t-on un surloyer ?")

    assert llm.built["provider"] == "ollama"
    assert response.cost_usd is None


def test_citation_inventee_rejette_la_reponse(llm):
    llm(answer("Voir la fiche F99999.", ["F21051", "F99999"]))

    response = rag.answer_with_rag("Quand paie-t-on un surloyer ?")

    assert response.abstained and response.stop_reason == "abstained"
    assert response.answer == rag.ABSTENTION_ANSWER
    assert response.citations == []
    assert "citation(s) hors contexte supprimée(s) : F99999" in response.warnings
    assert any("absents du contexte" in w for w in response.warnings)


def test_reponse_sans_source_rejetee(llm):
    llm(answer("Le surloyer est dû au-delà de 20 % du plafond.", []))

    response = rag.answer_with_rag("Quand paie-t-on un surloyer ?")

    assert response.abstained
    assert "réponse rejetée : aucune source citée" in response.warnings


def test_abstention_sans_citation_et_reformulee(llm):
    llm(answer("abstained", ["F21051"], abstained=True))

    response = rag.answer_with_rag("Quel est le loyer moyen à Lyon ?")

    assert response.abstained and response.stop_reason == "abstained"
    assert response.citations == []
    assert response.answer == rag.ABSTENTION_ANSWER


@pytest.mark.parametrize(
    "error",
    [
        LLMRefusal("Le modèle a décliné la requête"),
        LLMConfigError("ANTHROPIC_API_KEY absente"),
    ],
)
def test_erreur_llm_ne_leve_pas(llm, error):
    llm(error)

    response = rag.answer_with_rag(
        "Quand paie-t-on un surloyer ?", provider="anthropic"
    )

    assert response.stop_reason == "error" and response.abstained
    assert response.retrieved_document_ids == ["F21051", "LEGIARTI000006824658"]


def test_json_invalide_devient_une_erreur(llm):
    llm('{"answer": "incomplet"}')

    response = rag.answer_with_rag("Quand paie-t-on un surloyer ?")

    assert response.stop_reason == "error"


def test_aucun_passage_abstention_sans_appel_llm(llm):
    client = llm(answer("x", []), results=[])

    response = rag.answer_with_rag("Question hors corpus")

    assert response.abstained and response.answer == rag.NO_SOURCE_ANSWER
    assert client.calls == []


def test_reponse_tronquee_signalee(llm):
    llm(
        answer("Le surloyer est dû au-delà de 20 % du plafond.", ["F21051"]),
        finish_reason="max_tokens",
    )

    response = rag.answer_with_rag("Quand paie-t-on un surloyer ?")

    assert "réponse tronquée : limite de tokens atteinte" in response.warnings


def test_chiffre_absent_des_sources_signale(llm):
    llm(answer("Le surloyer est dû au-delà de 37 504 € de revenus.", ["F21051"]))

    response = rag.answer_with_rag("Quand paie-t-on un surloyer ?")

    assert response.stop_reason == "answered"
    assert "chiffre(s) absent(s) des sources : 37504" in response.warnings
