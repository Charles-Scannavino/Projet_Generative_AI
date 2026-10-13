"""Tests du contrat `search_docs` (lancer : uv run pytest -q)."""

from __future__ import annotations

import pytest

from src.retrieval import search_docs, warmup
from src.retrieval.common import load_chunks
from src.retrieval.config import DEFAULT_RERANKER_MODEL, MAX_K
from src.retrieval.schemas import SearchResponse


# q001 du golden set, dont le document attendu est F10007.
Q001 = "Comment peut-on déposer une demande de logement social (HLM) ?"


@pytest.fixture(scope="module", autouse=True)
def models_loaded():
    # Chargement unique des index et modèles pour tout le module.
    warmup()


@pytest.fixture(scope="module")
def corpus_document_ids() -> set[str]:
    return {str(chunk["document_id"]) for chunk in load_chunks()}


@pytest.mark.parametrize("k", [1, 5, 10, MAX_K])
@pytest.mark.parametrize("rerank", [True, False])
def test_renvoie_exactement_k_resultats(k, rerank):
    response = search_docs(Q001, k=k, rerank=rerank)

    assert len(response.results) == k
    assert [result.rank for result in response.results] == list(range(1, k + 1))


@pytest.mark.parametrize("source", ["SERVICE_PUBLIC", "LEGI"])
def test_filtre_source_respecte(source):
    response = search_docs(Q001, k=10, source=source)

    assert len(response.results) == 10
    assert {result.source for result in response.results} == {source}


def test_document_ids_presents_dans_le_corpus(corpus_document_ids):
    response = search_docs(Q001, k=10)

    assert response.document_ids
    assert set(response.document_ids) <= corpus_document_ids

    # IDs uniques, dans l'ordre des résultats.
    assert response.document_ids == list(
        dict.fromkeys(result.document_id for result in response.results)
    )


def test_resultats_deterministes():
    first = search_docs(Q001, k=10)
    second = search_docs(Q001, k=10)

    assert [r.chunk_id for r in first.results] == [r.chunk_id for r in second.results]
    assert [r.score for r in first.results] == pytest.approx(
        [r.score for r in second.results]
    )


def test_schema_pydantic_valide():
    response = search_docs(Q001, k=5)

    assert isinstance(response, SearchResponse)
    assert SearchResponse.model_validate_json(response.model_dump_json()) == response

    assert response.method == "hybrid_rrf+rerank"
    assert response.models["reranker"] == DEFAULT_RERANKER_MODEL
    assert response.latency_ms > 0

    for result in response.results:
        assert result.score == result.rerank_score
        assert result.rrf_score is not None
        assert result.bm25_rank is not None or result.dense_rank is not None


def test_sans_reranker_score_rrf():
    response = search_docs(Q001, k=5, rerank=False)

    assert response.method == "hybrid_rrf"
    assert "reranker" not in response.models

    for result in response.results:
        assert result.rerank_score is None
        assert result.score == result.rrf_score


def test_q001_trouve_F10007_dans_le_top_5():
    response = search_docs(Q001, k=5)

    assert "F10007" in response.document_ids


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_requete_vide_erreur_claire(query):
    with pytest.raises(ValueError, match="requête est vide"):
        search_docs(query)


@pytest.mark.parametrize(
    "kwargs",
    [{"k": 0}, {"k": MAX_K + 1}, {"k": True}, {"source": "AUTRE"}],
)
def test_arguments_invalides(kwargs):
    with pytest.raises(ValueError):
        search_docs(Q001, **kwargs)
