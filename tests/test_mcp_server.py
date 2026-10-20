"""Tests du serveur MCP de l'Assistant HLM."""

import asyncio

from mcp import Client

from src import mcp_server
from src.retrieval.schemas import RetrievedChunk, SearchResponse


def test_mcp_discovery():
    """Le client MCP doit découvrir search_docs et son schéma."""

    async def run():
        async with Client(mcp_server.mcp) as client:
            response = await client.list_tools()

            tools = {
                tool.name: tool
                for tool in response.tools
            }

            assert "search_docs" in tools
            assert tools["search_docs"].output_schema is not None

    asyncio.run(run())


def test_mcp_search_docs(monkeypatch):
    """Le MCP transmet correctement la question au retrieval."""

    calls = []

    def fake_search(*, query, k, rerank):
        calls.append((query, k, rerank))

        passage = RetrievedChunk(
            rank=1,
            chunk_id="F18005_test_1",
            document_id="F18005",
            source="SERVICE_PUBLIC",
            title="Droit au logement opposable (Dalo)",
            url="https://www.service-public.gouv.fr/",
            text="Le DALO signifie droit au logement opposable.",
            score=0.95,
        )

        return SearchResponse(
            query=query,
            results=[passage],
            document_ids=["F18005"],
            method="hybrid_rrf+rerank",
            models={"bm25": "test"},
            latency_ms=1.5,
        )

    monkeypatch.setattr(
        mcp_server,
        "retrieval_search_docs",
        fake_search,
    )

    async def run():
        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool(
                "search_docs",
                {
                    "query": "Qu'est-ce que le DALO ?",
                    "k": 3,
                },
            )

            assert not result.is_error

            data = result.structured_content

            assert isinstance(data, dict)
            assert data["document_ids"] == ["F18005"]
            assert data["method"] == "hybrid_rrf+rerank"
            assert data["results"][0]["document_id"] == "F18005"

    asyncio.run(run())

    assert calls == [
        ("Qu'est-ce que le DALO ?", 3, True)
    ]


async def _assert_mcp_error(arguments, expected_message):
    """Vérifie qu'une requête invalide produit une erreur MCP lisible."""

    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool(
            "search_docs",
            arguments,
        )

        assert result.is_error

        message = " ".join(
            getattr(item, "text", "")
            for item in result.content
        )

        assert expected_message in message


def test_mcp_rejects_empty_query():
    """Une question vide doit être rejetée."""

    asyncio.run(
        _assert_mcp_error(
            {"query": "", "k": 5},
            "requête est vide",
        )
    )


def test_mcp_rejects_invalid_k():
    """Le nombre de passages doit être compris entre 1 et 30."""

    asyncio.run(
        _assert_mcp_error(
            {"query": "DALO", "k": 0},
            "entre 1 et 30",
        )
    )
