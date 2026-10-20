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


def test_mcp_four_tools_discovery():
    """Les quatre outils doivent être accessibles au client MCP."""

    async def run():
        async with Client(mcp_server.mcp) as client:
            response = await client.list_tools()
            names = {tool.name for tool in response.tools}
            assert names == {
                "search_docs",
                "check_eligibility",
                "social_housing_stats",
                "market_prices",
            }

    asyncio.run(run())


def test_mcp_eligibility_real_data():
    """Le MCP doit retourner le calcul déterministe du barème 2026."""

    async def run():
        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool(
                "check_eligibility",
                {
                    "revenu_fiscal_reference": 28000.0,
                    "nb_personnes": 3,
                    "nb_personnes_a_charge": 1,
                    "commune": "Lyon",
                    "annee": 2026,
                },
            )
            assert not result.is_error
            data = result.structured_content
            assert data["plafonds"] == {
                "plai": 22551,
                "plus": 37584,
                "pls": 48859,
            }
            assert data["eligible"] == {
                "plai": False,
                "plus": True,
                "pls": True,
            }
            assert data["provenance"]["data_year"] == 2026

    asyncio.run(run())


def test_mcp_rpls_real_data():
    """Le MCP conserve les statistiques et la provenance RPLS."""

    async def run():
        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool(
                "social_housing_stats",
                {"commune": "Créteil"},
            )
            assert not result.is_error
            data = result.structured_content
            assert data["nb_logements"] == 15133
            assert data["millesime"] == 2025
            assert data["provenance"]["data_year"] == 2025
            assert data["caveats"]

    asyncio.run(run())


def test_mcp_dvf_real_data():
    """Le MCP conserve les statistiques DVF et leurs restrictions."""

    async def run():
        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool(
                "market_prices",
                {
                    "commune": "Villeurbanne",
                    "type_local": "Appartement",
                    "annee": 2025,
                },
            )
            assert not result.is_error
            data = result.structured_content
            stats = data["statistiques"]
            assert len(stats) == 1
            assert stats[0]["n_ventes"] == 1890
            assert stats[0]["prix_m2_median"] == 3567.0
            assert data["provenance"]["data_year"] == 2025
            assert data["caveats"]

    asyncio.run(run())


def test_mcp_invalid_business_inputs():
    """Les entrées invalides doivent produire des erreurs MCP."""

    cases = [
        (
            "check_eligibility",
            {
                "revenu_fiscal_reference": 28000.0,
                "nb_personnes": 2,
                "nb_personnes_a_charge": 2,
                "commune": "Lyon",
            },
        ),
        (
            "check_eligibility",
            {
                "revenu_fiscal_reference": 28000.0,
                "nb_personnes": 3,
                "commune": "Lyon",
                "annee": 2030,
            },
        ),
        (
            "social_housing_stats",
            {"commune": "CommuneInexistanteXYZ"},
        ),
        (
            "market_prices",
            {
                "commune": "Villeurbanne",
                "type_local": "Appartement",
                "annee": 2030,
            },
        ),
        (
            "market_prices",
            {
                "commune": "Villeurbanne",
                "type_local": "Chateau",
                "annee": 2025,
            },
        ),
    ]

    async def run():
        async with Client(mcp_server.mcp) as client:
            for name, arguments in cases:
                result = await client.call_tool(name, arguments)
                assert result.is_error, (
                    f"{name} a accepté des arguments invalides : "
                    f"{arguments}"
                )

    asyncio.run(run())
