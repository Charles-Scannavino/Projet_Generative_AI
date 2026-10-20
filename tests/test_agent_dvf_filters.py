from types import SimpleNamespace

from src.agent import agent
from src.agent.schemas import AssistantResponse


def test_agent_transmet_filtres_dvf(monkeypatch):
    question = (
        "Quel est le prix médian au m² des appartements "
        "vendus à Villeurbanne en 2025 ?"
    )

    route = agent.DEFAULT_ROUTE.model_copy(
        update={
            "action": "prix_commune",
            "commune": "Villeurbanne",
            "type_local": "Appartement",
            "annee": None,
        }
    )

    received = []

    monkeypatch.setattr(
        agent,
        "get_search",
        lambda retriever: lambda question, k: SimpleNamespace(
            document_ids=[],
            method="test",
            latency_ms=1,
        ),
    )
    monkeypatch.setattr(agent, "passages_from", lambda search: [])
    monkeypatch.setattr(
        agent.ollama,
        "chat",
        lambda *args, **kwargs: SimpleNamespace(
            content=route.model_dump_json(),
            latency_ms=1,
            tokens_in=10,
            tokens_out=10,
        ),
    )

    def fake_market_prices(arguments):
        received.append(arguments.copy())
        return {
            "statistiques": [
                {
                    "annee": 2025,
                    "type_local": "Appartement",
                    "n_ventes": 1890,
                    "prix_m2_median": 3567.0,
                }
            ],
            "provenance": {
                "source_name": "DGFiP DVF",
                "source_url": "https://www.data.gouv.fr/",
            },
        }

    monkeypatch.setattr(
        agent,
        "stats_tools",
        lambda: {"prix_commune": ("market_prices", fake_market_prices)},
    )
    monkeypatch.setattr(
        agent,
        "synthesize",
        lambda *args: (
            None,
            SimpleNamespace(
                tokens_in=10,
                tokens_out=10,
                latency_ms=1,
            ),
        ),
    )
    monkeypatch.setattr(
        agent,
        "build_response",
        lambda **kwargs: AssistantResponse(
            question=kwargs["question"],
            answer="3 567 euros par m²",
            backend="agent",
            model="fake-model",
            tool_calls=kwargs["tool_calls"],
        ),
    )

    response = agent.run_agent(question)

    assert received == [
        {
            "commune": "Villeurbanne",
            "type_local": "Appartement",
            "annee": 2025,
        }
    ]
    assert response.citations[0].document_id == "DVF"
    assert response.stop_reason != "error"


def test_extract_dvf_year():
    assert agent.extract_dvf_year(
        "Prix des appartements vendus en 2025 ?"
    ) == 2025

    assert agent.extract_dvf_year(
        "Prix des maisons vendues en 2024 ?"
    ) == 2024

    assert agent.extract_dvf_year(
        "Prix des appartements à Villeurbanne ?"
    ) is None

    assert agent.extract_dvf_year(
        "Comparer les ventes de 2024 et 2025."
    ) is None

    assert agent.extract_dvf_year(
        "Comparer deux communes pour 2025, année 2025."
    ) == 2025
