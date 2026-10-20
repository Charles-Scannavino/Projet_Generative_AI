import json
from types import SimpleNamespace

import pytest

from src.agent import agent
from src.agent.schemas import AssistantResponse, Citation, ToolCall


@pytest.mark.parametrize(
    "action,tool_name,source_id",
    [
        ("stats_commune", "social_housing_stats", "RPLS"),
        ("prix_commune", "market_prices", "DVF"),
    ],
)
def test_citation_provenance_outil(monkeypatch, action, tool_name, source_id):
    route = agent.DEFAULT_ROUTE.model_copy(
        update={"action": action, "commune": "Créteil"}
    )

    monkeypatch.setattr(
        agent, "get_search",
        lambda retriever: lambda question, k: SimpleNamespace(
            document_ids=[], method="test", latency_ms=1
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

    source_url = "https://example.org/donnees-officielles"
    monkeypatch.setattr(
        agent,
        "stats_tools",
        lambda: {
            action: (
                tool_name,
                lambda arguments: {
                    "provenance": {
                        "source_name": "Source officielle de test",
                        "source_url": source_url,
                        "data_year": 2025,
                    }
                },
            )
        },
    )
    monkeypatch.setattr(
        agent,
        "synthesize",
        lambda *args: (
            None,
            SimpleNamespace(tokens_in=10, tokens_out=10, latency_ms=1),
        ),
    )
    monkeypatch.setattr(
        agent,
        "build_response",
        lambda **kwargs: AssistantResponse(
            question=kwargs["question"],
            answer="Résultat statistique de test",
            citations=[
                Citation(
                    document_id="F920",
                    source="SERVICE_PUBLIC",
                    title="Document juridique sans rapport",
                    url="https://www.service-public.gouv.fr/",
                )
            ],
            backend="agent",
            model="fake-model",
            tool_calls=kwargs["tool_calls"],
        ),
    )

    result = agent.run_agent("Statistiques de Créteil")
    citations = [c for c in result.citations if c.document_id == source_id]

    assert len(result.citations) == 1
    assert len(citations) == 1
    assert citations[0].url == source_url
    assert citations[0].title == "Source officielle de test"
    assert any(c.name == tool_name for c in result.tool_calls)
