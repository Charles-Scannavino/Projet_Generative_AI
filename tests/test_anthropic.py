"""Fournisseur Anthropic (Claude Opus 5.5) avec un SDK simulé : sans réseau ni clé.

Lancer : uv run pytest -q tests/test_anthropic.py
Test réel : uv run pytest -m integration (ANTHROPIC_API_KEY dans .env)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agent import ask
from src.agent.llm import (
    NATIVE_KEY,
    AnthropicClient,
    LLMConfigError,
    LLMRefusal,
    build_client,
)
from src.agent.tool_agent import run_tool_agent
from src.agent.tools_registry import build_registry
from src.settings import get_settings
from tests.test_agent import SURLOYER, fake_search

# ============================================================
# SDK SIMULÉ
# ============================================================


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def thinking() -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking="", signature="sig")


def message(*blocks, stop_reason="end_turn", tokens=(1000, 200), stop_details=None):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
        model="claude-opus-5-5",
    )


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(request)
        return self.responses.pop(0)


def fake_client(*responses) -> tuple[AnthropicClient, FakeMessages]:
    messages = FakeMessages(responses)
    sdk = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return AnthropicClient(sdk_client=sdk), messages


OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Cherche dans le corpus.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}


# ============================================================
# REQUÊTE CONFORME À CLAUDE OPUS 5.5
# ============================================================


def test_requete_conforme_opus_5_5():
    client, api = fake_client(
        message(
            thinking(),
            tool_use("t1", "search_docs", {"query": "dalo"}),
            stop_reason="tool_use",
        )
    )

    turn = client.chat(
        [
            {"role": "system", "content": "règles"},
            {"role": "user", "content": "Dalo ?"},
        ],
        [OPENAI_TOOL],
    )

    (request,) = api.requests
    assert request["model"] == "claude-opus-5-5"
    # Interdits sur Opus 5.5 (400) : temperature, seed, tool_choice forcé, thinking désactivé.
    for forbidden in ("temperature", "seed", "tool_choice", "thinking"):
        assert forbidden not in request
    assert request["output_config"] == {"effort": "medium"}
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["fallbacks"] == "default"
    assert request["system"] == "règles"
    assert request["messages"] == [{"role": "user", "content": "Dalo ?"}]
    assert request["tools"] == [
        {
            "name": "search_docs",
            "description": "Cherche dans le corpus.",
            "input_schema": OPENAI_TOOL["function"]["parameters"],
        }
    ]

    (call,) = turn.tool_calls
    assert (call.id, call.name, json.loads(call.arguments)) == (
        "t1",
        "search_docs",
        {"query": "dalo"},
    )
    assert turn.cost_usd == pytest.approx((1000 * 4.0 + 200 * 20.0) / 1e6)
    assert [block.type for block in turn.native] == ["thinking", "tool_use"]


def test_historique_converti_et_blocs_natifs_renvoyes_tels_quels():
    native = [thinking(), tool_use("t1", "search_docs", {"query": "dalo"})]
    history = [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "search_docs",
                        "arguments": '{"query": "dalo"}',
                    },
                }
            ],
            NATIVE_KEY: native,
        },
        {"role": "tool", "tool_call_id": "t1", "content": '{"status": "ok"}'},
        {"role": "tool", "tool_call_id": "t2", "content": '{"status": "error"}'},
    ]

    system, converted = AnthropicClient.to_anthropic(history)

    assert system == ""
    assert converted[1] == {"role": "assistant", "content": native}  # thinking conservé
    # Résultats d'outils regroupés dans UN message user, erreurs marquées.
    assert converted[2]["role"] == "user"
    assert [(b["tool_use_id"], b["is_error"]) for b in converted[2]["content"]] == [
        ("t1", False),
        ("t2", True),
    ]
    assert len(converted) == 3


def test_message_assistant_sans_contenu_natif():
    _, converted = AnthropicClient.to_anthropic(
        [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "Je cherche.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "market_prices",
                            "arguments": '{"commune": "Lyon"}',
                        },
                    }
                ],
            },
        ]
    )

    assert converted[1]["content"] == [
        {"type": "text", "text": "Je cherche."},
        {
            "type": "tool_use",
            "id": "c1",
            "name": "market_prices",
            "input": {"commune": "Lyon"},
        },
    ]


def test_sortie_structuree():
    client, api = fake_client(
        message(text('{"verdict": "correct", "justification": "ok"}'))
    )
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
        "additionalProperties": False,
    }

    turn = client.complete_json(
        [{"role": "user", "content": "Juge."}], schema, "verdict"
    )

    (request,) = api.requests
    assert request["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": schema},
    }
    assert "tools" not in request
    assert json.loads(turn.content)["verdict"] == "correct"


def test_refus_leve_llm_refusal():
    client, _ = fake_client(
        message(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))
    )

    with pytest.raises(LLMRefusal, match="cyber"):
        client.chat([{"role": "user", "content": "Q"}], [])


def test_cle_absente(monkeypatch):
    without_key = get_settings().model_copy(update={"anthropic_api_key": None})
    monkeypatch.setattr("src.agent.llm.get_settings", lambda: without_key)

    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        build_client("anthropic")


# ============================================================
# AGENT À OUTILS DE BOUT EN BOUT SUR ANTHROPIC (SDK simulé)
# ============================================================


def test_agent_tools_sur_anthropic():
    first = message(
        thinking(),
        tool_use("t1", "search_docs", {"query": "surloyer"}),
        stop_reason="tool_use",
    )
    final = message(
        thinking(),
        tool_use(
            "t2",
            "final_answer",
            {
                "answer": "Le surloyer est dû au-delà de 20 % du plafond.",
                "cited_document_ids": ["F21051"],
                "abstained": False,
            },
        ),
        stop_reason="tool_use",
    )
    client, api = fake_client(first, final)
    registry = build_registry(search_fn=fake_search(SURLOYER))

    response = run_tool_agent(
        "Quand paie-t-on un surloyer ?", client=client, registry=registry
    )

    assert response.stop_reason == "answered"
    assert [c.document_id for c in response.citations] == ["F21051"]
    assert response.cost_usd == pytest.approx(2 * (1000 * 4.0 + 200 * 20.0) / 1e6)

    second = api.requests[1]["messages"]
    assert second[1] == {
        "role": "assistant",
        "content": first.content,
    }  # blocs d'origine
    (result,) = second[2]["content"]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"


def test_refus_dans_l_agent_devient_une_erreur():
    client, _ = fake_client(message(stop_reason="refusal"))
    registry = build_registry(search_fn=fake_search(SURLOYER))

    response = ask("Question", backend="agent_tools", client=client, registry=registry)

    assert response.stop_reason == "error" and response.abstained
    assert "décliné" in response.answer


def test_juge_run_eval_sur_anthropic(monkeypatch):
    from eval.run_eval import VERDICT_SCHEMA, judge
    from src.agent.schemas import AssistantResponse

    client, api = fake_client(
        message(
            text('{"verdict": "partial", "justification": "incomplet"}'),
            tokens=(300, 40),
        )
    )
    monkeypatch.setattr("src.agent.llm.build_client", lambda provider, model: client)
    row = {"question": "Q", "answer": "R", "should_abstain": False}
    response = AssistantResponse(question="Q", answer="R'", backend="rag", model="m")

    verdict = judge(row, response, None, provider="anthropic")

    assert verdict["judge_score"] == 0.5
    assert verdict["judge_model"] == "claude-opus-5-5"
    assert verdict["judge_cost_usd"] == pytest.approx((300 * 4.0 + 40 * 20.0) / 1e6)
    assert api.requests[0]["output_config"]["format"]["schema"] == VERDICT_SCHEMA


# ============================================================
# INTÉGRATION (réseau + ANTHROPIC_API_KEY)
# ============================================================


@pytest.mark.integration
def test_integration_anthropic():
    if get_settings().anthropic_api_key is None:
        pytest.skip("ANTHROPIC_API_KEY absente")

    response = ask(
        "Combien de logements sociaux compte Créteil ?",
        backend="agent_tools",
        provider="anthropic",
    )

    assert response.stop_reason == "answered"
    assert "social_housing_stats" in [c.name for c in response.tool_calls]
    assert "15 133" in response.answer or "15133" in response.answer
