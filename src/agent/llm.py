"""Clients LLM de l'agent à outils : OpenRouter (API compatible OpenAI) et Ollama.

Ollama passe par son API native /api/chat (client partagé src/agent/ollama.py) et non
par son endpoint compatible OpenAI : ce dernier ignore la taille de contexte demandée
et applique celle du serveur, 4 096 tokens par défaut sur une machine à 8 Go, alors
que le 2e tour de l'agent (outils + passages) dépasse 5 000 tokens (mesuré le 18/10).
"""

from __future__ import annotations

import itertools
import json
import logging
from typing import Any, Protocol

import openai
from openai import OpenAI
from pydantic import BaseModel

from src.agent import ollama
from src.settings import Backend, get_settings

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # JSON brut produit par le modèle, validé plus tard


class LLMTurn(BaseModel):
    content: str | None
    tool_calls: list[ToolCall]
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    finish_reason: str | None
    model: str


class LLMClient(Protocol):
    backend: str
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMTurn: ...


class LLMConfigError(RuntimeError):
    pass


# Erreurs « attendues » d'un appel LLM : l'agent les transforme en réponse d'erreur.
LLM_ERRORS: tuple[type[Exception], ...] = (openai.OpenAIError, ollama.OllamaError)


class OpenAICompatibleClient:
    def __init__(self, backend: Backend, model: str | None = None) -> None:
        settings = get_settings()
        agent = settings.agent
        config = agent.backends[backend]

        if settings.openrouter_api_key is None:
            raise LLMConfigError(
                "OPENROUTER_API_KEY absente : la renseigner dans .env "
                "(voir .env.example) ou utiliser --provider ollama."
            )
        api_key = settings.openrouter_api_key.get_secret_value()

        headers = {"X-Title": config.app_name} if config.app_name else None

        self.backend = backend
        self.model = model or config.model
        self._tool_choice = config.tool_choice
        self._temperature = agent.temperature
        self._seed = agent.seed
        self._prices = agent.prices_usd_per_mtok.get(self.model)
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=agent.request_timeout_s,
            default_headers=headers,
            max_retries=2,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMTurn:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=self._tool_choice,
            temperature=self._temperature,
            seed=self._seed,
        )

        choice = response.choices[0]
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        return LLMTurn(
            content=choice.message.content,
            tool_calls=[
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in choice.message.tool_calls or []
            ],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost(usage, tokens_in, tokens_out),
            finish_reason=choice.finish_reason,
            model=response.model or self.model,
        )

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        name: str,
    ) -> LLMTurn:
        """Réponse contrainte par un schéma JSON (sortie structurée, sans outils).

        Utilisé par le juge LLM de eval/run_eval.py. Le schéma doit être « strict »
        au sens OpenAI : tous les champs requis, `additionalProperties: false`.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self._temperature,
            seed=self._seed,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )

        choice = response.choices[0]
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        return LLMTurn(
            content=choice.message.content,
            tool_calls=[],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost(usage, tokens_in, tokens_out),
            finish_reason=choice.finish_reason,
            model=response.model or self.model,
        )

    def _cost(self, usage: Any, tokens_in: int, tokens_out: int) -> float | None:
        # OpenRouter ajoute `usage.cost` (champ hors spécification OpenAI).
        reported = getattr(usage, "cost", None) if usage else None
        if reported is not None:
            return float(reported)

        if self._prices is None:
            logger.warning("Prix inconnu pour %s : coût non calculé.", self.model)
            return None

        return (tokens_in * self._prices.input + tokens_out * self._prices.output) / 1e6


class OllamaNativeClient:
    """Tool calling via l'API native d'Ollama (`num_ctx` par requête, `think=False`).

    Hôte et taille de contexte : OLLAMA_HOST et OLLAMA_NUM_CTX (.env), communs à tous
    les backends Ollama du projet. Les messages au format OpenAI de la boucle de
    l'agent sont convertis au format natif (arguments en objet, `tool_name`).
    """

    backend = "ollama"

    def __init__(self, model: str | None = None) -> None:
        agent = get_settings().agent
        self.model = model or agent.backends["ollama"].model
        self._temperature = agent.temperature
        self._ids = itertools.count(1)

    @staticmethod
    def to_native(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        names: dict[str, str] = {}
        native = []

        for message in messages:
            message = {**message, "content": message.get("content") or ""}

            if message.get("tool_calls"):
                calls = []
                for call in message["tool_calls"]:
                    function = call["function"]
                    names[call["id"]] = function["name"]
                    try:
                        arguments = json.loads(function["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    calls.append(
                        {"function": {"name": function["name"], "arguments": arguments}}
                    )
                message["tool_calls"] = calls

            if message["role"] == "tool":
                message["tool_name"] = names.get(message.pop("tool_call_id", ""), "")

            native.append(message)

        return native

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMTurn:
        result = ollama.chat(
            self.to_native(messages),
            model=self.model,
            tools=tools,
            temperature=self._temperature,
        )

        calls = []
        for call in result.tool_calls:
            function = call.get("function", {})
            arguments = function.get("arguments") or {}
            calls.append(
                ToolCall(
                    id=call.get("id") or f"call_{next(self._ids)}",
                    name=function.get("name", ""),
                    arguments=arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False),
                )
            )

        return LLMTurn(
            content=result.content or None,
            tool_calls=calls,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=0.0,
            finish_reason=result.done_reason,
            model=self.model,
        )

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        name: str,
    ) -> LLMTurn:
        """Réponse contrainte par un schéma JSON (décodage contraint d'Ollama)."""
        result = ollama.chat(
            self.to_native(messages),
            model=self.model,
            schema=schema,
            temperature=self._temperature,
        )
        return LLMTurn(
            content=result.content,
            tool_calls=[],
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=0.0,
            finish_reason=result.done_reason,
            model=self.model,
        )


def build_client(
    backend: Backend | None = None, model: str | None = None
) -> OpenAICompatibleClient | OllamaNativeClient:
    """Client du fournisseur demandé ; `model` remplace le modèle de configs/agent.yaml."""
    backend = backend or get_settings().agent.backend
    if backend == "ollama":
        return OllamaNativeClient(model)
    return OpenAICompatibleClient(backend, model)
