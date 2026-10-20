"""Clients LLM : OpenRouter (API compatible OpenAI), Ollama et Anthropic.

Tous exposent `chat(messages, tools)` (tool calling) et `complete_json(messages,
schema, name)` (sortie JSON contrainte) sur des messages au format OpenAI ; chaque
client les convertit au format de son API.

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

import anthropic
import openai
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from src.agent import ollama
from src.settings import Backend, get_settings

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # JSON brut produit par le modèle, validé plus tard


class LLMTurn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str | None
    tool_calls: list[ToolCall]
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    finish_reason: str | None
    model: str
    # Contenu natif de la réponse (Anthropic : blocs thinking / tool_use), à renvoyer
    # tel quel au tour suivant : Claude Opus 5.5 refuse un historique retouché.
    native: list[Any] | None = None


# Clé privée d'un message assistant portant ce contenu natif ; retirée avant envoi
# aux autres fournisseurs.
NATIVE_KEY = "_native_content"


def public_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in m.items() if k != NATIVE_KEY} for m in messages]


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


class LLMRefusal(RuntimeError):
    """Le modèle a décliné la requête (stop_reason « refusal »), même après repli."""


# Erreurs « attendues » d'un appel LLM : l'agent les transforme en réponse d'erreur.
LLM_ERRORS: tuple[type[Exception], ...] = (
    openai.OpenAIError,
    ollama.OllamaError,
    anthropic.APIError,
    LLMRefusal,
)


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
            messages=public_messages(messages),
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
            messages=public_messages(messages),
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

        for message in public_messages(messages):
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


class AnthropicClient:
    """API Messages d'Anthropic via le SDK officiel (Claude Opus 5.5 par défaut).

    Contraintes de Claude Opus 5.5 respectées :
    - thinking adaptatif toujours actif : pas de paramètre `thinking`, profondeur réglée
      par `output_config.effort` (fixé explicitement, le défaut du modèle étant `medium`) ;
    - ni `temperature` ni `seed` (400) ; `tool_choice` laissé à `auto` (forcer un outil
      renvoie 400) : l'agent garde son repli par sortie structurée ;
    - blocs de réponse (thinking, tool_use) renvoyés tels quels au tour suivant ;
    - refus : repli serveur `fallbacks: "default"` (bêta server-side-fallback-2026-07-01),
      puis LLMRefusal si le refus persiste.
    """

    backend = "anthropic"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str | None = None, sdk_client: Any | None = None) -> None:
        settings = get_settings()
        agent = settings.agent
        config = agent.backends["anthropic"]

        self.model = model or config.model
        self._effort = config.effort or "medium"
        self._max_tokens = config.max_tokens or 16000
        self._prices = agent.prices_usd_per_mtok.get(self.model)

        if sdk_client is not None:
            self._client = sdk_client
            return

        if settings.anthropic_api_key is None:
            raise LLMConfigError(
                "ANTHROPIC_API_KEY absente : la renseigner dans .env "
                "(voir .env.example) ou utiliser un autre fournisseur."
            )
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=agent.request_timeout_s,
            max_retries=2,
        )

    @staticmethod
    def _is_error_result(content: str) -> bool:
        try:
            return json.loads(content).get("status") == "error"
        except (json.JSONDecodeError, AttributeError):
            return False

    @classmethod
    def to_anthropic(
        cls, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Messages OpenAI -> (system, messages Anthropic).

        Les résultats d'outils consécutifs sont regroupés dans un seul message `user`
        (sinon le modèle cesse d'appeler plusieurs outils en parallèle).
        """
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []

        def flush_results() -> None:
            if results:
                converted.append({"role": "user", "content": list(results)})
                results.clear()

        for message in messages:
            role = message["role"]
            content = message.get("content") or ""

            if role == "system":
                system_parts.append(content)
                continue

            if role == "tool":
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": content,
                        "is_error": cls._is_error_result(content),
                    }
                )
                continue

            flush_results()

            if role == "assistant":
                native = message.get(NATIVE_KEY)
                if native is not None:
                    converted.append({"role": "assistant", "content": native})
                    continue
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for call in message.get("tool_calls") or []:
                    try:
                        arguments = json.loads(call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": arguments,
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
                continue

            converted.append({"role": "user", "content": content})

        flush_results()
        return "\n\n".join(part for part in system_parts if part), converted

    @staticmethod
    def to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"],
            }
            for tool in tools
        ]

    def _create(
        self,
        messages: list[dict[str, Any]],
        output_config: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMTurn:
        system, converted = self.to_anthropic(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": converted,
            "output_config": {"effort": self._effort, **output_config},
            "betas": [self.FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = self.to_anthropic_tools(tools)

        response = self._client.beta.messages.create(**request)

        if response.stop_reason == "refusal":
            details = response.stop_details
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(
                f"Le modèle a décliné la requête (catégorie : {category or 'non précisée'})."
            )

        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            ToolCall(
                id=block.id,
                name=block.name,
                arguments=json.dumps(block.input, ensure_ascii=False),
            )
            for block in response.content
            if block.type == "tool_use"
        ]
        usage = response.usage
        tokens_in = usage.input_tokens if usage else 0
        tokens_out = usage.output_tokens if usage else 0

        return LLMTurn(
            content=text or None,
            tool_calls=calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._cost(response.model, tokens_in, tokens_out),
            finish_reason=response.stop_reason,
            model=response.model or self.model,
            native=list(response.content),
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMTurn:
        return self._create(messages, {}, tools)

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        name: str,
    ) -> LLMTurn:
        """Sortie structurée (`output_config.format`) : le texte renvoyé est du JSON valide."""
        return self._create(
            messages, {"format": {"type": "json_schema", "schema": schema}}
        )

    def _cost(self, model: str, tokens_in: int, tokens_out: int) -> float | None:
        # Thinking compris dans output_tokens. Si le repli serveur a servi un autre
        # modèle, son tarif n'est pas forcément connu.
        prices = self._prices if model in (None, self.model) else None
        if prices is None:
            logger.warning("Prix inconnu pour %s : coût non calculé.", model)
            return None
        return (tokens_in * prices.input + tokens_out * prices.output) / 1e6


def build_client(
    backend: Backend | None = None, model: str | None = None
) -> OpenAICompatibleClient | OllamaNativeClient | AnthropicClient:
    """Client du fournisseur demandé ; `model` remplace le modèle de configs/agent.yaml."""
    backend = backend or get_settings().agent.backend
    if backend == "ollama":
        return OllamaNativeClient(model)
    if backend == "anthropic":
        return AnthropicClient(model)
    return OpenAICompatibleClient(backend, model)
