"""Backend « agent_tools » : tool calling natif, le LLM choisit et enchaîne les outils.

    question ──> LLM ──(appels d'outils)──> ToolExecutor ──> résultats JSON ──> LLM …
                  └──(final_answer)──> validation des citations ──> AssistantResponse

Complémentaire du backend « agent » (src/agent/agent.py), conçu pour un petit modèle
local (recherche systématique + formulaire JSON) : ici le modèle décide lui-même des
appels, ce qui suppose un modèle fiable en tool calling (API via OpenRouter, ou Ollama).
Un seul client compatible OpenAI sert les deux fournisseurs (src/agent/llm.py) ; les
garde-fous d'exécution sont dans src/agent/tools_registry.py et src/agent/guards.py.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any

from pydantic import Field, StringConstraints, ValidationError

from src.agent.guards import detect_injection, ungrounded_numbers
from src.agent.llm import LLM_ERRORS, LLMClient, LLMTurn, build_client
from src.agent.prompts import prompt_version, system_prompt
from src.agent.rag import passages_from
from src.agent.retrieval import RetrieverName, get_search
from src.agent.schemas import AssistantResponse, Citation, Passage, StopReason, ToolCall
from src.agent.tools_registry import (
    ToolCallTrace,
    ToolExecutor,
    ToolSpec,
    build_registry,
    load_tool_descriptions,
    validation_message,
)
from src.settings import Backend, get_settings
from src.tools.schemas import StrictInput

logger = logging.getLogger(__name__)

BACKEND_NAME = "agent_tools"
FINAL_TOOL = "final_answer"
MAX_REPEATS = 2

# Schéma JSON de la réponse finale pour la sortie structurée (repli quand le modèle
# répond en texte libre). « Strict » au sens OpenAI : tous les champs requis ; les
# contraintes fines (motif des identifiants, longueurs) sont validées par FinalAnswer.
FINAL_ANSWER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_document_ids": {"type": "array", "items": {"type": "string"}},
        "abstained": {"type": "boolean"},
        "needs_clarification": {"type": "boolean"},
        "abstention_reason": {"type": ["string", "null"]},
    },
    "required": [
        "answer",
        "cited_document_ids",
        "abstained",
        "needs_clarification",
        "abstention_reason",
    ],
    "additionalProperties": False,
}

STRUCTURE_INSTRUCTION = (
    "Reformule ta réponse précédente dans le format JSON demandé (answer, "
    "cited_document_ids, abstained, needs_clarification, abstention_reason), "
    "sans ajouter d'information."
)

DocumentId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{1,40}$")]


class FinalAnswer(StrictInput):
    """Arguments du pseudo-outil `final_answer`, validés strictement (pas d'URL, pas de champ libre)."""

    answer: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ] = Field(description="Réponse finale en français.")
    cited_document_ids: Annotated[list[DocumentId], Field(max_length=10)] = Field(
        default_factory=list,
        description="document_id des passages search_docs utilisés.",
    )
    abstained: bool = Field(
        default=False,
        description="True si la réponse ne peut pas être donnée de façon fiable.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True si une information indispensable manque et est demandée à l'utilisateur.",
    )
    abstention_reason: Annotated[str, StringConstraints(max_length=500)] | None = None


# Arrêts sans réponse finale du modèle : message et statut du contrat commun.
FALLBACKS: dict[str, tuple[str, StopReason]] = {
    "max_steps": (
        "Je n'ai pas pu aboutir à une réponse fiable dans le nombre d'étapes autorisé.",
        "max_steps",
    ),
    "repeated_call": (
        "Je n'ai pas pu aboutir à une réponse fiable : la recherche tournait en rond.",
        "max_steps",
    ),
    "empty_response": ("Je n'ai pas pu produire de réponse.", "error"),
}


def _final_answer_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": FINAL_TOOL,
            "description": load_tool_descriptions()[FINAL_TOOL],
            "parameters": FinalAnswer.model_json_schema(),
        },
    }


def _assistant_message(turn: LLMTurn) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": turn.content}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in turn.tool_calls
        ]
    return message


def _tool_error_message(call_id: str, message: str) -> dict[str, Any]:
    payload = {
        "tool": FINAL_TOOL,
        "status": "error",
        "error": {"code": "invalid_input", "message": message},
    }
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def _contract_tool_calls(executor: ToolExecutor) -> list[ToolCall]:
    """Trace interne -> ToolCall du contrat ; search_docs résumé (les passages sont à part)."""
    searches = iter(executor.searches)
    calls = []

    for trace in executor.traces:
        output = trace.output
        if trace.name == "search_docs" and trace.ok:
            search = next(searches)
            output = {"document_ids": search.document_ids, "method": search.method}

        calls.append(
            ToolCall(
                name=trace.name,
                arguments=trace.arguments or {},
                output=output,
                ok=trace.ok,
                error=None
                if trace.ok
                else f"{trace.error_code}: {trace.error_message}",
                latency_ms=trace.latency_ms,
            )
        )

    return calls


def _passages(executor: ToolExecutor) -> list[Passage]:
    passages, seen = [], set()
    for search in executor.searches:
        for passage in passages_from(search):
            key = (passage.document_id, passage.text)
            if key not in seen:
                seen.add(key)
                passages.append(passage)
    return passages


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _dropped_argument_warnings(traces: list[ToolCallTrace]) -> list[str]:
    return [
        f"argument « {argument} » ignoré pour {trace.name} : absent de la question "
        "alors que la commune est homonyme"
        for trace in traces
        for argument in trace.dropped_arguments
    ]


def _structure_free_text(
    client: LLMClient,
    messages: list[dict[str, Any]],
    content: str,
) -> tuple[FinalAnswer, LLMTurn | None]:
    """Réponse en texte libre -> FinalAnswer, par sortie JSON contrainte si possible.

    Les petits modèles locaux répondent souvent en texte au lieu d'appeler
    final_answer (mesuré le 18/10 avec qwen3:4b-instruct) : citations et indicateurs
    d'abstention seraient perdus. Repli : le texte brut, sans citation.
    """
    fallback = FinalAnswer(answer=content.strip()[:4000])
    complete_json = getattr(client, "complete_json", None)
    if complete_json is None:
        return fallback, None

    followup = [
        *messages,
        {"role": "assistant", "content": content},
        {"role": "user", "content": STRUCTURE_INSTRUCTION},
    ]
    try:
        turn = complete_json(followup, FINAL_ANSWER_JSON_SCHEMA, FINAL_TOOL)
        return FinalAnswer.model_validate_json(turn.content or "{}"), turn
    except (*LLM_ERRORS, ValidationError) as exc:
        logger.warning("Sortie structurée de la réponse finale impossible : %s", exc)
        return fallback, None


def run_tool_agent(
    question: str,
    *,
    k: int | None = None,
    retriever: RetrieverName = "hybrid",
    model: str | None = None,
    provider: Backend | None = None,
    client: LLMClient | None = None,
    registry: dict[str, ToolSpec] | None = None,
) -> AssistantResponse:
    """Répond à une question en orchestrant les outils ; ne lève pas sur une erreur LLM.

    Args:
        k: passages par appel à search_docs (défaut : configs/agent.yaml).
        retriever: "hybrid" (défaut), "hybrid_norerank" ou "bm25".
        model: remplace le modèle du fournisseur dans configs/agent.yaml.
        provider: "openrouter" ou "ollama" (défaut : configs/agent.yaml).
    """
    start = time.perf_counter()
    settings = get_settings().agent

    client = client or build_client(provider, model)
    registry = registry or build_registry(get_search(retriever), default_k=k)
    executor = ToolExecutor(
        registry=registry,
        max_calls_per_tool=settings.max_calls_per_tool,
        user_question=question,
    )

    system = system_prompt(client.backend, [*registry, FINAL_TOOL])
    tools = [spec.openai_schema() for spec in registry.values()] + [
        _final_answer_schema()
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    tokens_in = tokens_out = llm_calls = repeats = 0
    llm_ms = 0.0
    costs: list[float | None] = []
    final: FinalAnswer | None = None
    reason = "max_steps"
    error: str | None = None
    used_model = client.model

    for _ in range(settings.max_steps):
        llm_start = time.perf_counter()
        try:
            turn = client.chat(messages, tools)
        except LLM_ERRORS as exc:
            logger.error("Erreur LLM (%s) : %s", type(exc).__name__, exc)
            reason, error = "llm_error", f"{type(exc).__name__}: {exc}"
            break
        finally:
            llm_ms += (time.perf_counter() - llm_start) * 1000

        llm_calls += 1
        tokens_in += turn.tokens_in
        tokens_out += turn.tokens_out
        costs.append(turn.cost_usd)
        used_model = turn.model

        if not turn.tool_calls:
            if turn.content and turn.content.strip():
                final, structured = _structure_free_text(client, messages, turn.content)
                if structured is not None:
                    llm_calls += 1
                    tokens_in += structured.tokens_in
                    tokens_out += structured.tokens_out
                    costs.append(structured.cost_usd)
                    reason = "structured_final"
                else:
                    reason = "no_final_tool"
            else:
                reason = "empty_response"
            break

        messages.append(_assistant_message(turn))
        finished = False

        for call in turn.tool_calls:
            if call.name == FINAL_TOOL:
                try:
                    final = FinalAnswer.model_validate_json(call.arguments or "{}")
                except ValidationError as exc:
                    messages.append(
                        _tool_error_message(call.id, validation_message(exc))
                    )
                    continue
                reason, finished = "final_answer", True
                break

            outcome = executor.execute(call.name, call.arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": outcome.content}
            )

            # Un appel répété reçoit une erreur ; à la deuxième répétition, on arrête.
            repeats += outcome.repeated
            if repeats >= MAX_REPEATS:
                reason, finished = "repeated_call", True
                break

        if finished:
            break

    warnings: list[str] = []

    if reason == "llm_error":
        answer, stop_reason, abstained = f"Erreur technique : {error}", "error", True
        cited_ids: list[str] = []
    elif final is None:
        answer, stop_reason = FALLBACKS[reason]
        abstained, cited_ids = True, []
        if reason == "repeated_call":
            warnings.append("boucle interrompue : appel d'outil répété")
    else:
        answer, cited_ids = final.answer, list(dict.fromkeys(final.cited_document_ids))
        abstained = final.abstained or final.needs_clarification
        if final.needs_clarification:
            stop_reason = "needs_clarification"
        elif final.abstained:
            stop_reason = "abstained"
        else:
            stop_reason = "answered"
        if reason == "no_final_tool":
            warnings.append("réponse en texte libre, sans appel à final_answer")
        elif reason == "structured_final":
            warnings.append(
                "réponse finale reformatée par sortie structurée "
                "(le modèle n'a pas appelé final_answer)"
            )

    # Citations : uniquement des documents récupérés pendant ce run ; titre et URL
    # viennent du corpus, jamais du modèle.
    citations, dropped = [], []
    for document_id in cited_ids:
        meta = executor.documents.get(document_id)
        if meta is None:
            dropped.append(document_id)
            continue
        citations.append(
            Citation(
                document_id=meta.document_id,
                source=meta.source,
                title=meta.title,
                url=meta.url,
            )
        )
    if dropped:
        warnings.append(
            f"citation(s) hors contexte supprimée(s) : {', '.join(dropped)}"
        )

    # Sources fiables pour l'ancrage : sorties d'outils, question, prompt système.
    # Un message d'erreur technique (codes HTTP, tailles de contexte) n'est pas contrôlé.
    missing = (
        []
        if stop_reason == "error"
        else ungrounded_numbers(answer, [*executor.source_texts, question, system])
    )
    if missing:
        warnings.append(
            "chiffre(s) absent(s) des sources : "
            + ", ".join(_format_number(value) for value in missing)
        )

    warnings += _dropped_argument_warnings(executor.traces)

    if detect_injection(question) or executor.injection_in_sources:
        warnings.append(
            "motif d'injection détecté dans la question ou un passage : "
            "contenu traité comme une donnée"
        )

    tools_ms = sum(trace.latency_ms for trace in executor.traces)

    return AssistantResponse(
        question=question,
        answer=answer,
        citations=citations,
        retrieved_document_ids=executor.retrieved_document_ids,
        passages=_passages(executor),
        tool_calls=_contract_tool_calls(executor),
        abstained=abstained,
        stop_reason=stop_reason,
        warnings=warnings,
        backend=BACKEND_NAME,
        model=used_model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        llm_calls=llm_calls,
        latency_ms=(time.perf_counter() - start) * 1000,
        latency_breakdown_ms={"llm": llm_ms, "tools": tools_ms},
        cost_usd=None
        if not costs or any(c is None for c in costs)
        else round(sum(costs), 6),
        prompt_version=prompt_version(system),
    )
