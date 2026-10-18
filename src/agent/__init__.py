"""Point d'entrée unique des backends : `from src.agent import ask`.

    ask("Qu'est-ce que le Dalo ?", backend="agent")   # -> AssistantResponse

Backends :
- "baseline"    : BM25 seul, sans LLM (src/agent/baseline.py)
- "rag"         : search_docs + Qwen3, sans choix d'outil (src/agent/rag.py)
- "agent"       : recherche systématique + formulaire JSON rempli par Qwen3 local,
                  Python choisit et exécute l'outil (src/agent/agent.py)
- "agent_tools" : tool calling natif, le LLM enchaîne les outils ; API via OpenRouter
                  ou Ollama, options `provider` et `model` (src/agent/tool_agent.py)
"""

from __future__ import annotations

from typing import Literal

from src.agent.schemas import AssistantResponse

Backend = Literal["baseline", "rag", "agent", "agent_tools"]
BACKENDS: tuple[str, ...] = ("baseline", "rag", "agent", "agent_tools")


def ask(question: str, backend: Backend = "agent", **options) -> AssistantResponse:
    if not question or not question.strip():
        raise ValueError("Question vide.")
    question = question.strip()
    if backend == "baseline":
        from src.agent.baseline import answer_baseline

        return answer_baseline(question, **options)
    if backend == "rag":
        from src.agent.rag import answer_with_rag

        return answer_with_rag(question, **options)
    if backend == "agent":
        from src.agent.agent import run_agent

        return run_agent(question, **options)
    if backend == "agent_tools":
        from src.agent.tool_agent import run_tool_agent

        return run_tool_agent(question, **options)
    raise ValueError(f"Backend inconnu : {backend!r} (choix : {BACKENDS})")


__all__ = ["ask", "AssistantResponse", "BACKENDS", "Backend"]
