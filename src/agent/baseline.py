"""Backend « baseline » : BM25 seul, SANS LLM (réponse = début du meilleur passage).

Deux usages :
- point de comparaison dans l'évaluation (que rapporte vraiment le LLM ?) ;
- développement de l'UI et de run_eval.py sans Ollama ni PyTorch (démarre en 1 s).
"""

from __future__ import annotations

import re
import time

from src.agent.rag import passages_from
from src.agent.retrieval import RetrieverName, get_search
from src.agent.schemas import AssistantResponse, Citation, ToolCall


def answer_baseline(question: str, *, k: int = 5, retriever: RetrieverName = "bm25", **_: object) -> AssistantResponse:
    started = time.perf_counter()
    search = get_search(retriever)(question, k=k)
    passages = passages_from(search)
    if not passages:
        return AssistantResponse(question=question, answer="Aucun passage trouvé.", abstained=True,
                                 stop_reason="abstained", backend="baseline", model="none")
    top = passages[0]
    sentences = re.split(r"(?<=[.!?])\s+", top.text.strip())
    return AssistantResponse(
        question=question,
        answer=" ".join(sentences[:2]),
        citations=[Citation(document_id=top.document_id, source=top.source, title=top.title, url=top.url)],
        retrieved_document_ids=search.document_ids,
        passages=passages,
        tool_calls=[ToolCall(name="search_docs", arguments={"query": question, "k": k},
                             output={"document_ids": search.document_ids}, latency_ms=search.latency_ms)],
        backend="baseline",
        model="none (extractif)",
        latency_ms=(time.perf_counter() - started) * 1000,
        latency_breakdown_ms={"retrieval": search.latency_ms},
    )
