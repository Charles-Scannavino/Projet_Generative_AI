"""Choix du retriever utilisé par les backends.

- "hybrid" (défaut) : search_docs complet (BM25 + E5 + RRF + reranker), charge PyTorch ;
- "bm25"            : BM25 seul, sans PyTorch : démarrage instantané, utile pour
                      développer l'UI et pour la baseline sans LLM.
Les deux renvoient le même `SearchResponse` (src/retrieval/schemas.py).
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Callable, Literal

from src.retrieval.schemas import RetrievedChunk, SearchResponse

RetrieverName = Literal["hybrid", "hybrid_norerank", "bm25"]
SearchFn = Callable[..., SearchResponse]


@lru_cache(maxsize=1)
def _bm25():
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.common import CHUNKS_PATH, load_chunks

    chunks = load_chunks(CHUNKS_PATH)
    return chunks, BM25Retriever(chunks)


def bm25_search(query: str, k: int = 5, source: str | None = None, **_: object) -> SearchResponse:
    start = time.perf_counter()
    chunks, retriever = _bm25()
    allowed = None
    if source:
        import numpy as np

        allowed = np.array([chunk["source"] == source for chunk in chunks])
    results = []
    for rank, (index, score) in enumerate(retriever.rank(query, top_k=k, allowed=allowed), start=1):
        chunk = chunks[index]
        results.append(
            RetrievedChunk(
                rank=rank, chunk_id=str(chunk["chunk_id"]), document_id=str(chunk["document_id"]),
                source=chunk["source"], title=str(chunk.get("title", "")), article=chunk.get("article"),
                url=str(chunk.get("url", "")), text=str(chunk.get("text", "")), score=float(score),
                bm25_rank=rank,
            )
        )
    return SearchResponse(
        query=query, results=results,
        document_ids=list(dict.fromkeys(r.document_id for r in results)),
        method="bm25", models={"bm25": "rank_bm25.BM25Okapi"},
        latency_ms=(time.perf_counter() - start) * 1000,
    )


def get_search(name: RetrieverName = "hybrid") -> SearchFn:
    if name == "bm25":
        return bm25_search
    from src.retrieval import search_docs

    if name == "hybrid_norerank":
        return lambda query, k=5, source=None, **_: search_docs(query, k=k, source=source, rerank=False)
    return lambda query, k=5, source=None, **_: search_docs(query, k=k, source=source, rerank=True)
