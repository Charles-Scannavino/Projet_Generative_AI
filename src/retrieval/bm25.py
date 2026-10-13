"""Retriever lexical BM25 sur le champ retrieval_text des chunks.

BM25 compte les mots communs entre la question et le chunk, en donnant
plus de poids aux mots rares du corpus (IDF) et en normalisant par la
longueur du chunk.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.common import rank_indices


def tokenize(text: str) -> list[str]:
    """Tokenisation simple, IDENTIQUE à eval/eval_bm25_chunking.py.

    Ne pas la modifier sans relancer l'évaluation : les résultats BM25
    de référence (MRR = 0.8306) en dépendent.
    """
    text = unicodedata.normalize(
        "NFKC",
        str(text),
    ).lower()

    # l'article -> l article
    # d'un -> d un
    text = (
        text
        .replace("’", " ")
        .replace("'", " ")
    )

    tokens = re.findall(
        r"[a-zà-öø-ÿ0-9]+",
        text,
        flags=re.IGNORECASE,
    )

    return tokens


@dataclass
class BM25SearchResult:
    chunk_id: str
    document_id: str
    title: str
    score: float
    text: str


class BM25Retriever:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

        tokenized_corpus = [
            tokenize(
                chunk.get(
                    "retrieval_text",
                    chunk.get("text", ""),
                )
            )
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def scores(self, query: str) -> np.ndarray:
        """Score BM25 de chaque chunk pour la question."""
        return self.bm25.get_scores(tokenize(query))

    def rank(
        self,
        query: str,
        top_k: int | None = None,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        """Couples (indice du chunk, score BM25), du meilleur au moins bon."""
        return rank_indices(
            self.scores(query),
            top_k=top_k,
            allowed=allowed,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[BM25SearchResult]:
        results = []

        for index, score in self.rank(query, top_k=top_k):
            chunk = self.chunks[index]

            results.append(
                BM25SearchResult(
                    chunk_id=str(chunk["chunk_id"]),
                    document_id=str(chunk["document_id"]),
                    title=str(chunk.get("title", "")),
                    score=score,
                    text=str(chunk.get("text", "")),
                )
            )

        return results
