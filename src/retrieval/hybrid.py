"""Recherche hybride : fusion des classements BM25 et dense par RRF.

RRF (Reciprocal Rank Fusion, Cormack, Clarke & Büttcher, SIGIR 2009) :

    score_RRF(chunk) = somme sur les retrievers de 1 / (k + rang)

- on fusionne des RANGS, jamais des scores bruts : un score BM25 n'est
  pas borné alors qu'une similarité cosinus est comprise entre -1 et 1,
  les additionner n'aurait pas de sens ;
- k = 60 est la valeur de l'article original, reprise par défaut par la
  plupart des moteurs ; plus k est grand, moins le rang 1 écrase les
  suivants ;
- chaque retriever fournit ses 100 meilleurs chunks : un chunk absent du
  top 100 d'un retriever ne reçoit simplement rien de ce retriever.

La fusion se fait au niveau des CHUNKS (comme BM25 et dense) ; le passage
aux documents se fait ensuite, pour l'évaluation uniquement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.config import CANDIDATE_DEPTH, RRF_K
from src.retrieval.dense import DenseRetriever


@dataclass
class HybridCandidate:
    """Un chunk candidat, avec la trace de son classement."""

    chunk_index: int
    chunk: dict
    rrf_score: float
    bm25_rank: int | None
    dense_rank: int | None
    rerank_score: float | None = None


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Fusionne plusieurs classements d'indices de chunks.

    Renvoie des couples (indice du chunk, score RRF) triés par score
    décroissant. En cas d'égalité, l'ordre du fichier de chunks départage :
    le résultat est donc toujours le même.
    """
    scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = RRF_K,
        candidate_depth: int = CANDIDATE_DEPTH,
    ) -> None:
        # Les deux retrievers doivent indexer exactement les mêmes chunks,
        # dans le même ordre : la fusion travaille sur les indices.
        bm25_ids = [chunk["chunk_id"] for chunk in bm25.chunks]
        dense_ids = [chunk["chunk_id"] for chunk in dense.chunks]

        if bm25_ids != dense_ids:
            raise ValueError(
                "BM25 et dense n'indexent pas les mêmes chunks."
            )

        self.bm25 = bm25
        self.dense = dense
        self.chunks = bm25.chunks
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth

        # Un masque booléen par source, calculé une seule fois,
        # pour le filtre `source` de search_docs.
        sources = [str(chunk["source"]) for chunk in self.chunks]

        self.source_masks = {
            source: np.array(sources) == source
            for source in sorted(set(sources))
        }

    def search(
        self,
        query: str,
        source: str | None = None,
    ) -> list[HybridCandidate]:
        """Candidats fusionnés par RRF, du meilleur au moins bon.

        Si `source` est donné, chaque retriever ne classe que les chunks
        de cette source : le filtre est appliqué AVANT la fusion, donc
        avant toute coupe aux k résultats.
        """
        if source is None:
            allowed = None
        elif source in self.source_masks:
            allowed = self.source_masks[source]
        else:
            raise ValueError(
                f"Source inconnue : {source!r}. "
                f"Valeurs possibles : {sorted(self.source_masks)}."
            )

        bm25_ranking = [
            index
            for index, _ in self.bm25.rank(
                query,
                top_k=self.candidate_depth,
                allowed=allowed,
            )
        ]

        dense_ranking = [
            index
            for index, _ in self.dense.rank(
                query,
                top_k=self.candidate_depth,
                allowed=allowed,
            )
        ]

        bm25_ranks = {
            index: rank
            for rank, index in enumerate(bm25_ranking, start=1)
        }

        dense_ranks = {
            index: rank
            for rank, index in enumerate(dense_ranking, start=1)
        }

        fused = reciprocal_rank_fusion(
            [bm25_ranking, dense_ranking],
            k=self.rrf_k,
        )

        return [
            HybridCandidate(
                chunk_index=index,
                chunk=self.chunks[index],
                rrf_score=score,
                bm25_rank=bm25_ranks.get(index),
                dense_rank=dense_ranks.get(index),
            )
            for index, score in fused
        ]
