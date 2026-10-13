"""Reranker cross-encoder : reclasse les meilleurs candidats de l'hybride.

Différence avec le dense E5 (bi-encodeur) :
- E5 encode la question et le chunk SÉPARÉMENT, puis compare deux
  vecteurs : rapide, car les chunks sont encodés une fois pour toutes ;
- le cross-encoder lit la question ET le chunk ENSEMBLE, dans une même
  entrée : chaque mot de la question peut « regarder » chaque mot du
  chunk. C'est plus précis, mais il faut un passage du modèle par paire
  (question, chunk) à chaque requête.

On ne l'applique donc qu'aux 30 premiers candidats de l'hybride ; les
candidats suivants gardent leur ordre RRF, à la suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
from sentence_transformers import CrossEncoder

from src.retrieval.config import RERANK_TOP_N, RERANKER_MAX_LENGTH
from src.retrieval.hybrid import HybridCandidate


def default_device() -> str:
    """MPS (GPU Apple Silicon) si disponible, sinon CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = RERANKER_MAX_LENGTH,
    ) -> None:
        self.model_name = model_name
        self.device = device or default_device()

        # max_length = 512 tokens pour la paire (question, chunk).
        # Mesuré avec la question q001 : 178 paires sur 1 357 (13 %)
        # dépassent 512 tokens (maximum 716) ; seule leur fin est tronquée.
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=max_length,
        )

    def score(
        self,
        query: str,
        texts: Sequence[str],
    ) -> list[float]:
        """Score de pertinence de chaque texte pour la question.

        L'échelle dépend du modèle : logit brut pour mmarco-mMiniLMv2,
        sigmoïde entre 0 et 1 pour bge-reranker-v2-m3. Seul l'ordre compte.
        """
        if not texts:
            return []

        scores = self.model.predict(
            [(query, text) for text in texts],
            # Les 30 candidats tiennent dans un seul lot.
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[HybridCandidate],
        top_n: int = RERANK_TOP_N,
    ) -> list[HybridCandidate]:
        """Reclasse les `top_n` premiers candidats ; les autres suivent.

        Entrée du modèle : (question, retrieval_text), c'est-à-dire le
        titre du document suivi du texte du chunk.
        """
        head = list(candidates[:top_n])
        tail = list(candidates[top_n:])

        scores = self.score(
            query,
            [
                str(
                    candidate.chunk.get(
                        "retrieval_text",
                        candidate.chunk.get("text", ""),
                    )
                )
                for candidate in head
            ],
        )

        rescored = [
            replace(candidate, rerank_score=score)
            for candidate, score in zip(head, scores)
        ]

        # Tri stable : à score égal, l'ordre RRF est conservé.
        rescored.sort(key=lambda candidate: -candidate.rerank_score)

        return rescored + tail
