from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


@dataclass
class DenseSearchResult:
    chunk_id: str
    document_id: str
    title: str
    score: float
    text: str


class DenseRetriever:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None

    @property
    def device(self) -> str:
        return str(self.model.device)

    @property
    def embedding_dimension(self) -> int:
        return int(
            self.model.get_embedding_dimension()
        )

    def build(
        self,
        chunks: list[dict],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> None:
        self.chunks = chunks

        passages = [
            "passage: "
            + str(
                chunk.get(
                    "retrieval_text",
                    chunk.get("text", ""),
                )
            )
            for chunk in chunks
        ]

        self.embeddings = self.model.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[DenseSearchResult]:

        if self.embeddings is None:
            raise RuntimeError(
                "L'index dense n'a pas encore été construit."
            )

        query_embedding = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        # Les embeddings étant normalisés,
        # le produit scalaire = similarité cosinus.
        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(
            -scores,
            kind="stable",
        )[:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[int(index)]

            results.append(
                DenseSearchResult(
                    chunk_id=str(chunk["chunk_id"]),
                    document_id=str(
                        chunk["document_id"]
                    ),
                    title=str(
                        chunk.get("title", "")
                    ),
                    score=float(
                        scores[int(index)]
                    ),
                    text=str(
                        chunk.get("text", "")
                    ),
                )
            )

        return results
