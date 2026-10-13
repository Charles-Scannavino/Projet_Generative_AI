from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.common import (
    INDEX_DIR,
    rank_indices,
    sha256_file,
)
from src.retrieval.config import DENSE_MODEL_NAME


DEFAULT_MODEL_NAME = DENSE_MODEL_NAME


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

    # ========================================================
    # CACHE DES EMBEDDINGS
    #
    # Encoder les 1 357 chunks prend ~20 s : on sauvegarde les
    # embeddings dans data/indexes/ (.npy) avec un fichier JSON
    # de métadonnées. Le cache n'est réutilisé que si le modèle,
    # le nombre de chunks ET le hash SHA-256 du fichier de chunks
    # sont identiques ; sinon l'index est reconstruit.
    # ========================================================

    def cache_paths(
        self,
        chunks_path: Path,
        cache_dir: Path = INDEX_DIR,
    ) -> tuple[Path, Path]:
        model_slug = self.model_name.replace("/", "__")
        stem = f"dense__{Path(chunks_path).stem}__{model_slug}"

        return (
            Path(cache_dir) / f"{stem}.npy",
            Path(cache_dir) / f"{stem}.json",
        )

    def load_or_build(
        self,
        chunks: list[dict],
        chunks_path: Path,
        cache_dir: Path = INDEX_DIR,
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> bool:
        """Charge les embeddings depuis le cache, ou les calcule.

        `chunks` doit avoir été lu depuis `chunks_path`.
        Renvoie True si le cache a été utilisé, False sinon.
        """
        npy_path, meta_path = self.cache_paths(
            chunks_path,
            cache_dir,
        )

        expected = {
            "model_name": self.model_name,
            "n_chunks": len(chunks),
            "chunks_sha256": sha256_file(chunks_path),
        }

        if npy_path.exists() and meta_path.exists():
            stored = json.loads(
                meta_path.read_text(encoding="utf-8")
            )

            if all(
                stored.get(key) == value
                for key, value in expected.items()
            ):
                embeddings = np.load(npy_path)

                if embeddings.shape == (
                    len(chunks),
                    self.embedding_dimension,
                ):
                    self.chunks = chunks
                    self.embeddings = embeddings
                    return True

        self.build(
            chunks,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        Path(cache_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(npy_path, self.embeddings)

        # Les métadonnées sont écrites en dernier : si l'écriture
        # est interrompue, le cache sera simplement reconstruit.
        meta = {
            **expected,
            "chunks_file": Path(chunks_path).name,
            "embedding_dimension": self.embedding_dimension,
            "normalized": True,
            "passage_prefix": "passage: ",
        }

        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return False

    # ========================================================
    # RECHERCHE
    # ========================================================

    def scores(self, query: str) -> np.ndarray:
        """Similarité cosinus entre la question et chaque chunk."""
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
        return self.embeddings @ query_embedding

    def rank(
        self,
        query: str,
        top_k: int | None = None,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        """Couples (indice du chunk, score cosinus), du meilleur au moins bon."""
        return rank_indices(
            self.scores(query),
            top_k=top_k,
            allowed=allowed,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[DenseSearchResult]:

        results = []

        for index, score in self.rank(query, top_k=top_k):
            chunk = self.chunks[index]

            results.append(
                DenseSearchResult(
                    chunk_id=str(chunk["chunk_id"]),
                    document_id=str(
                        chunk["document_id"]
                    ),
                    title=str(
                        chunk.get("title", "")
                    ),
                    score=score,
                    text=str(
                        chunk.get("text", "")
                    ),
                )
            )

        return results
