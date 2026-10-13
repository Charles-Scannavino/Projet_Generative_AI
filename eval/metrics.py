"""Métriques de retrieval, calculées au niveau DOCUMENT.

Les retrievers classent des chunks, mais le golden set indique des
documents attendus (gold_docs). On convertit donc d'abord le classement
de chunks en classement de documents, puis on calcule :

- Recall@k : part des gold_docs présents dans les k premiers documents ;
- Hit@1    : 1 si le premier document est un gold_doc, 0 sinon ;
- MRR      : moyenne de 1 / rang du premier gold_doc (0 s'il est absent).

Recall@1 et Hit@1 ne diffèrent que pour les questions à plusieurs
gold_docs (q017) : un seul document ne peut couvrir que 1/2 des attendus.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


RECALL_KS = (1, 5, 10)


def chunks_to_documents(chunk_document_ids: Iterable[str]) -> list[str]:
    """Classement de documents à partir d'un classement de chunks.

    Les chunks étant triés du meilleur au moins bon, le premier chunk
    rencontré pour un document donne le meilleur rang de ce document ;
    les chunks suivants du même document sont ignorés.
    """
    ranked_documents = []
    seen = set()

    for document_id in chunk_document_ids:
        if document_id in seen:
            continue

        seen.add(document_id)
        ranked_documents.append(document_id)

    return ranked_documents


def first_relevant_rank(
    ranked_documents: Sequence[str],
    gold_docs: Sequence[str],
) -> int | None:
    """Rang (à partir de 1) du premier gold_doc, None s'il est absent."""
    gold = set(gold_docs)

    for rank, document_id in enumerate(ranked_documents, start=1):
        if document_id in gold:
            return rank

    return None


def recall_at_k(
    ranked_documents: Sequence[str],
    gold_docs: Sequence[str],
    k: int,
) -> float:
    gold = set(gold_docs)

    return len(gold & set(ranked_documents[:k])) / len(gold)


def hit_at_k(
    ranked_documents: Sequence[str],
    gold_docs: Sequence[str],
    k: int,
) -> float:
    return float(bool(set(gold_docs) & set(ranked_documents[:k])))


def reciprocal_rank(
    ranked_documents: Sequence[str],
    gold_docs: Sequence[str],
) -> float:
    rank = first_relevant_rank(ranked_documents, gold_docs)

    return 1.0 / rank if rank else 0.0


def evaluate_question(
    ranked_documents: Sequence[str],
    gold_docs: Sequence[str],
) -> dict:
    """Toutes les métriques d'une question."""
    row = {
        f"recall_at_{k}": recall_at_k(ranked_documents, gold_docs, k)
        for k in RECALL_KS
    }

    row["hit_at_1"] = hit_at_k(ranked_documents, gold_docs, 1)
    row["reciprocal_rank"] = reciprocal_rank(ranked_documents, gold_docs)
    row["first_relevant_rank"] = first_relevant_rank(
        ranked_documents,
        gold_docs,
    )

    return row


def mean_metrics(rows: Sequence[dict]) -> dict:
    """Moyenne des métriques sur toutes les questions (MRR = moyenne des RR)."""
    n = len(rows)

    summary = {
        f"recall_at_{k}": sum(row[f"recall_at_{k}"] for row in rows) / n
        for k in RECALL_KS
    }

    summary["mrr"] = sum(row["reciprocal_rank"] for row in rows) / n
    summary["hit_at_1"] = sum(row["hit_at_1"] for row in rows) / n

    return summary
