from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.retrieval.dense import (
    DEFAULT_MODEL_NAME,
    DenseRetriever,
)


GOLDEN_PATH = Path("eval/golden.jsonl")

CHUNKS_PATH = Path(
    "data/processed/chunks/fixed_300_60.jsonl"
)

OUTPUT_DIR = Path(
    "eval/dense"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


golden = load_jsonl(
    GOLDEN_PATH
)

questions = [
    row
    for row in golden
    if (
        row.get("type") == "rag"
        and row.get("gold_docs")
    )
]

chunks = load_jsonl(
    CHUNKS_PATH
)


print("=" * 90)
print("ÉVALUATION DENSE RETRIEVAL")
print("=" * 90)

print("Modèle :", DEFAULT_MODEL_NAME)
print("Questions RAG :", len(questions))
print("Chunks :", len(chunks))
print()


# ============================================================
# CONSTRUCTION INDEX
# ============================================================

retriever = DenseRetriever()

print("Device :", retriever.device)
print(
    "Dimension embeddings :",
    retriever.embedding_dimension,
)

print()
print("Encodage des chunks...")


start_build = time.perf_counter()

retriever.build(
    chunks,
    batch_size=32,
)

build_seconds = (
    time.perf_counter()
    - start_build
)


print(
    f"Index dense construit en "
    f"{build_seconds:.2f} s"
)


# ============================================================
# ÉVALUATION
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


details = []

recall_5_values = []
recall_10_values = []
reciprocal_ranks = []
query_latencies_ms = []


for question in questions:

    start_query = time.perf_counter()

    # On classe tous les chunks afin que
    # le calcul du MRR soit exact.
    results = retriever.search(
        question["question"],
        top_k=len(chunks),
    )

    query_latency_ms = (
        time.perf_counter()
        - start_query
    ) * 1000

    query_latencies_ms.append(
        query_latency_ms
    )


    # ========================================================
    # CHUNKS -> DOCUMENTS
    # ========================================================

    ranked_documents = []

    seen_documents = set()


    for result in results:

        document_id = (
            result.document_id
        )

        if document_id in seen_documents:
            continue

        seen_documents.add(
            document_id
        )

        ranked_documents.append(
            {
                "document_id": (
                    document_id
                ),
                "chunk_id": (
                    result.chunk_id
                ),
                "score": (
                    result.score
                ),
                "title": (
                    result.title
                ),
            }
        )


    gold_docs = [
        str(doc)
        for doc in question[
            "gold_docs"
        ]
    ]


    top_5 = [
        row["document_id"]
        for row
        in ranked_documents[:5]
    ]


    top_10 = [
        row["document_id"]
        for row
        in ranked_documents[:10]
    ]


    recall_at_5 = (
        len(
            set(gold_docs)
            & set(top_5)
        )
        / len(gold_docs)
    )


    recall_at_10 = (
        len(
            set(gold_docs)
            & set(top_10)
        )
        / len(gold_docs)
    )


    first_relevant_rank = None


    for rank, row in enumerate(
        ranked_documents,
        start=1,
    ):
        if (
            row["document_id"]
            in gold_docs
        ):
            first_relevant_rank = rank
            break


    reciprocal_rank = (
        1.0
        / first_relevant_rank

        if first_relevant_rank
        else 0.0
    )


    recall_5_values.append(
        recall_at_5
    )

    recall_10_values.append(
        recall_at_10
    )

    reciprocal_ranks.append(
        reciprocal_rank
    )


    details.append(
        {
            "id": question["id"],
            "question": (
                question["question"]
            ),
            "gold_docs": "|".join(
                gold_docs
            ),
            "recall_at_5": (
                recall_at_5
            ),
            "recall_at_10": (
                recall_at_10
            ),
            "first_relevant_rank": (
                first_relevant_rank
                if first_relevant_rank
                is not None
                else ""
            ),
            "reciprocal_rank": (
                reciprocal_rank
            ),
            "top_1": (
                ranked_documents[0][
                    "document_id"
                ]
                if ranked_documents
                else ""
            ),
            "top_5": "|".join(
                top_5
            ),
            "top_10": "|".join(
                top_10
            ),
            "query_latency_ms": (
                query_latency_ms
            ),
        }
    )


mean_recall_5 = float(
    np.mean(recall_5_values)
)

mean_recall_10 = float(
    np.mean(recall_10_values)
)

mean_mrr = float(
    np.mean(reciprocal_ranks)
)

mean_query_ms = float(
    np.mean(query_latencies_ms)
)


# ============================================================
# EXPORT DÉTAILLÉ
# ============================================================

details_path = (
    OUTPUT_DIR
    / "details.csv"
)


with details_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            details[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        details
    )


# ============================================================
# RÉSUMÉ
# ============================================================

summary = {
    "model": DEFAULT_MODEL_NAME,
    "chunking": "fixed_300_60",
    "questions": len(questions),
    "chunks": len(chunks),
    "device": retriever.device,
    "embedding_dimension": (
        retriever.embedding_dimension
    ),
    "recall_at_5": (
        mean_recall_5
    ),
    "recall_at_10": (
        mean_recall_10
    ),
    "mrr": (
        mean_mrr
    ),
    "build_seconds": (
        build_seconds
    ),
    "mean_query_latency_ms": (
        mean_query_ms
    ),
}


summary_path = (
    OUTPUT_DIR
    / "summary.csv"
)


with summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary.keys()
        ),
    )

    writer.writeheader()

    writer.writerow(
        summary
    )


# ============================================================
# AFFICHAGE
# ============================================================

print()
print("=" * 90)
print("RÉSULTATS DENSE")
print("=" * 90)

print(
    f"Recall@5          : "
    f"{mean_recall_5:.4f}"
)

print(
    f"Recall@10         : "
    f"{mean_recall_10:.4f}"
)

print(
    f"MRR               : "
    f"{mean_mrr:.4f}"
)

print(
    f"Temps indexation  : "
    f"{build_seconds:.2f} s"
)

print(
    f"Latence moyenne   : "
    f"{mean_query_ms:.2f} ms/question"
)


print()
print("=" * 90)
print("COMPARAISON AVEC BM25")
print("=" * 90)

print(
    f"{'Méthode':<20}"
    f"{'Recall@5':>12}"
    f"{'Recall@10':>12}"
    f"{'MRR':>10}"
)

print("-" * 54)

print(
    f"{'BM25':<20}"
    f"{1.0000:>12.4f}"
    f"{1.0000:>12.4f}"
    f"{0.8306:>10.4f}"
)

print(
    f"{'Dense E5':<20}"
    f"{mean_recall_5:>12.4f}"
    f"{mean_recall_10:>12.4f}"
    f"{mean_mrr:>10.4f}"
)

print()
print(
    "Résultats écrits dans :",
    OUTPUT_DIR,
)
