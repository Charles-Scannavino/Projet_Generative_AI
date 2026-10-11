from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_PATH = Path("eval/golden.jsonl")

CHUNK_DIR = Path(
    "data/processed/chunks"
)

OUTPUT_DIR = Path(
    "eval/bm25_chunking"
)

STRATEGIES = [
    "fixed_300_60",
    "fixed_500_100",
    "paragraph_400_80",
]


# ============================================================
# LECTURE JSONL
# ============================================================

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


# ============================================================
# TOKENISATION SIMPLE POUR BM25
# ============================================================

def tokenize(text: str) -> list[str]:

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


# ============================================================
# GOLDEN SET
# ============================================================

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


print("=" * 90)
print("ÉVALUATION BM25 DES STRATÉGIES DE CHUNKING")
print("=" * 90)

print(
    "Questions RAG évaluées :",
    len(questions),
)


if len(questions) != 18:
    print(
        "ATTENTION : 18 questions RAG étaient attendues."
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


summary_results = []


# ============================================================
# ÉVALUATION DES STRATÉGIES
# ============================================================

for strategy in STRATEGIES:

    chunk_path = (
        CHUNK_DIR
        / f"{strategy}.jsonl"
    )

    chunks = load_jsonl(
        chunk_path
    )


    print()
    print("-" * 90)
    print(strategy)
    print("-" * 90)

    print(
        "Chunks indexés :",
        len(chunks),
    )


    # ========================================================
    # CONSTRUCTION DE L'INDEX BM25
    # ========================================================

    tokenized_corpus = [
        tokenize(
            chunk.get(
                "retrieval_text",
                chunk.get("text", ""),
            )
        )
        for chunk in chunks
    ]


    bm25 = BM25Okapi(
        tokenized_corpus
    )


    question_results = []

    recall_5_values = []
    recall_10_values = []
    reciprocal_ranks = []


    # ========================================================
    # ÉVALUATION QUESTION PAR QUESTION
    # ========================================================

    for question in questions:

        query_tokens = tokenize(
            question["question"]
        )


        scores = bm25.get_scores(
            query_tokens
        )


        # Tri décroissant.
        # stable = résultat reproductible en cas d'égalité.
        ranked_chunk_indices = np.argsort(
            -scores,
            kind="stable",
        )


        # ====================================================
        # CHUNKS -> DOCUMENTS
        #
        # Comme les chunks sont déjà triés par score,
        # le premier chunk rencontré pour un document
        # correspond à son meilleur score BM25.
        # ====================================================

        ranked_documents = []

        seen_documents = set()


        for index in ranked_chunk_indices:

            chunk = chunks[
                int(index)
            ]

            document_id = str(
                chunk["document_id"]
            )


            if document_id in seen_documents:
                continue


            seen_documents.add(
                document_id
            )


            ranked_documents.append(
                {
                    "document_id": document_id,
                    "chunk_id": chunk[
                        "chunk_id"
                    ],
                    "score": float(
                        scores[
                            int(index)
                        ]
                    ),
                    "title": chunk.get(
                        "title",
                        "",
                    ),
                }
            )


            # Suffisant pour Recall@10
            # et diagnostic des erreurs.
            if len(ranked_documents) >= 20:
                break


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


        # ====================================================
        # RECALL@5
        # ====================================================

        recall_at_5 = (
            len(
                set(gold_docs)
                & set(top_5)
            )
            / len(gold_docs)
        )


        # ====================================================
        # RECALL@10
        # ====================================================

        recall_at_10 = (
            len(
                set(gold_docs)
                & set(top_10)
            )
            / len(gold_docs)
        )


        # ====================================================
        # MRR
        # ====================================================

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


        question_results.append(
            {
                "id": question["id"],
                "question": question[
                    "question"
                ],
                "gold_docs": "|".join(
                    gold_docs
                ),
                "recall_at_5": recall_at_5,
                "recall_at_10": recall_at_10,
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
            }
        )


    # ========================================================
    # MÉTRIQUES MOYENNES
    # ========================================================

    mean_recall_5 = float(
        np.mean(
            recall_5_values
        )
    )

    mean_recall_10 = float(
        np.mean(
            recall_10_values
        )
    )

    mean_mrr = float(
        np.mean(
            reciprocal_ranks
        )
    )


    summary_results.append(
        {
            "strategy": strategy,
            "questions": len(
                questions
            ),
            "chunks": len(
                chunks
            ),
            "recall_at_5": (
                mean_recall_5
            ),
            "recall_at_10": (
                mean_recall_10
            ),
            "mrr": mean_mrr,
        }
    )


    print(
        f"Recall@5  : {mean_recall_5:.4f}"
    )

    print(
        f"Recall@10 : {mean_recall_10:.4f}"
    )

    print(
        f"MRR        : {mean_mrr:.4f}"
    )


    # ========================================================
    # EXPORT DÉTAILLÉ
    # ========================================================

    detail_path = (
        OUTPUT_DIR
        / f"{strategy}_details.csv"
    )


    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                question_results[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            question_results
        )


# ============================================================
# RÉSUMÉ GLOBAL
# ============================================================

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
        fieldnames=[
            "strategy",
            "questions",
            "chunks",
            "recall_at_5",
            "recall_at_10",
            "mrr",
        ],
    )

    writer.writeheader()

    writer.writerows(
        summary_results
    )


# ============================================================
# AFFICHAGE FINAL
# ============================================================

print()
print("=" * 90)
print("COMPARAISON BM25")
print("=" * 90)

print(
    f"{'Stratégie':<24}"
    f"{'Chunks':>8}"
    f"{'Recall@5':>12}"
    f"{'Recall@10':>12}"
    f"{'MRR':>10}"
)

print("-" * 66)


for result in summary_results:

    print(
        f"{result['strategy']:<24}"
        f"{result['chunks']:>8}"
        f"{result['recall_at_5']:>12.4f}"
        f"{result['recall_at_10']:>12.4f}"
        f"{result['mrr']:>10.4f}"
    )


print()
print(
    "Résumé écrit dans :",
    summary_path,
)
