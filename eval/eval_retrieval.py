"""Ablation unifiée du retrieval sur les questions RAG du golden set.

Méthodes comparées, toutes sur le chunking de référence fixed_300_60 :
- bm25                 : BM25 seul ;
- dense                : E5 seul ;
- hybrid               : fusion BM25 + dense par RRF ;
- hybrid+rerank_minilm : hybride, puis cross-encoder léger sur le top 30 ;
- hybrid+rerank_bge    : hybride, puis cross-encoder lourd sur le top 30.

Sorties :
- eval/retrieval/summary.csv : métriques moyennes, latence, temps de chargement ;
- eval/retrieval/details.csv : rang du premier gold_doc par question et méthode.

Commande :
    uv run python eval/eval_retrieval.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from eval.metrics import (
    chunks_to_documents,
    evaluate_question,
    mean_metrics,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.common import (
    CHUNKS_PATH,
    GOLDEN_PATH,
    load_chunks,
    load_jsonl,
)
from src.retrieval.config import (
    BM25_NAME,
    CANDIDATE_DEPTH,
    DENSE_MODEL_NAME,
    RERANK_TOP_N,
    RERANKER_MODELS,
    RRF_K,
)
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker


OUTPUT_DIR = ROOT / "eval" / "retrieval"

# Requête hors golden set, lancée une fois par méthode avant les mesures :
# le premier appel sur MPS est plus lent (initialisation des noyaux GPU).
WARMUP_QUERY = "logement social"

# Règle de choix du reranker par défaut, FIXÉE AVANT l'expérience :
# - on prend le reranker au meilleur MRR ;
# - mais si l'écart de MRR est < 0,02 (bruit sur 18 questions) ou si la
#   latence p95 du plus lourd dépasse 2 s, on prend le plus léger.
MRR_NOISE = 0.02
MAX_P95_MS = 2000.0
LIGHT_METHOD = "hybrid+rerank_minilm"
HEAVY_METHOD = "hybrid+rerank_bge"


# ============================================================
# DONNÉES
# ============================================================

questions = [
    row
    for row in load_jsonl(GOLDEN_PATH)
    if row.get("type") == "rag" and row.get("gold_docs")
]

chunks = load_chunks(CHUNKS_PATH)


print("=" * 90)
print("ABLATION RETRIEVAL — BM25 / DENSE / HYBRIDE / RERANKERS")
print("=" * 90)
print("Questions RAG :", len(questions))
print("Chunks        :", len(chunks), f"({CHUNKS_PATH.name})")
print("PyTorch       :", torch.__version__)
print()


# ============================================================
# CHARGEMENT DES COMPOSANTS (chronométré)
# ============================================================

start = time.perf_counter()
bm25 = BM25Retriever(chunks)
bm25_seconds = time.perf_counter() - start
print(f"BM25 construit en {bm25_seconds:.2f} s")

start = time.perf_counter()
dense = DenseRetriever(DENSE_MODEL_NAME)
from_cache = dense.load_or_build(chunks, CHUNKS_PATH)
dense_seconds = time.perf_counter() - start
print(
    f"Dense prêt en {dense_seconds:.2f} s "
    f"(modèle + embeddings {'chargés du cache' if from_cache else 'calculés'}) "
    f"| device {dense.device}"
)

hybrid = HybridRetriever(
    bm25,
    dense,
    rrf_k=RRF_K,
    candidate_depth=CANDIDATE_DEPTH,
)

rerankers = {}
reranker_seconds = {}

for key, model_name in RERANKER_MODELS.items():
    start = time.perf_counter()
    rerankers[key] = CrossEncoderReranker(model_name)
    reranker_seconds[key] = time.perf_counter() - start
    print(
        f"Reranker {key} chargé en {reranker_seconds[key]:.2f} s "
        f"| device {rerankers[key].device}"
    )

print()


# ============================================================
# MÉTHODES
#
# Chaque méthode renvoie la liste ordonnée des document_id des chunks
# classés. BM25 et dense classent TOUS les chunks (MRR exact) ;
# l'hybride classe les candidats fusionnés (100 par retriever).
# ============================================================

def run_bm25(query: str) -> list[str]:
    return [chunks[index]["document_id"] for index, _ in bm25.rank(query)]


def run_dense(query: str) -> list[str]:
    return [chunks[index]["document_id"] for index, _ in dense.rank(query)]


def run_hybrid(query: str) -> list[str]:
    return [
        candidate.chunk["document_id"]
        for candidate in hybrid.search(query)
    ]


def make_run_rerank(reranker: CrossEncoderReranker):
    def run_rerank(query: str) -> list[str]:
        candidates = reranker.rerank(
            query,
            hybrid.search(query),
            top_n=RERANK_TOP_N,
        )
        return [candidate.chunk["document_id"] for candidate in candidates]

    return run_rerank


hybrid_models = f"{BM25_NAME} + {DENSE_MODEL_NAME} (RRF k={RRF_K})"

methods = {
    "bm25": {
        "run": run_bm25,
        "setup_seconds": bm25_seconds,
        "models": BM25_NAME,
    },
    "dense": {
        "run": run_dense,
        "setup_seconds": dense_seconds,
        "models": DENSE_MODEL_NAME,
    },
    "hybrid": {
        "run": run_hybrid,
        "setup_seconds": bm25_seconds + dense_seconds,
        "models": hybrid_models,
    },
}

for key, reranker in rerankers.items():
    methods[f"hybrid+rerank_{key}"] = {
        "run": make_run_rerank(reranker),
        "setup_seconds": (
            bm25_seconds + dense_seconds + reranker_seconds[key]
        ),
        "models": f"{hybrid_models} + {reranker.model_name}",
    }


# ============================================================
# ÉVALUATION
# ============================================================

details = []
summary = []

for method, spec in methods.items():
    spec["run"](WARMUP_QUERY)

    rows = []
    latencies_ms = []

    for question in questions:
        gold_docs = [str(doc) for doc in question["gold_docs"]]

        start = time.perf_counter()
        chunk_documents = spec["run"](question["question"])
        latency_ms = (time.perf_counter() - start) * 1000

        ranked_documents = chunks_to_documents(chunk_documents)
        row = evaluate_question(ranked_documents, gold_docs)

        rows.append(row)
        latencies_ms.append(latency_ms)

        details.append(
            {
                "id": question["id"],
                "method": method,
                "question": question["question"],
                "gold_docs": "|".join(gold_docs),
                "first_relevant_rank": (
                    row["first_relevant_rank"]
                    if row["first_relevant_rank"] is not None
                    else ""
                ),
                "reciprocal_rank": row["reciprocal_rank"],
                "recall_at_1": row["recall_at_1"],
                "recall_at_5": row["recall_at_5"],
                "recall_at_10": row["recall_at_10"],
                "hit_at_1": row["hit_at_1"],
                "top_5": "|".join(ranked_documents[:5]),
                "latency_ms": latency_ms,
            }
        )

    means = mean_metrics(rows)

    summary.append(
        {
            "method": method,
            "questions": len(questions),
            "chunks": len(chunks),
            "recall_at_1": means["recall_at_1"],
            "recall_at_5": means["recall_at_5"],
            "recall_at_10": means["recall_at_10"],
            "mrr": means["mrr"],
            "hit_at_1": means["hit_at_1"],
            "mean_latency_ms": float(np.mean(latencies_ms)),
            # Percentile 95 par interpolation linéaire : sur 18 requêtes,
            # il se situe entre la 17e et la 18e latence triée.
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
            "setup_seconds": spec["setup_seconds"],
            "models": spec["models"],
        }
    )

    print(f"{method} évalué")


# ============================================================
# EXPORTS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for filename, rows in (
    ("summary.csv", summary),
    ("details.csv", details),
):
    with (OUTPUT_DIR / filename).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# TABLEAU COMPARATIF
# ============================================================

print()
print("=" * 90)
print("COMPARAISON")
print("=" * 90)

print(
    f"{'Méthode':<24}"
    f"{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}"
    f"{'moy. ms':>10}{'p95 ms':>10}{'charg. s':>10}"
)
print("-" * 86)

for row in summary:
    print(
        f"{row['method']:<24}"
        f"{row['recall_at_1']:>8.4f}"
        f"{row['recall_at_5']:>8.4f}"
        f"{row['recall_at_10']:>8.4f}"
        f"{row['mrr']:>8.4f}"
        f"{row['mean_latency_ms']:>10.1f}"
        f"{row['p95_latency_ms']:>10.1f}"
        f"{row['setup_seconds']:>10.2f}"
    )


# ============================================================
# QUESTIONS OÙ LES MÉTHODES DIVERGENT
# ============================================================

ranks = {}

for row in details:
    ranks.setdefault(row["id"], {})[row["method"]] = row["first_relevant_rank"]

method_names = list(methods)

print()
print("Rang du premier gold_doc, questions où les méthodes divergent :")
print(f"{'id':<6}" + "".join(f"{name:>22}" for name in method_names))

n_divergent = 0

for question_id, question_ranks in ranks.items():
    if len({str(question_ranks[name]) for name in method_names}) > 1:
        n_divergent += 1
        print(
            f"{question_id:<6}"
            + "".join(
                f"{str(question_ranks[name]) or '-':>22}"
                for name in method_names
            )
        )

print(
    f"{n_divergent} question(s) divergente(s) sur {len(ranks)} ; "
    "les autres ont le même rang pour toutes les méthodes."
)


# ============================================================
# CHOIX DU RERANKER PAR DÉFAUT (règle fixée à l'avance)
# ============================================================

by_method = {row["method"]: row for row in summary}

light = by_method[LIGHT_METHOD]
heavy = by_method[HEAVY_METHOD]
mrr_gap = heavy["mrr"] - light["mrr"]

if heavy["p95_latency_ms"] > MAX_P95_MS:
    chosen = light
    reason = (
        f"latence p95 de {HEAVY_METHOD} = "
        f"{heavy['p95_latency_ms']:.0f} ms > {MAX_P95_MS:.0f} ms"
    )
elif abs(mrr_gap) < MRR_NOISE:
    chosen = light
    reason = (
        f"écart de MRR |{mrr_gap:+.4f}| < {MRR_NOISE} "
        "(bruit sur 18 questions)"
    )
elif mrr_gap > 0:
    chosen = heavy
    reason = f"MRR supérieur de {mrr_gap:+.4f} et p95 ≤ {MAX_P95_MS:.0f} ms"
else:
    chosen = light
    reason = f"MRR supérieur de {-mrr_gap:+.4f}"

improves = chosen["mrr"] > by_method["hybrid"]["mrr"]

print()
print("=" * 90)
print("CHOIX DU RERANKER PAR DÉFAUT")
print("=" * 90)
print(f"Retenu : {chosen['method']} — {reason}")
print(
    f"MRR hybride = {by_method['hybrid']['mrr']:.4f} ; "
    f"MRR retenu = {chosen['mrr']:.4f} ; "
    f"le reranker {'améliore' if improves else 'N AMÉLIORE PAS'} l'hybride."
)


# ============================================================
# NON-RÉGRESSION : BM25 ET DENSE DOIVENT RETROUVER LEURS RÉSULTATS
# ============================================================

def read_reference(path: Path, **filters) -> dict:
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if all(row[key] == value for key, value in filters.items()):
                return row

    raise ValueError(f"Référence introuvable dans {path}")


references = {
    "bm25": read_reference(
        ROOT / "eval" / "bm25_chunking" / "summary.csv",
        strategy="fixed_300_60",
    ),
    "dense": read_reference(ROOT / "eval" / "dense" / "summary.csv"),
}

regression = False

for method, reference in references.items():
    for metric in ("recall_at_5", "recall_at_10", "mrr"):
        if abs(by_method[method][metric] - float(reference[metric])) > 1e-9:
            regression = True
            print(
                f"ÉCART {method} {metric} : "
                f"{by_method[method][metric]:.4f} "
                f"au lieu de {float(reference[metric]):.4f}"
            )

print()
print(
    "Non-régression BM25 / dense :",
    "ÉCHEC" if regression else "OK",
)
print("Résultats écrits dans :", OUTPUT_DIR.relative_to(ROOT))

if regression:
    sys.exit(1)
