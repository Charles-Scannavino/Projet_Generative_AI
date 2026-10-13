"""Configuration du retrieval, regroupée en un seul endroit.

Toutes les valeurs ci-dessous sont des choix standard et documentés.
Elles n'ont PAS été optimisées sur le golden set : avec 18 questions,
régler des hyperparamètres reviendrait à sur-apprendre ces questions.
"""

from __future__ import annotations

from pathlib import Path


# ============================================================
# CHEMINS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

# Chunking de référence figé le 11 octobre 2026 (1 357 chunks).
CHUNKS_PATH = (
    ROOT_DIR / "data" / "processed" / "chunks" / "fixed_300_60.jsonl"
)

GOLDEN_PATH = ROOT_DIR / "eval" / "golden.jsonl"

# Cache des embeddings denses (ignoré par Git).
INDEX_DIR = ROOT_DIR / "data" / "indexes"


# ============================================================
# MODÈLES
# ============================================================

# BM25 : implémentation rank_bm25, paramètres par défaut
# (k1 = 1.5, b = 0.75, epsilon = 0.25).
BM25_NAME = "rank_bm25.BM25Okapi"

# Dense : E5 multilingue small, 384 dimensions.
DENSE_MODEL_NAME = "intfloat/multilingual-e5-small"


# ============================================================
# FUSION HYBRIDE (RRF)
# ============================================================

# Constante k de la formule 1 / (k + rang) : valeur de l'article
# original (Cormack et al., 2009), utilisée par défaut partout.
RRF_K = 60

# Nombre de chunks fournis par chaque retriever avant la fusion.
CANDIDATE_DEPTH = 100


# ============================================================
# RERANKER (CROSS-ENCODER)
# ============================================================

# Deux cross-encoders multilingues comparés à l'étape d'ablation.
RERANKER_MODELS = {
    # Léger : MiniLM multilingue entraîné sur mMARCO (MS MARCO traduit).
    "minilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    # Plus précis et plus lourd : base XLM-RoBERTa large (bge-m3).
    "bge": "BAAI/bge-reranker-v2-m3",
}

# Nombre de candidats de l'hybride reclassés par le cross-encoder.
RERANK_TOP_N = 30

# Longueur maximale (en tokens) de la paire (question, chunk).
RERANKER_MAX_LENGTH = 512

# Reranker par défaut, choisi par la règle fixée avant l'ablation
# (eval/eval_retrieval.py, 13 octobre 2026). bge a le meilleur MRR
# (0.9722 contre 0.9444), mais sa latence p95 (4 428 ms par requête)
# dépasse le seuil de 2 s sur le MacBook Pro M1 : on garde MiniLM
# (p95 = 503 ms). bge reste utilisable en changeant cette clé.
DEFAULT_RERANKER = "minilm"
DEFAULT_RERANKER_MODEL = RERANKER_MODELS[DEFAULT_RERANKER]


# ============================================================
# search_docs
# ============================================================

DEFAULT_K = 5

# Au-delà de RERANK_TOP_N, les passages ne seraient plus reclassés.
MAX_K = RERANK_TOP_N
