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
