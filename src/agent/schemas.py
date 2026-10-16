"""Contrat de sortie de l'assistant, partagé par l'UI et eval/run_eval.py.

Toute implémentation (baseline BM25, RAG simple, agent avec outils) renvoie
un `AssistantResponse`. L'interface Streamlit et l'évaluation ne dépendent
que de ce contrat : on peut changer de backend sans toucher à l'une ou l'autre.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

StopReason = Literal[
    "answered",             # réponse appuyée sur des sources ou un outil
    "abstained",            # sources insuffisantes / hors périmètre / refus
    "needs_clarification",  # il manque une information (commune, foyer...)
    "max_steps",            # l'agent a atteint sa limite d'étapes
    "error",                # erreur technique (Ollama arrêté, timeout...)
]


class Citation(BaseModel):
    document_id: str
    source: str
    title: str
    url: str
    article: str | None = None


class Passage(BaseModel):
    """Passage fourni au modèle, affiché dans la trace de l'UI."""

    rank: int
    document_id: str
    source: str
    title: str
    url: str
    text: str
    score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0


class AssistantResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    abstained: bool = False
    stop_reason: StopReason = "answered"
    warnings: list[str] = Field(
        default_factory=list,
        description="Anomalies non bloquantes (ex. citation inventée supprimée).",
    )
    backend: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    latency_ms: float = 0.0
    latency_breakdown_ms: dict[str, float] = Field(default_factory=dict)
    cost_eur: float = 0.0
