"""Retrieval du corpus logement social : `from src.retrieval import search_docs`.

L'import de search_docs est paresseux (PEP 562) : importer seulement
src.retrieval.bm25, par exemple, ne charge pas PyTorch, et
`python -m src.retrieval.search` ne charge pas le module deux fois.
"""

__all__ = ["search_docs", "warmup"]


def __getattr__(name: str):
    if name in __all__:
        from src.retrieval import search

        return getattr(search, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
