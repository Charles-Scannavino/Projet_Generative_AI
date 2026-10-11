from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, median

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CORPUS_PATH = Path(
    "data/processed/rag_documents.jsonl"
)

OUTPUT_DIR = Path(
    "data/processed/chunks"
)

STATS_PATH = Path(
    "data/processed/chunking_stats.csv"
)


STRATEGIES = {
    "fixed_300_60": {
        "type": "fixed",
        "size": 300,
        "overlap": 60,
    },
    "fixed_500_100": {
        "type": "fixed",
        "size": 500,
        "overlap": 100,
    },
    "paragraph_400_80": {
        "type": "paragraph",
        "size": 400,
        "overlap": 80,
    },
}


# ============================================================
# UTILITAIRES
# ============================================================

def load_jsonl(path: Path) -> list[dict]:

    documents = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                documents.append(
                    json.loads(line)
                )

    return documents


def word_count(text: str) -> int:

    return len(
        str(text).split()
    )


# ============================================================
# CHUNKING FIXE
# ============================================================

def fixed_chunks(
    text: str,
    size: int,
    overlap: int,
) -> list[str]:

    words = str(text).split()

    if not words:
        return []

    if len(words) <= size:
        return [
            " ".join(words)
        ]

    step = size - overlap

    if step <= 0:
        raise ValueError(
            "L'overlap doit être inférieur à la taille du chunk."
        )

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + size,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += step

    return chunks


# ============================================================
# CHUNKING PAR PARAGRAPHES / BLOCS
# ============================================================

def split_blocks(
    text: str,
) -> list[str]:

    # Les corpus Service-Public et LEGI conservent
    # des retours à la ligne utiles.
    blocks = [
        block.strip()
        for block in re.split(
            r"\n+",
            str(text),
        )
        if block.strip()
    ]

    return blocks


def trailing_overlap_blocks(
    blocks: list[str],
    overlap_words: int,
) -> list[str]:

    if overlap_words <= 0:
        return []

    selected = []
    total = 0

    for block in reversed(blocks):

        block_words = word_count(
            block
        )

        selected.insert(
            0,
            block,
        )

        total += block_words

        if total >= overlap_words:
            break

    return selected


def paragraph_chunks(
    text: str,
    size: int,
    overlap: int,
) -> list[str]:

    blocks = split_blocks(
        text
    )

    if not blocks:
        return []

    if word_count(text) <= size:
        return [
            str(text).strip()
        ]

    chunks = []

    current = []
    current_words = 0


    def emit_current():

        if not current:
            return

        chunk = "\n\n".join(
            current
        ).strip()

        if chunk:
            chunks.append(chunk)


    for block in blocks:

        block_words = word_count(
            block
        )

        # Cas exceptionnel :
        # un seul paragraphe dépasse la taille cible.
        if block_words > size:

            if current:

                previous = list(
                    current
                )

                emit_current()

                current = trailing_overlap_blocks(
                    previous,
                    overlap,
                )

                current_words = sum(
                    word_count(x)
                    for x in current
                )

                # On ne mélange pas artificiellement
                # le paragraphe géant avec le précédent.
                current = []
                current_words = 0

            long_parts = fixed_chunks(
                block,
                size=size,
                overlap=overlap,
            )

            chunks.extend(
                long_parts
            )

            continue


        # Le bloc suivant ferait dépasser la cible.
        if (
            current
            and current_words + block_words > size
        ):

            previous = list(
                current
            )

            emit_current()

            current = trailing_overlap_blocks(
                previous,
                overlap,
            )

            current_words = sum(
                word_count(x)
                for x in current
            )

            # Eviter qu'un overlap trop gros empêche
            # d'ajouter le nouveau bloc.
            while (
                current
                and current_words + block_words > size
            ):

                removed = current.pop(0)

                current_words -= word_count(
                    removed
                )


        current.append(
            block
        )

        current_words += block_words


    emit_current()

    return chunks


# ============================================================
# CONSTRUCTION DES CHUNKS
# ============================================================

def build_strategy(
    documents: list[dict],
    strategy_name: str,
    config: dict,
) -> list[dict]:

    output = []

    for document in documents:

        document_id = str(
            document["document_id"]
        )

        text = str(
            document.get(
                "text",
                "",
            )
        ).strip()

        if config["type"] == "fixed":

            parts = fixed_chunks(
                text,
                size=config["size"],
                overlap=config["overlap"],
            )

        elif config["type"] == "paragraph":

            parts = paragraph_chunks(
                text,
                size=config["size"],
                overlap=config["overlap"],
            )

        else:

            raise ValueError(
                f"Type inconnu : {config['type']}"
            )


        if not parts:

            raise RuntimeError(
                f"Aucun chunk créé pour {document_id}"
            )


        n_chunks = len(parts)


        for chunk_index, chunk_text in enumerate(
            parts
        ):

            chunk_id = (
                f"{document_id}"
                f"__{strategy_name}"
                f"__{chunk_index:04d}"
            )


            # Le titre est ajouté à retrieval_text
            # pour améliorer la recherche,
            # mais le corps du chunk reste séparé.
            title = str(
                document.get(
                    "title",
                    "",
                )
            ).strip()


            retrieval_text = (
                f"{title}\n\n{chunk_text}"
                if title
                else chunk_text
            )


            output.append(
                {
                    "chunk_id": chunk_id,
                    "strategy": strategy_name,
                    "document_id": document_id,
                    "source": document.get(
                        "source"
                    ),
                    "source_name": document.get(
                        "source_name"
                    ),
                    "title": title,
                    "article": document.get(
                        "article"
                    ),
                    "url": document.get(
                        "url"
                    ),
                    "chunk_index": chunk_index,
                    "n_chunks_document": n_chunks,
                    "text": chunk_text,
                    "retrieval_text": retrieval_text,
                    "n_words": word_count(
                        chunk_text
                    ),
                }
            )

    return output


# ============================================================
# CHARGEMENT DU CORPUS
# ============================================================

documents = load_jsonl(
    CORPUS_PATH
)


print("=" * 80)
print("CHUNKING DU CORPUS RAG")
print("=" * 80)

print(
    "Documents source :",
    len(documents),
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


stats_rows = []


# ============================================================
# EXÉCUTION DES 3 STRATÉGIES
# ============================================================

for strategy_name, config in STRATEGIES.items():

    chunks = build_strategy(
        documents,
        strategy_name,
        config,
    )


    # --------------------------------------------------------
    # CONTRÔLES
    # --------------------------------------------------------

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    document_ids = {
        chunk["document_id"]
        for chunk in chunks
    }

    empty_chunks = [
        chunk["chunk_id"]
        for chunk in chunks
        if not chunk["text"].strip()
    ]

    duplicate_chunk_ids = (
        len(chunk_ids)
        - len(set(chunk_ids))
    )


    if empty_chunks:
        raise RuntimeError(
            f"{strategy_name}: chunks vides détectés."
        )

    if duplicate_chunk_ids:
        raise RuntimeError(
            f"{strategy_name}: chunk_id dupliqués."
        )

    if len(document_ids) != len(documents):
        raise RuntimeError(
            f"{strategy_name}: tous les documents "
            "ne sont pas couverts."
        )


    # --------------------------------------------------------
    # SAUVEGARDE JSONL
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{strategy_name}.jsonl"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )


    # --------------------------------------------------------
    # STATISTIQUES
    # --------------------------------------------------------

    lengths = [
        chunk["n_words"]
        for chunk in chunks
    ]

    chunks_per_document = (
        pd.Series(
            [
                chunk["document_id"]
                for chunk in chunks
            ]
        )
        .value_counts()
    )


    stats_rows.append(
        {
            "strategy": strategy_name,
            "documents": len(
                document_ids
            ),
            "chunks": len(
                chunks
            ),
            "chunks_per_document_mean": round(
                float(
                    chunks_per_document.mean()
                ),
                2,
            ),
            "words_min": min(
                lengths
            ),
            "words_mean": round(
                mean(lengths),
                1,
            ),
            "words_median": round(
                median(lengths),
                1,
            ),
            "words_p95": round(
                float(
                    pd.Series(
                        lengths
                    ).quantile(
                        0.95
                    )
                ),
                1,
            ),
            "words_max": max(
                lengths
            ),
        }
    )


    print()
    print("-" * 80)
    print(
        strategy_name
    )
    print("-" * 80)

    print(
        "Chunks :",
        len(chunks),
    )

    print(
        "Documents couverts :",
        len(document_ids),
    )

    print(
        "Chunks vides :",
        len(empty_chunks),
    )

    print(
        "Doublons chunk_id :",
        duplicate_chunk_ids,
    )

    print(
        "Chunks/document moyen :",
        round(
            float(
                chunks_per_document.mean()
            ),
            2,
        ),
    )

    print(
        "Mots/chunk :",
        f"min={min(lengths)}",
        f"moy={mean(lengths):.1f}",
        f"mediane={median(lengths):.1f}",
        f"p95={pd.Series(lengths).quantile(0.95):.1f}",
        f"max={max(lengths)}",
    )

    print(
        "Fichier :",
        output_path,
    )


# ============================================================
# TABLEAU COMPARATIF
# ============================================================

stats = pd.DataFrame(
    stats_rows
)

stats.to_csv(
    STATS_PATH,
    index=False,
)


print()
print("=" * 80)
print("COMPARAISON DES STRATÉGIES")
print("=" * 80)

print(
    stats.to_string(
        index=False
    )
)

print()
print(
    "Statistiques écrites dans :",
    STATS_PATH,
)
