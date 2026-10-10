from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SERVICE_PUBLIC_PATH = Path(
    "data/processed/service_public_hlm_documents.jsonl"
)

LEGI_PATH = Path(
    "data/processed/legi_hlm_documents.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/rag_documents.jsonl"
)

AUDIT_PATH = Path(
    "data/processed/rag_documents_audit.csv"
)


# ============================================================
# LECTURE JSONL
# ============================================================

def load_jsonl(path: Path) -> list[dict]:

    documents = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            documents.append(
                json.loads(line)
            )

    return documents


service_public = load_jsonl(
    SERVICE_PUBLIC_PATH
)

legi = load_jsonl(
    LEGI_PATH
)


print(
    "Service-Public :",
    len(service_public),
)

print(
    "LEGI           :",
    len(legi),
)


# ============================================================
# NORMALISATION MINIMALE
# ============================================================

normalized = []


for doc in service_public:

    item = dict(doc)

    item["source"] = "SERVICE_PUBLIC"

    item.setdefault(
        "source_name",
        "Service-Public.fr",
    )

    normalized.append(item)


for doc in legi:

    item = dict(doc)

    item["source"] = "LEGI"

    item.setdefault(
        "source_name",
        "Code de la construction et de l'habitation",
    )

    normalized.append(item)


# ============================================================
# CONTRÔLES
# ============================================================

ids = [
    str(doc.get("document_id", ""))
    for doc in normalized
]


empty_ids = [
    i
    for i in ids
    if not i.strip()
]


empty_texts = [
    doc.get("document_id")
    for doc in normalized
    if not str(
        doc.get("text", "")
    ).strip()
]


duplicate_ids = (
    len(ids)
    - len(set(ids))
)


missing_titles = [
    doc.get("document_id")
    for doc in normalized
    if not str(
        doc.get("title", "")
    ).strip()
]


print()
print("=" * 80)
print("CORPUS RAG COMBINÉ")
print("=" * 80)

print(
    "Documents totaux :",
    len(normalized),
)

print(
    "Identifiants vides :",
    len(empty_ids),
)

print(
    "Textes vides :",
    len(empty_texts),
)

print(
    "Titres vides :",
    len(missing_titles),
)

print(
    "Doublons document_id :",
    duplicate_ids,
)


if empty_ids:
    raise RuntimeError(
        "Des documents n'ont pas de document_id."
    )


if empty_texts:
    raise RuntimeError(
        "Des documents ont un texte vide."
    )


if duplicate_ids:
    raise RuntimeError(
        "Des document_id sont dupliqués."
    )


# ============================================================
# STATISTIQUES PAR SOURCE
# ============================================================

source_counts = pd.Series(
    [
        doc["source"]
        for doc in normalized
    ]
).value_counts()


print()
print("Répartition par source :")

print(
    source_counts.to_string()
)


lengths = pd.Series(
    [
        len(
            str(
                doc.get("text", "")
            ).split()
        )
        for doc in normalized
    ]
)


print()
print(
    "Longueur min (mots) :",
    int(lengths.min()),
)

print(
    "Longueur max (mots) :",
    int(lengths.max()),
)

print(
    "Longueur moyenne (mots) :",
    round(
        float(lengths.mean()),
        1,
    ),
)


# ============================================================
# SAUVEGARDE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    for doc in normalized:

        f.write(
            json.dumps(
                doc,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# AUDIT
# ============================================================

audit_rows = []

for doc in normalized:

    text = str(
        doc.get("text", "")
    )

    audit_rows.append(
        {
            "document_id": doc.get(
                "document_id"
            ),
            "source": doc.get(
                "source"
            ),
            "title": doc.get(
                "title"
            ),
            "n_chars": len(text),
            "n_words": len(
                text.split()
            ),
        }
    )


pd.DataFrame(
    audit_rows
).to_csv(
    AUDIT_PATH,
    index=False,
)


print()
print(
    "Corpus écrit dans :",
    OUTPUT_PATH,
)

print(
    "Audit écrit dans :",
    AUDIT_PATH,
)
