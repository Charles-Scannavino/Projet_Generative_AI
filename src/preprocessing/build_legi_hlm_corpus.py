from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RAW_ROOT = Path(
    "data/raw/legi/huggingface"
)

OUTPUT_PATH = Path(
    "data/processed/legi_hlm_documents.jsonl"
)

AUDIT_PATH = Path(
    "data/processed/legi_hlm_documents_audit.csv"
)


# ============================================================
# LOCALISER LE PARQUET CCH
# ============================================================

files = list(
    RAW_ROOT.rglob(
        "legi_code_de_la_construction_et_de_l_habitation_part_0.parquet"
    )
)

if len(files) != 1:
    raise RuntimeError(
        f"Nombre inattendu de fichiers CCH : {len(files)}"
    )

parquet_path = files[0]

print(
    "Parquet utilisé :",
    parquet_path,
)


# ============================================================
# CHARGER LES COLONNES UTILES
# ============================================================

columns = [
    "doc_id",
    "chunk_index",
    "nature",
    "category",
    "status",
    "title",
    "full_title",
    "subtitles",
    "number",
    "start_date",
    "end_date",
    "nota",
    "links",
    "text",
]

df = pd.read_parquet(
    parquet_path,
    columns=columns,
)


# ============================================================
# ARTICLES EN VIGUEUR
# ============================================================

current = df[
    (df["nature"] == "Article")
    & (df["category"] == "CODE")
    & (df["status"] == "VIGUEUR")
].copy()


print(
    "Lignes/chunks VIGUEUR :",
    len(current),
)

print(
    "doc_id uniques VIGUEUR :",
    current["doc_id"].nunique(),
)


# ============================================================
# PÉRIMÈTRE CIBLE
# ============================================================

subtitles = (
    current["subtitles"]
    .fillna("")
    .astype(str)
)

numbers = (
    current["number"]
    .fillna("")
    .astype(str)
)


title_iv = (
    subtitles.str.contains(
        "Titre IV : Rapports des organismes",
        case=False,
        regex=False,
    )
    &
    subtitles.str.contains(
        "loyer modéré et des bénéficiaires",
        case=False,
        regex=False,
    )
)

dalo = numbers.eq(
    "L300-1"
)

hlm_general = numbers.eq(
    "L411"
)


target_rows = current[
    title_iv
    | dalo
    | hlm_general
].copy()


print(
    "Lignes/chunks dans le périmètre cible :",
    len(target_rows),
)

print(
    "doc_id uniques dans le périmètre cible :",
    target_rows["doc_id"].nunique(),
)


# ============================================================
# RECONSTRUCTION :
# 1 doc_id = 1 article
# ============================================================

documents = []


for doc_id, group in target_rows.groupby(
    "doc_id",
    sort=False,
):

    group = (
        group
        .sort_values("chunk_index")
        .reset_index(drop=True)
    )

    first = group.iloc[0]

    chunks = []

    for text in group["text"]:

        text = str(
            text if pd.notna(text) else ""
        ).strip()

        if text:
            chunks.append(text)

    reconstructed_text = "\n\n".join(
        chunks
    ).strip()


    document = {
        "document_id": str(doc_id),
        "source": "LEGI",
        "source_name": (
            "Code de la construction et de l'habitation"
        ),
        "article": str(
            first["number"]
        ),
        "title": (
            f"Article {first['number']} — "
            "Code de la construction et de l'habitation"
        ),
        "legal_title": str(
            first["title"]
        ),
        "full_title": str(
            first["full_title"]
            if pd.notna(first["full_title"])
            else ""
        ),
        "subtitles": str(
            first["subtitles"]
            if pd.notna(first["subtitles"])
            else ""
        ),
        "status": str(
            first["status"]
        ),
        "start_date": str(
            first["start_date"]
            if pd.notna(first["start_date"])
            else ""
        ),
        "end_date": str(
            first["end_date"]
            if pd.notna(first["end_date"])
            else ""
        ),
        "nota": str(
            first["nota"]
            if pd.notna(first["nota"])
            else ""
        ),
        "links": (
            first["links"]
            if isinstance(
                first["links"],
                (list, dict)
            )
            else str(
                first["links"]
                if pd.notna(first["links"])
                else ""
            )
        ),
        "url": (
            "https://www.legifrance.gouv.fr/"
            f"codes/article_lc/{doc_id}"
        ),
        "n_source_chunks": int(
            len(group)
        ),
        "text": reconstructed_text,
        "n_chars": len(
            reconstructed_text
        ),
        "n_words": len(
            reconstructed_text.split()
        ),
    }

    documents.append(
        document
    )


# ============================================================
# CONTRÔLES QUALITÉ
# ============================================================

document_ids = [
    doc["document_id"]
    for doc in documents
]

empty_documents = [
    doc["document_id"]
    for doc in documents
    if not doc["text"].strip()
]

duplicate_doc_ids = (
    len(document_ids)
    - len(set(document_ids))
)

numbers_final = [
    doc["article"]
    for doc in documents
]


print()
print("=" * 100)
print("CORPUS LEGI HLM FINAL")
print("=" * 100)

print(
    "Documents reconstruits :",
    len(documents),
)

print(
    "Documents vides :",
    len(empty_documents),
)

print(
    "Doublons doc_id :",
    duplicate_doc_ids,
)

print(
    "Numéros d'article uniques :",
    len(set(numbers_final)),
)

print(
    "Documents multi-chunks :",
    sum(
        doc["n_source_chunks"] > 1
        for doc in documents
    ),
)


if empty_documents:

    raise RuntimeError(
        "Documents vides : "
        + ", ".join(
            empty_documents
        )
    )


if duplicate_doc_ids:

    raise RuntimeError(
        "Des doc_id sont dupliqués."
    )


# ============================================================
# STATISTIQUES
# ============================================================

prefixes = pd.Series(
    numbers_final
).str.extract(
    r"^([A-Z]+)",
    expand=False,
).value_counts()


print()
print(
    "Répartition par préfixe :"
)

print(
    prefixes.to_string()
)


lengths = pd.Series(
    [
        doc["n_words"]
        for doc in documents
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
# SAUVEGARDE JSONL
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    for doc in documents:

        f.write(
            json.dumps(
                doc,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# AUDIT CSV
# ============================================================

audit = pd.DataFrame(
    [
        {
            "document_id": doc["document_id"],
            "article": doc["article"],
            "status": doc["status"],
            "start_date": doc["start_date"],
            "end_date": doc["end_date"],
            "n_source_chunks": doc["n_source_chunks"],
            "n_chars": doc["n_chars"],
            "n_words": doc["n_words"],
            "subtitles": doc["subtitles"],
        }
        for doc in documents
    ]
)

audit.to_csv(
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
