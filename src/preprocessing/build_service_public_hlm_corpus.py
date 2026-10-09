from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DOCUMENTS_PATH = Path(
    "data/processed/service_public_documents.jsonl"
)

DECISIONS_PATH = Path(
    "data/processed/service_public_scope_decisions.csv"
)

GOLDEN_PATH = Path(
    "eval/golden.jsonl"
)

FINAL_DECISIONS_PATH = Path(
    "data/processed/service_public_scope_decisions_final.csv"
)

OUTPUT_PATH = Path(
    "data/processed/service_public_hlm_documents.jsonl"
)


# ------------------------------------------------------------
# Décisions manuelles sur les 13 social_keyword
# ------------------------------------------------------------

SOCIAL_KEYWORD_KEEP = {
    "F288",    # amiante, y compris bailleurs sociaux
    "F13737",  # troubles / gestionnaire HLM
    "F1702",   # insonorisation spécifique HLM
    "F39367",  # expropriation et priorité relogement HLM
    "F33667",  # chèque énergie en logement social
}

SOCIAL_KEYWORD_EXCLUDE = {
    "F37932",
    "F2407",
    "F1626",
    "F16104",
    "F16158",
    "F35779",
    "F1616",
    "F23263",
}


# ------------------------------------------------------------
# Charger les documents
# ------------------------------------------------------------

documents = {}

with DOCUMENTS_PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        doc = json.loads(line)
        doc_id = doc["document_id"]

        if doc_id in documents:
            raise RuntimeError(
                f"Document dupliqué : {doc_id}"
            )

        documents[doc_id] = doc


print("Documents source :", len(documents))


# ------------------------------------------------------------
# Charger les décisions initiales
# ------------------------------------------------------------

df = pd.read_csv(
    DECISIONS_PATH
).fillna("")


print("\nDécisions initiales :")
print(df["decision"].value_counts().to_string())


# ------------------------------------------------------------
# Résoudre les 33 REVIEW
# ------------------------------------------------------------

review_mask = (
    df["decision"] == "REVIEW"
)

n_review = int(review_mask.sum())

df.loc[
    review_mask,
    "decision",
] = "EXCLUDE"

df.loc[
    review_mask,
    "reason",
] = "manual_review_outside_core_hlm_scope"


# ------------------------------------------------------------
# Résoudre les social_keyword
# ------------------------------------------------------------

for doc_id in SOCIAL_KEYWORD_KEEP:

    mask = (
        df["document_id"] == doc_id
    )

    if mask.sum() != 1:
        raise RuntimeError(
            f"Document KEEP introuvable ou dupliqué : {doc_id}"
        )

    df.loc[
        mask,
        "decision",
    ] = "KEEP"

    df.loc[
        mask,
        "reason",
    ] = "manual_social_keyword_keep"


for doc_id in SOCIAL_KEYWORD_EXCLUDE:

    mask = (
        df["document_id"] == doc_id
    )

    if mask.sum() != 1:
        raise RuntimeError(
            f"Document EXCLUDE introuvable ou dupliqué : {doc_id}"
        )

    df.loc[
        mask,
        "decision",
    ] = "EXCLUDE"

    df.loc[
        mask,
        "reason",
    ] = "manual_social_keyword_exclude"


# ------------------------------------------------------------
# Vérifier qu'il ne reste aucun REVIEW
# ------------------------------------------------------------

remaining_review = df[
    df["decision"] == "REVIEW"
]

if not remaining_review.empty:
    raise RuntimeError(
        "Il reste des documents REVIEW."
    )


# ------------------------------------------------------------
# Lire les gold_docs
# ------------------------------------------------------------

gold_docs = set()

with GOLDEN_PATH.open(
    encoding="utf-8"
) as f:

    for line in f:
        line = line.strip()

        if not line:
            continue

        question = json.loads(line)

        gold_docs.update(
            question.get(
                "gold_docs",
                []
            )
        )


print(
    "Gold docs attendus :",
    len(gold_docs),
)


# ------------------------------------------------------------
# Construire le corpus final
# ------------------------------------------------------------

kept = df[
    df["decision"] == "KEEP"
].copy()


final_documents = []

for _, row in kept.iterrows():

    doc_id = row["document_id"]

    if doc_id not in documents:
        raise RuntimeError(
            f"{doc_id} absent du corpus source."
        )

    doc = dict(
        documents[doc_id]
    )

    # Métadonnées de périmètre utiles ensuite
    for col in [
        "subtheme_id",
        "subtheme_title",
        "dossier_id",
        "dossier_title",
        "breadcrumb",
    ]:
        if col in row.index:
            doc[col] = str(
                row.get(col, "")
            )

    doc["scope_decision"] = "KEEP"
    doc["scope_reason"] = str(
        row.get("reason", "")
    )

    final_documents.append(doc)


# ------------------------------------------------------------
# Contrôles qualité
# ------------------------------------------------------------

final_ids = {
    doc["document_id"]
    for doc in final_documents
}

missing_gold = (
    gold_docs - final_ids
)

empty_docs = [
    doc["document_id"]
    for doc in final_documents
    if not str(
        doc.get("text", "")
    ).strip()
]

duplicate_count = (
    len(final_documents)
    - len(final_ids)
)


if missing_gold:
    raise RuntimeError(
        "Gold docs manquants : "
        + ", ".join(
            sorted(missing_gold)
        )
    )

if empty_docs:
    raise RuntimeError(
        "Documents vides : "
        + ", ".join(empty_docs)
    )

if duplicate_count:
    raise RuntimeError(
        "Doublons dans le corpus final."
    )


# ------------------------------------------------------------
# Sauvegarde
# ------------------------------------------------------------

FINAL_DECISIONS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    FINAL_DECISIONS_PATH,
    index=False,
)


with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    for doc in final_documents:
        f.write(
            json.dumps(
                doc,
                ensure_ascii=False
            )
            + "\n"
        )


# ------------------------------------------------------------
# Résumé
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CORPUS SERVICE-PUBLIC HLM FINAL")
print("=" * 80)

print("Documents source       :", len(documents))
print("Documents conservés    :", len(final_documents))
print("Documents exclus       :", len(df) - len(final_documents))
print("Documents vides        :", len(empty_docs))
print("Doublons               :", duplicate_count)

print()
print("Gold docs attendus     :", len(gold_docs))
print(
    "Gold docs présents     :",
    len(gold_docs & final_ids)
)
print(
    "Gold docs manquants    :",
    sorted(missing_gold)
)

print()
print("Décisions finales :")
print(
    df["decision"]
    .value_counts()
    .to_string()
)

print()
print("33 REVIEW traités      :", n_review)
print(
    "Corpus final           :",
    OUTPUT_PATH
)
print(
    "Décisions finales      :",
    FINAL_DECISIONS_PATH
)
