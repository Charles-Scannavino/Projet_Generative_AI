from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "AgentPublic/legi"

CCH_PREFIX = (
    "data/legi-latest/"
    "legi_code_de_la_construction_et_de_l_habitation/"
)

RAW_DIR = Path(
    "data/raw/legi/huggingface"
)

AUDIT_PATH = Path(
    "data/processed/legi_cch_hlm_candidates.csv"
)


# ============================================================
# IDENTIFIER UNIQUEMENT LES FICHIERS DU CCH
# ============================================================

api = HfApi()

repo_files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset",
)

parquet_files = sorted(
    path
    for path in repo_files
    if path.startswith(CCH_PREFIX)
    and path.endswith(".parquet")
)


if not parquet_files:
    raise RuntimeError(
        "Aucun fichier Parquet du Code de la construction "
        "et de l'habitation n'a été trouvé."
    )


print("=" * 100)
print("FICHIERS CCH IDENTIFIÉS")
print("=" * 100)

print("Nombre de fichiers :", len(parquet_files))

for path in parquet_files:
    print("-", path)


# ============================================================
# TÉLÉCHARGEMENT
# ============================================================

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

local_paths = []

for filename in parquet_files:

    print()
    print("Téléchargement :", filename)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
    )

    local_paths.append(
        Path(local_path)
    )


# ============================================================
# CHARGEMENT DU CODE
# ============================================================

frames = []

for path in local_paths:

    print(
        "Lecture :",
        path.name,
    )

    frame = pd.read_parquet(path)

    frames.append(frame)


df = pd.concat(
    frames,
    ignore_index=True,
)


print()
print("=" * 100)
print("CODE DE LA CONSTRUCTION ET DE L'HABITATION")
print("=" * 100)

print("Nombre total de lignes :", len(df))

print()
print("Colonnes :")

for column in df.columns:
    print("-", column)


# ============================================================
# CONTRÔLES GÉNÉRAUX
# ============================================================

print()
print("Nature :")

print(
    df["nature"]
    .fillna("<NA>")
    .value_counts()
    .to_string()
)


print()
print("Statuts :")

print(
    df["status"]
    .fillna("<NA>")
    .value_counts()
    .to_string()
)


# ============================================================
# ARTICLES ACTUELLEMENT EN VIGUEUR
# ============================================================

current = df[
    (df["nature"] == "Article")
    & (df["category"] == "CODE")
    & (df["status"] == "VIGUEUR")
].copy()


print()
print(
    "Articles actuellement en vigueur :",
    len(current),
)


# ============================================================
# CANDIDATS HLM
#
# On utilise d'abord la hiérarchie juridique ("subtitles")
# plutôt qu'un simple mot-clé dans le texte.
# ============================================================

subtitles = (
    current["subtitles"]
    .fillna("")
    .astype(str)
)


hlm_structure = subtitles.str.contains(
    r"habitations?\s+à\s+loyer\s+modéré",
    case=False,
    regex=True,
)


# Bloc droit au logement / DALO.
# On ajoute provisoirement les articles L300-* afin de les auditer.
numbers = (
    current["number"]
    .fillna("")
    .astype(str)
)

dalo_structure = numbers.str.match(
    r"^L300(?:-|$)"
)


candidates = current[
    hlm_structure
    | dalo_structure
].copy()


# ============================================================
# MÉTADONNÉES D'AUDIT
# ============================================================

candidates["n_chars"] = (
    candidates["text"]
    .fillna("")
    .astype(str)
    .str.len()
)

candidates["n_words"] = (
    candidates["text"]
    .fillna("")
    .astype(str)
    .str.split()
    .str.len()
)


columns = [
    "doc_id",
    "number",
    "nature",
    "category",
    "status",
    "title",
    "start_date",
    "end_date",
    "subtitles",
    "n_chars",
    "n_words",
]

audit = candidates[
    [
        column
        for column in columns
        if column in candidates.columns
    ]
].copy()


AUDIT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

audit.to_csv(
    AUDIT_PATH,
    index=False,
)


# ============================================================
# RÉSUMÉ
# ============================================================

print()
print("=" * 100)
print("AUDIT HLM / DALO")
print("=" * 100)

print(
    "Articles CCH en vigueur :",
    len(current),
)

print(
    "Articles candidats HLM/DALO :",
    len(candidates),
)

print()
print(
    "Répartition législatif / réglementaire :"
)

print(
    candidates["number"]
    .fillna("")
    .astype(str)
    .str.extract(r"^([A-Z]+)", expand=False)
    .value_counts()
    .to_string()
)


print()
print("Premiers candidats :")

preview_columns = [
    "doc_id",
    "number",
    "subtitles",
]

print(
    candidates[
        preview_columns
    ]
    .head(30)
    .to_string(
        index=False,
        max_colwidth=100,
    )
)


print()
print(
    "Audit écrit dans :",
    AUDIT_PATH,
)
