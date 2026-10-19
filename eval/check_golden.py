"""Contrôle d'intégrité du golden set.

Usage :
    uv run python eval/check_golden.py                 # contrôle seul
    uv run python eval/check_golden.py --bm25          # + difficulté BM25 par split

Vérifie, pour chaque ligne de eval/golden.jsonl :
- JSON valide, champs obligatoires présents, identifiants uniques ;
- `type`, `split` et `expected_tool` dans les valeurs autorisées ;
- chaque `gold_docs` / `acceptable_docs` existe dans le corpus RAG ;
- cohérence : question `rag` => au moins un gold_doc ; question à abstention
  => pas de gold_docs ; question sans réponse => `status: pending_data`.

Code de sortie 1 en cas d'erreur : utilisable en CI ou avant chaque commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden.jsonl"
CORPUS = ROOT / "data" / "processed" / "rag_documents.jsonl"
CHUNKS = ROOT / "data" / "processed" / "chunks" / "fixed_300_60.jsonl"

REQUIRED = {"id", "split", "type", "question", "answer", "gold_docs", "expected_tool", "should_abstain"}
TYPES = {"rag", "eligibility", "clarification", "rpls", "dvf", "out_of_scope", "prompt_injection", "causal_trap"}
SPLITS = {"dev", "test"}
TOOLS = {None, "search_docs", "check_eligibility", "social_housing_stats", "market_prices"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                sys.exit(f"{path.name}:{number}: JSON invalide ({error})")
    return rows


def check(rows: list[dict], corpus_ids: set[str]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    ids = Counter(row.get("id") for row in rows)
    errors += [f"id en double : {i}" for i, n in ids.items() if n > 1]

    for row in rows:
        rid = row.get("id", "?")
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"{rid}: champs manquants {sorted(missing)}")
            continue
        if row["type"] not in TYPES:
            errors.append(f"{rid}: type inconnu {row['type']!r}")
        if row["split"] not in SPLITS:
            errors.append(f"{rid}: split inconnu {row['split']!r}")
        if row["expected_tool"] not in TOOLS:
            errors.append(f"{rid}: expected_tool inconnu {row['expected_tool']!r}")
        for field in ("gold_docs", "acceptable_docs"):
            for doc in row.get(field, []):
                if doc not in corpus_ids:
                    errors.append(f"{rid}: {field} {doc} absent du corpus")
        if row["type"] == "rag" and not row["gold_docs"]:
            errors.append(f"{rid}: question rag sans gold_docs")
        if row["should_abstain"] and row["gold_docs"]:
            errors.append(f"{rid}: abstention attendue mais gold_docs non vide")
        if row["answer"] is None and row.get("status") != "pending_data":
            errors.append(f"{rid}: answer vide sans status pending_data")
        twin = row.get("twin_of")
        if twin and twin not in {r.get("id") for r in rows if r.get("split") == "dev"}:
            errors.append(f"{rid}: twin_of {twin} n'est pas une question dev")
        if row.get("status") == "pending_data":
            warnings.append(f"{rid}: réponse à compléter ({row['type']})")
    return errors, warnings


def bm25_difficulty(rows: list[dict]) -> None:
    """Hit@5 et MRR de BM25 seul, par split : un split trop facile n'apprend rien."""
    sys.path.insert(0, str(ROOT))
    from eval.metrics import chunks_to_documents, hit_at_k, reciprocal_rank
    from src.retrieval.bm25 import BM25Retriever

    chunks = load_jsonl(CHUNKS)
    retriever = BM25Retriever(chunks)
    by_split: dict[str, list[tuple[float, float]]] = {}
    rr_by_id: dict[str, float] = {}
    print("\nBM25 seul (Hit@5 / RR) sur les questions rag :")
    for row in rows:
        if row["type"] != "rag":
            continue
        relevant = row["gold_docs"] + row.get("acceptable_docs", [])
        ranked = chunks_to_documents(r.document_id for r in retriever.search(row["question"], top_k=100))
        hit, rr = hit_at_k(ranked, relevant, 5), reciprocal_rank(ranked, relevant)
        by_split.setdefault(row["split"], []).append((hit, rr))
        rr_by_id[row["id"]] = rr
        flag = "" if hit else "   <-- raté par BM25"
        print(f"  {row['id']} [{row['split']}] hit@5={hit:.0f} rr={rr:.2f}{flag}")
    for split, values in by_split.items():
        n = len(values)
        print(f"  => {split}: n={n} Hit@5={sum(v[0] for v in values) / n:.3f} MRR={sum(v[1] for v in values) / n:.3f}")
    twins = [(row["twin_of"], row["id"]) for row in rows if row.get("twin_of")]
    if twins:
        print("\nJumeaux (même sujet) : RR BM25 formulation titre (dev) -> langage courant (test)")
        for dev_id, test_id in twins:
            print(f"  {dev_id} {rr_by_id.get(dev_id, 0):.2f} -> {test_id} {rr_by_id.get(test_id, 0):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--bm25", action="store_true", help="mesure la difficulté avec BM25 seul")
    args = parser.parse_args()

    rows = load_jsonl(args.golden)
    corpus_ids = {doc["document_id"] for doc in load_jsonl(CORPUS)}
    errors, warnings = check(rows, corpus_ids)

    print(f"{len(rows)} questions | split : {dict(Counter(r.get('split') for r in rows))}")
    print(f"types : {dict(Counter(r.get('type') for r in rows))}")
    for warning in warnings:
        print("ATTENTION", warning)
    for error in errors:
        print("ERREUR", error)
    if args.bm25:
        bm25_difficulty(rows)
    if errors:
        sys.exit(1)
    print("OK : golden set valide.")


if __name__ == "__main__":
    main()
