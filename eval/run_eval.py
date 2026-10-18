"""Évaluation end-to-end sur le golden set (commande exigée par l'énoncé).

    uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/

Options utiles :
    --backend baseline|rag|agent|agent_tools   (défaut : agent)
    --provider openrouter|ollama   (agent_tools uniquement, défaut : configs/agent.yaml)
    --retriever hybrid|hybrid_norerank|bm25
    --model qwen3:4b-instruct      (tout modèle Ollama)
    --judge                        juge LLM : correct / partiel / incorrect vs réponse de référence
    --split dev|test               ne garder qu'un split
    --run-name agent_qwen3_4b      copie aussi les résultats dans eval/runs/<nom>/
    --resume                       reprend un run interrompu (ids déjà présents ignorés)
    --compare                      tableau comparatif de tous les eval/runs/*/summary.json

Écrit :
- <out>/results.jsonl : une ligne par question (réponse, sources citées, documents récupérés,
  tokens in/out, latence, appels d'outils, stop reason, métriques) ;
- <out>/summary.json et <out>/summary.md : métriques agrégées par split et par type.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import hit_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from src.agent import BACKENDS, ask  # noqa: E402
from src.agent import ollama as _ollama  # noqa: E402,F401  (charge le .env avant de lire les variables)
from src.agent.schemas import AssistantResponse  # noqa: E402

ACTION_TOOLS = {"check_eligibility", "social_housing_stats", "market_prices"}
# Coût par million de tokens (0 en local). Pour un modèle API : exporter ces variables.
PRICE_IN = float(os.environ.get("COST_EUR_PER_MTOK_IN", "0"))
PRICE_OUT = float(os.environ.get("COST_EUR_PER_MTOK_OUT", "0"))

JUDGE_PROMPT = """Tu évalues la réponse d'un assistant sur le logement social.
Compare la RÉPONSE à la RÉFÉRENCE (qui peut décrire un comportement attendu, ex. une abstention).
- "correct" : mêmes faits essentiels / bon comportement, sans erreur ;
- "partial" : en partie juste, ou incomplet sur un point important ;
- "incorrect" : faux, inventé, ou mauvais comportement (ex. répond alors qu'il fallait s'abstenir).
Ne tiens pas compte du style. Réponds au format JSON demandé."""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_question(row: dict, response: AssistantResponse) -> dict:
    """Métriques déterministes d'une question (sans juge)."""
    metrics: dict = {}
    relevant = row.get("gold_docs", []) + row.get("acceptable_docs", [])
    retrieved = response.retrieved_document_ids
    if row["type"] == "rag" and row.get("gold_docs"):
        metrics["hit_at_5"] = hit_at_k(retrieved, relevant, 5)
        metrics["recall_at_5"] = recall_at_k(retrieved, row["gold_docs"], 5)
        metrics["rr"] = reciprocal_rank(retrieved, relevant)
    cited = [c.document_id for c in response.citations]
    if row["type"] == "rag" and relevant and not row["should_abstain"]:
        metrics["citation_precision"] = (sum(d in relevant for d in cited) / len(cited)) if cited else 0.0
        metrics["cites_relevant_doc"] = float(any(d in relevant for d in cited))
    metrics["invented_citation"] = float(any("hors contexte" in w for w in response.warnings))
    metrics["unsupported_number"] = float(any(w.startswith("chiffre") for w in response.warnings))
    metrics["abstention_ok"] = float(response.abstained == row["should_abstain"])
    called = {c.name for c in response.tool_calls}
    expected = row.get("expected_tool")
    if expected:
        metrics["routing_ok"] = float(expected in called)
    else:
        metrics["routing_ok"] = float(not (called & ACTION_TOOLS))
    if "expected" in row:
        outputs = [c.output for c in response.tool_calls if c.name == "check_eligibility" and c.ok]
        exp = row["expected"]
        metrics["eligibility_exact"] = float(any(
            o.get("zone") == exp["zone"] and o.get("categorie") == exp["categorie"]
            and o.get("eligible") == {"plai": exp["plai"], "plus": exp["plus"], "pls": exp["pls"]}
            for o in outputs
        ))
    metrics["error"] = float(response.stop_reason == "error")
    return metrics


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "justification": {"type": "string"},
    },
    "required": ["verdict", "justification"],
    "additionalProperties": False,
}


def judge(row: dict, response: AssistantResponse, model: str, provider: str = "ollama") -> dict:
    """Juge LLM : Ollama (local) ou OpenRouter (API, coût réel enregistré)."""
    from pydantic import BaseModel

    class Verdict(BaseModel):
        verdict: str
        justification: str

    payload = {"question": row["question"], "reference": row["answer"], "reponse": response.answer,
               "abstention_attendue": row["should_abstain"]}
    messages = [{"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    cost_usd = None
    if provider == "openrouter":
        from src.agent.llm import build_client

        result = build_client("openrouter", model).complete_json(messages, VERDICT_SCHEMA, "verdict")
        content, tokens_in, tokens_out, cost_usd = result.content, result.tokens_in, result.tokens_out, result.cost_usd
    else:
        from src.agent import ollama

        result = ollama.chat(messages, model=model, schema=VERDICT_SCHEMA)
        content, tokens_in, tokens_out = result.content, result.tokens_in, result.tokens_out
    verdict = Verdict.model_validate_json(content)
    return {"judge_verdict": verdict.verdict, "judge_reason": verdict.justification,
            "judge_score": {"correct": 1.0, "partial": 0.5}.get(verdict.verdict, 0.0),
            "judge_model": model, "judge_tokens_in": tokens_in, "judge_tokens_out": tokens_out,
            "judge_cost_usd": cost_usd}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def aggregate(records: list[dict]) -> dict:
    def mean_of(rows: list[dict], key: str) -> float | None:
        values = [r["metrics"][key] for r in rows if key in r["metrics"]]
        return round(sum(values) / len(values), 4) if values else None

    keys = ["hit_at_5", "recall_at_5", "rr", "citation_precision", "cites_relevant_doc", "invented_citation",
            "abstention_ok", "routing_ok", "eligibility_exact", "judge_score", "unsupported_number", "error"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups["ALL"].append(record)
        groups[f"split={record['split']}"].append(record)
        groups[f"type={record['type']}"].append(record)
    order = lambda name: (name != "ALL", not name.startswith("split"), name)  # noqa: E731
    groups = dict(sorted(groups.items(), key=lambda item: order(item[0])))
    latencies = [r["latency_ms"] for r in records]
    return {
        "n": len(records),
        "backend": records[0]["backend"] if records else None,
        "model": records[0]["model"] if records else None,
        "groups": {name: {"n": len(rows), **{k: mean_of(rows, k) for k in keys}} for name, rows in groups.items()},
        "latency_ms": {"mean": round(statistics.mean(latencies), 1) if latencies else 0,
                       "p50": round(percentile(latencies, 0.5), 1), "p95": round(percentile(latencies, 0.95), 1)},
        "tokens_in_total": sum(r["tokens_in"] for r in records),
        "tokens_out_total": sum(r["tokens_out"] for r in records),
        "tool_calls_total": sum(len(r["tool_calls"]) for r in records),
        "cost_eur_total": round(sum(r["cost_eur"] for r in records), 6),
        # Coût réel renvoyé par l'API (backend agent_tools via OpenRouter) ; None en local.
        "judge_cost_usd_total": (
            round(sum(r["metrics"].get("judge_cost_usd") or 0 for r in records), 6)
            if any(r["metrics"].get("judge_cost_usd") is not None for r in records) else None
        ),
        "cost_usd_total": (
            round(sum(r["cost_usd"] for r in records if r.get("cost_usd") is not None), 6)
            if any(r.get("cost_usd") is not None for r in records) else None
        ),
    }


def summary_markdown(summary: dict, title: str) -> str:
    cols = ["n", "hit_at_5", "rr", "cites_relevant_doc", "citation_precision", "abstention_ok", "routing_ok",
            "eligibility_exact", "judge_score", "invented_citation", "unsupported_number", "error"]
    lines = [f"# {title}", "", f"Backend : `{summary['backend']}` | modèle : `{summary['model']}`", "",
             "| groupe | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, values in summary["groups"].items():
        cells = ["-" if values.get(c) is None else (f"{values[c]:.2f}" if isinstance(values[c], float) else str(values[c])) for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lat = summary["latency_ms"]
    lines += ["", f"Latence : moyenne {lat['mean']:.0f} ms, p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms",
              f"Tokens : {summary['tokens_in_total']} in / {summary['tokens_out_total']} out | "
              f"appels d'outils : {summary['tool_calls_total']} | coût : {summary['cost_eur_total']:.4f} €"
              + (f" | coût API : {summary['cost_usd_total']:.4f} $" if summary.get("cost_usd_total") is not None else "")]
    return "\n".join(lines) + "\n"


def compare_runs(runs_dir: Path) -> None:
    rows = []
    for path in sorted(runs_dir.glob("*/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        test = s["groups"].get("split=test", {})
        all_ = s["groups"]["ALL"]
        rows.append((path.parent.name, s["backend"], s["model"], test.get("hit_at_5"), test.get("rr"),
                     all_.get("judge_score"), all_.get("abstention_ok"), all_.get("routing_ok"),
                     s["latency_ms"]["p50"], s["latency_ms"]["p95"], s["tokens_in_total"] + s["tokens_out_total"],
                     s["cost_eur_total"]))
    header = "| run | backend | modèle | Hit@5 test | MRR test | juge | abstention | routage | p50 ms | p95 ms | tokens | coût € |"
    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
    text = "\n".join([header, "|---" * 12 + "|"] + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows])
    (runs_dir / "comparison.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation end-to-end sur le golden set.")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "golden.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "eval")
    parser.add_argument("--backend", choices=BACKENDS, default="agent")
    parser.add_argument("--retriever", choices=["hybrid", "hybrid_norerank", "bm25"], default=None)
    parser.add_argument("--model", default=None, help="modèle Ollama (défaut : OLLAMA_MODEL ou qwen3:4b-instruct), "
                        "ou modèle du fournisseur pour agent_tools")
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=None,
                        help="fournisseur du backend agent_tools (défaut : configs/agent.yaml)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-provider", choices=["ollama", "openrouter"], default="ollama",
                        help="fournisseur du juge ; avec openrouter, --judge-model est obligatoire")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.judge and args.judge_provider == "openrouter" and not args.judge_model:
        parser.error("--judge-provider openrouter exige --judge-model (ex. google/gemini-2.5-flash)")

    runs_dir = args.out / "runs"
    if args.compare:
        compare_runs(runs_dir)
        return

    rows = load_jsonl(args.questions)
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    rows = [r for r in rows if r.get("status") != "pending_data"][: args.limit]
    options: dict = {"k": args.k}
    if args.retriever:
        options["retriever"] = args.retriever
    if args.model and args.backend != "baseline":
        options["model"] = args.model
    if args.provider and args.backend == "agent_tools":
        options["provider"] = args.provider

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    done: dict[str, dict] = {}
    if args.resume and results_path.exists():
        done = {r["id"]: r for r in load_jsonl(results_path)}
    records = [done[r["id"]] for r in rows if r["id"] in done]
    mode = "a" if args.resume else "w"

    if args.backend in ("rag", "agent", "agent_tools"):
        from src.agent.retrieval import get_search

        get_search(options.get("retriever", "hybrid"))("échauffement logement social", k=1)  # charge les modèles

    with results_path.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            if row["id"] in done:
                continue
            start = time.perf_counter()
            try:
                response = ask(row["question"], backend=args.backend, **options)
            except Exception as error:  # une question ne doit pas arrêter tout le run
                response = AssistantResponse(question=row["question"], answer=f"Exception : {error}", abstained=True,
                                             stop_reason="error", backend=args.backend, model=str(options.get("model")))
            metrics = score_question(row, response)
            record = {
                "id": row["id"], "split": row["split"], "type": row["type"], "question": row["question"],
                "reference_answer": row["answer"], "gold_docs": row.get("gold_docs", []),
                "answer": response.answer,
                "cited_sources": [c.model_dump() for c in response.citations],
                "retrieved_document_ids": response.retrieved_document_ids,
                "tokens_in": response.tokens_in, "tokens_out": response.tokens_out,
                "latency_ms": round(response.latency_ms, 1),
                "latency_breakdown_ms": {k: round(v, 1) for k, v in response.latency_breakdown_ms.items()},
                "tool_calls": [c.model_dump(mode="json") for c in response.tool_calls],
                "stop_reason": response.stop_reason, "abstained": response.abstained,
                "warnings": response.warnings, "backend": response.backend, "model": response.model,
                "llm_calls": response.llm_calls,
                "cost_eur": (response.tokens_in * PRICE_IN + response.tokens_out * PRICE_OUT) / 1e6,
                "cost_usd": response.cost_usd, "prompt_version": response.prompt_version,
                "metrics": metrics,
            }
            if args.judge and row["answer"]:
                try:
                    judge_model = args.judge_model or options.get("model") or _ollama.DEFAULT_MODEL
                    record["metrics"].update(judge(row, response, judge_model, args.judge_provider))
                except Exception as error:
                    record["judge_error"] = str(error)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            print(f"[{index}/{len(rows)}] {row['id']} {response.stop_reason:<19} "
                  f"{(time.perf_counter() - start):6.1f}s  {json.dumps(metrics)[:110]}")

    summary = aggregate(records)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = summary_markdown(summary, f"Résultats ({args.run_name or args.backend})")
    (args.out / "summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    if args.run_name:
        target = runs_dir / args.run_name
        target.mkdir(parents=True, exist_ok=True)
        for name in ("results.jsonl", "summary.json", "summary.md"):
            shutil.copy(args.out / name, target / name)
        print(f"Copié dans {target}")


if __name__ == "__main__":
    main()
