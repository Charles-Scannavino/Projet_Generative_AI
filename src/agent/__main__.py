"""Démonstration en ligne de commande, pour n'importe quel backend de `ask()`.

uv run python -m src.agent "Combien de logements sociaux à Créteil ?"
uv run python -m src.agent "…" --backend agent_tools --provider ollama --json
uv run python -m src.agent "…" --backend agent --model qwen3:4b-instruct
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import get_args

from src.agent import BACKENDS, ask
from src.settings import Backend as Provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant HLM.")
    parser.add_argument("question")
    parser.add_argument("--backend", choices=BACKENDS, default="agent_tools")
    parser.add_argument(
        "--provider",
        choices=get_args(Provider),
        help="rag (défaut : ollama) et agent_tools (défaut : configs/agent.yaml).",
    )
    parser.add_argument("--model", help="Remplace le modèle par défaut du backend.")
    parser.add_argument("--retriever", choices=["hybrid", "hybrid_norerank", "bm25"])
    parser.add_argument(
        "--json",
        action="store_true",
        help="Affiche l'AssistantResponse complète en JSON.",
    )
    parser.add_argument(
        "--no-warmup", action="store_true", help="Ne pas précharger le retrieval."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    options: dict = {}
    if args.model:
        options["model"] = args.model
    if args.retriever:
        options["retriever"] = args.retriever
    if args.provider and args.backend in ("rag", "agent_tools"):
        options["provider"] = args.provider

    if not args.no_warmup and args.retriever != "bm25":
        from src.retrieval import warmup

        start = time.perf_counter()
        warmup()
        print(f"Retrieval préchargé en {time.perf_counter() - start:.1f} s\n")

    response = ask(args.question, backend=args.backend, **options)

    if args.json:
        print(response.model_dump_json(indent=2))
        return

    print(response.answer)
    print()

    if response.citations:
        print("Sources :")
        for citation in response.citations:
            print(f"  - {citation.document_id} — {citation.title} — {citation.url}")

    print("Outils :")
    for call in response.tool_calls:
        status = "ok" if call.ok else f"erreur {call.error}"
        print(f"  - {call.name}({call.arguments}) : {status}, {call.latency_ms:.0f} ms")

    cost = f"{response.cost_usd:.5f} $" if response.cost_usd is not None else "-"
    print(
        f"\n{response.backend} / {response.model} | statut : {response.stop_reason} | "
        f"abstention : {response.abstained} | {response.llm_calls} appels LLM | "
        f"tokens {response.tokens_in} in / {response.tokens_out} out | coût {cost} | "
        f"{response.latency_ms / 1000:.1f} s"
    )

    for warning in response.warnings:
        print(f"Avertissement : {warning}")


if __name__ == "__main__":
    main()
