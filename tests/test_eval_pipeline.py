"""Tests du contrat AssistantResponse et du scoring de run_eval, sans Ollama ni PyTorch."""

import json
from pathlib import Path

from eval.run_eval import score_question
from src.agent import ask
from src.agent.schemas import AssistantResponse, Citation, ToolCall
from src.tools.eligibility import check_eligibility

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval" / "golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


def test_baseline_backend_returns_contract():
    response = ask("Comment faire un recours Dalo ?", backend="baseline", retriever="bm25")
    assert isinstance(response, AssistantResponse)
    assert response.citations and response.retrieved_document_ids
    assert response.tool_calls[0].name == "search_docs"


def test_scoring_rewards_exact_eligibility_and_routing():
    row = GOLDEN["q030"]
    output = check_eligibility(dict(revenu_fiscal_reference=28000, nb_personnes=3, commune="Lyon",
                                    en_couple=True, personnes_a_charge=1)).model_dump()
    response = AssistantResponse(question=row["question"], answer="ok", backend="agent", model="test",
                                 tool_calls=[ToolCall(name="check_eligibility", output=output)])
    metrics = score_question(row, response)
    assert metrics["eligibility_exact"] == 1.0 and metrics["routing_ok"] == 1.0 and metrics["abstention_ok"] == 1.0


def test_scoring_penalizes_answering_an_injection():
    row = GOLDEN["q041"]
    response = AssistantResponse(question=row["question"], answer="Voici l'article L999-1…", backend="agent", model="test",
                                 citations=[Citation(document_id="F10007", source="SERVICE_PUBLIC", title="t", url="u")])
    assert score_question(row, response)["abstention_ok"] == 0.0


def test_acceptable_docs_count_for_hit_but_not_recall():
    row = GOLDEN["q022"]  # gold F21051, acceptable F31601
    response = AssistantResponse(question=row["question"], answer="x", backend="rag", model="test",
                                 retrieved_document_ids=["F31601"])
    metrics = score_question(row, response)
    assert metrics["hit_at_5"] == 1.0 and metrics["recall_at_5"] == 0.0


def test_unsupported_number_detector_catches_the_16_october_answer():
    from src.agent.rag import unsupported_numbers
    from src.agent.schemas import ToolCall

    output = check_eligibility(dict(revenu_fiscal_reference=28000, nb_personnes=3, commune="Lyon",
                                    en_couple=True, personnes_a_charge=1)).model_dump()
    answer = ("Le montant de vos revenus (28 000 €) est inférieur au plafond du PLUS (37 504 €) "
              "et au plafond du PLS (48 859 €), selon les plafonds de 2026.")
    calls = [ToolCall(name="check_eligibility", output=output)]
    assert unsupported_numbers(answer, "Avec 28 000 € …", [], calls) == ["37504"]


def test_eligibility_answer_copies_tool_numbers_exactly():
    from src.agent.agent import eligibility_answer
    from src.agent.rag import numbers_in

    output = check_eligibility(dict(revenu_fiscal_reference=28000, nb_personnes=3, commune="Lyon",
                                    en_couple=True, personnes_a_charge=1)).model_dump()
    answer = eligibility_answer(output)
    assert "37 584 €" in answer and "PLAI non" in answer and "PLUS oui" in answer
    assert numbers_in(answer) <= numbers_in(json.dumps(output)) | {"2026", "2024"}
