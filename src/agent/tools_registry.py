"""Registre fermé des outils de l'agent et exécution contrôlée de leurs appels.

Chaque appel venant du LLM passe par ToolExecutor.execute :
    1. nom d'outil dans le registre (sinon refus) ;
    2. quota d'appels par outil et refus des appels identiques répétés ;
    3. parsing JSON + validation Pydantic stricte des arguments ;
    4. exécution ;
    5. sortie sérialisée en JSON dans une enveloppe `untrusted_data`, textes
       assainis (src.agent.guards).
Les erreurs sont renvoyées au modèle sous une forme construite par nous,
sans recopier les valeurs reçues.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from src.agent.guards import detect_injection, sanitize_text
from src.config import PROMPTS_DIR
from src.retrieval.schemas import SearchResponse
from src.settings import get_settings
from src.tools.communes import get_resolver
from src.tools.eligibility import HouseholdSituation, check_eligibility
from src.tools.errors import ToolError
from src.tools.market_prices import MarketPricesInput, market_prices
from src.tools.schemas import StrictInput
from src.tools.social_housing import SocialHousingInput, social_housing_stats

logger = logging.getLogger(__name__)

TOOL_DESCRIPTIONS_PATH = PROMPTS_DIR / "tool_descriptions.yaml"


# ============================================================
# search_docs (adaptateur du retrieval pour l'agent)
# ============================================================


class SearchDocsInput(StrictInput):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)
    ] = Field(description="Question ou mots-clés en français.")

    # Ni `k` ni filtre de source exposés au modèle : mesuré le 17/10, gpt-4.1-mini
    # restreignait 50 recherches sur 53 à Service-Public (Hit@5 LEGI = 0,14) et
    # baissait parfois k à 3. La recherche couvre toujours les deux sources.

    @field_validator("query")
    @classmethod
    def _strip_invisible(cls, value: str) -> str:
        return sanitize_text(value)


class Passage(BaseModel):
    document_id: str
    source: str
    title: str
    article: str | None
    text: str


class SearchDocsOutput(BaseModel):
    passages: list[Passage]


@dataclass(frozen=True)
class DocumentMeta:
    document_id: str
    title: str
    url: str
    source: str


def make_search_docs(
    search_fn: Callable[..., Any] | None = None,
    default_k: int | None = None,
) -> Callable[[SearchDocsInput], tuple[SearchDocsOutput, SearchResponse]]:
    """Adapte `src.retrieval.search_docs` : sortie allégée pour le LLM + réponse complète.

    `search_fn(query, k=..., source=...)` doit renvoyer un `SearchResponse` ; par
    défaut `src.retrieval.search_docs`, sinon par exemple `src.agent.retrieval.get_search`.
    """

    def run(request: SearchDocsInput) -> tuple[SearchDocsOutput, SearchResponse]:
        nonlocal search_fn
        if search_fn is None:
            from src.retrieval import search_docs  # import paresseux : charge PyTorch

            search_fn = search_docs

        settings = get_settings().agent
        k = default_k or settings.search_k
        response = search_fn(request.query, k=k, source=None)

        passages = [
            Passage(
                document_id=result.document_id,
                source=result.source,
                title=sanitize_text(result.title, 200),
                article=result.article,
                text=sanitize_text(result.text, settings.max_chunk_chars),
            )
            for result in response.results
        ]
        return SearchDocsOutput(passages=passages), response

    return run


# ============================================================
# REGISTRE
# ============================================================


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[StrictInput]
    fn: Callable[[Any], Any]
    returns_documents: bool = False

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }


@lru_cache(maxsize=1)
def load_tool_descriptions(path: Path = TOOL_DESCRIPTIONS_PATH) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        descriptions = yaml.safe_load(f)

    return {name: " ".join(text.split()) for name, text in descriptions.items()}


def build_registry(
    search_fn: Callable[..., Any] | None = None,
    default_k: int | None = None,
) -> dict[str, ToolSpec]:
    descriptions = load_tool_descriptions()
    specs = [
        ToolSpec(
            "search_docs",
            descriptions["search_docs"],
            SearchDocsInput,
            make_search_docs(search_fn, default_k),
            returns_documents=True,
        ),
        ToolSpec(
            "check_eligibility",
            descriptions["check_eligibility"],
            HouseholdSituation,
            check_eligibility,
        ),
        ToolSpec(
            "social_housing_stats",
            descriptions["social_housing_stats"],
            SocialHousingInput,
            social_housing_stats,
        ),
        ToolSpec(
            "market_prices",
            descriptions["market_prices"],
            MarketPricesInput,
            market_prices,
        ),
    ]
    return {spec.name: spec for spec in specs}


# ============================================================
# EXÉCUTION
# ============================================================


class ToolCallTrace(BaseModel):
    name: str
    arguments: dict[str, Any] | None = Field(
        description="Arguments parsés ; None si le JSON reçu était invalide.",
    )
    ok: bool
    output: Any = Field(
        default=None,
        description="Sortie de l'outil (données) ou erreur renvoyée au modèle.",
    )
    error_code: str | None = None
    error_message: str | None = None
    dropped_arguments: list[str] = Field(
        default_factory=list,
        description="Arguments retirés car inventés par le modèle (absents de la question).",
    )
    latency_ms: float


@dataclass
class ToolOutcome:
    content: str  # message `tool` renvoyé au LLM (JSON)
    trace: ToolCallTrace
    repeated: bool = False


def validation_message(error: ValidationError) -> str:
    """Message d'erreur sans les valeurs reçues : champs et types d'erreur.

    Seuls les messages de nos propres validateurs (`value_error`, rédigés dans le
    code) sont recopiés ; ceux de Pydantic pourraient citer la valeur reçue.
    """
    problems = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "arguments"
        if item["type"] == "value_error":
            message = str(item.get("ctx", {}).get("error", "")) or item["type"]
            problems.append(f"{location} : {message}")
        else:
            problems.append(f"{location} ({item['type']})")
    return "Arguments invalides : " + ", ".join(problems) + ". Corriger et réessayer."


@dataclass
class ToolExecutor:
    """État d'une question : appels, quotas, documents récupérés, textes sources."""

    registry: dict[str, ToolSpec]
    max_calls_per_tool: int
    user_question: str = ""
    calls: Counter = field(default_factory=Counter)
    signatures: set[tuple[str, str]] = field(default_factory=set)
    documents: dict[str, DocumentMeta] = field(default_factory=dict)
    source_texts: list[str] = field(default_factory=list)
    searches: list[SearchResponse] = field(default_factory=list)
    traces: list[ToolCallTrace] = field(default_factory=list)
    injection_in_sources: bool = False

    @property
    def retrieved_document_ids(self) -> list[str]:
        return list(self.documents)

    def _ground_departement(
        self, request: StrictInput
    ) -> tuple[StrictInput, list[str]]:
        """Retire un département inventé par le modèle pour lever une homonymie.

        Si la commune est ambiguë sans département et que ni le code ni le nom
        du département ne figurent dans la question, l'outil doit renvoyer
        l'ambiguïté (le modèle demandera la précision) au lieu de deviner.
        """
        departement = getattr(request, "departement", None)
        commune = getattr(request, "commune", None)

        if departement is None or commune is None:
            return request, []

        resolver = get_resolver()
        if resolver.departement_mentioned(departement, self.user_question):
            return request, []
        if not resolver.is_ambiguous(commune):
            return request, []

        return request.model_copy(update={"departement": None}), ["departement"]

    def execute(self, name: str, raw_arguments: str) -> ToolOutcome:
        start = time.perf_counter()
        parsed: dict[str, Any] | None = None
        dropped: list[str] = []

        def done(
            payload: dict[str, Any], ok: bool, error_code: str | None = None
        ) -> ToolOutcome:
            trace = ToolCallTrace(
                name=name if name in self.registry else "<inconnu>",
                arguments=parsed,
                ok=ok,
                output=payload.get("untrusted_data", payload.get("error")),
                error_code=error_code,
                error_message=None if ok else payload["error"].get("message"),
                dropped_arguments=dropped,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
            self.traces.append(trace)
            if dropped:
                payload["note"] = (
                    "L'utilisateur n'a PAS indiqué de département : la valeur proposée a "
                    "été ignorée. Ne pas la deviner ni l'attribuer à l'utilisateur ; "
                    "présenter les communes candidates et demander laquelle est concernée."
                )
            content = json.dumps(
                {"tool": trace.name, **payload}, ensure_ascii=False, default=str
            )
            return ToolOutcome(content=content, trace=trace)

        def error(code: str, message: str) -> ToolOutcome:
            return done(
                {"status": "error", "error": {"code": code, "message": message}},
                ok=False,
                error_code=code,
            )

        spec = self.registry.get(name)
        if spec is None:
            return error(
                "unknown_tool",
                f"Outil inconnu. Outils disponibles : {', '.join(self.registry)}.",
            )

        try:
            parsed = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            return error("invalid_json", "Les arguments ne sont pas un JSON valide.")

        if not isinstance(parsed, dict):
            parsed = None
            return error("invalid_json", "Les arguments doivent être un objet JSON.")

        signature = (name, json.dumps(parsed, sort_keys=True, ensure_ascii=False))
        if signature in self.signatures:
            outcome = error(
                "repeated_call",
                "Appel identique déjà effectué : utiliser le résultat précédent.",
            )
            outcome.repeated = True
            return outcome

        if self.calls[name] >= self.max_calls_per_tool:
            return error(
                "quota_exceeded",
                f"Nombre maximal d'appels à {name} atteint pour cette question.",
            )

        self.calls[name] += 1
        self.signatures.add(signature)

        try:
            request = spec.input_model.model_validate_json(raw_arguments or "{}")
        except ValidationError as exc:
            return error("invalid_input", validation_message(exc))

        # Une commune absente de la question a été inventée par le modèle (mesuré le
        # 17/10 : « Paris » ajouté à « Je gagne 25 000 €, ai-je droit à un HLM ? »).
        commune = getattr(request, "commune", None)
        if (
            commune is not None
            and self.user_question
            and not get_resolver().commune_mentioned(commune, self.user_question)
        ):
            return error(
                "not_in_question",
                "La commune fournie n'apparaît pas dans la question de l'utilisateur : "
                "ne pas la deviner. Demander dans quelle commune il cherche "
                "(needs_clarification).",
            )

        request, dropped = self._ground_departement(request)

        try:
            result = spec.fn(request)
        except ToolError as exc:
            return done(
                {"status": "error", "error": exc.to_dict()},
                ok=False,
                error_code=exc.code,
            )
        except Exception:
            logger.exception("Erreur inattendue dans l'outil %s", name)
            return error("internal_error", "Erreur interne de l'outil.")

        if spec.returns_documents:
            result, response = result
            self.searches.append(response)
            for hit in response.results:
                self.documents.setdefault(
                    hit.document_id,
                    DocumentMeta(hit.document_id, hit.title, hit.url, hit.source),
                )
            if any(detect_injection(passage.text) for passage in result.passages):
                self.injection_in_sources = True

        data = result.model_dump(mode="json")
        self.source_texts.append(json.dumps(data, ensure_ascii=False))

        return done({"status": "ok", "untrusted_data": data}, ok=True)
