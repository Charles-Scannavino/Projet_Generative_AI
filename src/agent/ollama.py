"""Client HTTP minimal pour l'API locale d'Ollama (/api/chat), sans dépendance.

Points importants :
- le tag Ollama `qwen3:4b` pointe désormais vers Qwen3-4B-**Thinking**-2507, qui raisonne
  toujours avant de répondre (même avec think=false) : c'est la cause probable des 67 à 138 s
  par question. On utilise `qwen3:4b-instruct` (Qwen3-4B-Instruct-2507, même taille, sans
  raisonnement) ; `think=False` reste envoyé pour les modèles hybrides ;
- Ollama renvoie `prompt_eval_count` / `eval_count` (tokens in/out) et
  `done_reason` : on les remonte tels quels pour l'évaluation.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def _load_dotenv(path: Path = Path(__file__).resolve().parents[2] / ".env") -> None:
    """Lit le .env (git-ignoré) sans dépendance ; une variable déjà définie n'est jamais écrasée."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct")
_host = os.environ.get("OLLAMA_HOST", "localhost:11434")
OLLAMA_URL = (_host if _host.startswith(("http://", "https://")) else f"http://{_host}").rstrip("/")
# Délai max d'UNE réponse Ollama (non streamée). 600 s : un portable sans GPU peut dépasser 3 min.
TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT", "600"))
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))


class OllamaError(RuntimeError):
    """Ollama injoignable ou réponse invalide."""


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    done_reason: str | None = None
    latency_ms: float = 0.0
    raw_message: dict[str, Any] = field(default_factory=dict)


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    schema: dict | None = None,
    tools: list[dict] | None = None,
    think: bool = False,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ChatResult:
    options: dict[str, Any] = {"temperature": temperature, "num_ctx": NUM_CTX}
    if max_tokens:
        options["num_predict"] = max_tokens   # plafond de tokens générés (latence bornée)
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": options,
    }
    if schema is not None:
        body["format"] = schema
    if tools:
        body["tools"] = tools
    request = Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, socket.timeout) as error:
        raise OllamaError(
            f"Ollama n'a pas répondu en {TIMEOUT_S:.0f} s (modèle {model} trop lent sur cette machine). "
            f"Augmenter OLLAMA_TIMEOUT, baisser OLLAMA_NUM_CTX ou prendre un modèle plus petit."
        ) from error
    except URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise OllamaError(f"Ollama n'a pas répondu en {TIMEOUT_S:.0f} s (modèle {model}).") from error
        raise OllamaError(
            f"Ollama injoignable sur {OLLAMA_URL} ({error.reason}). Lancer l'application Ollama "
            f"(ou `ollama serve`) et vérifier `ollama list` (modèle {model})."
        ) from error
    except OSError as error:
        raise OllamaError(f"Erreur réseau avec Ollama sur {OLLAMA_URL} : {error}") from error
    if "error" in payload:
        raise OllamaError(f"Ollama : {payload['error']} (modèle {model} ; `ollama pull {model}` ?)")
    message = payload.get("message")
    if not isinstance(message, dict):
        raise OllamaError(f"Réponse Ollama sans message : {str(payload)[:200]}")
    return ChatResult(
        content=message.get("content") or "",
        tool_calls=message.get("tool_calls") or [],
        tokens_in=int(payload.get("prompt_eval_count") or 0),
        tokens_out=int(payload.get("eval_count") or 0),
        done_reason=payload.get("done_reason"),
        latency_ms=(time.perf_counter() - start) * 1000,
        raw_message=message,
    )
